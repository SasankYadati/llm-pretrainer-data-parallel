import torch
import torch.nn as nn
import torch.nn.functional as F
import torch._dynamo
from flash_attn.flash_attn_interface import flash_attn_func
from flash_attn.layers.rotary import apply_rotary_emb as _apply_rotary_emb
from flash_attn.ops.triton.layer_norm import layer_norm_fn

# Disable torch.compile tracing for flash-attn's Triton kernels
# These are already highly optimized - torch.compile can't improve them
# and the inductor backend is incompatible with their triton.heuristics decorators

@torch._dynamo.disable
def flash_attention(q, k, v, causal=True):
    q = q.permute(0, 2, 1, 3)
    k = k.permute(0, 2, 1, 3)
    v = v.permute(0, 2, 1, 3)
    return flash_attn_func(q, k, v, causal=causal)

@torch._dynamo.disable
def apply_rotary_emb(q, cos, sin, interleaved=False):
    return _apply_rotary_emb(q, cos, sin, interleaved=interleaved)

def get_cos_sin(seq_length, head_dim, base=500000.0):
    assert head_dim % 2 == 0
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float().to('cpu') / head_dim))
    dtype = torch.bfloat16
    device = torch.device('cuda')
    position = torch.arange(seq_length).to(device).unsqueeze(1).float()
    theta = theta.to(device)
    angles = position.float() * theta.float()
    return (
        torch.cos(angles).to(dtype).repeat(1, 2),
        torch.sin(angles).to(dtype).repeat(1, 2)
    )

class TritonRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.register_parameter("bias", None)

    @torch._dynamo.disable
    def forward(
        self,
        hidden_states,
        residual=None,
        dropout_p=0.0,
        prenorm=False,
        residual_in_fp32=False,
        return_dropout_mask=False
    ):
        return layer_norm_fn(
            hidden_states,
            self.weight,
            None,
            residual=residual,
            eps=self.eps,
            dropout_p=dropout_p,
            prenorm=prenorm,
            residual_in_fp32=residual_in_fp32,
            is_rms_norm=True,
            return_dropout_mask=return_dropout_mask
        )

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size//self.num_heads
        self.num_key_values = config.num_key_value_heads

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_values * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_values * self.head_dim, bias=False)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, x, cos, sin, attention_mask=None, position_ids=None):
        batch_size, seq_length, hidden_dim = x.size()
        q = self.q_proj(x) # [batch_size, seq_length, num_heads*head_dim]
        k = self.k_proj(x) # [batch_size, seq_length, num_key_values*head_dim]
        v = self.v_proj(x) # [batch_size, seq_length, num_key_values*head_dim]

        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim)       # [batch_size, seq_length, num_heads, head_dim]
        k = k.view(batch_size, seq_length, self.num_key_values, self.head_dim)  # [batch_size, seq_length, num_key_values, head_dim]
        q = apply_rotary_emb(q,cos[:, :self.head_dim // 2], sin[:, :self.head_dim // 2],interleaved=False) # [batch_size, seq_length, num_heads, head_dim]
        k = apply_rotary_emb(k,cos[:, :self.head_dim // 2], sin[:, :self.head_dim // 2],interleaved=False) # [batch_size, seq_length, num_key_values, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.view(batch_size, seq_length, self.num_key_values, self.head_dim).transpose(1,2)

        k = k.repeat_interleave(self.num_heads // self.num_key_values, dim=1)
        v = v.repeat_interleave(self.num_heads // self.num_key_values, dim=1)

        # During decoding phase. The lenghth of q is usually 1 (previous token).
        causal = True if q.size(2) == k.size(2) else False

        out = flash_attention(q, k, v, causal = causal) # [batch_size, seq_length, num_heads, head_dim]

        out = out.reshape(batch_size, seq_length, self.num_heads * self.head_dim) # [batch_size, seq_length, hidden_dim]
        out = self.out_proj(out) # [batch_size, seq_length, hidden_dim]
        return out


class MLP(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.up_proj = nn.Linear(
            model_config.hidden_size,
            model_config.intermediate_size,
            bias=False
        )
        self.gate_proj = nn.Linear(
            model_config.hidden_size,
            model_config.intermediate_size,
            bias=False
        )
        self.down_proj = nn.Linear(
            model_config.intermediate_size,
            model_config.hidden_size,
            bias=False
        )

    def forward(self, x):
        x = F.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(x)

class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = TritonRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = TritonRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention = Attention(config)
        self.mlp = MLP(config)
        head_dim = config.hidden_size // config.num_attention_heads
        rope_theta = getattr(config, 'rope_theta', 10000.0)
        self.cos, self.sin = get_cos_sin(
            config.max_position_embeddings,
            head_dim=head_dim,
            base=rope_theta
        )

    def forward(self, x, attention_mask=None, position_ids=None):
        x = x + self.attention(self.input_layernorm(x), self.cos, self.sin, attention_mask, position_ids)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x

class Llama(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        assert model_config.hidden_size % model_config.num_attention_heads == 0
        assert model_config.num_attention_heads % model_config.num_key_value_heads == 0

        # model params
        self.vocab_size = model_config.vocab_size
        self.hidden_size = model_config.hidden_size
        self.num_attention_heads = model_config.num_attention_heads
        self.num_key_values = model_config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.max_position_embeddings = model_config.max_position_embeddings
        self.num_layers = model_config.num_hidden_layers
        self.model_config = model_config

        self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
        self.decoders = nn.ModuleList([
            DecoderLayer(model_config) for _ in range(self.num_layers)
        ])
        self.final_proj = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        self.final_norm = TritonRMSNorm(self.hidden_size, eps=model_config.rms_norm_eps)

    def forward(self, input_ids, attention_mask=None, position_ids: torch.Tensor = None):
        x = self.embedding(input_ids)
        for decoder in self.decoders:
            x = decoder(x)
        x = self.final_norm(x)
        logits = self.final_proj(x)
        return logits
