"""
torchrun --nproc_per_node=N train.py
"""
import argparse
import os
from utils import set_all_seed, init_distributed
from transformers import AutoConfig
from model import Llama
import torch
import torch.distributed as dist
from torch.optim import AdamW
import time
import wandb
from dataloader import MicroBatchDataLoader
import torch.nn.functional as F

# A100 80GB BF16 peak throughput
A100_PEAK_FLOPS = 312 * 10 ** 12

def calculate_mfu(tokens_per_second, num_params, world_size=1):
    """
    Calculate per-GPU Model FLOPs Utilization (MFU).

    Uses 6*N*D as approximation for total training flops.
    Divides by (peak_flops * world_size) to get per-GPU utilization.
    """
    actual_flops = 6 * num_params * tokens_per_second
    return 100 * actual_flops / (A100_PEAK_FLOPS * world_size)

def naive_sync_gradients(model, world_size):
    """Average gradients across all ranks using one all_reduce per parameter."""
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad, dist.ReduceOp.SUM)
            param.grad /= world_size


def train_step(model, data_loader, device, dtype, grad_acc_steps):
    loss_acc = 0.0
    for i in range(grad_acc_steps):
        batch = next(data_loader)
        input_ids = batch['input_ids'].to(device)
        target_ids = batch['target_ids'].to(device)

        with torch.autocast(device_type='cuda', dtype=dtype):
            outputs = model(input_ids=input_ids)
            batch_size, seq_len = input_ids.shape
            target_ids = target_ids.reshape(-1)
            outputs = outputs.view(seq_len * batch_size, -1)
            loss = F.cross_entropy(outputs, target_ids, reduction='mean') / grad_acc_steps

        loss.backward()
        loss_acc += loss.item()
    return loss_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training script for Llama")


    # model config
    parser.add_argument("--model_name", type=str, default="HuggingFaceTB/SmolLM-1.7B")

    # training
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--seq_len", type=int, default=512)
    # Global batch: 32 * 8 * 512 = 131K tokens per optimizer step
    # Also holds => 262K = 16 * 8 * 512
    parser.add_argument("--micro_batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    # Batch size warmup: ramp grad_acc_steps from 1 to target over this many tokens (0 = disabled)
    parser.add_argument("--batch_size_warmup_tokens", type=int, default=0)
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--dataset_config", type=str, default="sample-10BT")
    parser.add_argument("--n_tokens", type=int, default=10000000)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_proc", type=int, default=48)

    # logging
    parser.add_argument("--run_name", type=str, default="dp_naive")

    args = parser.parse_args()

    # Initialize distributed training
    rank, local_rank, world_size = init_distributed()
    device = torch.device(f"cuda:{local_rank}")
    dtype = torch.bfloat16

    set_all_seed(args.seed)

    # Adjust grad_acc_steps so global batch size stays constant across GPU counts
    assert args.gradient_accumulation_steps % world_size == 0, \
        f"grad_acc_steps ({args.gradient_accumulation_steps}) must be divisible by world_size ({world_size})"
    local_grad_acc_steps = args.gradient_accumulation_steps // world_size

    # Only rank 0 logs to wandb
    if rank == 0:
        wandb.init(
            project="llm-pretrainer",
            name=f"{args.run_name}",
            config={
                "model": args.model_name,
                "learning_rate": args.learning_rate,
                "seed": args.seed,
                "world_size": world_size,
            },
        )

    model_config = AutoConfig.from_pretrained(args.model_name)

    model = Llama(model_config=model_config)
    model = torch.compile(model)
    model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    # Create dataloader — now with rank/world_size for distributed sampling
    dataloader = MicroBatchDataLoader(
        seq_len=args.seq_len,
        micro_batch_size=args.micro_batch_size,
        grad_acc_steps=local_grad_acc_steps,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        tokenizer_name=args.model_name,
        n_tokens=args.n_tokens,
        num_workers=args.num_workers,
        num_proc=args.num_proc,
        rank=rank,
        world_size=world_size,
    )

    target_tokens_per_step = args.micro_batch_size * args.gradient_accumulation_steps * args.seq_len
    if rank == 0:
        print(f"Target tokens per step: {target_tokens_per_step:,}")

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if rank == 0:
        print(f"Trainable parameters: {num_params:,}")

    trained_token, step = 0, 0

    # Warmup steps to exclude from MFU measurement (torch.compile overhead)
    warmup_steps = 5

    # Training loop
    while trained_token < args.n_tokens:

        # Batch size warmup: linearly ramp grad_acc_steps from 1 to target
        if args.batch_size_warmup_tokens == 0:
            current_grad_acc_steps = local_grad_acc_steps
        else:
            current_grad_acc_steps = min(
                1 + (trained_token * (local_grad_acc_steps - 1)) // args.batch_size_warmup_tokens,
                local_grad_acc_steps
            )
        tokens_per_step = args.micro_batch_size * current_grad_acc_steps * args.seq_len * world_size

        optimizer.zero_grad()

        torch.cuda.synchronize()
        t0 = time.time()
        loss = train_step(model, dataloader, device, dtype, current_grad_acc_steps)
        torch.cuda.synchronize()
        t1 = time.time()
        naive_sync_gradients(model, world_size)
        torch.cuda.synchronize()
        t2 = time.time()
        compute_time, comm_time = t1 - t0, t2 - t1
        step_duration = t2 - t0

        optimizer.step()

        trained_token += tokens_per_step
        step += 1

        tokens_per_second = tokens_per_step / step_duration if step_duration > 0 else 0

        if rank == 0:
            is_warmup = step <= warmup_steps
            mfu = None if is_warmup else calculate_mfu(tokens_per_second, num_params, world_size)
            mfu_str = "warming up" if is_warmup else f"{mfu:.1f}%"
            comm_pct = 100 * comm_time / step_duration if step_duration > 0 else 0
            print(
                f"Step: {step}, Loss: {loss:.4f}, "
                f"Tokens/s: {tokens_per_second:.0f}, "
                f"MFU: {mfu_str}, "
                f"Compute: {compute_time*1000:.1f}ms, "
                f"Comm: {comm_time*1000:.1f}ms ({comm_pct:.1f}%), "
                f"Tokens: {trained_token}/{args.n_tokens}, "
                f"Memory: {torch.cuda.memory_reserved() / 1e9:.2f}GB"
            )

            wandb.log({
                "loss": loss,
                "tokens_per_step": tokens_per_step,
                "tokens_per_second": tokens_per_second,
                "mfu": mfu,
                "grad_acc_steps": current_grad_acc_steps,
                "memory_usage": torch.cuda.memory_reserved() / 1e9,
                "trained_tokens": trained_token,
                "compute_time_ms": compute_time * 1000,
                "comm_time_ms": comm_time * 1000,
                "comm_pct": comm_pct,
            })

    if rank == 0:
        wandb.finish()
    dist.destroy_process_group()
