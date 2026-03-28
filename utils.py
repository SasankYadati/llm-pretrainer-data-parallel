import os
import numpy as np
import random
import torch
import torch.distributed as dist

def set_all_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_distributed():
    """Initialize distributed training and return (rank, local_rank, world_size).

    When launched via `torchrun`, the following env vars are set automatically:
        - RANK: global rank of this process
        - LOCAL_RANK: rank on this node (used to pick the GPU)
        - WORLD_SIZE: total number of processes

    """
    RANK, LOCAL_RANK, WORLD_SIZE = int(os.environ['RANK']), int(os.environ['LOCAL_RANK']), int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(LOCAL_RANK)
    dist.init_process_group(backend='nccl')
    return (RANK, LOCAL_RANK, WORLD_SIZE)
