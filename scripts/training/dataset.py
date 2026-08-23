"""Memory-mapped random-window batching over the flat token streams produced by
scripts/training/prepare_bin.py.

Windows are NOT pre-cut and stored — every call to get_batch draws a fresh random
offset per sequence in the batch, so the model sees different chunk boundaries each
epoch instead of a fixed, repeated set of them.
"""

from pathlib import Path

import numpy as np
import torch


def load_memmap(bin_path: Path) -> np.memmap:
    return np.memmap(bin_path, dtype=np.uint16, mode="r")


def get_batch(
    data: np.memmap,
    block_size: int,
    batch_size: int,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = len(data) - block_size - 1
    if max_start < 1:
        raise ValueError(
            f"token stream ({len(data)} tokens) too short for block_size={block_size}")

    starts = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([
        torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in starts
    ])
    y = torch.stack([
        torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in starts
    ])

    if "cuda" in str(device):
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y
