"""PyTorch reference implementations of the three operations the CUDA kernels implement.

They define correctness: every kernel is checked against these (max absolute error) on the
GPU, and the pure-PyTorch versions also run on CPU so the unit tests need no GPU.

- preprocess: nearest-neighbour resize of a wafer map to 64x64, then per-wafer standardisation
  (x - mean) / (std + eps), as float32.
- conv3x3: a 2-D cross-correlation with one 3x3 kernel, zero padding, same output size.
- median3x3: a 3x3 median filter with edge replication (a classic denoise step for the
  salt-and-pepper look of die-level test noise on wafer maps).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

EPS = 1e-6


def preprocess(wafers: torch.Tensor, size: int = 64) -> torch.Tensor:
    """wafers: (N, H, W) uint8/int. Returns (N, size, size) float32, standardised per wafer."""
    x = wafers.to(torch.float32).unsqueeze(1)
    x = F.interpolate(x, size=(size, size), mode="nearest-exact")  # same index rule as the kernel
    x = x.squeeze(1)
    mean = x.mean(dim=(1, 2), keepdim=True)
    std = x.std(dim=(1, 2), keepdim=True, unbiased=False)
    return (x - mean) / (std + EPS)


def conv3x3(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """x: (N, H, W) float32; weight: (3, 3). Zero-padded 'same' cross-correlation."""
    return F.conv2d(x.unsqueeze(1), weight.view(1, 1, 3, 3), padding=1).squeeze(1)


def median3x3(x: torch.Tensor) -> torch.Tensor:
    """x: (N, H, W) float32. 3x3 median with replicate padding."""
    padded = F.pad(x.unsqueeze(1), (1, 1, 1, 1), mode="replicate")
    patches = F.unfold(padded, kernel_size=3)  # (N, 9, H*W)
    med = patches.median(dim=1).values
    return med.view(x.shape)
