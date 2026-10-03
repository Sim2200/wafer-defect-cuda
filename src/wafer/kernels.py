"""Builds and wraps the CUDA extension (kernels/wafer_ops.cu) with torch.utils.cpp_extension.

`load()` compiles on first use (nvcc, about a minute on a T4 host) and caches the binary. The
wrappers accept the same arguments as the PyTorch references in `reference.py`, so a benchmark
or a test can call either side with one code path.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

KERNEL_SRC = Path(__file__).resolve().parents[2] / "kernels" / "wafer_ops.cu"


@lru_cache(maxsize=1)
def load():
    from torch.utils.cpp_extension import load as cpp_load

    return cpp_load(name="wafer_ops", sources=[str(KERNEL_SRC)], extra_cuda_cflags=["-O3"], verbose=False)


def pack_wafers(wafers: list[np.ndarray], device: str = "cuda"):
    """Flatten variable-size uint8 wafer maps into one buffer plus (height, width, offset) tables,
    the layout the preprocess kernel reads (one thread block tile per 16x16 output patch)."""
    heights = torch.tensor([w.shape[0] for w in wafers], dtype=torch.int32)
    widths = torch.tensor([w.shape[1] for w in wafers], dtype=torch.int32)
    sizes = (heights.to(torch.int64) * widths.to(torch.int64))
    offsets = torch.zeros(len(wafers), dtype=torch.int64)
    offsets[1:] = torch.cumsum(sizes, 0)[:-1]
    flat = torch.from_numpy(np.concatenate([np.asarray(w, dtype=np.uint8).ravel() for w in wafers]))
    return flat.to(device), heights.to(device), widths.to(device), offsets.to(device)


def preprocess(wafers: list[np.ndarray], size: int = 64, device: str = "cuda") -> torch.Tensor:
    flat, heights, widths, offsets = pack_wafers(wafers, device)
    return load().preprocess(flat, heights, widths, offsets, size)


def conv3x3(x: torch.Tensor, weight: torch.Tensor, tiled: bool = True) -> torch.Tensor:
    return load().conv3x3(x.contiguous(), weight.contiguous().to(x.device), tiled)


def median3x3(x: torch.Tensor, tiled: bool = True) -> torch.Tensor:
    return load().median3x3(x.contiguous(), tiled)
