"""The 3x3 median and 3x3 convolution kernels written in OpenAI Triton, for comparison with the
hand-written CUDA versions in wafer_ops.cu and the PyTorch references in wafer/reference.py.

Same contracts: median3x3 replicates the border, conv3x3 zero-pads, both take (N, H, W) float32
and return the same shape. One Triton program handles a BLOCK_H x BLOCK_W tile of one wafer; the
nine neighbour tiles are loaded as shifted 2-D blocks, so the compiler is free to vectorise the
loads and keep the whole working set in registers (no shared memory to manage by hand).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _sort2(a, b):
    return tl.minimum(a, b), tl.maximum(a, b)


@triton.jit
def _median9(v0, v1, v2, v3, v4, v5, v6, v7, v8):
    """Median of nine values with the same 19-comparison network as the CUDA kernel."""
    v1, v2 = _sort2(v1, v2)
    v4, v5 = _sort2(v4, v5)
    v7, v8 = _sort2(v7, v8)
    v0, v1 = _sort2(v0, v1)
    v3, v4 = _sort2(v3, v4)
    v6, v7 = _sort2(v6, v7)
    v1, v2 = _sort2(v1, v2)
    v4, v5 = _sort2(v4, v5)
    v7, v8 = _sort2(v7, v8)
    v0, v3 = _sort2(v0, v3)
    v5, v8 = _sort2(v5, v8)
    v4, v7 = _sort2(v4, v7)
    v3, v6 = _sort2(v3, v6)
    v1, v4 = _sort2(v1, v4)
    v2, v5 = _sort2(v2, v5)
    v4, v7 = _sort2(v4, v7)
    v4, v2 = _sort2(v4, v2)
    v6, v4 = _sort2(v6, v4)
    v4, v2 = _sort2(v4, v2)
    return v4


@triton.jit
def _load_replicate(base, rows, cols, dy, dx, H, W, valid):
    """One shifted neighbour tile with replicate padding: the index is clamped into the image."""
    r = tl.minimum(tl.maximum(rows + dy, 0), H - 1)
    c = tl.minimum(tl.maximum(cols + dx, 0), W - 1)
    return tl.load(base + r[:, None] * W + c[None, :], mask=valid, other=0.0)


@triton.jit
def _median3x3_kernel(x_ptr, out_ptr, H, W, BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr):
    b = tl.program_id(0)
    rows = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    cols = tl.arange(0, BLOCK_W)
    base = x_ptr + b.to(tl.int64) * H * W
    valid = (rows[:, None] < H) & (cols[None, :] < W)
    v0 = _load_replicate(base, rows, cols, -1, -1, H, W, valid)
    v1 = _load_replicate(base, rows, cols, -1, 0, H, W, valid)
    v2 = _load_replicate(base, rows, cols, -1, 1, H, W, valid)
    v3 = _load_replicate(base, rows, cols, 0, -1, H, W, valid)
    v4 = _load_replicate(base, rows, cols, 0, 0, H, W, valid)
    v5 = _load_replicate(base, rows, cols, 0, 1, H, W, valid)
    v6 = _load_replicate(base, rows, cols, 1, -1, H, W, valid)
    v7 = _load_replicate(base, rows, cols, 1, 0, H, W, valid)
    v8 = _load_replicate(base, rows, cols, 1, 1, H, W, valid)
    med = _median9(v0, v1, v2, v3, v4, v5, v6, v7, v8)
    tl.store(out_ptr + b.to(tl.int64) * H * W + rows[:, None] * W + cols[None, :], med, mask=valid)


@triton.jit
def _conv3x3_kernel(x_ptr, w_ptr, out_ptr, H, W, BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr):
    b = tl.program_id(0)
    rows = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    cols = tl.arange(0, BLOCK_W)
    base = x_ptr + b.to(tl.int64) * H * W
    valid = (rows[:, None] < H) & (cols[None, :] < W)
    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)
    for dy in tl.static_range(-1, 2):
        for dx in tl.static_range(-1, 2):
            r = rows + dy
            c = cols + dx
            # zero padding: out-of-image neighbours contribute 0
            inside = (r[:, None] >= 0) & (r[:, None] < H) & (c[None, :] >= 0) & (c[None, :] < W)
            v = tl.load(base + r[:, None] * W + c[None, :], mask=inside, other=0.0)
            wgt = tl.load(w_ptr + (dy + 1) * 3 + (dx + 1))
            acc += v * wgt
    tl.store(out_ptr + b.to(tl.int64) * H * W + rows[:, None] * W + cols[None, :], acc, mask=valid)


def _grid(x: torch.Tensor, block_h: int, block_w: int) -> tuple[int, int]:
    n, h, w = x.shape
    assert w <= block_w, "one program covers a full row; raise BLOCK_W for wider images"
    return (n, triton.cdiv(h, block_h))


def median3x3(x: torch.Tensor, block_h: int = 16) -> torch.Tensor:
    x = x.contiguous()
    out = torch.empty_like(x)
    n, h, w = x.shape
    block_w = triton.next_power_of_2(w)
    _median3x3_kernel[_grid(x, block_h, block_w)](x, out, h, w, BLOCK_H=block_h, BLOCK_W=block_w)
    return out


def conv3x3(x: torch.Tensor, weight: torch.Tensor, block_h: int = 16) -> torch.Tensor:
    x = x.contiguous()
    weight = weight.contiguous().to(x.device, torch.float32)
    out = torch.empty_like(x)
    n, h, w = x.shape
    block_w = triton.next_power_of_2(w)
    _conv3x3_kernel[_grid(x, block_h, block_w)](x, weight, out, h, w, BLOCK_H=block_h, BLOCK_W=block_w)
    return out
