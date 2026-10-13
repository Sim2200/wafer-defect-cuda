"""Correctness and speed of the CUDA kernels against the PyTorch references. Writes results/kernels.json.

    python -m wafer.bench_kernels --data data/wm811k_64.npz --raw data/raw_sample.npz --out results/kernels.json

For each op: max absolute error vs the reference on real wafer batches, then timing with CUDA
events, `--repeats` repeats of `--iters` launches each, median reported, ms per batch and
wafers/s. The PyTorch column is what a practitioner gets for free (cuDNN for the conv,
unfold+median for the median filter, F.interpolate + mean/std for preprocessing).
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import torch

from . import kernels as K
from . import reference as R


def timed(fn, iters: int, repeats: int) -> float:
    """Median over repeats of the mean ms per call within a repeat (CUDA events, warm-up first)."""
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        out.append(start.elapsed_time(end) / iters)
    return statistics.median(out)


def code_sizes() -> dict:
    """Non-blank, non-comment lines of each implementation of the two ops: the CUDA kernels plus
    their host wrapper in wafer_ops.cu versus the Triton kernels plus wrappers in wafer_triton.py."""
    import re

    root = Path(__file__).resolve().parents[2] / "kernels"
    cu = (root / "wafer_ops.cu").read_text()
    py = (root / "wafer_triton.py").read_text()

    def count(text: str, comment: str) -> int:
        return sum(1 for line in text.splitlines() if line.strip() and not line.strip().startswith(comment))

    def section(text: str, start: str, end: str) -> str:
        i = text.index(start)
        j = text.index(end, i + len(start))
        return text[i:j]

    cuda_median = section(cu, "// ---------------------------------------------------------------- median3x3", "PYBIND11_MODULE")
    cuda_conv = section(cu, "// ---------------------------------------------------------------- conv3x3", "// ---------------------------------------------------------------- median3x3")
    triton_median = section(py, "@triton.jit\ndef _sort2", "@triton.jit\ndef _conv3x3_kernel") + section(py, "def median3x3(", "def conv3x3(")
    triton_conv = section(py, "@triton.jit\ndef _conv3x3_kernel", "def _grid(") + py[py.index("def conv3x3("):]
    return {"cuda_median3x3_naive_and_tiled": count(cuda_median, "//"), "triton_median3x3": count(triton_median, "#"),
            "cuda_conv3x3_naive_and_tiled": count(cuda_conv, "//"), "triton_conv3x3": count(triton_conv, "#"),
            "note": "CUDA counts include both the naive and the shared-memory tiled kernel plus the host wrapper; Triton counts one kernel plus its wrapper (and the shared _grid helper for the conv)"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="npz with x (N,64,64) uint8")
    ap.add_argument("--raw", default="", help="npz with an object array of variable-size wafer maps (for preprocess)")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default="results/kernels.json")
    ap.add_argument("--triton", action="store_true", help="also time the OpenAI Triton versions (kernels/wafer_triton.py)")
    a = ap.parse_args()
    dev = torch.device("cuda")
    x8 = np.load(a.data)["x"][: a.batch]
    x = torch.from_numpy(x8).to(torch.float32).to(dev).contiguous()
    weight = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32, device=dev) / 16  # 3x3 Gaussian
    rows = []

    def add(op, variant, err, ms):
        rows.append({"op": op, "variant": variant, "batch": a.batch, "max_abs_error_vs_reference": err,
                     "ms_per_batch": round(ms, 4), "wafers_per_second": round(a.batch / ms * 1000, 1)})

    ref_conv = R.conv3x3(x, weight)
    add("conv3x3", "pytorch_cudnn", 0.0, timed(lambda: R.conv3x3(x, weight), a.iters, a.repeats))
    for tiled, name in ((False, "cuda_naive"), (True, "cuda_tiled")):
        out = K.conv3x3(x, weight, tiled)
        add("conv3x3", name, float((out - ref_conv).abs().max()), timed(lambda: K.conv3x3(x, weight, tiled), a.iters, a.repeats))

    ref_med = R.median3x3(x)
    add("median3x3", "pytorch_unfold", 0.0, timed(lambda: R.median3x3(x), a.iters, a.repeats))
    for tiled, name in ((False, "cuda_naive"), (True, "cuda_tiled")):
        out = K.median3x3(x, tiled)
        add("median3x3", name, float((out - ref_med).abs().max()), timed(lambda: K.median3x3(x, tiled), a.iters, a.repeats))

    triton_info = None
    if a.triton:
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "kernels"))
        import triton as triton_lib
        import wafer_triton as T

        for bh in (8, 16, 32):
            out = T.conv3x3(x, weight, bh)
            add("conv3x3", f"triton_block{bh}", float((out - ref_conv).abs().max()), timed(lambda: T.conv3x3(x, weight, bh), a.iters, a.repeats))
            out = T.median3x3(x, bh)
            add("median3x3", f"triton_block{bh}", float((out - ref_med).abs().max()), timed(lambda: T.median3x3(x, bh), a.iters, a.repeats))
        triton_info = {"triton": triton_lib.__version__, "lines_of_code": code_sizes()}

    if a.raw:
        raw = list(np.load(a.raw, allow_pickle=True)["maps"][: a.batch])
        flat, heights, widths, offsets = K.pack_wafers(raw, "cuda")
        ext = K.load()
        ref_pre = torch.stack([R.preprocess(torch.from_numpy(np.asarray(w))[None].to(dev))[0] for w in raw])
        out = ext.preprocess(flat, heights, widths, offsets, 64)
        err = float((out - ref_pre).abs().max())
        ms_kernel = timed(lambda: ext.preprocess(flat, heights, widths, offsets, 64), a.iters, a.repeats)
        # The PyTorch path has to resize each variable-size map separately: that loop is the comparison.
        maps_dev = [torch.from_numpy(np.asarray(w))[None].to(dev) for w in raw]
        ms_torch = timed(lambda: torch.stack([R.preprocess(m)[0] for m in maps_dev]), max(1, a.iters // 10), a.repeats)
        add("preprocess", "pytorch_per_wafer_loop", 0.0, ms_torch)
        add("preprocess", "cuda_packed", err, ms_kernel)

    out = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda,
           "batch": a.batch, "iters": a.iters, "repeats": a.repeats, "timing": "CUDA events, median of repeats of mean ms per call",
           "rows": rows}
    if triton_info:
        out.update(triton_info)
    for r in rows:
        print(f"{r['op']:11s} {r['variant']:22s} err {r['max_abs_error_vs_reference']:.2e}  {r['ms_per_batch']:8.3f} ms/batch  {r['wafers_per_second']:>10,.0f} wafers/s")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
