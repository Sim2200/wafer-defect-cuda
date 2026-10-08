"""Small, repeatable workloads for Nsight Systems / Nsight Compute.

    nsys profile -o median python -m wafer.profile_targets median --data wm811k_64.npz
    nsys profile -o trt    python -m wafer.profile_targets trt    --engine trt_work/wafer_cnn_fp16.plan --data wm811k_64.npz
    ncu --set default -k regex:median3x3 --launch-count 3 python -m wafer.profile_targets median --data ... --iters 3

Each target does a warm-up, then `--iters` launches inside an NVTX range named after the target,
so the kernel summary can be filtered to the steady-state part of the trace.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .train import standardise


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", choices=["median", "trt"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--engine", default="")
    ap.add_argument("--batch", type=int, default=1024, help="the trt target is capped at the engine's largest profile shape (128)")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--tiled", type=int, default=1)
    ap.add_argument("--torch-profile", default="", help="also record the steady-state loop with torch.profiler (CUPTI) "
                                                        "and write the per-kernel table here as JSON")
    a = ap.parse_args()
    dev = torch.device("cuda")
    x8 = np.load(a.data)["x"]

    if a.target == "median":
        from . import kernels as K

        x = torch.from_numpy(x8[: a.batch]).to(torch.float32).to(dev).contiguous()
        run = lambda: K.median3x3(x, tiled=bool(a.tiled))  # noqa: E731
    else:
        from .trt_bench import trt_runner

        trt_run, _ = trt_runner(Path(a.engine))
        # engines are built with an optimisation profile up to batch 128; a larger input would make
        # set_input_shape fail and the run would silently execute nothing
        x = standardise(x8[: min(a.batch, 128)]).to(dev).contiguous()
        run = lambda: trt_run(x)  # noqa: E731

    for _ in range(10):
        run()
    torch.cuda.synchronize()
    if a.torch_profile:
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(a.iters):
                run()
            torch.cuda.synchronize()
        rows = []
        for ev in prof.key_averages():
            if ev.device_type == torch.autograd.DeviceType.CUDA and ev.self_device_time_total > 0:
                rows.append({"kernel": ev.key, "calls": ev.count, "total_us": round(ev.self_device_time_total, 1),
                             "avg_us": round(ev.self_device_time_total / ev.count, 1)})
        total = sum(r["total_us"] for r in rows) or 1.0
        for r in rows:
            r["time_pct"] = round(100 * r["total_us"] / total, 1)
        rows.sort(key=lambda r: -r["total_us"])
        import json

        json.dump({"target": a.target, "iters": a.iters, "batch": a.batch, "kernels": rows[:12]}, open(a.torch_profile, "w"), indent=1)
        return
    with torch.cuda.nvtx.range(f"steady_{a.target}"):
        for _ in range(a.iters):
            run()
        torch.cuda.synchronize()


if __name__ == "__main__":
    main()
