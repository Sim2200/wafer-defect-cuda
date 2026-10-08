"""Export the trained classifier to ONNX, build TensorRT engines (FP32, FP16, INT8 with real
calibration) and benchmark every backend on the same inputs. Writes two files:

    results/tensorrt_bench.json      latency p50/p95, throughput, GPU memory per backend x batch size
    results/tensorrt_accuracy.json   test macro-F1 and per-class recall per backend, parity vs PyTorch

    python -m wafer.trt_bench --data wm811k_64.npz --weights cnn.pt --out-dir results

Backends: PyTorch eager (fp32 and autocast fp16), torch.compile, ONNX Runtime CUDA EP, TensorRT
FP32 / FP16 / INT8. Inputs already live on the GPU for every backend, so the numbers are the
model's compute time plus framework overhead, not PCIe copies.

INT8 calibration uses `--calib` wafers from the *train* split only (never val or test) through
TensorRT's entropy calibrator, which picks a per-tensor scale that minimises the KL divergence
between the fp32 activation histogram and its int8 quantisation. The calibration cache is written
next to the engine so a rebuild is reproducible.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .data import CLASSES
from .metrics import report
from .model import build
from .train import standardise

BATCHES = (1, 8, 32, 128)


# ----------------------------------------------------------------------------- helpers
def gpu_mem_mib() -> float:
    """Memory in use on GPU 0 as the driver sees it (covers TensorRT and ORT allocations too,
    which torch.cuda.memory_allocated does not)."""
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"],
                         capture_output=True, text=True)
    return float(out.stdout.strip().splitlines()[0])


def percentile(xs: list[float], p: float) -> float:
    s = sorted(xs)
    return s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))]


def time_backend(run, x: torch.Tensor, warmup: int = 20, iters: int = 100) -> dict:
    """Each iteration is one call on `x` (a GPU tensor), synchronised, timed on the host."""
    for _ in range(warmup):
        run(x)
    torch.cuda.synchronize()
    base = gpu_mem_mib()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run(x)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return {"batch": x.shape[0], "iters": iters, "p50_ms": round(statistics.median(times), 4),
            "p95_ms": round(percentile(times, 95), 4), "mean_ms": round(statistics.fmean(times), 4),
            "wafers_per_second": round(x.shape[0] / statistics.median(times) * 1000, 1),
            "gpu_memory_used_mib": round(max(base, gpu_mem_mib()), 1)}


def predict_all(run, x: torch.Tensor, batch: int = 128) -> np.ndarray:
    out = []
    for i in range(0, len(x), batch):
        xb = x[i:i + batch]
        if xb.shape[0] != batch:  # engines are built for fixed batch sizes; pad the tail
            pad = torch.zeros(batch - xb.shape[0], *xb.shape[1:], device=xb.device, dtype=xb.dtype)
            out.append(run(torch.cat([xb, pad]))[: xb.shape[0]].argmax(1).cpu())
        else:
            out.append(run(xb).argmax(1).cpu())
    return torch.cat(out).numpy()


# ----------------------------------------------------------------------------- ONNX Runtime
def ort_runner(onnx_path: Path):
    import onnxruntime as ort

    # The pip wheel finds CUDA and cuDNN through the nvidia-* pip packages only if they are loaded
    # first; without this the CUDA provider silently falls back to the CPU one.
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()
    sess = ort.InferenceSession(str(onnx_path), providers=["CUDAExecutionProvider"])
    assert "CUDAExecutionProvider" in sess.get_providers(), sess.get_providers()

    def run(x: torch.Tensor) -> torch.Tensor:
        # IOBinding keeps input and output on the GPU: no host round trip inside the timed region.
        io = sess.io_binding()
        io.bind_input("input", "cuda", 0, np.float32, tuple(x.shape), x.data_ptr())
        y = torch.empty(x.shape[0], len(CLASSES), device="cuda", dtype=torch.float32)
        io.bind_output("logits", "cuda", 0, np.float32, tuple(y.shape), y.data_ptr())
        sess.run_with_iobinding(io)
        return y

    return run, ort.__version__


# ----------------------------------------------------------------------------- TensorRT
def build_engine(onnx_path: Path, engine_path: Path, precision: str, calib: torch.Tensor | None, log) -> dict:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        raise RuntimeError("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    # One engine serves every batch size in BATCHES: the optimisation profile spans 1..128 and
    # TensorRT tunes kernels for the "opt" shape.
    profile = builder.create_optimization_profile()
    profile.set_shape("input", (1, 1, 64, 64), (32, 1, 64, 64), (max(BATCHES), 1, 64, 64))
    config.add_optimization_profile(profile)

    calibrator = None
    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16)  # layers that gain nothing from int8 may still use fp16
        calibrator = EntropyCalibrator(calib, engine_path.with_suffix(".calib"))
        config.int8_calibrator = calibrator
        config.set_calibration_profile(profile)

    t0 = time.perf_counter()
    blob = builder.build_serialized_network(network, config)
    build_s = time.perf_counter() - t0
    if blob is None:
        raise RuntimeError(f"TensorRT build failed for {precision}")
    engine_path.write_bytes(bytes(blob))
    info = {"precision": precision, "build_seconds": round(build_s, 1), "engine_bytes": engine_path.stat().st_size,
            "layers_in_network": network.num_layers}
    if calibrator is not None:
        info["calibration"] = {"algorithm": "IInt8EntropyCalibrator2 (KL divergence)", "wafers": int(calib.shape[0]),
                               "batch": calibrator.batch, "source": "train split only"}
    log(f"  {precision}: built in {build_s:.1f} s, {info['engine_bytes'] / 1e6:.2f} MB")
    return info


def make_calibrator_class():
    import tensorrt as trt

    class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
        """Feeds calibration batches from a GPU tensor. TensorRT calls get_batch until it returns
        None, collects activation histograms, then writes the per-tensor scales to the cache."""

        def __init__(self, data: torch.Tensor, cache: Path, batch: int = 32):
            super().__init__()
            self.data, self.cache, self.batch, self.i = data.contiguous(), cache, batch, 0

        def get_batch_size(self) -> int:
            return self.batch

        def get_batch(self, names):
            if self.i + self.batch > self.data.shape[0]:
                return None
            chunk = self.data[self.i:self.i + self.batch]
            self.i += self.batch
            self._keep = chunk  # keep the slice alive until TensorRT has read it
            return [int(chunk.data_ptr())]

        def read_calibration_cache(self):
            return self.cache.read_bytes() if self.cache.exists() else None

        def write_calibration_cache(self, cache):
            self.cache.write_bytes(bytes(cache))

    return EntropyCalibrator


def trt_runner(engine_path: Path):
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    context = engine.create_execution_context()
    stream = torch.cuda.current_stream().cuda_stream

    def run(x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        y = torch.empty(x.shape[0], len(CLASSES), device="cuda", dtype=torch.float32)
        context.set_input_shape("input", tuple(x.shape))
        context.set_tensor_address("input", x.data_ptr())
        context.set_tensor_address("logits", y.data_ptr())
        context.execute_async_v3(stream)
        return y

    return run, trt.__version__


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="npz with x, y, train, val, test")
    ap.add_argument("--weights", required=True, help="state_dict from wafer.train --save")
    ap.add_argument("--model", default="cnn")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--work", default="trt_work", help="where ONNX, engines and the calibration cache go")
    ap.add_argument("--calib", type=int, default=512, help="train wafers used for INT8 calibration")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--iters", type=int, default=100)
    a = ap.parse_args()
    out_dir, work = Path(a.out_dir), Path(a.work)
    out_dir.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    log = lambda *s: print(*s, flush=True)  # noqa: E731
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    d = np.load(a.data)
    x, y = d["x"], d["y"]
    te, tr = d["test"], d["train"]
    x_te = standardise(x[te]).to(dev)
    y_te = y[te]
    rng = np.random.default_rng(0)
    calib = standardise(x[rng.choice(tr, size=a.calib, replace=False)]).to(dev)
    log(f"test wafers {len(te)}, calibration wafers {a.calib} (train split)")

    model = build(a.model, len(CLASSES)).to(dev).eval()
    model.load_state_dict(torch.load(a.weights, map_location=dev))

    # 1. ONNX export with a dynamic batch axis
    onnx_path = work / "wafer_cnn.onnx"
    torch.onnx.export(model, torch.zeros(1, 1, 64, 64, device=dev), str(onnx_path), opset_version=a.opset,
                      input_names=["input"], output_names=["logits"], dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
                      dynamo=False)
    log(f"ONNX opset {a.opset}: {onnx_path.stat().st_size / 1e6:.2f} MB")

    backends: dict[str, tuple] = {}

    @torch.no_grad()
    def torch_fp32(xb):
        return model(xb)

    @torch.no_grad()
    def torch_fp16(xb):
        with torch.autocast("cuda", dtype=torch.float16):
            return model(xb).float()

    backends["pytorch_fp32"] = (torch_fp32, torch.__version__)
    backends["pytorch_autocast_fp16"] = (torch_fp16, torch.__version__)
    try:
        compiled = torch.compile(model)
        with torch.no_grad():
            compiled(x_te[:32])
        backends["torch_compile_fp32"] = (torch.no_grad()(compiled), torch.__version__)
    except Exception as exc:  # noqa: BLE001 - report, do not abort the benchmark
        log("torch.compile unavailable:", str(exc)[:200])
        backends["torch_compile_fp32"] = (None, str(exc)[:200])

    run, ort_version = ort_runner(onnx_path)
    backends["onnxruntime_cuda_fp32"] = (run, ort_version)

    global EntropyCalibrator
    EntropyCalibrator = make_calibrator_class()
    builds = {}
    for precision in ("fp32", "fp16", "int8"):
        engine_path = work / f"wafer_cnn_{precision}.plan"
        builds[precision] = build_engine(onnx_path, engine_path, precision, calib if precision == "int8" else None, log)
        run, trt_version = trt_runner(engine_path)
        backends[f"tensorrt_{precision}"] = (run, trt_version)

    # 2. accuracy and parity on the full test set
    ref_logits = torch.cat([torch_fp32(x_te[i:i + 128]) for i in range(0, len(x_te), 128)])
    ref_pred = ref_logits.argmax(1).cpu().numpy()
    accuracy = {}
    for name, (run, _) in backends.items():
        if run is None:
            continue
        pred = predict_all(run, x_te)
        rep = report(y_te, pred)
        logits = torch.cat([run(x_te[i:i + 128]) for i in range(0, len(x_te) - len(x_te) % 128, 128)])
        n = logits.shape[0]
        accuracy[name] = {"macro_f1": rep["macro_f1"], "accuracy": rep["accuracy"], "macro_recall": rep["macro_recall"],
                          "per_class_recall": {c: rep["per_class"][c]["recall"] for c in CLASSES},
                          "agreement_with_pytorch": round(float((pred == ref_pred).mean()), 5),
                          "max_abs_logit_diff_vs_pytorch": round(float((logits - ref_logits[:n]).abs().max()), 5)}
        log(f"{name:24s} macro-F1 {rep['macro_f1']:.4f}  agree {accuracy[name]['agreement_with_pytorch']:.4f}  "
            f"max|dlogit| {accuracy[name]['max_abs_logit_diff_vs_pytorch']:.4f}")

    # 3. latency and throughput
    bench = {}
    for name, (run, _) in backends.items():
        if run is None:
            continue
        bench[name] = []
        for b in BATCHES:
            row = time_backend(run, x_te[:b].contiguous(), iters=a.iters)
            bench[name].append(row)
            log(f"{name:24s} batch {b:4d}  p50 {row['p50_ms']:8.3f} ms  p95 {row['p95_ms']:8.3f} ms  {row['wafers_per_second']:10.0f} wafers/s  {row['gpu_memory_used_mib']:6.0f} MiB")

    env = {"gpu": torch.cuda.get_device_name(0), "driver": subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                                                            capture_output=True, text=True).stdout.strip(),
           "torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
           "onnxruntime": ort_version, "tensorrt": trt_version, "onnx_opset": a.opset, "python": sys.version.split()[0]}
    base_f1 = accuracy["pytorch_fp32"]["macro_f1"]
    (out_dir / "tensorrt_bench.json").write_text(json.dumps({
        "env": env, "engine_builds": builds, "batch_sizes": list(BATCHES), "iters_per_point": a.iters, "warmup": 20,
        "method": "inputs resident on the GPU; one synchronised call per iteration timed on the host; p50/p95 over iters; "
                  "GPU memory is nvidia-smi memory.used while that backend runs (includes the CUDA context)",
        "backends": bench, "versions": {k: v[1] for k, v in backends.items()}}, indent=2))
    (out_dir / "tensorrt_accuracy.json").write_text(json.dumps({
        "env": env, "test_wafers": int(len(te)), "pytorch_fp32_macro_f1": base_f1,
        "backends": {k: {**v, "macro_f1_delta_vs_pytorch": round(v["macro_f1"] - base_f1, 4)} for k, v in accuracy.items()},
        "int8_calibration": builds["int8"].get("calibration")}, indent=2))
    log("wrote", out_dir / "tensorrt_bench.json", out_dir / "tensorrt_accuracy.json")


if __name__ == "__main__":
    main()
