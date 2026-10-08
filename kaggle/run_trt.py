"""TensorRT experiment on one Kaggle T4. Pushed by scripts/run_on_kaggle.py --experiment trt.

Writes under /kaggle/working/results:
  trt_env.json              GPU, driver, torch/CUDA/TensorRT/ONNX Runtime versions, tool availability
  trt_train.json            the 1-GPU training run whose weights are exported (test macro-F1 etc.)
  tensorrt_bench.json       latency/throughput/memory per backend x batch size (wafer.trt_bench)
  tensorrt_accuracy.json    test macro-F1 and per-class recall per backend, parity vs PyTorch
  nsight_summary.md         top kernels and time share from Nsight Systems, Nsight Compute metrics
                            for the median kernel (when the tools run on Kaggle; otherwise it says so)
"""

import csv
import glob
import io
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

WORK = Path("/kaggle/working")
RES = WORK / "results"
RES.mkdir(parents=True, exist_ok=True)
SRC = Path(glob.glob("/kaggle/input/**/src/wafer/data.py", recursive=True)[0]).parents[2]
sys.path.insert(0, str(SRC / "src"))
os.environ["PYTHONPATH"] = str(SRC / "src")
os.environ["TORCH_EXTENSIONS_DIR"] = str(WORK / "torch_ext")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # one T4 for everything here
EPOCHS = int(os.environ.get("EPOCHS", "6"))
PY = sys.executable


def sh(*args, check=True, capture=False, env=None):
    print("$", " ".join(str(a) for a in args), flush=True)
    t0 = time.time()
    r = subprocess.run([str(a) for a in args], text=True, capture_output=capture, env={**os.environ, **(env or {})})
    print(f"  -> exit {r.returncode} in {time.time() - t0:.0f} s", flush=True)
    if check and r.returncode != 0:
        if capture:
            print(r.stdout[-2000:], r.stderr[-2000:])
        raise SystemExit(f"step failed: {args[0]}")
    return r


def pip(*pkgs):
    sh(PY, "-m", "pip", "install", "-q", *pkgs, check=False)


ORT_CUDA12_INDEX = "https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/"


def ort_cuda_works() -> bool:
    """Open a one-op model on the CUDA provider in a fresh interpreter (import state is sticky)."""
    code = (
        "import onnx, onnxruntime as ort\nfrom onnx import helper, TensorProto\n"
        "g = helper.make_graph([helper.make_node('Relu', ['x'], ['y'])], 'g', "
        "[helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 4])], [helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 4])])\n"
        "m = helper.make_model(g, opset_imports=[helper.make_opsetid('', 17)]); m.ir_version = 9; onnx.save(m, '/tmp/relu.onnx')\n"
        "ort.preload_dlls() if hasattr(ort, 'preload_dlls') else None\n"
        "s = ort.InferenceSession('/tmp/relu.onnx', providers=['CUDAExecutionProvider'])\n"
        "print(ort.__version__, s.get_providers())\nassert 'CUDAExecutionProvider' in s.get_providers()"
    )
    r = subprocess.run([PY, "-c", code], capture_output=True, text=True)
    print((r.stdout + r.stderr)[-600:], flush=True)
    return r.returncode == 0


def ensure_ort_cuda():
    """The default onnxruntime-gpu wheel is built for CUDA 13; this image has CUDA 12.8. The
    CUDA 12 builds live on ONNX Runtime's own package index, with an older pinned release as
    the fallback. Each attempt is verified by actually running on the CUDA provider."""
    attempts = (
        ["onnxruntime-gpu", "--extra-index-url", ORT_CUDA12_INDEX],
        ["onnxruntime-gpu==1.22.0"],
        ["onnxruntime-gpu==1.20.1"],
    )
    for spec in attempts:
        sh(PY, "-m", "pip", "uninstall", "-y", "-q", "onnxruntime-gpu", "onnxruntime", check=False)
        sh(PY, "-m", "pip", "install", "-q", *spec, check=False)
        if ort_cuda_works():
            return
    raise SystemExit("no onnxruntime-gpu build with a working CUDA provider on this image")


def nsys_kernel_summary(report: Path, label: str) -> str:
    """`nsys stats` kernel table for one report as a Markdown table (top 8 kernels by time)."""
    r = subprocess.run(["nsys", "stats", "--report", "cuda_gpu_kern_sum", "--format", "csv", "--force-export=true", str(report)],
                       capture_output=True, text=True)
    text = r.stdout
    start = text.find("Time (%)")
    if start < 0:
        return f"### {label}\n\nnsys stats produced no kernel table:\n```\n{(r.stdout + r.stderr)[-800:]}\n```\n"
    rows = list(csv.DictReader(io.StringIO(text[start:])))
    rows.sort(key=lambda row: -float(row["Time (%)"]))
    lines = [f"### {label}", "", "| Time % | Total ms | Calls | Avg us | Kernel |", "|---|---|---|---|---|"]
    for row in rows[:8]:
        name = row["Name"]
        name = (name[:90] + "...") if len(name) > 90 else name
        lines.append(f"| {float(row['Time (%)']):.1f} | {float(row['Total Time (ns)']) / 1e6:.2f} | {row['Instances']} | "
                     f"{float(row['Avg (ns)']) / 1e3:.1f} | `{name}` |")
    return "\n".join(lines) + "\n"


def ncu_metrics(text: str) -> str:
    """Pull the headline metrics out of `ncu --set default` text output."""
    keep = ("Duration", "Memory Throughput", "DRAM Throughput", "Compute (SM) Throughput", "Achieved Occupancy",
            "Theoretical Occupancy", "Registers Per Thread", "Block Size", "Grid Size", "L1/TEX Hit Rate", "L2 Hit Rate",
            "Achieved Active Warps Per SM", "Waves Per SM")
    out = []
    for line in text.splitlines():
        if any(line.strip().startswith(k) for k in keep):
            out.append("- " + " ".join(line.split()))
    return "\n".join(out[:24]) or "(no metrics parsed)"


def main():
    import torch

    env = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda,
           "python": sys.version.split()[0]}
    # TensorRT and ONNX Runtime GPU come from pip; the Kaggle image has neither. Plain "tensorrt"
    # resolves to a CUDA 13 build; the image has CUDA 12.8.
    # TensorRT 10.x: TensorRT 11 removed implicit (calibrator-based) INT8 quantization, which is
    # what wafer.trt_bench uses; the 10 line still supports it and the T4.
    pip("tensorrt-cu12<11", "onnx")
    ensure_ort_cuda()
    # Nsight Systems is not on the image. Try NVIDIA's CUDA apt repository; if that fails the kernel
    # table comes from torch.profiler (same CUPTI timings) and the summary says so.
    sh("bash", "-c", "set -e; cd /tmp && curl -fsSLo cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb "
       "&& dpkg -i cuda-keyring.deb >/dev/null && apt-get update -qq >/dev/null && apt-get install -y -qq cuda-nsight-systems-12-8 >/dev/null 2>&1 "
       "&& ln -sf /usr/local/cuda-12.8/bin/nsys /usr/local/bin/nsys || ln -sf $(ls -d /opt/nvidia/nsight-systems/*/bin/nsys 2>/dev/null | head -1) /usr/local/bin/nsys", check=False)
    import onnxruntime as ort
    import tensorrt as trt

    env.update({"tensorrt": trt.__version__, "onnxruntime": ort.__version__, "ort_providers": ort.get_available_providers()})
    sh("nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader", check=False)
    env["driver"] = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    env["tools"] = {t: shutil.which(t) for t in ("nsys", "ncu")}
    (RES / "trt_env.json").write_text(json.dumps(env, indent=2))
    print(env, flush=True)

    # 1. data (same loader and split as the main run)
    from wafer import data as D

    pkl = glob.glob("/kaggle/input/**/LSWMD.pkl", recursive=True)[0]
    data_npz = WORK / "wm811k_64.npz"
    t0 = time.time()
    df = D.load_labelled(pkl)
    x = D.to_array(df)
    y = df["y"].to_numpy()
    split = D.lot_grouped_split(df)
    np.savez_compressed(data_npz, x=x, y=y, train=split.train, val=split.val, test=split.test)
    del df
    print("data ready in", round(time.time() - t0), "s", flush=True)

    # 2. train once on 1 GPU and keep the weights
    weights = WORK / "cnn.pt"
    sh(PY, "-m", "wafer.train", "--data", data_npz, "--epochs", EPOCHS, "--out", RES / "trt_train.json", "--save", weights)

    # 3. export, build engines, benchmark, accuracy
    sh(PY, "-m", "wafer.trt_bench", "--data", data_npz, "--weights", weights, "--out-dir", RES, "--work", WORK / "trt_work")

    # 4. Nsight
    md = ["# Nsight summary", "", f"GPU {env['gpu']}, driver {env['driver']}, CUDA {env['cuda']}, TensorRT {env['tensorrt']}, "
          f"torch {env['torch']}. Produced on Kaggle by `kaggle/run_trt.py`; trace files are not committed.", ""]
    targets = (("median", [], "Custom median3x3 kernel (tiled), batch 1024, 50 launches"),
               ("trt", ["--engine", str(WORK / "trt_work" / "wafer_cnn_fp16.plan")], "TensorRT FP16 engine, batch 128, 50 runs"))
    if shutil.which("nsys"):
        for target, extra, label in targets:
            rep = WORK / f"nsys_{target}"
            r = sh("nsys", "profile", "-o", rep, "--force-overwrite", "true", "-t", "cuda,nvtx", PY, "-m", "wafer.profile_targets",
                   target, "--data", data_npz, *extra, check=False, capture=True)
            report = next(iter(glob.glob(str(rep) + ".nsys-rep")), None)
            if report:
                md.append(nsys_kernel_summary(Path(report), label + " (Nsight Systems)"))
            else:
                md.append(f"### {target}\n\nnsys profile failed:\n```\n{(r.stdout + r.stderr)[-800:]}\n```\n")
    else:
        md.append("Nsight Systems (`nsys`) is not on the Kaggle image and could not be installed, so the kernel tables below "
                  "come from `torch.profiler` (CUPTI, the same kernel timings Nsight Systems reports) via "
                  "`wafer.profile_targets --torch-profile`.\n")
        for target, extra, label in targets:
            out = WORK / f"torchprof_{target}.json"
            r = sh(PY, "-m", "wafer.profile_targets", target, "--data", data_npz, *extra, "--torch-profile", out, check=False, capture=True)
            if out.exists():
                prof = json.loads(out.read_text())
                lines = [f"### {label} (torch.profiler)", "", "| Time % | Total ms | Calls | Avg us | Kernel |", "|---|---|---|---|---|"]
                for k in prof["kernels"][:8]:
                    name = k["kernel"]
                    name = (name[:90] + "...") if len(name) > 90 else name
                    lines.append(f"| {k['time_pct']:.1f} | {k['total_us'] / 1e3:.2f} | {k['calls']} | {k['avg_us']:.1f} | `{name}` |")
                md.append("\n".join(lines) + "\n")
                shutil.copy(out, RES / out.name)
            else:
                md.append(f"### {target}\n\ntorch.profiler run failed:\n```\n{(r.stdout + r.stderr)[-800:]}\n```\n")
    if shutil.which("ncu"):
        r = sh("ncu", "--set", "default", "-k", "regex:median3x3", "--launch-skip", "10", "--launch-count", "1", PY, "-m",
               "wafer.profile_targets", "median", "--data", data_npz, "--iters", "3", check=False, capture=True)
        metrics = ncu_metrics(r.stdout)
        md.append("### Nsight Compute: median3x3_tiled_kernel (one launch)\n\n" + metrics + "\n")
        if r.returncode != 0 or metrics == "(no metrics parsed)":
            md.append("ncu output tail (exit " + str(r.returncode) + "):\n```\n" + (r.stdout + r.stderr)[-1200:] + "\n```\n")
    else:
        md.append("Nsight Compute (`ncu`) is not available on this Kaggle image, so no kernel metrics were collected.\n")
    (RES / "nsight_summary.md").write_text("\n".join(md))

    for f in sorted(RES.glob("*")):
        print("==", f.name, f.read_text()[:1200] if f.suffix in (".json", ".md") else "", flush=True)
    shutil.rmtree(WORK / "torch_ext", ignore_errors=True)
    shutil.rmtree(WORK / "trt_work", ignore_errors=True)
    for f in (data_npz, weights, *WORK.glob("nsys_*")):
        f.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
