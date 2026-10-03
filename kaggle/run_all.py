"""The whole experiment on one Kaggle kernel (2x T4). Pushed by scripts/run_on_kaggle.py.

Reads the project source from the attached private dataset (wafer-defect-cuda-src) and the
WM-811K pickle from the attached public dataset; writes everything under /kaggle/working/results:

  data_summary.json   labelled wafers, lots, class counts per split
  kernels.json        correctness + timing of the CUDA kernels vs PyTorch
  run_1gpu.json       training on 1 GPU (metrics on the held-out test set)
  run_2gpu.json       the same with torchrun on 2 GPUs (DDP)
  metrics.json        the 1-GPU test report (macro-F1, per-class, confusion) + confusion_matrix.png
  ddp_scaling.json    throughput, epoch time, scaling efficiency, accuracy parity
  env.json            GPU names, torch/CUDA versions
"""

import glob
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

WORK = Path("/kaggle/working")
RES = WORK / "results"
RES.mkdir(parents=True, exist_ok=True)
SRC = Path(glob.glob("/kaggle/input/**/src/wafer/data.py", recursive=True)[0]).parents[2]
sys.path.insert(0, str(SRC / "src"))
os.environ["PYTHONPATH"] = str(SRC / "src")
os.environ["TORCH_EXTENSIONS_DIR"] = str(WORK / "torch_ext")
EPOCHS = int(os.environ.get("EPOCHS", "6"))

from wafer import data as D  # noqa: E402


def sh(*args, env=None):
    print("$", " ".join(args), flush=True)
    t0 = time.time()
    r = subprocess.run(args, text=True, env={**os.environ, **(env or {})})
    print(f"  -> exit {r.returncode} in {time.time() - t0:.0f} s", flush=True)
    if r.returncode != 0:
        raise SystemExit(f"step failed: {args[0]}")


def main():
    env = {"gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
           "torch": torch.__version__, "cuda": torch.version.cuda, "python": sys.version.split()[0],
           "cpus": os.cpu_count()}
    (RES / "env.json").write_text(json.dumps(env, indent=2))
    print(env, flush=True)

    # 1. data
    pkl = glob.glob("/kaggle/input/**/LSWMD.pkl", recursive=True)[0]
    data_npz = WORK / "wm811k_64.npz"
    if not data_npz.exists():
        t0 = time.time()
        df = D.load_labelled(pkl)
        print("labelled", len(df), "loaded in", round(time.time() - t0), "s", flush=True)
        x = D.to_array(df)
        y = df["y"].to_numpy()
        split = D.lot_grouped_split(df)
        np.savez_compressed(data_npz, x=x, y=y, train=split.train, val=split.val, test=split.test)
        summary = {"labelled_wafers": int(len(df)), "lots": int(df["lotName"].nunique()), "size": D.SIZE, "classes": D.CLASSES,
                   "counts": {"all": D.class_counts(y), "train": D.class_counts(y[split.train]),
                              "val": D.class_counts(y[split.val]), "test": D.class_counts(y[split.test])},
                   "split_sizes": {"train": int(len(split.train)), "val": int(len(split.val)), "test": int(len(split.test))},
                   "none_share": round(float((y == D.CLASS_INDEX["none"]).mean()), 4),
                   "split": "lot-grouped, per-class stratified by lot majority label, seed 0",
                   "raw_shape_examples": {str(k): int(v) for k, v in df["waferMap"].head(20000).apply(lambda m: m.shape).value_counts().head(8).items()}}
        (RES / "data_summary.json").write_text(json.dumps(summary, indent=2))
        rng = np.random.default_rng(0)
        pick = rng.choice(len(df), size=1024, replace=False)
        maps = np.empty(1024, dtype=object)
        for i, j in enumerate(pick):
            maps[i] = np.asarray(df["waferMap"].iloc[j], dtype=np.uint8)
        np.savez(WORK / "raw_sample.npz", maps=maps)
        del df
        print(json.dumps(summary["counts"]["all"]), flush=True)

    # 2. kernels
    sh(sys.executable, "-m", "wafer.bench_kernels", "--data", str(data_npz), "--raw", str(WORK / "raw_sample.npz"),
       "--out", str(RES / "kernels.json"))

    # 3. train on 1 GPU, 4. on 2 GPUs
    sh(sys.executable, "-m", "wafer.train", "--data", str(data_npz), "--epochs", str(EPOCHS), "--out", str(RES / "run_1gpu.json"),
       "--confusion-png", str(RES / "confusion_matrix.png"), env={"CUDA_VISIBLE_DEVICES": "0"})
    sh(sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=2", "--standalone", "-m", "wafer.train", "--data", str(data_npz),
       "--epochs", str(EPOCHS), "--out", str(RES / "run_2gpu.json"))

    one = json.loads((RES / "run_1gpu.json").read_text())
    two = json.loads((RES / "run_2gpu.json").read_text())
    (RES / "metrics.json").write_text(json.dumps({"model": one["model"], "params": one["params"], "epochs": one["epochs"],
                                                   "test": one["test"], "test_counts": one["test_counts"]}, indent=2))
    tp1, tp2 = one["mean_wafers_per_second"], two["mean_wafers_per_second"]
    ep1 = float(np.mean([e["epoch_seconds"] for e in one["per_epoch"][1:]]))
    ep2 = float(np.mean([e["epoch_seconds"] for e in two["per_epoch"][1:]]))
    (RES / "ddp_scaling.json").write_text(json.dumps({
        "gpus": env["gpus"], "epochs": EPOCHS, "batch_per_rank": one["batch_per_rank"],
        "wafers_per_second": {"1_gpu": tp1, "2_gpu": tp2}, "speedup": round(tp2 / tp1, 3), "scaling_efficiency": round(tp2 / tp1 / 2, 3),
        "mean_epoch_seconds_excluding_first": {"1_gpu": round(ep1, 2), "2_gpu": round(ep2, 2)},
        "total_train_seconds": {"1_gpu": one["total_train_seconds"], "2_gpu": two["total_train_seconds"]},
        "test_macro_f1": {"1_gpu": one["test"]["macro_f1"], "2_gpu": two["test"]["macro_f1"]},
        "test_accuracy": {"1_gpu": one["test"]["accuracy"], "2_gpu": two["test"]["accuracy"]},
        "note": "same model, same per-rank batch (global batch doubles on 2 GPUs), same epochs; throughput is wafers seen by all ranks per second, mean over epochs after the first",
    }, indent=2))
    for f in sorted(RES.glob("*.json")):
        print("==", f.name, f.read_text()[:1500], flush=True)
    shutil.rmtree(WORK / "torch_ext", ignore_errors=True)
    (WORK / "wm811k_64.npz").unlink(missing_ok=True)
    (WORK / "raw_sample.npz").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
