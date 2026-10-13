"""OpenAI Triton vs hand-written CUDA vs PyTorch for the 3x3 median and conv kernels, on one T4.
Pushed by scripts/run_on_kaggle.py --experiment triton. Writes results/kernels_triton.json (the
original results/kernels.json is left as it is) and results/triton_env.json.
"""

import glob
import json
import os
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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
PY = sys.executable


def sh(*args, check=True):
    print("$", " ".join(str(a) for a in args)[:300], flush=True)
    t0 = time.time()
    r = subprocess.run([str(a) for a in args], text=True)
    print(f"  -> exit {r.returncode} in {time.time() - t0:.0f} s", flush=True)
    if check and r.returncode != 0:
        raise SystemExit(f"step failed: {args[0]}")


def main():
    import torch
    import triton

    env = {"gpu": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0)), "torch": torch.__version__,
           "cuda": torch.version.cuda, "triton": triton.__version__, "python": sys.version.split()[0]}
    (RES / "triton_env.json").write_text(json.dumps(env, indent=2))
    print(env, flush=True)

    from wafer import data as D

    pkl = glob.glob("/kaggle/input/**/LSWMD.pkl", recursive=True)[0]
    df = D.load_labelled(pkl)
    x = D.to_array(df.head(4096))  # the benchmark uses the first 1,024 wafers; parity is checked on real maps
    np.savez_compressed(WORK / "wm811k_64.npz", x=x)
    del df

    # the Triton parity tests, then the benchmark with the Triton variants
    tests = subprocess.run([PY, "-m", "pytest", "-q", str(SRC / "tests" / "test_triton_kernels.py"), "-p", "no:cacheprovider",
                            "--rootdir", str(SRC)], text=True, capture_output=True)
    (RES / "triton_tests.txt").write_text(tests.stdout[-3000:] + tests.stderr[-1000:])
    print(tests.stdout[-800:], flush=True)
    if tests.returncode != 0:
        raise SystemExit("Triton parity tests failed")
    sh(PY, "-m", "wafer.bench_kernels", "--data", WORK / "wm811k_64.npz", "--out", RES / "kernels_triton.json", "--triton")
    print((RES / "kernels_triton.json").read_text()[:3000], flush=True)


if __name__ == "__main__":
    main()
