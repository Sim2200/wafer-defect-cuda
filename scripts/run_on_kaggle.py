"""Run the experiment on Kaggle's free 2x T4 and pull the results back.

1. Packages src/ and kernels/ into a private Kaggle dataset (<user>/wafer-defect-cuda-src) so the
   kernel can import the code (kernels can only attach datasets, not git repos).
2. Pushes kaggle/run_all.py as a private GPU script kernel attached to that dataset and to
   qingyi/wm811k-wafer-map, polls until it finishes, then downloads /kaggle/working/results/*
   into results/.

    python scripts/run_on_kaggle.py                     # push source, run, wait, pull
    python scripts/run_on_kaggle.py --pull-only
    python scripts/run_on_kaggle.py --experiment trt    # kaggle/run_trt.py: TensorRT engines + Nsight (1 GPU,
                                                        # internet on for the tensorrt/onnxruntime-gpu wheels)
    python scripts/run_on_kaggle.py --experiment diffusion   # kaggle/run_diffusion.py: DDPM for rare classes (2 GPUs)
Needs the Kaggle CLI authenticated (~/.kaggle/kaggle.json or ~/.kaggle/access_token).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KAGGLE = [sys.executable, "-m", "kaggle"]


def kaggle(*args: str, check: bool = True) -> str:
    r = subprocess.run([*KAGGLE, *args], text=True, capture_output=True)
    if check and r.returncode != 0:
        raise SystemExit(f"kaggle {' '.join(args)} failed:\n{r.stdout}\n{r.stderr}")
    return r.stdout + r.stderr


def username(explicit: str | None) -> str:
    if explicit:
        return explicit
    out = kaggle("config", "view", check=False)
    for line in out.splitlines():
        if line.strip().startswith("username") and ":" in line:
            v = line.split(":", 1)[1].strip()
            if v and v != "None":
                return v
    raise SystemExit("pass --user <kaggle username> (access-token auth does not expose it)")


def push_source(user: str) -> str:
    slug = f"{user}/wafer-defect-cuda-src"
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        for sub in ("src", "kernels", "tests", "kaggle"):
            shutil.copytree(ROOT / sub, d / sub, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        (d / "dataset-metadata.json").write_text(json.dumps({"title": "wafer-defect-cuda-src", "id": slug, "licenses": [{"name": "CC0-1.0"}]}))
        exists = slug.split("/")[1] in kaggle("datasets", "list", "--mine", check=False)
        out = kaggle("datasets", "version" if exists else "create", "-p", str(d), *(["-m", "update"] if exists else []), "--dir-mode", "zip", check=False)
        print(out.strip().splitlines()[-1])
    time.sleep(20)  # the new version takes a moment to become attachable
    return slug


EXPERIMENTS = {  # name -> (kernel script, internet needed)
    "all": ("run_all.py", False),
    "trt": ("run_trt.py", True),
    "diffusion": ("run_diffusion.py", True),  # internet for the diffusers wheel
    "nsight": ("run_nsight.py", True),  # run_trt.py in profiling-only mode
}


def push_kernel(user: str, src_slug: str, epochs: int, experiment: str = "all") -> str:
    script, internet = EXPERIMENTS[experiment]
    kid = f"{user}/wafer-defect-cuda-{experiment}" if experiment != "all" else f"{user}/wafer-defect-cuda-run"
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        shutil.copy(ROOT / "kaggle" / script, d / script)
        sources = ["qingyi/wm811k-wafer-map", src_slug]
        if experiment == "diffusion" and f"{user}/wafer-defect-cuda-ddpm" in kaggle("datasets", "list", "--mine", check=False):
            sources.append(f"{user}/wafer-defect-cuda-ddpm")  # a finished DDPM run's checkpoint and samples (see push_ddpm)
        (d / "kernel-metadata.json").write_text(json.dumps({
            "id": kid, "title": kid.split("/")[1], "code_file": script, "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": internet,
            "dataset_sources": sources, "competition_sources": [], "kernel_sources": []}))
        print(kaggle("kernels", "push", "-p", str(d)).strip().splitlines()[-1])
    return kid


def push_ddpm(user: str, folder: Path) -> None:
    """Upload a finished diffusion run's ddpm/ckpt.pt, synth_*.npz and study_*.npz (+ .json) as the
    private dataset <user>/wafer-defect-cuda-ddpm, so the evaluation can be redone without training."""
    slug = f"{user}/wafer-defect-cuda-ddpm"
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "ddpm").mkdir()
        shutil.copy(folder / "ddpm" / "ckpt.pt", d / "ddpm" / "ckpt.pt")
        shutil.copy(folder / "ddpm" / "train.json", d / "ddpm" / "train.json")
        for f in list(folder.glob("synth_*.npz")) + list(folder.glob("study_*.npz")) + list(folder.glob("*.json")):
            if f.name.startswith(("synth_", "study_")):
                shutil.copy(f, d / f.name)
        (d / "dataset-metadata.json").write_text(json.dumps({"title": "wafer-defect-cuda-ddpm", "id": slug, "licenses": [{"name": "CC0-1.0"}]}))
        exists = slug.split("/")[1] in kaggle("datasets", "list", "--mine", check=False)
        out = kaggle("datasets", "version" if exists else "create", "-p", str(d), *(["-m", "update"] if exists else []), "--dir-mode", "zip", check=False)
        print(out.strip().splitlines()[-1])


def wait(kid: str, poll: int = 30) -> str:
    t0 = time.time()
    # Right after a push the status endpoint still reports the previous version for a while;
    # wait until the new version shows up as queued or running before waiting for completion.
    while time.time() - t0 < 300:
        status = kaggle("kernels", "status", kid, check=False).strip().splitlines()[-1].lower()
        if "queued" in status or "running" in status:
            break
        time.sleep(10)
    while True:
        status = kaggle("kernels", "status", kid, check=False).strip().splitlines()[-1]
        print(f"  {time.time() - t0:6.0f} s  {status}", flush=True)
        if any(k in status.lower() for k in ("complete", "error", "cancel")):
            return status
        time.sleep(poll)


def pull(kid: str) -> None:
    out = ROOT / "kaggle" / "out"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    print(kaggle("kernels", "output", kid, "-p", str(out)).strip().splitlines()[-1])
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    for f in out.rglob("*"):
        if f.suffix in (".json", ".png", ".txt", ".md") and f.is_file():
            shutil.copy(f, results / f.name)
            print("  pulled", f.name)
    log = next(out.glob("*.log"), None)
    if log:
        lines = []
        for line in log.read_text().splitlines():
            line = line.strip().lstrip(",")
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            if o.get("stream_name") == "stdout" or "Error" in o.get("data", "") or "error" in o.get("data", ""):
                lines.append(o["data"].rstrip())
        (results / "kaggle_run.log").write_text("\n".join(lines))
        print("\n".join(lines[-25:]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pull-only", action="store_true")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--user", default="", help="Kaggle username")
    ap.add_argument("--experiment", default="all", choices=sorted(EXPERIMENTS))
    ap.add_argument("--push-ddpm", default="", help="folder with a finished run's ddpm/, synth_*.npz, study_*.npz to upload as a dataset")
    a = ap.parse_args()
    user = username(a.user or None)
    if a.push_ddpm:
        push_ddpm(user, Path(a.push_ddpm))
        return
    kid = f"{user}/wafer-defect-cuda-{a.experiment}" if a.experiment != "all" else f"{user}/wafer-defect-cuda-run"
    if not a.pull_only:
        slug = push_source(user)
        kid = push_kernel(user, slug, a.epochs, a.experiment)
        status = wait(kid)
        print("final:", status)
    pull(kid)


if __name__ == "__main__":
    main()
