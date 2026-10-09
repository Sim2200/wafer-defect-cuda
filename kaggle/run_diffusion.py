"""Diffusion for rare defect classes, on Kaggle's 2x T4. Pushed by scripts/run_on_kaggle.py --experiment diffusion.

Writes under /kaggle/working/results:
  diffusion_train.json       the class-conditional DDPM training run (loss per epoch, time, config)
  diffusion_samples.json     generated sets: memorisation check (nearest-neighbour distance to the
                             train split, vs real test wafers), feature-FID per class, DDPM vs DDIM
                             step counts (samples/s vs FID)
  diffusion_experiment.json  the classifier experiment: baseline, flips, oversampling, +500 and
                             +2000 synthetic wafers per rare class; 3 seeds each, mean and std of
                             test macro-F1 and per-class recall
  figures/diffusion_grid.png real vs generated wafers per rare class

Time budget on 2x T4 (session limit 12 h): data ~1 min, DDPM training ~EPOCHS x 20 s on 2 GPUs,
sampling ~5 min, 15 classifier runs ~15 min (two at a time, one per GPU).
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

WORK = Path("/kaggle/working")
RES = WORK / "results"
FIG = RES / "figures"
RES.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
SRC = Path(glob.glob("/kaggle/input/**/src/wafer/data.py", recursive=True)[0]).parents[2]
sys.path.insert(0, str(SRC / "src"))
os.environ["PYTHONPATH"] = str(SRC / "src")
PY = sys.executable
EPOCHS = int(os.environ.get("DDPM_EPOCHS", "60"))
CLS_EPOCHS = int(os.environ.get("EPOCHS", "6"))
RARE = ["Near-full", "Donut", "Random", "Scratch"]
SEEDS = (0, 1, 2)
SYNTH_SIZES = (500, 2000)


def sh(*args, check=True, env=None, capture=False):
    print("$", " ".join(str(a) for a in args)[:300], flush=True)
    t0 = time.time()
    r = subprocess.run([str(a) for a in args], text=True, env={**os.environ, **(env or {})}, capture_output=capture)
    print(f"  -> exit {r.returncode} in {time.time() - t0:.0f} s", flush=True)
    if check and r.returncode != 0:
        if capture:
            print(r.stdout[-2000:], r.stderr[-2000:])
        raise SystemExit(f"step failed: {args[0]}")
    return r


def run_pair(jobs: list[tuple[list[str], dict]]) -> None:
    """Run classifier trainings two at a time, one per GPU."""
    pending = list(jobs)
    running: list[tuple[subprocess.Popen, str]] = []
    while pending or running:
        while pending and len(running) < 2:
            gpu = next(g for g in ("0", "1") if g not in {r[1] for r in running})
            cmd, env = pending.pop(0)
            print("$ [GPU", gpu + "]", " ".join(cmd)[:200], flush=True)
            running.append((subprocess.Popen(cmd, env={**os.environ, "CUDA_VISIBLE_DEVICES": gpu, **env},
                                             stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT), gpu))
        time.sleep(5)
        still = []
        for p, gpu in running:
            if p.poll() is None:
                still.append((p, gpu))
            elif p.returncode != 0:
                raise SystemExit(f"classifier run failed on GPU {gpu}")
        running = still


def main():
    import torch

    from wafer import data as D
    from wafer.model import build
    from wafer.synth import classifier_features, frechet_distance, memorization_report, nearest_neighbour_distance, sample_grid
    from wafer.train import standardise

    sh(PY, "-m", "pip", "install", "-q", "diffusers", check=False)
    import diffusers

    env = {"gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())], "torch": torch.__version__,
           "cuda": torch.version.cuda, "diffusers": diffusers.__version__, "python": sys.version.split()[0]}
    print(env, flush=True)

    # 1. data
    pkl = glob.glob("/kaggle/input/**/LSWMD.pkl", recursive=True)[0]
    data_npz = WORK / "wm811k_64.npz"
    df = D.load_labelled(pkl)
    x, y = D.to_array(df), df["y"].to_numpy()
    split = D.lot_grouped_split(df)
    np.savez_compressed(data_npz, x=x, y=y, train=split.train, val=split.val, test=split.test)
    del df
    tr, te = split.train, split.test

    # 2. train the DDPM (DDP on both GPUs), resumable. A finished run's checkpoint and samples can
    # be attached as a dataset (wafer-defect-cuda-ddpm) to skip the 3 GPU-hours of training and
    # sampling and redo only the evaluation.
    ddpm = WORK / "ddpm"
    prior = next(iter(glob.glob("/kaggle/input/**/ddpm/ckpt.pt", recursive=True)), None)
    if prior:
        src = Path(prior).parents[1]
        print("reusing checkpoint and samples from", src, flush=True)
        shutil.copytree(src / "ddpm", ddpm, dirs_exist_ok=True)
        for f in src.glob("*.npz"):
            shutil.copy(f, WORK / f.name)
        for f in src.glob("*.json"):
            shutil.copy(f, WORK / f.name)
    else:
        sh(PY, "-m", "torch.distributed.run", "--nproc_per_node=2", "--standalone", "-m", "wafer.diffusion", "train",
           "--data", data_npz, "--epochs", EPOCHS, "--out-dir", ddpm, "--resume")
    shutil.copy(ddpm / "train.json", RES / "diffusion_train.json")

    # 3. samples: 2,000 per rare class with DDIM-50 (the bulk set), plus the step-count study
    synth = WORK / "synth_2000.npz"
    if not synth.exists():
        sh(PY, "-m", "wafer.diffusion", "sample", "--ckpt", ddpm / "ckpt.pt", "--classes", ",".join(RARE), "--n-per-class", 2000,
           "--scheduler", "ddim", "--steps", 50, "--batch", 500, "--out", synth)
    g = np.load(synth)
    rng = np.random.default_rng(0)
    keep = np.concatenate([rng.choice(np.where(g["y"] == D.CLASS_INDEX[c])[0], 500, replace=False) for c in RARE])
    np.savez_compressed(WORK / "synth_500.npz", x=g["x"][keep], y=g["y"][keep])
    steps_study = {}
    for sched, steps in (("ddpm", 1000), ("ddim", 50), ("ddim", 10)):
        out = WORK / f"study_{sched}{steps}.npz"
        if not out.exists():
            sh(PY, "-m", "wafer.diffusion", "sample", "--ckpt", ddpm / "ckpt.pt", "--classes", ",".join(RARE), "--n-per-class", 256,
               "--scheduler", sched, "--steps", steps, "--batch", 512, "--out", out)
        steps_study[f"{sched}_{steps}"] = json.loads(out.with_suffix(".json").read_text())

    # 4. baseline classifier (seed 0) once more with saved weights: its penultimate features give the FID
    base_w = WORK / "cnn_base.pt"
    sh(PY, "-m", "wafer.train", "--data", data_npz, "--epochs", CLS_EPOCHS, "--seed", 0, "--out", WORK / "cls_base0.json", "--save", base_w,
       env={"CUDA_VISIBLE_DEVICES": "0"})
    dev = torch.device("cuda:0")
    clf = build("cnn", len(D.CLASSES)).to(dev)
    clf.load_state_dict(torch.load(base_w, map_location=dev))

    def feats(arr: np.ndarray) -> np.ndarray:
        return classifier_features(clf, standardise(arr), dev)

    x_tr, y_tr, x_te, y_te = x[tr], y[tr], x[te], y[te]
    per_class = {}
    real_dict, gen_dict = {}, {}
    for c in RARE:
        k = D.CLASS_INDEX[c]
        real_te = x_te[y_te == k]
        real_tr = x_tr[y_tr == k]
        gen = g["x"][g["y"] == k]
        d_gen = nearest_neighbour_distance(gen, x_tr, device="cuda")
        d_test = nearest_neighbour_distance(real_te, x_tr, device="cuda")
        f_tr, f_te, f_gen = feats(real_tr), feats(real_te), feats(gen)
        per_class[c] = {
            "train_wafers": int(len(real_tr)), "test_wafers": int(len(real_te)), "generated": int(len(gen)),
            "memorization": memorization_report(d_gen, d_test),
            "fid_generated_vs_real_test": round(frechet_distance(f_gen, f_te), 3),
            "fid_real_train_vs_real_test": round(frechet_distance(f_tr, f_te), 3),
            "fid_generated_vs_real_train": round(frechet_distance(f_gen, f_tr), 3),
            "fid_by_steps_vs_real_test": {},
            "fail_pixel_share": {"real_train": round(float((real_tr == 2).mean()), 4), "generated": round(float((gen == 2).mean()), 4)},
        }
        for name in steps_study:
            s = np.load(WORK / f"study_{name.replace('_', '')}.npz")
            per_class[c]["fid_by_steps_vs_real_test"][name] = round(frechet_distance(feats(s["x"][s["y"] == k]), f_te), 3)
        real_dict[c], gen_dict[c] = real_tr[:8], gen[:8]
    sample_grid(real_dict, gen_dict, RARE, 8, str(FIG / "diffusion_grid.png"))
    (RES / "diffusion_samples.json").write_text(json.dumps({
        "env": env, "rare_classes": RARE, "bulk_set": json.loads(synth.with_suffix(".json").read_text()),
        "steps_study": steps_study, "per_class": per_class,
        "fid_features": "256-d penultimate features of the baseline WaferCNN (seed 0), not Inception",
        "memorization_metric": "fraction of differing pixels to the nearest train wafer; threshold = 1st percentile of real test wafers' distance to train"},
        indent=2))

    # 5. the classifier experiment: 5 conditions x 3 seeds, two runs at a time
    conditions = {"baseline": [], "flips": ["--augment", "flips"], "oversample_x4": ["--oversample", "4"],
                  "synthetic_500": ["--synthetic", str(WORK / "synth_500.npz")], "synthetic_2000": ["--synthetic", str(synth)]}
    jobs = []
    for cond, extra in conditions.items():
        for seed in SEEDS:
            out = WORK / f"cls_{cond}_s{seed}.json"
            jobs.append(([PY, "-m", "wafer.train", "--data", str(data_npz), "--epochs", str(CLS_EPOCHS), "--seed", str(seed),
                          "--out", str(out), *extra], {}))
    run_pair(jobs)
    summary = {"conditions": {}, "seeds": list(SEEDS), "epochs": CLS_EPOCHS, "classes": D.CLASSES, "rare_classes": RARE}
    for cond in conditions:
        runs = [json.loads((WORK / f"cls_{cond}_s{s}.json").read_text()) for s in SEEDS]
        f1 = np.array([r["test"]["macro_f1"] for r in runs])
        acc = np.array([r["test"]["accuracy"] for r in runs])
        rec = {c: np.array([r["test"]["per_class"][c]["recall"] for r in runs]) for c in D.CLASSES}
        summary["conditions"][cond] = {
            "macro_f1_mean": round(float(f1.mean()), 4), "macro_f1_std": round(float(f1.std()), 4), "macro_f1_runs": f1.round(4).tolist(),
            "accuracy_mean": round(float(acc.mean()), 4), "accuracy_std": round(float(acc.std()), 4),
            "per_class_recall_mean": {c: round(float(v.mean()), 4) for c, v in rec.items()},
            "per_class_recall_std": {c: round(float(v.std()), 4) for c, v in rec.items()},
            "train_counts_effective": runs[0].get("train_counts_effective", runs[0]["train_counts"]),
            "synthetic_wafers": runs[0].get("synthetic_wafers", 0),
        }
    (RES / "diffusion_experiment.json").write_text(json.dumps(summary, indent=2))
    for f in sorted(RES.glob("*.json")):
        print("==", f.name, f.read_text()[:1500], flush=True)
    # the checkpoint and samples stay in the kernel output (about 0.5 GB) so a later run can reuse them
    data_npz.unlink(missing_ok=True)
    (ddpm / "ckpt.pt").exists() and torch.save({k: v for k, v in torch.load(ddpm / "ckpt.pt", map_location="cpu").items() if k != "opt"},
                                               ddpm / "ckpt.pt")  # drop the optimizer state: sampling needs only the EMA weights


if __name__ == "__main__":
    main()
