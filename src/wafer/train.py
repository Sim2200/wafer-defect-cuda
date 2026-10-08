"""Training with optional DistributedDataParallel.

Single GPU:   python -m wafer.train --data data/wm811k_64.npz --epochs 5 --out results/run_1gpu.json
Two GPUs:     torchrun --nproc_per_node=2 -m wafer.train --data ... --out results/run_2gpu.json

Every rank trains on its own balanced slice (DistributedBalancedSampler); gradients are averaged
by DDP. Rank 0 evaluates on the full validation set each epoch and on the test set at the end,
and writes a JSON with per-epoch throughput (wafers/s, all ranks combined), epoch time, and the
metrics from `wafer.metrics.report`.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset

from .data import CLASSES, class_counts
from .metrics import report, save_confusion_png
from .model import build
from .sampler import DistributedBalancedSampler


class Augment:
    def __init__(self, augment_type: str, seed: int, rank: int = 0):
        self.augment_type = augment_type
        self.gen = torch.Generator()
        self.gen.manual_seed(seed + rank)

    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        if self.augment_type == "none":
            return batch
        if self.augment_type != "flips":
            return batch

        b = batch.shape[0]
        for i in range(b):
            choice = torch.randint(0, 8, (1,), generator=self.gen).item()
            if choice & 1:
                batch[i] = torch.flip(batch[i], dims=[-2])
            if choice & 2:
                batch[i] = torch.flip(batch[i], dims=[-1])
            rot = (choice >> 2) % 4
            for _ in range(rot):
                batch[i] = torch.rot90(batch[i], dims=[-2, -1])
        return batch


def oversample_indices(y: np.ndarray, factor: float) -> tuple[np.ndarray, dict]:
    """Duplicate minority class indices until each class has at least factor x its original count, capped at median."""
    if factor <= 0:
        return np.arange(len(y)), {}

    counts = np.bincount(y)
    orig_counts = counts.copy()
    nonzero_counts = counts[counts > 0]
    if len(nonzero_counts) == 0:
        return np.arange(len(y)), {}
    median_count = np.median(nonzero_counts)

    indices = []
    for c in range(len(counts)):
        mask = np.where(y == c)[0]
        if len(mask) == 0:
            continue
        target = min(int(counts[c] * factor), int(median_count))
        if target > counts[c]:
            reps = np.random.choice(mask, size=target - counts[c], replace=True)
            indices.append(mask)
            indices.append(reps)
        else:
            indices.append(mask)

    indices_arr = np.concatenate(indices)
    new_y = y[indices_arr]
    binned = np.bincount(new_y, minlength=len(CLASSES))
    new_counts = {CLASSES[i]: int(binned[i]) for i in range(len(CLASSES))}

    return indices_arr, new_counts


def standardise(x: np.ndarray) -> torch.Tensor:
    """(N, 64, 64) uint8 -> (N, 1, 64, 64) float32, per-wafer standardised (same as the kernel)."""
    t = torch.from_numpy(x).to(torch.float32)
    mean = t.mean(dim=(1, 2), keepdim=True)
    std = t.std(dim=(1, 2), keepdim=True, unbiased=False)
    return ((t - mean) / (std + 1e-6)).unsqueeze(1)


def setup() -> tuple[int, int, torch.device]:
    if "RANK" in os.environ:
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
        rank, world = dist.get_rank(), dist.get_world_size()
    else:
        rank, world = 0, 1
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return rank, world, device


@torch.no_grad()
def predict(model: nn.Module, x: torch.Tensor, device: torch.device, batch: int = 1024) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(x), batch):
        out.append(model(x[i:i + batch].to(device, non_blocking=True)).argmax(1).cpu())
    return torch.cat(out).numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="cnn")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=256, help="per-rank batch size")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--samples-per-epoch", type=int, default=0, help="0 = one pass over the train size")
    ap.add_argument("--out", required=True)
    ap.add_argument("--confusion-png", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default="", help="write the trained weights (state_dict) here, for export")
    ap.add_argument("--augment", choices=["none", "flips"], default="none")
    ap.add_argument("--oversample", type=float, default=0.0, help="duplication factor for minority classes")
    ap.add_argument("--synthetic", default="", help="path to npz with generated wafers")
    a = ap.parse_args()

    rank, world, device = setup()
    torch.manual_seed(a.seed + rank)
    np.random.seed(a.seed + rank)
    d = np.load(a.data)
    x, y = d["x"], d["y"]
    tr, va, te = d["train"], d["val"], d["test"]

    x_tr_raw = x[tr].copy()
    y_tr_raw = y[tr].copy()

    synth_counts = {}
    n_synth = 0
    if a.synthetic:
        synth_d = np.load(a.synthetic)
        x_synth, y_synth = synth_d["x"], synth_d["y"]
        n_synth = len(x_synth)
        x_tr_raw = np.concatenate([x_tr_raw, x_synth])
        y_tr_raw = np.concatenate([y_tr_raw, y_synth])
        binned = np.bincount(y_synth, minlength=len(CLASSES))
        synth_counts = {CLASSES[i]: int(binned[i]) for i in range(len(CLASSES))}

    if a.oversample > 0:
        tr_indices, counts_after = oversample_indices(y_tr_raw, a.oversample)
        x_tr_raw = x_tr_raw[tr_indices]
        y_tr_raw = y_tr_raw[tr_indices]
    else:
        counts_after = {}

    x_tr = standardise(x_tr_raw)
    y_tr = torch.from_numpy(y_tr_raw)
    sampler = DistributedBalancedSampler(y_tr_raw, num_replicas=world, rank=rank,
                                         samples_per_epoch=a.samples_per_epoch or len(y_tr_raw), seed=a.seed)
    augment = Augment(a.augment, a.seed, rank)
    loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=a.batch, sampler=sampler, num_workers=2,
                        pin_memory=device.type == "cuda", drop_last=True)
    model = build(a.model, len(CLASSES)).to(device)
    if world > 1:
        model = DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * len(loader))
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    loss_fn = nn.CrossEntropyLoss()

    x_va, x_te = standardise(x[va]), standardise(x[te])
    epochs = []
    t_start = time.perf_counter()
    for epoch in range(a.epochs):
        sampler.set_epoch(epoch)
        model.train()
        n_seen, loss_sum = 0, 0.0
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for xb, yb in loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            xb = augment(xb)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss = loss_fn(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            n_seen += len(xb)
            loss_sum += loss.item() * len(xb)
        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_s = time.perf_counter() - t0
        seen = torch.tensor([n_seen], device=device)
        if world > 1:
            dist.all_reduce(seen)
        row = {"epoch": epoch, "epoch_seconds": round(epoch_s, 2), "wafers_seen_all_ranks": int(seen.item()),
               "wafers_per_second": round(float(seen.item()) / epoch_s, 1), "train_loss": round(loss_sum / max(1, n_seen), 4)}
        if rank == 0:
            val = report(y[va], predict(model.module if world > 1 else model, x_va, device))
            row.update({"val_macro_f1": val["macro_f1"], "val_accuracy": val["accuracy"]})
            print(json.dumps(row), flush=True)
        epochs.append(row)
    total_s = time.perf_counter() - t_start

    if rank == 0:
        core = model.module if world > 1 else model
        test = report(y[te], predict(core, x_te, device))
        out = {"model": a.model, "world_size": world, "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
               "epochs": a.epochs, "batch_per_rank": a.batch, "global_batch": a.batch * world, "lr": a.lr,
               "params": int(sum(p.numel() for p in core.parameters())), "sampler": "DistributedBalancedSampler (inverse-frequency)",
               "train_counts": class_counts(y[tr]), "test_counts": class_counts(y[te]),
               "total_train_seconds": round(total_s, 1), "per_epoch": epochs,
               "mean_wafers_per_second": round(float(np.mean([e["wafers_per_second"] for e in epochs[1:] or epochs])), 1),
               "test": test, "augment": a.augment}
        if a.synthetic:
            out["synthetic_path"] = a.synthetic
            out["synthetic_wafers"] = n_synth
            out["synthetic_counts"] = synth_counts
        if a.oversample > 0:
            out["oversample_factor"] = a.oversample
            out["train_counts_effective"] = counts_after
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(out, indent=2))
        if a.confusion_png:
            save_confusion_png(test["confusion"], a.confusion_png)
        if a.save:
            torch.save(core.state_dict(), a.save)
        print("test macro-F1", test["macro_f1"], "accuracy", test["accuracy"], "->", a.out, flush=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
