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
    a = ap.parse_args()

    rank, world, device = setup()
    torch.manual_seed(a.seed + rank)
    d = np.load(a.data)
    x, y = d["x"], d["y"]
    tr, va, te = d["train"], d["val"], d["test"]
    x_tr = standardise(x[tr])
    y_tr = torch.from_numpy(y[tr])
    sampler = DistributedBalancedSampler(y[tr], num_replicas=world, rank=rank,
                                         samples_per_epoch=a.samples_per_epoch or len(tr), seed=a.seed)
    loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=a.batch, sampler=sampler, num_workers=2,
                        pin_memory=device.type == "cuda", drop_last=True)
    model = build(a.model, len(CLASSES)).to(device)
    if world > 1:
        model = DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * len(loader))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
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
               "test": test}
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
