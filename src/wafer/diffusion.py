"""Class-conditional DDPM for 64x64 wafer maps (HF diffusers UNet2DModel), to generate rare defect classes.

    # train (DDP on 2 GPUs), only on the lot-grouped train split
    torchrun --nproc_per_node=2 -m wafer.diffusion train --data wm811k_64.npz --epochs 60 --out-dir ddpm
    # sample 2,000 wafers for each rare class with 50 DDIM steps
    python -m wafer.diffusion sample --ckpt ddpm/ckpt.pt --classes Near-full,Donut,Random,Scratch \
        --n-per-class 2000 --scheduler ddim --steps 50 --out synth.npz

Pixels: the npz stores maps as uint8 {0 background, 1 pass, 2 fail}. The model sees (x - 1) in
{-1, 0, 1} and predicts noise on that scale; generated floats are rounded back to the three
levels with `wafer.synth.from_signed`. The UNet is conditioned on the class through diffusers'
class embedding, so one model serves every class and the rare ones share what it learns from
the common ones (ring and edge structure, the circular wafer mask).

Training uses the balanced sampler from the classifier so each class is seen about equally per
epoch, an EMA copy of the weights for sampling (the usual DDPM practice; the raw weights give
noisier samples), fp16 autocast and a checkpoint after every epoch, so a run can resume if the
session ends.
"""

from __future__ import annotations

import argparse
import copy
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

from .data import CLASS_INDEX, CLASSES
from .sampler import DistributedBalancedSampler
from .synth import from_signed, to_signed
from .train import setup

DEFECT_CLASSES = [c for c in CLASSES if c != "none"]


def build_unet(channels: tuple[int, ...] = (64, 128, 256, 256), layers_per_block: int = 2) -> nn.Module:
    from diffusers import UNet2DModel

    n = len(channels)
    # attention only at the 16x16 level (third block): cheap and enough for 64x64 maps
    down = ["DownBlock2D"] * n
    up = ["UpBlock2D"] * n
    if n >= 3:
        down[2] = "AttnDownBlock2D"
        up[n - 3] = "AttnUpBlock2D"
    return UNet2DModel(sample_size=64, in_channels=1, out_channels=1, layers_per_block=layers_per_block,
                       block_out_channels=tuple(channels), down_block_types=tuple(down), up_block_types=tuple(up),
                       num_class_embeds=len(CLASSES))


def parse_classes(text: str) -> list[str]:
    names = [c.strip() for c in text.split(",")] if text else DEFECT_CLASSES
    unknown = [c for c in names if c not in CLASS_INDEX]
    if unknown:
        raise SystemExit(f"unknown classes {unknown}; choose from {CLASSES}")
    return names


@torch.no_grad()
def ema_update(ema: nn.Module, model: nn.Module, decay: float) -> None:
    for pe, pm in zip(ema.parameters(), model.parameters()):
        pe.mul_(decay).add_(pm.detach(), alpha=1 - decay)


def train(a: argparse.Namespace) -> None:
    from diffusers import DDPMScheduler

    rank, world, device = setup()
    torch.manual_seed(a.seed + rank)
    d = np.load(a.data)
    x, y, tr = d["x"], d["y"], d["train"]
    classes = parse_classes(a.classes)
    keep = tr[np.isin(y[tr], [CLASS_INDEX[c] for c in classes])]
    x_tr = torch.from_numpy(to_signed(x[keep])).unsqueeze(1)  # (N, 1, 64, 64) in {-1, 0, 1}
    y_tr = torch.from_numpy(y[keep])
    sampler = DistributedBalancedSampler(y[keep], num_replicas=world, rank=rank,
                                         samples_per_epoch=a.samples_per_epoch or len(keep), seed=a.seed)
    loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=a.batch, sampler=sampler, num_workers=2,
                        pin_memory=device.type == "cuda", drop_last=True)

    channels = tuple(int(c) for c in a.channels.split(","))
    model = build_unet(channels, a.layers_per_block).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    scheduler = DDPMScheduler(num_train_timesteps=a.timesteps, beta_schedule="squaredcos_cap_v2")
    # torch.amp.GradScaler exists from torch 2.3; the cuda one is the same class on older versions
    grad_scaler = getattr(torch.amp, "GradScaler", None)
    scaler = grad_scaler("cuda", enabled=device.type == "cuda") if grad_scaler else torch.cuda.amp.GradScaler(enabled=False)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "ckpt.pt"
    start_epoch, history = 0, []
    if a.resume and ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        start_epoch, history = ck["epoch"] + 1, ck["history"]
        if rank == 0:
            print(f"resumed from epoch {ck['epoch']}", flush=True)
    if world > 1:
        model = DistributedDataParallel(model, device_ids=[device.index])
    core = model.module if world > 1 else model

    t_start = time.perf_counter()
    for epoch in range(start_epoch, a.epochs):
        sampler.set_epoch(epoch)
        model.train()
        loss_sum, n = 0.0, 0
        t0 = time.perf_counter()
        for xb, yb in loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            noise = torch.randn_like(xb)
            t = torch.randint(0, a.timesteps, (xb.shape[0],), device=device)
            noisy = scheduler.add_noise(xb, noise, t)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                pred = model(noisy, t, class_labels=yb).sample
                loss = nn.functional.mse_loss(pred.float(), noise)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            ema_update(ema, core, a.ema)
            loss_sum += loss.item() * xb.shape[0]
            n += xb.shape[0]
        if device.type == "cuda":
            torch.cuda.synchronize()
        row = {"epoch": epoch, "loss": round(loss_sum / max(1, n), 5), "epoch_seconds": round(time.perf_counter() - t0, 1)}
        history.append(row)
        if rank == 0:
            print(json.dumps(row), flush=True)
            if (epoch + 1) % a.save_every == 0 or epoch + 1 == a.epochs:
                torch.save({"model": core.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "epoch": epoch,
                            "history": history, "channels": channels, "layers_per_block": a.layers_per_block,
                            "timesteps": a.timesteps, "classes": classes}, ckpt_path)
        if world > 1:
            dist.barrier()
    if rank == 0:
        (out_dir / "train.json").write_text(json.dumps({
            "model": "UNet2DModel class-conditional DDPM", "params": int(sum(p.numel() for p in core.parameters())),
            "channels": channels, "layers_per_block": a.layers_per_block, "timesteps": a.timesteps,
            "beta_schedule": "squaredcos_cap_v2", "prediction": "epsilon", "ema_decay": a.ema, "lr": a.lr,
            "batch_per_rank": a.batch, "world_size": world, "epochs": a.epochs, "classes_trained_on": classes,
            "train_wafers": int(len(keep)), "train_counts": {c: int((y[keep] == CLASS_INDEX[c]).sum()) for c in classes},
            "total_train_seconds": round(time.perf_counter() - t_start, 1), "per_epoch": history,
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"}, indent=2))
    if world > 1:
        dist.destroy_process_group()


@torch.no_grad()
def sample(a: argparse.Namespace) -> None:
    from diffusers import DDIMScheduler, DDPMScheduler

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location=device)
    model = build_unet(tuple(ck["channels"]), ck["layers_per_block"]).to(device).eval()
    model.load_state_dict(ck["ema"])
    if a.scheduler == "ddpm":
        sched = DDPMScheduler(num_train_timesteps=ck["timesteps"], beta_schedule="squaredcos_cap_v2")
    else:
        sched = DDIMScheduler(num_train_timesteps=ck["timesteps"], beta_schedule="squaredcos_cap_v2")
    sched.set_timesteps(a.steps)
    torch.manual_seed(a.seed)
    classes = parse_classes(a.classes)
    xs, ys = [], []
    t0 = time.perf_counter()
    for c in classes:
        label = CLASS_INDEX[c]
        done = 0
        while done < a.n_per_class:
            b = min(a.batch, a.n_per_class - done)
            x = torch.randn(b, 1, 64, 64, device=device)
            labels = torch.full((b,), label, device=device, dtype=torch.long)
            for t in sched.timesteps:
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    eps = model(x, t, class_labels=labels).sample.float()
                x = sched.step(eps, t, x).prev_sample
            xs.append(from_signed(x.squeeze(1).cpu()))
            ys.append(np.full(b, label, dtype=np.int64))
            done += b
    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    x_all, y_all = np.concatenate(xs), np.concatenate(ys)
    np.savez_compressed(a.out, x=x_all, y=y_all)
    info = {"ckpt": a.ckpt, "scheduler": a.scheduler, "steps": a.steps, "classes": classes, "n_per_class": a.n_per_class,
            "wafers": int(len(x_all)), "seconds": round(seconds, 1), "wafers_per_second": round(len(x_all) / seconds, 2),
            "batch": a.batch, "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"}
    Path(a.out).with_suffix(".json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--classes", default="", help="comma-separated; default: the eight defect classes (not 'none')")
    t.add_argument("--epochs", type=int, default=60)
    t.add_argument("--batch", type=int, default=128, help="per rank")
    t.add_argument("--lr", type=float, default=2e-4)
    t.add_argument("--ema", type=float, default=0.999)
    t.add_argument("--timesteps", type=int, default=1000)
    t.add_argument("--channels", default="64,128,256,256")
    t.add_argument("--layers-per-block", type=int, default=2)
    t.add_argument("--samples-per-epoch", type=int, default=0)
    t.add_argument("--save-every", type=int, default=1)
    t.add_argument("--resume", action="store_true")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--out-dir", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--ckpt", required=True)
    s.add_argument("--classes", default="Near-full,Donut,Random,Scratch")
    s.add_argument("--n-per-class", type=int, default=2000)
    s.add_argument("--scheduler", choices=["ddpm", "ddim"], default="ddim")
    s.add_argument("--steps", type=int, default=50)
    s.add_argument("--batch", type=int, default=256)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", required=True)
    a = ap.parse_args()
    (train if a.cmd == "train" else sample)(a)


if __name__ == "__main__":
    main()
