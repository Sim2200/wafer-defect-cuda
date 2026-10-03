"""WM-811K loading, labelling and a wafer-level stratified split.

LSWMD.pkl is a pandas DataFrame with one row per wafer: `waferMap` (a 2-D uint8 array of
0 = outside the wafer, 1 = good die, 2 = defective die; sizes vary from ~6x21 to 300x202),
`failureType` (an array holding the label, or empty for the ~639K unlabelled wafers),
`lotName`, `waferIndex`, `dieSize`, `trianTestLabel`.

Only labelled wafers are used (nine classes). The split is stratified by class and grouped
by lot so that wafers from one lot never straddle train and test (wafers in a lot share a
process window and look alike; splitting them would leak).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "Random", "Scratch", "none"]
CLASS_INDEX = {c: i for i, c in enumerate(CLASSES)}
SIZE = 64


def label_of(value) -> str | None:
    """`failureType` is an ndarray like [['Center']] for labelled wafers and [] otherwise."""
    if isinstance(value, str):
        return value if value in CLASS_INDEX else None
    if isinstance(value, (list, np.ndarray)) and len(value) > 0:
        v = value[0]
        v = v[0] if isinstance(v, (list, np.ndarray)) and len(v) > 0 else v
        return str(v) if str(v) in CLASS_INDEX else None
    return None


def load_labelled(pkl_path: str | Path) -> pd.DataFrame:
    df = pd.read_pickle(pkl_path)
    df["label"] = df["failureType"].map(label_of)
    df = df[df["label"].notna()].reset_index(drop=True)
    df["y"] = df["label"].map(CLASS_INDEX).astype(np.int64)
    return df[["waferMap", "lotName", "waferIndex", "label", "y"]]


def resize_nearest(wafer: np.ndarray, size: int = SIZE) -> np.ndarray:
    """Nearest-neighbour resize to size x size with the "nearest-exact" index rule
    (src = floor((dst + 0.5) * in / out)), the same rule as the CUDA kernel and torch."""
    h, w = wafer.shape
    rows = np.minimum(((np.arange(size) + 0.5) * h / size).astype(np.int64), h - 1)
    cols = np.minimum(((np.arange(size) + 0.5) * w / size).astype(np.int64), w - 1)
    return wafer[rows][:, cols].astype(np.uint8)


def to_array(df: pd.DataFrame, size: int = SIZE) -> np.ndarray:
    """Stack every wafer map as a (N, size, size) uint8 array (values 0, 1, 2)."""
    out = np.empty((len(df), size, size), dtype=np.uint8)
    for i, m in enumerate(df["waferMap"].to_numpy()):
        out[i] = resize_nearest(np.asarray(m), size)
    return out


@dataclass(frozen=True)
class Split:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def lot_grouped_split(df: pd.DataFrame, val_frac: float = 0.1, test_frac: float = 0.2, seed: int = 0) -> Split:
    """Index arrays for train/val/test. Lots are assigned whole; the assignment is done per class
    of the lot's majority label so every class lands in every split in roughly the right share."""
    rng = np.random.default_rng(seed)
    lot_label = df.groupby("lotName")["y"].agg(lambda s: s.value_counts().idxmax())
    train, val, test = [], [], []
    for cls in range(len(CLASSES)):
        lots = lot_label[lot_label == cls].index.to_numpy()
        rng.shuffle(lots)
        n_test = int(round(len(lots) * test_frac))
        n_val = int(round(len(lots) * val_frac))
        test += list(lots[:n_test])
        val += list(lots[n_test:n_test + n_val])
        train += list(lots[n_test + n_val:])
    lot_to_split = {**{l: 0 for l in train}, **{l: 1 for l in val}, **{l: 2 for l in test}}
    assignment = df["lotName"].map(lot_to_split).to_numpy()
    idx = np.arange(len(df))
    return Split(idx[assignment == 0], idx[assignment == 1], idx[assignment == 2])


def class_counts(y: np.ndarray) -> dict[str, int]:
    counts = np.bincount(y, minlength=len(CLASSES))
    return {c: int(counts[i]) for i, c in enumerate(CLASSES)}


def prepare(pkl_path: str | Path, out_path: str | Path, seed: int = 0) -> dict:
    """Load, label, resize and split once; save a compact npz plus a JSON summary."""
    df = load_labelled(pkl_path)
    x = to_array(df)
    y = df["y"].to_numpy()
    split = lot_grouped_split(df, seed=seed)
    np.savez_compressed(out_path, x=x, y=y, train=split.train, val=split.val, test=split.test)
    summary = {
        "labelled_wafers": int(len(df)),
        "lots": int(df["lotName"].nunique()),
        "size": SIZE,
        "classes": CLASSES,
        "counts": {"all": class_counts(y), "train": class_counts(y[split.train]),
                   "val": class_counts(y[split.val]), "test": class_counts(y[split.test])},
        "split_sizes": {"train": int(len(split.train)), "val": int(len(split.val)), "test": int(len(split.test))},
        "none_share": round(float((y == CLASS_INDEX["none"]).mean()), 4),
        "split": "lot-grouped, per-class stratified by lot majority label, seed %d" % seed,
    }
    Path(out_path).with_suffix(".json").write_text(json.dumps(summary, indent=2))
    return summary
