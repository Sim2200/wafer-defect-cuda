"""Tests for train.py augment, oversample, and synthetic options."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from wafer.data import CLASSES
from wafer.train import Augment, oversample_indices


@pytest.fixture
def synthetic_data(tmp_path: Path):
    """Create a tiny synthetic dataset for testing."""
    np.random.seed(42)
    x = np.random.randint(0, 3, size=(300, 64, 64), dtype=np.uint8)
    y = np.concatenate([
        np.zeros(80, dtype=np.int64),
        np.ones(60, dtype=np.int64),
        np.full(50, 2, dtype=np.int64),
        np.full(40, 3, dtype=np.int64),
        np.full(30, 4, dtype=np.int64),
        np.full(20, 5, dtype=np.int64),
        np.full(15, 6, dtype=np.int64),
        np.full(3, 7, dtype=np.int64),
        np.full(2, 8, dtype=np.int64),
    ])
    np.random.shuffle(y)

    train = np.arange(200)
    val = np.arange(200, 250)
    test = np.arange(250, 300)

    npz_path = tmp_path / "data.npz"
    np.savez(npz_path, x=x, y=y, train=train, val=val, test=test)
    return npz_path


@pytest.fixture
def synthetic_wafers(tmp_path: Path):
    """Create a small synthetic wafers npz."""
    np.random.seed(123)
    x_synth = np.random.randint(0, 3, size=(20, 64, 64), dtype=np.uint8)
    y_synth = np.concatenate([
        np.zeros(8, dtype=np.int64),
        np.ones(5, dtype=np.int64),
        np.full(4, 2, dtype=np.int64),
        np.full(3, 3, dtype=np.int64),
    ])

    synth_path = tmp_path / "synthetic.npz"
    np.savez(synth_path, x=x_synth, y=y_synth)
    return synth_path


def run_train(repo_root: Path, data_path: Path, out_path: Path,
              synthetic: Path | None = None, augment: str = "none",
              oversample: float = 0.0) -> dict:
    """Run train.py via subprocess and return parsed output JSON."""
    cmd = [
        sys.executable, "-m", "wafer.train",
        "--data", str(data_path),
        "--out", str(out_path),
        "--epochs", "1",
        "--batch", "16",
        "--samples-per-epoch", "64",
        "--augment", augment,
        "--oversample", str(oversample),
    ]
    if synthetic:
        cmd.extend(["--synthetic", str(synthetic)])

    env = {"PYTHONPATH": str(repo_root / "src")}
    result = subprocess.run(cmd, cwd=repo_root, env=env, capture_output=True, text=True)
    assert result.returncode == 0, f"train.py failed:\n{result.stderr}"

    out_dict = json.loads(out_path.read_text())
    return out_dict


def test_default_no_synthetic(synthetic_data: Path, tmp_path: Path):
    """Default run should not have synthetic_wafers key or it should be 0."""
    repo_root = Path(__file__).parent.parent
    out = tmp_path / "out.json"

    result = run_train(repo_root, synthetic_data, out)

    assert result.get("synthetic_wafers", 0) == 0
    assert "synthetic_path" not in result


def test_augment_flips(synthetic_data: Path, tmp_path: Path):
    """--augment flips should record augment: flips."""
    repo_root = Path(__file__).parent.parent
    out = tmp_path / "out.json"

    result = run_train(repo_root, synthetic_data, out, augment="flips")

    assert result["augment"] == "flips"


def test_oversample_factor(synthetic_data: Path, tmp_path: Path):
    """--oversample should duplicate minority classes and record effective counts."""
    repo_root = Path(__file__).parent.parent
    out = tmp_path / "out.json"

    result = run_train(repo_root, synthetic_data, out, oversample=3.0)

    assert "train_counts_effective" in result
    assert "oversample_factor" in result
    assert result["oversample_factor"] == 3.0

    eff_counts = result["train_counts_effective"]
    orig_counts = result["train_counts"]

    orig_median = np.median([v for v in orig_counts.values() if v > 0])

    for cls_name, eff_count in eff_counts.items():
        orig_count = orig_counts.get(cls_name, 0)
        if orig_count > 0 and orig_count < orig_median:
            assert eff_count >= orig_count
            assert eff_count <= orig_median
        elif orig_count > 0:
            assert eff_count >= orig_count


def test_synthetic_wafers(synthetic_data: Path, synthetic_wafers: Path, tmp_path: Path):
    """--synthetic should record synthetic_wafers count and per-class counts."""
    repo_root = Path(__file__).parent.parent
    out = tmp_path / "out.json"

    result = run_train(repo_root, synthetic_data, out, synthetic=synthetic_wafers)

    assert result["synthetic_wafers"] == 20
    assert "synthetic_counts" in result
    assert result["synthetic_path"] == str(synthetic_wafers)

    synth_counts = result["synthetic_counts"]
    assert sum(synth_counts.values()) == 20


def test_augment_callable_shape():
    """Augment callable should preserve batch shape."""
    aug = Augment("flips", seed=42)

    batch = torch.randn(8, 1, 64, 64)
    result = aug(batch)

    assert result.shape == batch.shape


def test_augment_deterministic():
    """Augment with seeded generator should be deterministic."""
    seed = 100
    batch1 = torch.randn(4, 1, 64, 64)
    batch2 = batch1.clone()

    aug1 = Augment("flips", seed=seed)
    aug2 = Augment("flips", seed=seed)

    result1 = aug1(batch1)
    result2 = aug2(batch2)

    assert torch.allclose(result1, result2)


def test_augment_none_passthrough():
    """Augment with 'none' should not modify batch."""
    aug = Augment("none", seed=42)
    batch = torch.randn(2, 1, 64, 64)
    batch_orig = batch.clone()

    result = aug(batch)

    assert torch.allclose(result, batch_orig)


def test_oversample_indices_rule():
    """oversample_indices should respect duplication and median cap."""
    np.random.seed(99)
    y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2])
    indices, counts_after = oversample_indices(y, factor=2.0)

    y_oversampled = y[indices]
    orig_counts = np.bincount(y, minlength=len(CLASSES))
    orig_nonzero = orig_counts[orig_counts > 0]
    median = np.median(orig_nonzero)

    for c in range(3):
        cls_name = CLASSES[c]
        if orig_counts[c] == 0:
            assert counts_after[cls_name] == 0
        elif orig_counts[c] < median:
            assert counts_after[cls_name] >= orig_counts[c]
            assert counts_after[cls_name] <= median
        else:
            assert counts_after[cls_name] >= orig_counts[c]


def test_oversample_zero_factor():
    """oversample_indices with factor 0 should return unchanged indices."""
    y = np.array([0, 1, 1, 2])
    indices, counts = oversample_indices(y, factor=0.0)

    assert np.array_equal(indices, np.arange(len(y)))
    assert len(counts) == 0
