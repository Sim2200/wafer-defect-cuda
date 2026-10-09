"""Diffusion model output postprocessing, memorization metrics, and feature extraction."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def to_signed(x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Convert uint8 {0,1,2} to float32 {-1,0,1} via (x - 1).

    Shape and type are preserved; accepts both numpy and torch tensors.
    """
    if isinstance(x, torch.Tensor):
        return (x.float() - 1)
    else:
        return (x.astype(np.float32) - 1)


def from_signed(x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Convert float in [-1,1] to uint8 {0,1,2} by rounding to nearest level then adding 1.

    Rounds to nearest of -1/0/1, adds 1, clips to [0,2], and converts to uint8.
    Preserves input type (numpy or torch).
    """
    if isinstance(x, torch.Tensor):
        rounded = torch.clamp(torch.round(x), -1, 1)
        return (rounded + 1).to(torch.uint8)
    else:
        rounded = np.clip(np.round(x), -1, 1)
        return (rounded + 1).astype(np.uint8)


def nearest_neighbour_distance(queries: np.ndarray, reference: np.ndarray,
                               batch: int = 512, device: str | None = None) -> np.ndarray:
    """For each query wafer, the smallest fraction of differing pixels to any reference wafer.

    Maps have three levels, so the number of matching pixels between two maps is the sum over
    levels of the dot product of their one-hot masks. That turns the all-pairs Hamming distance
    into three matrix multiplications, (Q, 4096) x (4096, R) each, instead of a (Q, R, 4096)
    comparison tensor: 2,000 queries against 120,000 train wafers needs about 1 GB of GPU memory
    in chunks rather than tens of GB.

    Args:
        queries: (Q, 64, 64) uint8 array with values 0, 1, 2
        reference: (R, 64, 64) uint8 array with values 0, 1, 2
        batch: queries per chunk
        device: 'cuda' or 'cpu'; None picks cuda when available

    Returns:
        (Q,) float32 array of minimum distances in [0, 1]
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    pixels = queries.shape[1] * queries.shape[2]
    q = torch.from_numpy(queries.reshape(len(queries), pixels)).to(device)
    r = torch.from_numpy(reference.reshape(len(reference), pixels)).to(device)
    dtype = torch.float16 if device == "cuda" else torch.float32  # exact for sums up to 4,096
    ref_chunk = 8192
    best = torch.full((len(queries),), float(pixels), device=device, dtype=torch.float32)
    for q0 in range(0, len(queries), batch):
        qb = q[q0:q0 + batch]
        q_onehot = [(qb == level).to(dtype) for level in (0, 1, 2)]
        for r0 in range(0, len(reference), ref_chunk):
            rb = r[r0:r0 + ref_chunk]
            matches = torch.zeros(len(qb), len(rb), device=device, dtype=torch.float32)
            for level, qo in enumerate(q_onehot):
                matches += (qo @ (rb == level).to(dtype).T).float()
            best[q0:q0 + batch] = torch.minimum(best[q0:q0 + batch], (pixels - matches).min(dim=1).values)
    return (best / pixels).cpu().numpy().astype(np.float32)


def near_copy_fraction(distances: np.ndarray, threshold: float) -> float:
    """Compute the fraction of distances <= threshold."""
    return float((distances <= threshold).mean())


def memorization_report(gen_dist: np.ndarray, test_dist: np.ndarray) -> dict:
    """Compute memorization statistics for generated vs. real test samples.

    Args:
        gen_dist: (G,) array of nearest-neighbour distances for generated wafers
        test_dist: (T,) array of distances for real test wafers to training set

    Returns:
        Dict with "generated", "real_test" stats, threshold, and near-copy fractions.
        Rounded to 4 decimals.
    """
    threshold = float(np.percentile(test_dist, 1))

    gen_stats = {
        "mean": round(float(gen_dist.mean()), 4),
        "median": round(float(np.median(gen_dist)), 4),
        "p05": round(float(np.percentile(gen_dist, 5)), 4),
        "min": round(float(gen_dist.min()), 4),
    }
    test_stats = {
        "mean": round(float(test_dist.mean()), 4),
        "median": round(float(np.median(test_dist)), 4),
        "p05": round(float(np.percentile(test_dist, 5)), 4),
        "min": round(float(test_dist.min()), 4),
    }

    return {
        "generated": gen_stats,
        "real_test": test_stats,
        "threshold": round(threshold, 4),
        "near_copy_fraction_generated": round(near_copy_fraction(gen_dist, threshold), 4),
        "near_copy_fraction_real_test": round(near_copy_fraction(test_dist, threshold), 4),
    }


def classifier_features(model: nn.Module, x: torch.Tensor, device,
                       batch: int = 1024) -> np.ndarray:
    """Extract penultimate features (after pool and flatten) from WaferCNN.

    Runs model.features, then model.head[0] (AdaptiveAvgPool2d) and model.head[1]
    (Flatten) in batches under torch.no_grad and eval mode.

    Args:
        model: WaferCNN instance
        x: (N, 1, 64, 64) standardised float tensor
        device: torch device
        batch: batch size for inference

    Returns:
        (N, 256) float32 array of features
    """
    model.eval()
    features = []

    with torch.no_grad():
        for i in range(0, len(x), batch):
            xb = x[i:i + batch].to(device)
            feat = model.features(xb)
            feat = model.head[0](feat)
            feat = model.head[1](feat)
            features.append(feat.cpu())

    return torch.cat(features).numpy().astype(np.float32)


def frechet_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Frechet distance between Gaussians fitted to two feature sets.

    FD = ||mu_a - mu_b||^2 + Tr(Ca + Cb - 2 * sqrt(Ca @ Cb))

    Uses scipy.linalg.sqrtm if available; otherwise numpy eigendecomposition with
    symmetry and numerical stability (1e-6 * I regularization).

    Args:
        a: (Na, D) feature array
        b: (Nb, D) feature array

    Returns:
        Frechet distance as float
    """
    mu_a = a.mean(axis=0)
    mu_b = b.mean(axis=0)

    ca = np.cov(a, rowvar=False)
    cb = np.cov(b, rowvar=False)

    if ca.ndim == 0:
        ca = ca.reshape(1, 1)
    if cb.ndim == 0:
        cb = cb.reshape(1, 1)

    # Numerical stability
    ca += 1e-6 * np.eye(ca.shape[0])
    cb += 1e-6 * np.eye(cb.shape[0])

    # Try scipy; fall back to numpy eigendecomposition
    try:
        from scipy.linalg import sqrtm
        sqrt_ca_cb = sqrtm(ca @ cb).real
    except (ImportError, RuntimeError):
        # Eigendecomposition-based sqrt
        evals_a, evecs_a = np.linalg.eigh(ca)
        evals_b, evecs_b = np.linalg.eigh(cb)

        evals_a = np.maximum(evals_a, 0)
        evals_b = np.maximum(evals_b, 0)

        sqrt_ca = evecs_a @ np.diag(np.sqrt(evals_a)) @ evecs_a.T
        sqrt_ca_cb_product = sqrt_ca @ cb @ sqrt_ca

        evals_prod, evecs_prod = np.linalg.eigh(sqrt_ca_cb_product)
        evals_prod = np.maximum(evals_prod, 0)
        sqrt_ca_cb = evecs_prod @ np.diag(np.sqrt(evals_prod)) @ evecs_prod.T

    mean_diff_sq = float(np.sum((mu_a - mu_b) ** 2))
    trace_term = float(np.trace(ca + cb - 2 * sqrt_ca_cb))

    return mean_diff_sq + trace_term


def sample_grid(real: dict[str, np.ndarray], generated: dict[str, np.ndarray],
                class_names: list[str], per_class: int, path: str) -> None:
    """Create a matplotlib grid of real and generated wafer maps.

    Args:
        real: Dict {class_name: (n, 64, 64) uint8 array}
        generated: Dict {class_name: (n, 64, 64) uint8 array}
        class_names: List of class names (row labels)
        per_class: Number of samples per class to display
        path: Output path for PNG (saved at dpi 120)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_classes = len(class_names)
    fig, axes = plt.subplots(n_classes, 2 * per_class + 1, figsize=(2 * per_class + 1, n_classes))

    if n_classes == 1:
        axes = axes.reshape(1, -1)

    # Colormap for 0, 1, 2 as three distinct greys
    cmap = plt.cm.Greys
    norm_colors = {0: 0.0, 1: 0.5, 2: 1.0}

    for i, class_name in enumerate(class_names):
        real_maps = real.get(class_name, np.zeros((per_class, 64, 64), dtype=np.uint8))
        gen_maps = generated.get(class_name, np.zeros((per_class, 64, 64), dtype=np.uint8))

        # Real maps
        for j in range(per_class):
            ax = axes[i, j]
            ax.imshow(real_maps[j], cmap=cmap, vmin=0, vmax=2)
            ax.axis("off")

        # Gap (blank column)
        axes[i, per_class].axis("off")

        # Generated maps
        for j in range(per_class):
            ax = axes[i, per_class + 1 + j]
            ax.imshow(gen_maps[j], cmap=cmap, vmin=0, vmax=2)
            ax.axis("off")

        # Class label on the left
        axes[i, 0].set_ylabel(class_name, rotation=0, ha="right", va="center", fontsize=10)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
