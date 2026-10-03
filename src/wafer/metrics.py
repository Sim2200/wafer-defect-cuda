"""Macro-F1, per-class precision/recall and a confusion matrix, from label and prediction arrays."""

from __future__ import annotations

import numpy as np

from .data import CLASSES


def confusion(y_true: np.ndarray, y_pred: np.ndarray, n: int = len(CLASSES)) -> np.ndarray:
    cm = np.zeros((n, n), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def report(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    cm = confusion(y_true, y_pred)
    tp = np.diag(cm).astype(float)
    support = cm.sum(axis=1).astype(float)
    predicted = cm.sum(axis=0).astype(float)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)
    present = support > 0
    return {
        "accuracy": round(float(tp.sum() / max(1, support.sum())), 4),
        "macro_f1": round(float(f1[present].mean()), 4),
        "macro_recall": round(float(recall[present].mean()), 4),
        "per_class": {c: {"support": int(support[i]), "precision": round(float(precision[i]), 4),
                          "recall": round(float(recall[i]), 4), "f1": round(float(f1[i]), 4)}
                      for i, c in enumerate(CLASSES)},
        "confusion": cm.tolist(),
    }


def save_confusion_png(cm: list[list[int]], path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = np.array(cm, dtype=float)
    norm = m / np.maximum(m.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{int(m[i, j])}", ha="center", va="center", fontsize=7, color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Test-set confusion matrix (counts; colour = row-normalised recall)", loc="left", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
