"""Minimal patient-level binary metrics used during model development."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def binary_metrics(
    targets: Iterable[float],
    probabilities: Iterable[float],
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    y = np.asarray(list(targets), dtype=np.float64)
    p = np.asarray(list(probabilities), dtype=np.float64)
    if y.ndim != 1 or p.shape != y.shape or y.size == 0:
        raise ValueError("Targets and probabilities must be aligned non-empty vectors")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilities must be finite and lie in [0, 1]")
    if set(np.unique(y)).difference({0.0, 1.0}):
        raise ValueError("Targets must be binary")
    predictions = p >= float(threshold)
    positives = y == 1
    negatives = ~positives
    tp = int(np.sum(predictions & positives))
    tn = int(np.sum(~predictions & negatives))
    fp = int(np.sum(predictions & negatives))
    fn = int(np.sum(~predictions & positives))
    result = {
        "auroc": float(roc_auc_score(y, p)) if np.unique(y).size == 2 else float("nan"),
        "auprc": float(average_precision_score(y, p)),
        "brier": float(np.mean((p - y) ** 2)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "ppv": float(tp / (tp + fp)) if tp + fp else float("nan"),
        "npv": float(tn / (tn + fn)) if tn + fn else float("nan"),
        "threshold": float(threshold),
        "n": int(y.size),
        "positive": int(positives.sum()),
    }
    return result
