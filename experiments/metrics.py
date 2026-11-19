"""Metrics shared by the synthetic controls and optional text experiments."""

import json
from pathlib import Path

import numpy as np


def auroc(positive, negative):
    """Probability a positive outranks a negative, with half credit for ties."""
    positive, negative = np.asarray(positive), np.sort(negative)
    if positive.ndim != 1 or negative.ndim != 1 or not positive.size or not negative.size:
        raise ValueError("AUROC requires two nonempty vectors")
    if not (np.isfinite(positive).all() and np.isfinite(negative).all()):
        raise ValueError("AUROC requires finite scores")
    lower = np.searchsorted(negative, positive, side="left")
    upper = np.searchsorted(negative, positive, side="right")
    return float((lower + upper).mean() / (2 * negative.size))


def classification(positive, predicted):
    positive, predicted = np.asarray(positive, dtype=bool), np.asarray(predicted, dtype=bool)
    if positive.shape != predicted.shape or positive.ndim != 1:
        raise ValueError("Expected equally sized label and prediction vectors")
    tp = int(np.count_nonzero(positive & predicted))
    fp = int(np.count_nonzero(~positive & predicted))
    fn = int(np.count_nonzero(positive & ~predicted))
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": len(positive) - tp - fp - fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
    }


def write_json(path: Path, result):
    serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized, encoding="utf-8")
