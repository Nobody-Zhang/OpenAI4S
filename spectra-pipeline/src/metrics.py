"""Spectral similarity metrics and evaluation-vs-ground-truth metrics."""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Spectral similarity (used for matching and reconstruction quality)
# ---------------------------------------------------------------------------
def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 0 else 0.0


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 0 else 0.0


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def sid(a: np.ndarray, b: np.ndarray) -> float:
    """Spectral information divergence (lower = more similar)."""
    eps = 1e-12
    p = np.clip(a, eps, None); p = p / p.sum()
    q = np.clip(b, eps, None); q = q / q.sum()
    return float(np.sum(p * np.log(p / q) + q * np.log(q / p)))


# ---------------------------------------------------------------------------
# Evaluation vs ground truth (only used to score/validate, not inside blind fit)
# ---------------------------------------------------------------------------
def component_prf(true_names, pred_names):
    """Precision / recall / F1 on the set of identified components."""
    t, p = set(true_names), set(pred_names)
    tp = len(t & p)
    precision = tp / len(p) if p else 0.0
    recall = tp / len(t) if t else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def fraction_mae(true_map: dict, pred_map: dict) -> float:
    """Mean absolute error of fractions over the union of components."""
    keys = set(true_map) | set(pred_map)
    if not keys:
        return 0.0
    return float(np.mean([abs(true_map.get(k, 0.0) - pred_map.get(k, 0.0)) for k in keys]))
