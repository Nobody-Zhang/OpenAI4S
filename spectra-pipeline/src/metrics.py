"""Spectral similarity metrics used for matching and reconstruction quality.

Ground-truth evaluation metrics (precision/recall/F1, fraction MAE) live in
``src/evaluate.py`` — they are only used once, after the blind loop.
"""
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

