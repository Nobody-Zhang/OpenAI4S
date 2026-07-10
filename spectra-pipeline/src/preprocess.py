"""Config-driven preprocessing: despike -> denoise -> baseline -> normalise.

All functions operate on a 1-D intensity array already resampled onto the
common grid (see ``data.common_grid``).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter


# ---------------------------------------------------------------------------
# Despike (cosmic-ray removal, Whitaker-Hayes modified z-score)
# ---------------------------------------------------------------------------
def despike(y: np.ndarray, threshold: float = 7.0, window: int = 5) -> np.ndarray:
    diff = np.diff(y, prepend=y[0])
    med = np.median(diff)
    mad = np.median(np.abs(diff - med)) or 1e-9
    mod_z = 0.6745 * (diff - med) / mad
    spikes = np.abs(mod_z) > threshold
    y = y.copy()
    n = len(y)
    for i in np.where(spikes)[0]:
        lo, hi = max(0, i - window), min(n, i + window + 1)
        neigh = [j for j in range(lo, hi) if not spikes[j]]
        if neigh:
            y[i] = np.mean(y[neigh])
    return y


# ---------------------------------------------------------------------------
# Denoise
# ---------------------------------------------------------------------------
def denoise(y: np.ndarray, method: str = "savgol", window: int = 7, poly: int = 3) -> np.ndarray:
    if method == "none":
        return y
    if method == "gaussian":
        return gaussian_filter1d(y, sigma=max(window / 3.0, 0.5))
    # savgol
    window = int(window)
    if window % 2 == 0:
        window += 1
    window = max(window, poly + 2)
    if window % 2 == 0:
        window += 1
    return savgol_filter(y, window_length=window, polyorder=poly)


# ---------------------------------------------------------------------------
# Baseline correction
# ---------------------------------------------------------------------------
def baseline_correct(y: np.ndarray, method: str = "asls", lam: float = 1e5,
                     p: float = 0.01, poly_order: int = 5) -> np.ndarray:
    if method == "none":
        return y
    if method == "poly":
        x = np.arange(len(y))
        coeffs = np.polyfit(x, y, poly_order)
        base = np.polyval(coeffs, x)
        return y - base
    from pybaselines import Baseline
    fitter = Baseline(x_data=np.arange(len(y)))
    if method == "airpls":
        base, _ = fitter.airpls(y, lam=lam)
    else:  # asls
        base, _ = fitter.asls(y, lam=lam, p=p)
    return y - base


# ---------------------------------------------------------------------------
# Normalise
# ---------------------------------------------------------------------------
def normalise(y: np.ndarray, method: str = "area") -> np.ndarray:
    y = np.clip(y, 0, None)
    if method == "max":
        d = y.max()
    elif method == "l2":
        d = np.linalg.norm(y)
    else:  # area
        d = y.sum()
    return y / d if d > 0 else y


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def preprocess(y: np.ndarray, config: dict) -> np.ndarray:
    """Run the full config-driven preprocessing chain on a gridded spectrum."""
    out = np.asarray(y, dtype=float)
    if config.get("despike_enabled", True):
        out = despike(out, config["despike_threshold"], config["despike_window"])
    out = denoise(out, config["denoise_method"], config["denoise_window"], config["savgol_poly"])
    out = baseline_correct(out, config["baseline_method"], config["baseline_lam"],
                           config["baseline_p"], config["baseline_poly_order"])
    out = np.clip(out, 0, None)
    out = normalise(out, config["normalise_method"])
    return out
