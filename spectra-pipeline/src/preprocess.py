"""Config-driven preprocessing: despike -> denoise -> baseline -> normalise.

All functions operate on a 1-D intensity array already resampled onto the
common grid (see ``data.common_grid``).

The cleaned spectrum is preprocessed exactly ONCE per run (not per loop
iteration): repeated smoothing broadens/flattens peaks and repeated baseline
removal distorts peak areas. ``save_clean_spectrum`` / ``load_clean_spectrum``
persist that one clean spectrum so the iterative subtraction loop reads it back
from disk and it can be inspected later.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, savgol_filter

from .config import GRID_STEP


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
# Second-derivative peak detection
# ---------------------------------------------------------------------------
def detect_peaks_2nd_deriv(y: np.ndarray, grid: np.ndarray, config: dict) -> np.ndarray:
    """Screen initial peak positions via the second-derivative method.

    A Raman band shows up as a sharp negative trough in the (smoothed) second
    derivative, regardless of any residual broad baseline — this is the classic
    robust way to pick peaks. We SavGol-smooth while differentiating (deriv=2),
    then run ``find_peaks`` on ``-d2`` with a MAD-based noise threshold.

    Operates on a spectrum already resampled onto ``grid``. Returns the peak
    positions in grid units (cm^-1), sorted ascending.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 5:
        return np.array([])

    window = int(config.get("peak_smooth_window", 11))
    poly = int(config.get("peak_savgol_poly", 3))
    if window % 2 == 0:
        window += 1
    window = min(window, n if n % 2 == 1 else n - 1)
    window = max(window, poly + 2)
    if window % 2 == 0:
        window += 1
    if window > n:  # spectrum too short to smooth at this order
        return np.array([])

    d2 = savgol_filter(y, window_length=window, polyorder=poly, deriv=2)
    neg = -d2  # troughs of d2 (i.e. peaks of y) become positive prominences

    noise = np.median(np.abs(np.diff(neg))) * 1.4826 or 1e-9
    prom = config.get("peak_prominence_sigma", 3.0) * noise
    distance = max(1, int(round(config.get("peak_min_distance_cm", 8.0) / GRID_STEP)))

    idx, _ = find_peaks(neg, prominence=prom, distance=distance)
    # only keep troughs that sit on actual positive signal, not baseline dips
    idx = idx[y[idx] > 0]
    return grid[idx]


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


# ---------------------------------------------------------------------------
# Persist / reload the one clean spectrum (produced once per run)
# ---------------------------------------------------------------------------
def save_clean_spectrum(path: str, grid: np.ndarray, y: np.ndarray) -> None:
    """Write the cleaned spectrum as a 2-column CSV (raman_shift,intensity).

    Same format as a case's ``spectrum.csv`` so it can be inspected/plotted the
    same way; the iterative loop reloads it via ``load_clean_spectrum``.
    """
    arr = np.column_stack([np.asarray(grid, dtype=float), np.asarray(y, dtype=float)])
    np.savetxt(path, arr, delimiter=",", header="raman_shift,intensity",
               comments="", fmt="%.6g")


def load_clean_spectrum(path: str):
    """Read back a clean spectrum written by ``save_clean_spectrum``.

    Returns ``(grid, intensity)``.
    """
    arr = np.loadtxt(path, delimiter=",", skiprows=1)
    return arr[:, 0], arr[:, 1]
