"""Pipeline configuration.

A ``config`` is a plain dict of fixed hyperparameters that fully determines the
analysis: global preprocessing (done once) -> iterative peak-driven matching &
spectral subtraction -> unmixing -> diagnosis. There is no config search — the
loop is the iterative subtraction over residuals, not a hyperparameter search.
"""
from __future__ import annotations

import copy


# ---------------------------------------------------------------------------
# Common wavenumber grid (cm^-1). Every library and target spectrum is
# resampled onto this grid so the NNLS reference matrix stays aligned.
# ---------------------------------------------------------------------------
GRID_MIN = 150.0
GRID_MAX = 1400.0
GRID_STEP = 2.0


# ---------------------------------------------------------------------------
# Default pipeline config (fixed hyperparameters).
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {
    # --- despike (cosmic-ray removal) ---
    "despike_enabled": True,
    "despike_threshold": 7.0,       # modified z-score threshold
    "despike_window": 5,            # neighbourhood for replacement

    # --- denoise (smoothing) ---
    "denoise_method": "savgol",     # {"savgol", "gaussian", "none"}
    "denoise_window": 7,            # odd; savgol window / gaussian ~sigma proxy
    "savgol_poly": 3,

    # --- baseline correction ---
    "baseline_method": "asls",      # {"asls", "airpls", "poly", "none"}
    "baseline_lam": 1e5,            # asls/airpls smoothness
    "baseline_p": 0.01,             # asls asymmetry
    "baseline_poly_order": 5,       # for poly

    # --- normalisation ---
    "normalise_method": "area",     # {"area", "max", "l2"}

    # --- second-derivative peak detection (produces the "clean" peak list) ---
    "peak_smooth_window": 11,        # SavGol window for the 2nd-derivative
    "peak_savgol_poly": 3,           # SavGol polyorder for the 2nd-derivative
    "peak_prominence_sigma": 3.0,    # trough prominence threshold (x MAD noise)
    "peak_min_distance_cm": 8.0,     # min spacing between detected peaks (cm^-1)

    # --- peak-driven candidate pre-filter ---
    "peak_prefilter_enabled": True,  # restrict matching to minerals whose ref
                                     # peaks coincide with the detected peaks
    "peak_match_tol_cm": 8.0,        # ref/target peak coincidence tolerance
    "peak_min_matches": 1,           # min coincident peaks for a mineral to pass

    # --- library matching (peak-driven, per residual step) ---
    "match_metric": "pearson",      # correlation metric for ranking a residual
    "top_k": 8,                     # max number of components (subtraction steps)
    "corr_threshold": 0.3,          # a candidate must exceed this correlation
    "greedy_min_gain": 0.01,        # min relative-residual drop to keep a component

    # --- unmixing (NNLS) ---
    "fraction_threshold": 0.05,     # drop components below this fraction, refit
}


def default_config() -> dict:
    """Return a fresh copy of the default config."""
    return copy.deepcopy(DEFAULT_CONFIG)
