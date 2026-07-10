"""Iterative peak-driven spectral-subtraction loop.

The spectrum is preprocessed exactly once (upstream, in ``run.py``) and the
cleaned spectrum is passed in here. This loop then repeats, on the *residual*:

    1. detect significant peaks on the current residual (2nd derivative)
    2. peak-driven match -> best-correlating library component
    3. NNLS-refit ALL selected components against the clean spectrum (OMP) and
       subtract -> new residual

Removing a dominant phase exposes the weaker/overlapping components hiding under
it, which the next iteration's peak search then picks up. The loop stops when the
residual has no significant peaks left, the best match falls below the
correlation threshold, or adding a component no longer reduces the residual
enough. Ground truth is never seen here — evaluation happens once, afterwards,
in ``src/evaluate.py``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import nnls

from . import diagnose as diag
from . import matching, unmix
from .data import Library
from .preprocess import detect_peaks_2nd_deriv


@dataclass
class PipelineResult:
    processed: np.ndarray                 # the clean spectrum the loop consumed
    recon: np.ndarray
    candidate_names: list
    fractions: dict                       # name -> fraction
    used_idx: np.ndarray
    used_coef: np.ndarray
    diagnostics: dict
    peaks: np.ndarray = field(default_factory=lambda: np.array([]))  # clean-spectrum 2nd-deriv peaks
    support: dict = field(default_factory=dict)   # name -> [supporting peak positions]


@dataclass
class LoopOutcome:
    best_result: PipelineResult
    history: list = field(default_factory=list)   # one record per subtraction step


def search(clean: np.ndarray, lib: Library, config: dict, log_path: str = None,
           verbose: bool = True) -> LoopOutcome:
    """Iterative peak-find -> match -> subtract loop on the (already cleaned)
    spectrum. Returns the identified components + per-step history."""
    clean = np.asarray(clean, dtype=float)
    grid = lib.grid
    max_components = config["top_k"]
    corr_thr = config["corr_threshold"]
    min_gain = config.get("greedy_min_gain", 0.01)
    tgt_norm = float(np.linalg.norm(clean)) or 1e-12

    selected: list[int] = []
    residual = clean.copy()
    prev_relres = 1.0
    history = []
    logf = open(log_path, "w") if log_path else None

    for step in range(max_components):
        # 1. re-detect peaks on the CURRENT residual
        peaks = detect_peaks_2nd_deriv(residual, grid, config)
        if len(peaks) == 0:
            if verbose:
                print(f"[{step:02d}] no significant residual peaks -> stop")
            break

        # 2. peak-driven pick of the next best component
        best, corr = matching.best_next_component(residual, peaks, lib, config, exclude=selected)
        if best is None or (corr < corr_thr and selected):
            if verbose:
                print(f"[{step:02d}] best corr {corr:.3f} < {corr_thr} -> stop")
            break

        # 3. OMP: refit all selected + candidate against the clean spectrum, subtract
        trial = selected + [best]
        A = lib.A[:, trial]
        coef, _ = nnls(A, clean)
        recon = A @ coef
        relres = float(np.linalg.norm(clean - recon) / tgt_norm)

        if selected and (prev_relres - relres) < min_gain:
            if verbose:
                print(f"[{step:02d}] gain {prev_relres - relres:.4f} < {min_gain} -> stop")
            break

        selected = trial
        residual = clean - recon
        prev_relres = relres
        res_rmse = float(np.sqrt(np.mean((clean - recon) ** 2)))

        record = {
            "step": step,
            "added_component": lib.names[best],
            "match_corr": round(corr, 4),
            "rel_residual": round(relres, 5),
            "residual_rmse": round(res_rmse, 6),
            "n_residual_peaks": int(len(peaks)),
            "residual_peak_positions": [round(float(p), 1) for p in peaks[:12]],
            "cumulative_components": [lib.names[j] for j in selected],
        }
        history.append(record)
        if logf:
            logf.write(json.dumps(record, ensure_ascii=False) + "\n")
            logf.flush()
        if verbose:
            print(f"[{step:02d}]+ add {lib.names[best]:<22} corr={corr:.3f} "
                  f"relres={relres:.4f} peaks={len(peaks)} -> {record['cumulative_components']}")

    if logf:
        logf.close()

    # --- finalize: NNLS unmix over the discovered components ---
    if selected:
        fractions, used_idx, used_coef = unmix.unmix(clean, lib, np.array(selected), config)
        recon = unmix.reconstruct(lib, used_idx, used_coef)
        cand_names = [lib.names[j] for j in selected]
    else:
        fractions, used_idx, used_coef = {}, np.array([], dtype=int), np.array([])
        recon = np.zeros_like(clean)
        cand_names = []

    diagnostics = diag.diagnose(clean, recon, grid, config)
    peaks_clean = detect_peaks_2nd_deriv(clean, grid, config)  # initial clean-spectrum peaks (report)

    support = {}
    for name in fractions:
        support[name] = diag.supporting_peaks(clean, lib, name, grid)

    result = PipelineResult(
        processed=clean, recon=recon, candidate_names=cand_names,
        fractions=fractions, used_idx=used_idx, used_coef=used_coef,
        diagnostics=diagnostics, peaks=peaks_clean, support=support,
    )
    return LoopOutcome(best_result=result, history=history)
