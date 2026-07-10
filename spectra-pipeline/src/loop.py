"""Outer loop: search over configs, score by reconstruction residual (blind),
validate against ground truth, keep the best, log every iteration."""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field

import numpy as np

from . import metrics
from .config import SEARCH_SPACE, default_config
from .data import Library
from .pipeline import PipelineResult, run_pipeline
from .synth import SynthCase

# parsimony penalty (in units of relative residual) so the fit isn't improved
# by piling on spectrally-similar spurious components
PARSIMONY = 0.01


def objective(result: PipelineResult) -> float:
    """Blind score to minimise: scale-invariant relative residual + parsimony.

    Uses relative residual (not raw RMSE) so configs with different
    normalisation methods are compared on the same footing.
    """
    n = len(result.fractions)
    return result.diagnostics["rel_residual"] + PARSIMONY * n


def gt_score(result: PipelineResult, case: SynthCase) -> dict:
    """Ground-truth validation metrics (not used for selection)."""
    prf = metrics.component_prf(case.true_names, list(result.fractions))
    mae = metrics.fraction_mae(case.true_fractions, result.fractions)
    return {**prf, "fraction_mae": mae}


def mutate(base: dict, hints: list, rng: np.random.Generator) -> dict:
    """Produce a new config: hint-guided greedy tweaks + random exploration."""
    cfg = copy.deepcopy(base)

    if "possible_missing_component" in hints:
        cfg["top_k"] = min(cfg["top_k"] + 4, 24)
        cfg["corr_threshold"] = max(cfg["corr_threshold"] - 0.1, 0.1)
    if "poor_baseline_or_denoise" in hints:
        cfg["baseline_method"] = rng.choice(SEARCH_SPACE["baseline_method"])
        cfg["denoise_method"] = rng.choice(SEARCH_SPACE["denoise_method"])
        cfg["baseline_lam"] = float(rng.choice(SEARCH_SPACE["baseline_lam"]))

    # random exploration: perturb a couple of random knobs
    keys = rng.choice(list(SEARCH_SPACE), size=2, replace=False)
    for k in keys:
        v = rng.choice(SEARCH_SPACE[k])
        cfg[k] = v.item() if hasattr(v, "item") else v
    return cfg


@dataclass
class LoopOutcome:
    best_config: dict
    best_result: PipelineResult
    best_objective: float
    history: list = field(default_factory=list)


def search(case: SynthCase, lib: Library, seed: int = 0, budget: int = 20,
           patience: int = 8, tol: float = 0.03, log_path: str = None,
           verbose: bool = True) -> LoopOutcome:
    """Iterate: run pipeline with a config, score, mutate toward better configs."""
    rng = np.random.default_rng(seed)
    cfg = default_config()

    best = None
    best_obj = np.inf
    best_cfg = None
    best_hints: list = []
    history = []
    since_improve = 0
    logf = open(log_path, "w") if log_path else None

    for it in range(budget):
        if it == 0:
            trial_cfg = cfg
        else:
            # explore from best config, guided by best diagnostics' hints
            trial_cfg = mutate(best_cfg, best_hints, rng)

        result = run_pipeline(case.spectrum, lib, trial_cfg)
        obj = objective(result)
        gt = gt_score(result, case)

        improved = obj < best_obj - 1e-9
        if improved:
            best, best_obj, best_cfg = result, obj, trial_cfg
            best_hints = result.diagnostics["hints"]
            since_improve = 0
        else:
            since_improve += 1

        record = {
            "iteration": it,
            "objective": round(obj, 6),
            "is_best": improved,
            "rel_residual": round(result.diagnostics["rel_residual"], 5),
            "residual_rmse": round(result.diagnostics["residual_rmse"], 6),
            "fit_corr": round(result.diagnostics["fit_corr"], 4),
            "reliability": result.diagnostics["reliability"],
            "n_residual_peaks": result.diagnostics["n_residual_peaks"],
            "hints": result.diagnostics["hints"],
            "identified": {k: round(v, 3) for k, v in result.fractions.items()},
            "gt": {k: round(v, 4) for k, v in gt.items()},
            "config": {k: trial_cfg[k] for k in (
                "denoise_method", "denoise_window", "baseline_method",
                "baseline_lam", "normalise_method", "top_k",
                "corr_threshold", "fraction_threshold")},
        }
        history.append(record)
        if logf:
            logf.write(json.dumps(record, ensure_ascii=False) + "\n")
            logf.flush()
        if verbose:
            flag = "*" if improved else " "
            print(f"[{it:02d}]{flag} obj={obj:.4f} relres={record['rel_residual']:.4f} "
                  f"corr={record['fit_corr']:.3f} F1={gt['f1']:.2f} "
                  f"MAE={gt['fraction_mae']:.3f} rel={record['reliability']} "
                  f"-> {list(result.fractions)}")

        if best_obj <= tol or since_improve >= patience:
            break

    if logf:
        logf.close()

    # recompute best result with supporting peaks for the report
    best = run_pipeline(case.spectrum, lib, best_cfg, with_support=True)
    return LoopOutcome(best_config=best_cfg, best_result=best,
                       best_objective=best_obj, history=history)
