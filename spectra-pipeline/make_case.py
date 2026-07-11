"""Dataset generation: synthesise dirty mixture spectra into ``cases/caseN/``.

This is deliberately separate from ``run.py``: it is the only place that knows
the ground truth at construction time. Each case is written as an observable
``spectrum.csv`` plus a hidden ``truth.json``, so the analysis loop in
``run.py`` can consume a case without ever seeing the answer.

Usage:
    python make_cases.py [--n N] [--seed S] [--out cases]
                         [--n-components 2|3] [--max-minerals N] [--noise F]

Produces cases/case1/, cases/case2/, ... each with:
    spectrum.csv   two columns (raman_shift, intensity) -- the blind input
    truth.json     {true_names, true_fractions, meta}   -- the answer key
    input.png      visualisation of the dirty spectrum
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data import build_library
from src.synth import save_case, synth_mixture

CASES_ROOT = os.path.join(os.path.dirname(__file__), "cases")


def plot_input(case, path):
    """Save a plot of the raw dirty spectrum (the observable input)."""
    plt.figure(figsize=(10, 3))
    plt.plot(case.grid, case.spectrum, color="gray", lw=0.9)
    plt.xlabel("Raman shift (cm$^{-1}$)"); plt.ylabel("intensity (a.u.)")
    plt.title("Synthetic dirty spectrum (input)")
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5, help="number of cases to generate")
    ap.add_argument("--seed", type=int, default=0, help="base seed (case i uses seed+i)")
    ap.add_argument("--out", type=str, default=CASES_ROOT, help="output root dir")
    ap.add_argument("--n-components", type=int, default=None)
    ap.add_argument("--max-minerals", type=int, default=120)
    ap.add_argument("--noise", type=float, default=0.02)
    args = ap.parse_args()

    print("Building library from RRUFF excellent_oriented ...")
    lib = build_library("excellent_oriented", max_minerals=args.max_minerals)
    print(f"  library: {len(lib.names)} minerals, grid {lib.grid[0]:.0f}-{lib.grid[-1]:.0f} "
          f"cm^-1 ({len(lib.grid)} pts)")

    os.makedirs(args.out, exist_ok=True)
    for i in range(args.n):
        rng = np.random.default_rng(args.seed + i)
        case = synth_mixture(lib, rng, n_components=args.n_components,
                             noise_level=args.noise)
        case_dir = os.path.join(args.out, f"case{i + 1}")
        save_case(case, case_dir)
        plot_input(case, os.path.join(case_dir, "input.png"))
        fr = {k: round(v, 3) for k, v in case.true_fractions.items()}
        print(f"  case{i + 1}: {case.true_names} fractions={fr} -> {case_dir}")

    print(f"\nDone. {args.n} case(s) written under: {args.out}")


if __name__ == "__main__":
    main()
