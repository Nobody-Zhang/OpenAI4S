"""Entry point: load a pre-generated case -> preprocess ONCE -> iterative
peak-driven subtraction loop -> evaluate against ground truth ONCE -> write a
diagnostic report + figures.

Pipeline shape:
    1. global preprocessing (SavGol denoise + ALS/poly baseline) applied exactly
       once; the clean spectrum is saved to ``clean_spectrum.csv`` for inspection
    2. the loop reads that clean spectrum back and iterates find-peaks -> match
       -> subtract on the residual (see ``src/loop.py``)
    3. the answer key is loaded only afterwards, for a single final evaluation

The loop never sees the ground truth. Cases are produced separately by
``make_cases.py`` (which writes cases/caseN/spectrum.csv + truth.json).

Usage:
    python run.py --case cases/case1 [--max-components N] [--max-minerals N]
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import default_config
from src.data import build_library, resample
from src.evaluate import evaluate
from src.loop import search
from src.preprocess import load_clean_spectrum, preprocess, save_clean_spectrum
from src.synth import load_spectrum, load_truth

OUT_ROOT = os.path.join(os.path.dirname(__file__), "outputs")


def make_figures(spectrum, clean, lib, outcome, fig_dir):
    os.makedirs(fig_dir, exist_ok=True)
    res = outcome.best_result
    grid = lib.grid

    # 1. raw dirty spectrum vs the one-time preprocessed clean spectrum
    fig, (axa, axb) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    axa.plot(grid, spectrum, color="gray", lw=0.9)
    axa.set_ylabel("raw intensity (a.u.)"); axa.set_title("Global preprocessing (done once)")
    axb.plot(grid, clean, color="tab:green", lw=1.0)
    axb.set_ylabel("clean (norm.)"); axb.set_xlabel("Raman shift (cm$^{-1}$)")
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "preprocess.png"), dpi=120); plt.close(fig)

    # 2. overlay: clean spectrum vs reconstruction
    plt.figure(figsize=(10, 4))
    plt.plot(grid, res.processed, label="clean (preprocessed)", lw=1.2)
    plt.plot(grid, res.recon, label="NNLS reconstruction", lw=1.2, alpha=0.8)
    plt.xlabel("Raman shift (cm$^{-1}$)"); plt.ylabel("norm. intensity")
    plt.title("Clean spectrum vs reconstruction"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(fig_dir, "overlay.png"), dpi=120); plt.close()

    # 3. final residual
    plt.figure(figsize=(10, 3))
    plt.plot(grid, res.processed - res.recon, color="crimson", lw=1.0)
    plt.axhline(0, color="k", lw=0.5)
    plt.xlabel("Raman shift (cm$^{-1}$)"); plt.ylabel("residual")
    plt.title("Final residual (clean - reconstruction)")
    plt.tight_layout(); plt.savefig(os.path.join(fig_dir, "residual.png"), dpi=120); plt.close()

    # 4. subtraction progress: relative residual per step
    if outcome.history:
        steps = [h["step"] for h in outcome.history]
        relres = [h["rel_residual"] for h in outcome.history]
        labels = [h["added_component"] for h in outcome.history]
        plt.figure(figsize=(8, 4))
        plt.plot(steps, relres, "o-", color="tab:blue")
        for s, r, name in zip(steps, relres, labels):
            plt.annotate(name, (s, r), fontsize=7, rotation=30,
                         textcoords="offset points", xytext=(4, 4))
        plt.xlabel("subtraction step"); plt.ylabel("relative residual")
        plt.title("Iterative subtraction progress")
        plt.tight_layout(); plt.savefig(os.path.join(fig_dir, "iterations.png"), dpi=120); plt.close()

    # 5. raw dirty spectrum (input)
    plt.figure(figsize=(10, 3))
    plt.plot(grid, spectrum, color="gray", lw=0.9)
    plt.xlabel("Raman shift (cm$^{-1}$)"); plt.ylabel("intensity (a.u.)")
    plt.title("Synthetic dirty spectrum (input)")
    plt.tight_layout(); plt.savefig(os.path.join(fig_dir, "input.png"), dpi=120); plt.close()


def write_report(truth, final, outcome, cfg, path):
    res = outcome.best_result
    d = res.diagnostics

    lines = []
    lines.append("# 光谱成分识别诊断报告\n")
    lines.append("- 流程: 全局预处理(一次) → 迭代寻峰-匹配-相减 → 真值评估(一次)\n")

    lines.append("## 1. 结论：识别到的成分与比例\n")
    lines.append("| 成分 | 估计比例 | 支持特征峰 (cm⁻¹) |")
    lines.append("|---|---|---|")
    for name, frac in sorted(res.fractions.items(), key=lambda kv: -kv[1]):
        peaks = ", ".join(str(p) for p in res.support.get(name, [])) or "—"
        lines.append(f"| {name} | {frac*100:.1f}% | {peaks} |")
    lines.append("")

    lines.append("## 2. 可信性诊断\n")
    npk = len(res.peaks)
    lines.append(f"- 二阶导数法在干净谱上检出峰: **{npk}** 个"
                 + (f"（位置 cm⁻¹: {[round(float(p), 1) for p in res.peaks[:20]]}）" if npk else ""))
    lines.append("- 说明: 全局预处理只做一次，干净谱已保存于本目录 `clean_spectrum.csv`；"
                 "循环从该文件读回后在残差上逐步寻峰-相减。")
    lines.append(f"- 重构拟合相关 (Pearson): **{d['fit_corr']:.3f}**")
    lines.append(f"- 残差 RMSE: **{d['residual_rmse']:.4f}**")
    lines.append(f"- 解释能量占比: **{d['explained_energy']*100:.1f}%**")
    lines.append(f"- 残差残留显著峰数: **{d['n_residual_peaks']}** "
                 f"{'(提示可能漏成分)' if d['n_residual_peaks'] else '(无明显未解释峰)'}")
    if d["residual_peak_positions"]:
        lines.append(f"  - 位置: {d['residual_peak_positions']}")
    lines.append(f"- 综合可信度: **{d['reliability'].upper()}**")
    lines.append("")

    lines.append("## 3. 与真值对比（循环结束后一次性评估）\n")
    lines.append("> 说明: 以下真值指标仅在迭代相减循环**结束后**计算一次，"
                 "循环过程中并未使用，以模拟科学家不知道答案的真实流程。\n")
    lines.append(f"- 真实成分: {truth['true_names']}")
    tf = {k: round(v, 3) for k, v in truth['true_fractions'].items()}
    lines.append(f"- 真实比例: {tf}")
    lines.append(f"- 成分识别 Precision/Recall/F1: "
                 f"{final['precision']:.2f} / {final['recall']:.2f} / **{final['f1']:.2f}**")
    lines.append(f"- 比例估计 MAE: **{final['fraction_mae']:.3f}**")
    lines.append("")

    lines.append("## 4. 使用的固定配置\n")
    lines.append("```json")
    lines.append(json.dumps(cfg, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append(f"\n- 相减步数: {len(outcome.history)}  |  最大成分数上限: {cfg['top_k']}\n")

    lines.append("## 5. 迭代相减过程（每步：残差寻峰 → 匹配 → 相减）\n")
    lines.append("| 步 | 新增成分 | 匹配相关 | rel_residual | 残差峰数 | 累计成分 |")
    lines.append("|---|---|---|---|---|---|")
    for h in outcome.history:
        lines.append(f"| {h['step']} | {h['added_component']} | {h['match_corr']:.3f} | "
                     f"{h['rel_residual']:.4f} | {h['n_residual_peaks']} | "
                     f"{h['cumulative_components']} |")
    lines.append("")

    lines.append("## 6. 图\n")
    lines.append("![input](figures/input.png)\n")
    lines.append("![preprocess](figures/preprocess.png)\n")
    lines.append("![overlay](figures/overlay.png)\n")
    lines.append("![residual](figures/residual.png)\n")
    lines.append("![iterations](figures/iterations.png)\n")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", type=str, required=True,
                    help="path to a case folder produced by make_cases.py (e.g. cases/case1)")
    ap.add_argument("--max-components", type=int, default=8,
                    help="max number of components / subtraction steps")
    ap.add_argument("--max-minerals", type=int, default=120)
    args = ap.parse_args()

    print("Building library from RRUFF excellent_oriented ...")
    lib = build_library("excellent_oriented", max_minerals=args.max_minerals)
    print(f"  library: {len(lib.names)} minerals, grid {lib.grid[0]:.0f}-{lib.grid[-1]:.0f} "
          f"cm^-1 ({len(lib.grid)} pts)")

    # --- load the BLIND observable only ---
    grid, spectrum = load_spectrum(args.case)
    if len(spectrum) != len(lib.grid) or not np.allclose(grid, lib.grid):
        # safety net: align an off-grid case onto the library grid
        spectrum = resample(grid, spectrum, lib.grid)
    print(f"Loaded case: {args.case} ({len(spectrum)} pts) -- ground truth withheld from loop")

    cfg = default_config()
    cfg["top_k"] = args.max_components

    case_name = os.path.basename(os.path.normpath(args.case))
    run_dir = os.path.join(OUT_ROOT, f"run_{case_name}")
    os.makedirs(run_dir, exist_ok=True)

    # --- global preprocessing, done ONCE; persist the clean spectrum ---
    clean = preprocess(spectrum, cfg)
    clean_path = os.path.join(run_dir, "clean_spectrum.csv")
    save_clean_spectrum(clean_path, lib.grid, clean)
    print(f"Preprocessed once -> saved clean spectrum: {clean_path}")

    # --- the loop reads the clean spectrum back from the folder ---
    _, clean = load_clean_spectrum(clean_path)

    log_path = os.path.join(run_dir, "iterations.jsonl")
    print("\nRunning iterative peak-find -> match -> subtract loop ...")
    outcome = search(clean, lib, cfg, log_path=log_path)

    # --- reveal the answer key ONCE, only now, for final evaluation ---
    truth = load_truth(args.case)
    final = evaluate(outcome.best_result, truth)
    print(f"\nFinal evaluation (revealed after loop): "
          f"F1={final['f1']:.2f} MAE={final['fraction_mae']:.3f} "
          f"true={truth['true_names']}")

    make_figures(spectrum, clean, lib, outcome, os.path.join(run_dir, "figures"))
    write_report(truth, final, outcome, cfg, os.path.join(run_dir, "report.md"))
    print(f"\nDone. Report + figures + log in: {run_dir}")


if __name__ == "__main__":
    main()
