#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reanalysis of the statistical validation reported in Section 6.6.

Extends src/evaluation/statistical_tests.py with:
  - Friedman tests on the three outcomes (accuracy, subset size, execution time)
  - Nemenyi critical difference and mean ranks
  - Pairwise Wilcoxon tests with Holm correction within each outcome family
  - Two one-sided tests (TOST) for the equivalence claim on accuracy

Run from the repository root:

    python src/evaluation/reanalysis_statistics.py

Reads results/comparison_table.csv, which holds the mean of each metric per
algorithm and dataset over the 15 independent runs.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

ROOT = Path(__file__).resolve().parents[2]
TABLE = ROOT / "results" / "comparison_table.csv"

ALGORITHMS = ["PSO", "BSO", "HHO", "bABER", "PSO-bABER", "HHO-bABER", "BSO-bABER"]
PAIRS = [("PSO-bABER", "PSO"), ("HHO-bABER", "HHO"), ("BSO-bABER", "BSO")]

# column name, label, whether a lower value is better, Wilcoxon alternative
OUTCOMES = [
    ("Accuracy_Mean", "accuracy", False, "greater"),
    ("Features_Mean", "features", True, "less"),
    ("Execution_Time_Mean", "execution time", True, "greater"),
]

# Nemenyi critical value for k = 7 algorithms at alpha = 0.05
Q_ALPHA_K7 = 2.949


def holm(p_values):
    """Holm step-down adjusted p-values."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adjusted = np.empty(n)
    running_max = 0.0
    for rank, index in enumerate(order):
        value = (n - rank) * p[index]
        running_max = max(running_max, value)
        adjusted[index] = min(running_max, 1.0)
    return adjusted


def load_table():
    if not TABLE.exists():
        raise SystemExit(f"Cannot find {TABLE}. Run the experiments first.")
    return pd.read_csv(TABLE)


def column(df, algorithm, metric):
    """Per-dataset means for one algorithm, in a fixed dataset order."""
    subset = df[df["Algorithm"] == algorithm].sort_values("Dataset")
    return subset[metric].to_numpy(dtype=float)


def friedman_section(df):
    print("=" * 78)
    print("FRIEDMAN TESTS")
    print("=" * 78)

    n_datasets = df["Dataset"].nunique()
    k = len(ALGORITHMS)
    critical_difference = Q_ALPHA_K7 * np.sqrt(k * (k + 1) / (6.0 * n_datasets))

    global_p = []
    for metric, label, lower_is_better, _ in OUTCOMES:
        arrays = [column(df, a, metric) for a in ALGORITHMS]
        chi2, p = friedmanchisquare(*arrays)
        global_p.append(p)

        matrix = np.column_stack(arrays)
        signed = matrix if lower_is_better else -matrix
        ranks = np.array([rankdata(row) for row in signed])
        mean_ranks = ranks.mean(axis=0)

        print(f"\n{label}: chi2 = {chi2:.3f}, p = {p:.4g}, df = {k - 1}")
        for algorithm, rank in sorted(zip(ALGORITHMS, mean_ranks), key=lambda t: t[1]):
            print(f"   {algorithm:>11}  mean rank {rank:.2f}")
        print(f"   sum of mean ranks = {mean_ranks.sum():.2f} (must equal {k * (k + 1) / 2:.0f})")

    adjusted = holm(global_p)
    print(f"\nNemenyi critical difference (k = {k}, N = {n_datasets}, "
          f"alpha = 0.05): {critical_difference:.2f} rank positions")
    print("Holm correction across the three global tests:")
    for (_, label, _, _), p, p_adj in zip(OUTCOMES, global_p, adjusted):
        print(f"   {label:>15}: p = {p:.4g}  ->  adjusted p = {p_adj:.4g}")


def wilcoxon_section(df):
    print("\n" + "=" * 78)
    print("PAIRWISE WILCOXON TESTS, HOLM-CORRECTED WITHIN EACH OUTCOME FAMILY")
    print("=" * 78)

    for metric, label, _, alternative in OUTCOMES:
        rows, p_values = [], []
        for hybrid, base in PAIRS:
            x = column(df, hybrid, metric)
            y = column(df, base, metric)
            statistic, p = wilcoxon(x, y, alternative=alternative,
                                    zero_method="wilcox", method="exact")
            difference = x - y
            pooled = np.std(np.concatenate([x, y]), ddof=1)
            cohens_d = difference.mean() / pooled if pooled > 0 else 0.0
            rows.append((hybrid, base, x.mean(), y.mean(),
                         difference.mean(), statistic, p, cohens_d))
            p_values.append(p)

        adjusted = holm(p_values)
        direction = ">" if alternative == "greater" else "<"
        print(f"\n--- {label} (alternative: hybrid {direction} base) ---")
        header = (f"{'comparison':<24}{'hybrid':>10}{'base':>10}{'diff':>10}"
                  f"{'W':>7}{'p':>9}{'p_Holm':>9}{'d':>8}")
        print(header)
        for (hybrid, base, hm, bm, dm, w, p, d), p_adj in zip(rows, adjusted):
            name = f"{hybrid} vs {base}"
            print(f"{name:<24}{hm:>10.3f}{bm:>10.3f}{dm:>+10.3f}"
                  f"{w:>7.1f}{p:>9.3f}{p_adj:>9.3f}{d:>+8.3f}")


def equivalence_section(df, margin=0.01):
    print("\n" + "=" * 78)
    print(f"EQUIVALENCE ON ACCURACY (TOST, margin = {margin})")
    print("=" * 78)

    for hybrid, base in PAIRS:
        difference = (column(df, hybrid, "Accuracy_Mean")
                      - column(df, base, "Accuracy_Mean"))
        _, p_lower = wilcoxon(difference + margin, alternative="greater",
                              zero_method="wilcox")
        _, p_upper = wilcoxon(difference - margin, alternative="less",
                              zero_method="wilcox")
        p = max(p_lower, p_upper)
        verdict = "equivalent" if p < 0.05 else "not established"
        print(f"   {hybrid:>11} vs {base:<6} max(p) = {p:.3f}  {verdict}"
              f"   (largest |difference| = {np.abs(difference).max():.4f})")


def main():
    df = load_table()
    friedman_section(df)
    wilcoxon_section(df)
    equivalence_section(df)
    print("\nDone.")


if __name__ == "__main__":
    main()
