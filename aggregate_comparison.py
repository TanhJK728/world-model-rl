#!/usr/bin/env python3
"""Aggregate per-checkpoint comparison rows into seed-level mean +/- std per method.

`compare_imagined_methods.py` produces one row per checkpoint. When several seeds of
the same imagined method are compared, they share a method label; this script groups by
that label and reports the seed-level mean and standard deviation (n = number of seeds)
for model return, real return, and the exploitation gap. Seed-level error bars are the
credible uncertainty for the claim "method X reliably produces / mitigates a gap",
because each per-checkpoint value is already averaged over the evaluation episodes.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Fixed presentation order (baseline first, then increasing horizon / mitigation).
METHOD_ORDER = [
    "Real PPO (mid init)",
    "Fixed H=5",
    "Fixed H=20",
    "Uncertainty termination",
    "Weighted advantage H=20",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-csv", required=True)
    parser.add_argument("--output-dir", default="runs/comparison_seeds")
    parser.add_argument("--baseline-relabel", default="Real PPO (mid init)",
                        help="rename the no-imagined-config baseline row to this label")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.comparison_csv)

    # Any row whose label starts with "Real PPO" is the shared warm-start baseline.
    df["method"] = df["method"].apply(
        lambda m: args.baseline_relabel if str(m).startswith("Real PPO") else m
    )

    agg_rows = []
    for method, g in df.groupby("method"):
        agg_rows.append({
            "method": method,
            "seeds": len(g),
            "model_return_mean": g["model_return_mean"].mean(),
            "model_return_sd": g["model_return_mean"].std(ddof=1) if len(g) > 1 else 0.0,
            "real_return_mean": g["real_return_mean"].mean(),
            "real_return_sd": g["real_return_mean"].std(ddof=1) if len(g) > 1 else 0.0,
            "gap_mean": g["exploitation_gap_mean"].mean(),
            "gap_sd": g["exploitation_gap_mean"].std(ddof=1) if len(g) > 1 else 0.0,
            "mean_training_horizon": g["mean_training_horizon"].mean(),
        })
    agg = pd.DataFrame(agg_rows)
    order = [m for m in METHOD_ORDER if m in set(agg["method"])]
    order += [m for m in agg["method"] if m not in order]
    agg = agg.set_index("method").loc[order].reset_index()

    agg.to_csv(out / "comparison_aggregated.csv", index=False)

    lines = [
        "| Method | Seeds | Model return | Real return | Exploitation gap | Mean training horizon |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in agg.itertuples():
        lines.append(
            f"| {r.method} | {r.seeds} "
            f"| {r.model_return_mean:.1f} ± {r.model_return_sd:.1f} "
            f"| {r.real_return_mean:.1f} ± {r.real_return_sd:.1f} "
            f"| {r.gap_mean:.1f} ± {r.gap_sd:.1f} "
            f"| {r.mean_training_horizon:.1f} |"
        )
    (out / "comparison_aggregated.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    x = np.arange(len(agg))

    # Exploitation gap with seed-level error bars.
    plt.figure(figsize=(9, 5))
    plt.bar(x, agg["gap_mean"], yerr=agg["gap_sd"], capsize=5)
    plt.axhline(0.0, linewidth=1.0, color="black")
    plt.xticks(x, agg["method"], rotation=20, ha="right")
    plt.ylabel("Model return − real return")
    plt.title("Model exploitation gap (mean ± std over seeds)")
    plt.tight_layout()
    plt.savefig(out / "exploitation_gap_seeds.png", dpi=180)
    plt.close()

    # Model vs real return, grouped, with error bars.
    width = 0.36
    plt.figure(figsize=(10, 5))
    plt.bar(x - width / 2, agg["model_return_mean"], width,
            yerr=agg["model_return_sd"], capsize=4, label="model return")
    plt.bar(x + width / 2, agg["real_return_mean"], width,
            yerr=agg["real_return_sd"], capsize=4, label="real return")
    plt.xticks(x, agg["method"], rotation=20, ha="right")
    plt.ylabel("200-step return")
    plt.title("Learned-model return vs real-environment return (mean ± std over seeds)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "model_vs_real_return_seeds.png", dpi=180)
    plt.close()

    print(agg.to_string(index=False))
    print(f"Saved aggregated comparison to {out}")


if __name__ == "__main__":
    main()
