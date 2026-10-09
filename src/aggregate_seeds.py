"""
Step 9 - Aggregate metrics across multiple random-seed runs.

Reads:
    results/metrics_seed_42.csv
    results/metrics_seed_123.csv
    results/metrics_seed_2024.csv

For each (model, dataset) pair, computes mean and standard deviation of:
    accuracy, precision_macro, recall_macro, f1_macro, f1_weighted

Writes:
    results/metrics_aggregated.csv

Prints a clean markdown table to stdout.

Usage:
    python src/aggregate_seeds.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from config import RESULTS_DIR
from utils import banner, load_csv, save_csv


SEEDS = (42, 123, 2024)
METRICS = ("accuracy", "precision_macro", "recall_macro",
           "f1_macro", "f1_weighted")


def _aggregate(frames: list[pd.DataFrame], out_path: Path, label: str) -> pd.DataFrame:
    big = pd.concat(frames, ignore_index=True)
    agg_funcs = {m: ["mean", "std"] for m in METRICS}
    grouped = (big.groupby(["model", "dataset"])
                   .agg(agg_funcs)
                   .reset_index())
    rename_map = {
        "accuracy_mean":        "acc_mean",
        "accuracy_std":         "acc_std",
        "precision_macro_mean": "precision_macro_mean",
        "precision_macro_std":  "precision_macro_std",
        "recall_macro_mean":    "recall_macro_mean",
        "recall_macro_std":     "recall_macro_std",
        "f1_macro_mean":        "f1_macro_mean",
        "f1_macro_std":         "f1_macro_std",
        "f1_weighted_mean":     "f1_weighted_mean",
        "f1_weighted_std":      "f1_weighted_std",
    }
    grouped.columns = [
        "_".join(c).rstrip("_") if isinstance(c, tuple) else c
        for c in grouped.columns
    ]
    grouped = grouped.rename(columns=rename_map)
    for col in grouped.columns:
        if col not in ("model", "dataset"):
            grouped[col] = grouped[col].round(6)
    grouped = grouped.sort_values(["dataset", "model"]).reset_index(drop=True)
    save_csv(grouped, out_path)
    print(f"  wrote {out_path.name} ({len(grouped)} rows, {label})")
    return grouped


def _aggregate_astd4() -> pd.DataFrame | None:
    frames = []
    for s in SEEDS:
        p = RESULTS_DIR / f"metrics_seed_{s}_astd4.csv"
        if not p.exists():
            print(f"  [astd4] skip: {p.name} missing")
            return None
        df = load_csv(p)
        df["seed"] = s
        frames.append(df)
        print(f"  loaded {p.name} ({len(df)} rows)")
    return _aggregate(frames, RESULTS_DIR / "metrics_aggregated_astd4.csv",
                      label="astd4 only")


def main() -> None:
    banner("Aggregating metrics across seeds")

    frames = []
    for s in SEEDS:
        p = RESULTS_DIR / f"metrics_seed_{s}.csv"
        if not p.exists():
            print(f"  ERROR: missing {p.name}", file=sys.stderr)
            sys.exit(1)
        df = load_csv(p)
        df["seed"] = s
        frames.append(df)
        print(f"  loaded {p.name} ({len(df)} rows)")

    big = pd.concat(frames, ignore_index=True)

    agg_funcs = {m: ["mean", "std"] for m in METRICS}
    grouped = (big.groupby(["model", "dataset"])
                   .agg(agg_funcs)
                   .reset_index())

    # Flatten the column MultiIndex: ("accuracy","mean") -> "acc_mean"
    rename_map = {
        "accuracy_mean":        "acc_mean",
        "accuracy_std":         "acc_std",
        "precision_macro_mean": "precision_macro_mean",
        "precision_macro_std":  "precision_macro_std",
        "recall_macro_mean":    "recall_macro_mean",
        "recall_macro_std":     "recall_macro_std",
        "f1_macro_mean":        "f1_macro_mean",
        "f1_macro_std":         "f1_macro_std",
        "f1_weighted_mean":     "f1_weighted_mean",
        "f1_weighted_std":      "f1_weighted_std",
    }
    grouped.columns = [
        "_".join(c).rstrip("_") if isinstance(c, tuple) else c
        for c in grouped.columns
    ]
    grouped = grouped.rename(columns=rename_map)

    # Round for readability
    for col in grouped.columns:
        if col not in ("model", "dataset"):
            grouped[col] = grouped[col].round(6)

    grouped = grouped.sort_values(["dataset", "model"]).reset_index(drop=True)

    out = RESULTS_DIR / "metrics_aggregated.csv"
    save_csv(grouped, out)
    print(f"\n  wrote {out.name} ({len(grouped)} rows)")

    banner("Aggregated metrics (mean ± std across 3 seeds)", char="-")
    # Pretty markdown: model | dataset | acc | f1_macro | f1_weighted
    show = grouped.copy()
    show["accuracy"]    = show.apply(lambda r: f"{r['acc_mean']:.4f} ± {r['acc_std']:.4f}", axis=1)
    show["f1_macro"]    = show.apply(lambda r: f"{r['f1_macro_mean']:.4f} ± {r['f1_macro_std']:.4f}", axis=1)
    show["f1_weighted"] = show.apply(lambda r: f"{r['f1_weighted_mean']:.4f} ± {r['f1_weighted_std']:.4f}", axis=1)
    md = show[["model", "dataset", "accuracy", "f1_macro", "f1_weighted"]]
    print(md.to_markdown(index=False))

    # --- optional: ASTD 4-class aggregation -------------------------------
    banner("ASTD 4-class aggregation (if seed snapshots exist)", char="-")
    agg4 = _aggregate_astd4()
    if agg4 is not None:
        show = agg4.copy()
        show["accuracy"]    = show.apply(lambda r: f"{r['acc_mean']:.4f} ± {r['acc_std']:.4f}", axis=1)
        show["f1_macro"]    = show.apply(lambda r: f"{r['f1_macro_mean']:.4f} ± {r['f1_macro_std']:.4f}", axis=1)
        show["f1_weighted"] = show.apply(lambda r: f"{r['f1_weighted_mean']:.4f} ± {r['f1_weighted_std']:.4f}", axis=1)
        md4 = show[["model", "dataset", "accuracy", "f1_macro", "f1_weighted"]]
        print(md4.to_markdown(index=False))


if __name__ == "__main__":
    main()
