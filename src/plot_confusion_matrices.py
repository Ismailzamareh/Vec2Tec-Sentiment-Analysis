"""
Plot publication-quality confusion matrices for the best classifier in
each model family on each dataset.

For every dataset in {imdb, labr, astd, astd4} we build one PNG with a
1x3 grid of heatmaps:
    [ best TF-IDF | best Word2Vec | best Vec2Tec ]

The "best" model in a family is selected by f1_macro_mean from the
aggregated table (results/tables/aggregated_results.csv).

Heatmaps:
    - cells show RAW counts (annotated)
    - colour scale is ROW-normalised, so the diagonal intensity gives
      per-class recall (most readable view).
    - axis labels are taken from the CM CSV column headers, so ASTD4
      automatically shows POS / NEG / NEUTRAL / OBJ.

Outputs:
    results/figures/cm_<dataset>.png  (300 DPI)

Usage:
    python src/plot_confusion_matrices.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import RESULTS_DIR, CM_DIR
from utils import banner, load_csv


FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
DPI = 300

DATASETS = ("imdb", "labr", "astd", "astd4")

FAMILIES = (
    ("TF-IDF",   ("tfidf_lr", "tfidf_svm")),
    ("Word2Vec", ("word2vec_lr", "word2vec_svm")),
    ("Vec2Tec",  ("vec2tec_lr", "vec2tec_svm")),
)


def best_in_family(agg: pd.DataFrame, ds: str,
                   family_models: tuple[str, ...]) -> str | None:
    sub = agg[(agg["dataset"] == ds) & (agg["model"].isin(family_models))]
    if sub.empty:
        return None
    return str(sub.loc[sub["f1_macro_mean"].idxmax(), "model"])


def load_cm(model: str, ds: str) -> tuple[np.ndarray, list[str]]:
    df = load_csv(CM_DIR / f"{model}_{ds}.csv")
    pred_cols = [c for c in df.columns if c.startswith("pred_")]
    labels = [c[len("pred_"):] for c in pred_cols]
    return df[pred_cols].to_numpy(dtype=int), labels


def plot_one(ds: str, agg: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, (fam_name, models) in zip(axes, FAMILIES):
        m = best_in_family(agg, ds, models)
        if m is None:
            ax.set_visible(False)
            continue
        cm, labels = load_cm(m, ds)
        row_total = cm.sum(axis=1, keepdims=True).clip(min=1)
        cm_norm = cm / row_total
        sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues", cbar=False,
                    ax=ax,
                    xticklabels=[f"pred {l}" for l in labels],
                    yticklabels=[f"true {l}" for l in labels],
                    annot_kws={"fontsize": 9})
        ax.set_title(f"{fam_name}\n{m}", fontsize=10)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    fig.suptitle(f"Confusion matrices on {ds.upper()} "
                 "(cells: counts; colour: row-normalised recall)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    out = FIG_DIR / f"cm_{ds}.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


def main() -> None:
    banner("Plotting confusion matrices for best-of-family per dataset")
    sns.set_theme(context="paper", style="white", font_scale=1.05)
    agg = load_csv(RESULTS_DIR / "tables" / "aggregated_results.csv")
    for ds in DATASETS:
        plot_one(ds, agg)


if __name__ == "__main__":
    main()
