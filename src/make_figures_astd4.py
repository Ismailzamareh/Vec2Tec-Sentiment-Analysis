"""
Step 15 - ASTD 4-class publication figures (300 DPI).

Reads:
    results/metrics_aggregated_astd4.csv    (mean ± std on astd4, 3 seeds)
    results/metrics_aggregated.csv          (mean ± std on imdb/labr/astd,
                                             used for the astd2-vs-astd4 chart)
    results/confusion_matrices/<model>_astd4.csv (4x4 CMs, last-seed state)

Writes:
    results/figures/fig_astd4_macro_f1.png
    results/figures/fig_astd4_confusion.png
    results/figures/fig_astd2_vs_astd4.png

Usage:
    python src/make_figures_astd4.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import RESULTS_DIR, CM_DIR, LABEL_NAMES_4
from utils import banner, load_csv


FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
DPI = 300

MODELS = ("tfidf_lr", "tfidf_svm",
          "word2vec_lr", "word2vec_svm",
          "vec2tec_lr", "vec2tec_svm")

FAMILY_OF = {
    "tfidf_lr": "TF-IDF", "tfidf_svm": "TF-IDF",
    "word2vec_lr": "Word2Vec", "word2vec_svm": "Word2Vec",
    "vec2tec_lr": "Vec2Tec",  "vec2tec_svm": "Vec2Tec",
}
FAMILY_COLOR = {
    "TF-IDF":   "#4C72B0",
    "Word2Vec": "#7F7F7F",
    "Vec2Tec":  "#DD8452",
}


# ---------- 1: macro-F1 bar chart ----------------------------------------- #

def fig_astd4_macro_f1(agg4: pd.DataFrame) -> None:
    sub = (agg4.set_index("model").reindex(MODELS)).reset_index()
    fig, ax = plt.subplots(figsize=(8.5, 5))
    x = np.arange(len(MODELS))
    colors = [FAMILY_COLOR[FAMILY_OF[m]] for m in MODELS]
    edges  = ["black" if FAMILY_OF[m] == "Vec2Tec" else "none" for m in MODELS]
    lws    = [1.2     if FAMILY_OF[m] == "Vec2Tec" else 0     for m in MODELS]

    bars = ax.bar(x, sub["f1_macro_mean"], yerr=sub["f1_macro_std"], capsize=4,
                  color=colors, edgecolor=edges, linewidth=lws)

    # value labels on top of each bar
    for xi, m, sd in zip(x, sub["f1_macro_mean"], sub["f1_macro_std"]):
        ax.text(xi, m + sd + 0.008, f"{m:.3f}",
                ha="center", va="bottom", fontsize=9)

    ax.axhline(0.25, linestyle="--", color="grey", alpha=0.7,
               label="random baseline (4-class)")
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, rotation=20, ha="right")
    ax.set_ylabel("Macro-F1 (mean ± std over 3 seeds)")
    ax.set_title("ASTD 4-class — macro-F1 across models")
    ax.set_ylim(0.20, max(0.55, sub["f1_macro_mean"].max() + 0.08))
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    fig.tight_layout()
    out = FIG_DIR / "fig_astd4_macro_f1.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


# ---------- 2: 4x4 confusion matrices ------------------------------------- #

def _best_in_family(agg4: pd.DataFrame, family_models: tuple[str, ...]) -> str:
    sub = agg4[agg4["model"].isin(family_models)]
    return str(sub.loc[sub["f1_macro_mean"].idxmax(), "model"])


def _load_cm4(model: str) -> tuple[np.ndarray, list[str]]:
    df = load_csv(CM_DIR / f"{model}_astd4.csv")
    pred_cols = [c for c in df.columns if c.startswith("pred_")]
    labels = [c[len("pred_"):] for c in pred_cols]
    return df[pred_cols].to_numpy(dtype=int), labels


def fig_astd4_confusion(agg4: pd.DataFrame) -> None:
    tfidf_best = _best_in_family(agg4, ("tfidf_lr", "tfidf_svm"))
    v2t_best   = _best_in_family(agg4, ("vec2tec_lr", "vec2tec_svm"))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, model, title_prefix in [
        (axes[0], tfidf_best,  "Best TF-IDF"),
        (axes[1], v2t_best,    "Best Vec2Tec"),
    ]:
        cm, labels = _load_cm4(model)
        # Normalised by true-class count (row-normalised), so the diagonal
        # gives per-class recall — most informative for an imbalanced view.
        row_totals = cm.sum(axis=1, keepdims=True).clip(min=1)
        cm_norm = cm / row_totals
        sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues",
                    cbar=True, ax=ax,
                    xticklabels=[f"pred {l}" for l in labels],
                    yticklabels=[f"true {l}" for l in labels],
                    annot_kws={"fontsize": 9})
        ax.set_title(f"{title_prefix}: {model}\n(cells: counts; "
                     "colour: row-normalised recall)", fontsize=10)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    fig.suptitle("ASTD 4-class confusion matrices "
                 "(test set, single-seed snapshot from last run)",
                 fontsize=12, y=1.03)
    fig.tight_layout()
    out = FIG_DIR / "fig_astd4_confusion.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


# ---------- 3: binary ASTD vs 4-class ASTD -------------------------------- #

def fig_astd2_vs_astd4(agg_bin: pd.DataFrame, agg4: pd.DataFrame) -> None:
    """Side-by-side macro-F1 on binary ASTD and 4-class ASTD per model."""
    bin_astd = (agg_bin[agg_bin["dataset"] == "astd"]
                .set_index("model").reindex(MODELS))
    cls4     = (agg4.set_index("model").reindex(MODELS))

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    x = np.arange(len(MODELS))
    width = 0.36

    ax.bar(x - width/2, bin_astd["f1_macro_mean"], width,
           yerr=bin_astd["f1_macro_std"], capsize=4,
           color="#4C72B0", label="Binary ASTD (POS vs NEG)")
    ax.bar(x + width/2, cls4["f1_macro_mean"], width,
           yerr=cls4["f1_macro_std"], capsize=4,
           color="#C44E52", label="4-class ASTD (POS/NEG/NEU/OBJ)")

    # Delta annotations
    for xi, m, (a, b) in zip(x, MODELS, zip(bin_astd["f1_macro_mean"],
                                            cls4["f1_macro_mean"])):
        gap_pp = (a - b) * 100
        y = max(a, b) + 0.02
        ax.annotate(f"-{gap_pp:.1f} pp", xy=(xi, y), ha="center",
                    fontsize=9, fontweight="bold", color="black")

    ax.axhline(0.50, linestyle="--", color="#4C72B0", alpha=0.4,
               label="random baseline (binary)")
    ax.axhline(0.25, linestyle="--", color="#C44E52", alpha=0.4,
               label="random baseline (4-class)")

    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, rotation=20, ha="right")
    ax.set_ylabel("Macro-F1 (mean ± std over 3 seeds)")
    ax.set_title("ASTD difficulty: binary vs 4-class on the same model family")
    ax.set_ylim(0.20, 0.95)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(fontsize=8, loc="upper right", frameon=False)
    fig.tight_layout()
    out = FIG_DIR / "fig_astd2_vs_astd4.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


# ---------- main ---------------------------------------------------------- #

def main() -> None:
    banner("Generating ASTD 4-class figures (300 DPI)")
    sns.set_theme(context="paper", style="whitegrid", font_scale=1.05)

    agg4    = load_csv(RESULTS_DIR / "metrics_aggregated_astd4.csv")
    agg_bin = load_csv(RESULTS_DIR / "metrics_aggregated.csv")

    fig_astd4_macro_f1(agg4)
    fig_astd4_confusion(agg4)
    fig_astd2_vs_astd4(agg_bin, agg4)

    banner(f"Wrote figures under {FIG_DIR.relative_to(RESULTS_DIR.parent)}",
           char="-")


if __name__ == "__main__":
    main()
