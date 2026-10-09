"""
Two new analysis figures + supporting tables.

1. results/figures/fig_macro_f1_by_dataset.png
   Bar chart: x-axis = (IMDb, LABR, ASTD), 3 grouped bars per dataset
   (TF-IDF best-in-family, Word2Vec best, Vec2Tec best). Error bars
   show std across 3 seeds.

2. results/figures/fig_efficiency_tradeoff_v2.png
   Annotated scatter of train_time (log scale) vs macro-F1 across
   every (model, dataset). Each point is labelled with its model
   short-name. A small companion table is also written to
   results/tables/efficiency_tradeoff.csv with one row per point.

3. results/tables/efficiency_tradeoff_summary.md
   Brief written summary of the efficiency-quality balance.

Usage:
    python src/make_figures_v2.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import RESULTS_DIR
from utils import banner, load_csv, save_csv


FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR = RESULTS_DIR / "tables"
TBL_DIR.mkdir(parents=True, exist_ok=True)
DPI = 300

FAMILY_COLOR = {
    "TF-IDF":   "#4C72B0",
    "Word2Vec": "#7F7F7F",
    "Vec2Tec":  "#DD8452",
}
SHORT = {
    "tfidf_lr":     "TFIDF-LR",   "tfidf_svm":    "TFIDF-SVM",
    "word2vec_lr":  "W2V-LR",     "word2vec_svm": "W2V-SVM",
    "vec2tec_lr":   "V2T-LR",     "vec2tec_svm":  "V2T-SVM",
}


def _best_per_family(agg: pd.DataFrame, ds: str,
                     family: str, models: list[str]) -> dict:
    sub = agg[(agg["dataset"] == ds) & (agg["model"].isin(models))]
    if sub.empty:
        return {}
    b = sub.loc[sub["f1_macro_mean"].idxmax()]
    return dict(family=family, model=str(b["model"]),
                f1=float(b["f1_macro_mean"]),
                std=float(b["f1_macro_std"]),
                tt=float(b["train_time_s_mean"]))


# ---------- 1: macro-F1 by dataset ---------------------------------------- #

def fig_macro_f1_by_dataset(agg: pd.DataFrame) -> None:
    datasets = ["imdb", "labr", "astd"]
    families = [("TF-IDF",   ["tfidf_lr", "tfidf_svm"]),
                ("Word2Vec", ["word2vec_lr", "word2vec_svm"]),
                ("Vec2Tec",  ["vec2tec_lr", "vec2tec_svm"])]

    table = []
    for ds in datasets:
        for fam, mods in families:
            row = _best_per_family(agg, ds, fam, mods)
            row["dataset"] = ds
            table.append(row)
    tdf = pd.DataFrame(table)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(datasets))
    width = 0.27

    for i, (fam, _mods) in enumerate(families):
        sub = tdf[tdf["family"] == fam].set_index("dataset").reindex(datasets)
        bars = ax.bar(x + (i - 1) * width, sub["f1"], width,
                      yerr=sub["std"], capsize=4,
                      label=fam, color=FAMILY_COLOR[fam])
        for xi, m, s, mod in zip(x + (i - 1) * width,
                                 sub["f1"], sub["std"], sub["model"]):
            ax.text(xi, m + s + 0.012, f"{m:.3f}",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([d.upper() for d in datasets])
    ax.set_ylabel("Macro-F1 (mean ± std over 3 seeds)")
    ax.set_title("Macro-F1 by dataset (best classifier in each family)")
    ax.set_ylim(0.55, 0.95)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    out = FIG_DIR / "fig_macro_f1_by_dataset.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


# ---------- 2: annotated efficiency tradeoff ----------------------------- #

def fig_efficiency_tradeoff_v2(agg: pd.DataFrame) -> None:
    # use binary datasets only (where x-scale isn't dwarfed by 4-class numbers)
    sub = agg[agg["dataset"].isin(["imdb", "labr", "astd"])].copy()
    sub["family"] = sub["model"].map({
        "tfidf_lr": "TF-IDF",       "tfidf_svm": "TF-IDF",
        "word2vec_lr": "Word2Vec",  "word2vec_svm": "Word2Vec",
        "vec2tec_lr": "Vec2Tec",    "vec2tec_svm": "Vec2Tec",
    })
    sub["short"] = sub["model"].map(SHORT)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    markers = {"imdb": "o", "labr": "s", "astd": "^"}
    for family, color in FAMILY_COLOR.items():
        for ds, marker in markers.items():
            grp = sub[(sub["family"] == family) & (sub["dataset"] == ds)]
            if grp.empty:
                continue
            ax.scatter(grp["train_time_s_mean"], grp["f1_macro_mean"],
                       s=130, c=color, marker=marker,
                       edgecolor="black", linewidth=0.6, alpha=0.9,
                       label=f"{family} / {ds.upper()}")

    # annotate every point with its short model name
    for _, r in sub.iterrows():
        ax.annotate(f"{r['short']}",
                    xy=(r["train_time_s_mean"], r["f1_macro_mean"]),
                    xytext=(6, 4), textcoords="offset points",
                    fontsize=7, alpha=0.85)

    ax.set_xscale("log")
    ax.set_xlabel("Classifier training time, seconds (log scale)")
    ax.set_ylabel("Macro-F1 (mean across seeds)")
    ax.set_title("Efficiency–quality trade-off (binary datasets)")
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    ax.legend(ncol=3, fontsize=7, loc="lower right", frameon=False)
    fig.tight_layout()
    out = FIG_DIR / "fig_efficiency_tradeoff_v2.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")

    # companion CSV
    show = sub[["dataset", "model", "family", "f1_macro_mean",
                "train_time_s_mean"]].copy()
    show.columns = ["dataset", "model", "family",
                    "f1_macro_mean", "train_time_s_mean"]
    show = show.sort_values(["dataset", "f1_macro_mean"], ascending=[True, False])
    out_csv = TBL_DIR / "efficiency_tradeoff.csv"
    save_csv(show, out_csv)
    print(f"  wrote {out_csv.relative_to(RESULTS_DIR.parent)}")


# ---------- 3: short narrative ------------------------------------------- #

def write_efficiency_summary(agg: pd.DataFrame) -> None:
    # Compute Pareto-front-like commentary: best F1 per family + its time
    families = ["TF-IDF", "Word2Vec", "Vec2Tec"]
    lines = ["# Efficiency–quality trade-off summary\n",
             "All numbers are mean across 3 seeds; training time is the "
             "classifier-fit step only (does not include vectorisation or "
             "Word2Vec pre-training).\n"]
    for ds in ["imdb", "labr", "astd"]:
        sub = agg[agg["dataset"] == ds]
        lines.append(f"\n## {ds.upper()}\n")
        lines.append("| family | best model | macro-F1 | classifier train (s) |")
        lines.append("|:-------|:-----------|---------:|---------------------:|")
        for fam in families:
            family_models = {
                "TF-IDF":   ["tfidf_lr", "tfidf_svm"],
                "Word2Vec": ["word2vec_lr", "word2vec_svm"],
                "Vec2Tec":  ["vec2tec_lr", "vec2tec_svm"],
            }[fam]
            row = _best_per_family(agg, ds, fam, family_models)
            if not row:
                continue
            lines.append(f"| {fam} | {row['model']} "
                         f"| {row['f1']:.4f} | {row['tt']:.3f} |")

    lines.append(
        "\n## Interpretation\n"
        "- TF-IDF + LR provides the best macro-F1 on every binary dataset "
        "*and* the lowest classifier training time on the small Arabic "
        "datasets (ASTD/LABR), so it dominates the Pareto frontier on this "
        "benchmark.\n"
        "- Vec2Tec consistently beats Word2Vec on macro-F1, at very similar "
        "classifier-fit cost. The Word2Vec doc-vector step is the bulk of "
        "wall-clock time for both, so adding lexicon enhancements is "
        "essentially free at inference time.\n"
        "- If you are interpretability- or domain-adaptation-constrained, "
        "Vec2Tec is the better lightweight pick over Word2Vec; if you "
        "care only about test-set accuracy on standard reviews/tweets, "
        "TF-IDF + LR is the strongest cheap baseline.\n"
    )
    out = TBL_DIR / "efficiency_tradeoff_summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"  wrote {out.relative_to(RESULTS_DIR.parent)}")


def main() -> None:
    banner("New analysis figures v2")
    sns.set_theme(context="paper", style="whitegrid", font_scale=1.05)
    agg = load_csv(RESULTS_DIR / "tables" / "aggregated_results.csv")

    fig_macro_f1_by_dataset(agg)
    fig_efficiency_tradeoff_v2(agg)
    write_efficiency_summary(agg)


if __name__ == "__main__":
    main()
