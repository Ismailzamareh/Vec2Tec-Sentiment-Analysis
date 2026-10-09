"""
Step 12 - Publication-quality figures.

Reads:
    results/metrics_aggregated.csv
    results/metrics_seed_{42,123,2024}.csv      (per-seed values for boxplots)
    results/confusion_matrices/<model>_<ds>.csv (for confusion heatmaps)

Writes (all 300 DPI):
    results/figures/fig_macro_f1_comparison.png
    results/figures/fig_vec2tec_vs_w2v.png
    results/figures/fig_confusion_matrices.png
    results/figures/fig_arabic_vs_english.png
    results/figures/fig_long_vs_short.png
    results/figures/fig_train_time_vs_f1.png

Usage:
    python src/make_figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import RESULTS_DIR, CM_DIR, DATASETS
from utils import banner, load_csv


FIG_DIR = RESULTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = (42, 123, 2024)
DPI = 300

MODELS = ("tfidf_lr", "tfidf_svm",
          "word2vec_lr", "word2vec_svm",
          "vec2tec_lr", "vec2tec_svm")

FAMILY_OF = {
    "tfidf_lr":     "TF-IDF",
    "tfidf_svm":    "TF-IDF",
    "word2vec_lr":  "Word2Vec",
    "word2vec_svm": "Word2Vec",
    "vec2tec_lr":   "Vec2Tec",
    "vec2tec_svm":  "Vec2Tec",
}
FAMILY_COLOR = {
    "TF-IDF":   "#4C72B0",
    "Word2Vec": "#7F7F7F",
    "Vec2Tec":  "#DD8452",
}


def _load_aggregated() -> pd.DataFrame:
    return load_csv(RESULTS_DIR / "metrics_aggregated.csv")


def _load_all_seeds() -> pd.DataFrame:
    frames = []
    for s in SEEDS:
        df = load_csv(RESULTS_DIR / f"metrics_seed_{s}.csv")
        df["seed"] = s
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ---------- 1.1 ----------------------------------------------------------- #

def fig_macro_f1_comparison(agg: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(DATASETS))
    width = 0.13

    for i, m in enumerate(MODELS):
        sub = agg[agg["model"] == m].set_index("dataset").reindex(DATASETS)
        means = sub["f1_macro_mean"].values
        stds  = sub["f1_macro_std"].values
        family = FAMILY_OF[m]
        edge = "black" if family == "Vec2Tec" else "none"
        lw   = 1.2  if family == "Vec2Tec" else 0
        ax.bar(x + (i - 2.5) * width, means, width,
               yerr=stds, capsize=3,
               label=m,
               color=FAMILY_COLOR[family],
               edgecolor=edge, linewidth=lw,
               alpha=0.75 if "svm" in m else 1.0)

    ax.set_xticks(x)
    ax.set_xticklabels([d.upper() for d in DATASETS])
    ax.set_ylabel("Macro-F1 (mean ± std over 3 seeds)")
    ax.set_title("Macro-F1 across datasets and models")
    ax.set_ylim(0.55, 0.95)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(ncol=3, fontsize=8, loc="lower center",
              bbox_to_anchor=(0.5, -0.22), frameon=False)
    fig.tight_layout()
    out = FIG_DIR / "fig_macro_f1_comparison.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


# ---------- 1.2 ----------------------------------------------------------- #

def _best_in_family(agg: pd.DataFrame, family_models: tuple[str, ...],
                    ds: str) -> tuple[float, float, str]:
    sub = agg[(agg["model"].isin(family_models)) & (agg["dataset"] == ds)]
    row = sub.loc[sub["f1_macro_mean"].idxmax()]
    return float(row["f1_macro_mean"]), float(row["f1_macro_std"]), str(row["model"])


def fig_vec2tec_vs_w2v(agg: pd.DataFrame) -> None:
    w2v_family = ("word2vec_lr", "word2vec_svm")
    v2t_family = ("vec2tec_lr",  "vec2tec_svm")

    rows = []
    for ds in DATASETS:
        w_mean, w_std, w_name = _best_in_family(agg, w2v_family, ds)
        v_mean, v_std, v_name = _best_in_family(agg, v2t_family, ds)
        rows.append(dict(dataset=ds,
                         w2v_mean=w_mean, w2v_std=w_std, w2v_name=w_name,
                         v2t_mean=v_mean, v2t_std=v_std, v2t_name=v_name,
                         delta=v_mean - w_mean))
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    x = np.arange(len(DATASETS))
    width = 0.36

    b1 = ax.bar(x - width/2, df["w2v_mean"], width,
                yerr=df["w2v_std"], capsize=4,
                label="Word2Vec (best in family)",
                color=FAMILY_COLOR["Word2Vec"])
    b2 = ax.bar(x + width/2, df["v2t_mean"], width,
                yerr=df["v2t_std"], capsize=4,
                label="Vec2Tec (best in family)",
                color=FAMILY_COLOR["Vec2Tec"])

    ymax = float((df[["w2v_mean", "v2t_mean"]].max(axis=1)).max())
    for i, r in df.iterrows():
        ax.annotate(f"Δ = {r['delta']*100:+.2f}%",
                    xy=(x[i], max(r["w2v_mean"], r["v2t_mean"]) + 0.012),
                    ha="center", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([d.upper() for d in DATASETS])
    ax.set_ylabel("Macro-F1 (mean ± std over 3 seeds)")
    ax.set_title("Vec2Tec vs Word2Vec (best classifier in each family)")
    ax.set_ylim(0.55, min(ymax + 0.08, 1.0))
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    out = FIG_DIR / "fig_vec2tec_vs_w2v.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


# ---------- 1.3 ----------------------------------------------------------- #

def _load_cm(model: str, ds: str) -> np.ndarray:
    df = load_csv(CM_DIR / f"{model}_{ds}.csv")
    # first column is "row" labels (true_negative / true_positive)
    pred_cols = [c for c in df.columns if c.startswith("pred_")]
    return df[pred_cols].to_numpy(dtype=int)


def fig_confusion_matrices(agg: pd.DataFrame) -> None:
    # Row 1 = best TF-IDF per dataset ; Row 2 = best Vec2Tec per dataset.
    tfidf_family = ("tfidf_lr", "tfidf_svm")
    v2t_family   = ("vec2tec_lr", "vec2tec_svm")

    fig, axes = plt.subplots(2, 3, figsize=(11, 7))
    for col, ds in enumerate(DATASETS):
        _, _, tfidf_best = _best_in_family(agg, tfidf_family, ds)
        _, _, v2t_best   = _best_in_family(agg, v2t_family,   ds)

        for row_idx, model in enumerate([tfidf_best, v2t_best]):
            cm = _load_cm(model, ds)
            ax = axes[row_idx, col]
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        cbar=False, ax=ax,
                        xticklabels=["pred neg", "pred pos"],
                        yticklabels=["true neg", "true pos"])
            ax.set_title(f"{ds.upper()} - {model}", fontsize=10)

    fig.suptitle("Confusion matrices (top: best TF-IDF, bottom: best Vec2Tec)",
                 fontsize=12, y=1.00)
    fig.tight_layout()
    out = FIG_DIR / "fig_confusion_matrices.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


# ---------- 1.4 ----------------------------------------------------------- #

def fig_arabic_vs_english(all_seeds: pd.DataFrame) -> None:
    df = all_seeds.copy()
    df["language"] = df["dataset"].map(
        {"imdb": "English (IMDb)", "labr": "Arabic (LABR+ASTD)",
         "astd": "Arabic (LABR+ASTD)"})

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(data=df, x="language", y="f1_macro",
                order=["English (IMDb)", "Arabic (LABR+ASTD)"],
                hue="language",
                palette={"English (IMDb)": "#4C72B0",
                         "Arabic (LABR+ASTD)": "#DD8452"},
                legend=False, ax=ax)
    sns.stripplot(data=df, x="language", y="f1_macro",
                  order=["English (IMDb)", "Arabic (LABR+ASTD)"],
                  color="black", size=4, alpha=0.6, ax=ax)

    en_mean = df[df["language"].str.startswith("English")]["f1_macro"].mean()
    ar_mean = df[df["language"].str.startswith("Arabic")]["f1_macro"].mean()
    gap = en_mean - ar_mean
    ax.set_title(f"Macro-F1 by language  (English − Arabic gap = {gap*100:+.2f} pp)")
    ax.set_ylabel("Macro-F1 (per model × seed point)")
    ax.set_xlabel("")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    out = FIG_DIR / "fig_arabic_vs_english.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


# ---------- 1.5 ----------------------------------------------------------- #

def fig_long_vs_short(all_seeds: pd.DataFrame) -> None:
    df = all_seeds.copy()
    df["length"] = df["dataset"].map(
        {"imdb": "Long text (IMDb+LABR)", "labr": "Long text (IMDb+LABR)",
         "astd": "Short text (ASTD tweets)"})

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(data=df, x="length", y="f1_macro",
                order=["Long text (IMDb+LABR)", "Short text (ASTD tweets)"],
                hue="length",
                palette={"Long text (IMDb+LABR)": "#4C72B0",
                         "Short text (ASTD tweets)": "#C44E52"},
                legend=False, ax=ax)
    sns.stripplot(data=df, x="length", y="f1_macro",
                  order=["Long text (IMDb+LABR)", "Short text (ASTD tweets)"],
                  color="black", size=4, alpha=0.6, ax=ax)

    long_mean  = df[df["length"].str.startswith("Long")]["f1_macro"].mean()
    short_mean = df[df["length"].str.startswith("Short")]["f1_macro"].mean()
    gap = long_mean - short_mean
    ax.set_title(f"Macro-F1 by text length  (Long − Short gap = {gap*100:+.2f} pp)")
    ax.set_ylabel("Macro-F1 (per model × seed point)")
    ax.set_xlabel("")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    out = FIG_DIR / "fig_long_vs_short.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


# ---------- 1.6 ----------------------------------------------------------- #

def fig_train_time_vs_f1(all_seeds: pd.DataFrame) -> None:
    # mean across seeds
    df = (all_seeds.groupby(["model", "dataset"], as_index=False)
                   .agg(train_time_s=("train_time_s", "mean"),
                        f1_macro=("f1_macro", "mean")))
    df["family"]  = df["model"].map(FAMILY_OF)
    df["marker"]  = df["dataset"].map({"imdb": "o", "labr": "s", "astd": "^"})

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for family, color in FAMILY_COLOR.items():
        for ds, marker in {"imdb": "o", "labr": "s", "astd": "^"}.items():
            sub = df[(df["family"] == family) & (df["dataset"] == ds)]
            if sub.empty:
                continue
            ax.scatter(sub["train_time_s"], sub["f1_macro"],
                       s=110, c=color, marker=marker,
                       edgecolor="black", linewidth=0.5, alpha=0.85,
                       label=f"{family} / {ds.upper()}")

    ax.set_xscale("log")
    ax.set_xlabel("Classifier training time (s, log scale)")
    ax.set_ylabel("Macro-F1 (mean across seeds)")
    ax.set_title("Efficiency–quality trade-off")
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    ax.legend(ncol=3, fontsize=7, loc="lower right", frameon=False)
    fig.tight_layout()
    out = FIG_DIR / "fig_train_time_vs_f1.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


# ---------- main ---------------------------------------------------------- #

def main() -> None:
    banner("Generating publication figures (300 DPI)")
    sns.set_theme(context="paper", style="whitegrid", font_scale=1.05)

    agg       = _load_aggregated()
    all_seeds = _load_all_seeds()

    fig_macro_f1_comparison(agg)
    fig_vec2tec_vs_w2v(agg)
    fig_confusion_matrices(agg)
    fig_arabic_vs_english(all_seeds)
    fig_long_vs_short(all_seeds)
    fig_train_time_vs_f1(all_seeds)

    banner(f"All figures saved under {FIG_DIR.relative_to(RESULTS_DIR.parent)}",
           char="-")


if __name__ == "__main__":
    main()
