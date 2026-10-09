"""
Build the complete long-format results table by stitching together every
per-seed snapshot we have on disk.

Inputs (whichever are present):
    results/metrics_seed_{42,123,2024}.csv          (binary: imdb/labr/astd)
    results/metrics_seed_{42,123,2024}_astd4.csv    (4-class)

Outputs:
    results/tables/full_results_long.csv
        columns:
            dataset, language, text_type, n_classes,
            model, family, classifier, seed,
            accuracy, precision_macro, recall_macro,
            f1_macro, f1_weighted,
            train_time_s, predict_time_s

    results/tables/aggregated_results.csv
        columns:
            dataset, model, family, classifier,
            <metric>_mean, <metric>_std for each metric,
            n_seeds

    results/tables/best_per_dataset.csv
        - one row per dataset with the best model (by f1_macro_mean) AND
          the best Word2Vec-based model (Word2Vec OR Vec2Tec family).

predict_time_s is set to NaN — the original pipeline scripts did not
record it. New scripts (ablation, transformer) DO record it.

Usage:
    python src/build_full_results.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import RESULTS_DIR
from utils import banner, load_csv, save_csv


TBL_DIR = RESULTS_DIR / "tables"
TBL_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = (42, 123, 2024)

DATASET_META = {
    "imdb":  dict(language="English", text_type="long review",  n_classes=2),
    "labr":  dict(language="Arabic",  text_type="long review",  n_classes=2),
    "astd":  dict(language="Arabic",  text_type="short tweet",  n_classes=2),
    "astd4": dict(language="Arabic",  text_type="short tweet",  n_classes=4),
}

FAMILY_OF = {
    "tfidf_lr":         "TF-IDF",
    "tfidf_svm":        "TF-IDF",
    "word2vec_lr":      "Word2Vec",
    "word2vec_svm":     "Word2Vec",
    "vec2tec_lr":       "Vec2Tec",
    "vec2tec_svm":      "Vec2Tec",
    "vec2tec_plain_lr": "Vec2Tec(plain)",
    "vec2tec_plain_svm":"Vec2Tec(plain)",
}
CLASSIFIER_OF = {
    "tfidf_lr":         "LR",  "tfidf_svm":        "SVM",
    "word2vec_lr":      "LR",  "word2vec_svm":     "SVM",
    "vec2tec_lr":       "LR",  "vec2tec_svm":      "SVM",
    "vec2tec_plain_lr": "LR",  "vec2tec_plain_svm":"SVM",
}


def _read_seed_files(suffix: str = "") -> pd.DataFrame:
    """Stack metrics_seed_<s>{suffix}.csv files into a long DataFrame."""
    frames = []
    for s in SEEDS:
        p = RESULTS_DIR / f"metrics_seed_{s}{suffix}.csv"
        if not p.exists():
            continue
        df = load_csv(p)
        df["seed"] = s
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _enrich(df: pd.DataFrame) -> pd.DataFrame:
    df["language"]  = df["dataset"].map(lambda d: DATASET_META[d]["language"])
    df["text_type"] = df["dataset"].map(lambda d: DATASET_META[d]["text_type"])
    df["n_classes"] = df["dataset"].map(lambda d: DATASET_META[d]["n_classes"])
    df["family"]    = df["model"].map(FAMILY_OF).fillna("other")
    df["classifier"]= df["model"].map(CLASSIFIER_OF).fillna("?")
    if "predict_time_s" not in df.columns:
        df["predict_time_s"] = np.nan
    return df


def build_long_table() -> pd.DataFrame:
    binary = _read_seed_files(suffix="")
    cls4   = _read_seed_files(suffix="_astd4")
    parts = [df for df in (binary, cls4) if not df.empty]
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    df = _enrich(df)

    # canonical column order
    cols = ["dataset", "language", "text_type", "n_classes",
            "model", "family", "classifier", "seed",
            "accuracy", "precision_macro", "recall_macro",
            "f1_macro", "f1_weighted",
            "train_time_s", "predict_time_s"]
    df = df[[c for c in cols if c in df.columns]]
    df = df.sort_values(["dataset", "model", "seed"]).reset_index(drop=True)
    return df


def build_aggregated(long_df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["accuracy", "precision_macro", "recall_macro",
               "f1_macro", "f1_weighted",
               "train_time_s", "predict_time_s"]
    agg_funcs = {m: ["mean", "std"] for m in metrics}

    grouped = (long_df.groupby(["dataset", "model"], as_index=False)
                       .agg(agg_funcs))
    grouped.columns = [
        "_".join(c).rstrip("_") if isinstance(c, tuple) else c
        for c in grouped.columns
    ]
    n_seeds = (long_df.groupby(["dataset", "model"]).size()
                       .rename("n_seeds").reset_index())
    grouped = grouped.merge(n_seeds, on=["dataset", "model"], how="left")

    grouped["family"]     = grouped["model"].map(FAMILY_OF).fillna("other")
    grouped["classifier"] = grouped["model"].map(CLASSIFIER_OF).fillna("?")

    # round
    for col in grouped.columns:
        if col not in ("dataset", "model", "family", "classifier", "n_seeds"):
            grouped[col] = pd.to_numeric(grouped[col], errors="coerce").round(6)

    front = ["dataset", "model", "family", "classifier", "n_seeds"]
    rest  = [c for c in grouped.columns if c not in front]
    grouped = grouped[front + rest]
    return grouped.sort_values(["dataset", "f1_macro_mean"],
                               ascending=[True, False]).reset_index(drop=True)


def build_best_per_dataset(agg: pd.DataFrame) -> pd.DataFrame:
    rows = []
    w2v_like = {"Word2Vec", "Vec2Tec", "Vec2Tec(plain)"}
    for ds, sub in agg.groupby("dataset"):
        best = sub.loc[sub["f1_macro_mean"].idxmax()]
        # best Word2Vec-based (W2V, Vec2Tec, Vec2Tec plain)
        wsub = sub[sub["family"].isin(w2v_like)]
        if wsub.empty:
            w2v_best_model = ""
            w2v_best_f1    = np.nan
            w2v_best_std   = np.nan
        else:
            w2v_best = wsub.loc[wsub["f1_macro_mean"].idxmax()]
            w2v_best_model = str(w2v_best["model"])
            w2v_best_f1    = float(w2v_best["f1_macro_mean"])
            w2v_best_std   = float(w2v_best["f1_macro_std"])

        rows.append(dict(
            dataset=ds,
            best_model=str(best["model"]),
            best_family=str(best["family"]),
            best_f1_macro_mean=float(best["f1_macro_mean"]),
            best_f1_macro_std=float(best["f1_macro_std"]),
            best_accuracy_mean=float(best["accuracy_mean"]),
            best_w2v_based_model=w2v_best_model,
            best_w2v_based_f1_macro_mean=w2v_best_f1,
            best_w2v_based_f1_macro_std=w2v_best_std,
            gap_best_minus_w2v_pp=round((best["f1_macro_mean"] - w2v_best_f1) * 100, 4)
                                   if not pd.isna(w2v_best_f1) else np.nan,
        ))
    return pd.DataFrame(rows).sort_values("dataset").reset_index(drop=True)


def main() -> None:
    banner("Building full long-format + aggregated result tables")

    long_df = build_long_table()
    if long_df.empty:
        print("  no metrics_seed_*.csv files found.")
        return
    out_long = TBL_DIR / "full_results_long.csv"
    save_csv(long_df, out_long)
    print(f"  wrote {out_long.relative_to(RESULTS_DIR.parent)} "
          f"({len(long_df)} rows)")

    agg = build_aggregated(long_df)
    out_agg = TBL_DIR / "aggregated_results.csv"
    save_csv(agg, out_agg)
    print(f"  wrote {out_agg.relative_to(RESULTS_DIR.parent)} "
          f"({len(agg)} rows)")

    best = build_best_per_dataset(agg)
    out_best = TBL_DIR / "best_per_dataset.csv"
    save_csv(best, out_best)
    print(f"  wrote {out_best.relative_to(RESULTS_DIR.parent)} "
          f"({len(best)} rows)")

    banner("Aggregated overview", char="-")
    print(agg[["dataset", "model", "family", "n_seeds",
               "accuracy_mean", "f1_macro_mean", "f1_macro_std",
               "train_time_s_mean"]].to_string(index=False))
    banner("Best per dataset", char="-")
    print(best.to_string(index=False))


if __name__ == "__main__":
    main()
