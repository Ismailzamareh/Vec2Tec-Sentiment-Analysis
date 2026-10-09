"""
Step 7 - Per-dataset comparison tables.

Reads metrics_per_model.csv and writes one comparison CSV per dataset:

    results/comparison_imdb.csv
    results/comparison_labr.csv
    results/comparison_astd.csv

Each table contains the headline metrics (Accuracy, Macro-F1, Weighted-F1)
for TF-IDF / Word2Vec / Vec2Tec families, sorted by macro-F1 descending.

Usage:
    python src/compare.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import RESULTS_DIR, DATASETS, DATASETS_4CLASS
from utils import banner, load_csv, save_csv


ALL_DATASETS = tuple(DATASETS) + tuple(DATASETS_4CLASS)


MODEL_FAMILY = {
    "tfidf_lr":         "TF-IDF",
    "tfidf_svm":        "TF-IDF",
    "word2vec_lr":      "Word2Vec",
    "word2vec_svm":     "Word2Vec",
    "vec2tec_lr":       "Vec2Tec",
    "vec2tec_svm":      "Vec2Tec",
    "vec2tec_plain_lr": "Vec2Tec (plain)",
    "vec2tec_plain_svm": "Vec2Tec (plain)",
}

PRETTY = {
    "tfidf_lr":         "TF-IDF + LR",
    "tfidf_svm":        "TF-IDF + SVM",
    "word2vec_lr":      "Word2Vec + LR",
    "word2vec_svm":     "Word2Vec + SVM",
    "vec2tec_lr":       "Vec2Tec + LR",
    "vec2tec_svm":      "Vec2Tec + SVM",
    "vec2tec_plain_lr": "Vec2Tec(plain) + LR",
    "vec2tec_plain_svm": "Vec2Tec(plain) + SVM",
}


def main() -> None:
    src = RESULTS_DIR / "metrics_per_model.csv"
    df = load_csv(src)
    df["family"] = df["model"].map(MODEL_FAMILY)
    df["model_pretty"] = df["model"].map(PRETTY).fillna(df["model"])

    banner("Per-dataset comparison tables")
    for ds in ALL_DATASETS:
        sub = df[df["dataset"] == ds].copy()
        if sub.empty:
            print(f"  no rows for {ds}, skipping.")
            continue
        sub = sub.sort_values("f1_macro", ascending=False).reset_index(drop=True)
        out = sub[["family", "model_pretty", "accuracy",
                   "precision_macro", "recall_macro",
                   "f1_macro", "f1_weighted", "train_time_s"]]
        out.columns = ["family", "model", "accuracy", "precision_macro",
                       "recall_macro", "f1_macro", "f1_weighted",
                       "train_time_s"]
        path = RESULTS_DIR / f"comparison_{ds}.csv"
        save_csv(out, path)
        print(f"\n  {ds.upper()}  -> {path.name}")
        print(out.to_string(index=False))


if __name__ == "__main__":
    main()
