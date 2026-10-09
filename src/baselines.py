"""
Step 3 - TF-IDF baselines.

Trains two baselines on each processed dataset:
    * TF-IDF (1,2-grams) + Logistic Regression
    * TF-IDF (1,2-grams) + Linear SVM

The trained vectoriser and classifier are pickled to  models/  so that
they can later be re-loaded by the evaluation script.

Usage:
    python src/baselines.py
    python src/baselines.py --dataset imdb         # one dataset only
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from config import (
    PROCESSED_DIR, MODELS_DIR, RESULTS_DIR, DATASETS, DATASETS_4CLASS,
    TFIDF_NGRAM, TFIDF_MAX, TFIDF_MIN_DF,
)
from utils import banner, load_csv, save_pickle


ALL_DATASETS = tuple(DATASETS) + tuple(DATASETS_4CLASS)


def _vectorizer():
    return TfidfVectorizer(
        max_features=TFIDF_MAX,
        ngram_range=TFIDF_NGRAM,
        min_df=TFIDF_MIN_DF,
        sublinear_tf=True,
    )


def train_one(ds: str) -> dict:
    banner(f"TF-IDF baselines on {ds.upper()}")
    train = load_csv(PROCESSED_DIR / f"{ds}_train.csv")
    test  = load_csv(PROCESSED_DIR / f"{ds}_test.csv")
    ytr, yte = train["label"].values, test["label"].values

    # For 4-class datasets that ship an official validation split, just
    # acknowledge it in the log (tuning is left for a future step).
    if ds in DATASETS_4CLASS:
        val_path = PROCESSED_DIR / f"{ds}_validation.csv"
        if val_path.exists():
            val = load_csv(val_path)
            print(f"  validation set available ({len(val)} rows) — not used "
                  "for tuning yet")
        n_classes = len(set(ytr.tolist()) | set(yte.tolist()))
        print(f"  multi-class run (n_classes={n_classes})")

    vec = _vectorizer()
    t = time.time()
    Xtr = vec.fit_transform(train["text"].fillna(""))
    Xte = vec.transform(test["text"].fillna(""))
    vec_t = time.time() - t
    print(f"  vectorised in {vec_t:.2f}s | features={Xtr.shape[1]}")

    save_pickle(vec, MODELS_DIR / f"tfidf_vectorizer_{ds}.pkl")

    out = {}
    # LR
    t = time.time()
    lr_solver = "lbfgs" if ds in DATASETS_4CLASS else "liblinear"
    lr = LogisticRegression(max_iter=1000, C=4.0, n_jobs=1,
                            class_weight="balanced", solver=lr_solver)
    lr.fit(Xtr, ytr)
    out["tfidf_lr"] = dict(
        model_name="tfidf_lr",
        train_time_s=time.time() - t,
        predictions=lr.predict(Xte).tolist(),
        y_true=yte.tolist(),
    )
    save_pickle(lr, MODELS_DIR / f"clf_tfidf_lr_{ds}.pkl")
    print(f"  tfidf_lr trained in {out['tfidf_lr']['train_time_s']:.2f}s")

    # SVM
    t = time.time()
    svm = LinearSVC(C=1.0, class_weight="balanced")
    svm.fit(Xtr, ytr)
    out["tfidf_svm"] = dict(
        model_name="tfidf_svm",
        train_time_s=time.time() - t,
        predictions=svm.predict(Xte).tolist(),
        y_true=yte.tolist(),
    )
    save_pickle(svm, MODELS_DIR / f"clf_tfidf_svm_{ds}.pkl")
    print(f"  tfidf_svm trained in {out['tfidf_svm']['train_time_s']:.2f}s")

    # persist predictions to disk -- consumed by evaluate.py
    save_pickle(out, RESULTS_DIR / f"_preds_tfidf_{ds}.pkl")
    return out


def main(only: str | None) -> None:
    targets = (only,) if only else DATASETS
    for ds in targets:
        train_one(ds)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=ALL_DATASETS, default=None)
    args = p.parse_args()
    main(args.dataset)
