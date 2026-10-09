"""
Step 4 - Word2Vec embedding models.

For each dataset:
    1. tokenise the cleaned text (whitespace tokens).
    2. train a Skip-gram Word2Vec model separately for that dataset.
    3. build a document vector as the mean of its word vectors.
    4. train a Logistic Regression  AND  a Linear SVM head on top.
    5. persist:
         models/w2v_<dataset>.model            (gensim format)
         models/clf_word2vec_lr_<dataset>.pkl
         models/clf_word2vec_svm_<dataset>.pkl
       and write predictions to
         results/_preds_word2vec_<dataset>.pkl
       so the evaluation script can consume them.

Usage:
    python src/train_word2vec.py
    python src/train_word2vec.py --dataset imdb
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from config import (
    PROCESSED_DIR, MODELS_DIR, RESULTS_DIR, DATASETS, DATASETS_4CLASS,
    W2V_DIM, W2V_WINDOW, W2V_MIN_COUNT, W2V_EPOCHS, W2V_SG, W2V_WORKERS,
    RANDOM_SEED,
)
from utils import banner, load_csv, save_pickle


ALL_DATASETS = tuple(DATASETS) + tuple(DATASETS_4CLASS)


def tokenize(text: str) -> list[str]:
    return str(text).split()


def doc_vector(tokens: list[str], wv, dim: int) -> np.ndarray:
    vecs = [wv[t] for t in tokens if t in wv]
    if not vecs:
        return np.zeros(dim, dtype=np.float32)
    return np.mean(vecs, axis=0)


def epochs_for(ds: str) -> int:
    """Adapt epoch budget so the run fits a single CPU pass."""
    return {"imdb": 10, "labr": 10, "astd": 10, "astd4": 10}.get(ds, W2V_EPOCHS)


def train_one(ds: str) -> dict:
    banner(f"Word2Vec on {ds.upper()}")
    train = load_csv(PROCESSED_DIR / f"{ds}_train.csv")
    test  = load_csv(PROCESSED_DIR / f"{ds}_test.csv")

    tr_tok = [tokenize(t) for t in train["text"].fillna("")]
    te_tok = [tokenize(t) for t in test["text"].fillna("")]
    ytr = train["label"].values
    yte = test["label"].values

    if ds in DATASETS_4CLASS:
        val_path = PROCESSED_DIR / f"{ds}_validation.csv"
        if val_path.exists():
            val = load_csv(val_path)
            print(f"  validation set available ({len(val)} rows) — not used "
                  "for tuning yet")
        n_classes = len(set(ytr.tolist()) | set(yte.tolist()))
        print(f"  multi-class run (n_classes={n_classes})")

    # 1) Train Word2Vec
    epochs = epochs_for(ds)
    t = time.time()
    w2v = Word2Vec(
        sentences=tr_tok,
        vector_size=W2V_DIM,
        window=W2V_WINDOW,
        min_count=W2V_MIN_COUNT,
        sg=W2V_SG,
        epochs=epochs,
        workers=W2V_WORKERS,
        seed=RANDOM_SEED,
    )
    train_time_w2v = time.time() - t
    print(f"  w2v trained ({epochs} epochs, vocab={len(w2v.wv)}) "
          f"in {train_time_w2v:.2f}s")

    # 2) Persist Word2Vec model
    w2v_path = MODELS_DIR / f"w2v_{ds}.model"
    w2v.save(str(w2v_path))
    print(f"  saved {w2v_path.name}")

    # 3) Build document vectors
    t = time.time()
    Xtr = np.vstack([doc_vector(t, w2v.wv, W2V_DIM) for t in tr_tok])
    Xte = np.vstack([doc_vector(t, w2v.wv, W2V_DIM) for t in te_tok])
    print(f"  doc-vectors built in {time.time()-t:.2f}s "
          f"(train={Xtr.shape}, test={Xte.shape})")

    out: dict = {}

    # 4a) Logistic Regression
    t = time.time()
    lr_solver = "lbfgs" if ds in DATASETS_4CLASS else "liblinear"
    lr = LogisticRegression(max_iter=1000, C=2.0, n_jobs=1,
                            class_weight="balanced", solver=lr_solver)
    lr.fit(Xtr, ytr)
    out["word2vec_lr"] = dict(
        model_name="word2vec_lr",
        train_time_s=time.time() - t,
        predictions=lr.predict(Xte).tolist(),
        y_true=yte.tolist(),
    )
    save_pickle(lr, MODELS_DIR / f"clf_word2vec_lr_{ds}.pkl")
    print(f"  word2vec_lr trained in {out['word2vec_lr']['train_time_s']:.2f}s")

    # 4b) Linear SVM
    t = time.time()
    svm = LinearSVC(C=1.0, class_weight="balanced")
    svm.fit(Xtr, ytr)
    out["word2vec_svm"] = dict(
        model_name="word2vec_svm",
        train_time_s=time.time() - t,
        predictions=svm.predict(Xte).tolist(),
        y_true=yte.tolist(),
    )
    save_pickle(svm, MODELS_DIR / f"clf_word2vec_svm_{ds}.pkl")
    print(f"  word2vec_svm trained in {out['word2vec_svm']['train_time_s']:.2f}s")

    save_pickle(out, RESULTS_DIR / f"_preds_word2vec_{ds}.pkl")
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
