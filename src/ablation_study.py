"""
Ablation study for Vec2Tec.

We hold the underlying Word2Vec model fixed (re-using the cached
w2v_<ds>.model produced by train_word2vec.py) and ONLY vary the
enhancement stack that turns word vectors into a document vector.

Variants:
    A. word2vec_plain   -- mean of word vectors (no enhancements)
    B. v2t_lexicon      -- SentimentLexiconEnhancement only
    C. v2t_synonym      -- SynonymExpansionEnhancement only (top_k=3)
    D. v2t_contextual   -- ContextualWeightingEnhancement only (tau=1.0)
    E. v2t_full         -- PolarityRetrofitEnhancement + SentimentLexicon
                           (the default Vec2Tec configuration)

For each variant on each binary dataset we measure:
    - accuracy, precision_macro, recall_macro, f1_macro, f1_weighted
    - classifier training time (LR fit only)
    - prediction time (clf.predict on the full test set)
    - delta vs. variant A (word2vec_plain) for accuracy and f1_macro

Single-seed run (uses whatever RANDOM_SEED is currently in config.py),
so variants are directly comparable on the same train/test split and
the same underlying w2v model.

Outputs:
    results/tables/ablation_results.csv
    results/tables/ablation_results.md

Usage:
    python src/ablation_study.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             f1_score)

from config import (PROCESSED_DIR, MODELS_DIR, RESULTS_DIR,
                    DATASETS, DATASETS_4CLASS, RANDOM_SEED, W2V_DIM)
from utils import banner, load_csv, save_csv
from vec2tec_module import (Vec2Tec,
                            SentimentLexiconEnhancement,
                            SynonymExpansionEnhancement,
                            ContextualWeightingEnhancement,
                            PolarityRetrofitEnhancement)


TBL_DIR = RESULTS_DIR / "tables"
TBL_DIR.mkdir(parents=True, exist_ok=True)


VARIANTS = [
    ("word2vec_plain", []),
    ("v2t_lexicon",    [SentimentLexiconEnhancement(beta=0.5)]),
    ("v2t_synonym",    [SynonymExpansionEnhancement(top_k=3)]),
    ("v2t_contextual", [ContextualWeightingEnhancement(tau=1.0)]),
    ("v2t_full",       [PolarityRetrofitEnhancement(alpha=0.3, tau=6.0, top_k=100),
                        SentimentLexiconEnhancement(beta=0.5)]),
]


def tokenize(text: str) -> list[str]:
    return str(text).split()


def _build_doc_vectors(model: Vec2Tec, tokens_list) -> np.ndarray:
    return np.vstack([model.document_vector(toks) for toks in tokens_list])


def run_on_dataset(ds: str) -> list[dict]:
    banner(f"Ablation on {ds.upper()}")
    train = load_csv(PROCESSED_DIR / f"{ds}_train.csv")
    test  = load_csv(PROCESSED_DIR / f"{ds}_test.csv")
    tr_tok = [tokenize(t) for t in train["text"].fillna("")]
    te_tok = [tokenize(t) for t in test["text"].fillna("")]
    ytr, yte = train["label"].values, test["label"].values
    print(f"  data: train={len(tr_tok)}, test={len(te_tok)}, "
          f"n_classes={len(set(ytr.tolist()) | set(yte.tolist()))}")

    w2v_path = MODELS_DIR / f"w2v_{ds}.model"
    lex_path = RESULTS_DIR / f"lex_{ds}.json"
    if not w2v_path.exists():
        print(f"  SKIP: {w2v_path.name} missing — run train_word2vec.py first")
        return []
    w2v = Word2Vec.load(str(w2v_path))
    lex: dict[str, float] = {}
    if lex_path.exists():
        with open(lex_path, "r", encoding="utf-8") as f:
            lex = {k: float(v) for k, v in json.load(f).items()}

    rows: list[dict] = []
    lr_solver = "lbfgs" if ds in DATASETS_4CLASS else "liblinear"

    for variant_name, enhancements in VARIANTS:
        v2t = Vec2Tec(w2v.wv, lex, enhancements, dim=W2V_DIM)
        v2t.fit()

        t0 = time.time()
        Xtr = _build_doc_vectors(v2t, tr_tok)
        Xte = _build_doc_vectors(v2t, te_tok)
        vec_time = time.time() - t0

        t0 = time.time()
        clf = LogisticRegression(max_iter=1000, C=2.0,
                                 class_weight="balanced", solver=lr_solver)
        clf.fit(Xtr, ytr)
        train_time = time.time() - t0

        t0 = time.time()
        ypred = clf.predict(Xte)
        predict_time = time.time() - t0

        acc = accuracy_score(yte, ypred)
        p_m, r_m, f1_m, _ = precision_recall_fscore_support(
            yte, ypred, average="macro", zero_division=0)
        f1_w = f1_score(yte, ypred, average="weighted", zero_division=0)

        rows.append(dict(
            dataset=ds,
            variant=variant_name,
            accuracy=round(float(acc), 6),
            precision_macro=round(float(p_m), 6),
            recall_macro=round(float(r_m), 6),
            f1_macro=round(float(f1_m), 6),
            f1_weighted=round(float(f1_w), 6),
            doc_vector_build_s=round(vec_time, 4),
            train_time_s=round(train_time, 4),
            predict_time_s=round(predict_time, 4),
            seed=RANDOM_SEED,
        ))
        print(f"  {variant_name:18s} | acc={acc:.4f} | "
              f"f1_macro={f1_m:.4f} | "
              f"vec={vec_time:.2f}s | train={train_time:.2f}s | "
              f"predict={predict_time:.3f}s")

    return rows


def main() -> None:
    banner(f"Vec2Tec ablation study (seed={RANDOM_SEED})")
    all_rows: list[dict] = []
    for ds in DATASETS:
        all_rows.extend(run_on_dataset(ds))

    df = pd.DataFrame(all_rows)
    # add delta vs word2vec_plain per dataset
    base = (df[df["variant"] == "word2vec_plain"]
              .set_index("dataset")[["accuracy", "f1_macro"]]
              .rename(columns={"accuracy": "acc_base",
                               "f1_macro": "f1_base"}))
    df = df.merge(base, left_on="dataset", right_index=True, how="left")
    df["delta_accuracy_pp"] = ((df["accuracy"] - df["acc_base"]) * 100).round(4)
    df["delta_f1_macro_pp"] = ((df["f1_macro"] - df["f1_base"]) * 100).round(4)
    df = df.drop(columns=["acc_base", "f1_base"])

    out_csv = TBL_DIR / "ablation_results.csv"
    save_csv(df, out_csv)
    print(f"\n  wrote {out_csv.relative_to(RESULTS_DIR.parent)}")

    # markdown view per dataset
    md = ["# Vec2Tec ablation study\n",
          f"Single-seed run (seed = {RANDOM_SEED}). "
          "Underlying Word2Vec model and corpus lexicon are held fixed; "
          "only the enhancement stack is varied.\n",
          "Deltas are vs. `word2vec_plain` (mean of word vectors, no "
          "enhancement) on the SAME dataset.\n"]
    for ds, sub in df.groupby("dataset"):
        sub = sub.sort_values("f1_macro", ascending=False)
        md.append(f"\n## {ds.upper()}\n")
        show = sub[["variant", "accuracy", "f1_macro", "f1_weighted",
                    "train_time_s", "predict_time_s",
                    "delta_accuracy_pp", "delta_f1_macro_pp"]]
        md.append(show.to_markdown(index=False))
    out_md = TBL_DIR / "ablation_results.md"
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"  wrote {out_md.relative_to(RESULTS_DIR.parent)}")


if __name__ == "__main__":
    main()
