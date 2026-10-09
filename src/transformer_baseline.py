"""
Transformer baseline: encode texts with a frozen multilingual sentence
encoder, then train the SAME LR head used by Vec2Tec on top.

Model chosen:
    sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
    - 384-dim outputs
    - covers English AND Arabic (~50 languages)
    - small enough (~120 MB) to run on CPU in minutes
    - frozen: we never fine-tune; we treat it as a fixed feature extractor.

This positions Vec2Tec as a lightweight / interpretable alternative,
NOT as a replacement for transformers. We expect the transformer to be
stronger on absolute macro-F1.

For each binary dataset we:
    1. Load processed train/test CSVs.
    2. Encode with the frozen MiniLM (with a small progress log).
    3. Fit a LogisticRegression head (lbfgs, balanced).
    4. Report accuracy, precision/recall/f1 macro+weighted,
       encoder_time, train_time, predict_time.

Outputs:
    results/tables/transformer_results.csv
    results/tables/transformer_results.md
    results/_preds_transformer_<ds>.pkl   (for downstream tests)

Usage:
    python src/transformer_baseline.py
    python src/transformer_baseline.py --dataset imdb
    python src/transformer_baseline.py --max-train 5000     # quick run
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             f1_score)

from config import (PROCESSED_DIR, RESULTS_DIR, RANDOM_SEED,
                    DATASETS, DATASETS_4CLASS)
from utils import banner, load_csv, save_pickle, save_csv


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TBL_DIR = RESULTS_DIR / "tables"
TBL_DIR.mkdir(parents=True, exist_ok=True)
ALL_DS = tuple(DATASETS) + tuple(DATASETS_4CLASS)


def _maybe_subsample(df: pd.DataFrame, max_n: int) -> pd.DataFrame:
    if max_n <= 0 or len(df) <= max_n:
        return df.reset_index(drop=True)
    return df.sample(n=max_n, random_state=RANDOM_SEED).reset_index(drop=True)


def encode(texts: list[str], model, batch: int = 64) -> np.ndarray:
    return model.encode(texts, batch_size=batch, show_progress_bar=False,
                        convert_to_numpy=True, normalize_embeddings=True)


def run_one(ds: str, model, max_train: int = 0, max_test: int = 0) -> dict:
    banner(f"Transformer baseline on {ds.upper()}")
    train = load_csv(PROCESSED_DIR / f"{ds}_train.csv")
    test  = load_csv(PROCESSED_DIR / f"{ds}_test.csv")
    if max_train > 0:
        train = _maybe_subsample(train, max_train)
        print(f"  sub-sampled train to {len(train)} rows (--max-train)")
    if max_test > 0:
        test = _maybe_subsample(test, max_test)
        print(f"  sub-sampled test to {len(test)} rows (--max-test)")

    print(f"  encoding {len(train)} train texts...")
    t0 = time.time()
    Xtr = encode(train["text"].fillna("").astype(str).tolist(), model)
    enc_train_time = time.time() - t0

    print(f"  encoding {len(test)} test texts...")
    t0 = time.time()
    Xte = encode(test["text"].fillna("").astype(str).tolist(), model)
    enc_test_time = time.time() - t0

    ytr = train["label"].values
    yte = test["label"].values

    t0 = time.time()
    clf = LogisticRegression(max_iter=2000, C=2.0,
                             class_weight="balanced", solver="lbfgs")
    clf.fit(Xtr, ytr)
    train_time = time.time() - t0

    t0 = time.time()
    ypred = clf.predict(Xte)
    predict_time = time.time() - t0

    acc = accuracy_score(yte, ypred)
    p_m, r_m, f1_m, _ = precision_recall_fscore_support(
        yte, ypred, average="macro", zero_division=0)
    f1_w = f1_score(yte, ypred, average="weighted", zero_division=0)

    out = {
        "transformer_lr": dict(
            model_name="transformer_lr",
            train_time_s=train_time,
            predictions=ypred.tolist(),
            y_true=yte.tolist(),
        )
    }
    save_pickle(out, RESULTS_DIR / f"_preds_transformer_{ds}.pkl")

    row = dict(
        dataset=ds,
        model="transformer_lr",
        encoder=MODEL_NAME,
        dim=int(Xtr.shape[1]),
        n_train=int(len(Xtr)),
        n_test=int(len(Xte)),
        accuracy=round(float(acc), 6),
        precision_macro=round(float(p_m), 6),
        recall_macro=round(float(r_m), 6),
        f1_macro=round(float(f1_m), 6),
        f1_weighted=round(float(f1_w), 6),
        encode_train_s=round(enc_train_time, 4),
        encode_test_s=round(enc_test_time, 4),
        train_time_s=round(train_time, 4),
        predict_time_s=round(predict_time, 4),
        seed=RANDOM_SEED,
    )
    print(f"  acc={acc:.4f} | f1_macro={f1_m:.4f} | "
          f"enc_train={enc_train_time:.1f}s | enc_test={enc_test_time:.1f}s "
          f"| train={train_time:.2f}s | predict={predict_time:.3f}s")
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=ALL_DS, default=None,
                    help="Run only this dataset.")
    ap.add_argument("--max-train", type=int, default=0,
                    help="If >0, subsample the train split.")
    ap.add_argument("--max-test",  type=int, default=0,
                    help="If >0, subsample the test split.")
    args = ap.parse_args()

    banner(f"Loading frozen encoder: {MODEL_NAME}")
    # Import here so the rest of the file can be parsed without
    # sentence-transformers installed.
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        sys.exit(f"ERROR: sentence-transformers not installed ({e}).\n"
                 "Install with:  python -m pip install sentence-transformers")

    t0 = time.time()
    model = SentenceTransformer(MODEL_NAME)
    print(f"  loaded in {time.time()-t0:.1f}s")

    targets = (args.dataset,) if args.dataset else ALL_DS
    rows: list[dict] = []
    for ds in targets:
        try:
            rows.append(run_one(ds, model,
                                max_train=args.max_train,
                                max_test=args.max_test))
        except Exception as exc:
            print(f"  ERROR on {ds}: {exc}")

    if not rows:
        print("  no rows produced; aborting.")
        return

    new_df = pd.DataFrame(rows)
    out_csv = TBL_DIR / "transformer_results.csv"
    if out_csv.exists():
        prev = load_csv(out_csv)
        # remove rows for datasets we just re-ran
        prev = prev[~prev["dataset"].isin(new_df["dataset"].tolist())]
        df = pd.concat([prev, new_df], ignore_index=True)
    else:
        df = new_df
    df = df.sort_values("dataset").reset_index(drop=True)
    save_csv(df, out_csv)
    print(f"\n  wrote {out_csv.relative_to(RESULTS_DIR.parent)}")

    out_md = TBL_DIR / "transformer_results.md"
    md = ["# Transformer baseline (frozen multilingual MiniLM + LR head)\n",
          f"Encoder: `{MODEL_NAME}` (no fine-tuning).\n",
          df.to_markdown(index=False), ""]
    out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"  wrote {out_md.relative_to(RESULTS_DIR.parent)}")


if __name__ == "__main__":
    main()
