"""
Build an extended dataset summary table covering every binary and 4-class
processed dataset on disk.

For each dataset we compute:
    - language, text type, number of classes
    - train / validation / test sizes
    - vocabulary size (whitespace tokens) from TRAIN
    - mean / median document length in tokens (across train+test)
    - class distribution (train)
    - imbalance ratio (max/min class count)

Inputs:
    data/processed/<ds>_train.csv, _test.csv, _validation.csv (if any)

Outputs:
    results/tables/dataset_stats_v2.csv
    results/tables/dataset_stats_v2.md

Usage:
    python src/dataset_stats_v2.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import PROCESSED_DIR, RESULTS_DIR
from utils import banner, load_csv, save_csv


TBL_DIR = RESULTS_DIR / "tables"
TBL_DIR.mkdir(parents=True, exist_ok=True)


DATASETS = (
    ("imdb",  "English", "long review"),
    ("labr",  "Arabic",  "long review"),
    ("astd",  "Arabic",  "short tweet"),
    ("astd4", "Arabic",  "short tweet"),
)


def _doc_lengths(df: pd.DataFrame) -> np.ndarray:
    return df["text"].fillna("").astype(str).str.split().str.len().to_numpy()


def stats_for(ds: str, language: str, text_type: str) -> dict | None:
    tr_p = PROCESSED_DIR / f"{ds}_train.csv"
    te_p = PROCESSED_DIR / f"{ds}_test.csv"
    va_p = PROCESSED_DIR / f"{ds}_validation.csv"
    if not (tr_p.exists() and te_p.exists()):
        return None

    tr = load_csv(tr_p)
    te = load_csv(te_p)
    va = load_csv(va_p) if va_p.exists() else None

    train_text = tr["text"].fillna("").astype(str)
    vocab: set[str] = set()
    for line in train_text:
        vocab.update(line.split())

    lens = np.concatenate([_doc_lengths(tr), _doc_lengths(te)])
    cls_tr = tr["label"].value_counts().sort_index().to_dict()
    cls_tr = {int(k): int(v) for k, v in cls_tr.items()}
    n_classes = len(cls_tr)
    counts = list(cls_tr.values())
    imbalance = round(max(counts) / max(min(counts), 1), 3) if counts else float("nan")

    return dict(
        dataset=ds.upper(),
        language=language,
        text_type=text_type,
        n_classes=n_classes,
        train_size=len(tr),
        validation_size=(len(va) if va is not None else None),
        test_size=len(te),
        vocab_size_train=len(vocab),
        avg_doc_len_tokens=round(float(lens.mean()), 1),
        median_doc_len_tokens=int(np.median(lens)),
        max_doc_len_tokens=int(lens.max()),
        class_distribution_train=str(cls_tr),
        imbalance_ratio_train=imbalance,
    )


def main() -> None:
    banner("Building dataset stats v2")
    rows = []
    for ds, lang, ttype in DATASETS:
        r = stats_for(ds, lang, ttype)
        if r is None:
            print(f"  skip {ds}: processed CSV(s) missing")
            continue
        rows.append(r)
        print(f"  {ds:6s} train={r['train_size']:>5d} "
              f"val={r['validation_size']} test={r['test_size']} "
              f"vocab={r['vocab_size_train']} "
              f"avg_len={r['avg_doc_len_tokens']} "
              f"median_len={r['median_doc_len_tokens']}")

    df = pd.DataFrame(rows)
    out_csv = TBL_DIR / "dataset_stats_v2.csv"
    save_csv(df, out_csv)
    print(f"  wrote {out_csv.relative_to(RESULTS_DIR.parent)}")

    md = ["# Dataset statistics (extended)\n",
          df.to_markdown(index=False), ""]
    out_md = TBL_DIR / "dataset_stats_v2.md"
    out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"  wrote {out_md.relative_to(RESULTS_DIR.parent)}")


if __name__ == "__main__":
    main()
