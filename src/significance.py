"""
Step 10 - McNemar significance tests on classifier predictions.

For each dataset, compares two pairs of classifiers on the *same* test set:
    1. vec2tec_lr  vs  word2vec_lr
    2. vec2tec_lr  vs  tfidf_lr

For each pair we build the disagreement contingency table
(per-example correctness):

                     model_B correct        model_B wrong
    model_A correct        a                       b
    model_A wrong          c                       d

McNemar's test looks at b vs c.  When (b + c) is small (< 25) we use
the exact binomial variant; otherwise the chi-square approximation with
continuity correction.

Predictions are read from the cached prediction pickles that the pipeline
wrote in train_word2vec.py / vec2tec_module.py / baselines.py:
    results/_preds_{tfidf,word2vec,vec2tec}_<ds>.pkl
Those pickles store the exact predictions produced by
models/clf_<model>_<ds>.pkl on the held-out test split, so re-loading them
is functionally identical to calling clf.predict on the test vectors.

Writes:
    results/significance_vec2tec_vs_w2v.csv     (paper-spec name)
    results/significance.csv                    (alias for backwards compat)

CSV columns:
    dataset, model_a, model_b,
    n_only_a_correct, n_only_b_correct,
    n_both_correct, n_both_wrong,
    test, statistic, p_value, significant_5pct

Usage:
    python src/significance.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

from config import RESULTS_DIR, DATASETS
from utils import banner, load_pickle, save_csv


# (group_prefix, model_key) -> identifies which _preds_*.pkl block to read
COMPARISONS = (
    # (model_a, group_a, model_b, group_b)
    ("vec2tec_lr", "vec2tec", "word2vec_lr", "word2vec"),
    ("vec2tec_lr", "vec2tec", "tfidf_lr",    "tfidf"),
)


def _load_preds(group: str, ds: str, model_key: str) -> tuple[np.ndarray, np.ndarray]:
    p = RESULTS_DIR / f"_preds_{group}_{ds}.pkl"
    if not p.exists():
        raise FileNotFoundError(f"missing prediction file: {p}")
    block = load_pickle(p)
    if model_key not in block:
        raise KeyError(f"{model_key!r} not in {p.name} (have: {list(block)})")
    payload = block[model_key]
    return (np.asarray(payload["y_true"]),
            np.asarray(payload["predictions"]))


def _mcnemar_row(ds: str, model_a: str, group_a: str,
                 model_b: str, group_b: str) -> dict:
    y_true_a, y_pred_a = _load_preds(group_a, ds, model_a)
    y_true_b, y_pred_b = _load_preds(group_b, ds, model_b)

    if not np.array_equal(y_true_a, y_true_b):
        raise RuntimeError(f"[{ds}] y_true mismatch between {model_a} and {model_b}")

    correct_a = (y_pred_a == y_true_a)
    correct_b = (y_pred_b == y_true_b)

    a = int(np.sum( correct_a &  correct_b))   # both correct
    b = int(np.sum( correct_a & ~correct_b))   # only A correct
    c = int(np.sum(~correct_a &  correct_b))   # only B correct
    d = int(np.sum(~correct_a & ~correct_b))   # both wrong

    table = [[a, b], [c, d]]
    use_exact = (b + c) < 25
    res = mcnemar(table, exact=use_exact, correction=not use_exact)
    stat = float(res.statistic) if res.statistic is not None else float("nan")
    pval = float(res.pvalue)

    return dict(
        dataset=ds,
        model_a=model_a,
        model_b=model_b,
        n_only_a_correct=b,
        n_only_b_correct=c,
        n_both_correct=a,
        n_both_wrong=d,
        test=("exact" if use_exact else "chi2_with_continuity"),
        statistic=round(stat, 6),
        p_value=round(pval, 6),
        significant_5pct=bool(pval < 0.05),
    )


def main() -> None:
    banner("McNemar significance tests")

    rows: list[dict] = []
    for ds in DATASETS:
        for (m_a, g_a, m_b, g_b) in COMPARISONS:
            row = _mcnemar_row(ds, m_a, g_a, m_b, g_b)
            rows.append(row)
            print(f"  {ds:5s} | {m_a:11s} vs {m_b:11s} "
                  f"| onlyA={row['n_only_a_correct']:5d} "
                  f"onlyB={row['n_only_b_correct']:5d} "
                  f"| stat={row['statistic']:7.4f} "
                  f"p={row['p_value']:.4g} "
                  f"sig@5%={row['significant_5pct']}")

    df = pd.DataFrame(rows)
    primary = RESULTS_DIR / "significance_vec2tec_vs_w2v.csv"
    save_csv(df, primary)
    alias   = RESULTS_DIR / "significance.csv"
    save_csv(df, alias)
    banner(f"wrote {primary.name} (+ alias significance.csv)", char="-")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
