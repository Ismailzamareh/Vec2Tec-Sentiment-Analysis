"""
Step 14 - Lexicon interpretability table.

Reads results/lex_<ds>.json for each dataset and writes a single combined
CSV listing the top-20 positive and top-20 negative words side-by-side
per dataset.

Writes:
    results/tables/top_polarity_words.csv     (long table, all 3 datasets)
    results/tables/top_polarity_words.md      (markdown preview)

Columns:
    dataset, rank, positive_word, positive_score, negative_word, negative_score

Usage:
    python src/make_lexicon_table.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from config import RESULTS_DIR, DATASETS
from utils import banner, save_csv


TBL_DIR = RESULTS_DIR / "tables"
TBL_DIR.mkdir(parents=True, exist_ok=True)

TOP_K = 20


def _load_lex(ds: str) -> dict[str, float]:
    p = RESULTS_DIR / f"lex_{ds}.json"
    if not p.exists():
        raise FileNotFoundError(p)
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _top_polarity(lex: dict[str, float], k: int = TOP_K) -> list[tuple]:
    items = sorted(lex.items(), key=lambda kv: kv[1])
    negatives = items[:k]                  # most negative first (lowest score)
    positives = list(reversed(items[-k:])) # most positive first (highest score)
    n = min(len(negatives), len(positives), k)
    return [(positives[i][0], float(positives[i][1]),
             negatives[i][0], float(negatives[i][1])) for i in range(n)]


def main() -> None:
    banner("Lexicon interpretability table")

    rows: list[dict] = []
    for ds in DATASETS:
        try:
            lex = _load_lex(ds)
        except FileNotFoundError:
            print(f"  skip {ds}: lex file missing", file=sys.stderr)
            continue
        for rank, (pw, ps, nw, ns) in enumerate(_top_polarity(lex), start=1):
            rows.append(dict(
                dataset=ds,
                rank=rank,
                positive_word=pw,
                positive_score=round(ps, 6),
                negative_word=nw,
                negative_score=round(ns, 6),
            ))
        print(f"  {ds}: lexicon size = {len(lex)}, kept top {TOP_K} of each polarity")

    df = pd.DataFrame(rows)
    out_csv = TBL_DIR / "top_polarity_words.csv"
    save_csv(df, out_csv)
    print(f"  wrote {out_csv.relative_to(RESULTS_DIR.parent)}")

    md_blocks = ["# Top polarity words per dataset (corpus-derived lexicon)\n"]
    for ds in DATASETS:
        sub = df[df["dataset"] == ds]
        if sub.empty:
            continue
        md_blocks.append(f"\n## {ds.upper()}\n")
        md_blocks.append(sub.to_markdown(index=False))
    out_md = TBL_DIR / "top_polarity_words.md"
    out_md.write_text("\n".join(md_blocks) + "\n", encoding="utf-8")
    print(f"  wrote {out_md.relative_to(RESULTS_DIR.parent)}")


if __name__ == "__main__":
    main()
