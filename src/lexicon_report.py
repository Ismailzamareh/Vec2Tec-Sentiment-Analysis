"""
Step 11 - Lexicon qualitative analysis.

For each dataset:
    1. Loads results/lex_<ds>.json (term -> polarity score).
    2. Picks the top-20 most positive and top-20 most negative terms.
    3. Saves results/top_polarity_<ds>.csv with columns:
         rank, polarity ('positive'|'negative'), term, score
    4. Prints both halves to stdout for human inspection.

Usage:
    python src/lexicon_report.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from config import RESULTS_DIR, DATASETS
from utils import banner, save_csv


TOP_K = 20


def _load_lex(ds: str) -> dict[str, float]:
    p = RESULTS_DIR / f"lex_{ds}.json"
    if not p.exists():
        raise FileNotFoundError(f"missing {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _rank_table(lex: dict[str, float]) -> pd.DataFrame:
    items = sorted(lex.items(), key=lambda kv: kv[1])
    negatives = items[:TOP_K]                  # most negative scores first
    positives = list(reversed(items[-TOP_K:])) # most positive scores first

    rows: list[dict] = []
    for rank, (term, score) in enumerate(positives, start=1):
        rows.append(dict(rank=rank, polarity="positive",
                         term=term, score=round(float(score), 6)))
    for rank, (term, score) in enumerate(negatives, start=1):
        rows.append(dict(rank=rank, polarity="negative",
                         term=term, score=round(float(score), 6)))
    return pd.DataFrame(rows)


def main() -> None:
    banner("Lexicon report - top polarity terms")

    for ds in DATASETS:
        try:
            lex = _load_lex(ds)
        except FileNotFoundError as e:
            print(f"  skipping {ds}: {e}", file=sys.stderr)
            continue

        df = _rank_table(lex)
        out = RESULTS_DIR / f"top_polarity_{ds}.csv"
        save_csv(df, out)

        banner(f"{ds.upper()} (lexicon size = {len(lex)})", char="-")
        pos = df[df["polarity"] == "positive"]
        neg = df[df["polarity"] == "negative"]
        print("  TOP-20 POSITIVE:")
        for _, r in pos.iterrows():
            print(f"    {int(r['rank']):2d}. {r['term']:30s} {r['score']:+.4f}")
        print("  TOP-20 NEGATIVE:")
        for _, r in neg.iterrows():
            print(f"    {int(r['rank']):2d}. {r['term']:30s} {r['score']:+.4f}")
        print(f"  wrote {out.name}")


if __name__ == "__main__":
    main()
