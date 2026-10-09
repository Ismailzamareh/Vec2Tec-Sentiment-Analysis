"""
Step 1 - Dataset validation.

Verifies that all required raw dataset files exist and have the expected
format, then prints sample counts, label set, and class distribution for
each dataset.

Usage:
    python src/validate_data.py
"""
from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

from config import (
    RAW_EN_DIR, RAW_AR_DIR, RESULTS_DIR, DATASETS,
)
from utils import banner, save_csv
import pandas as pd


# ---------------------------------------------------------------------------
# Required files per dataset
# ---------------------------------------------------------------------------
IMDB_REQUIRED = {
    "train/pos":      "directory",
    "train/neg":      "directory",
    "test/pos":       "directory",
    "test/neg":       "directory",
    "imdb.vocab":     "file",
    "README":         "file",
}

LABR_REQUIRED = {
    "reviews.tsv":               "file",
    "2class-balanced-train.txt": "file",
    "2class-balanced-test.txt":  "file",
}

ASTD_REQUIRED = {
    "Tweets.txt":                "file",
}

ASTD4_REQUIRED = {
    "Tweets.txt":                       "file",
    "4class-balanced-train.txt":        "file",
    "4class-balanced-validation.txt":   "file",
    "4class-balanced-test.txt":         "file",
}


# ---------------------------------------------------------------------------
# Per-dataset validators
# ---------------------------------------------------------------------------
def _check_paths(root: Path, expected: dict) -> list[dict]:
    findings = []
    for rel, kind in expected.items():
        p = root / rel
        ok = (p.is_dir() if kind == "directory" else p.is_file())
        findings.append(dict(
            path=str(p), kind=kind, exists=ok,
            size_bytes=(p.stat().st_size if ok and kind == "file" else (
                "-" if not ok else "<dir>")),
        ))
    return findings


def validate_imdb() -> dict:
    banner("IMDb (English)")
    findings = _check_paths(RAW_EN_DIR, IMDB_REQUIRED)
    missing = [f for f in findings if not f["exists"]]
    for f in findings:
        flag = "OK " if f["exists"] else "!! "
        print(f"  {flag}{f['path']}  ({f['size_bytes']})")
    if missing:
        print(f"  ERROR: {len(missing)} required file(s) missing.")
        return dict(dataset="imdb", status="MISSING_FILES",
                    findings=findings, missing=missing)

    # count train/test pos/neg
    counts = {}
    for split in ("train", "test"):
        for lbl in ("pos", "neg"):
            d = RAW_EN_DIR / split / lbl
            try:
                n = len([x for x in os.listdir(d) if x.endswith(".txt")])
            except OSError as e:
                return dict(dataset="imdb", status="OS_ERROR", error=str(e))
            counts[f"{split}_{lbl}"] = n
            print(f"  {split}/{lbl}: {n} reviews")

    total = sum(counts.values())
    print(f"  TOTAL: {total} labeled reviews")
    return dict(
        dataset="imdb", status="OK", total=total,
        counts=counts,
        labels=["pos", "neg"],
        format="text files; filename = '<id>_<rating>.txt'",
    )


def validate_labr() -> dict:
    banner("LABR (Arabic book reviews)")
    findings = _check_paths(RAW_AR_DIR, LABR_REQUIRED)
    missing = [f for f in findings if not f["exists"]]
    for f in findings:
        flag = "OK " if f["exists"] else "!! "
        print(f"  {flag}{f['path']}  ({f['size_bytes']})")
    if missing:
        print(f"  ERROR: {len(missing)} required file(s) missing.")
        return dict(dataset="labr", status="MISSING_FILES",
                    findings=findings, missing=missing)

    # check tsv format: 5 tab-separated columns
    bad_lines = 0
    rating_counter = Counter()
    with open(RAW_AR_DIR / "reviews.tsv", encoding="utf-8") as f:
        total_lines = 0
        for line in f:
            total_lines += 1
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                bad_lines += 1
                continue
            try:
                rating = int(parts[0])
                rating_counter[rating] += 1
            except ValueError:
                bad_lines += 1
    print(f"  reviews.tsv lines       : {total_lines}")
    print(f"  reviews.tsv malformed   : {bad_lines}")
    print(f"  rating distribution      : {dict(sorted(rating_counter.items()))}")

    # check split files
    for fn in ("2class-balanced-train.txt", "2class-balanced-test.txt"):
        with open(RAW_AR_DIR / fn, encoding="utf-8") as f:
            ids = [l.strip() for l in f if l.strip()]
        bad = sum(1 for x in ids if not x.isdigit())
        print(f"  {fn}: {len(ids)} entries (malformed={bad})")

    return dict(
        dataset="labr",
        status="OK" if bad_lines == 0 else "WARN_MALFORMED",
        total_reviews=total_lines,
        malformed=bad_lines,
        rating_distribution=dict(rating_counter),
        labels="rating in {1..5}  ->  binary: {1,2}=neg, {4,5}=pos, drop 3",
        format="reviews.tsv columns: rating\\tuser_id\\tbook_id\\treview_id\\ttext",
    )


def validate_astd() -> dict:
    banner("ASTD (Arabic Sentiment Tweets)")
    findings = _check_paths(RAW_AR_DIR, ASTD_REQUIRED)
    missing = [f for f in findings if not f["exists"]]
    for f in findings:
        flag = "OK " if f["exists"] else "!! "
        print(f"  {flag}{f['path']}  ({f['size_bytes']})")
    if missing:
        print(f"  ERROR: {len(missing)} required file(s) missing.")
        return dict(dataset="astd", status="MISSING_FILES",
                    findings=findings, missing=missing)

    label_counter = Counter()
    bad_lines = 0
    with open(RAW_AR_DIR / "Tweets.txt", encoding="utf-8") as f:
        total = 0
        for line in f:
            total += 1
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                bad_lines += 1
                continue
            label_counter[parts[-1].strip().upper()] += 1
    print(f"  Tweets.txt lines         : {total}")
    print(f"  Tweets.txt malformed     : {bad_lines}")
    print(f"  label distribution        : {dict(label_counter)}")
    return dict(
        dataset="astd",
        status="OK" if bad_lines == 0 else "WARN_MALFORMED",
        total_tweets=total,
        malformed=bad_lines,
        label_distribution=dict(label_counter),
        labels="POS / NEG / NEUTRAL / OBJ  ->  binary: keep POS+NEG only",
        format="Tweets.txt: text\\tLABEL",
    )


def validate_astd4() -> dict:
    """Validate the OFFICIAL ASTD 4-class split files.

    Checks:
      - all three split files exist
      - prints train/validation/test sizes
      - dereferences IDs against Tweets.txt and reports per-split class
        distribution
      - flags any class imbalance > 1.5x (max/min ratio across the four
        labels in a single split)
    """
    banner("ASTD 4-class (official balanced splits)")
    findings = _check_paths(RAW_AR_DIR, ASTD4_REQUIRED)
    missing = [f for f in findings if not f["exists"]]
    for f in findings:
        flag = "OK " if f["exists"] else "!! "
        print(f"  {flag}{f['path']}  ({f['size_bytes']})")
    if missing:
        print(f"  ERROR: {len(missing)} required file(s) missing.")
        return dict(dataset="astd4", status="MISSING_FILES",
                    findings=findings, missing=missing)

    # Pre-read Tweets.txt once and index by line (0-based).
    tweets_labels: list[str] = []
    with open(RAW_AR_DIR / "Tweets.txt", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").rstrip("\r").split("\t")
            tweets_labels.append(parts[-1].strip().upper() if len(parts) >= 2 else "")

    splits = {
        "train":      "4class-balanced-train.txt",
        "validation": "4class-balanced-validation.txt",
        "test":       "4class-balanced-test.txt",
    }

    per_split: dict[str, dict] = {}
    worst_ratio = 0.0
    imbalance_flags: list[str] = []

    for split, fname in splits.items():
        with open(RAW_AR_DIR / fname, encoding="utf-8") as f:
            ids = [int(l.strip()) for l in f if l.strip().lstrip("-").isdigit()]

        oob = sum(1 for i in ids if not (0 <= i < len(tweets_labels)))
        dist = Counter(tweets_labels[i] for i in ids
                       if 0 <= i < len(tweets_labels))
        n = sum(dist.values())
        # max/min ratio across labels
        if dist:
            mn = min(dist.values())
            mx = max(dist.values())
            ratio = mx / max(mn, 1)
        else:
            ratio = float("nan")
        worst_ratio = max(worst_ratio, ratio)

        flag = ratio > 1.5
        flag_str = " (IMBALANCE>1.5x)" if flag else ""
        if flag:
            imbalance_flags.append(f"{split}: ratio={ratio:.2f}")

        print(f"  {split:11s} size={n:5d}  oob={oob}  dist={dict(dist)} "
              f"max/min={ratio:.2f}{flag_str}")
        per_split[split] = dict(size=n, oob_ids=oob,
                                distribution=dict(dist), max_min_ratio=ratio)

    status = ("OK"
              if not imbalance_flags
              and all(s["oob_ids"] == 0 for s in per_split.values())
              else "WARN_IMBALANCED")

    return dict(
        dataset="astd4",
        status=status,
        per_split=per_split,
        worst_max_min_ratio=round(worst_ratio, 4),
        imbalance_flags=imbalance_flags,
        labels="POS / NEG / NEUTRAL / OBJ  ->  0/1/2/3 (4-class)",
        format="4class-balanced-{train,validation,test}.txt: one 0-based "
               "line index per row into Tweets.txt",
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> int:
    report = {
        "imdb":  validate_imdb(),
        "labr":  validate_labr(),
        "astd":  validate_astd(),
        "astd4": validate_astd4(),
    }
    # Persist a flat summary as CSV
    flat = []
    for ds, info in report.items():
        row = {"dataset": ds, "status": info.get("status", "UNKNOWN")}
        for k, v in info.items():
            if k in ("dataset", "status"):
                continue
            row[k] = str(v)
        flat.append(row)
    save_csv(pd.DataFrame(flat), RESULTS_DIR / "dataset_validation.csv")
    banner("Validation summary written to results/dataset_validation.csv", char="-")

    n_bad = sum(1 for v in report.values() if v["status"] not in ("OK",))
    return 0 if n_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
