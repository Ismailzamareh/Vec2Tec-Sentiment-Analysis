"""
Step 2 - Preprocessing.

For each dataset:
    * load raw text
    * clean and normalise (Arabic-specific rules for AR datasets)
    * build train/test splits  -- official splits used when available
    * save as CSV inside data/processed/

Output files (UTF-8, comma-separated, columns = text,label):
    data/processed/imdb_train.csv
    data/processed/imdb_test.csv
    data/processed/labr_train.csv
    data/processed/labr_test.csv
    data/processed/astd_train.csv
    data/processed/astd_test.csv

Usage:
    python src/preprocess.py
    python src/preprocess.py --full          # disables sub-sampling
"""
from __future__ import annotations

import argparse
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from config import (
    RAW_EN_DIR, RAW_AR_DIR, PROCESSED_DIR, RESULTS_DIR, RANDOM_SEED,
    SAMPLE_IMDB_PER_CLASS, SAMPLE_LABR_TRAIN_PER_CLASS,
    SAMPLE_LABR_TEST_PER_CLASS, USE_SAMPLING,
)
from utils import banner, save_csv


# ---------------------------------------------------------------------------
# Text cleaning - English
# ---------------------------------------------------------------------------
_HTML_BR  = re.compile(r"<br\s*/?>", re.IGNORECASE)
_HTML_TAG = re.compile(r"<[^>]+>")
_EN_KEEP  = re.compile(r"[^a-z0-9\s'!?\.,]")
_MULTI_WS = re.compile(r"\s+")


def clean_en(text: str) -> str:
    text = _HTML_BR.sub(" ", text)
    text = _HTML_TAG.sub(" ", text)
    text = text.lower()
    text = _EN_KEEP.sub(" ", text)
    text = _MULTI_WS.sub(" ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Text cleaning - Arabic
# ---------------------------------------------------------------------------
# Arabic diacritics (Tashkeel) and small marks
_AR_DIACRITICS = re.compile(r"[ً-ٰٟۖ-ۭ]")
# Tatweel (kashida)
_AR_TATWEEL = re.compile(r"ـ")
# URLs, mentions, hashtags
_URL        = re.compile(r"https?://\S+|www\.\S+")
_MENTION    = re.compile(r"@\w+")
_HASHTAG    = re.compile(r"#")
# Arabic-specific punctuation (keep separated)
_AR_PUNCT   = re.compile(r"[،؛؟!]")
# Anything that is not Arabic letters, Latin letters, digits, or whitespace
_AR_NON_TXT = re.compile(r"[^؀-ۿa-zA-Z0-9\s]")


def normalize_arabic_letters(text: str) -> str:
    """Standard letter normalisation: Alef, Yaa, Taa-marbuta."""
    text = _AR_DIACRITICS.sub("", text)
    text = _AR_TATWEEL.sub("", text)
    text = re.sub(r"[إأآا]", "ا", text)
    text = text.replace("ى", "ي")
    text = text.replace("ة", "ه")
    return text


def clean_ar(text: str) -> str:
    text = _URL.sub(" ", text)
    text = _MENTION.sub(" ", text)
    text = _HASHTAG.sub(" ", text)
    text = _AR_PUNCT.sub(" ", text)   # handled as whitespace
    text = normalize_arabic_letters(text)
    text = _AR_NON_TXT.sub(" ", text)
    text = _MULTI_WS.sub(" ", text).strip()
    return text


# ---------------------------------------------------------------------------
# IMDb loader (uses official train/ and test/ directory split)
# ---------------------------------------------------------------------------
def _list_txt(d: Path) -> List[str]:
    return sorted(f for f in os.listdir(d) if f.endswith(".txt"))


def load_imdb(use_sampling: bool, per_class: int = SAMPLE_IMDB_PER_CLASS):
    rng = random.Random(RANDOM_SEED)
    rows_train, rows_test = [], []
    for split, sink in (("train", rows_train), ("test", rows_test)):
        for lbl_name, lbl_id in (("pos", 1), ("neg", 0)):
            d = RAW_EN_DIR / split / lbl_name
            files = _list_txt(d)
            if use_sampling:
                rng.shuffle(files)
                files = files[:per_class]
            for fn in files:
                with open(d / fn, "r", encoding="utf-8", errors="ignore") as f:
                    txt = f.read()
                sink.append(dict(text=clean_en(txt), label=lbl_id))
    return pd.DataFrame(rows_train), pd.DataFrame(rows_test)


# ---------------------------------------------------------------------------
# LABR loader -- uses official 2class-balanced split files (0-indexed)
# ---------------------------------------------------------------------------
def _read_ids(path: Path) -> List[int]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s.isdigit():
                out.append(int(s))
    return out


def load_labr(use_sampling: bool,
              train_per_class: int = SAMPLE_LABR_TRAIN_PER_CLASS,
              test_per_class: int = SAMPLE_LABR_TEST_PER_CLASS):
    # 0-indexed line numbers into reviews.tsv (verified empirically)
    train_ids = _read_ids(RAW_AR_DIR / "2class-balanced-train.txt")
    test_ids  = _read_ids(RAW_AR_DIR / "2class-balanced-test.txt")

    with open(RAW_AR_DIR / "reviews.tsv", "r", encoding="utf-8") as f:
        lines = f.readlines()

    def build(id_list):
        rng = random.Random(RANDOM_SEED)
        rng.shuffle(id_list)
        pos, neg = [], []
        for idx in id_list:
            if idx < 0 or idx >= len(lines):
                continue
            parts = lines[idx].rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            try:
                rating = int(parts[0])
            except ValueError:
                continue
            if rating in (1, 2):
                lab, sink = 0, neg
            elif rating in (4, 5):
                lab, sink = 1, pos
            else:
                continue
            text = clean_ar(parts[4])
            if len(text.split()) >= 3:
                sink.append(dict(text=text, label=lab))
        return pos, neg

    pos_tr, neg_tr = build(train_ids)
    pos_te, neg_te = build(test_ids)

    if use_sampling:
        pos_tr = pos_tr[:train_per_class]
        neg_tr = neg_tr[:train_per_class]
        pos_te = pos_te[:test_per_class]
        neg_te = neg_te[:test_per_class]

    train = pd.DataFrame(pos_tr + neg_tr).sample(
        frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    test = pd.DataFrame(pos_te + neg_te).sample(
        frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    return train, test


# ---------------------------------------------------------------------------
# ASTD loader -- balanced binary split (80/20) from Tweets.txt
# (the 2class-balanced-*.txt files reference an external ID space that is
#  not redistributable with Tweets.txt, so we build a stratified split.)
# ---------------------------------------------------------------------------
def load_astd(test_size: float = 0.2):
    rows = []
    with open(RAW_AR_DIR / "Tweets.txt", "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            text_raw, lab_raw = parts[0], parts[-1].strip().upper()
            if lab_raw == "POS":
                lab = 1
            elif lab_raw == "NEG":
                lab = 0
            else:
                continue                          # drop NEU and OBJ
            text = clean_ar(text_raw)
            if len(text.split()) >= 2:
                rows.append(dict(text=text, label=lab))

    df = pd.DataFrame(rows)
    # stratified shuffle
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(
        df, test_size=test_size, random_state=RANDOM_SEED, stratify=df["label"])
    train = train.reset_index(drop=True)
    test  = test.reset_index(drop=True)
    return train, test


# ---------------------------------------------------------------------------
# ASTD 4-class loader -- OFFICIAL splits from 4class-balanced-*.txt
# Each split file contains 0-based line numbers into Tweets.txt.
# Verified empirically in src/explore_astd4.py.
# ---------------------------------------------------------------------------
ASTD4_LABEL_MAP = {"NEG": 0, "POS": 1, "NEUTRAL": 2, "OBJ": 3}

ASTD4_EXPECTED_SIZES = {"train": 1924, "validation": 636, "test": 636}
ASTD4_EXPECTED_DIST = {
    "train":      {0: 481, 1: 481, 2: 481, 3: 481},
    "validation": {0: 159, 1: 159, 2: 159, 3: 159},
    "test":       {0: 159, 1: 159, 2: 159, 3: 159},
}


def _read_int_ids(path: Path) -> List[int]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s.lstrip("-").isdigit():
                out.append(int(s))
    return out


def _read_tweets_table() -> List[tuple[str, str]]:
    """Return Tweets.txt as a list of (raw_text, raw_label) by line index."""
    rows: List[tuple[str, str]] = []
    with open(RAW_AR_DIR / "Tweets.txt", "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n").rstrip("\r")
            if not line:
                rows.append(("", ""))
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                rows.append(("\t".join(parts[:-1]), parts[-1].strip().upper()))
            else:
                rows.append((line, ""))
    return rows


def load_astd_4class():
    """Build (train, validation, test) DataFrames from the official ASTD
    4class-balanced split files, with strict integrity assertions.

    Returns
    -------
    (train_df, validation_df, test_df, audit) where audit is a dict
    describing what was read (used to write the dataset card).
    """
    files = {
        "train":      RAW_AR_DIR / "4class-balanced-train.txt",
        "validation": RAW_AR_DIR / "4class-balanced-validation.txt",
        "test":       RAW_AR_DIR / "4class-balanced-test.txt",
    }
    for k, p in files.items():
        if not p.exists():
            sys.exit(f"ERROR: ASTD4 file missing: {p}")

    tweets = _read_tweets_table()
    n_tweets = len(tweets)

    audit = dict(
        source_files={k: str(v) for k, v in files.items()},
        tweets_file=str(RAW_AR_DIR / "Tweets.txt"),
        tweets_lines=n_tweets,
        indexing="0-based (verified via explore_astd4.py)",
        label_map=ASTD4_LABEL_MAP,
        ids_failed_to_map={"train": 0, "validation": 0, "test": 0},
        sizes={},
        class_dist={},
    )

    # --- (c) leakage check across splits ---
    id_sets: dict[str, set[int]] = {}
    for split, path in files.items():
        id_sets[split] = set(_read_int_ids(path))
    inter_tv = id_sets["train"]      & id_sets["validation"]
    inter_tt = id_sets["train"]      & id_sets["test"]
    inter_vt = id_sets["validation"] & id_sets["test"]
    if inter_tv or inter_tt or inter_vt:
        sys.exit(
            "ERROR: ID leakage detected between ASTD4 splits\n"
            f"  train ∩ validation = {len(inter_tv)} ids\n"
            f"  train ∩ test       = {len(inter_tt)} ids\n"
            f"  validation ∩ test  = {len(inter_vt)} ids"
        )

    frames: dict[str, pd.DataFrame] = {}
    for split, path in files.items():
        ids = _read_int_ids(path)
        rows = []
        n_failed = 0
        for idx in ids:
            if not (0 <= idx < n_tweets):
                n_failed += 1
                continue
            text_raw, lab_raw = tweets[idx]
            if lab_raw not in ASTD4_LABEL_MAP:
                n_failed += 1
                continue
            text = clean_ar(text_raw)
            rows.append(dict(text=text, label=ASTD4_LABEL_MAP[lab_raw]))
        df = pd.DataFrame(rows)

        # --- (a) exact size check ---
        expected_n = ASTD4_EXPECTED_SIZES[split]
        if len(df) != expected_n:
            sys.exit(
                f"ERROR: ASTD4 {split} size mismatch:\n"
                f"  expected {expected_n}, got {len(df)} "
                f"(unmapped IDs={n_failed})"
            )

        # --- (b) perfect class balance check ---
        got_dist = {int(k): int(v) for k, v in
                    df["label"].value_counts().items()}
        # canonicalise order for printing
        got_dist_sorted = {k: got_dist.get(k, 0) for k in (0, 1, 2, 3)}
        if got_dist_sorted != ASTD4_EXPECTED_DIST[split]:
            sys.exit(
                f"ERROR: ASTD4 {split} class balance mismatch:\n"
                f"  expected {ASTD4_EXPECTED_DIST[split]}\n"
                f"  got      {got_dist_sorted}"
            )

        audit["ids_failed_to_map"][split] = n_failed
        audit["sizes"][split] = len(df)
        audit["class_dist"][split] = got_dist_sorted

        # shuffle (deterministic) so downstream classifiers see mixed-class order
        df = df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
        frames[split] = df

    return frames["train"], frames["validation"], frames["test"], audit


def _write_astd4_dataset_card(audit: dict) -> None:
    out = RESULTS_DIR / "astd4_dataset_card.md"
    total_used = sum(audit["sizes"].values())
    total_failed = sum(audit["ids_failed_to_map"].values())

    lines = [
        "# ASTD 4-class dataset card",
        "",
        "## Source files",
        f"- Tweets.txt (reference corpus) — `{audit['tweets_file']}`, "
        f"{audit['tweets_lines']:,} lines",
    ]
    for split, p in audit["source_files"].items():
        lines.append(f"- {split} ID list — `{p}`")
    lines += [
        "",
        "## Indexing decision",
        f"- {audit['indexing']}.",
        "- Each line in a `4class-balanced-*.txt` file is treated as a "
        "0-based line number into `Tweets.txt`. The 1-based reading was "
        "ruled out empirically in `src/explore_astd4.py`: only 0-based "
        "produced the documented `{POS, NEG, NEUTRAL, OBJ}` × 481 / 159 "
        "balance.",
        "",
        "## Label mapping",
        "| raw label | numeric id |",
        "|:----------|-----------:|",
    ]
    for k, v in audit["label_map"].items():
        lines.append(f"| {k} | {v} |")

    lines += [
        "",
        "## Final split sizes",
        "| split | size | class distribution |",
        "|:------|-----:|:-------------------|",
    ]
    for split in ("train", "validation", "test"):
        lines.append(
            f"| {split} | {audit['sizes'][split]} | "
            f"{audit['class_dist'][split]} |"
        )

    lines += [
        "",
        "## Coverage and balancing",
        f"- Total tweets used across the three splits: **{total_used:,}** "
        f"out of {audit['tweets_lines']:,} available in Tweets.txt.",
        "- Balancing rationale: the official splits are downsampled per class "
        "so that all four classes are equally represented in every split. "
        "The minority class in the raw corpus is **POS = 799**, which is the "
        "binding constraint on the size of the balanced design "
        "(train + val + test ≈ 4 × 799 minus held-out reserve).",
        f"- IDs that failed to map to a labelled tweet: **{total_failed}** "
        f"({audit['ids_failed_to_map']}). Zero is expected and required by "
        "the runtime assertions in `load_astd_4class`.",
        "",
        "## Provenance of this file",
        "Written automatically by `src/preprocess.py --astd4`. Re-run that "
        "command to regenerate.",
    ]

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  wrote {out.relative_to(RESULTS_DIR.parent)}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def _summary(name: str, train: pd.DataFrame, test: pd.DataFrame) -> None:
    def dist(df):
        return Counter(df["label"].tolist())
    print(f"  {name}: train={len(train)} | test={len(test)} | "
          f"train_dist={dict(dist(train))} | test_dist={dict(dist(test))}")


def main(use_sampling: bool, run_astd4: bool, only_astd4: bool) -> None:
    banner(f"Preprocessing (use_sampling={use_sampling}, "
           f"astd4={run_astd4}, only_astd4={only_astd4})")

    if not only_astd4:
        print("[1/3] IMDb")
        t = time.time()
        tr, te = load_imdb(use_sampling)
        save_csv(tr, PROCESSED_DIR / "imdb_train.csv")
        save_csv(te, PROCESSED_DIR / "imdb_test.csv")
        print(f"  saved imdb_train.csv ({len(tr)}) + imdb_test.csv ({len(te)}) "
              f"in {time.time()-t:.1f}s")
        _summary("imdb", tr, te)

        print("[2/3] LABR (official 2class-balanced splits)")
        t = time.time()
        tr, te = load_labr(use_sampling)
        save_csv(tr, PROCESSED_DIR / "labr_train.csv")
        save_csv(te, PROCESSED_DIR / "labr_test.csv")
        print(f"  saved labr_train.csv ({len(tr)}) + labr_test.csv ({len(te)}) "
              f"in {time.time()-t:.1f}s")
        _summary("labr", tr, te)

        print("[3/3] ASTD (stratified 80/20 of POS+NEG)")
        t = time.time()
        tr, te = load_astd()
        save_csv(tr, PROCESSED_DIR / "astd_train.csv")
        save_csv(te, PROCESSED_DIR / "astd_test.csv")
        print(f"  saved astd_train.csv ({len(tr)}) + astd_test.csv ({len(te)}) "
              f"in {time.time()-t:.1f}s")
        _summary("astd", tr, te)

    if run_astd4:
        print("[+] ASTD 4-class (official 4class-balanced splits)")
        t = time.time()
        tr, va, te, audit = load_astd_4class()
        save_csv(tr, PROCESSED_DIR / "astd4_train.csv")
        save_csv(va, PROCESSED_DIR / "astd4_validation.csv")
        save_csv(te, PROCESSED_DIR / "astd4_test.csv")
        print(f"  saved astd4_train.csv ({len(tr)}) + "
              f"astd4_validation.csv ({len(va)}) + "
              f"astd4_test.csv ({len(te)}) in {time.time()-t:.1f}s")
        print(f"  astd4 train_dist = {audit['class_dist']['train']}")
        print(f"  astd4 val_dist   = {audit['class_dist']['validation']}")
        print(f"  astd4 test_dist  = {audit['class_dist']['test']}")
        _write_astd4_dataset_card(audit)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--full", action="store_true",
                   help="Disable sub-sampling and process all available data.")
    p.add_argument("--astd4", action="store_true",
                   help="Also build the ASTD 4-class official splits.")
    p.add_argument("--only-astd4", action="store_true",
                   help="Skip the binary datasets and only build ASTD 4-class.")
    args = p.parse_args()
    main(
        use_sampling=(not args.full) and USE_SAMPLING,
        run_astd4=(args.astd4 or args.only_astd4),
        only_astd4=args.only_astd4,
    )
