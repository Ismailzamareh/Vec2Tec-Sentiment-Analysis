"""
Read-only exploration of the official ASTD 4-class split files.

Determines the file format of:
    ../AR Data/4class-balanced-train.txt
    ../AR Data/4class-balanced-validation.txt
    ../AR Data/4class-balanced-test.txt

For each file:
    - prints first 5 raw lines
    - infers schema: integer-ID / 'id\tlabel' / 'text\tlabel'
    - if integer IDs, dereferences against Tweets.txt and shows the
      resulting (text, label) for the first 5 entries
    - prints total entries and class distribution (when reachable)
    - prints min/max ID value (when the file is integer-IDs)

This script writes nothing; it just prints to stdout.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from config import RAW_AR_DIR


FILES = {
    "train":      "4class-balanced-train.txt",
    "validation": "4class-balanced-validation.txt",
    "test":       "4class-balanced-test.txt",
}
TWEETS_PATH = RAW_AR_DIR / "Tweets.txt"


def _read_lines(path: Path) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.rstrip("\n").rstrip("\r") for ln in f]


def _infer_schema(lines: list[str]) -> str:
    """Return one of: 'int_id', 'id_label', 'text_label', 'unknown'."""
    sample = [ln for ln in lines if ln.strip()][:50]
    if not sample:
        return "unknown"

    # All entirely numeric -> integer-only IDs
    if all(ln.strip().lstrip("-").isdigit() for ln in sample):
        return "int_id"

    # Tab-separated 'id<TAB>label' (id is digits, label is one of the 4 names)
    label_vocab = {"NEG", "POS", "NEUTRAL", "OBJ"}
    tab_split = [ln.split("\t") for ln in sample]
    if all(len(parts) == 2 and parts[0].strip().isdigit()
           and parts[1].strip().upper() in label_vocab for parts in tab_split):
        return "id_label"

    # text<TAB>label  (label is one of the 4 known names, text is non-numeric)
    if all(len(parts) >= 2 and parts[-1].strip().upper() in label_vocab
           for parts in tab_split):
        return "text_label"

    return "unknown"


def _load_tweets(path: Path) -> list[tuple[str, str]]:
    """Read Tweets.txt as a list of (text, label) by line.

    The file is tab-separated; we keep the LAST column as label and
    everything before it as text.
    """
    out: list[tuple[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.rstrip("\n").rstrip("\r")
            if not ln:
                out.append(("", ""))
                continue
            parts = ln.split("\t")
            if len(parts) >= 2:
                out.append(("\t".join(parts[:-1]), parts[-1]))
            else:
                out.append((ln, ""))
    return out


def main() -> None:
    print("=" * 72)
    print(f"Tweets reference file: {TWEETS_PATH}")
    print(f"  exists: {TWEETS_PATH.exists()}")
    tweets = _load_tweets(TWEETS_PATH) if TWEETS_PATH.exists() else []
    print(f"  total tweet lines: {len(tweets)}")
    if tweets:
        labels_global = Counter(lbl for _, lbl in tweets)
        print(f"  global label distribution: {dict(labels_global)}")
    print("=" * 72)

    for split_name, fname in FILES.items():
        p = RAW_AR_DIR / fname
        print()
        print("-" * 72)
        print(f"FILE [{split_name}] = {p}")
        print(f"  exists: {p.exists()}")
        if not p.exists():
            continue

        lines = _read_lines(p)
        non_empty = [ln for ln in lines if ln.strip()]
        print(f"  total lines       : {len(lines)}")
        print(f"  non-empty entries : {len(non_empty)}")

        print("  first 5 raw lines :")
        for i, ln in enumerate(lines[:5]):
            print(f"    [{i}] {ln!r}")

        schema = _infer_schema(lines)
        print(f"  inferred schema   : {schema}")

        if schema == "int_id":
            ids = [int(ln.strip()) for ln in non_empty]
            print(f"  id range          : min={min(ids)}  max={max(ids)}")
            if tweets:
                # Try both 0-based and 1-based addressing; the one whose
                # max id stays in-bounds is the correct convention.
                max_id = max(ids)
                zero_based_ok = max_id < len(tweets)
                one_based_ok  = (max_id - 1) < len(tweets) and min(ids) >= 1
                print(f"  fits 0-based index (max < N={len(tweets)}): {zero_based_ok}")
                print(f"  fits 1-based index (min>=1 and max-1 < N): {one_based_ok}")

                # Show first 5 dereferenced entries with both conventions if
                # both seem plausible; otherwise show whichever works.
                for label, offset, ok in [
                    ("0-based", 0, zero_based_ok),
                    ("1-based", 1, one_based_ok),
                ]:
                    if not ok:
                        continue
                    print(f"  first 5 entries ({label} indexing into Tweets.txt):")
                    for i, idx in enumerate(ids[:5]):
                        ref = tweets[idx - offset]
                        text, lab = ref
                        text_short = (text[:80] + "...") if len(text) > 80 else text
                        print(f"    id={idx:5d} | label={lab:7s} | text={text_short}")

                # Class distribution by dereferencing (use whichever convention
                # appears valid; prefer 0-based if both work).
                offset = 0 if zero_based_ok else (1 if one_based_ok else None)
                if offset is not None:
                    labels = [tweets[i - offset][1] for i in ids]
                    dist = Counter(labels)
                    print(f"  class distribution: {dict(dist)}")
                    ratio = max(dist.values()) / max(min(dist.values()), 1)
                    print(f"  max/min class ratio: {ratio:.2f}")
            else:
                print("  (Tweets.txt unavailable; cannot dereference)")

        elif schema == "id_label":
            parts = [ln.split("\t") for ln in non_empty]
            labels = [p[1].strip().upper() for p in parts]
            dist = Counter(labels)
            print(f"  class distribution: {dict(dist)}")

        elif schema == "text_label":
            parts = [ln.split("\t") for ln in non_empty]
            labels = [p[-1].strip().upper() for p in parts]
            dist = Counter(labels)
            print(f"  class distribution: {dict(dist)}")
            tab = "\t"
            sample_lens = [len(tab.join(p[:-1])) for p in parts[:5]]
            print(f"  example text length (chars) first 5: {sample_lens}")
        else:
            print("  schema could not be inferred from first 50 lines")


if __name__ == "__main__":
    main()
