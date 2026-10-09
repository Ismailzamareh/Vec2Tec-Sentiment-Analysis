"""
Convenience runner -- executes the whole pipeline end-to-end:

    Step 1   validate_data.py
    Step 2   preprocess.py
    Step 3   baselines.py        (TF-IDF + LR / SVM)
    Step 4   train_word2vec.py   (Word2Vec + LR / SVM)
    Step 5   vec2tec_module.py   (Vec2Tec + LR / SVM)
    Step 6   evaluate.py         (metrics + confusion matrices + reports)
    Step 7   compare.py          (per-dataset comparison tables)
    Step 8   summarize.py        (summary + research-question answers)

Usage:
    python src/run_all.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

STEPS = [
    ("validate_data.py",   []),
    ("preprocess.py",      []),
    ("baselines.py",       []),
    ("train_word2vec.py",  []),
    ("vec2tec_module.py",  []),
    ("evaluate.py",        []),
    ("compare.py",         []),
    ("summarize.py",       []),
]


def main() -> int:
    for fname, args in STEPS:
        cmd = [sys.executable, "-u", str(ROOT / fname), *args]
        print(f"\n>>> {' '.join(cmd)}", flush=True)
        rc = subprocess.call(cmd)
        if rc != 0:
            print(f"!! step {fname} failed (exit={rc})")
            return rc
    print("\nAll steps completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
