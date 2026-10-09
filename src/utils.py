"""Common helpers used across the Vec2Tec experimental pipeline."""
from __future__ import annotations

import csv as _csv
import json
import pickle
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def save_pickle(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_csv(path):
    """Robust CSV reader (falls back to python engine on parser error)."""
    try:
        return pd.read_csv(path, encoding="utf-8", low_memory=False)
    except Exception:
        return pd.read_csv(path, encoding="utf-8", engine="python")


def save_csv(df, path):
    """Save a DataFrame as UTF-8 CSV with full quoting of non-numeric fields."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8",
              quoting=_csv.QUOTE_NONNUMERIC)


@contextmanager
def timer(label):
    t0 = time.time()
    yield
    print(f"  [{label}] {time.time() - t0:.2f}s", flush=True)


def banner(text, char="="):
    print()
    print(char * 72)
    print(text)
    print(char * 72, flush=True)


def label_distribution(y):
    y = np.asarray(y)
    vals, counts = np.unique(y, return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, counts)}
