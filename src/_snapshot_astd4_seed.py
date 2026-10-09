"""Internal helper: filter metrics_per_model.csv to astd4 rows and save
as results/metrics_seed_<SEED>_astd4.csv. Reads current RANDOM_SEED."""
from __future__ import annotations
import sys
from pathlib import Path
from config import RESULTS_DIR, RANDOM_SEED
from utils import load_csv, save_csv

src = RESULTS_DIR / "metrics_per_model.csv"
df = load_csv(src)
sub = df[df["dataset"] == "astd4"].copy()
if sub.empty:
    sys.exit("ERROR: no astd4 rows in metrics_per_model.csv")
out = RESULTS_DIR / f"metrics_seed_{RANDOM_SEED}_astd4.csv"
save_csv(sub, out)
print(f"  wrote {out.name} ({len(sub)} rows)")
