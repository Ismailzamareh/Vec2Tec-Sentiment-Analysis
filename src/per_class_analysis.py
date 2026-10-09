"""
Per-class analysis: stitch every classification_reports/*.csv into one
tidy long-format table and produce a per-dataset summary that highlights
which class is hardest and whether Vec2Tec helps a specific class over
standard Word2Vec.

Inputs:
    results/classification_reports/<model>_<dataset>.csv

Outputs:
    results/tables/per_class_long.csv
        columns: dataset, model, family, class, precision, recall,
                 f1_score, support
    results/tables/per_class_<dataset>.csv
        wide table per dataset for easy reading
    results/tables/per_class_vec2tec_vs_w2v.csv
        per-class F1 delta (vec2tec_lr - word2vec_lr) per dataset class
    results/tables/per_class_vec2tec_vs_w2v.md
        narrative / pretty markdown version of the above

Usage:
    python src/per_class_analysis.py
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from config import RESULTS_DIR, CR_DIR
from utils import banner, load_csv, save_csv


TBL_DIR = RESULTS_DIR / "tables"
TBL_DIR.mkdir(parents=True, exist_ok=True)


FAMILY_OF = {
    "tfidf_lr":         "TF-IDF",       "tfidf_svm":        "TF-IDF",
    "word2vec_lr":      "Word2Vec",     "word2vec_svm":     "Word2Vec",
    "vec2tec_lr":       "Vec2Tec",      "vec2tec_svm":      "Vec2Tec",
    "vec2tec_plain_lr": "Vec2Tec(plain)","vec2tec_plain_svm":"Vec2Tec(plain)",
}

# only "real" classes, not the summary rows in sklearn classification_report
SUMMARY_ROWS = {"accuracy", "macro avg", "weighted avg"}


def parse_one(path: Path) -> pd.DataFrame:
    """Parse a single classification_report CSV.

    Filename pattern: '<model>_<dataset>.csv'. The dataset is the LAST
    underscore-segment.
    """
    stem = path.stem  # e.g. 'tfidf_lr_astd4'
    # dataset = last token after final '_'
    m = re.match(r"^(.*)_([^_]+)$", stem)
    if not m:
        return pd.DataFrame()
    model, ds = m.group(1), m.group(2)

    df = load_csv(path)
    df = df[~df["group"].isin(SUMMARY_ROWS)].copy()
    df = df.rename(columns={"group": "class", "f1-score": "f1_score"})
    df["dataset"] = ds
    df["model"]   = model
    df["family"]  = FAMILY_OF.get(model, "other")
    keep = ["dataset", "model", "family", "class",
            "precision", "recall", "f1_score", "support"]
    return df[keep]


def main() -> None:
    banner("Per-class performance analysis")

    files = sorted(CR_DIR.glob("*.csv"))
    if not files:
        print("  no classification_reports found.")
        return

    long_frames = [parse_one(p) for p in files]
    long_df = pd.concat([f for f in long_frames if not f.empty],
                        ignore_index=True)
    long_df = long_df.sort_values(["dataset", "model", "class"]).reset_index(drop=True)

    out_long = TBL_DIR / "per_class_long.csv"
    save_csv(long_df, out_long)
    print(f"  wrote {out_long.relative_to(RESULTS_DIR.parent)} "
          f"({len(long_df)} rows)")

    # --- per-dataset wide tables -----------------------------------------
    for ds, sub in long_df.groupby("dataset"):
        wide = (sub.pivot_table(index=["class"], columns="model",
                                values="f1_score", aggfunc="first")
                   .round(4))
        sup = (sub.groupby("class")["support"].first().astype(int)
                  .reindex(wide.index))
        wide.insert(0, "support", sup.values)
        wide = wide.reset_index()
        out = TBL_DIR / f"per_class_{ds}.csv"
        save_csv(wide, out)
        print(f"  wrote {out.relative_to(RESULTS_DIR.parent)} "
              f"(classes: {wide['class'].tolist()})")

    # --- vec2tec_lr vs word2vec_lr delta ---------------------------------
    delta_rows = []
    for ds, sub in long_df.groupby("dataset"):
        v = sub[sub["model"] == "vec2tec_lr"].set_index("class")
        w = sub[sub["model"] == "word2vec_lr"].set_index("class")
        common = sorted(set(v.index) & set(w.index))
        for cls in common:
            delta_rows.append(dict(
                dataset=ds,
                **{"class": cls},
                support=int(v.loc[cls, "support"]),
                f1_word2vec_lr=round(float(w.loc[cls, "f1_score"]), 4),
                f1_vec2tec_lr=round(float(v.loc[cls, "f1_score"]), 4),
                delta_f1_pp=round((v.loc[cls, "f1_score"]
                                   - w.loc[cls, "f1_score"]) * 100, 4),
            ))
    delta_df = pd.DataFrame(delta_rows)
    out_d = TBL_DIR / "per_class_vec2tec_vs_w2v.csv"
    save_csv(delta_df, out_d)
    print(f"  wrote {out_d.relative_to(RESULTS_DIR.parent)} "
          f"({len(delta_df)} rows)")

    # pretty markdown
    md = ["# Per-class F1: Vec2Tec(LR) vs Word2Vec(LR)\n",
          "Single-seed snapshot (last evaluate.py run).\n"]
    for ds, sub in delta_df.groupby("dataset"):
        md.append(f"\n## {ds.upper()}\n")
        md.append(sub.to_markdown(index=False))
    out_md = TBL_DIR / "per_class_vec2tec_vs_w2v.md"
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"  wrote {out_md.relative_to(RESULTS_DIR.parent)}")

    # --- hardest class per dataset (lowest mean F1 across all models) -----
    hardest = (long_df.groupby(["dataset", "class"])["f1_score"]
                       .mean().reset_index()
                       .sort_values(["dataset", "f1_score"]))
    summary_rows = []
    for ds, g in hardest.groupby("dataset"):
        g = g.sort_values("f1_score")
        summary_rows.append(dict(
            dataset=ds,
            hardest_class=g.iloc[0]["class"],
            hardest_class_mean_f1=round(float(g.iloc[0]["f1_score"]), 4),
            easiest_class=g.iloc[-1]["class"],
            easiest_class_mean_f1=round(float(g.iloc[-1]["f1_score"]), 4),
        ))
    summary = pd.DataFrame(summary_rows)
    out_s = TBL_DIR / "hardest_class_per_dataset.csv"
    save_csv(summary, out_s)
    print(f"  wrote {out_s.relative_to(RESULTS_DIR.parent)}")
    banner("Hardest class per dataset (mean F1 across all 6 models)", char="-")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
