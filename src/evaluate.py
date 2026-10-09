"""
Step 6 - Full evaluation.

Reads the prediction files written by baselines.py, train_word2vec.py and
vec2tec_module.py, then computes for every (model, dataset) pair:

    * accuracy
    * macro precision / recall / F1
    * weighted F1 (for reference)
    * confusion matrix (saved as CSV)
    * classification report (saved as CSV)
    * one cumulative metrics table for all models

Outputs:
    results/metrics_per_model.csv
    results/confusion_matrices/<model>_<dataset>.csv
    results/classification_reports/<model>_<dataset>.csv

Usage:
    python src/evaluate.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
)

from config import (
    RESULTS_DIR, CM_DIR, CR_DIR,
    DATASETS, DATASETS_4CLASS, LABEL_NAMES, LABEL_NAMES_4,
)
from utils import banner, load_pickle, save_csv


PRED_GROUPS = ("tfidf", "word2vec", "vec2tec", "vec2tec_plain")
ALL_DATASETS = tuple(DATASETS) + tuple(DATASETS_4CLASS)


def _label_names_for(y_true, y_pred) -> tuple[list[int], dict[int, str]]:
    """Return (sorted_label_ids, id->name) inferred from predictions."""
    labels = sorted(set(np.asarray(y_true).tolist())
                    | set(np.asarray(y_pred).tolist()))
    if len(labels) > 2:
        name_map = LABEL_NAMES_4
    else:
        name_map = LABEL_NAMES
    # ensure every label has a printable name
    return labels, {lbl: name_map.get(lbl, str(lbl)) for lbl in labels}


def _load_predictions() -> Dict[str, Dict[str, dict]]:
    """Returns nested dict {dataset: {model: {predictions, y_true, ...}}}"""
    all_preds: Dict[str, Dict[str, dict]] = {ds: {} for ds in ALL_DATASETS}
    for ds in ALL_DATASETS:
        for group in PRED_GROUPS:
            p = RESULTS_DIR / f"_preds_{group}_{ds}.pkl"
            if not p.exists():
                continue
            block = load_pickle(p)
            for model_name, payload in block.items():
                all_preds[ds][model_name] = payload
    return all_preds


def evaluate_pair(model_name: str, ds: str, y_true, y_pred,
                  train_time_s: float | None = None) -> dict:
    acc = accuracy_score(y_true, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0)

    labels, name_map = _label_names_for(y_true, y_pred)
    target_names = [name_map[l] for l in labels]

    # confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{name_map[l]}" for l in labels],
        columns=[f"pred_{name_map[l]}" for l in labels],
    )
    save_csv(cm_df.reset_index().rename(columns={"index": "row"}),
             CM_DIR / f"{model_name}_{ds}.csv")

    # classification report
    cr = classification_report(
        y_true, y_pred, labels=labels, target_names=target_names,
        zero_division=0, output_dict=True,
    )
    cr_rows = []
    for k, v in cr.items():
        if isinstance(v, dict):
            cr_rows.append(dict(group=k, **{kk: vv for kk, vv in v.items()}))
        else:
            cr_rows.append(dict(group=k, value=v))
    save_csv(pd.DataFrame(cr_rows), CR_DIR / f"{model_name}_{ds}.csv")

    return dict(
        model=model_name, dataset=ds,
        accuracy=round(acc, 6),
        precision_macro=round(p_macro, 6),
        recall_macro=round(r_macro, 6),
        f1_macro=round(f1_macro, 6),
        f1_weighted=round(f1_w, 6),
        train_time_s=round(train_time_s, 4) if train_time_s is not None else None,
    )


def main() -> None:
    banner("Evaluating all models")
    all_preds = _load_predictions()

    rows: List[dict] = []
    for ds in ALL_DATASETS:
        if not all_preds[ds]:
            print(f"  no predictions found for dataset '{ds}', skipping.")
            continue
        for model_name, payload in all_preds[ds].items():
            y_true = payload["y_true"]
            y_pred = payload["predictions"]
            train_time = payload.get("train_time_s")
            row = evaluate_pair(model_name, ds, y_true, y_pred, train_time)
            rows.append(row)
            print(f"  {model_name:20s} {ds:5s} | "
                  f"acc={row['accuracy']:.4f} | "
                  f"f1_macro={row['f1_macro']:.4f} | "
                  f"f1_weighted={row['f1_weighted']:.4f}")

    df = pd.DataFrame(rows)
    df = df.sort_values(["dataset", "model"]).reset_index(drop=True)
    save_csv(df, RESULTS_DIR / "metrics_per_model.csv")
    banner("metrics_per_model.csv written", char="-")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
