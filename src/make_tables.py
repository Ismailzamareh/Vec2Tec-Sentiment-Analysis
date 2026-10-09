"""
Step 13 - LaTeX + Markdown tables for the paper.

Reads:
    data/processed/<ds>_{train,test}.csv
    results/metrics_aggregated.csv

Writes to results/tables/:
    table_dataset_stats.{tex,md}
    table_main_results.{tex,md}
    table_vec2tec_vs_w2v.{tex,md}

Usage:
    python src/make_tables.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import PROCESSED_DIR, RESULTS_DIR, DATASETS, LABEL_NAMES
from utils import banner, load_csv


TBL_DIR = RESULTS_DIR / "tables"
TBL_DIR.mkdir(parents=True, exist_ok=True)

MODELS_ORDER = ("tfidf_lr", "tfidf_svm",
                "word2vec_lr", "word2vec_svm",
                "vec2tec_lr", "vec2tec_svm")

MODEL_LABEL = {
    "tfidf_lr":     "TF-IDF + LR",
    "tfidf_svm":    "TF-IDF + SVM",
    "word2vec_lr":  "Word2Vec + LR",
    "word2vec_svm": "Word2Vec + SVM",
    "vec2tec_lr":   "Vec2Tec + LR",
    "vec2tec_svm":  "Vec2Tec + SVM",
}


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  wrote {path.relative_to(RESULTS_DIR.parent)}")


# ---------- 2.1 ----------------------------------------------------------- #

def _dataset_stats() -> pd.DataFrame:
    rows = []
    for ds in DATASETS:
        tr = load_csv(PROCESSED_DIR / f"{ds}_train.csv")
        te = load_csv(PROCESSED_DIR / f"{ds}_test.csv")

        train_text = tr["text"].fillna("").astype(str)
        test_text  = te["text"].fillna("").astype(str)
        train_tokens = train_text.str.split()
        test_tokens  = test_text.str.split()

        vocab = set()
        for toks in train_tokens:
            vocab.update(toks)

        avg_len_train = float(train_tokens.str.len().mean())
        avg_len_test  = float(test_tokens.str.len().mean())
        avg_len = (avg_len_train + avg_len_test) / 2.0

        cls_tr = tr["label"].value_counts().to_dict()
        n_pos = int(cls_tr.get(1, 0))
        n_neg = int(cls_tr.get(0, 0))
        balance = f"{n_pos / (n_pos + n_neg):.2f}/{n_neg / (n_pos + n_neg):.2f}"

        rows.append(dict(
            dataset=ds.upper(),
            train_size=len(tr),
            test_size=len(te),
            vocab_size=len(vocab),
            avg_doc_len=round(avg_len, 1),
            class_balance_pos_neg=balance,
        ))
    return pd.DataFrame(rows)


def write_dataset_stats() -> None:
    df = _dataset_stats()
    headers = {
        "dataset": "Dataset",
        "train_size": "Train size",
        "test_size": "Test size",
        "vocab_size": "Vocab size (train)",
        "avg_doc_len": "Avg doc length (tokens)",
        "class_balance_pos_neg": "Class balance (pos/neg)",
    }
    show = df.rename(columns=headers)

    _write(TBL_DIR / "table_dataset_stats.md", show.to_markdown(index=False))

    tex = (show.to_latex(index=False, escape=True,
                         caption="Dataset statistics after preprocessing.",
                         label="tab:dataset_stats"))
    _write(TBL_DIR / "table_dataset_stats.tex", tex)


# ---------- 2.2 ----------------------------------------------------------- #

def _format_pm(mean: float, std: float) -> str:
    return f"{mean:.4f} ± {std:.4f}"


def write_main_results(agg: pd.DataFrame) -> None:
    blocks_md  = ["# Main results (mean ± std over 3 seeds)\n"]
    blocks_tex = []

    for ds in DATASETS:
        sub = agg[agg["dataset"] == ds].copy()
        # Order rows deterministically
        sub["__order"] = sub["model"].map({m: i for i, m in enumerate(MODELS_ORDER)})
        sub = sub.sort_values("__order").drop(columns="__order")

        best_idx = sub["f1_macro_mean"].idxmax()

        # Markdown
        rows_md = [f"\n## {ds.upper()}\n",
                   "| Model | Accuracy | Precision (macro) | Recall (macro) | F1-macro |",
                   "|:------|---------:|------------------:|---------------:|---------:|"]
        rows_tex = [
            r"\begin{table}[ht]",
            r"\centering",
            rf"\caption{{Main results on {ds.upper()} (mean $\pm$ std over 3 seeds; best F1-macro in \textbf{{bold}}).}}",
            rf"\label{{tab:main_results_{ds}}}",
            r"\begin{tabular}{lcccc}",
            r"\toprule",
            r"Model & Accuracy & Precision (macro) & Recall (macro) & F1-macro \\",
            r"\midrule",
        ]

        for idx, r in sub.iterrows():
            f1   = _format_pm(r["f1_macro_mean"],        r["f1_macro_std"])
            acc  = _format_pm(r["acc_mean"],             r["acc_std"])
            prec = _format_pm(r["precision_macro_mean"], r["precision_macro_std"])
            rec  = _format_pm(r["recall_macro_mean"],    r["recall_macro_std"])
            name = MODEL_LABEL[r["model"]]

            if idx == best_idx:
                rows_md.append(f"| **{name}** | {acc} | {prec} | {rec} | **{f1}** |")
                rows_tex.append(rf"\textbf{{{name}}} & {acc} & {prec} & {rec} & \textbf{{{f1}}} \\")
            else:
                rows_md.append(f"| {name} | {acc} | {prec} | {rec} | {f1} |")
                rows_tex.append(rf"{name} & {acc} & {prec} & {rec} & {f1} \\")

        rows_tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
        blocks_md.append("\n".join(rows_md))
        blocks_tex.append("\n".join(rows_tex))

    _write(TBL_DIR / "table_main_results.md",  "\n".join(blocks_md) + "\n")
    _write(TBL_DIR / "table_main_results.tex", "\n".join(blocks_tex))


# ---------- 2.3 ----------------------------------------------------------- #

def write_vec2tec_vs_w2v(agg: pd.DataFrame) -> None:
    # Compare best-in-family per dataset on accuracy and f1_macro.
    metrics = [("acc_mean",      "acc_std",      "Accuracy"),
               ("f1_macro_mean", "f1_macro_std", "F1-macro")]
    w2v_family = ("word2vec_lr", "word2vec_svm")
    v2t_family = ("vec2tec_lr",  "vec2tec_svm")

    def _best(family, ds, mean_col):
        sub = agg[(agg["model"].isin(family)) & (agg["dataset"] == ds)]
        i = sub[mean_col].idxmax()
        return sub.loc[i]

    headers_md = ["| Dataset | Metric | Word2Vec (best) | Vec2Tec (best) | Δ (V2T − W2V) |",
                  "|:--------|:-------|:---------------:|:--------------:|:-------------:|"]
    rows_md = []

    tex = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Vec2Tec vs Word2Vec (best classifier per family, mean $\pm$ std over 3 seeds).}",
        r"\label{tab:vec2tec_vs_w2v}",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Dataset & Metric & Word2Vec (best) & Vec2Tec (best) & $\Delta$ (V2T $-$ W2V) \\",
        r"\midrule",
    ]

    for ds in DATASETS:
        for mean_col, std_col, label in metrics:
            wrow = _best(w2v_family, ds, mean_col)
            vrow = _best(v2t_family, ds, mean_col)
            wstr = _format_pm(wrow[mean_col], wrow[std_col]) + f"  ({MODEL_LABEL[wrow['model']]})"
            vstr = _format_pm(vrow[mean_col], vrow[std_col]) + f"  ({MODEL_LABEL[vrow['model']]})"
            delta = (vrow[mean_col] - wrow[mean_col]) * 100.0
            dstr  = f"{delta:+.2f} pp"

            rows_md.append(f"| {ds.upper()} | {label} | {wstr} | {vstr} | **{dstr}** |")
            tex.append(rf"{ds.upper()} & {label} & {wstr} & {vstr} & \textbf{{{dstr}}} \\")

    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    _write(TBL_DIR / "table_vec2tec_vs_w2v.md",
           "# Vec2Tec vs Word2Vec\n\n" + "\n".join(headers_md + rows_md) + "\n")
    _write(TBL_DIR / "table_vec2tec_vs_w2v.tex", "\n".join(tex))


# ---------- main ---------------------------------------------------------- #

def main() -> None:
    banner("Generating LaTeX + Markdown tables")
    agg = load_csv(RESULTS_DIR / "metrics_aggregated.csv")

    write_dataset_stats()
    write_main_results(agg)
    write_vec2tec_vs_w2v(agg)

    banner(f"All tables saved under {TBL_DIR.relative_to(RESULTS_DIR.parent)}",
           char="-")


if __name__ == "__main__":
    main()
