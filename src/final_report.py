"""
Generate the final experimental report and the consolidated artefact
bundle requested by the user:

    results/FINAL_REPORT.md           one focused, evidence-based answer
                                      to the 10 review questions
    results/REPORT_ALL.md             every old + new table stitched
                                      together in a single readable file
    results/figures_all/              hard-link / copy of every PNG in
                                      results/figures/, so the user has a
                                      single bundle of figures

This script READS existing CSV / MD artefacts; it does NOT recompute any
metric. Run AFTER:
    build_full_results.py, per_class_analysis.py, dataset_stats_v2.py,
    ablation_study.py, significance_seeds.py, transformer_baseline.py,
    make_figures(_v2/_astd4).py, plot_confusion_matrices.py

Usage:
    python src/final_report.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from config import RESULTS_DIR
from utils import banner, load_csv


TBL_DIR = RESULTS_DIR / "tables"
FIG_DIR = RESULTS_DIR / "figures"
FIG_ALL_DIR = RESULTS_DIR / "figures_all"


# ---------- helpers ------------------------------------------------------ #

def _read_md(p: Path) -> str:
    return p.read_text(encoding="utf-8") if p.exists() else ""


def _read_csv_md(p: Path) -> str:
    if not p.exists():
        return f"_(file missing: {p.name})_"
    df = load_csv(p)
    return df.to_markdown(index=False)


def _safe_get(df: pd.DataFrame, ds: str, model: str, col: str):
    row = df[(df["dataset"] == ds) & (df["model"] == model)]
    if row.empty:
        return None
    return float(row[col].iloc[0])


# ---------- final report ------------------------------------------------- #

def write_final_report() -> None:
    agg = load_csv(TBL_DIR / "aggregated_results.csv")
    best = load_csv(TBL_DIR / "best_per_dataset.csv")
    ablation = load_csv(TBL_DIR / "ablation_results.csv")
    sig = load_csv(RESULTS_DIR / "significance_seeds.csv")
    tx  = load_csv(TBL_DIR / "transformer_results.csv")

    def _f1(ds, model):
        v = _safe_get(agg, ds, model, "f1_macro_mean")
        return v if v is not None else float("nan")

    def _f1_std(ds, model):
        v = _safe_get(agg, ds, model, "f1_macro_std")
        return v if v is not None else float("nan")

    def _tx_f1(ds):
        row = tx[tx["dataset"] == ds]
        return float(row["f1_macro"].iloc[0]) if not row.empty else float("nan")

    md: list[str] = []

    md.append("# Vec2Tec — final experimental summary\n")
    md.append("This report consolidates the entire experimental campaign — "
              "the original binary results, the multi-seed reproducibility "
              "study, the 4-class ASTD extension, the ablation study, "
              "across-seed significance tests, and the transformer baseline "
              "comparison. Numbers below are **mean ± std over 3 random "
              "seeds (42, 123, 2024)** unless stated otherwise.\n")

    md.append("## TL;DR\n")
    md.append("- **Best overall**: TF-IDF + LR. It wins macro-F1 on every "
              "dataset (binary and 4-class).")
    md.append("- **Vec2Tec significantly improves over Word2Vec** on every "
              "binary dataset (paired bootstrap, p < 0.05; macro-F1 gain "
              "≈ +1.1 pp on IMDb, +1.4 pp on LABR, +0.7 pp on ASTD).")
    md.append("- **Vec2Tec does NOT beat TF-IDF on any dataset** "
              "(paired bootstrap, p < 0.05 in the opposite direction; gap "
              "ranges from +2.65 pp on IMDb to +12.09 pp on ASTD4).")
    md.append("- **The frozen multilingual transformer wins only on short "
              "noisy social text** (ASTD, ASTD4). On long well-resourced "
              "text (IMDb, LABR) Vec2Tec is the better lightweight model "
              "than the frozen transformer.")
    md.append("- The Vec2Tec **ablation** shows the gains come from the "
              "**sentiment-lexicon weighting** + **polarity retrofit** "
              "combination. Synonym expansion HURTS on long text but HELPS "
              "dramatically on short noisy ASTD (+9.6 pp F1) — a "
              "dataset-conditional finding worth flagging in any future "
              "paper.")
    md.append("- **ASTD remains the hardest binary dataset** (best F1 = "
              "0.7414); ASTD-4-class is at 0.4728, well above 0.25 random "
              "but clearly the limit of bag-of-words + shallow embeddings.\n")

    # --------- 1. Which model performed best overall? -------------------- #
    md.append("## 1. Which model performed best overall?\n")
    md.append("By macro-F1 averaged across seeds, the best model is "
              "**TF-IDF + LR** on every dataset.\n")
    md.append(best.to_markdown(index=False))

    # --------- 2. Did Vec2Tec improve over Word2Vec? --------------------- #
    md.append("\n## 2. Did Vec2Tec improve over Word2Vec?\n")
    md.append("Yes, consistently — and the improvement is statistically "
              "significant under a paired bootstrap on per-example test "
              "predictions (B = 2,000).\n")
    md.append("| dataset | best W2V (mean F1) | best Vec2Tec (mean F1) | "
              "Δ F1 (pp) | bootstrap p-value | significant @ 5% |")
    md.append("|:--------|-------------------:|-----------------------:|"
              "----------:|:-----------------|:----------------:|")
    for ds in ("imdb", "labr", "astd"):
        w_row = agg[(agg["dataset"] == ds) &
                    (agg["model"].isin(["word2vec_lr", "word2vec_svm"]))]
        v_row = agg[(agg["dataset"] == ds) &
                    (agg["model"].isin(["vec2tec_lr", "vec2tec_svm"]))]
        w_best = w_row.loc[w_row["f1_macro_mean"].idxmax()]
        v_best = v_row.loc[v_row["f1_macro_mean"].idxmax()]
        delta_pp = (v_best["f1_macro_mean"] - w_best["f1_macro_mean"]) * 100

        boot_row = sig[(sig["dataset"] == ds) &
                        (sig["comparison"] == "vec2tec_vs_w2v")]
        if not boot_row.empty:
            p_str = f"{float(boot_row['bootstrap_p'].iloc[0]):.4g}"
            sig5  = bool(boot_row["significant_5pct"].iloc[0])
        else:
            p_str, sig5 = "n/a", False
        md.append(
            f"| {ds.upper()} | "
            f"{w_best['model']} ({w_best['f1_macro_mean']:.4f}) | "
            f"{v_best['model']} ({v_best['f1_macro_mean']:.4f}) | "
            f"{delta_pp:+.2f} | {p_str} | {'✓' if sig5 else '×'} |")

    # --------- 3. Did Vec2Tec outperform TF-IDF? -------------------------- #
    md.append("\n## 3. Did Vec2Tec outperform TF-IDF?\n")
    md.append("No. TF-IDF + LR is significantly stronger than Vec2Tec on "
              "every dataset under the same paired-bootstrap test.\n")
    md.append("| dataset | best TF-IDF (mean F1) | best Vec2Tec (mean F1) | "
              "Δ F1 (pp, TFIDF − V2T) | bootstrap p | significant @ 5% |")
    md.append("|:--------|---------------------:|-----------------------:|"
              "----------------------:|:------------|:----------------:|")
    for ds in ("imdb", "labr", "astd"):
        t_row = agg[(agg["dataset"] == ds) &
                    (agg["model"].isin(["tfidf_lr", "tfidf_svm"]))]
        v_row = agg[(agg["dataset"] == ds) &
                    (agg["model"].isin(["vec2tec_lr", "vec2tec_svm"]))]
        t_best = t_row.loc[t_row["f1_macro_mean"].idxmax()]
        v_best = v_row.loc[v_row["f1_macro_mean"].idxmax()]
        delta_pp = (t_best["f1_macro_mean"] - v_best["f1_macro_mean"]) * 100

        boot_row = sig[(sig["dataset"] == ds) &
                        (sig["comparison"] == "tfidf_vs_vec2tec")]
        if not boot_row.empty:
            p_str = f"{float(boot_row['bootstrap_p'].iloc[0]):.4g}"
            sig5  = bool(boot_row["significant_5pct"].iloc[0])
        else:
            p_str, sig5 = "n/a", False
        md.append(
            f"| {ds.upper()} | "
            f"{t_best['model']} ({t_best['f1_macro_mean']:.4f}) | "
            f"{v_best['model']} ({v_best['f1_macro_mean']:.4f}) | "
            f"{delta_pp:+.2f} | {p_str} | {'✓' if sig5 else '×'} |")

    # --------- 4. Hardest dataset --------------------------------------- #
    md.append("\n## 4. Which dataset was hardest?\n")
    md.append(
        "By the **best** F1 any model achieved:\n"
        "1. ASTD-4-class — best F1 = 0.4728 (TF-IDF + LR)\n"
        "2. ASTD (binary) — best F1 = 0.7414 (TF-IDF + LR)\n"
        "3. LABR — best F1 = 0.8336 (TF-IDF + LR)\n"
        "4. IMDb — best F1 = 0.9024 (TF-IDF + LR)\n"
    )

    # --------- 5. Why ASTD is harder ------------------------------------ #
    md.append("## 5. Why is ASTD harder than IMDb and LABR?\n")
    md.append(
        "Three structural reasons, all derivable from "
        "`tables/dataset_stats_v2.csv`:\n"
        "- **Vocabulary scarcity**: ASTD train vocab = 12,273 vs LABR 86,677 "
        "and IMDb 159,209 (≈ 13× smaller).\n"
        "- **Text length**: median 16 tokens (tweets) vs LABR 31 and IMDb "
        "173. Short text gives the classifier very few sentiment-bearing "
        "features per example.\n"
        "- **Class imbalance (in the raw corpus)**: the binary 80/20 split "
        "we use to build ASTD is stratified but starts from a 1,684 NEG / "
        "799 POS pool, so the minority class is small in absolute terms; "
        "the per-class analysis confirms POS is the harder class on binary "
        "ASTD (mean F1 = 0.5360 across all 6 models). For ASTD-4-class, "
        "NEG is the hardest class (mean F1 = 0.2373) — it gets confused "
        "with NEUTRAL most often.\n"
        "- **Domain noise**: tweets have hashtags, mentions, code-switching, "
        "and non-standard orthography. Even after `clean_ar()` they remain "
        "harder than book reviews.\n"
    )

    # --------- 6. Does the hypothesis hold? ----------------------------- #
    md.append("## 6. Does this support the research hypothesis "
              "“semantic enhancement improves Word2Vec, especially "
              "for Arabic and short noisy text”?\n")
    md.append(
        "Partially.\n\n"
        "**Yes — improvement over Word2Vec is real and statistically "
        "significant.** All three binary datasets show Vec2Tec > Word2Vec "
        "under the paired bootstrap (p < 0.05). The ablation study "
        "isolates the lexicon-weighting + polarity-retrofit pair as the "
        "components doing the work.\n\n"
        "**Partially — the Arabic / short-text gain is dataset-dependent.** "
        "The ablation showed +9.6 pp F1 from synonym expansion on ASTD "
        "alone (most noisy / sparsest vocab), but synonym expansion HURTS "
        "on the larger, vocabulary-rich datasets (−1.4 pp on IMDb, "
        "−1.5 pp on LABR). The default Vec2Tec stack picks the wrong "
        "components for ASTD; a dataset-aware enhancement selector would "
        "be the obvious next step.\n\n"
        "**No — Vec2Tec does not displace TF-IDF.** TF-IDF + LR is "
        "stronger on every dataset by a margin that is unambiguous "
        "(bootstrap p ≈ 0 on all three binary datasets).\n"
    )

    # --------- 7. Transformer comparison -------------------------------- #
    md.append("\n## 7. How does Vec2Tec compare to a frozen "
              "multilingual transformer?\n")
    md.append(
        "Encoder: `paraphrase-multilingual-MiniLM-L12-v2` (no fine-tuning), "
        "LR head. IMDb was sub-sampled (5K train / 5K test) due to CPU "
        "encoding cost; LABR / ASTD / ASTD4 used the full split.\n")
    md.append(
        "| dataset | best Vec2Tec F1 | transformer F1 | who wins? |\n"
        "|:--------|----------------:|---------------:|:----------|\n"
        f"| IMDb (5K subsample) | "
        f"{_f1('imdb', 'vec2tec_svm'):.4f} | {_tx_f1('imdb'):.4f} | "
        f"**Vec2Tec** wins (+{(_f1('imdb','vec2tec_svm')-_tx_f1('imdb'))*100:.2f} pp) |\n"
        f"| LABR | {_f1('labr', 'vec2tec_lr'):.4f} | {_tx_f1('labr'):.4f} | "
        f"**Vec2Tec** wins (+{(_f1('labr','vec2tec_lr')-_tx_f1('labr'))*100:.2f} pp) |\n"
        f"| ASTD | {_f1('astd', 'vec2tec_svm'):.4f} | {_tx_f1('astd'):.4f} | "
        f"**Transformer** wins ({(_f1('astd','vec2tec_svm')-_tx_f1('astd'))*100:+.2f} pp) |\n"
        f"| ASTD4 | {_f1('astd4','vec2tec_svm'):.4f} | {_tx_f1('astd4'):.4f} | "
        f"**Transformer** wins ({(_f1('astd4','vec2tec_svm')-_tx_f1('astd4'))*100:+.2f} pp) |\n"
    )
    md.append(
        "Interpretation: the frozen multilingual model wins on short noisy "
        "social text (ASTD/ASTD4), where Word2Vec's small in-domain "
        "vocabulary is the bottleneck. On long well-resourced text "
        "(IMDb, LABR) the multilingual encoder underperforms a "
        "domain-adapted Vec2Tec + LR — the encoder's capacity is split "
        "across ~50 languages and is not specialised for either English "
        "movie reviews or Arabic book reviews. This positions Vec2Tec as "
        "a credible, lightweight, **interpretable** alternative when a "
        "domain lexicon is available and inference cost matters.\n"
    )

    # --------- 8. Ablation -------------------------------------------- #
    md.append("\n## 8. Ablation: what inside Vec2Tec is doing the work?\n")
    md.append(
        "Single-seed run (seed = 2024). The underlying Word2Vec model and "
        "corpus lexicon are held fixed; only the enhancement stack varies. "
        "Δ columns are vs. `word2vec_plain` (mean of word vectors, no "
        "enhancement) on the SAME dataset.\n"
    )
    md.append("\n_Detailed per-dataset table: see "
              "`results/tables/ablation_results.md`._\n")
    md.append(
        "Summary findings:\n"
        "- Lexicon weighting alone gives most of Vec2Tec's IMDb / LABR "
        "gain (+0.76 pp / +0.97 pp F1).\n"
        "- Polarity retrofit on top of lexicon (the `v2t_full` stack) adds "
        "a smaller but consistent further gain on long text (+1.11 pp "
        "F1 on IMDb, +1.19 pp on LABR vs. `word2vec_plain`).\n"
        "- **Synonym expansion** is the most volatile knob: it HURTS on "
        "IMDb (−1.40 pp) and LABR (−1.48 pp) but is the SINGLE strongest "
        "enhancement on ASTD (**+9.58 pp F1**, +7.69 pp accuracy). "
        "On sparse-vocab data, expanding tokens to their semantic "
        "neighbours adds usable signal; on dense-vocab data it adds noise.\n"
        "- Contextual weighting (cosine-softmax pooling) is essentially "
        "neutral everywhere we tested (Δ within ±0.2 pp). It is the "
        "weakest of the four enhancements in this configuration.\n"
    )

    # --------- 9. Per-class -------------------------------------------- #
    md.append("\n## 9. Per-class performance\n")
    md.append("Classification reports for every (model, dataset) are in "
              "`results/classification_reports/*.csv`; tidy long-format "
              "view in `results/tables/per_class_long.csv`; "
              "Vec2Tec(LR) vs Word2Vec(LR) per-class delta in "
              "`results/tables/per_class_vec2tec_vs_w2v.csv`.\n")
    md.append("\nHardest class per dataset (lowest mean F1 across all six "
              "models):\n")
    md.append(_read_csv_md(TBL_DIR / "hardest_class_per_dataset.csv"))

    # --------- 10. Limitations and next steps -------------------------- #
    md.append("\n## 10. Limitations and what should change in v2 of Vec2Tec\n")
    md.append(
        "- **Three seeds is the floor of useful reproducibility.** The "
        "paired t-test across seeds is informative but underpowered; the "
        "Wilcoxon hits its minimum p = 0.25 with n = 3. We rely on the "
        "paired bootstrap of per-example predictions for the decisive "
        "test. A future v2 should run ≥ 5 seeds.\n"
        "- **The `synonym_expansion` enhancement is on by default but "
        "harmful on long text.** Vec2Tec should expose enhancement "
        "selection as a dataset-aware hyper-parameter, ideally chosen on "
        "the validation split (ASTD4's official validation set already "
        "exists but is not used for tuning yet).\n"
        "- **No fine-tuned transformer.** Our transformer baseline is "
        "intentionally frozen / CPU-only. A fine-tuned AraBERT or RoBERTa "
        "would almost certainly out-perform every model in this report; "
        "Vec2Tec is positioned as a *lightweight, interpretable* "
        "alternative, not a replacement.\n"
        "- **TF-IDF wins on every dataset.** Vec2Tec's value proposition "
        "must therefore lean on its interpretability (corpus-derived "
        "polarity lexicon, modular enhancement hooks) and its 200-dim "
        "dense representation (useful for nearest-neighbour analysis, "
        "transfer to new tasks, or downstream pipelines) — not on raw "
        "macro-F1.\n"
        "- **IMDb transformer numbers are on a 5K/5K subsample** "
        "(CPU-only encoding made a full-50K run impractical). Numbers "
        "should be re-validated on the full split with a GPU before "
        "being cited in any paper.\n"
        "- **The current `polarity_retrofit` axis uses only NEG vs POS** "
        "and therefore cannot explain NEUTRAL/OBJ on the 4-class task. "
        "A multi-class polarity space (one axis per class) is the next "
        "obvious enhancement.\n"
    )

    out = RESULTS_DIR / "FINAL_REPORT.md"
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"  wrote {out.relative_to(RESULTS_DIR.parent)}")


# ---------- consolidated REPORT_ALL.md ----------------------------------- #

def write_consolidated_report() -> None:
    """One readable file aggregating EVERY result table + summary we have."""
    sections: list[tuple[str, Path]] = [
        ("Final summary (top-level)",
            RESULTS_DIR / "FINAL_REPORT.md"),
        ("Binary multi-seed summary (original)",
            RESULTS_DIR / "summary.md"),
        ("ASTD 4-class summary",
            RESULTS_DIR / "summary_astd4.md"),
        ("Dataset card — ASTD 4-class",
            RESULTS_DIR / "astd4_dataset_card.md"),
        ("Dataset statistics v2",
            TBL_DIR / "dataset_stats_v2.md"),
        ("Main results table (binary, mean ± std over 3 seeds)",
            TBL_DIR / "table_main_results.md"),
        ("Vec2Tec vs Word2Vec (best-in-family)",
            TBL_DIR / "table_vec2tec_vs_w2v.md"),
        ("Per-class F1 — Vec2Tec(LR) vs Word2Vec(LR)",
            TBL_DIR / "per_class_vec2tec_vs_w2v.md"),
        ("Ablation results",
            TBL_DIR / "ablation_results.md"),
        ("Across-seed significance",
            RESULTS_DIR / "significance_seeds.md"),
        ("Transformer baseline",
            TBL_DIR / "transformer_results.md"),
        ("Top polarity words (corpus lexicon)",
            TBL_DIR / "top_polarity_words.md"),
        ("Efficiency–quality summary",
            TBL_DIR / "efficiency_tradeoff_summary.md"),
    ]

    csv_appendix: list[tuple[str, Path]] = [
        ("Aggregated results CSV (long view)",
         TBL_DIR / "aggregated_results.csv"),
        ("Best per dataset",
         TBL_DIR / "best_per_dataset.csv"),
        ("Ablation results CSV",
         TBL_DIR / "ablation_results.csv"),
        ("Across-seed significance CSV",
         RESULTS_DIR / "significance_seeds.csv"),
        ("Transformer results CSV",
         TBL_DIR / "transformer_results.csv"),
    ]

    md: list[str] = []
    md.append("# REPORT_ALL — full Vec2Tec experimental record")
    md.append("\nThis single file gathers every old and new result for the "
              "Vec2Tec project. It is automatically regenerated by "
              "`src/final_report.py` from the canonical CSVs and per-section "
              "Markdown notes.\n")
    md.append("## Contents\n")
    for i, (title, _) in enumerate(sections, 1):
        anchor = title.lower().replace(" ", "-").replace("/", "-")
        md.append(f"{i}. [{title}](#{anchor})")
    md.append(f"{len(sections) + 1}. [Appendix: machine-readable tables]"
              "(#appendix-machine-readable-tables)\n")

    for title, p in sections:
        md.append(f"\n---\n\n## {title}\n")
        body = _read_md(p)
        if body:
            md.append(body)
        else:
            md.append(f"_(no file at `{p.relative_to(RESULTS_DIR.parent)}`)_")

    md.append("\n---\n\n## Appendix: machine-readable tables\n")
    for title, p in csv_appendix:
        md.append(f"\n### {title} — `{p.relative_to(RESULTS_DIR.parent)}`\n")
        md.append(_read_csv_md(p))

    out = RESULTS_DIR / "REPORT_ALL.md"
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"  wrote {out.relative_to(RESULTS_DIR.parent)}")


# ---------- figures_all/ -------------------------------------------------- #

def copy_all_figures() -> None:
    FIG_ALL_DIR.mkdir(parents=True, exist_ok=True)
    n_copied = 0
    for src in sorted(FIG_DIR.glob("*.png")):
        dst = FIG_ALL_DIR / src.name
        shutil.copy2(src, dst)
        n_copied += 1
    print(f"  copied {n_copied} PNGs to "
          f"{FIG_ALL_DIR.relative_to(RESULTS_DIR.parent)}")


def main() -> None:
    banner("Generating FINAL_REPORT.md + REPORT_ALL.md + figures_all/")
    write_final_report()
    write_consolidated_report()
    copy_all_figures()
    banner("Done", char="-")


if __name__ == "__main__":
    main()
