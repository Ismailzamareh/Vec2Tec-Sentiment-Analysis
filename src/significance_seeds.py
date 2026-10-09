"""
Across-seed statistical significance tests.

For every binary dataset and every (model_a, model_b) pair listed below,
this script runs:

    1. Paired t-test across the 3 seeds (a small-N parametric test;
       interpret with care — three observations is at the edge of what
       any test can resolve).
    2. Wilcoxon signed-rank test across the 3 seeds (non-parametric
       counterpart). With n=3 the test cannot reach a two-sided p < 0.05,
       so we report it but flag the limitation.
    3. Paired bootstrap on per-example test-set predictions (B=2000
       resamples). For each bootstrap sample we compute
       f1_macro(model_a) - f1_macro(model_b); we report the 95%
       confidence interval and the two-sided p-value (proportion of
       bootstrap diffs that crossed zero).

Comparisons performed:
    - vec2tec_lr  vs  word2vec_lr
    - tfidf_lr    vs  vec2tec_lr

Inputs:
    results/metrics_seed_{42,123,2024}.csv      (binary only)
    results/_preds_{group}_{ds}.pkl             (last-run predictions
                                                 on the test set)

Outputs:
    results/significance_seeds.csv
    results/significance_seeds.md

Usage:
    python src/significance_seeds.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score

from config import RESULTS_DIR, DATASETS
from utils import banner, load_csv, load_pickle, save_csv


SEEDS = (42, 123, 2024)
B_BOOTSTRAP = 2000
RNG_SEED = 12345

COMPARISONS = [
    # (label,           model_a,      group_a,    model_b,      group_b)
    ("vec2tec_vs_w2v",  "vec2tec_lr", "vec2tec",  "word2vec_lr","word2vec"),
    ("tfidf_vs_vec2tec","tfidf_lr",   "tfidf",    "vec2tec_lr", "vec2tec"),
]


# ---------- 1+2: across-seed parametric / non-parametric ------------------ #

def collect_seed_f1(model: str) -> dict[str, list[float]]:
    """Return {dataset: [f1_seed42, f1_seed123, f1_seed2024]}."""
    out: dict[str, list[float]] = {ds: [] for ds in DATASETS}
    for s in SEEDS:
        p = RESULTS_DIR / f"metrics_seed_{s}.csv"
        if not p.exists():
            continue
        df = load_csv(p)
        for ds in DATASETS:
            row = df[(df["model"] == model) & (df["dataset"] == ds)]
            if not row.empty:
                out[ds].append(float(row["f1_macro"].iloc[0]))
    return out


def paired_t(a: list[float], b: list[float]) -> tuple[float, float]:
    if len(a) < 2 or len(b) < 2 or len(a) != len(b):
        return float("nan"), float("nan")
    try:
        t, p = stats.ttest_rel(a, b)
        return float(t), float(p)
    except Exception:
        return float("nan"), float("nan")


def paired_wilcoxon(a: list[float], b: list[float]) -> tuple[float, float]:
    if len(a) < 2 or len(b) < 2 or len(a) != len(b):
        return float("nan"), float("nan")
    diffs = np.asarray(a) - np.asarray(b)
    if np.all(diffs == 0):
        return float("nan"), 1.0
    try:
        w, p = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        return float(w), float(p)
    except Exception:
        return float("nan"), float("nan")


# ---------- 3: paired bootstrap on per-example predictions --------------- #

def _load_pred(group: str, ds: str, model: str) -> tuple[np.ndarray, np.ndarray]:
    p = RESULTS_DIR / f"_preds_{group}_{ds}.pkl"
    block = load_pickle(p)
    payload = block[model]
    return (np.asarray(payload["y_true"]),
            np.asarray(payload["predictions"]))


def paired_bootstrap_f1(y_true: np.ndarray,
                        y_a: np.ndarray,
                        y_b: np.ndarray,
                        B: int = B_BOOTSTRAP,
                        rng_seed: int = RNG_SEED) -> dict:
    rng = np.random.default_rng(rng_seed)
    n = len(y_true)
    diffs = np.empty(B, dtype=np.float64)
    obs_a = f1_score(y_true, y_a, average="macro", zero_division=0)
    obs_b = f1_score(y_true, y_b, average="macro", zero_division=0)
    obs_diff = obs_a - obs_b

    for i in range(B):
        idx = rng.integers(0, n, n)
        ya = y_a[idx]; yb = y_b[idx]; yt = y_true[idx]
        diffs[i] = (f1_score(yt, ya, average="macro", zero_division=0)
                    - f1_score(yt, yb, average="macro", zero_division=0))

    lo, hi = np.percentile(diffs, [2.5, 97.5])
    # two-sided p-value: proportion of bootstrap diffs whose sign disagrees
    # with the observed sign (a quick, common convention).
    if obs_diff >= 0:
        p_two = 2 * float((diffs < 0).mean())
    else:
        p_two = 2 * float((diffs > 0).mean())
    p_two = min(p_two, 1.0)

    return dict(obs_a_f1=round(float(obs_a), 6),
                obs_b_f1=round(float(obs_b), 6),
                obs_diff=round(float(obs_diff), 6),
                ci_lo=round(float(lo), 6),
                ci_hi=round(float(hi), 6),
                p_bootstrap=round(p_two, 6),
                n_test_examples=int(n))


# ---------- driver ------------------------------------------------------- #

def main() -> None:
    banner("Across-seed significance tests (paired t / Wilcoxon / bootstrap)")

    rows = []
    md_blocks = ["# Across-seed significance tests\n",
                 "Comparisons are performed on binary datasets only "
                 "(imdb, labr, astd). Three seeds — paired t and Wilcoxon "
                 "are reported for completeness but have very low power "
                 "with n=3 and should be read alongside the paired-bootstrap "
                 "result, which uses 2000 resamples of the per-example "
                 "test-set predictions.\n"]

    for label, m_a, g_a, m_b, g_b in COMPARISONS:
        md_blocks.append(f"\n## {label}: `{m_a}` vs `{m_b}`\n")
        md_blocks.append(
            "| dataset | mean A | mean B | mean Δ | paired t (p) | "
            "Wilcoxon (p) | bootstrap Δ obs | bootstrap 95% CI | "
            "bootstrap p | sig (5%) |")
        md_blocks.append(
            "|:--------|------:|------:|------:|:------------|:------------|"
            ":---------------|:-----------------|:-----------|:-------:|")
        a_per_seed = collect_seed_f1(m_a)
        b_per_seed = collect_seed_f1(m_b)
        for ds in DATASETS:
            a = a_per_seed.get(ds, [])
            b = b_per_seed.get(ds, [])
            mean_a = float(np.mean(a)) if a else float("nan")
            mean_b = float(np.mean(b)) if b else float("nan")
            mean_diff = mean_a - mean_b
            std_diff = (float(np.std(np.asarray(a) - np.asarray(b), ddof=1))
                        if len(a) == len(b) and len(a) >= 2 else float("nan"))
            t_stat, t_p = paired_t(a, b)
            w_stat, w_p = paired_wilcoxon(a, b)

            # bootstrap on the current-state predictions
            try:
                y_true, y_a = _load_pred(g_a, ds, m_a)
                y_true_b, y_b = _load_pred(g_b, ds, m_b)
                if not np.array_equal(y_true, y_true_b):
                    print(f"  [{ds}] WARNING: y_true mismatch between {m_a} and {m_b}",
                          file=sys.stderr)
                boot = paired_bootstrap_f1(y_true, y_a, y_b)
            except FileNotFoundError as e:
                print(f"  [{ds}] bootstrap skipped: {e}")
                boot = {k: float("nan") for k in
                        ("obs_a_f1", "obs_b_f1", "obs_diff",
                         "ci_lo", "ci_hi", "p_bootstrap", "n_test_examples")}

            sig5 = bool((not np.isnan(boot["p_bootstrap"]))
                        and boot["p_bootstrap"] < 0.05)
            row = dict(
                comparison=label, dataset=ds,
                model_a=m_a, model_b=m_b,
                mean_f1_a=round(mean_a, 6) if a else None,
                mean_f1_b=round(mean_b, 6) if b else None,
                mean_diff_f1=round(mean_diff, 6) if a and b else None,
                std_diff_f1=(round(std_diff, 6)
                             if not np.isnan(std_diff) else None),
                t_stat=round(t_stat, 6) if not np.isnan(t_stat) else None,
                t_p=round(t_p, 6) if not np.isnan(t_p) else None,
                wilcoxon_stat=round(w_stat, 6) if not np.isnan(w_stat) else None,
                wilcoxon_p=round(w_p, 6) if not np.isnan(w_p) else None,
                bootstrap_obs_diff=boot["obs_diff"],
                bootstrap_ci_lo=boot["ci_lo"],
                bootstrap_ci_hi=boot["ci_hi"],
                bootstrap_p=boot["p_bootstrap"],
                bootstrap_n_test=boot["n_test_examples"],
                significant_5pct=sig5,
            )
            rows.append(row)

            print(f"  {label:18s} | {ds:5s} "
                  f"| meanΔ={mean_diff:+.4f} "
                  f"| t_p={t_p if not np.isnan(t_p) else 'NA'} "
                  f"| W_p={w_p if not np.isnan(w_p) else 'NA'} "
                  f"| bootΔ={boot['obs_diff']:+.4f} "
                  f"[{boot['ci_lo']:+.4f},{boot['ci_hi']:+.4f}] "
                  f"| boot_p={boot['p_bootstrap']:.4g} sig={sig5}")

            md_blocks.append(
                f"| {ds.upper()} | {mean_a:.4f} | {mean_b:.4f} "
                f"| {mean_diff:+.4f} "
                f"| t={t_stat:.3f}, p={t_p:.4g} "
                f"| W={w_stat:.3f}, p={w_p:.4g} "
                f"| {boot['obs_diff']:+.4f} "
                f"| [{boot['ci_lo']:+.4f}, {boot['ci_hi']:+.4f}] "
                f"| {boot['p_bootstrap']:.4g} "
                f"| {'✓' if sig5 else '×'} |"
            )

    df = pd.DataFrame(rows)
    out_csv = RESULTS_DIR / "significance_seeds.csv"
    save_csv(df, out_csv)
    print(f"\n  wrote {out_csv.relative_to(RESULTS_DIR.parent)}")

    out_md = RESULTS_DIR / "significance_seeds.md"
    out_md.write_text("\n".join(md_blocks) + "\n", encoding="utf-8")
    print(f"  wrote {out_md.relative_to(RESULTS_DIR.parent)}")


if __name__ == "__main__":
    main()
