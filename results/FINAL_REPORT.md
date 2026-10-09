# Vec2Tec — final experimental summary

This report consolidates the entire experimental campaign — the original binary results, the multi-seed reproducibility study, the 4-class ASTD extension, the ablation study, across-seed significance tests, and the transformer baseline comparison. Numbers below are **mean ± std over 3 random seeds (42, 123, 2024)** unless stated otherwise.

## TL;DR

- **Best overall**: TF-IDF + LR. It wins macro-F1 on every dataset (binary and 4-class).
- **Vec2Tec significantly improves over Word2Vec** on every binary dataset (paired bootstrap, p < 0.05; macro-F1 gain ≈ +1.1 pp on IMDb, +1.4 pp on LABR, +0.7 pp on ASTD).
- **Vec2Tec does NOT beat TF-IDF on any dataset** (paired bootstrap, p < 0.05 in the opposite direction; gap ranges from +2.65 pp on IMDb to +12.09 pp on ASTD4).
- **The frozen multilingual transformer wins only on short noisy social text** (ASTD, ASTD4). On long well-resourced text (IMDb, LABR) Vec2Tec is the better lightweight model than the frozen transformer.
- The Vec2Tec **ablation** shows the gains come from the **sentiment-lexicon weighting** + **polarity retrofit** combination. Synonym expansion HURTS on long text but HELPS dramatically on short noisy ASTD (+9.6 pp F1) — a dataset-conditional finding worth flagging in any future paper.
- **ASTD remains the hardest binary dataset** (best F1 = 0.7414); ASTD-4-class is at 0.4728, well above 0.25 random but clearly the limit of bag-of-words + shallow embeddings.

## 1. Which model performed best overall?

By macro-F1 averaged across seeds, the best model is **TF-IDF + LR** on every dataset.

| dataset   | best_model   | best_family   |   best_f1_macro_mean |   best_f1_macro_std |   best_accuracy_mean | best_w2v_based_model   |   best_w2v_based_f1_macro_mean |   best_w2v_based_f1_macro_std |   gap_best_minus_w2v_pp |
|:----------|:-------------|:--------------|---------------------:|--------------------:|---------------------:|:-----------------------|-------------------------------:|------------------------------:|------------------------:|
| astd      | tfidf_lr     | TF-IDF        |             0.741428 |             0.00485 |             0.777328 | vec2tec_svm            |                       0.623701 |                      0.009721 |                 11.7727 |
| astd4     | tfidf_lr     | TF-IDF        |             0.472772 |             0       |             0.471698 | vec2tec_svm            |                       0.351904 |                      0.014766 |                 12.0868 |
| imdb      | tfidf_lr     | TF-IDF        |             0.9024   |             0       |             0.9024   | vec2tec_svm            |                       0.875907 |                      0.000751 |                  2.6493 |
| labr      | tfidf_lr     | TF-IDF        |             0.833643 |             0       |             0.833647 | vec2tec_lr             |                       0.788972 |                      0.000479 |                  4.4671 |

## 2. Did Vec2Tec improve over Word2Vec?

Yes, consistently — and the improvement is statistically significant under a paired bootstrap on per-example test predictions (B = 2,000).

| dataset | best W2V (mean F1) | best Vec2Tec (mean F1) | Δ F1 (pp) | bootstrap p-value | significant @ 5% |
|:--------|-------------------:|-----------------------:|----------:|:-----------------|:----------------:|
| IMDB | word2vec_svm (0.8607) | vec2tec_svm (0.8759) | +1.52 | 0 | ✓ |
| LABR | word2vec_lr (0.7750) | vec2tec_lr (0.7890) | +1.40 | 0 | ✓ |
| ASTD | word2vec_svm (0.6172) | vec2tec_svm (0.6237) | +0.65 | 0.027 | ✓ |

## 3. Did Vec2Tec outperform TF-IDF?

No. TF-IDF + LR is significantly stronger than Vec2Tec on every dataset under the same paired-bootstrap test.

| dataset | best TF-IDF (mean F1) | best Vec2Tec (mean F1) | Δ F1 (pp, TFIDF − V2T) | bootstrap p | significant @ 5% |
|:--------|---------------------:|-----------------------:|----------------------:|:------------|:----------------:|
| IMDB | tfidf_lr (0.9024) | vec2tec_svm (0.8759) | +2.65 | 0 | ✓ |
| LABR | tfidf_lr (0.8336) | vec2tec_lr (0.7890) | +4.47 | 0 | ✓ |
| ASTD | tfidf_lr (0.7414) | vec2tec_svm (0.6237) | +11.77 | 0 | ✓ |

## 4. Which dataset was hardest?

By the **best** F1 any model achieved:
1. ASTD-4-class — best F1 = 0.4728 (TF-IDF + LR)
2. ASTD (binary) — best F1 = 0.7414 (TF-IDF + LR)
3. LABR — best F1 = 0.8336 (TF-IDF + LR)
4. IMDb — best F1 = 0.9024 (TF-IDF + LR)

## 5. Why is ASTD harder than IMDb and LABR?

Three structural reasons, all derivable from `tables/dataset_stats_v2.csv`:
- **Vocabulary scarcity**: ASTD train vocab = 12,273 vs LABR 86,677 and IMDb 159,209 (≈ 13× smaller).
- **Text length**: median 16 tokens (tweets) vs LABR 31 and IMDb 173. Short text gives the classifier very few sentiment-bearing features per example.
- **Class imbalance (in the raw corpus)**: the binary 80/20 split we use to build ASTD is stratified but starts from a 1,684 NEG / 799 POS pool, so the minority class is small in absolute terms; the per-class analysis confirms POS is the harder class on binary ASTD (mean F1 = 0.5360 across all 6 models). For ASTD-4-class, NEG is the hardest class (mean F1 = 0.2373) — it gets confused with NEUTRAL most often.
- **Domain noise**: tweets have hashtags, mentions, code-switching, and non-standard orthography. Even after `clean_ar()` they remain harder than book reviews.

## 6. Does this support the research hypothesis “semantic enhancement improves Word2Vec, especially for Arabic and short noisy text”?

Partially.

**Yes — improvement over Word2Vec is real and statistically significant.** All three binary datasets show Vec2Tec > Word2Vec under the paired bootstrap (p < 0.05). The ablation study isolates the lexicon-weighting + polarity-retrofit pair as the components doing the work.

**Partially — the Arabic / short-text gain is dataset-dependent.** The ablation showed +9.6 pp F1 from synonym expansion on ASTD alone (most noisy / sparsest vocab), but synonym expansion HURTS on the larger, vocabulary-rich datasets (−1.4 pp on IMDb, −1.5 pp on LABR). The default Vec2Tec stack picks the wrong components for ASTD; a dataset-aware enhancement selector would be the obvious next step.

**No — Vec2Tec does not displace TF-IDF.** TF-IDF + LR is stronger on every dataset by a margin that is unambiguous (bootstrap p ≈ 0 on all three binary datasets).


## 7. How does Vec2Tec compare to a frozen multilingual transformer?

Encoder: `paraphrase-multilingual-MiniLM-L12-v2` (no fine-tuning), LR head. IMDb was sub-sampled (5K train / 5K test) due to CPU encoding cost; LABR / ASTD / ASTD4 used the full split.

| dataset | best Vec2Tec F1 | transformer F1 | who wins? |
|:--------|----------------:|---------------:|:----------|
| IMDb (5K subsample) | 0.8759 | 0.7754 | **Vec2Tec** wins (+10.05 pp) |
| LABR | 0.7890 | 0.7517 | **Vec2Tec** wins (+3.73 pp) |
| ASTD | 0.6237 | 0.7585 | **Transformer** wins (-13.48 pp) |
| ASTD4 | 0.3519 | 0.4635 | **Transformer** wins (-11.16 pp) |

Interpretation: the frozen multilingual model wins on short noisy social text (ASTD/ASTD4), where Word2Vec's small in-domain vocabulary is the bottleneck. On long well-resourced text (IMDb, LABR) the multilingual encoder underperforms a domain-adapted Vec2Tec + LR — the encoder's capacity is split across ~50 languages and is not specialised for either English movie reviews or Arabic book reviews. This positions Vec2Tec as a credible, lightweight, **interpretable** alternative when a domain lexicon is available and inference cost matters.


## 8. Ablation: what inside Vec2Tec is doing the work?

Single-seed run (seed = 2024). The underlying Word2Vec model and corpus lexicon are held fixed; only the enhancement stack varies. Δ columns are vs. `word2vec_plain` (mean of word vectors, no enhancement) on the SAME dataset.


_Detailed per-dataset table: see `results/tables/ablation_results.md`._

Summary findings:
- Lexicon weighting alone gives most of Vec2Tec's IMDb / LABR gain (+0.76 pp / +0.97 pp F1).
- Polarity retrofit on top of lexicon (the `v2t_full` stack) adds a smaller but consistent further gain on long text (+1.11 pp F1 on IMDb, +1.19 pp on LABR vs. `word2vec_plain`).
- **Synonym expansion** is the most volatile knob: it HURTS on IMDb (−1.40 pp) and LABR (−1.48 pp) but is the SINGLE strongest enhancement on ASTD (**+9.58 pp F1**, +7.69 pp accuracy). On sparse-vocab data, expanding tokens to their semantic neighbours adds usable signal; on dense-vocab data it adds noise.
- Contextual weighting (cosine-softmax pooling) is essentially neutral everywhere we tested (Δ within ±0.2 pp). It is the weakest of the four enhancements in this configuration.


## 9. Per-class performance

Classification reports for every (model, dataset) are in `results/classification_reports/*.csv`; tidy long-format view in `results/tables/per_class_long.csv`; Vec2Tec(LR) vs Word2Vec(LR) per-class delta in `results/tables/per_class_vec2tec_vs_w2v.csv`.


Hardest class per dataset (lowest mean F1 across all six models):

| dataset   | hardest_class   |   hardest_class_mean_f1 | easiest_class   |   easiest_class_mean_f1 |
|:----------|:----------------|------------------------:|:----------------|------------------------:|
| astd      | positive        |                  0.536  | negative        |                  0.7768 |
| astd4     | negative        |                  0.2373 | objective       |                  0.4399 |
| imdb      | positive        |                  0.8765 | negative        |                  0.8769 |
| labr      | negative        |                  0.7965 | positive        |                  0.7974 |

## 10. Limitations and what should change in v2 of Vec2Tec

- **Three seeds is the floor of useful reproducibility.** The paired t-test across seeds is informative but underpowered; the Wilcoxon hits its minimum p = 0.25 with n = 3. We rely on the paired bootstrap of per-example predictions for the decisive test. A future v2 should run ≥ 5 seeds.
- **The `synonym_expansion` enhancement is on by default but harmful on long text.** Vec2Tec should expose enhancement selection as a dataset-aware hyper-parameter, ideally chosen on the validation split (ASTD4's official validation set already exists but is not used for tuning yet).
- **No fine-tuned transformer.** Our transformer baseline is intentionally frozen / CPU-only. A fine-tuned AraBERT or RoBERTa would almost certainly out-perform every model in this report; Vec2Tec is positioned as a *lightweight, interpretable* alternative, not a replacement.
- **TF-IDF wins on every dataset.** Vec2Tec's value proposition must therefore lean on its interpretability (corpus-derived polarity lexicon, modular enhancement hooks) and its 200-dim dense representation (useful for nearest-neighbour analysis, transfer to new tasks, or downstream pipelines) — not on raw macro-F1.
- **IMDb transformer numbers are on a 5K/5K subsample** (CPU-only encoding made a full-50K run impractical). Numbers should be re-validated on the full split with a GPU before being cited in any paper.
- **The current `polarity_retrofit` axis uses only NEG vs POS** and therefore cannot explain NEUTRAL/OBJ on the 4-class task. A multi-class polarity space (one axis per class) is the next obvious enhancement.

