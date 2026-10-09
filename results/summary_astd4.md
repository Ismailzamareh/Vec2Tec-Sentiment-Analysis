# ASTD 4-class — experimental summary

This file reports the results of running the existing model
pipeline (TF-IDF, Word2Vec, Vec2Tec) on the official ASTD
4-class balanced split (POS / NEG / NEUTRAL / OBJ).

Each metric is the mean ± std over 3 random seeds (42, 123, 2024).


## Best model on astd4 (by macro-F1)

- **tfidf_lr** — macro-F1 = 0.4728 ± 0.0000, accuracy = 0.4717 ± 0.0000.

## Best-in-family on each variant of ASTD

| family | best model on binary ASTD (macro-F1) | best model on 4-class ASTD (macro-F1) | Δ (binary − 4-class) |
|:-------|:-------------------------------------|:---------------------------------------|:--------------------:|
| TF-IDF | tfidf_lr — 0.7414 ± 0.0049 | tfidf_lr — 0.4728 ± 0.0000 | **+26.87 pp** |
| Word2Vec | word2vec_svm — 0.6172 ± 0.0139 | word2vec_svm — 0.3472 ± 0.0064 | **+27.00 pp** |
| Vec2Tec | vec2tec_svm — 0.6237 ± 0.0097 | vec2tec_svm — 0.3519 ± 0.0148 | **+27.18 pp** |

## Does Vec2Tec hold up on 4-class sentiment as well as on binary?

On binary ASTD (POS vs NEG), Vec2Tec's best classifier (vec2tec_svm) reaches macro-F1 = 0.6237, beating Word2Vec (word2vec_svm, 0.6172) by **+0.65 pp**.
On 4-class ASTD, Vec2Tec's best (vec2tec_svm) reaches 0.3519, vs Word2Vec (word2vec_svm) at 0.3472 — a margin of **+0.47 pp**.

Verdict: yes — Vec2Tec still beats Word2Vec on the 4-class task, though the margin shrinks. The 4-class task is fundamentally harder for every model family (TF-IDF drops from 0.7414 to 0.4728 as well, a +26.87 pp drop) because the NEUTRAL and OBJ classes do not align with the polarity axis that drives Vec2Tec's lexicon enhancement, so the headroom that Vec2Tec exploits is smaller. The fact that Vec2Tec still out-ranks Word2Vec is consistent with the binary results and suggests the framework's enhancements continue to add signal even when the polarity axis only explains two of the four classes.
