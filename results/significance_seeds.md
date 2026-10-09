# Across-seed significance tests

Comparisons are performed on binary datasets only (imdb, labr, astd). Three seeds — paired t and Wilcoxon are reported for completeness but have very low power with n=3 and should be read alongside the paired-bootstrap result, which uses 2000 resamples of the per-example test-set predictions.


## vec2tec_vs_w2v: `vec2tec_lr` vs `word2vec_lr`

| dataset | mean A | mean B | mean Δ | paired t (p) | Wilcoxon (p) | bootstrap Δ obs | bootstrap 95% CI | bootstrap p | sig (5%) |
|:--------|------:|------:|------:|:------------|:------------|:---------------|:-----------------|:-----------|:-------:|
| IMDB | 0.8692 | 0.8586 | +0.0106 | t=41.517, p=0.0005796 | W=0.000, p=0.25 | +0.0111 | [+0.0085, +0.0139] | 0 | ✓ |
| LABR | 0.7890 | 0.7750 | +0.0140 | t=3.757, p=0.06412 | W=0.000, p=0.25 | +0.0116 | [+0.0044, +0.0192] | 0 | ✓ |
| ASTD | 0.6060 | 0.5988 | +0.0072 | t=0.943, p=0.445 | W=3.000, p=1 | +0.0225 | [+0.0028, +0.0434] | 0.027 | ✓ |

## tfidf_vs_vec2tec: `tfidf_lr` vs `vec2tec_lr`

| dataset | mean A | mean B | mean Δ | paired t (p) | Wilcoxon (p) | bootstrap Δ obs | bootstrap 95% CI | bootstrap p | sig (5%) |
|:--------|------:|------:|------:|:------------|:------------|:---------------|:-----------------|:-----------|:-------:|
| IMDB | 0.9024 | 0.8692 | +0.0332 | t=60.901, p=0.0002695 | W=0.000, p=0.25 | +0.0331 | [+0.0293, +0.0368] | 0 | ✓ |
| LABR | 0.8336 | 0.7890 | +0.0447 | t=161.387, p=3.839e-05 | W=0.000, p=0.25 | +0.0443 | [+0.0320, +0.0568] | 0 | ✓ |
| ASTD | 0.7414 | 0.6060 | +0.1354 | t=17.045, p=0.003424 | W=0.000, p=0.25 | +0.1352 | [+0.0803, +0.1887] | 0 | ✓ |
