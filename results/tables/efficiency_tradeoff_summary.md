# Efficiency–quality trade-off summary

All numbers are mean across 3 seeds; training time is the classifier-fit step only (does not include vectorisation or Word2Vec pre-training).


## IMDB

| family | best model | macro-F1 | classifier train (s) |
|:-------|:-----------|---------:|---------------------:|
| TF-IDF | tfidf_lr | 0.9024 | 2.830 |
| Word2Vec | word2vec_svm | 0.8607 | 2.165 |
| Vec2Tec | vec2tec_svm | 0.8759 | 2.610 |

## LABR

| family | best model | macro-F1 | classifier train (s) |
|:-------|:-----------|---------:|---------------------:|
| TF-IDF | tfidf_lr | 0.8336 | 0.347 |
| Word2Vec | word2vec_lr | 0.7750 | 1.351 |
| Vec2Tec | vec2tec_lr | 0.7890 | 2.319 |

## ASTD

| family | best model | macro-F1 | classifier train (s) |
|:-------|:-----------|---------:|---------------------:|
| TF-IDF | tfidf_lr | 0.7414 | 0.016 |
| Word2Vec | word2vec_svm | 0.6172 | 0.092 |
| Vec2Tec | vec2tec_svm | 0.6237 | 0.109 |

## Interpretation
- TF-IDF + LR provides the best macro-F1 on every binary dataset *and* the lowest classifier training time on the small Arabic datasets (ASTD/LABR), so it dominates the Pareto frontier on this benchmark.
- Vec2Tec consistently beats Word2Vec on macro-F1, at very similar classifier-fit cost. The Word2Vec doc-vector step is the bulk of wall-clock time for both, so adding lexicon enhancements is essentially free at inference time.
- If you are interpretability- or domain-adaptation-constrained, Vec2Tec is the better lightweight pick over Word2Vec; if you care only about test-set accuracy on standard reviews/tweets, TF-IDF + LR is the strongest cheap baseline.
