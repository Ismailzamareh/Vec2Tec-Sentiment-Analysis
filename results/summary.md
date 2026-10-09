# Vec2Tec - Experimental summary

## Best model per dataset (by macro-F1)

| dataset   | best_model   |   accuracy |   f1_macro |   f1_weighted |
|:----------|:-------------|-----------:|-----------:|--------------:|
| imdb      | tfidf_lr     |     0.9024 |     0.9024 |        0.9024 |
| labr      | tfidf_lr     |     0.8336 |     0.8336 |        0.8336 |
| astd      | tfidf_lr     |     0.7814 |     0.7444 |        0.7794 |


## Research questions - evidence-based answers

### Q1

```
Does Word2Vec perform better than TF-IDF baselines?
Word2Vec beats TF-IDF on 0/3 datasets.
  - IMDB: TF-IDF best=0.9024 (tfidf_lr)  |  W2V best=0.8602 (word2vec_svm)  ->  winner: TF-IDF
  - LABR: TF-IDF best=0.8336 (tfidf_lr)  |  W2V best=0.7778 (word2vec_lr)  ->  winner: TF-IDF
  - ASTD: TF-IDF best=0.7444 (tfidf_lr)  |  W2V best=0.6274 (word2vec_svm)  ->  winner: TF-IDF
```

### Q2

```
Does Arabic sentiment analysis behave differently from English?
  - mean macro-F1 (English, IMDb)  = 0.8767
  - mean macro-F1 (Arabic, LABR+ASTD) = 0.7267
  - gap (EN - AR) = +0.1500  (English is easier on average.)
```

### Q3

```
Are long reviews easier than short noisy tweets?
  - mean macro-F1 (IMDb+LABR, long text) = 0.8368
  - mean macro-F1 (ASTD, tweets)         = 0.6564
  - gap = +0.1805  (long text is easier on average.)
```

### Q4

```
Does Vec2Tec improve macro-F1 over Word2Vec?
  - improvement in 3/3 datasets, average gain = +0.0116
  - IMDB: W2V best macro-F1 = 0.8602  |  Vec2Tec best macro-F1 = 0.8758  |  delta = +0.0156
  - LABR: W2V best macro-F1 = 0.7778  |  Vec2Tec best macro-F1 = 0.7894  |  delta = +0.0116
  - ASTD: W2V best macro-F1 = 0.6274  |  Vec2Tec best macro-F1 = 0.6349  |  delta = +0.0075
```

### Q5

```
Can Vec2Tec balance performance, simplicity, interpretability and efficiency?
  - mean classifier train time, Vec2Tec  = 1.2967s
  - mean classifier train time, TF-IDF   = 0.9093s
  - Vec2Tec is implemented as a modular framework with explicit
    enhancement hooks (sentiment lexicon, polarity retrofit, ...)
    and a corpus-derived lexicon that can be audited by a human.
  - Trade-off: at the current training budget Vec2Tec improves
    over standard Word2Vec but does not yet match TF-IDF;
    see Q4 for the exact deltas.
```
