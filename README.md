# Vec2Tec — A Semantically Enhanced Word2Vec Framework for Adaptive Sentiment Intelligence

Sentiment analysis experimental framework producing **measurable, reproducible
results** for English (IMDb) and Arabic (LABR book reviews + ASTD tweets).
This repository is the official code release accompanying the research paper
of the same name. The code is organised as one Python script per step and is
designed to run in **VS Code** under Python 3.10+.

## Paper

**Vec2Tec: A Semantically Enhanced Word2Vec Framework for Adaptive Sentiment
Intelligence**
Ismail Zama'reh, Jihad Turman, Mohammad Sawad (2026)

The manuscript itself is not distributed in this repository. If you use this
code or build on this work, please cite it — see [`CITATION.cff`](CITATION.cff).

## Directory layout

```
vec2tec/
├── data/
│   ├── raw/                       (not used - raw files live in ../EN Data / ../AR Data)
│   └── processed/                 cleaned CSV outputs
│       ├── imdb_train.csv
│       ├── imdb_test.csv
│       ├── labr_train.csv
│       ├── labr_test.csv
│       ├── astd_train.csv
│       └── astd_test.csv
├── models/                        trained models
│   ├── tfidf_vectorizer_<ds>.pkl  TF-IDF vectoriser per dataset
│   ├── clf_tfidf_lr_<ds>.pkl      TF-IDF + LR
│   ├── clf_tfidf_svm_<ds>.pkl     TF-IDF + Linear SVM
│   ├── w2v_<ds>.model             gensim Word2Vec model
│   ├── clf_word2vec_lr_<ds>.pkl   Word2Vec + LR
│   ├── clf_word2vec_svm_<ds>.pkl  Word2Vec + SVM
│   ├── clf_vec2tec_lr_<ds>.pkl    Vec2Tec + LR
│   └── clf_vec2tec_svm_<ds>.pkl   Vec2Tec + SVM
├── results/
│   ├── dataset_validation.csv     output of Step 1
│   ├── metrics_per_model.csv      all metrics in one table (Step 6)
│   ├── confusion_matrices/        one CSV per (model, dataset)
│   ├── classification_reports/    one CSV per (model, dataset)
│   ├── comparison_imdb.csv        per-dataset comparison (Step 7)
│   ├── comparison_labr.csv
│   ├── comparison_astd.csv
│   ├── lex_<ds>.json              corpus-derived sentiment lexicon
│   ├── summary.csv                best model per dataset (Step 8)
│   └── summary.md                 final summary + research-question answers
└── src/
    ├── config.py                  paths + hyperparameters
    ├── utils.py                   IO helpers + timer + label_distribution
    ├── validate_data.py           Step 1 - check & validate raw datasets
    ├── preprocess.py              Step 2 - clean text -> CSV
    ├── baselines.py               Step 3 - TF-IDF + LR / SVM
    ├── train_word2vec.py          Step 4 - Word2Vec embedding + LR / SVM
    ├── vec2tec_module.py          Step 5 - modular Vec2Tec framework
    ├── evaluate.py                Step 6 - metrics + confusion + reports
    ├── compare.py                 Step 7 - per-dataset comparison tables
    ├── summarize.py               Step 8 - final summary + Q&A
    └── run_all.py                 convenience: run every step in order
```

## Required raw data (NOT redistributed)

The scripts expect the user-supplied raw data in the sibling folders:

```
<New folder>/
├── EN Data/                       IMDb (aclImdb-style)
│   ├── train/{pos,neg}/*.txt
│   ├── test/{pos,neg}/*.txt
│   ├── imdb.vocab
│   └── README
└── AR Data/
    ├── reviews.tsv                LABR raw reviews
    ├── 2class-balanced-train.txt  LABR official binary split (line indices)
    ├── 2class-balanced-test.txt
    └── Tweets.txt                 ASTD labelled tweets
```

## Running the pipeline

### One-shot

```bash
cd vec2tec
python src/run_all.py
```

### Step by step (recommended for iteration in VS Code)

```bash
python src/validate_data.py        # Step 1
python src/preprocess.py           # Step 2  (add --full to disable sampling)
python src/baselines.py            # Step 3
python src/train_word2vec.py       # Step 4
python src/vec2tec_module.py       # Step 5
python src/evaluate.py             # Step 6
python src/compare.py              # Step 7
python src/summarize.py            # Step 8
```

Every step is also runnable for a single dataset via `--dataset {imdb|labr|astd}`.

## Dependencies

Python 3.10+. Install everything with:

```bash
pip install -r requirements.txt
```

## How Vec2Tec is wired (Step 5)

`vec2tec_module.py` defines an `Enhancement` interface with concrete
implementations:

* `SentimentLexiconEnhancement` — adaptive pooling by `|z(w)|`
* `PolarityRetrofitEnhancement` — closed-form retrofit along p_pos − p_neg axis
* `SynonymExpansionEnhancement` — **stub** (raises `NotImplementedError`)
* `ContextualWeightingEnhancement` — **stub**
* `DomainAdaptationEnhancement` — **stub**
* `SenseAwareWeightingEnhancement` — **stub**

To add a new enhancement, subclass `Enhancement`, implement `transform(...)`,
and pass an instance to the `Vec2Tec` constructor. The rest of the pipeline
is untouched.

To run a pure baseline that bypasses every enhancement (= standard Word2Vec
representation) pass `--plain`:

```bash
python src/vec2tec_module.py --plain --dataset astd
```

## Outputs the pipeline produces

After running all steps you get:

1. **Cleaned datasets** — `data/processed/*.csv`
2. **Trained models** — `models/*.pkl`, `models/*.model`
3. **Evaluation reports** — `results/metrics_per_model.csv`
4. **Confusion matrices** — `results/confusion_matrices/*.csv`
5. **Classification reports** — `results/classification_reports/*.csv`
6. **Comparison tables** — `results/comparison_{imdb,labr,astd}.csv`
7. **Summary file** — `results/summary.md` + `results/summary.csv`

All of these are plain CSV / Markdown / JSON files — easy to inspect in VS
Code, Excel, or by `pandas.read_csv`.

## Reproducibility notes

* `RANDOM_SEED = 42` is applied to splits and Word2Vec.
* Sub-sampling is **on** by default (see `USE_SAMPLING` in `config.py`) so
  the full pipeline finishes in a few minutes on CPU. Pass `--full` to
  `preprocess.py` (and disable `USE_SAMPLING`) to process all available data.
* LABR is loaded via the **official** `2class-balanced-train.txt` /
  `2class-balanced-test.txt` files (0-indexed line numbers into
  `reviews.tsv`, verified empirically).
* ASTD's `2class-balanced-*.txt` files reference an external ID space not
  bundled with `Tweets.txt`, so for ASTD we build a stratified 80/20 split
  from `Tweets.txt` directly.

## What's in this repository

`data/processed/` (cleaned, pre-split CSVs) and `models/` (trained
vectorizers, classifiers, and Word2Vec embeddings) are tracked in this
repository as the exact artifacts used to produce the results reported in
the paper. They can also be regenerated from scratch via
`python src/run_all.py` given the raw IMDb/LABR/ASTD inputs described above.
Note that `data/processed/*` is *derived* from those raw datasets, not a
redistribution of the raw files themselves (which remain outside this repo).
`results/` (metrics, confusion matrices, classification reports, figures,
tables, final reports) is tracked as the reported findings themselves.
Large cached prediction dumps (`results/_preds_*.pkl`) are git-ignored since
they are just intermediate scratch files.

## License

Code is released under the [MIT License](LICENSE). See [`CITATION.cff`](CITATION.cff)
for how to cite the accompanying paper.
