"""
Step 5 - Vec2Tec extension module (placeholder + extension hooks).

This file defines a modular framework that wraps a standard Word2Vec
model and exposes well-defined *hooks* where future semantic enhancement
techniques can be plugged in without touching the rest of the pipeline:

    1. sentiment_lexicon      - inject corpus-derived or external polarity scores
    2. synonym_expansion      - bring synonyms / paraphrases closer in space
    3. contextual_weighting   - re-weight word vectors per document context
    4. domain_adaptation      - shift the embedding space toward the active domain
    5. sense_aware_weighting  - assign multiple polarity-conditioned weights
                                to a single surface form

The default Vec2Tec configuration (`enhancements=[]`) is mathematically
identical to standard Word2Vec averaged document vectors, which makes
the first experimental row a sanity check that the framework introduces
no regression.

Every enhancement is implemented as a subclass of `Enhancement` with a
`transform(word_vectors, lexicon, document_tokens) -> np.ndarray` method
that returns the *document* vector.  This way the rest of the pipeline
calls `Vec2Tec.document_vector(...)` and never has to know which
enhancements are active.

Default we ship with TWO concrete enhancements wired up:

    * SentimentLexiconEnhancement (corpus log-odds-ratio)
    * AdaptivePolarityPoolingEnhancement (re-weight tokens by |z(w)|)

Adding more (synonyms, sense disambiguation, ...) is a matter of writing
a new subclass and appending it to the `enhancements` list.

Usage:
    python src/vec2tec_module.py                # run Vec2Tec on all datasets
    python src/vec2tec_module.py --dataset imdb
    python src/vec2tec_module.py --plain        # baseline = vanilla W2V
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors, Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from config import (
    PROCESSED_DIR, MODELS_DIR, RESULTS_DIR, DATASETS, DATASETS_4CLASS, W2V_DIM,
)
from utils import banner, load_csv, save_pickle, save_json


ALL_DATASETS = tuple(DATASETS) + tuple(DATASETS_4CLASS)


# ===========================================================================
#  Enhancement interface
# ===========================================================================
class Enhancement(ABC):
    """Abstract hook applied to a (word_vector, lexicon, doc) tuple."""

    name: str = "abstract"

    @abstractmethod
    def transform(
        self,
        tokens: List[str],
        word_vectors: KeyedVectors,
        lexicon: Dict[str, float],
        dim: int,
    ) -> np.ndarray:
        """Return the document vector for `tokens`."""
        raise NotImplementedError


# ===========================================================================
#  Default enhancement #1 - corpus-derived sentiment lexicon weighting
# ===========================================================================
class SentimentLexiconEnhancement(Enhancement):
    """Re-weights each token by  1 + beta * |z(w)|  (adaptive pooling).

    The lexicon is computed in :func:`build_sentiment_lexicon` from the
    training corpus using log-odds-ratio with informative Dirichlet prior
    (Monroe et al., 2008).  A token's absolute polarity score becomes its
    soft attention weight in the mean-pooling layer, so sentiment-bearing
    words contribute more strongly than function words.
    """

    name = "sentiment_lexicon"

    def __init__(self, beta: float = 0.5):
        self.beta = float(beta)

    def transform(self, tokens, word_vectors, lexicon, dim):
        if not tokens:
            return np.zeros(dim, dtype=np.float32)
        vecs, weights = [], []
        for t in tokens:
            if t in word_vectors:
                vecs.append(word_vectors[t])
                weights.append(1.0 + self.beta * abs(lexicon.get(t, 0.0)))
        if not vecs:
            return np.zeros(dim, dtype=np.float32)
        arr = np.vstack(vecs)
        w = np.asarray(weights, dtype=np.float32).reshape(-1, 1)
        return (arr * w).sum(axis=0) / w.sum()


# ===========================================================================
#  Default enhancement #2 - polarity-axis embedding retrofit
# ===========================================================================
class PolarityRetrofitEnhancement(Enhancement):
    """Shifts each word vector along a learned polarity axis.

        E'[w] = (1 - alpha) * E[w]  +  alpha * tanh(z(w)/tau) * (p_pos - p_neg)

    where p_pos / p_neg are the mean embeddings of the top-K positive
    and top-K negative words by polarity score z(w).  The shift is
    re-computed once per dataset and cached on the instance.
    """

    name = "polarity_retrofit"

    def __init__(self, alpha: float = 0.3, tau: float = 6.0, top_k: int = 100):
        self.alpha = float(alpha)
        self.tau   = float(tau)
        self.top_k = int(top_k)
        self._cache: Dict[str, np.ndarray] = {}
        self._axis: Optional[np.ndarray] = None

    def fit(self, wv: KeyedVectors, lexicon: Dict[str, float]) -> None:
        """Pre-compute the polarity axis from the corpus lexicon."""
        items = sorted(lexicon.items(), key=lambda kv: kv[1])
        neg_words = [w for w, _ in items            if w in wv][: self.top_k]
        pos_words = [w for w, _ in items[::-1]      if w in wv][: self.top_k]
        if not pos_words or not neg_words:
            self._axis = np.zeros(wv.vector_size, dtype=np.float32)
            return
        p_pos = np.mean([wv[w] for w in pos_words], axis=0)
        p_neg = np.mean([wv[w] for w in neg_words], axis=0)
        self._axis = (p_pos - p_neg).astype(np.float32)
        # pre-shift the vocabulary
        for w in wv.index_to_key:
            z = lexicon.get(w, 0.0)
            shift = math.tanh(z / self.tau)
            self._cache[w] = (1 - self.alpha) * wv[w] + self.alpha * shift * self._axis

    def transform(self, tokens, word_vectors, lexicon, dim):
        if self._axis is None:
            self.fit(word_vectors, lexicon)
        if not tokens:
            return np.zeros(dim, dtype=np.float32)
        vecs = [self._cache[t] for t in tokens if t in self._cache]
        if not vecs:
            return np.zeros(dim, dtype=np.float32)
        return np.mean(vecs, axis=0)


# ===========================================================================
#  Placeholders for future enhancements (left as stubs)
# ===========================================================================
class SynonymExpansionEnhancement(Enhancement):
    """Replaces every token by the mean of itself and its top_k nearest
    neighbours in embedding space (a cheap proxy for synonym/paraphrase
    smoothing). Neighbour lookups are cached so the cost is paid once
    per vocabulary item, not once per document.
    """

    name = "synonym_expansion"

    def __init__(self, top_k: int = 3):
        self.top_k = int(top_k)
        self._cache: Dict[str, np.ndarray] = {}

    def _expand(self, w: str, wv: KeyedVectors, dim: int) -> np.ndarray:
        if w in self._cache:
            return self._cache[w]
        if w not in wv:
            self._cache[w] = np.zeros(dim, dtype=np.float32)
            return self._cache[w]
        vecs = [wv[w]]
        try:
            for nb, _ in wv.most_similar(w, topn=self.top_k):
                vecs.append(wv[nb])
        except (KeyError, ValueError):
            pass
        self._cache[w] = np.mean(vecs, axis=0).astype(np.float32)
        return self._cache[w]

    def transform(self, tokens, word_vectors, lexicon, dim):
        if not tokens:
            return np.zeros(dim, dtype=np.float32)
        vecs = [self._expand(t, word_vectors, dim) for t in tokens
                if t in word_vectors]
        if not vecs:
            return np.zeros(dim, dtype=np.float32)
        return np.mean(vecs, axis=0)


class ContextualWeightingEnhancement(Enhancement):
    """Weights each token by a softmax over its cosine similarity to the
    document centroid:
        a_i = softmax_tau(cos(E[w_i], centroid))
    so context-central words dominate the pooled representation.
    """

    name = "contextual_weighting"

    def __init__(self, tau: float = 1.0):
        self.tau = float(tau)

    def transform(self, tokens, word_vectors, lexicon, dim):
        if not tokens:
            return np.zeros(dim, dtype=np.float32)
        vecs = [word_vectors[t] for t in tokens if t in word_vectors]
        if not vecs:
            return np.zeros(dim, dtype=np.float32)
        arr = np.vstack(vecs).astype(np.float32)
        centroid = arr.mean(axis=0)
        # cosine similarity to centroid
        norms = np.linalg.norm(arr, axis=1) * np.linalg.norm(centroid) + 1e-9
        sims = (arr @ centroid) / norms
        # numerically stable softmax(sims / tau)
        z = sims / max(self.tau, 1e-6)
        z = z - z.max()
        w = np.exp(z)
        w = w / w.sum()
        return (arr * w.reshape(-1, 1)).sum(axis=0)


class DomainAdaptationEnhancement(Enhancement):
    """STUB: project embeddings onto a domain-specific subspace."""
    name = "domain_adaptation"

    def transform(self, tokens, word_vectors, lexicon, dim):
        raise NotImplementedError("DomainAdaptationEnhancement not implemented yet")


class SenseAwareWeightingEnhancement(Enhancement):
    """STUB: support multiple polarity-conditioned vectors per surface form."""
    name = "sense_aware"

    def transform(self, tokens, word_vectors, lexicon, dim):
        raise NotImplementedError("SenseAwareWeightingEnhancement not implemented yet")


# ===========================================================================
#  Vec2Tec orchestrator
# ===========================================================================
class Vec2Tec:
    """Wrap a Word2Vec model + an ordered list of enhancements.

    The class is intentionally tiny - it is the *integration point*
    between the standard Word2Vec ancestor and any future semantic
    enhancement.  To extend Vec2Tec, write a new Enhancement subclass
    and pass it to the constructor; nothing else changes.
    """

    def __init__(
        self,
        wv: KeyedVectors,
        lexicon: Dict[str, float] | None = None,
        enhancements: List[Enhancement] | None = None,
        dim: int = W2V_DIM,
    ):
        self.wv = wv
        self.dim = dim
        self.lexicon = lexicon or {}
        self.enhancements = enhancements or []

    # ------- vector builders ------------------------------------------------
    def _baseline_doc(self, tokens: List[str]) -> np.ndarray:
        """Mean of word vectors -- the standard W2V representation."""
        if not tokens:
            return np.zeros(self.dim, dtype=np.float32)
        vecs = [self.wv[t] for t in tokens if t in self.wv]
        if not vecs:
            return np.zeros(self.dim, dtype=np.float32)
        return np.mean(vecs, axis=0)

    def document_vector(self, tokens: List[str]) -> np.ndarray:
        """Document representation: average across all active enhancements."""
        if not self.enhancements:
            return self._baseline_doc(tokens)
        out = np.zeros(self.dim, dtype=np.float32)
        for enh in self.enhancements:
            out += enh.transform(tokens, self.wv, self.lexicon, self.dim)
        return out / len(self.enhancements)

    # ------- fitting helpers ------------------------------------------------
    def fit(self) -> None:
        for enh in self.enhancements:
            if hasattr(enh, "fit"):
                enh.fit(self.wv, self.lexicon)


# ===========================================================================
#  Sentiment lexicon builder (corpus log-odds-ratio with Dirichlet prior)
# ===========================================================================
def build_sentiment_lexicon(
    df: pd.DataFrame, min_count: int = 5, alpha_prior: float = 0.01,
) -> Dict[str, float]:
    train = df[df.get("split", "train") == "train"] if "split" in df.columns else df
    pos_c, neg_c = Counter(), Counter()
    for _, row in train.iterrows():
        toks = str(row["text"]).split()
        # POS (1) and NEG (0) define the polarity axis. For multi-class
        # datasets, NEUTRAL (2) and OBJ (3) carry no polarity signal and
        # are deliberately excluded so they do not bias the log-odds.
        if row["label"] == 1:
            pos_c.update(toks)
        elif row["label"] == 0:
            neg_c.update(toks)
    n_pos = sum(pos_c.values()); n_neg = sum(neg_c.values())
    lex: Dict[str, float] = {}
    for w in (set(pos_c) | set(neg_c)):
        if pos_c[w] + neg_c[w] < min_count:
            continue
        yp, yn = pos_c[w], neg_c[w]
        delta = (math.log((yp + alpha_prior) / (n_pos - yp + alpha_prior))
               - math.log((yn + alpha_prior) / (n_neg - yn + alpha_prior)))
        sigma2 = 1.0 / (yp + alpha_prior) + 1.0 / (yn + alpha_prior)
        lex[w] = delta / math.sqrt(sigma2)
    return lex


# ===========================================================================
#  Runner
# ===========================================================================
def tokenize(text: str) -> List[str]:
    return str(text).split()


def run_one(ds: str, plain: bool = False) -> dict:
    banner(f"Vec2Tec on {ds.upper()}  (plain={plain})")
    train = load_csv(PROCESSED_DIR / f"{ds}_train.csv")
    test  = load_csv(PROCESSED_DIR / f"{ds}_test.csv")
    tr_tok = [tokenize(t) for t in train["text"].fillna("")]
    te_tok = [tokenize(t) for t in test["text"].fillna("")]

    if ds in DATASETS_4CLASS:
        val_path = PROCESSED_DIR / f"{ds}_validation.csv"
        if val_path.exists():
            val = load_csv(val_path)
            print(f"  validation set available ({len(val)} rows) — not used "
                  "for tuning yet")
        n_classes = len(set(train["label"].tolist()) | set(test["label"].tolist()))
        print(f"  multi-class run (n_classes={n_classes})")

    # load the standard Word2Vec model produced in Step 4
    w2v_path = MODELS_DIR / f"w2v_{ds}.model"
    if not w2v_path.exists():
        sys.exit(f"ERROR: {w2v_path} not found - run train_word2vec.py first.")
    w2v = Word2Vec.load(str(w2v_path))

    # build corpus lexicon
    lex = build_sentiment_lexicon(train.assign(split="train"))
    save_json(lex, RESULTS_DIR / f"lex_{ds}.json")
    print(f"  sentiment lexicon size: {len(lex)} terms")

    # configure enhancements
    if plain:
        enhancements: List[Enhancement] = []
    else:
        enhancements = [
            PolarityRetrofitEnhancement(alpha=0.3, tau=6.0, top_k=100),
            SentimentLexiconEnhancement(beta=0.5),
        ]
    vec2tec = Vec2Tec(w2v.wv, lex, enhancements, dim=W2V_DIM)
    vec2tec.fit()

    # build document vectors
    t = time.time()
    Xtr = np.vstack([vec2tec.document_vector(t) for t in tr_tok])
    Xte = np.vstack([vec2tec.document_vector(t) for t in te_tok])
    print(f"  doc vectors built in {time.time()-t:.2f}s")

    out: dict = {}
    ytr, yte = train["label"].values, test["label"].values

    # LR
    t = time.time()
    lr_solver = "lbfgs" if ds in DATASETS_4CLASS else "liblinear"
    lr = LogisticRegression(max_iter=1000, C=2.0, n_jobs=1,
                            class_weight="balanced", solver=lr_solver)
    lr.fit(Xtr, ytr)
    model_label = "vec2tec_lr" if not plain else "vec2tec_plain_lr"
    out[model_label] = dict(
        model_name=model_label,
        train_time_s=time.time() - t,
        predictions=lr.predict(Xte).tolist(),
        y_true=yte.tolist(),
    )
    save_pickle(lr, MODELS_DIR / f"clf_{model_label}_{ds}.pkl")
    print(f"  {model_label} trained in {out[model_label]['train_time_s']:.2f}s")

    # SVM
    t = time.time()
    svm = LinearSVC(C=1.0, class_weight="balanced")
    svm.fit(Xtr, ytr)
    model_label2 = "vec2tec_svm" if not plain else "vec2tec_plain_svm"
    out[model_label2] = dict(
        model_name=model_label2,
        train_time_s=time.time() - t,
        predictions=svm.predict(Xte).tolist(),
        y_true=yte.tolist(),
    )
    save_pickle(svm, MODELS_DIR / f"clf_{model_label2}_{ds}.pkl")
    print(f"  {model_label2} trained in {out[model_label2]['train_time_s']:.2f}s")

    tag = "vec2tec_plain" if plain else "vec2tec"
    save_pickle(out, RESULTS_DIR / f"_preds_{tag}_{ds}.pkl")
    return out


def main(only: str | None, plain: bool) -> None:
    targets = (only,) if only else DATASETS
    for ds in targets:
        run_one(ds, plain=plain)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=ALL_DATASETS, default=None)
    p.add_argument("--plain", action="store_true",
                   help="Disable all enhancements (= vanilla W2V baseline).")
    args = p.parse_args()
    main(args.dataset, args.plain)
