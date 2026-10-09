"""Vec2Tec - Central configuration."""
from __future__ import annotations
from pathlib import Path

# Path roots
ROOT          = Path(__file__).resolve().parent.parent
PARENT        = ROOT.parent

# Raw input folders provided by the user
RAW_EN_DIR    = PARENT / "EN Data"
RAW_AR_DIR    = PARENT / "AR Data"

# Project-managed folders
DATA_DIR      = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR    = ROOT / "models"
RESULTS_DIR   = ROOT / "results"
CM_DIR        = RESULTS_DIR / "confusion_matrices"
CR_DIR        = RESULTS_DIR / "classification_reports"

for p in (DATA_DIR, PROCESSED_DIR, MODELS_DIR, RESULTS_DIR, CM_DIR, CR_DIR):
    p.mkdir(parents=True, exist_ok=True)

# Dataset constants
DATASETS      = ("imdb", "labr", "astd")
DATASETS_4CLASS = ("astd4",)
NEG, POS      = 0, 1
LABEL_NAMES   = {NEG: "negative", POS: "positive"}
LABEL_NAMES_4 = {0: "negative", 1: "positive", 2: "neutral", 3: "objective"}
RANDOM_SEED   = 2024

# Word2Vec hyperparameters
W2V_DIM       = 200
W2V_WINDOW    = 5
W2V_MIN_COUNT = 2
W2V_EPOCHS    = 10
W2V_SG        = 1
W2V_WORKERS   = 7

# TF-IDF hyperparameters
TFIDF_NGRAM   = (1, 2)
TFIDF_MAX     = 30_000
TFIDF_MIN_DF  = 2

# Sub-sampling (set USE_SAMPLING=False to disable)
SAMPLE_IMDB_PER_CLASS       = 1000
SAMPLE_LABR_TRAIN_PER_CLASS = 3000
SAMPLE_LABR_TEST_PER_CLASS  = 750
USE_SAMPLING                = False
