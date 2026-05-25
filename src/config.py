"""
config.py - Central Configuration File
=======================================
All project-wide constants, paths, hyperparameters, and settings
are defined here. Import this module wherever configuration is needed
to ensure consistency across the entire project.
"""

import os

# ─────────────────────────────────────────────
# BASE DIRECTORY
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ─────────────────────────────────────────────
# DIRECTORY PATHS
# ─────────────────────────────────────────────
DATASET_DIR  = os.path.join(BASE_DIR, "dataset")
MODEL_DIR    = os.path.join(BASE_DIR, "models")
REPORT_DIR   = os.path.join(BASE_DIR, "reports")
LOG_DIR      = os.path.join(BASE_DIR, "logs")

# ─────────────────────────────────────────────
# FILE PATHS
# ─────────────────────────────────────────────
DATASET_PATH           = os.path.join(DATASET_DIR,  "emails.csv")
MODEL_PATH             = os.path.join(MODEL_DIR,     "phishing_model.pkl")
VECTORIZER_PATH        = os.path.join(MODEL_DIR,     "vectorizer.pkl")
SCALER_PATH            = os.path.join(MODEL_DIR,     "scaler.pkl")
CONFUSION_MATRIX_PATH  = os.path.join(REPORT_DIR,   "confusion_matrix.png")
ACCURACY_CHART_PATH    = os.path.join(REPORT_DIR,   "accuracy_chart.png")
FEATURE_IMP_PATH       = os.path.join(REPORT_DIR,   "feature_importance.png")
LOG_FILE_PATH          = os.path.join(LOG_DIR,       "app.log")

# ─────────────────────────────────────────────
# DATASET CONFIGURATION
# ─────────────────────────────────────────────
TEXT_COLUMN   = "text"
LABEL_COLUMN  = "label"
PHISHING_LABEL = "phishing"
SAFE_LABEL     = "safe"

# ─────────────────────────────────────────────
# TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
TEST_SIZE       = 0.2
RANDOM_STATE    = 42
CV_FOLDS        = 5       # cross-validation folds

# ─────────────────────────────────────────────
# TF-IDF VECTORIZER SETTINGS
# ─────────────────────────────────────────────
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE  = (1, 2)   # unigrams + bigrams
TFIDF_MIN_DF       = 1
TFIDF_MAX_DF       = 0.95

# ─────────────────────────────────────────────
# MODEL HYPERPARAMETERS
# ─────────────────────────────────────────────

# Logistic Regression
LR_PARAMS = {
    "C": 1.0,
    "max_iter": 1000,
    "solver": "lbfgs",
    "random_state": RANDOM_STATE,
}

# Naive Bayes (MultinomialNB – no random_state)
NB_PARAMS = {
    "alpha": 1.0,
}

# Random Forest
RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": None,
    "min_samples_split": 2,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# Support Vector Machine
SVM_PARAMS = {
    "C": 1.0,
    "kernel": "linear",
    "probability": True,
    "random_state": RANDOM_STATE,
}

# ─────────────────────────────────────────────
# HYPERPARAMETER GRID FOR TUNING (Random Forest)
# ─────────────────────────────────────────────
RF_PARAM_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5],
}

# ─────────────────────────────────────────────
# PHISHING DETECTION KEYWORDS
# ─────────────────────────────────────────────
PHISHING_KEYWORDS = [
    "urgent", "verify", "suspended", "limited", "click here",
    "act now", "immediately", "expires", "account suspended",
    "confirm your", "update your", "your account", "password",
    "credentials", "bank account", "credit card", "social security",
    "winner", "congratulations", "free gift", "lottery", "prize",
    "claim", "offer expires", "final warning", "security alert",
    "unauthorized", "suspicious activity", "compromised", "locked",
    "validate", "reactivate", "dear user", "dear customer",
    "valued customer", "dear account", "billing information",
    "payment failed", "invoice attached", "reset password",
    "unusual activity", "verify identity", "confirm account",
]

# ─────────────────────────────────────────────
# SUSPICIOUS URL PATTERNS
# ─────────────────────────────────────────────
URL_SHORTENERS = [
    "bit.ly", "tinyurl.com", "goo.gl", "ow.ly", "t.co",
    "buff.ly", "adf.ly", "is.gd", "cli.gs", "yfrog.com",
]

SUSPICIOUS_TLDS = [
    ".tk", ".ml", ".ga", ".cf", ".gq", ".xyz", ".top",
    ".click", ".loan", ".online", ".site", ".club",
]

# ─────────────────────────────────────────────
# VISUALIZATION SETTINGS
# ─────────────────────────────────────────────
FIGURE_DPI    = 150
FIGURE_SIZE   = (10, 7)
COLOR_PALETTE = "Blues"

# ─────────────────────────────────────────────
# LOGGING SETTINGS
# ─────────────────────────────────────────────
LOG_LEVEL  = "DEBUG"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE   = "%Y-%m-%d %H:%M:%S"
