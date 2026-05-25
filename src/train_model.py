"""
train_model.py - Model Training, Comparison & Persistence
===========================================================
Trains four classifiers (Logistic Regression, Naive Bayes, Random
Forest, SVM) on the combined TF-IDF + cyber-feature matrix, performs
cross-validation, optionally tunes the best model with GridSearchCV,
and persists the winner plus its vectorizer/scaler with Joblib.
"""

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes   import MultinomialNB
from sklearn.ensemble       import RandomForestClassifier
from sklearn.svm            import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline        import Pipeline
from sklearn.preprocessing   import MinMaxScaler

try:
    from src.config import (
        MODEL_PATH, VECTORIZER_PATH, SCALER_PATH,
        LR_PARAMS, NB_PARAMS, RF_PARAMS, SVM_PARAMS,
        RF_PARAM_GRID, RANDOM_STATE, CV_FOLDS,
        PHISHING_LABEL,
    )
    from src.utils import setup_logger, timer, encode_labels
    from src.preprocess import preprocess_pipeline, EmailPreprocessor
    from src.feature_extraction import FeatureBuilder
except ModuleNotFoundError:
    from src.config import (
        MODEL_PATH, VECTORIZER_PATH, SCALER_PATH,
        LR_PARAMS, NB_PARAMS, RF_PARAMS, SVM_PARAMS,
        RF_PARAM_GRID, RANDOM_STATE, CV_FOLDS,
        PHISHING_LABEL,
    )
    from src.utils import setup_logger, timer, encode_labels
    from src.preprocess import preprocess_pipeline, EmailPreprocessor
    from src.feature_extraction import FeatureBuilder

logger = setup_logger(__name__)


# ─────────────────────────────────────────────
# MODEL REGISTRY
# ─────────────────────────────────────────────

def get_models() -> dict:
    """
    Return a dictionary of named classifiers with their hyperparameters
    sourced from config.py.

    Notes
    -----
    MultinomialNB requires non-negative features.  We add a MinMaxScaler
    step that ensures the sparse cyber-feature columns (which are already
    standardised with a StandardScaler that can produce negatives) are
    clipped to [0, ∞).  For all other classifiers the raw combined matrix
    is used directly.
    """
    return {
        "Logistic Regression": LogisticRegression(**LR_PARAMS),
        "Naive Bayes":         MultinomialNB(**NB_PARAMS),
        "Random Forest":       RandomForestClassifier(**RF_PARAMS),
        "SVM":                 SVC(**SVM_PARAMS),
    }


# ─────────────────────────────────────────────
# CROSS-VALIDATION HELPER
# ─────────────────────────────────────────────

def cross_validate_model(
    model,
    X: sp.csr_matrix,
    y: np.ndarray,
    cv: int = CV_FOLDS,
) -> dict:
    """
    Run stratified k-fold cross-validation and return mean ± std for
    accuracy, precision, recall, and F1-score.

    Parameters
    ----------
    model  : sklearn estimator
    X      : feature matrix (sparse)
    y      : binary label array
    cv     : number of folds

    Returns
    -------
    dict with keys accuracy, precision, recall, f1 and their _std variants
    """
    metrics = {}
    for scoring in ("accuracy", "precision", "recall", "f1"):
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        metrics[scoring]             = scores.mean()
        metrics[f"{scoring}_std"]    = scores.std()
        logger.debug(
            "CV %s: %.4f ± %.4f", scoring, scores.mean(), scores.std()
        )
    return metrics


# ─────────────────────────────────────────────
# TRAINING PIPELINE
# ─────────────────────────────────────────────

@timer
def train_all_models(
    X_train_raw: pd.Series,
    y_train: pd.Series,
    feature_builder: FeatureBuilder,
) -> tuple[dict, FeatureBuilder]:
    """
    Train every classifier and collect cross-validation scores.

    Parameters
    ----------
    X_train_raw   : raw (cleaned) text series for training
    y_train       : string label series
    feature_builder : fitted FeatureBuilder (vectorizer + scaler already fit)

    Returns
    -------
    results : dict  {model_name: {"model": ..., "cv_metrics": ...}}
    feature_builder : same object (returned for convenience)
    """
    y_enc = encode_labels(y_train, phishing_label=PHISHING_LABEL)

    # The feature builder is already fit; just get the matrix
    X_feat = feature_builder.transform(X_train_raw, X_train_raw)

    # For Naive Bayes we need non-negative values – apply MinMaxScaler
    from sklearn.preprocessing import MaxAbsScaler
    X_feat_nn = MaxAbsScaler().fit_transform(X_feat)

    results = {}
    models  = get_models()

    for name, model in models.items():
        logger.info("Training %s …", name)
        # Naive Bayes needs non-negative matrix
        X_use = X_feat_nn if isinstance(model, MultinomialNB) else X_feat

        try:
            cv_metrics = cross_validate_model(model, X_use, y_enc)
            model.fit(X_use, y_enc)
            results[name] = {
                "model":      model,
                "cv_metrics": cv_metrics,
                "uses_nn_matrix": isinstance(model, MultinomialNB),
            }
            logger.info(
                "%s  |  CV Accuracy: %.4f ± %.4f  |  F1: %.4f",
                name,
                cv_metrics["accuracy"],
                cv_metrics["accuracy_std"],
                cv_metrics["f1"],
            )
        except Exception as exc:
            logger.error("Failed to train %s: %s", name, exc)

    return results


# ─────────────────────────────────────────────
# SELECT BEST MODEL
# ─────────────────────────────────────────────

def select_best_model(results: dict) -> tuple[str, dict]:
    """
    Pick the model with the highest mean CV F1-score.

    Returns
    -------
    best_name  : str
    best_entry : dict  {"model": ..., "cv_metrics": ...}
    """
    best_name = max(
        results,
        key=lambda k: results[k]["cv_metrics"]["f1"],
    )
    logger.info("Best model: %s (F1=%.4f)", best_name,
                results[best_name]["cv_metrics"]["f1"])
    return best_name, results[best_name]


# ─────────────────────────────────────────────
# HYPERPARAMETER TUNING (Random Forest)
# ─────────────────────────────────────────────

@timer
def tune_random_forest(
    X_train: sp.csr_matrix,
    y_train: np.ndarray,
    param_grid: dict = None,
    cv: int = CV_FOLDS,
) -> RandomForestClassifier:
    """
    Run GridSearchCV on Random Forest and return the best estimator.

    Parameters
    ----------
    X_train    : feature matrix
    y_train    : binary labels
    param_grid : dict of hyperparameter grids (uses config default if None)
    cv         : number of cross-validation folds

    Returns
    -------
    Best-fit RandomForestClassifier
    """
    if param_grid is None:
        param_grid = RF_PARAM_GRID

    logger.info("Starting GridSearchCV for Random Forest …")
    rf    = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    grid  = GridSearchCV(rf, param_grid, cv=cv, scoring="f1", n_jobs=-1, verbose=0)
    grid.fit(X_train, y_train)

    logger.info("Best RF params: %s  |  Best F1: %.4f",
                grid.best_params_, grid.best_score_)
    return grid.best_estimator_


# ─────────────────────────────────────────────
# MODEL PERSISTENCE
# ─────────────────────────────────────────────

def save_model(model, feature_builder: FeatureBuilder) -> None:
    """
    Persist the trained model and the FeatureBuilder (vectorizer + scaler)
    to disk using Joblib.

    Parameters
    ----------
    model          : fitted sklearn estimator
    feature_builder : FeatureBuilder instance (contains vectorizer & scaler)
    """
    import os
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    joblib.dump(model,           MODEL_PATH,      compress=3)
    joblib.dump(feature_builder, VECTORIZER_PATH, compress=3)

    logger.info("Model saved     → %s", MODEL_PATH)
    logger.info("FeatureBuilder  → %s", VECTORIZER_PATH)


def load_model():
    """
    Load the saved model and FeatureBuilder from disk.

    Returns
    -------
    model          : sklearn estimator
    feature_builder : FeatureBuilder
    """
    if not all([
        __import__("os").path.isfile(MODEL_PATH),
        __import__("os").path.isfile(VECTORIZER_PATH),
    ]):
        raise FileNotFoundError(
            "Saved model or vectorizer not found. Run training first."
        )

    model           = joblib.load(MODEL_PATH)
    feature_builder = joblib.load(VECTORIZER_PATH)
    logger.info("Model and FeatureBuilder loaded from disk.")
    return model, feature_builder


# ─────────────────────────────────────────────
# MAIN TRAINING ENTRY POINT
# ─────────────────────────────────────────────

@timer
def run_training() -> tuple:
    """
    Full training pipeline:
    1. Preprocess data
    2. Build features (fit on training set)
    3. Train all models with CV
    4. Select best model
    5. Optionally tune with GridSearchCV
    6. Save best model

    Returns
    -------
    best_model, feature_builder, results, X_test_feat, y_test
    """
    # ── Step 1: preprocess ──────────────────────────────────────────
    X_train_clean, X_test_clean, y_train, y_test = preprocess_pipeline()

    # ── Step 2: build features ──────────────────────────────────────
    feature_builder = FeatureBuilder()
    X_train_feat    = feature_builder.fit_transform(X_train_clean, X_train_clean)
    X_test_feat     = feature_builder.transform(X_test_clean, X_test_clean)

    # ── Step 3: train all models ────────────────────────────────────
    results = train_all_models(X_train_clean, y_train, feature_builder)

    # ── Step 4: select best ─────────────────────────────────────────
    best_name, best_entry = select_best_model(results)
    best_model = best_entry["model"]

    # ── Step 5: optionally tune RF if it is the best ─────────────
    if "Random Forest" in best_name:
        logger.info("Tuning best Random Forest model …")
        y_enc      = encode_labels(y_train, phishing_label=PHISHING_LABEL)
        best_model = tune_random_forest(X_train_feat, y_enc)
        best_model.fit(X_train_feat, y_enc)

    # ── Step 6: save ────────────────────────────────────────────────
    save_model(best_model, feature_builder)

    return best_model, feature_builder, results, X_test_feat, y_test


# ─────────────────────────────────────────────
# STANDALONE USAGE
# ─────────────────────────────────────────────

if __name__ == "__main__":
    model, fb, results, X_test, y_test = run_training()
    print("\nTraining complete.")
    for name, entry in results.items():
        m = entry["cv_metrics"]
        print(f"  {name:<22} Acc={m['accuracy']:.4f}  F1={m['f1']:.4f}")
