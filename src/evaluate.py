"""
evaluate.py - Model Evaluation & Reporting
============================================
Provides detailed evaluation of a trained classifier:
  • Classification report (precision / recall / F1 per class)
  • Confusion matrix
  • ROC-AUC score
  • Cross-validation summary table
All results are logged and returned as structured dictionaries.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

try:
    from src.utils import setup_logger, timer, encode_labels, decode_label
    from src.config import PHISHING_LABEL, CV_FOLDS
except ModuleNotFoundError:
    from src.utils import setup_logger, timer, encode_labels, decode_label
    from src.config import PHISHING_LABEL, CV_FOLDS

logger = setup_logger(__name__)


# ─────────────────────────────────────────────
# SINGLE-HOLDOUT EVALUATION
# ─────────────────────────────────────────────

@timer
def evaluate_model(
    model,
    X_test,
    y_test,
    model_name: str = "Model",
) -> dict:
    """
    Evaluate a fitted model on the hold-out test set.

    Parameters
    ----------
    model      : fitted sklearn estimator
    X_test     : feature matrix (sparse or dense)
    y_test     : true labels (string or binary int)
    model_name : display name for logging

    Returns
    -------
    dict containing:
        accuracy, precision, recall, f1, roc_auc,
        confusion_matrix (ndarray), report (str)
    """
    # Encode labels to binary integers if they are strings
    if hasattr(y_test.iloc[0] if hasattr(y_test, "iloc") else y_test[0], "__len__"):
        y_true = encode_labels(y_test, phishing_label=PHISHING_LABEL)
    else:
        y_true = np.array(y_test)

    y_pred = model.predict(X_test)

    # ROC-AUC requires probability scores
    try:
        y_prob   = model.predict_proba(X_test)[:, 1]
        roc_auc  = roc_auc_score(y_true, y_prob)
    except (AttributeError, IndexError):
        roc_auc = None
        logger.warning("predict_proba not available – ROC-AUC skipped.")

    acc   = accuracy_score(y_true, y_pred)
    prec  = precision_score(y_true, y_pred, zero_division=0)
    rec   = recall_score(y_true, y_pred, zero_division=0)
    f1    = f1_score(y_true, y_pred, zero_division=0)
    cm    = confusion_matrix(y_true, y_pred)
    report_str = classification_report(
        y_true, y_pred,
        target_names=["Safe", "Phishing"],
        zero_division=0,
    )

    logger.info("=" * 55)
    logger.info("Evaluation Report – %s", model_name)
    logger.info("=" * 55)
    logger.info("Accuracy  : %.4f", acc)
    logger.info("Precision : %.4f", prec)
    logger.info("Recall    : %.4f", rec)
    logger.info("F1-Score  : %.4f", f1)
    if roc_auc is not None:
        logger.info("ROC-AUC   : %.4f", roc_auc)
    logger.info("\n%s", report_str)

    return {
        "model_name":       model_name,
        "accuracy":         acc,
        "precision":        prec,
        "recall":           rec,
        "f1":               f1,
        "roc_auc":          roc_auc,
        "confusion_matrix": cm,
        "report":           report_str,
        "y_true":           y_true,
        "y_pred":           y_pred,
    }


# ─────────────────────────────────────────────
# MULTI-MODEL COMPARISON TABLE
# ─────────────────────────────────────────────

def compare_models(results: dict) -> pd.DataFrame:
    """
    Build a summary DataFrame from the training results dictionary
    produced by train_model.train_all_models().

    Parameters
    ----------
    results : dict  { model_name: { "cv_metrics": {...} } }

    Returns
    -------
    pd.DataFrame sorted by F1-score descending
    """
    rows = []
    for name, entry in results.items():
        m = entry["cv_metrics"]
        rows.append({
            "Model":     name,
            "CV Acc":    f"{m['accuracy']:.4f} ± {m['accuracy_std']:.4f}",
            "Precision": f"{m['precision']:.4f}",
            "Recall":    f"{m['recall']:.4f}",
            "F1":        f"{m['f1']:.4f}",
        })
    df = pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)
    logger.info("\n%s\n", df.to_string(index=False))
    return df


# ─────────────────────────────────────────────
# PRETTY PRINT EVALUATION RESULTS
# ─────────────────────────────────────────────

def print_evaluation(eval_result: dict) -> None:
    """
    Print a formatted evaluation report to stdout.

    Parameters
    ----------
    eval_result : dict returned by evaluate_model()
    """
    name = eval_result["model_name"]
    sep  = "─" * 55

    print(f"\n{sep}")
    print(f"  Evaluation Report → {name}")
    print(sep)
    print(f"  Accuracy  : {eval_result['accuracy']:.4f}")
    print(f"  Precision : {eval_result['precision']:.4f}")
    print(f"  Recall    : {eval_result['recall']:.4f}")
    print(f"  F1-Score  : {eval_result['f1']:.4f}")
    if eval_result["roc_auc"] is not None:
        print(f"  ROC-AUC   : {eval_result['roc_auc']:.4f}")
    print(sep)
    print("\nClassification Report:")
    print(eval_result["report"])
    print("Confusion Matrix:")
    cm = eval_result["confusion_matrix"]
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}  TP={cm[1,1]}")
    print(sep)


# ─────────────────────────────────────────────
# STANDALONE USAGE
# ─────────────────────────────────────────────

if __name__ == "__main__":
    from src.train_model import load_model
    from src.preprocess  import preprocess_pipeline
    from src.feature_extraction import FeatureBuilder

    print("Loading saved model …")
    model, fb = load_model()

    _, X_test_clean, _, y_test = preprocess_pipeline()
    X_test_feat = fb.transform(X_test_clean, X_test_clean)

    result = evaluate_model(model, X_test_feat, y_test, model_name="Saved Model")
    print_evaluation(result)
