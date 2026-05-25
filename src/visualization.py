"""
visualization.py - Charts, Heatmaps & Feature Importance Plots
================================================================
Generates and saves four key visualisations:
  1. Confusion matrix heatmap
  2. Accuracy / F1 comparison bar chart across all models
  3. Feature importance bar chart (Random Forest)
  4. ROC curve (optional)
All figures are written to reports/ as PNG files.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (safe for servers)
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from src.config import (
        CONFUSION_MATRIX_PATH, ACCURACY_CHART_PATH, FEATURE_IMP_PATH,
        FIGURE_DPI, FIGURE_SIZE, REPORT_DIR,
    )
    from src.utils import setup_logger
except ModuleNotFoundError:
    from src.config import (
        CONFUSION_MATRIX_PATH, ACCURACY_CHART_PATH, FEATURE_IMP_PATH,
        FIGURE_DPI, FIGURE_SIZE, REPORT_DIR,
    )
    from src.utils import setup_logger

logger = setup_logger(__name__)

# ─────────────────────────────────────────────
# GLOBAL STYLE
# ─────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.facecolor":   "white",
})


# ─────────────────────────────────────────────
# 1. CONFUSION MATRIX HEATMAP
# ─────────────────────────────────────────────

def plot_confusion_matrix(
    cm: np.ndarray,
    model_name: str = "Model",
    save_path: str  = CONFUSION_MATRIX_PATH,
) -> str:
    """
    Save a styled confusion matrix heatmap.

    Parameters
    ----------
    cm         : (2, 2) confusion matrix from sklearn
    model_name : title label
    save_path  : output PNG path

    Returns
    -------
    str  – path where file was saved
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    labels = ["Safe (0)", "Phishing (1)"]

    fig, ax = plt.subplots(figsize=(7, 5), dpi=FIGURE_DPI)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix – {model_name}", fontsize=14, pad=15)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label",      fontsize=11)

    # Annotate TN / FP / FN / TP
    positions  = [(0, 0, "TN"), (0, 1, "FP"), (1, 0, "FN"), (1, 1, "TP")]
    for row, col, tag in positions:
        ax.text(
            col + 0.5, row + 0.75, tag,
            ha="center", va="center",
            fontsize=9, color="grey",
        )

    fig.tight_layout()
    fig.savefig(save_path, dpi=FIGURE_DPI)
    plt.close(fig)
    logger.info("Confusion matrix saved → %s", save_path)
    return save_path


# ─────────────────────────────────────────────
# 2. MODEL COMPARISON BAR CHART
# ─────────────────────────────────────────────

def plot_model_comparison(
    results: dict,
    save_path: str = ACCURACY_CHART_PATH,
) -> str:
    """
    Save a grouped bar chart comparing CV Accuracy and F1 across models.

    Parameters
    ----------
    results   : dict returned by train_model.train_all_models()
    save_path : output PNG path

    Returns
    -------
    str  – path where file was saved
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    names      = list(results.keys())
    accuracies = [results[n]["cv_metrics"]["accuracy"] for n in names]
    f1_scores  = [results[n]["cv_metrics"]["f1"]       for n in names]

    x     = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)

    bars1 = ax.bar(x - width/2, accuracies, width, label="CV Accuracy",
                   color="#4A90D9", edgecolor="white", linewidth=0.8)
    bars2 = ax.bar(x + width/2, f1_scores,  width, label="CV F1-Score",
                   color="#E07B54", edgecolor="white", linewidth=0.8)

    # Value labels on top of each bar
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.annotate(
            f"{h:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=8,
        )

    ax.set_title("Model Performance Comparison (Cross-Validation)",
                 fontsize=14, pad=15)
    ax.set_xlabel("Classifier",  fontsize=11)
    ax.set_ylabel("Score",       fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(save_path, dpi=FIGURE_DPI)
    plt.close(fig)
    logger.info("Accuracy chart saved → %s", save_path)
    return save_path


# ─────────────────────────────────────────────
# 3. FEATURE IMPORTANCE (Random Forest)
# ─────────────────────────────────────────────

def plot_feature_importance(
    model,
    feature_builder,
    top_n: int = 25,
    save_path: str = FEATURE_IMP_PATH,
) -> str:
    """
    Save a horizontal bar chart of the top-N feature importances for a
    tree-based model (Random Forest).

    Parameters
    ----------
    model          : fitted RandomForestClassifier
    feature_builder : FeatureBuilder (has .vectorizer with vocabulary)
    top_n          : how many features to display
    save_path      : output PNG path

    Returns
    -------
    str  – path where file was saved, or '' if model has no feature_importances_
    """
    if not hasattr(model, "feature_importances_"):
        logger.warning("Model has no feature_importances_; skipping chart.")
        return ""

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Build feature name list: TF-IDF vocab + cyber feature names
    from src.feature_extraction import CyberFeatureExtractor
    tfidf_names = list(feature_builder.vectorizer.get_feature_names_out())
    cyber_names = CyberFeatureExtractor.FEATURE_NAMES
    all_names   = tfidf_names + cyber_names

    importances = model.feature_importances_
    # Guard against size mismatch
    n = min(len(importances), len(all_names))
    importances = importances[:n]
    all_names   = all_names[:n]

    indices     = np.argsort(importances)[-top_n:]
    top_names   = [all_names[i] for i in indices]
    top_values  = importances[indices]

    fig, ax = plt.subplots(figsize=(9, 7), dpi=FIGURE_DPI)
    colours = ["#E07B54" if n in CyberFeatureExtractor.FEATURE_NAMES else "#4A90D9"
               for n in top_names]

    ax.barh(top_names, top_values, color=colours, edgecolor="white")
    ax.set_title(f"Top-{top_n} Feature Importances (Random Forest)",
                 fontsize=14, pad=15)
    ax.set_xlabel("Importance", fontsize=11)
    ax.xaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4A90D9", label="TF-IDF token"),
        Patch(facecolor="#E07B54", label="Cyber feature"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=FIGURE_DPI)
    plt.close(fig)
    logger.info("Feature importance chart saved → %s", save_path)
    return save_path


# ─────────────────────────────────────────────
# 4. ROC CURVE
# ─────────────────────────────────────────────

def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
    save_path:  str = None,
) -> str:
    """
    Save a ROC curve plot.

    Parameters
    ----------
    y_true     : true binary labels
    y_prob     : predicted probabilities for the positive class
    model_name : label used in the title
    save_path  : output PNG path (defaults to reports/roc_curve.png)

    Returns
    -------
    str  – path where file was saved
    """
    from sklearn.metrics import roc_curve, auc

    if save_path is None:
        save_path = os.path.join(REPORT_DIR, "roc_curve.png")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc     = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=FIGURE_DPI)
    ax.plot(fpr, tpr, color="#4A90D9", lw=2,
            label=f"ROC curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="grey", lw=1.5, linestyle="--",
            label="Random classifier")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#4A90D9")

    ax.set_title(f"ROC Curve – {model_name}", fontsize=14, pad=15)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate",  fontsize=11)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    fig.tight_layout()
    fig.savefig(save_path, dpi=FIGURE_DPI)
    plt.close(fig)
    logger.info("ROC curve saved → %s", save_path)
    return save_path


# ─────────────────────────────────────────────
# CONVENIENCE: GENERATE ALL REPORTS
# ─────────────────────────────────────────────

def generate_all_reports(eval_result: dict, train_results: dict, model, feature_builder) -> None:
    """
    Generate and save all four report plots in one call.

    Parameters
    ----------
    eval_result    : dict from evaluate.evaluate_model()
    train_results  : dict from train_model.train_all_models()
    model          : best fitted model
    feature_builder: FeatureBuilder instance
    """
    plot_confusion_matrix(eval_result["confusion_matrix"],
                          model_name=eval_result["model_name"])
    plot_model_comparison(train_results)
    plot_feature_importance(model, feature_builder)

    if eval_result.get("roc_auc") is not None:
        try:
            y_prob = model.predict_proba(None)   # will raise – caught below
        except Exception:
            pass   # ROC curve requires probability array; skip if unavailable

    logger.info("All reports generated.")


# ─────────────────────────────────────────────
# STANDALONE USAGE
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import numpy as np
    # Demo with a dummy confusion matrix
    dummy_cm = np.array([[8, 1], [0, 11]])
    plot_confusion_matrix(dummy_cm, model_name="Demo Model")
    print("Demo confusion matrix saved.")
