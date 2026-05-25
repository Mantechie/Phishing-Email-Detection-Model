"""
main.py - Non-Interactive Pipeline Runner
==========================================
Runs the full pipeline end-to-end without user interaction.
Useful for automated testing, CI/CD pipelines, or quick demos.

Usage:
    python main.py
    python main.py --predict "Your email text here"
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils          import setup_logger, print_header, ensure_directories
from src.preprocess     import preprocess_pipeline
from src.feature_extraction import FeatureBuilder
from src.train_model    import run_training, load_model
from src.evaluate       import evaluate_model, compare_models, print_evaluation
from src.predict        import predict_email, print_prediction
from src.visualization  import (
    plot_confusion_matrix,
    plot_model_comparison,
    plot_feature_importance,
)

logger = setup_logger("main")


# ─────────────────────────────────────────────
# FULL PIPELINE
# ─────────────────────────────────────────────

def run_full_pipeline() -> None:
    """
    Execute every stage in sequence:
    1. Preprocess
    2. Feature extraction
    3. Train all classifiers
    4. Evaluate best model
    5. Generate report charts
    6. Demo prediction
    """
    ensure_directories()
    print_header("PHISHING EMAIL DETECTION – FULL PIPELINE")

    # ── Stage 1-4: Training ─────────────────────────────────────────
    print("\n[1/5]  Training models …\n")
    model, feature_builder, train_results, X_test_feat, y_test = run_training()

    # ── Stage 4: Comparison table ───────────────────────────────────
    print("\n[2/5]  Model comparison:\n")
    df = compare_models(train_results)
    print(df.to_string(index=False))

    # ── Stage 4b: Holdout evaluation ────────────────────────────────
    print("\n[3/5]  Evaluating best model on hold-out test set …\n")
    eval_result = evaluate_model(model, X_test_feat, y_test, model_name="Best Model")
    print_evaluation(eval_result)

    # ── Stage 5: Visualisations ─────────────────────────────────────
    print("\n[4/5]  Generating report charts …\n")
    plot_confusion_matrix(eval_result["confusion_matrix"], model_name="Best Model")
    plot_model_comparison(train_results)
    if hasattr(model, "feature_importances_"):
        plot_feature_importance(model, feature_builder)

    # ROC curve
    try:
        from src.visualization import plot_roc_curve
        from src.utils         import encode_labels
        from src.config        import PHISHING_LABEL
        import numpy as np

        y_true = encode_labels(y_test, phishing_label=PHISHING_LABEL)
        y_prob = model.predict_proba(X_test_feat)[:, 1]
        plot_roc_curve(y_true, y_prob, model_name="Best Model")
    except Exception as exc:
        logger.warning("ROC curve skipped: %s", exc)

    print("  ✔  Charts saved to reports/")

    # ── Stage 6: Demo predictions ────────────────────────────────────
    print("\n[5/5]  Demo predictions …\n")

    demo_emails = [
        (
            "URGENT: Your PayPal account has been LIMITED! "
            "Verify IMMEDIATELY at http://bit.ly/paypal-verify or your "
            "account will be suspended PERMANENTLY within 24 HOURS!!!",
            "Expected: PHISHING",
        ),
        (
            "Hi Sarah, Just a reminder about the team lunch tomorrow "
            "at noon. The restaurant is at 45 Main Street. See you there!",
            "Expected: SAFE",
        ),
    ]

    for email_text, expected in demo_emails:
        result = predict_email(email_text, model=model, feature_builder=feature_builder)
        print(f"  {expected}")
        print_prediction(result, email_text=email_text)

    print_header("PIPELINE COMPLETE")
    logger.info("Full pipeline completed successfully.")


# ─────────────────────────────────────────────
# PREDICT-ONLY MODE
# ─────────────────────────────────────────────

def run_predict_only(email_text: str) -> None:
    """
    Load a saved model and classify a single email passed from CLI args.

    Parameters
    ----------
    email_text : str – raw email content to classify
    """
    ensure_directories()
    logger.info("Predict-only mode.")

    try:
        model, feature_builder = load_model()
    except FileNotFoundError as exc:
        print(f"\n  ✖  {exc}")
        print("  Run 'python main.py' first to train and save a model.")
        sys.exit(1)

    result = predict_email(email_text, model=model, feature_builder=feature_builder)
    print_prediction(result, email_text=email_text)


# ─────────────────────────────────────────────
# ARGUMENT PARSING
# ─────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phishing Email Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py\n"
            "  python main.py --predict \"URGENT: Click here http://bit.ly/x\"\n"
        ),
    )
    parser.add_argument(
        "--predict",
        type=str,
        default=None,
        metavar="EMAIL_TEXT",
        help="Classify a single email and exit (requires a saved model).",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    if args.predict:
        run_predict_only(args.predict)
    else:
        run_full_pipeline()
