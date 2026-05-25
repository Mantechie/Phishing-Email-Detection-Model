"""
app.py - Interactive Command-Line Interface
============================================
The main user-facing CLI.  It exposes a numbered menu:

  1. Train model
  2. Evaluate model
  3. Test custom email
  4. Save model (re-save after evaluation)
  5. Exit

Run directly:
    python app.py
"""

import sys
import os

# Allow imports from src/ whether running from project root or src/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils         import setup_logger, print_header, print_separator, ensure_directories
from src.preprocess    import preprocess_pipeline, EmailPreprocessor
from src.feature_extraction import FeatureBuilder
from src.train_model   import run_training, save_model, load_model
from src.evaluate      import evaluate_model, compare_models, print_evaluation
from src.predict       import predict_email, print_prediction
from src.visualization import (
    plot_confusion_matrix,
    plot_model_comparison,
    plot_feature_importance,
)
from src.config import MODEL_PATH, VECTORIZER_PATH

logger = setup_logger("app")

# ─────────────────────────────────────────────
# STATE (held in memory for one CLI session)
# ─────────────────────────────────────────────
_model           = None
_feature_builder = None
_train_results   = None
_X_test_feat     = None
_y_test          = None


# ─────────────────────────────────────────────
# MENU ACTIONS
# ─────────────────────────────────────────────

def action_train() -> None:
    """Train all models, select best, and save to disk."""
    global _model, _feature_builder, _train_results, _X_test_feat, _y_test

    print_header("TRAINING MODELS")
    try:
        (
            _model,
            _feature_builder,
            _train_results,
            _X_test_feat,
            _y_test,
        ) = run_training()

        print("\n  Model Comparison (Cross-Validation):")
        df = compare_models(_train_results)
        print(df.to_string(index=False))

        print("\n  ✔  Training complete.  Best model saved to disk.")
        logger.info("Training workflow completed successfully.")
    except Exception as exc:
        logger.error("Training failed: %s", exc, exc_info=True)
        print(f"\n  ✖  Training failed: {exc}")


def action_evaluate() -> None:
    """Evaluate the current best model on the hold-out test set."""
    global _model, _feature_builder, _X_test_feat, _y_test

    # Try to load from disk if not in memory
    if _model is None:
        try:
            _model, _feature_builder = load_model()
            _, X_test_clean, _, _y_test = preprocess_pipeline()
            _X_test_feat = _feature_builder.transform(X_test_clean, X_test_clean)
        except FileNotFoundError as exc:
            print(f"\n  ✖  {exc}")
            return

    print_header("EVALUATING MODEL")
    try:
        result = evaluate_model(
            _model, _X_test_feat, _y_test, model_name="Best Model"
        )
        print_evaluation(result)

        # Save visualisations
        print("  Generating report charts …")
        plot_confusion_matrix(result["confusion_matrix"], model_name="Best Model")

        if _train_results:
            plot_model_comparison(_train_results)

        if hasattr(_model, "feature_importances_"):
            plot_feature_importance(_model, _feature_builder)

        print("  ✔  Charts saved to reports/")
    except Exception as exc:
        logger.error("Evaluation failed: %s", exc, exc_info=True)
        print(f"\n  ✖  Evaluation failed: {exc}")


def action_predict() -> None:
    """Interactively classify a user-entered email."""
    global _model, _feature_builder

    # Load model if needed
    if _model is None:
        try:
            _model, _feature_builder = load_model()
        except FileNotFoundError as exc:
            print(f"\n  ✖  {exc}")
            return

    print_header("TEST CUSTOM EMAIL")
    print("  Paste your email text below.")
    print("  Enter a blank line followed by 'END' to finish.\n")

    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip().upper() == "END":
            break
        lines.append(line)

    email_text = "\n".join(lines).strip()

    if not email_text:
        print("  No email text entered.")
        return

    try:
        result = predict_email(email_text, model=_model, feature_builder=_feature_builder)
        print_prediction(result, email_text=email_text)
    except Exception as exc:
        logger.error("Prediction failed: %s", exc, exc_info=True)
        print(f"\n  ✖  Prediction failed: {exc}")


def action_save() -> None:
    """Re-save the in-memory model (useful after manual evaluation)."""
    global _model, _feature_builder

    if _model is None or _feature_builder is None:
        print("\n  ✖  No model in memory.  Train a model first.")
        return

    try:
        save_model(_model, _feature_builder)
        print(f"\n  ✔  Model saved → {MODEL_PATH}")
        print(f"  ✔  FeatureBuilder saved → {VECTORIZER_PATH}")
    except Exception as exc:
        logger.error("Save failed: %s", exc, exc_info=True)
        print(f"\n  ✖  Save failed: {exc}")


# ─────────────────────────────────────────────
# MENU RENDERER
# ─────────────────────────────────────────────

def show_menu() -> None:
    """Print the main CLI menu."""
    print_separator()
    print("  🛡️  PHISHING EMAIL DETECTION SYSTEM")
    print_separator()
    print("  1. Train Model")
    print("  2. Evaluate Model")
    print("  3. Test Custom Email")
    print("  4. Save Model")
    print("  5. Exit")
    print_separator()


# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────

def main() -> None:
    """Entry point for the interactive CLI."""
    ensure_directories()
    logger.info("Phishing Email Detection System started.")

    dispatch = {
        "1": action_train,
        "2": action_evaluate,
        "3": action_predict,
        "4": action_save,
    }

    while True:
        show_menu()
        try:
            choice = input("  Enter choice (1-5): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n  Goodbye!")
            break

        if choice == "5":
            print("\n  Goodbye!")
            logger.info("CLI session ended by user.")
            break
        elif choice in dispatch:
            dispatch[choice]()
        else:
            print("  ✖  Invalid choice.  Please enter 1–5.")


if __name__ == "__main__":
    main()
