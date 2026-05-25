"""
predict.py - Single Email Prediction
======================================
Loads a saved model + FeatureBuilder and provides:
  • predict_email()  – classify one email and return label + confidence
  • explain_email()  – surface the top suspicious signals detected
Both functions are usable standalone or imported by app.py / main.py.
"""

import re

try:
    from src.train_model       import load_model
    from src.preprocess        import EmailPreprocessor
    from src.feature_extraction import CyberFeatureExtractor
    from src.utils             import setup_logger, decode_label
    from src.config            import PHISHING_KEYWORDS, URL_SHORTENERS, SUSPICIOUS_TLDS
except ModuleNotFoundError:
    from src.train_model       import load_model
    from src.preprocess        import EmailPreprocessor
    from src.feature_extraction import CyberFeatureExtractor
    from src.utils             import setup_logger, decode_label
    from src.config            import PHISHING_KEYWORDS, URL_SHORTENERS, SUSPICIOUS_TLDS

import pandas as pd
import numpy as np

logger = setup_logger(__name__)

# URL regex reused from feature_extraction
_RE_URL    = re.compile(r"http[s]?://[^\s]+|www\.[^\s]+", re.IGNORECASE)
_RE_IP_URL = re.compile(r"http[s]?://\d{1,3}(?:\.\d{1,3}){3}", re.IGNORECASE)


# ─────────────────────────────────────────────
# CORE PREDICTION FUNCTION
# ─────────────────────────────────────────────

def predict_email(email_text: str, model=None, feature_builder=None) -> dict:
    """
    Classify a single email as PHISHING or SAFE.

    Parameters
    ----------
    email_text      : str  – raw email body
    model           : trained sklearn estimator (loaded from disk if None)
    feature_builder : FeatureBuilder (loaded from disk if None)

    Returns
    -------
    dict with keys:
        label           – "PHISHING" or "SAFE"
        confidence      – float, probability of predicted class
        phishing_prob   – float, raw probability of phishing class
        safe_prob       – float, raw probability of safe class
        suspicious_signals – list of human-readable warning strings
    """
    if model is None or feature_builder is None:
        model, feature_builder = load_model()

    # 1. Clean text
    preprocessor = EmailPreprocessor()
    cleaned      = preprocessor.clean_text(email_text)

    cleaned_series = pd.Series([cleaned])
    raw_series     = pd.Series([email_text])

    # 2. Build features (transform only – vectorizer already fit)
    X_feat = feature_builder.transform(cleaned_series, raw_series)

    # 3. Predict
    pred_int  = model.predict(X_feat)[0]
    label_str = decode_label(pred_int)

    # 4. Confidence
    try:
        proba        = model.predict_proba(X_feat)[0]
        phishing_prob = float(proba[1])
        safe_prob     = float(proba[0])
        confidence    = max(phishing_prob, safe_prob)
    except AttributeError:
        phishing_prob = float(pred_int)
        safe_prob     = 1.0 - float(pred_int)
        confidence    = 1.0

    # 5. Collect suspicious signals for explainability
    signals = _explain_signals(email_text)

    result = {
        "label":             label_str,
        "confidence":        confidence,
        "phishing_prob":     phishing_prob,
        "safe_prob":         safe_prob,
        "suspicious_signals": signals,
    }

    logger.info(
        "Prediction: %s (phishing_prob=%.4f) | signals=%d",
        label_str, phishing_prob, len(signals),
    )
    return result


# ─────────────────────────────────────────────
# EXPLAINABILITY HELPERS
# ─────────────────────────────────────────────

def _explain_signals(text: str) -> list[str]:
    """
    Return a list of human-readable phishing signals found in *text*.

    Parameters
    ----------
    text : str – raw email body

    Returns
    -------
    list of str – each entry is a short description of a detected signal
    """
    signals   = []
    text_lower = text.lower()

    # ── URL signals ────────────────────────────────────────────────
    urls = _RE_URL.findall(text)
    if urls:
        signals.append(f"Contains {len(urls)} URL(s): {', '.join(urls[:3])}")

    ip_urls = _RE_IP_URL.findall(text)
    if ip_urls:
        signals.append(f"IP-based URL(s) detected: {', '.join(ip_urls)}")

    for url in urls:
        if any(svc in url.lower() for svc in URL_SHORTENERS):
            signals.append(f"URL shortener detected: {url}")
            break

    for url in urls:
        for tld in SUSPICIOUS_TLDS:
            if url.lower().endswith(tld) or f"{tld}/" in url.lower():
                signals.append(f"Suspicious TLD in URL: {url}")
                break

    # ── Keyword signals ─────────────────────────────────────────────
    matched_kws = [kw for kw in PHISHING_KEYWORDS if kw in text_lower]
    if matched_kws:
        signals.append(f"Phishing keywords: {', '.join(matched_kws[:8])}")

    # ── Urgency / style signals ──────────────────────────────────────
    excl = text.count("!")
    if excl >= 3:
        signals.append(f"High urgency: {excl} exclamation marks")

    upper_words = re.findall(r"\b[A-Z]{3,}\b", text)
    if len(upper_words) >= 3:
        signals.append(f"Aggressive capitalisation: {', '.join(set(upper_words[:6]))}")

    # ── HTML signals ─────────────────────────────────────────────────
    html_tags = re.findall(r"<[^>]+>", text)
    if html_tags:
        signals.append(f"HTML tags detected ({len(html_tags)} tags)")

    return signals


# ─────────────────────────────────────────────
# PRETTY-PRINT HELPER
# ─────────────────────────────────────────────

def print_prediction(result: dict, email_text: str = "") -> None:
    """
    Print a formatted prediction result to stdout.

    Parameters
    ----------
    result     : dict returned by predict_email()
    email_text : optional raw email for display (truncated to 200 chars)
    """
    sep = "─" * 55

    print(f"\n{sep}")
    print("  PHISHING EMAIL DETECTION RESULT")
    print(sep)

    if email_text:
        preview = email_text[:200] + ("…" if len(email_text) > 200 else "")
        print(f"  Email Preview  : {preview}")
        print(sep)

    label = result["label"]
    emoji = "🚨" if label == "PHISHING" else "✅"
    print(f"  Verdict        : {emoji}  {label}")
    print(f"  Confidence     : {result['confidence']*100:.1f}%")
    print(f"  Phishing Prob  : {result['phishing_prob']*100:.1f}%")
    print(f"  Safe Prob      : {result['safe_prob']*100:.1f}%")

    signals = result["suspicious_signals"]
    if signals:
        print(f"\n  ⚠  Suspicious Signals ({len(signals)}):")
        for sig in signals:
            print(f"     • {sig}")
    else:
        print("\n  ✔  No suspicious signals detected.")

    print(sep)


# ─────────────────────────────────────────────
# STANDALONE USAGE
# ─────────────────────────────────────────────

if __name__ == "__main__":
    sample_phishing = (
        "URGENT: Your account has been suspended! Click here immediately "
        "to verify: http://bit.ly/fake-verify or your account will be "
        "PERMANENTLY DELETED within 24 HOURS!!!"
    )

    sample_safe = (
        "Hi John, just a reminder about tomorrow's 10 AM standup. "
        "Please have your sprint updates ready. Thanks, Sarah."
    )

    for email in [sample_phishing, sample_safe]:
        result = predict_email(email)
        print_prediction(result, email_text=email)
