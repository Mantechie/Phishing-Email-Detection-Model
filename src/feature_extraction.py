"""
feature_extraction.py - NLP & Cybersecurity Feature Engineering
================================================================
Combines TF-IDF vectorisation with hand-crafted cybersecurity
features (URL counts, suspicious domains, phishing keywords, etc.)
into a single, unified feature matrix.

Two public functions are exposed:
  • build_features_train() – fit+transform on training data
  • build_features_predict() – transform-only on new data
"""

import re
import scipy.sparse as sp
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

try:
    from src.config import (
        TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE, TFIDF_MIN_DF, TFIDF_MAX_DF,
        PHISHING_KEYWORDS, URL_SHORTENERS, SUSPICIOUS_TLDS,
    )
    from src.utils import setup_logger, timer
except ModuleNotFoundError:
    from src.config import (
        TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE, TFIDF_MIN_DF, TFIDF_MAX_DF,
        PHISHING_KEYWORDS, URL_SHORTENERS, SUSPICIOUS_TLDS,
    )
    from src.utils import setup_logger, timer

logger = setup_logger(__name__)

# Pre-compiled regexes used across feature functions
_RE_URL        = re.compile(r"http[s]?://[^\s]+|www\.[^\s]+", re.IGNORECASE)
_RE_IP_URL     = re.compile(r"http[s]?://\d{1,3}(?:\.\d{1,3}){3}", re.IGNORECASE)
_RE_HTML_TAG   = re.compile(r"<[^>]+>")
_RE_UPPER_WORD = re.compile(r"\b[A-Z]{3,}\b")  # words of 3+ uppercase letters
_RE_EMAIL_ADDR = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")


# ─────────────────────────────────────────────
# HAND-CRAFTED CYBERSECURITY FEATURE EXTRACTOR
# ─────────────────────────────────────────────

class CyberFeatureExtractor:
    """
    Extracts 10 hand-crafted numeric features from raw email text.

    Feature vector (per email)
    --------------------------
    0  url_count          – total number of URLs
    1  ip_url_count        – URLs that use an IP address instead of a domain
    2  url_shortener_count – URLs pointing to known shortener services
    3  suspicious_tld_count– URLs whose TLD is on the suspicious list
    4  phishing_kw_count   – matches against the phishing keyword list
    5  uppercase_word_count– number of ALL-CAPS words (urgency signals)
    6  exclamation_count   – number of '!' characters
    7  email_length        – total character count of the raw email
    8  html_tag_count      – number of HTML tags detected
    9  email_addr_count    – number of distinct email addresses in the body
    """

    FEATURE_NAMES = [
        "url_count",
        "ip_url_count",
        "url_shortener_count",
        "suspicious_tld_count",
        "phishing_kw_count",
        "uppercase_word_count",
        "exclamation_count",
        "email_length",
        "html_tag_count",
        "email_addr_count",
    ]

    # ------------------------------------------------------------------
    def transform(self, texts: pd.Series) -> np.ndarray:
        """
        Convert a Series of raw email strings to a 2-D numeric array.

        Parameters
        ----------
        texts : pd.Series
            Raw (not pre-cleaned) email texts.

        Returns
        -------
        np.ndarray, shape (n_samples, 10)
        """
        return np.array([self._extract_one(t) for t in texts], dtype=float)

    # ------------------------------------------------------------------
    def _extract_one(self, text: str) -> list[float]:
        """Extract all 10 features from a single email string."""
        if not isinstance(text, str):
            text = ""

        text_lower = text.lower()

        # ── URL-based features ──────────────────────────────────────
        urls = _RE_URL.findall(text)
        url_count = len(urls)

        ip_url_count = len(_RE_IP_URL.findall(text))

        url_shortener_count = sum(
            1 for u in urls if any(svc in u.lower() for svc in URL_SHORTENERS)
        )

        suspicious_tld_count = sum(
            1 for u in urls if any(u.lower().endswith(tld) for tld in SUSPICIOUS_TLDS)
        )

        # ── Keyword features ─────────────────────────────────────────
        phishing_kw_count = sum(
            1 for kw in PHISHING_KEYWORDS if kw in text_lower
        )

        # ── Structural / stylistic features ─────────────────────────
        uppercase_word_count = len(_RE_UPPER_WORD.findall(text))
        exclamation_count    = text.count("!")
        email_length         = len(text)
        html_tag_count       = len(_RE_HTML_TAG.findall(text))
        email_addr_count     = len(_RE_EMAIL_ADDR.findall(text))

        return [
            url_count,
            ip_url_count,
            url_shortener_count,
            suspicious_tld_count,
            phishing_kw_count,
            uppercase_word_count,
            exclamation_count,
            email_length,
            html_tag_count,
            email_addr_count,
        ]


# ─────────────────────────────────────────────
# COMBINED FEATURE BUILDER
# ─────────────────────────────────────────────

class FeatureBuilder:
    """
    Combines TF-IDF vectors with CyberFeatures into one sparse matrix.

    Attributes
    ----------
    vectorizer : TfidfVectorizer
    scaler     : StandardScaler
    cyber_extractor : CyberFeatureExtractor
    """

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            ngram_range=TFIDF_NGRAM_RANGE,
            min_df=TFIDF_MIN_DF,
            max_df=TFIDF_MAX_DF,
            sublinear_tf=True,       # apply log(1+tf) scaling
        )
        self.scaler          = StandardScaler(with_mean=False)
        self.cyber_extractor = CyberFeatureExtractor()

    # ------------------------------------------------------------------
    @timer
    def fit_transform(
        self,
        cleaned_texts: pd.Series,
        raw_texts: pd.Series,
    ) -> sp.csr_matrix:
        """
        Fit all transformers and return combined feature matrix for training.

        Parameters
        ----------
        cleaned_texts : pd.Series
            Pre-processed (stop-word removed, lemmatised) text.
        raw_texts : pd.Series
            Original email texts (for cyber features).

        Returns
        -------
        sp.csr_matrix, shape (n_samples, TFIDF_MAX_FEATURES + 10)
        """
        logger.info("Fitting TF-IDF vectoriser …")
        tfidf_matrix = self.vectorizer.fit_transform(cleaned_texts)
        logger.info("TF-IDF matrix: %s", tfidf_matrix.shape)

        logger.info("Extracting cybersecurity features …")
        cyber_matrix = self.cyber_extractor.transform(raw_texts)
        cyber_scaled = self.scaler.fit_transform(cyber_matrix)

        combined = sp.hstack(
            [tfidf_matrix, sp.csr_matrix(cyber_scaled)],
            format="csr",
        )
        logger.info("Combined feature matrix: %s", combined.shape)
        return combined

    # ------------------------------------------------------------------
    @timer
    def transform(
        self,
        cleaned_texts: pd.Series,
        raw_texts: pd.Series,
    ) -> sp.csr_matrix:
        """
        Transform-only (use after fit_transform on training data).

        Parameters
        ----------
        cleaned_texts : pd.Series or list
            Pre-processed texts.
        raw_texts : pd.Series or list
            Original email texts.

        Returns
        -------
        sp.csr_matrix
        """
        tfidf_matrix = self.vectorizer.transform(cleaned_texts)
        cyber_matrix = self.cyber_extractor.transform(raw_texts)
        cyber_scaled = self.scaler.transform(cyber_matrix)
        return sp.hstack(
            [tfidf_matrix, sp.csr_matrix(cyber_scaled)],
            format="csr",
        )


# ─────────────────────────────────────────────
# STANDALONE USAGE
# ─────────────────────────────────────────────

if __name__ == "__main__":
    sample = pd.Series([
        "URGENT! Click here http://bit.ly/fake to verify your account!!!",
        "Hi team, meeting at 10 AM tomorrow. See you then.",
    ])
    extractor = CyberFeatureExtractor()
    features  = extractor.transform(sample)
    print("Cyber feature names:", CyberFeatureExtractor.FEATURE_NAMES)
    print("Feature matrix:\n", features)
