"""
preprocess.py - Email Text Preprocessing Pipeline
===================================================
Loads the raw email CSV dataset, cleans and normalises text, and returns
train/test splits ready for feature extraction.  All NLP cleaning steps
are encapsulated in the EmailPreprocessor class so they can be applied
consistently at both training time and prediction time.
"""

import re
import string

import pandas as pd
from sklearn.model_selection import train_test_split

try:
    import nltk
    # Download required NLTK resources on first run
    for resource in ("stopwords", "punkt", "wordnet", "punkt_tab", "omw-1.4"):
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            try:
                nltk.download(resource, quiet=True)
            except Exception:
                pass
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    _NLTK_AVAILABLE = True
except ImportError:
    _NLTK_AVAILABLE = False

try:
    from src.config import (
        DATASET_PATH, TEXT_COLUMN, LABEL_COLUMN,
        TEST_SIZE, RANDOM_STATE,
    )
    from src.utils import setup_logger, timer
except ModuleNotFoundError:
    from src.config import (
        DATASET_PATH, TEXT_COLUMN, LABEL_COLUMN,
        TEST_SIZE, RANDOM_STATE,
    )
    from src.utils import setup_logger, timer

logger = setup_logger(__name__)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
if _NLTK_AVAILABLE:
    try:
        _STOP_WORDS = set(stopwords.words("english"))
        _LEMMATIZER = WordNetLemmatizer()
    except Exception:
        _NLTK_AVAILABLE = False

if not _NLTK_AVAILABLE:
    # Minimal fallback stop-word list when NLTK is unavailable
    _STOP_WORDS = {
        "a","an","the","and","or","but","in","on","at","to","for",
        "of","with","is","was","are","were","be","been","have","has",
        "do","does","did","will","would","could","should","may","might",
        "this","that","these","those","it","its","i","you","he","she",
        "we","they","my","your","his","her","our","their","me","him",
        "us","them","if","as","by","from","up","about","into","through",
        "not","no","so","just","also","only","very","more","most",
    }
    class _FallbackLemmatizer:
        def lemmatize(self, word):
            return word
    _LEMMATIZER = _FallbackLemmatizer()

# URL regex – broad enough to catch HTTP, HTTPS and bare domains
_URL_PATTERN = re.compile(
    r"http[s]?://(?:[a-zA-Z]|[0-9]|[$\-_@.&+]|[!*(),]|%[0-9a-fA-F]{2})+"
    r"|www\.[^\s]+",
    re.IGNORECASE,
)


# ─────────────────────────────────────────────
# EMAILPREPROCESSOR CLASS
# ─────────────────────────────────────────────

class EmailPreprocessor:
    """
    Stateless text-cleaning utility.

    Methods are static / class-level so this object can be instantiated
    once and shared across modules without carrying mutable state.
    """

    # ------------------------------------------------------------------
    # PUBLIC: clean a single email string
    # ------------------------------------------------------------------
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Apply the full normalisation pipeline to one email text.

        Steps
        -----
        1. Lower-case
        2. Replace URLs with placeholder token ``urltoken``
        3. Remove HTML tags
        4. Remove punctuation (keep spaces)
        5. Remove digits
        6. Tokenise
        7. Remove stop-words
        8. Lemmatise each token
        9. Rejoin into a single string

        Parameters
        ----------
        text : str
            Raw email body.

        Returns
        -------
        str
            Cleaned, normalised text.
        """
        if not isinstance(text, str):
            return ""

        # 1. Lower-case
        text = text.lower()

        # 2. Replace URLs with a special token so the TF-IDF vectoriser
        #    can learn "this email contains a URL" as a feature.
        text = _URL_PATTERN.sub(" urltoken ", text)

        # 3. Strip HTML tags
        text = re.sub(r"<[^>]+>", " ", text)

        # 4. Remove punctuation
        text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))

        # 5. Remove digits
        text = re.sub(r"\d+", " ", text)

        # 6. Tokenise on whitespace
        tokens = text.split()

        # 7. Remove stop-words (keep "urltoken" placeholder)
        tokens = [t for t in tokens if t not in _STOP_WORDS or t == "urltoken"]

        # 8. Lemmatise
        tokens = [_LEMMATIZER.lemmatize(t) for t in tokens if len(t) > 1]

        # 9. Rejoin
        return " ".join(tokens)

    # ------------------------------------------------------------------
    # PUBLIC: clean a whole pandas Series
    # ------------------------------------------------------------------
    @staticmethod
    def clean_series(series: pd.Series) -> pd.Series:
        """Apply ``clean_text`` to every row of a pandas Series."""
        return series.fillna("").apply(EmailPreprocessor.clean_text)


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

@timer
def load_dataset(path: str = DATASET_PATH) -> pd.DataFrame:
    """
    Load the email CSV file and perform basic sanity checks.

    Parameters
    ----------
    path : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame with at least ``text`` and ``label`` columns.

    Raises
    ------
    FileNotFoundError
        If the CSV path does not exist.
    ValueError
        If required columns are missing.
    """
    logger.info("Loading dataset from: %s", path)

    if not pd.io.common.file_exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")

    df = pd.read_csv(path)
    logger.info("Loaded %d rows, %d columns.", len(df), df.shape[1])

    # Validate required columns
    missing = [c for c in (TEXT_COLUMN, LABEL_COLUMN) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Drop rows with null text or label
    before = len(df)
    df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN], inplace=True)
    df.reset_index(drop=True, inplace=True)
    dropped = before - len(df)
    if dropped:
        logger.warning("Dropped %d rows with null values.", dropped)

    logger.info(
        "Label distribution:\n%s",
        df[LABEL_COLUMN].value_counts().to_string(),
    )
    return df


# ─────────────────────────────────────────────
# FULL PREPROCESSING PIPELINE
# ─────────────────────────────────────────────

@timer
def preprocess_pipeline(
    path: str = DATASET_PATH,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    End-to-end preprocessing: load → clean → split.

    Returns
    -------
    X_train, X_test, y_train, y_test : pd.Series
        Raw (uncleaned) text splits and their string labels.
        Feature extraction is handled downstream by feature_extraction.py
        so it can operate on both raw and cleaned text simultaneously.
    """
    df = load_dataset(path)

    logger.info("Cleaning email texts …")
    df["cleaned_text"] = EmailPreprocessor.clean_series(df[TEXT_COLUMN])

    X = df["cleaned_text"]
    y = df[LABEL_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    logger.info(
        "Split → train: %d  |  test: %d", len(X_train), len(X_test)
    )
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────
# STANDALONE USAGE
# ─────────────────────────────────────────────

if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te = preprocess_pipeline()
    print("Train samples:", len(X_tr))
    print("Test  samples:", len(X_te))
    print("\nSample cleaned text:\n", X_tr.iloc[0])
