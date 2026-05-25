"""
utils.py - Reusable Utility Functions
======================================
Provides logging setup, directory management, label encoding helpers,
timing decorators, and any other cross-cutting concerns used throughout
the project.
"""

import os
import time
import logging
import functools

import numpy as np

# Import config using a path-safe approach so the module works
# whether called from project root or from inside src/.
try:
    from src.config import (
        LOG_FILE_PATH, LOG_DIR, LOG_LEVEL, LOG_FORMAT, LOG_DATE,
        DATASET_DIR, MODEL_DIR, REPORT_DIR,
    )
except ModuleNotFoundError:
    from src.config import (
        LOG_FILE_PATH, LOG_DIR, LOG_LEVEL, LOG_FORMAT, LOG_DATE,
        DATASET_DIR, MODEL_DIR, REPORT_DIR,
    )


# ─────────────────────────────────────────────
# ENSURE ALL PROJECT DIRECTORIES EXIST
# ─────────────────────────────────────────────

def ensure_directories() -> None:
    """Create all required project directories if they do not already exist."""
    for directory in [DATASET_DIR, MODEL_DIR, REPORT_DIR, LOG_DIR]:
        os.makedirs(directory, exist_ok=True)


# ─────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────

def setup_logger(name: str = "phishing_detector") -> logging.Logger:
    """
    Configure and return a logger that writes to both the console
    and a rotating log file.

    Parameters
    ----------
    name : str
        Name of the logger (usually the module name).

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    ensure_directories()

    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if logger was already configured
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, LOG_LEVEL, logging.DEBUG))

    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=LOG_DATE)

    # ── Console handler ──────────────────────────────────────────────
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # ── File handler ─────────────────────────────────────────────────
    file_handler = logging.FileHandler(LOG_FILE_PATH, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


# ─────────────────────────────────────────────
# TIMING DECORATOR
# ─────────────────────────────────────────────

def timer(func):
    """
    Decorator that logs the execution time of any function.

    Usage
    -----
    @timer
    def some_function():
        ...
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        _logger = setup_logger()
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        _logger.debug("'%s' completed in %.4f seconds.", func.__name__, elapsed)
        return result
    return wrapper


# ─────────────────────────────────────────────
# LABEL HELPERS
# ─────────────────────────────────────────────

def encode_labels(labels, phishing_label: str = "phishing") -> np.ndarray:
    """
    Convert string labels to binary integers.

    Parameters
    ----------
    labels : array-like
        Sequence of string labels ('phishing' or 'safe').
    phishing_label : str
        The label string that maps to 1.

    Returns
    -------
    np.ndarray
        Integer array where phishing=1, safe=0.
    """
    return np.array([1 if lbl == phishing_label else 0 for lbl in labels])


def decode_label(value: int) -> str:
    """
    Convert a binary integer back to a human-readable label.

    Parameters
    ----------
    value : int
        0 or 1.

    Returns
    -------
    str
        'PHISHING' or 'SAFE'.
    """
    return "PHISHING" if value == 1 else "SAFE"


# ─────────────────────────────────────────────
# SAFE FILE CHECK
# ─────────────────────────────────────────────

def file_exists(path: str) -> bool:
    """Return True if *path* points to an existing file."""
    return os.path.isfile(path)


# ─────────────────────────────────────────────
# SEPARATOR PRINTER (CLI helper)
# ─────────────────────────────────────────────

def print_separator(char: str = "─", width: int = 60) -> None:
    """Print a visual separator line to stdout."""
    print(char * width)


def print_header(title: str, width: int = 60) -> None:
    """Print a styled section header to stdout."""
    print_separator(width=width)
    print(f"  {title}")
    print_separator(width=width)
