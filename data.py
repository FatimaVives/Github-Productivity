"""data.py

Helpers to load and prepare the repository dataset for modeling.

This module reads `data/repositories.csv`, validates required columns,
parses dates, creates engineered numeric features, and builds the
`productivity` target used by training. Defensive checks and helpful
errors are provided for missing columns.
"""
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import RAW_CSV, RANDOM_SEED, TEST_SIZE, VAL_SIZE
from utils import setup_logging

logger = setup_logging()

NUMERIC_COLS = [
    "Size",
    "Stars",
    "Forks",
    "Issues",
    "Watchers",
]

BOOL_COLS = [
    "Has Issues",
    "Has Projects",
    "Has Downloads",
    "Has Wiki",
    "Has Pages",
    "Has Discussions",
    "Is Fork",
    "Is Archived",
    "Is Template",
]

DATE_COLS = ["Created At", "Updated At"]


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    missing_dates = [c for c in DATE_COLS if c not in df.columns]
    if missing_dates:
        raise ValueError(
            f"Missing required date columns in CSV: {missing_dates}. These are needed to compute repository age."
        )

    for c in NUMERIC_COLS:
        if c not in df.columns:
            logger.warning("Column '%s' missing from CSV; adding default 0", c)
            df[c] = 0

    # legacy tests / downstream code expect a commits_count column; add default
    if "commits_count" not in df.columns:
        logger.debug("Adding missing 'commits_count' column with default 0")
        df["commits_count"] = 0

    for c in BOOL_COLS:
        if c not in df.columns:
            logger.warning("Column '%s' missing from CSV; adding default False", c)
            df[c] = False

    return df

def load_raw(path=RAW_CSV) -> pd.DataFrame:
    """Read the raw CSV and perform minimal validation.

    - parses `Created At` and `Updated At` as datetimes
    - ensures required numeric and boolean columns exist (adds defaults)
    """
    logger.info("Loading CSV from %s", path)
    df = pd.read_csv(path, parse_dates=DATE_COLS, dayfirst=False)

    df = _ensure_columns(df)

    # normalize column names (strip)
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]

    # convert date columns to datetimes (coerce errors)
    for d in DATE_COLS:
        df[d] = pd.to_datetime(df[d], errors="coerce")
        if df[d].isna().any():
            logger.warning("Column '%s' contains unparsable dates; some values set to NaT", d)

    return df


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Lightweight cleaning: drop duplicates and normalize booleans/numerics."""
    df = df.drop_duplicates().copy()

    # strip whitespace from textual columns if present
    if "Name" in df.columns:
        df["Name"] = df["Name"].astype(str).str.strip()

    # ensure numeric types
    for c in NUMERIC_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(float)

    # ensure boolean columns are 0/1 integers
    for c in BOOL_COLS:
        df[c] = df[c].fillna(False).astype(bool).astype(int)

    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create engineered numeric features required for modeling.

    Adds:
      - repo_age_days
      - stars_per_fork
      - issues_per_star
      - preserves/ensures numeric/boolean columns
      - creates `productivity` target based on `Stars`
    """
    df = df.copy()

    # repo age in days (non-negative)
    df["repo_age_days"] = (df["Updated At"] - df["Created At"]).dt.days
    df["repo_age_days"] = df["repo_age_days"].clip(lower=0).fillna(0).astype(float)

    # engineered ratios (guard against division by zero)
    df["stars_per_fork"] = df["Stars"] / (df["Forks"] + 1)
    df["issues_per_star"] = df["Issues"] / (df["Stars"] + 1)

    # ensure no infinite values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

    # create categorical target 'productivity'
    # high: Stars > 200000
    # medium: 50000 <= Stars <= 200000
    # low: Stars < 50000
    conditions = [
        df["Stars"] > 200_000,
        (df["Stars"] >= 50_000) & (df["Stars"] <= 200_000),
        df["Stars"] < 50_000,
    ]
    choices = ["high", "medium", "low"]
    df["productivity"] = np.select(conditions, choices, default="low")

    # make sure target is a categorical dtype
    df["productivity"] = pd.Categorical(df["productivity"], categories=["low", "medium", "high"])

    return df


def train_val_test_split(df: pd.DataFrame, target_col: str = "productivity") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the dataframe into train/val/test with stratification on the target.

    Raises a helpful ValueError if the target is missing or if stratification
    cannot be performed (e.g., only one class present).
    """
    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in dataframe. "
            "Ensure your input CSV or preprocessing produces this column before calling training."
        )

    # verify we have at least two classes for stratification
    class_counts = df[target_col].value_counts()
    if class_counts.shape[0] < 2:
        raise ValueError(
            f"Not enough classes in target '{target_col}' for stratified split. Found classes: {list(class_counts.index)}."
        )

    train_val, test = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=df[target_col])
    relative_val = VAL_SIZE / (1 - TEST_SIZE)
    train, val = train_test_split(train_val, test_size=relative_val, random_state=RANDOM_SEED, stratify=train_val[target_col])

    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
