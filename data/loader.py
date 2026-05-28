"""
data/loader.py

Handles loading and caching the Bitext Customer Service dataset.
The dataset is downloaded from HuggingFace on first use and cached in memory.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import pandas as pd
from datasets import load_dataset

logger = logging.getLogger(__name__)

DATASET_NAME = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"


@lru_cache(maxsize=1)
def get_dataframe() -> pd.DataFrame:
    """
    Load the Bitext customer service dataset as a pandas DataFrame.

    Downloads from HuggingFace on first call, then returns the cached copy.
    Columns of interest:
      - instruction: the customer's message / query
      - response:    the agent's reply
      - category:    high-level grouping (e.g. ACCOUNT, REFUND, SHIPPING)
      - intent:      specific intent label (e.g. get_refund, cancel_order)
      - flags:       dialogue-act tags (not used in most queries)

    Returns
    -------
    pd.DataFrame
        The full dataset with normalised column names.
    """
    logger.info("Loading Bitext dataset from HuggingFace (this may take a moment)…")
    ds = load_dataset(DATASET_NAME, split="train")
    df = ds.to_pandas()

    # Normalise text columns so comparisons are case-insensitive-safe
    df["category"] = df["category"].str.upper().str.strip()
    df["intent"] = df["intent"].str.lower().str.strip()

    logger.info("Dataset loaded: %d rows, columns: %s", len(df), list(df.columns))
    return df


def get_categories() -> list[str]:
    """Return a sorted list of unique category names."""
    return sorted(get_dataframe()["category"].unique().tolist())


def get_intents(category: str | None = None) -> list[str]:
    """
    Return a sorted list of unique intent names.

    Parameters
    ----------
    category:
        If provided, restrict to intents belonging to that category.
    """
    df = get_dataframe()
    if category:
        df = df[df["category"] == category.upper().strip()]
    return sorted(df["intent"].unique().tolist())