"""
agent/tools.py

All tools available to the Bitext analyst agent.

Design principles:
- Every tool has a clear name, a descriptive docstring (used as the LLM-facing description), 
  and a Pydantic input schema so the LLM knows exactly what arguments to supply.
- Tools are pure functions over the shared DataFrame.
- Return values are plain Python objects (dicts / lists / strings) 
  so they serialise cleanly into tool-result messages.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from data.loader import get_dataframe, get_categories, get_intents

logger = logging.getLogger(__name__)


# Input schemas (Pydantic models)

class ListIntentsInput(BaseModel):
    """Input for the list_intents tool."""

    category: str | None = Field(
        default=None,
        description=(
            "Upper-case category name to filter by (e.g. 'REFUND', 'SHIPPING'). "
            "Leave blank to list all intents across the entire dataset."
        ),
    )


class FilterRowsInput(BaseModel):
    """Input for the filter_rows tool."""

    category: str | None = Field(
        default=None,
        description="Upper-case category name to filter by (e.g. 'ACCOUNT').",
    )
    intent: str | None = Field(
        default=None,
        description="Snake-case intent label to filter by (e.g. 'get_refund').",
    )
    n_samples: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of example rows to return (1–20).",
    )


class CountRowsInput(BaseModel):
    """Input for the count_rows tool."""

    category: str | None = Field(
        default=None,
        description="Upper-case category name to count (e.g. 'REFUND').",
    )
    intent: str | None = Field(
        default=None,
        description="Snake-case intent label to count (e.g. 'get_refund').",
    )


class GetDistributionInput(BaseModel):
    """Input for the get_distribution tool."""

    category: str = Field(
        description=(
            "Upper-case category whose intent distribution you want "
            "(e.g. 'ACCOUNT', 'REFUND')."
        )
    )


class SummariseSamplesInput(BaseModel):
    """Input for the summarise_samples tool."""

    category: str | None = Field(
        default=None,
        description="Upper-case category to sample from.",
    )
    intent: str | None = Field(
        default=None,
        description="Snake-case intent to sample from.",
    )
    n_samples: int = Field(
        default=30,
        ge=5,
        le=100,
        description="Number of rows to sample before summarising (5–100).",
    )
    focus: str = Field(
        default="both customer messages and agent responses",
        description=(
            "What aspect to focus the summary on, e.g. "
            "'agent responses', 'customer messages', or 'both'."
        ),
    )


class SearchByKeywordInput(BaseModel):
    """Input for the search_by_keyword tool."""

    keyword: str = Field(
        description=(
            "Keyword or short phrase to search for inside customer messages "
            "(case-insensitive substring match)."
        )
    )
    n_samples: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of matching rows to return (1–20).",
    )


# Tool implementations

@tool("list_categories")
def list_categories() -> dict[str, Any]:
    """
    List all unique category names that exist in the Bitext dataset.

    Use this tool when the user asks what categories, topics, or high-level
    groupings are available in the dataset.

    Returns a dict with key 'categories' containing a sorted list of strings.
    """
    cats = get_categories()
    return {"categories": cats, "count": len(cats)}


@tool("list_intents", args_schema=ListIntentsInput)
def list_intents(category: str | None = None) -> dict[str, Any]:
    """
    List intent labels in the dataset, optionally filtered to one category.

    Use this tool when the user asks what intents or sub-topics exist,
    either overall or within a specific category.

    Parameters
    ----------
    category:
        Optional upper-case category name (e.g. 'REFUND').
        Omit to list all intents in the whole dataset.
    """
    intents = get_intents(category)
    result: dict[str, Any] = {"intents": intents, "count": len(intents)}
    if category:
        result["category"] = category.upper()
    return result


@tool("filter_rows", args_schema=FilterRowsInput)
def filter_rows(
    category: str | None = None, 
    intent: str | None = None, 
    n_samples: int = 5,
) -> dict[str, Any]:
    """
    Return a sample of rows from the dataset, filtered by category and/or intent.

    Use this tool when the user asks to 'show me examples' or 'give me samples'
    from a particular category or intent.  Also useful as the first step when
    you need to see raw text before summarising.

    Parameters
    ----------
    category:
        Upper-case category filter (e.g. 'SHIPPING').
    intent:
        Snake-case intent filter (e.g. 'track_order').
    n_samples:
        How many rows to return (default 5, max 20).
    """
    df = get_dataframe()

    if category:
        df = df[df["category"] == category.upper().strip()]
    if intent:
        df = df[df["intent"] == intent.lower().strip()]

    total = len(df)
    sample = df[["instruction", "response", "category", "intent"]].head(n_samples)

    return {
        "total_matching_rows": total,
        "returned_samples": len(sample),
        "rows": sample.to_dict(orient="records"),
    }


@tool("count_rows", args_schema=CountRowsInput)
def count_rows(
    category: str | None = None,
    intent: str | None = None,
) -> dict[str, Any]:
    """
    Count the number of rows matching a category and/or intent filter.

    Use this tool when the user asks 'how many …' questions, such as
    'how many refund requests', 'how many rows in SHIPPING', etc.

    Parameters
    ----------
    category:
        Upper-case category to count (e.g. 'REFUND').
    intent:
        Snake-case intent to count (e.g. 'get_refund').
    """
    df = get_dataframe()

    if category:
        df = df[df["category"] == category.upper().strip()]
    if intent:
        df = df[df["intent"] == intent.lower().strip()]

    result: dict[str, Any] = {"count": len(df)}
    if category:
        result["category"] = category.upper()
    if intent:
        result["intent"] = intent.lower()
    return result


@tool("get_distribution", args_schema=GetDistributionInput)
def get_distribution(category: str) -> dict[str, Any]:
    """
    Return the distribution of intents within a given category.

    Use this tool when the user asks about the breakdown, distribution, or
    proportion of intents inside a specific category
    (e.g. 'What is the distribution of intents in the ACCOUNT category?').

    Parameters
    ----------
    category:
        Upper-case category name (e.g. 'ACCOUNT').
    """
    df = get_dataframe()
    df = df[df["category"] == category.upper().strip()]

    if df.empty:
        return {"error": f"No rows found for category '{category}'."}

    counts = df["intent"].value_counts()
    total = counts.sum()
    distribution = [
        {
            "intent": intent,
            "count": int(cnt),
            "percentage": round(100 * cnt / total, 1),
        }
        for intent, cnt in counts.items()
    ]
    return {
        "category": category.upper(),
        "total_rows": int(total),
        "distribution": distribution,
    }


@tool("summarise_samples", args_schema=SummariseSamplesInput)
def summarise_samples(
    category: str | None = None,
    intent: str | None = None,
    n_samples: int = 30,
    focus: str = "both customer messages and agent responses",
) -> dict[str, Any]:
    """
    Fetch a sample of rows and return their raw text so the agent can summarise them.

    Use this tool for open-ended / unstructured questions such as:
    - 'Summarise the FEEDBACK category'
    - 'How do agents respond to cancellation requests?'
    - 'What themes appear in REFUND complaints?'

    This tool does NOT summarise itself 
    rather it returns the raw instruction/response pairs 
    so the LLM can synthesise a meaningful summary.

    Parameters
    ----------
    category:
        Upper-case category to sample from.
    intent:
        Snake-case intent to sample from.
    n_samples:
        Number of rows to sample (default 30).
    focus:
        Hint about what to focus on ('agent responses', 'customer messages', 'both').
    """
    df = get_dataframe()

    if category:
        df = df[df["category"] == category.upper().strip()]
    if intent:
        df = df[df["intent"] == intent.lower().strip()]

    if df.empty:
        return {"error": "No rows found matching the given filters."}

    sample = df[["instruction", "response", "category", "intent"]].sample(
        min(n_samples, len(df)), random_state=42
    )

    return {
        "total_matching_rows": len(df),
        "sampled_rows": len(sample),
        "focus": focus,
        "rows": sample.to_dict(orient="records"),
    }


@tool("search_by_keyword", args_schema=SearchByKeywordInput)
def search_by_keyword(keyword: str, n_samples: int = 5) -> dict[str, Any]:
    """
    Search for rows whose customer message (instruction) contains a keyword or phrase.

    Use this tool when the user describes what they're looking for in plain language
    rather than naming a specific intent or category. For example:
    - 'Show me examples of people wanting their money back'
    - 'Find messages about delayed packages'
    - 'Show me complaints about billing'

    Parameters
    ----------
    keyword:
        Word or short phrase to search for (case-insensitive substring match).
    n_samples:
        How many matching rows to return (default 5, max 20).
    """
    df = get_dataframe()
    mask = df["instruction"].str.contains(keyword, case=False, na=False)
    matches = df[mask]

    sample = matches[["instruction", "response", "category", "intent"]].head(n_samples)

    return {
        "keyword": keyword,
        "total_matches": len(matches),
        "returned_samples": len(sample),
        "rows": sample.to_dict(orient="records"),
    }


# Exported tool list (used by the graph and the MCP server)

ALL_TOOLS = [
    list_categories,
    list_intents,
    filter_rows,
    count_rows,
    get_distribution,
    summarise_samples,
    search_by_keyword,
]