"""
agent/profile.py

Helpers for loading, saving, and formatting per-session user profiles.

The profile is a lightweight JSON file stored at:
    sessions/profiles/<session_id>.json

It captures distilled facts about the user — name, frequent topics,
preferences — updated after every agent response.  It is intentionally
NOT a replay of messages; it is a concise, structured summary.

Example profile file
--------------------
{
    "name": "Yoni",
    "topics_of_interest": ["REFUND", "SHIPPING"],
    "preferences": "prefers concise bullet-point answers",
    "other_facts": [],
    "last_updated": "2025-01-15T10:32:00"
}
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROFILES_DIR = Path("sessions") / "profiles"

# Default structure for a brand-new profile
_DEFAULT_PROFILE: dict[str, Any] = {
    "name": None,
    "topics_of_interest": [],
    "preferences": None,
    "other_facts": [],
    "last_updated": None,
}


def _profile_path(session_id: str) -> Path:
    """Return the filesystem path for a session's profile file."""
    return PROFILES_DIR / f"{session_id}.json"


def load_profile(session_id: str) -> dict[str, Any]:
    """
    Load the user profile for a session from disk.

    Returns the stored profile dict, or a fresh default profile if none exists.

    Parameters
    ----------
    session_id:
        The session identifier used as the filename stem.
    """
    path = _profile_path(session_id)
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                profile = json.load(f)
            logger.info("Loaded profile for session '%s'.", session_id)
            return profile
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load profile for '%s': %s", session_id, exc)

    return dict(_DEFAULT_PROFILE)


def save_profile(session_id: str, profile: dict[str, Any]) -> None:
    """
    Persist a user profile to disk.

    Creates the profiles directory if it does not exist.

    Parameters
    ----------
    session_id:
        The session identifier used as the filename stem.
    profile:
        The profile dict to serialise.
    """
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    path = _profile_path(session_id)
    profile["last_updated"] = datetime.now().isoformat(timespec="seconds")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(profile, f, indent=2, ensure_ascii=False)
        logger.info("Saved profile for session '%s'.", session_id)
    except OSError as exc:
        logger.warning("Failed to save profile for '%s': %s", session_id, exc)


def build_profile_context(profile: dict[str, Any]) -> str:
    """
    Format a profile dict as a human-readable string for injection into
    the agent system prompt.

    Returns an empty string if the profile contains no meaningful data yet,
    so the system prompt isn't cluttered on the first turn.

    Parameters
    ----------
    profile:
        The profile dict returned by :func:`load_profile`.
    """
    lines: list[str] = []

    if profile.get("name"):
        lines.append(f"- Name: {profile['name']}")

    topics = profile.get("topics_of_interest") or []
    if topics:
        lines.append(f"- Frequently asks about: {', '.join(topics)}")

    if profile.get("preferences"):
        lines.append(f"- Preferences: {profile['preferences']}")

    other = profile.get("other_facts") or []
    for fact in other:
        lines.append(f"- {fact}")

    if not lines:
        return ""

    return "User profile (use this to personalise your answers):\n" + "\n".join(lines)


def is_empty_profile(profile: dict[str, Any]) -> bool:
    """Return True if the profile contains no meaningful data yet."""
    return (
        not profile.get("name")
        and not profile.get("topics_of_interest")
        and not profile.get("preferences")
        and not profile.get("other_facts")
    )