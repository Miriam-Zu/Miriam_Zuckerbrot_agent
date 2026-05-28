"""
agent/state.py

Defines the AgentState that is threaded through every node in the LangGraph graph.
"""

from __future__ import annotations

from typing import Annotated, Literal

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


QueryType = Literal["structured", "unstructured", "out_of_scope"]


class AgentState(TypedDict):
    """
    Shared state for the Bitext analyst agent graph.

    Fields
    ------
    messages:
        Full conversation history.  Uses the add_messages reducer so new
        messages are appended rather than replacing the list.
    query_type:
        Classification assigned by the router node.  Drives conditional edges.
    iteration_count:
        Number of agent→tool round-trips completed in the current turn.
        Checked after every tool execution to enforce the max-iterations guard.
    """

    messages: Annotated[list, add_messages]
    query_type: QueryType | None
    iteration_count: int