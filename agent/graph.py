"""
agent/graph.py

Assembles the full LangGraph ReAct agent graph.

Graph topology
--------------

  [router] --out_of_scope--> [decline] --> END
     |
  structured / unstructured
     |
     v
  [agent] <----------------------------------------------+
     |                                                   |
     | tool call                           (under limit) |
     v                                                   |
  [tools] --> [check_iterations] ------------------------+
                    |
                    | over limit
                    v
                   END (fallback message)

  [agent] -- no tool call --> [profile_updater] --> END
"""

from __future__ import annotations

import json
import logging
from typing import Any, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from agent.profile import load_profile, save_profile
from agent.prompts import (
    AGENT_SYSTEM_PROMPT,
    OUT_OF_SCOPE_MESSAGE,
    MAX_ITERATIONS_MESSAGE,
    PROFILE_UPDATER_SYSTEM_PROMPT,
    PROFILE_UPDATER_HUMAN_TEMPLATE,
)
from agent.router import build_router_node
from agent.state import AgentState
from agent.tools import ALL_TOOLS

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 12


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _last_human_text(state: AgentState) -> str:
    """Extract the text of the most recent HumanMessage in state."""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            return msg.content if isinstance(msg.content, str) else str(msg.content)
    return ""


def _last_ai_text(state: AgentState) -> str:
    """Extract the text of the most recent AIMessage (without tool calls) in state."""
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            return msg.content if isinstance(msg.content, str) else str(msg.content)
    return ""


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def _build_nodes(llm, session_id: str | None = None):
    """
    Build all graph node functions for the given LLM and session.

    Returns a dict of node_name -> callable, plus the ToolNode instance.

    Parameters
    ----------
    llm:
        A LangChain chat model with tool-calling support.
    session_id:
        Used by the profile_updater node to persist profile updates to disk.
        If None, profile updates are written to state only (not persisted).
    """
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    # ------------------------------------------------------------------
    # Node: router
    # ------------------------------------------------------------------
    router_node = build_router_node(llm)

    # ------------------------------------------------------------------
    # Node: decline
    # ------------------------------------------------------------------
    def decline_node(state: AgentState) -> dict:
        """Return a polite refusal for out-of-scope queries."""
        return {"messages": [AIMessage(content=OUT_OF_SCOPE_MESSAGE)]}

    # ------------------------------------------------------------------
    # Node: agent
    # ------------------------------------------------------------------
    def agent_node(state: AgentState) -> dict:
        """
        Run one ReAct reasoning step.

        Injects query_type and the user profile into the system prompt so the
        model is aware of both the query classification and user context.
        """
        from agent.profile import build_profile_context
 
        query_type = state.get("query_type") or "structured"
        profile = state.get("user_profile") or {}
        profile_context = build_profile_context(profile)
 
        system_msg = SystemMessage(
            content=AGENT_SYSTEM_PROMPT.format(
                query_type=query_type,
                profile_context=profile_context,
            )
        )
 
        all_messages = [system_msg] + list(state["messages"])
        response = llm_with_tools.invoke(all_messages)
        return {"messages": [response]}

    # ------------------------------------------------------------------
    # Node: tools
    # ------------------------------------------------------------------
    tool_node = ToolNode(ALL_TOOLS)

    # ------------------------------------------------------------------
    # Node: check_iterations
    # ------------------------------------------------------------------
    def check_iterations_node(state: AgentState) -> dict:
        """Increment the iteration counter; the routing edge checks the value."""
        current = state.get("iteration_count") or 0
        return {"iteration_count": current + 1}

    # ------------------------------------------------------------------
    # Node: profile_updater
    # ------------------------------------------------------------------
    def profile_updater_node(state: AgentState) -> dict[str, Any]:
        """
        Extract new user facts from the latest exchange and update the profile.

        Runs after the agent produces a final answer.  Uses a lightweight LLM
        call to identify durable facts (name, topics, preferences) and merges
        them into the existing profile.  Persists the result to disk when a
        session_id is available.
        """
        human_text = _last_human_text(state)
        assistant_text = _last_ai_text(state)

        # Nothing to learn from if either side is empty
        if not human_text or not assistant_text:
            return {}

        current_profile = state.get("user_profile") or {}

        prompt = PROFILE_UPDATER_HUMAN_TEMPLATE.format(
            current_profile=json.dumps(current_profile, indent=2),
            human_message=human_text,
            assistant_message=assistant_text[:1000],  # cap to keep prompt short
        )

        try:
            response = llm.invoke(
                [
                    SystemMessage(content=PROFILE_UPDATER_SYSTEM_PROMPT),
                    HumanMessage(content=prompt),
                ]
            )
            raw = response.content.strip()
            # Strip accidental markdown fences
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            updated_profile: dict[str, Any] = json.loads(raw)
        except Exception as exc:
            logger.warning("Profile updater failed to parse LLM output: %s", exc)
            return {}

        # Persist to disk if we have a session_id
        if session_id:
            save_profile(session_id, updated_profile)

        return {"user_profile": updated_profile}

    return {
        "router": router_node,
        "decline": decline_node,
        "agent": agent_node,
        "tools": tool_node,
        "check_iterations": check_iterations_node,
        "profile_updater": profile_updater_node,
    }


# ---------------------------------------------------------------------------
# Conditional edge helpers (shared between both build functions)
# ---------------------------------------------------------------------------

# After the router, if the query is out_of_scope we route to decline; otherwise to the agent.
def _route_after_router(state: AgentState) -> Literal["decline", "agent"]:
    """Route out-of-scope queries to decline; everything else to the agent."""
    return "decline" if state.get("query_type") == "out_of_scope" else "agent"

# After the agent node, if the last message contains a tool call, we need to execute it; 
# otherwise we can finish with the final answer.
def _route_after_agent(
    state: AgentState,
) -> Literal["tools", "profile_updater"]:
    """
    If the last message contains a tool call, execute the tools.
    Otherwise the agent has produced a final answer — run the profile updater.
    """
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "profile_updater"

# After check_iterations, if we've hit the iteration ceiling we route to END with a fallback message; 
# otherwise we loop back to the agent for another reasoning step.
def _route_after_iterations(
    state: AgentState,
) -> Literal["agent", "profile_updater"]:
    """
    If we've hit the iteration ceiling emit a fallback via profile_updater path;
    otherwise continue the ReAct loop.
    """
    if (state.get("iteration_count") or 0) >= MAX_ITERATIONS:
        logger.warning("Max iterations (%d) reached.", MAX_ITERATIONS)
        return "profile_updater"
    return "agent"


# ---------------------------------------------------------------------------
# Public graph builders
# ---------------------------------------------------------------------------

def build_graph(llm):
    """
    Build and compile the agent graph WITHOUT a checkpointer.

    Useful for one-off calls or testing.

    Parameters
    ----------
    llm:
        A LangChain chat model with tool-calling support.
    """
    nodes = _build_nodes(llm, session_id=None)

    graph = StateGraph(AgentState)
    for name, fn in nodes.items():
        graph.add_node(name, fn)

    graph.set_entry_point("router")
    graph.add_conditional_edges("router", _route_after_router, ["decline", "agent"])
    graph.add_edge("decline", END)
    graph.add_conditional_edges(
        "agent", _route_after_agent, ["tools", "profile_updater"]
    )
    graph.add_edge("tools", "check_iterations")
    graph.add_conditional_edges(
        "check_iterations", _route_after_iterations, ["agent", "profile_updater"]
    )
    graph.add_edge("profile_updater", END)

    return graph.compile()


def build_graph_with_memory(llm, checkpointer, session_id: str | None = None):
    """
    Build the agent graph WITH a persistence checkpointer for multi-turn memory.

    Conversation history is stored by the checkpointer (SQLite or MemorySaver).
    The user profile is stored separately in sessions/profiles/<session_id>.json.

    Parameters
    ----------
    llm:
        A LangChain chat model with tool-calling support.
    checkpointer:
        A LangGraph checkpointer (SqliteSaver for persistence, MemorySaver for tests).
    session_id:
        Session identifier passed to the profile updater for disk persistence.
    """
    nodes = _build_nodes(llm, session_id=session_id)

    graph = StateGraph(AgentState)
    for name, fn in nodes.items():
        graph.add_node(name, fn)

    graph.set_entry_point("router")
    graph.add_conditional_edges("router", _route_after_router, ["decline", "agent"])
    graph.add_edge("decline", END)
    graph.add_conditional_edges(
        "agent", _route_after_agent, ["tools", "profile_updater"]
    )
    graph.add_edge("tools", "check_iterations")
    graph.add_conditional_edges(
        "check_iterations", _route_after_iterations, ["agent", "profile_updater"]
    )
    graph.add_edge("profile_updater", END)

    return graph.compile(checkpointer=checkpointer)