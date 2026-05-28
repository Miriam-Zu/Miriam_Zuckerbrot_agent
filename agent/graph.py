"""
agent/graph.py

Assembles the full LangGraph ReAct agent graph.

Graph topology
--------------

  [router] ──out_of_scope──► [decline] ──► END
     │
  structured / unstructured
     │
     ▼
  [agent] ──── tool_call ────► [tools] ──► [check_iterations]
     ▲                                           │
     │                   under limit ◄───────────┘
     │                                           │
     └───────────────────────────────────────────┘
                                    │
                             over limit ──► END (fallback)
  [agent] ── no tool call ──► END (final answer)
"""

from __future__ import annotations

import logging
from typing import Literal

from langchain_core.messages import AIMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from agent.prompts import (
    AGENT_SYSTEM_PROMPT,
    OUT_OF_SCOPE_MESSAGE,
    MAX_ITERATIONS_MESSAGE,
)
from agent.router import build_router_node
from agent.state import AgentState
from agent.tools import ALL_TOOLS

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 12


def build_graph(llm):
    """
    Build and compile the full agent StateGraph.

    Parameters
    ----------
    llm:
        A LangChain chat model.  Must support tool-calling (bind_tools).

    Returns
    -------
    CompiledGraph
        A compiled LangGraph graph ready to invoke or stream.
    """
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    # Node: router
    router_node = build_router_node(llm)

    # Node: decline (out-of-scope terminal)
    def decline_node(state: AgentState) -> dict:
        """Return a polite refusal for out-of-scope queries."""
        return {"messages": [AIMessage(content=OUT_OF_SCOPE_MESSAGE)]}

    # Node: agent (ReAct reasoning step)
    def agent_node(state: AgentState) -> dict:
        """
        Run one ReAct reasoning step.

        Injects the query_type into the system prompt so the model knows
        whether to focus on data retrieval or text synthesis.
        """
        from langchain_core.messages import SystemMessage

        query_type = state.get("query_type") or "structured"
        system_msg = SystemMessage(
            content=AGENT_SYSTEM_PROMPT.format(query_type=query_type)
        )

        # The agent needs to see the full conversation history to make informed decisions, 
        # we inject the system prompt at the front so it doesn't get buried in the middle of the messages.
        # The LLM decides by itself when it has enough information to stop calling tools and give a final answer.
        all_messages = [system_msg] + list(state["messages"])
        response = llm_with_tools.invoke(all_messages)
        return {"messages": [response]}

    # Node: tools (executes the tool call chosen by the agent)
    tool_node = ToolNode(ALL_TOOLS)

    # Node: check_iterations (enforces the max-iterations guard)
    def check_iterations_node(state: AgentState) -> dict:
        """Increment the iteration counter; the routing edge checks the value."""
        current = state.get("iteration_count") or 0
        return {"iteration_count": current + 1}

    # Conditional edge helpers

    # After the router, if the query is out_of_scope we route to decline; otherwise to the agent.
    def route_after_router(
        state: AgentState,
    ) -> Literal["decline", "agent"]:
        """Route out-of-scope queries to decline; everything else to the agent."""
        return "decline" if state.get("query_type") == "out_of_scope" else "agent"

    # After the agent node, if the last message contains a tool call, 
    # we need to execute it; 
    # otherwise we can finish with the final answer.
    def route_after_agent(
        state: AgentState,
    ) -> Literal["tools", "__end__"]:
        """If the last message contains a tool call, run the tools; otherwise finish."""
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return "__end__"

    # After check_iterations, if we've hit the iteration ceiling we route to END with a fallback message; 
    # otherwise we loop back to the agent for another reasoning step.
    def route_after_iterations(
        state: AgentState,
    ) -> Literal["agent", "__end__"]:
        """If we've hit the iteration ceiling, return a fallback message; else continue."""
        if (state.get("iteration_count") or 0) >= MAX_ITERATIONS:
            logger.warning("Max iterations (%d) reached.", MAX_ITERATIONS)
            return "__end__"
        return "agent"

    # Build the graph
    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("decline", decline_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_node("check_iterations", check_iterations_node)

    graph.set_entry_point("router")

    graph.add_conditional_edges("router", route_after_router, ["decline", "agent"])
    graph.add_edge("decline", END)

    graph.add_conditional_edges("agent", route_after_agent, ["tools", END])
    graph.add_edge("tools", "check_iterations")
    
    graph.add_conditional_edges("check_iterations", route_after_iterations, ["agent", END])

    compiled = graph.compile()
    return compiled


def build_graph_with_memory(llm, checkpointer):
    """
    Build the graph with a persistence checkpointer for multi-turn memory.

    Parameters
    ----------
    llm:
        A LangChain chat model with tool-calling support.
    checkpointer:
        A LangGraph checkpointer (e.g. MemorySaver or SqliteSaver).

    Returns
    -------
    CompiledGraph
        A compiled LangGraph graph with memory enabled.
    """
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    router_node = build_router_node(llm)

    def decline_node(state: AgentState) -> dict:
        return {"messages": [AIMessage(content=OUT_OF_SCOPE_MESSAGE)]}

    def agent_node(state: AgentState) -> dict:
        from langchain_core.messages import SystemMessage

        query_type = state.get("query_type") or "structured"
        system_msg = SystemMessage(content=AGENT_SYSTEM_PROMPT.format(query_type=query_type))
        all_messages = [system_msg] + list(state["messages"])
        response = llm_with_tools.invoke(all_messages)
        return {"messages": [response]}

    tool_node = ToolNode(ALL_TOOLS)

    def check_iterations_node(state: AgentState) -> dict:
        current = state.get("iteration_count") or 0
        return {"iteration_count": current + 1}

    def route_after_router(state: AgentState) -> Literal["decline", "agent"]:
        return "decline" if state.get("query_type") == "out_of_scope" else "agent"

    def route_after_agent(state: AgentState) -> Literal["tools", "__end__"]:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return "__end__"

    def route_after_iterations(state: AgentState) -> Literal["agent", "__end__"]:
        if (state.get("iteration_count") or 0) >= MAX_ITERATIONS:
            logger.warning("Max iterations (%d) reached.", MAX_ITERATIONS)
            return "__end__"
        return "agent"

    graph = StateGraph(AgentState)
    graph.add_node("router", router_node)
    graph.add_node("decline", decline_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_node("check_iterations", check_iterations_node)

    graph.set_entry_point("router")
    
    graph.add_conditional_edges("router", route_after_router, ["decline", "agent"])
    graph.add_edge("decline", END)
    
    graph.add_conditional_edges("agent", route_after_agent, ["tools", END])
    graph.add_edge("tools", "check_iterations")
    
    graph.add_conditional_edges("check_iterations", route_after_iterations, ["agent", END])

    compiled = graph.compile(checkpointer=checkpointer)
    return compiled