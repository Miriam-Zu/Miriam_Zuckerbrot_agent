"""
agent/router.py

Implements the query-router node.  This node runs before the ReAct agent loop
and classifies the incoming user query into one of three types:

  - structured   : concrete data questions (counts, lists, distributions)
  - unstructured : open-ended questions requiring text synthesis
  - out_of_scope : questions unrelated to the Bitext dataset

The classification is stored in AgentState.query_type 
and drives the conditional edge that either sends the query to the agent 
or immediately returns a polite decline.
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from agent.prompts import ROUTER_SYSTEM_PROMPT, ROUTER_HUMAN_TEMPLATE
from agent.state import AgentState, QueryType

logger = logging.getLogger(__name__)

_VALID_LABELS: set[str] = {"structured", "unstructured", "out_of_scope"}


def build_router_node(llm):
    """
    Factory that returns a router node function bound to the given LLM.

    The returned function is compatible with LangGraph's node signature:
    it receives the current AgentState and returns a partial-state update dict.

    Parameters
    ----------
    llm:
        A LangChain chat model instance used to classify queries.
        A small / fast model is preferred here since the task is simple.
    """

    def router_node(state: AgentState) -> dict:
        """
        Classify the latest user message and update query_type in state.

        Reads the last HumanMessage from state.messages, 
        sends it to the router LLM, and parses the returned label.  
        Falls back to 'structured' if the model returns an unrecognised label.
        """
        # Extract the most recent human message text
        last_human = next(
            (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            None,
        )
        if last_human is None:
            logger.warning("Router found no HumanMessage; defaulting to structured.")
            return {"query_type": "structured", "iteration_count": 0}

        query_text = (
            last_human.content
            if isinstance(last_human.content, str)
            else str(last_human.content)
        )

        messages = [
            SystemMessage(content=ROUTER_SYSTEM_PROMPT),
            HumanMessage(content=ROUTER_HUMAN_TEMPLATE.format(query=query_text)),
        ]

        response = llm.invoke(messages)
        raw_label = response.content.strip().lower()

        if raw_label not in _VALID_LABELS:
            logger.warning(
                "Router returned unexpected label %r; defaulting to 'structured'.",
                raw_label,
            )
            raw_label = "structured"

        query_type: QueryType = raw_label  # type: ignore[assignment]
        logger.info("Router classified query as: %s", query_type)

        return {"query_type": query_type, "iteration_count": 0}

    return router_node