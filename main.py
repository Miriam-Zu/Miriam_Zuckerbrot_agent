"""
main.py

Interactive CLI for the Bitext Customer Service analyst agent.

Usage
-----
    python main.py                        # new ephemeral session
    python main.py --session my_session   # named persistent session

The agent prints every reasoning step (tool calls + observations) and then
the final answer.  

Conversation history is persisted in sessions/checkpoints.db via SqliteSaver.
The user profile is persisted in sessions/profiles/<session_id>.json.
Re-using the same --session name after restarting the app fully restores the
conversation, so follow-up queries like "show me 3 more" work across restarts.

Type 'quit', 'exit', or press Ctrl-C to end the session.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
# from langgraph.checkpoint.memory import MemorySaver
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

load_dotenv()

# Logging — show INFO so reasoning steps are visible
logging.basicConfig(
    level=logging.WARNING,  # suppress library noise
    format="%(levelname)s | %(name)s | %(message)s",
)
logging.getLogger("agent").setLevel(logging.INFO)

console = Console()

SESSION_DIR = Path("sessions") ### Make sure this directory exists for storing checkpoints and profiles


# LLM factory
def build_llm() -> ChatOpenAI:
    """
    Instantiate the main agent LLM via the Nebius Token Factory (OpenAI-compatible).

    Model: meta-llama/Meta-Llama-3.1-70B-Instruct-fast
    Chosen because it offers strong instruction-following and tool-calling
    capability within Nebius's catalogue, at a reasonable speed/cost trade-off.
    """
    api_key = os.getenv("NEBIUS_API_KEY")
    if not api_key:
        console.print(
            "[bold red]Error:[/bold red] NEBIUS_API_KEY environment variable is not set.\n"
            "Add it to your .env file or export it before running."
        )
        sys.exit(1)

    return ChatOpenAI(
        model="meta-llama/Llama-3.3-70B-Instruct",
        openai_api_key=api_key,
        openai_api_base="https://api.tokenfactory.nebius.com/v1/",
        temperature=0,
    )


# Checkpointer factory 
# def build_checkpointer():
#     """
#     Build a SqliteSaver checkpointer that persists conversation state to disk.
 
#     The database is stored at sessions/checkpoints.db.
#     All sessions share one database file; they are separated by thread_id.
 
#     Returns
#     -------
#     SqliteSaver
#         A LangGraph checkpointer backed by SQLite.
#     """

#     # from langgraph.checkpoint.memory import MemorySaver
#     # return MemorySaver()

#     from langgraph.checkpoint.sqlite import SqliteSaver
 
#     SESSION_DIR.mkdir(parents=True, exist_ok=True)
#     db_path = SESSION_DIR / "checkpoints.db"
#     return SqliteSaver.from_conn_string(str(db_path))


 
 
# Session resume detection
def get_message_count(checkpointer, thread_id: str) -> int:
    """
    Return the number of messages already stored for a session thread.
 
    Parameters
    ----------
    checkpointer:
        The SqliteSaver instance.
    thread_id:
        The session/thread identifier.
    """
    try:
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = checkpointer.get(config)
        if snapshot and snapshot.values.get("messages"):
            return len(snapshot.values["messages"])
    except Exception:
        pass
    return 0
 

# Reasoning-step printer
def print_reasoning_steps(events: list[dict]) -> str | None:
    """
    Iterate over streamed graph events and pretty-print reasoning steps.

    Returns the final answer text (last AIMessage without tool calls),
    or None if no answer was produced.

    Parameters
    ----------
    events:
        List of state-update dicts returned by graph.stream().
    """
    final_answer: str | None = None

    for event in events:
        for node_name, node_output in event.items():
            if node_name in ("__start__", "__end__", "profile_updater"):
                continue

            messages = node_output.get("messages", [])

            for msg in messages:
                msg_type = type(msg).__name__

                # ---- Tool call chosen by the agent ----
                if msg_type == "AIMessage" and getattr(msg, "tool_calls", None):
                    console.print(Rule(style="dim cyan"))
                    console.print(
                        Text("Agent reasoning step", style="bold cyan")
                    )
                    for tc in msg.tool_calls:
                        tool_text = Text()
                        tool_text.append("  Tool: ", style="bold")
                        tool_text.append(tc["name"], style="green")
                        tool_text.append("  Args: ", style="bold")
                        tool_text.append(str(tc["args"]), style="yellow")
                        console.print(tool_text)

                # ---- Tool observation / result ----
                elif msg_type == "ToolMessage":
                    console.print(Rule(style="dim yellow"))
                    console.print(
                        Text(f"Tool result: {msg.name}", style="bold yellow")
                    )
                    # Truncate very long outputs for readability
                    content = str(msg.content)
                    if len(content) > 800:
                        content = content[:800] + "\n  … [truncated]"
                    console.print(f"  {content}", style="dim")

                # ---- Final answer (AIMessage without tool calls) ----
                elif msg_type == "AIMessage" and not getattr(msg, "tool_calls", None):
                    final_answer = msg.content

    return final_answer


# Main conversation loop
# def run_cli(session_id: str) -> None:
#     """
#     Start the interactive CLI conversation loop.
#     Loads conversation history and user profile for the given session,
#     then enters a REPL that processes one user message per iteration.

#     Parameters
#     ----------
#     session_id:
#         Unique identifier for this session; used by the checkpointer so
#         conversation history persists across restarts when re-supplied.
#     """
#     from agent.graph import build_graph_with_memory
#     from agent.profile import load_profile, is_empty_profile
#     from agent.prompts import MAX_ITERATIONS_MESSAGE

#     checkpointer = build_checkpointer()
#     msg_count = get_message_count(checkpointer, session_id)
#     is_resuming = msg_count > 0
 
#     # Load user profile from disk
#     user_profile = load_profile(session_id)
#     profile_is_new = is_empty_profile(user_profile)
 
#     llm = build_llm()
#     graph = build_graph_with_memory(llm, checkpointer, session_id = session_id)
#     config = {"configurable": {"thread_id": session_id}}

#     # ---- Welcome banner ----
#     if is_resuming:
#         resume_note = (
#             f"[green]Resuming session[/green] [bold]'{session_id}'[/bold] "
#             f"([dim]{msg_count} messages in history[/dim])"
#         )
#         if not profile_is_new:
#             from agent.profile import build_profile_context
#             profile_summary = build_profile_context(user_profile)
#             resume_note += f"\n[dim]{profile_summary}[/dim]"
#     else:
#         resume_note = f"[dim]New session: {session_id}[/dim]"

#     console.print(
#         Panel.fit(
#             "[bold white]Bitext Customer Service Analyst[/bold white]\n"
#             + resume_note
#             + "[dim]Type 'exit' or 'quit' to end the session.[/dim]",
#             border_style="blue",
#         )
#     )


#     # Eagerly load the dataset so the first query doesn't stall
#     from data.loader import get_dataframe
#     get_dataframe()
#     console.print("[green] Dataset ready.[/green]\n")

#     while True:
#         try:
#             user_input = console.input("[bold blue]You:[/bold blue] ").strip()
#         except (EOFError, KeyboardInterrupt):
#             console.print("\n[dim]Goodbye![/dim]")
#             break

#         if not user_input:
#             continue
#         if user_input.lower() in {"exit", "quit"}:
#             console.print("[dim]Goodbye![/dim]")
#             break

#         from langchain_core.messages import HumanMessage
#         # Pass the current profile into the initial state 
#         # so the agent node can inject it into the system prompt from the very first turn.
#         initial_state = {
#             "messages": [HumanMessage(content=user_input)],
#             "user_profile": user_profile,
#         }

#         console.print()
#         try:
#             events = list(
#                 graph.stream(initial_state, config=config, stream_mode="updates")
#             )
#         except Exception as exc:
#             console.print(f"[bold red]Error during agent execution:[/bold red] {exc}")
#             continue

#         final_answer = print_reasoning_steps(events)

#         # Refresh the in-memory profile from the latest state snapshot
#         # (the profile_updater node may have updated it)
#         try:
#             snapshot = checkpointer.get(config)
#             if snapshot and snapshot.values.get("user_profile"):
#                 user_profile = snapshot.values["user_profile"]
#         except Exception:
#             pass

#         console.print(Rule(style="bold blue"))
#         if final_answer:
#             console.print(Text("Agent:", style="bold blue"))
#             console.print(Markdown(final_answer))
#         else:
#             console.print(
#                 Text("Agent:", style="bold blue"),
#                 MAX_ITERATIONS_MESSAGE,
#             )
#         console.print()


def run_cli(session_id: str) -> None:
    """
    Start the interactive CLI conversation loop.
 
    Loads conversation history and user profile for the given session,
    then enters a REPL that processes one user message per iteration.
 
    Parameters
    ----------
    session_id:
        Unique identifier for this session.  Re-using the same ID across
        restarts restores full conversation history and user profile.
    """
    from agent.graph import build_graph_with_memory
    from agent.profile import load_profile, is_empty_profile, build_profile_context
    from agent.prompts import MAX_ITERATIONS_MESSAGE
    from langchain_core.messages import HumanMessage
    from langgraph.checkpoint.sqlite import SqliteSaver
 
    llm = build_llm()
    user_profile = load_profile(session_id)
    profile_is_new = is_empty_profile(user_profile)
 
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    db_path = str(SESSION_DIR / "checkpoints.db")
    # checkpointer = SqliteSaver.from_conn_string(db_path)
 
    # msg_count = get_message_count(checkpointer, session_id)
    with SqliteSaver.from_conn_string(db_path) as checkpointer:
        msg_count = get_message_count(checkpointer, session_id)
        is_resuming = msg_count > 0
    
        graph = build_graph_with_memory(llm, checkpointer, session_id=session_id)
        config = {"configurable": {"thread_id": session_id}}
    
        # ---- Welcome banner ----
        if is_resuming:
            resume_note = (
                "[green]Resuming session[/green] [bold]'" + session_id + "'[/bold] "
                "([dim]" + str(msg_count) + " messages in history[/dim])"
            )
            if not profile_is_new:
                profile_summary = build_profile_context(user_profile)
                resume_note += "\n[dim]" + profile_summary + "[/dim]"
        else:
            resume_note = "[dim]New session: " + session_id + "[/dim]"
    
        console.print(
            Panel.fit(
                "[bold white]Bitext Customer Service Analyst[/bold white]\n"
                + resume_note
                + "\n[dim]Type 'exit' or 'quit' to end the session.[/dim]",
                border_style="blue",
            )
        )
    
        console.print("[dim]Loading dataset… (first run downloads from HuggingFace)[/dim]")
        from data.loader import get_dataframe
        get_dataframe()
        console.print("[green]✓ Dataset ready.[/green]\n")
    
        while True:
            try:
                user_input = console.input("[bold blue]You:[/bold blue] ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Goodbye![/dim]")
                break
    
            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit"}:
                console.print("[dim]Goodbye![/dim]")
                break
    
            initial_state = {
                "messages": [HumanMessage(content=user_input)],
                "user_profile": user_profile,
            }
    
            console.print()
            try:
                events = list(
                    graph.stream(initial_state, config=config, stream_mode="updates")
                )
            except Exception as exc:
                console.print("[bold red]Error during agent execution:[/bold red] " + str(exc))
                continue
    
            final_answer = print_reasoning_steps(events)
    
            # Refresh the in-memory profile from the latest state snapshot
            try:
                snapshot = checkpointer.get(config)
                if snapshot and snapshot.values.get("user_profile"):
                    user_profile = snapshot.values["user_profile"]
            except Exception:
                pass
    
            console.print(Rule(style="bold blue"))
            if final_answer:
                console.print(Text("🤖  Agent:", style="bold blue"))
                console.print(Markdown(final_answer))
            else:
                console.print(Text("🤖  Agent:", style="bold blue"))
                console.print(MAX_ITERATIONS_MESSAGE)
            console.print()
 
 

# Entry point
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Bitext Customer Service data analyst agent"
    )
    parser.add_argument(
        "--session",
        type=str,
        default=None,
        help=(
            "Session ID for persistent memory.  "
            "Re-using the same ID restores conversation history and user profile."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    session_id = args.session or str(uuid.uuid4())
    run_cli(session_id)