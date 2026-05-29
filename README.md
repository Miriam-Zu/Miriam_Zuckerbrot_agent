# Miriam_Zuckerbrot_agent

# Bitext Customer Service Analyst Agent

A LangGraph-based ReAct agent that answers questions about the [Bitext Customer Service dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset). Supports structured queries, open-ended summarisation, and gracefully declines out-of-scope questions.

---

## Table of Contents

- [Setup](#setup)
- [How to Run the CLI](#how-to-run-the-cli)
- [Example Queries](#example-queries)
- [Architecture Overview](#architecture-overview)
- [Tools](#tools)
- [Project Structure](#project-structure)

---

## Setup

### Prerequisites

- Python 3.11+
- A [Nebius Token Factory](https://studio.nebius.com/) API key
- Git

### 1. Clone the repository

```bash
git clone https://github.com/Miriam-Zu/Miriam_Zuckerbrot_agent.git
cd your-repo-name
```

### 2. Create and activate a virtual environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set your API key

Create a `.env` file in the project root:

```
NEBIUS_API_KEY=your_key_here
```

> **Note:** The `.env` file is listed in `.gitignore` and will never be committed to GitHub.

### 5. Dataset

The Bitext dataset is downloaded automatically from HuggingFace on first run (~27,000 rows). No manual download is needed. It is cached in memory for the duration of the session.

---

## How to Run the CLI

### Start a new session

```bash
python main.py
```

Each run without `--session` creates a new ephemeral session with a random ID.

### Start or resume a named session

```bash
python main.py --session my_session
```

Using the same session name after restarting the app **restores the full conversation history**, so follow-up questions like "show me 3 more" work across restarts.

### What you'll see

The CLI prints every reasoning step as it happens:

```
You: How many refund requests did we get?

──────────────────────────────────────────
🔧  Agent reasoning step
  Tool: list_intents   Args: {'category': 'REFUND'}

──────────────────────────────────────────
📊  Tool result: list_intents
  {'intents': ['get_refund', 'track_refund'], 'count': 2}

──────────────────────────────────────────
🔧  Agent reasoning step
  Tool: count_rows   Args: {'intent': 'get_refund'}

──────────────────────────────────────────
📊  Tool result: count_rows
  {'count': 1380, 'intent': 'get_refund'}

══════════════════════════════════════════
🤖  Agent:
There were **1,380 refund requests** (rows with intent `get_refund`) in the dataset.
```

Type `exit` or `quit` (or press `Ctrl+C`) to end the session.

---

## Example Queries

### Structured — concrete data questions

| Query | What the agent does |
|---|---|
| `What categories exist in the dataset?` | Calls `list_categories` |
| `How many refund requests did we get?` | Calls `list_intents` → `count_rows` |
| `Show me 5 examples of the SHIPPING category` | Calls `filter_rows` |
| `What is the distribution of intents in the ACCOUNT category?` | Calls `get_distribution` |
| `Show me examples of people wanting their money back` | Calls `search_by_keyword` |

### Unstructured — open-ended summarisation

| Query | What the agent does |
|---|---|
| `Summarise the FEEDBACK category` | Calls `summarise_samples` → synthesises answer |
| `How do agents respond to cancellation requests?` | Calls `summarise_samples` with focus on responses |
| `Summarise how agents handle complaint intents` | Calls `list_intents` → `summarise_samples` |

### Out of scope — politely declined

| Query | Response |
|---|---|
| `Who won the 2024 Champions League?` | Polite decline — not dataset-related |
| `Write me a poem about customer service` | Polite decline — not dataset-related |
| `What's the best CRM for handling complaints?` | Polite decline — not dataset-related |

---

## Architecture Overview

### Graph topology

```
User input
    │
    ▼
[router] ──── out_of_scope ────► [decline] ──► END
    │
    │ structured / unstructured
    ▼
[agent] ◄─────────────────────────────────────┐
    │                                          │
    │ tool call                                │ (under limit)
    ▼                                          │
[tools] ──► [check_iterations] ───────────────┘
                    │
                    │ over limit
                    ▼
                   END (fallback message)
    │
    │ no tool call
    ▼
   END (final answer)
```

### Key design decisions

**Router as a dedicated node, not a prompt instruction.**
The router runs a separate LLM call *before* the ReAct loop starts. Out-of-scope queries are hard-blocked at this stage — the agent never sees them, no tools are ever called. This prevents the LLM from "helpfully" answering general knowledge questions it shouldn't.

**ReAct loop with full message history.**
On every pass through the agent node, the LLM receives the complete conversation history: the original question, every tool call it made, and every tool result it received. This is what enables multi-step reasoning — the model can see what it already tried and decide what to do next.

**`summarise_samples` bridges structured and unstructured.**
Rather than having the LLM hallucinate summaries, `summarise_samples` fetches real rows from the dataset and returns them as raw text. The LLM then synthesises a grounded answer from actual data.

**Max iterations guard.**
The agent loop is capped at 12 iterations. If the agent hasn't produced a final answer by then, a graceful fallback message is returned instead of looping forever.

**Persistent memory via `MemorySaver`.**
The `--session` flag binds the graph to a named thread in a `MemorySaver` checkpointer. The same session ID always restores the same conversation, enabling follow-up queries like "show me 3 more" or "what about refunds?" to resolve against prior context.

### Model choice

| Role | Model |
|---|---|
| Router | `meta-llama/Meta-Llama-3.1-70B-Instruct-fast` |
| ReAct agent | `meta-llama/Meta-Llama-3.1-70B-Instruct-fast` |

**Why this model?**
`Meta-Llama-3.1-70B-Instruct-fast` is available on Nebius Token Factory and offers strong instruction-following, reliable tool/function calling, and good reasoning quality. The 70B size hits a practical sweet spot: capable enough for multi-step tool chaining and open-ended summarisation, while fast enough for an interactive CLI. A smaller model (e.g. 8B) was considered for the router, but the classification task benefits from the same model's consistency, and the router call is cheap (single short prompt).

---

## Tools

Six tools are available to the agent. Each has a Pydantic input schema so the LLM knows exactly what arguments to provide.

| Tool | Purpose | Key inputs |
|---|---|---|
| `list_categories` | List all unique category names | — |
| `list_intents` | List intent labels, optionally filtered by category | `category` |
| `filter_rows` | Return sample rows filtered by category / intent | `category`, `intent`, `n_samples` |
| `count_rows` | Count rows matching a filter | `category`, `intent` |
| `get_distribution` | Intent breakdown (counts + percentages) within a category | `category` |
| `summarise_samples` | Fetch raw text samples for LLM summarisation | `category`, `intent`, `n_samples`, `focus` |
| `search_by_keyword` | Find rows whose customer message contains a keyword | `keyword`, `n_samples` |

---

## Project Structure

```
bitext-agent/
├── main.py               # CLI entry point and conversation loop
├── requirements.txt      # Pinned dependencies
├── .env                  # API key (not committed)
├── .gitignore
│
├── agent/
│   ├── __init__.py
│   ├── graph.py          # LangGraph StateGraph definition
│   ├── router.py         # Query classification node
│   ├── state.py          # AgentState TypedDict
│   ├── tools.py          # All tool definitions with Pydantic schemas
│   └── prompts.py        # All prompt strings
│
└── data/
    ├── __init__.py
    └── loader.py         # Dataset download and caching
```

---

## Memory (Task 2)

### 2a — Conversation Memory

Conversation history is persisted across restarts using **SqliteSaver**, a LangGraph checkpointer backed by a local SQLite database:

```
sessions/
└── checkpoints.db    ← all sessions, separated by thread_id
```

Every turn is checkpointed automatically. Restarting the app and passing the same `--session` name fully restores the conversation — the agent sees all prior messages as if the session never ended.

**Follow-up queries work out of the box** because `AgentState.messages` accumulates the full history via the `add_messages` reducer. The LLM sees every prior exchange on each new turn, so references like "show me 3 more" or "what about the last two counts?" resolve naturally.

Example multi-turn session:

```
python main.py --session demo

You: Show me 3 examples from the REFUND category
🤖  [shows 3 rows]

You: Show me 3 more
🤖  [shows 3 different rows — agent remembers what it already showed]

You: How many complaints did we get?
🤖  There were 1,200 complaints.

You: What about refunds?
🤖  There were 1,380 refund requests.

You: What is the total count of the last two?
🤖  The total is 2,580 (1,200 complaints + 1,380 refunds).

# Restart the app — history is preserved:
python main.py --session demo
# Banner shows: "Resuming session 'demo' (12 messages in history)"
You: Remind me what categories we discussed
🤖  We looked at COMPLAINT and REFUND ...
```

### 2b — User Profile

A lightweight user profile is maintained **separately** from conversation history. It captures distilled facts — not a replay of messages:

```
sessions/
└── profiles/
    └── my_session.json
```

**Profile structure:**
```json
{
  "name": "Yoni",
  "topics_of_interest": ["REFUND", "SHIPPING"],
  "preferences": "prefers concise answers",
  "other_facts": [],
  "last_updated": "2025-01-15T10:32:00"
}
```

**How it works:**

After every agent response, a `profile_updater` node runs a lightweight LLM call that extracts any new durable facts from the exchange (name mentions, topics queried, stated preferences) and merges them into the profile JSON. The profile is then injected into the agent's system prompt at the start of each turn.

**Asking about the profile:**
```
You: My name is Yoni, I'm mostly interested in refund patterns
🤖  Got it! I'll keep that in mind ...

You: What do you remember about me?
🤖  Here's what I know about you:
    - Name: Yoni
    - You frequently ask about: REFUND
    - Preferences: interested in refund patterns
```

The profile persists across restarts. When resuming a session, the welcome banner shows the current profile summary.