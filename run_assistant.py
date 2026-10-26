"""
Phase 7: LangGraph orchestration layer on top of the retrieval pipeline.

Flow (per user turn):

    reformulate_query       -> make the query standalone (uses history)
        |
    classify_query          -> intent + per-intent graph weight
        |
    retrieve                -> RetrievalResult + rendered context
        |
    generate                -> grounded answer
        |
    verify                  -> JSON grounding check
        |
    [retry | END]           -> if ungrounded, reformulate answer and retry once

State is persisted per thread_id (conversation). The CLI runs an
interactive REPL so you can ask follow-up queries in the same thread.
"""

import json
import os
import re

from typing import TypedDict, Literal

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from groq import Groq

from retrieval import (
    GraphRAGPipeline,
    GROQ_API_KEY,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MAX_RETRIES = 1
MIN_ANSWER_CHARS = 40
MAX_HISTORY_TURNS = 10

groq_client = Groq(api_key=GROQ_API_KEY)


# ---------------------------------------------------------------------------
# Per-intent graph weight table
# ---------------------------------------------------------------------------

INTENT_GRAPH_WEIGHTS: dict[str, float] = {
    "factual":      0.3,
    "procedural":   0.9,
    "safety":       1.0,
    "comparison":   0.7,
    "troubleshoot": 0.9,
    "out_of_scope": 0.0,
}


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

REFORMULATOR_SYSTEM_PROMPT = """You are a query rewriter for a pump manual Q&A system.

Your job: decide whether the user's new query can be answered standalone,
or whether it depends on the conversation history for context. If it
depends on history, rewrite it into a self-contained query. If it does
not, return it unchanged.

Rules:
- Return the query UNCHANGED if it is already self-contained.
  Examples of self-contained queries:
    "What are the warnings for the coupling?"
    "How do I start the pump?"
    "What is the max pressure of HL125M?"
- REWRITE the query if it depends on prior turns. Signals:
  - Pronouns with no antecedent in the current query ("it", "its", "this",
    "that", "them", "they").
  - Elliptical follow-ups ("what about the impeller?", "and for HL125M?",
    "does it need lubrication?").
  - Bare references to concepts introduced earlier ("the same one",
    "the other one").
  - Short queries that only make sense with context.
- When rewriting, produce ONE self-contained query. Combine the prior
  turn's subject with the new intent. Examples:
  - Prior: "What warnings apply to the coupling?"
    New:   "Does it need lubrication?"
    Rewrite: "Does the coupling need lubrication?"
  - Prior: "What about the impeller?"
    New:   "Does it need lubrication?"
    Rewrite: "Does the impeller need lubrication?"
  - Prior: "How do I start the pump?"
    New:   "What about the air compressor?"
    Rewrite: "How do I start the air compressor?"
- If the new query mentions a NEW subject with no dependency on history,
  return it unchanged even if it is short:
  - Prior: "What warnings apply to the coupling?"
    New:   "How do I align the pump?"
    Rewrite: "How do I align the pump?"  (self-contained, no history needed)
- Never invent details that are not present in either the new query or
  the conversation history.
- Preserve the user's original intent and wording as much as possible.

Respond with JSON only:
{
  "rewritten": "<the standalone query>",
  "changed": true | false,
  "reason": "<brief explanation>"
}
"""


CLASSIFIER_SYSTEM_PROMPT = """You are a query classifier for a pump operation and maintenance manual Q&A system.

Classify the user's query into exactly one of these categories:

- "factual": asking for a specific value, spec, part name, or definition.
- "procedural": asking how to perform a task or procedure.
- "safety": asking about warnings, hazards, or safety precautions.
- "comparison": comparing items, or asking for highest/lowest/best.
- "troubleshoot": asking why something is wrong or how to fix a problem.
- "out_of_scope": the query is not about the pump manual at all.

The query you receive is already self-contained. Classify it on its own
merits without referencing prior conversation.

Respond with JSON only:
{"intent": "<category>"}
"""


VERIFIER_SYSTEM_PROMPT = """You are a grounding verifier for a pump manual Q&A system.

You will receive a QUESTION, an ANSWER, and the CONTEXT that was provided
to the answerer. Identify any claim in the answer that is not fully
supported by the context.

Respond with JSON only:
{
  "grounded": true | false,
  "issues": [
    {
      "claim": "the specific sentence or phrase from the answer",
      "problem": "not_in_context" | "contradicts_context" | "overstated" | "citation_mismatch",
      "explanation": "brief explanation"
    }
  ]
}

Rules:
- Numerical values must match exactly (including units).
- Warning levels (DANGER/WARNING/CAUTION/NOTICE) must match exactly.
- Paraphrases that preserve meaning are fine.
"""


ANSWER_REFORMULATOR_SYSTEM_PROMPT = """You rewrite pump-manual answers that failed a grounding check.

You will receive the original query, the previous answer, and the list
of grounding issues. Produce a corrected answer that:
- Removes or corrects every issue.
- Keeps any parts of the original answer that were correct.
- Still only uses information from the provided context.

Respond with JSON only:
{"answer": "<corrected answer>"}
"""


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class AgentState(TypedDict, total=False):
    # input
    user_query: str            # the raw query the user typed
    standalone_query: str      # user_query after reformulation if needed

    # routing
    intent: str

    # retrieval
    retrieval: dict
    context: str

    # generation
    answer: str

    # verification
    verified: bool
    verify_issues: list[str]

    # retry
    retry_count: int
    reformulation_reason: str  # why the query was rewritten (empty if unchanged)

    # memory
    conversation_history: list[dict]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_history_for_prompt(history: list[dict]) -> str:
    """
    Render the last N turns as a plain-text block. Used only by the
    reformulator node. Every other node sees only the standalone query.
    """
    if not history:
        return "(no prior turns)"

    lines: list[str] = []
    for turn in history[-MAX_HISTORY_TURNS:]:
        q = (turn.get("query") or "").strip()
        a = (turn.get("answer") or "").strip()
        # Trim the answer — first 300 chars are enough for reformulation.
        a_short = a[:300] + ("…" if len(a) > 300 else "")
        lines.append(f"User: {q}")
        lines.append(f"Assistant: {a_short}")
        lines.append("")

    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# Node 0: reformulate_query  (NEW — runs first on every turn)
# ---------------------------------------------------------------------------

def reformulate_query_node(state: AgentState) -> dict:
    """
    Decide whether the user's raw query is self-contained or depends on
    history. If it depends, rewrite it into a standalone query. If not,
    pass it through unchanged.

    This is the ONLY node that ever sees the full conversation history.
    All downstream nodes operate on the standalone query.
    """
    user_query = state.get("user_query") or state.get("query") or ""
    history = state.get("conversation_history") or []

    # Fast path: no history → nothing to reformulate against.
    if not history:
        return {
            "standalone_query": user_query,
            "reformulation_reason": "",
            "retry_count": 0,
        }

    history_block = _format_history_for_prompt(history)

    try:
        resp = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": REFORMULATOR_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Conversation history:\n{history_block}\n\n"
                        f"New user query: {user_query}"
                    ),
                },
            ],
            temperature=0.0,
            max_completion_tokens=256,
            response_format={"type": "json_object"},
            reasoning_effort="low",
            reasoning_format="hidden",
        )
        payload = json.loads(resp.choices[0].message.content or "{}")

        rewritten = (payload.get("rewritten") or user_query).strip()
        changed = bool(payload.get("changed", False))
        reason = (payload.get("reason") or "").strip()

        # Safety: if the model returns empty or garbled, fall back.
        if not rewritten:
            rewritten = user_query
            changed = False

        return {
            "standalone_query": rewritten,
            "reformulation_reason": reason if changed else "",
            "retry_count": 0,
        }

    except Exception as e:
        print(f"[reformulate_query] failed, using raw query: {e}")
        return {
            "standalone_query": user_query,
            "reformulation_reason": "",
            "retry_count": 0,
        }


# ---------------------------------------------------------------------------
# Node 1: classify_query
# ---------------------------------------------------------------------------

def classify_query_node(state: AgentState) -> dict:
    """
    Classify the standalone query. No history is passed — the query is
    already self-contained by the time it reaches this node.
    """
    query = state.get("standalone_query") or state.get("user_query") or ""

    try:
        resp = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0.0,
            max_completion_tokens=64,
            response_format={"type": "json_object"},
            reasoning_effort="low",
            reasoning_format="hidden",
        )
        payload = json.loads(resp.choices[0].message.content or "{}")
        intent = payload.get("intent", "factual")
    except Exception as e:
        print(f"[classify] falling back to 'factual': {e}")
        intent = "factual"

    if intent not in INTENT_GRAPH_WEIGHTS:
        intent = "factual"

    return {"intent": intent}


# ---------------------------------------------------------------------------
# Node 2: retrieve
# ---------------------------------------------------------------------------

def retrieve_node(state: AgentState) -> dict:
    """
    Retrieve using the standalone query. No history heuristics.
    """
    pipeline: GraphRAGPipeline = state["_pipeline"]  # type: ignore
    query = state.get("standalone_query") or state.get("user_query") or ""
    intent = state.get("intent", "factual")

    graph_weight = INTENT_GRAPH_WEIGHTS.get(intent, 0.5)

    if intent == "out_of_scope":
        return {
            "retrieval": {"chunks": [], "chunk_ids": [], "warnings": [], "steps": []},
            "context": "",
        }

    result, context = pipeline.retrieve(query, graph_weight=graph_weight)

    return {
        "retrieval": {
            "chunk_ids": result.chunk_ids,
            "linked_entities": result.linked_entities,
            "used_graph": result.used_graph,
            "warnings_count": len(result.warnings),
            "steps_count": len(result.steps),
        },
        "context": context,
    }


# ---------------------------------------------------------------------------
# Node 3: generate
# ---------------------------------------------------------------------------

def generate_node(state: AgentState) -> dict:
    """
    Generate an answer using the standalone query. No history is injected
    into the context — the query already carries whatever context it needs.
    """
    pipeline: GraphRAGPipeline = state["_pipeline"]  # type: ignore
    query = state.get("standalone_query") or state.get("user_query") or ""
    context = state.get("context", "")

    if state.get("intent") == "out_of_scope":
        return {
            "answer": (
                "This question is outside the scope of the HL Series pump "
                "manual. I can only answer questions about the pump, its "
                "operation, maintenance, and safety."
            )
        }

    if not context.strip():
        return {
            "answer": (
                "The manual does not contain information needed to answer "
                "this question. Please rephrase or consult a Xylem "
                "representative."
            )
        }

    answer = pipeline.generate(query, context)
    return {"answer": answer}


# ---------------------------------------------------------------------------
# Node 4: verify
# ---------------------------------------------------------------------------

def verify_node(state: AgentState) -> dict:
    intent = state.get("intent")
    if intent == "out_of_scope":
        return {"verified": True, "verify_issues": []}

    query = state.get("standalone_query") or state.get("user_query") or ""
    answer = state.get("answer", "")
    context = state.get("context", "")

    if len(answer.strip()) < MIN_ANSWER_CHARS:
        return {
            "verified": False,
            "verify_issues": ["Answer is too short to verify."],
        }

    try:
        resp = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"QUESTION:\n{query}\n\n"
                        f"ANSWER:\n{answer}\n\n"
                        f"CONTEXT:\n{context}"
                    ),
                },
            ],
            temperature=0.0,
            max_completion_tokens=1024,
            response_format={"type": "json_object"},
            reasoning_effort="low",
            reasoning_format="hidden",
        )
        payload = json.loads(resp.choices[0].message.content or "{}")
        grounded = bool(payload.get("grounded", False))
        issues = payload.get("issues", []) or []
        return {"verified": grounded, "verify_issues": issues}
    except Exception as e:
        print(f"[verify] verifier call failed: {e}")
        return {"verified": True, "verify_issues": []}


# ---------------------------------------------------------------------------
# Node 5: reformulate answer (used on retry)
# ---------------------------------------------------------------------------

def reformulate_answer_node(state: AgentState) -> dict:
    """
    Called on retry. Instead of reformulating the query, we reformulate
    the ANSWER — the query is already correct, so the previous answer's
    issues should be fixed by regenerating from the same context with
    explicit feedback from the verifier.
    """
    query = state.get("standalone_query") or state.get("user_query") or ""
    answer = state.get("answer", "")
    context = state.get("context", "")
    issues = state.get("verify_issues", [])

    issue_text = "\n".join(
        f"- {i.get('problem', '?')}: {i.get('claim', '?')} — {i.get('explanation', '')}"
        for i in issues if isinstance(i, dict)
    ) or "(no specific issues listed)"

    try:
        resp = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": ANSWER_REFORMULATOR_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Query: {query}\n\n"
                        f"Previous answer:\n{answer}\n\n"
                        f"Grounding issues:\n{issue_text}\n\n"
                        f"Context (the only source you may use):\n{context}"
                    ),
                },
            ],
            temperature=0.2,
            max_completion_tokens=4096,
            response_format={"type": "json_object"},
            reasoning_effort="medium",
            reasoning_format="hidden",
        )
        payload = json.loads(resp.choices[0].message.content or "{}")
        new_answer = (payload.get("answer") or answer).strip()
    except Exception as e:
        print(f"[reformulate_answer] failed, keeping original: {e}")
        new_answer = answer

    return {
        "answer": new_answer or answer,
        "retry_count": state.get("retry_count", 0) + 1,
        "verified": False,       # will be re-verified in verify node
        "verify_issues": [],
    }


# ---------------------------------------------------------------------------
# Node 6: memory
# ---------------------------------------------------------------------------

def memory_node(state: AgentState) -> dict:
    """
    Append this turn to the conversation history. Stores the user's
    ORIGINAL query (not the standalone rewrite) so the reformulator for
    the next turn sees what the user actually typed.
    """
    history = list(state.get("conversation_history") or [])
    history.append({
        "query": state.get("user_query") or "",
        "standalone_query": state.get("standalone_query") or "",
        "answer": state.get("answer", ""),
        "verified": state.get("verified", True),
        "intent": state.get("intent", "factual"),
    })
    return {"conversation_history": history[-MAX_HISTORY_TURNS:]}


# ---------------------------------------------------------------------------
# Conditional edges
# ---------------------------------------------------------------------------

def route_after_verify(state: AgentState) -> Literal["reformulate_answer", "memory"]:
    if state.get("verified", False):
        return "memory"

    if state.get("intent") == "out_of_scope":
        return "memory"

    retries = state.get("retry_count", 0)
    if retries >= MAX_RETRIES:
        return "memory"

    return "reformulate_answer"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph(pipeline: GraphRAGPipeline):
    builder = StateGraph(AgentState)

    def bound_retrieve(state):
        state = dict(state)
        state["_pipeline"] = pipeline
        return retrieve_node(state)

    def bound_generate(state):
        state = dict(state)
        state["_pipeline"] = pipeline
        return generate_node(state)

    builder.add_node("reformulate_query",  reformulate_query_node)
    builder.add_node("classify",           classify_query_node)
    builder.add_node("retrieve",           bound_retrieve)
    builder.add_node("generate",           bound_generate)
    builder.add_node("verify",             verify_node)
    builder.add_node("reformulate_answer", reformulate_answer_node)
    builder.add_node("memory",             memory_node)

    builder.set_entry_point("reformulate_query")

    builder.add_edge("reformulate_query",  "classify")
    builder.add_edge("classify",           "retrieve")
    builder.add_edge("retrieve",           "generate")
    builder.add_edge("generate",           "verify")
    builder.add_conditional_edges(
        "verify",
        route_after_verify,
        {
            "reformulate_answer": "reformulate_answer",
            "memory":             "memory",
        },
    )
    builder.add_edge("reformulate_answer", "verify")
    builder.add_edge("memory",             END)

    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Conversation session helper
# ---------------------------------------------------------------------------

class ConversationSession:
    def __init__(self, graph, thread_id: str):
        self.graph = graph
        self.thread_id = thread_id
        self.config = {"configurable": {"thread_id": thread_id}}
        self.history: list[dict] = []

    def ask(self, query: str) -> dict:
        final_state = self.graph.invoke(
            {
                "user_query": query,
                "conversation_history": self.history,
            },
            config=self.config,
        )
        self.history = final_state.get("conversation_history", [])
        return final_state

    def reset(self):
        self.history = []
        print("[session] local history cleared; thread state preserved")


# ---------------------------------------------------------------------------
# Interactive REPL
# ---------------------------------------------------------------------------

BANNER = """
==============================================================
  HL Series Pump Manual — GraphRAG Assistant (interactive)
==============================================================
Type your question and press Enter.
Commands:
  /new        start a new conversation (fresh thread)
  /history    print the current conversation history
  /state      dump the full checkpointed state
  /exit       quit
==============================================================
"""


def _print_turn_summary(state: dict) -> None:
    original = state.get("user_query") or ""
    standalone = state.get("standalone_query") or ""
    if standalone and standalone != original:
        print(f"  reformulated: {standalone}")
        reason = state.get("reformulation_reason")
        if reason:
            print(f"  reason      : {reason}")

    print(f"  intent      : {state.get('intent')}")
    print(f"  verified    : {state.get('verified')}")
    if state.get("retry_count"):
        print(f"  retries     : {state.get('retry_count')}")
    issues = state.get("verify_issues") or []
    if issues:
        print(f"  verify warns: {len(issues)}")
    used_graph = state.get("retrieval", {}).get("used_graph")
    if used_graph is not None:
        print(f"  used graph  : {used_graph}")


def run_repl(graph):
    print(BANNER)

    thread_counter = 0
    session = None

    def new_session():
        nonlocal thread_counter, session
        thread_counter += 1
        session = ConversationSession(graph, thread_id=f"session_{thread_counter}")
        print(f"[session] started thread session_{thread_counter}\n")

    new_session()

    while True:
        try:
            line = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[exit]")
            break

        if not line:
            continue

        if line in ("/exit", "/quit"):
            print("[exit]")
            break

        if line == "/new":
            new_session()
            continue

        if line == "/history":
            if not session.history:
                print("(no history)\n")
                continue
            for i, turn in enumerate(session.history, start=1):
                print(f"--- turn {i} ---")
                print(f"Q:  {turn.get('query')}")
                sa = turn.get("standalone_query")
                if sa and sa != turn.get("query"):
                    print(f"Q': {sa}")
                print(f"A:  {(turn.get('answer') or '')[:200]}…")
                print()
            continue

        if line == "/state":
            try:
                snapshot = graph.get_state(config=session.config)
                print(json.dumps(snapshot.values, indent=2, default=str)[:4000])
                print()
            except Exception as e:
                print(f"[state] {e}\n")
            continue

        # Normal query
        try:
            final_state = session.ask(line)
        except Exception as e:
            print(f"[error] {e}\n")
            continue

        print()
        print("Assistant >")
        print(final_state.get("answer", "(no answer)"))
        print()
        _print_turn_summary(final_state)
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pipeline = GraphRAGPipeline(entities_path="entities.json")
    graph = build_graph(pipeline)
    run_repl(graph)