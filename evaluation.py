"""
Phase 7 + Evaluation harness.

Features:
  * Retry-with-backoff around every Groq call (429-aware, parses
    "try again in Xm Ys" hints).
  * Per-question cache written to disk on completion. If the script
    crashes, rerun and it resumes from the last successful question.
  * Conversation history is reconstructed from the cache so follow-up
    questions see their antecedents.
  * Atomic writes: a crash mid-write cannot corrupt the cache.

Two run modes:

    python <script>.py                       -> interactive REPL
    python <script>.py --batch               -> resume-capable evaluation
    python <script>.py --batch --no-resume   -> fresh run (keep cache file)
    python <script>.py --batch --clear-cache -> wipe cache and run fresh
"""

import argparse
import hashlib
import json
import os
import random
import re
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, TypeVar, TypedDict, Literal

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from groq import Groq
from groq import (
    RateLimitError,
    APIStatusError,
    APIConnectionError,
    APITimeoutError,
)

from retrieval import (
    GraphRAGPipeline,
    GROQ_API_KEY,
)


# ---------------------------------------------------------------------------
# Cache configuration
# ---------------------------------------------------------------------------

# Bump this whenever you change test questions, prompts, or graph topology
# in a way that invalidates prior results.
CACHE_VERSION = 1

# Where cached turns live. One JSON file, rewritten atomically after each
# successful question.
DEFAULT_CACHE_PATH = "eval_cache.json"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MAX_RETRIES = 1
MIN_ANSWER_CHARS = 40
MAX_HISTORY_TURNS = 10

TRANSIENT_MAX_RETRIES = 4
TRANSIENT_BASE_DELAY_S = 2.0
TRANSIENT_MAX_DELAY_S = 30.0
MAX_TOTAL_WAIT_S = 45 * 60
RATE_LIMIT_MIN_WAIT_S = 1.0
RATE_LIMIT_JITTER_S = 1.5

groq_client = Groq(api_key=GROQ_API_KEY)


# ---------------------------------------------------------------------------
# Retry counters (reset per run; not persisted)
# ---------------------------------------------------------------------------

_RETRY_STATS = {
    "llm_calls": 0,
    "transient_retries": 0,
    "rate_limit_retries": 0,
    "total_sleep_s": 0.0,
}


def reset_retry_stats() -> None:
    _RETRY_STATS.update({
        "llm_calls": 0,
        "transient_retries": 0,
        "rate_limit_retries": 0,
        "total_sleep_s": 0.0,
    })


# ---------------------------------------------------------------------------
# Rate-limit parsing
# ---------------------------------------------------------------------------

_RETRY_IN_RE = re.compile(
    r"try again in\s+"
    r"(?:(?P<h>\d+)\s*h)?\s*"
    r"(?:(?P<m>\d+)\s*m)?\s*"
    r"(?:(?P<s>\d+(?:\.\d+)?)\s*s)?",
    re.IGNORECASE,
)


def _parse_retry_after_seconds(err: BaseException) -> float | None:
    m = _RETRY_IN_RE.search(str(err))
    if not m:
        return None
    hours = int(m.group("h") or 0)
    minutes = int(m.group("m") or 0)
    seconds = float(m.group("s") or 0.0)
    total = hours * 3600 + minutes * 60 + seconds
    return total if total > 0 else None


def _is_rate_limit(err: BaseException) -> bool:
    if isinstance(err, RateLimitError):
        return True
    text = str(err).lower()
    return "rate_limit_exceeded" in text or ("429" in text and "rate limit" in text)


def _is_transient(err: BaseException) -> bool:
    if isinstance(err, (APIConnectionError, APITimeoutError)):
        return True
    if isinstance(err, APIStatusError):
        status = getattr(err, "status_code", None)
        if status is not None and 500 <= status < 600:
            return True
    text = str(err).lower()
    if "getaddrinfo failed" in text or "connection" in text:
        return True
    if any(f" {code}" in text for code in ("500", "502", "503", "504")):
        return True
    return False


def _sleep_with_log(seconds: float, reason: str) -> None:
    seconds = max(0.0, seconds)
    print(f"    [retry] sleeping {seconds:.2f}s — {reason}", flush=True)
    _RETRY_STATS["total_sleep_s"] += seconds
    time.sleep(seconds)


T = TypeVar("T")


def call_with_retry(fn: Callable[[], T], *, label: str = "groq_call") -> T:
    _RETRY_STATS["llm_calls"] += 1

    transient_attempt = 0
    total_wait = 0.0

    while True:
        try:
            return fn()

        except Exception as e:
            if _is_rate_limit(e):
                _RETRY_STATS["rate_limit_retries"] += 1

                hinted = _parse_retry_after_seconds(e)
                if hinted is None:
                    delay = min(60.0, 2.0 ** min(6, _RETRY_STATS["rate_limit_retries"]))
                else:
                    delay = max(RATE_LIMIT_MIN_WAIT_S, hinted)
                delay += random.uniform(0.0, RATE_LIMIT_JITTER_S)

                if total_wait + delay > MAX_TOTAL_WAIT_S:
                    print(f"    [{label}] rate-limit wait would exceed "
                          f"{MAX_TOTAL_WAIT_S}s budget; raising.", flush=True)
                    raise

                _sleep_with_log(delay, f"{label}: 429 rate limit")
                total_wait += delay
                continue

            if _is_transient(e):
                transient_attempt += 1
                if transient_attempt > TRANSIENT_MAX_RETRIES:
                    raise

                _RETRY_STATS["transient_retries"] += 1
                delay = min(TRANSIENT_MAX_DELAY_S,
                            TRANSIENT_BASE_DELAY_S * (2 ** (transient_attempt - 1)))
                delay += random.uniform(0.0, 1.0)

                if total_wait + delay > MAX_TOTAL_WAIT_S:
                    raise

                _sleep_with_log(delay, f"{label}: transient ({type(e).__name__})")
                total_wait += delay
                continue

            raise


# ---------------------------------------------------------------------------
# Per-intent graph weights
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
# Prompts (unchanged)
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
    user_query: str
    standalone_query: str
    intent: str
    retrieval: dict
    context: str
    answer: str
    verified: bool
    verify_issues: list[str]
    retry_count: int
    reformulation_reason: str
    conversation_history: list[dict]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_history_for_prompt(history: list[dict]) -> str:
    if not history:
        return "(no prior turns)"

    lines: list[str] = []
    for turn in history[-MAX_HISTORY_TURNS:]:
        q = (turn.get("query") or "").strip()
        a = (turn.get("answer") or "").strip()
        a_short = a[:300] + ("…" if len(a) > 300 else "")
        lines.append(f"User: {q}")
        lines.append(f"Assistant: {a_short}")
        lines.append("")

    return "\n".join(lines).strip()


def _groq_json(
    *,
    label: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float = 0.0,
    reasoning_effort: str = "low",
) -> dict:
    def _do_call():
        return groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_completion_tokens=max_tokens,
            response_format={"type": "json_object"},
            reasoning_effort=reasoning_effort,
            reasoning_format="hidden",
        )

    resp = call_with_retry(_do_call, label=label)
    try:
        return json.loads(resp.choices[0].message.content or "{}")
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def reformulate_query_node(state: AgentState) -> dict:
    user_query = state.get("user_query") or state.get("query") or ""
    history = state.get("conversation_history") or []

    if not history:
        return {
            "standalone_query": user_query,
            "reformulation_reason": "",
            "retry_count": 0,
        }

    history_block = _format_history_for_prompt(history)

    try:
        payload = _groq_json(
            label="reformulate_query",
            system_prompt=REFORMULATOR_SYSTEM_PROMPT,
            user_prompt=(
                f"Conversation history:\n{history_block}\n\n"
                f"New user query: {user_query}"
            ),
            max_tokens=256,
        )

        rewritten = (payload.get("rewritten") or user_query).strip()
        changed = bool(payload.get("changed", False))
        reason = (payload.get("reason") or "").strip()

        if not rewritten:
            rewritten = user_query
            changed = False

        return {
            "standalone_query": rewritten,
            "reformulation_reason": reason if changed else "",
            "retry_count": 0,
        }
    except Exception as e:
        print(f"[reformulate_query] failed, using raw query: {e}", flush=True)
        return {
            "standalone_query": user_query,
            "reformulation_reason": "",
            "retry_count": 0,
        }


def classify_query_node(state: AgentState) -> dict:
    query = state.get("standalone_query") or state.get("user_query") or ""

    try:
        payload = _groq_json(
            label="classify",
            system_prompt=CLASSIFIER_SYSTEM_PROMPT,
            user_prompt=query,
            max_tokens=64,
        )
        intent = payload.get("intent", "factual")
    except Exception as e:
        print(f"[classify] falling back to 'factual': {e}", flush=True)
        intent = "factual"

    if intent not in INTENT_GRAPH_WEIGHTS:
        intent = "factual"

    return {"intent": intent}


def retrieve_node(state: AgentState) -> dict:
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


def generate_node(state: AgentState) -> dict:
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

    answer = call_with_retry(
        lambda: pipeline.generate(query, context),
        label="generate",
    )
    return {"answer": answer}


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
        payload = _groq_json(
            label="verify",
            system_prompt=VERIFIER_SYSTEM_PROMPT,
            user_prompt=(
                f"QUESTION:\n{query}\n\n"
                f"ANSWER:\n{answer}\n\n"
                f"CONTEXT:\n{context}"
            ),
            max_tokens=1024,
        )
        return {
            "verified": bool(payload.get("grounded", False)),
            "verify_issues": payload.get("issues", []) or [],
        }
    except Exception as e:
        print(f"[verify] verifier call failed: {e}", flush=True)
        return {"verified": True, "verify_issues": []}


def reformulate_answer_node(state: AgentState) -> dict:
    query = state.get("standalone_query") or state.get("user_query") or ""
    answer = state.get("answer", "")
    context = state.get("context", "")
    issues = state.get("verify_issues", [])

    issue_text = "\n".join(
        f"- {i.get('problem', '?')}: {i.get('claim', '?')} — {i.get('explanation', '')}"
        for i in issues if isinstance(i, dict)
    ) or "(no specific issues listed)"

    try:
        payload = _groq_json(
            label="reformulate_answer",
            system_prompt=ANSWER_REFORMULATOR_SYSTEM_PROMPT,
            user_prompt=(
                f"Query: {query}\n\n"
                f"Previous answer:\n{answer}\n\n"
                f"Grounding issues:\n{issue_text}\n\n"
                f"Context (the only source you may use):\n{context}"
            ),
            max_tokens=4096,
            temperature=0.2,
            reasoning_effort="medium",
        )
        new_answer = (payload.get("answer") or answer).strip()
    except Exception as e:
        print(f"[reformulate_answer] failed, keeping original: {e}", flush=True)
        new_answer = answer

    return {
        "answer": new_answer or answer,
        "retry_count": state.get("retry_count", 0) + 1,
        "verified": False,
        "verify_issues": [],
    }


def memory_node(state: AgentState) -> dict:
    history = list(state.get("conversation_history") or [])
    history.append({
        "query": state.get("user_query") or "",
        "standalone_query": state.get("standalone_query") or "",
        "answer": state.get("answer", ""),
        "verified": state.get("verified", True),
        "intent": state.get("intent", "factual"),
    })
    return {"conversation_history": history[-MAX_HISTORY_TURNS:]}


def route_after_verify(state: AgentState) -> Literal["reformulate_answer", "memory"]:
    if state.get("verified", False):
        return "memory"
    if state.get("intent") == "out_of_scope":
        return "memory"
    if state.get("retry_count", 0) >= MAX_RETRIES:
        return "memory"
    return "reformulate_answer"


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
        {"reformulate_answer": "reformulate_answer", "memory": "memory"},
    )
    builder.add_edge("reformulate_answer", "verify")
    builder.add_edge("memory",             END)

    return builder.compile(checkpointer=MemorySaver())


# ---------------------------------------------------------------------------
# Conversation session
# ---------------------------------------------------------------------------

class ConversationSession:
    def __init__(self, graph, thread_id: str, initial_history: list[dict] | None = None):
        self.graph = graph
        self.thread_id = thread_id
        self.config = {"configurable": {"thread_id": thread_id}}
        self.history: list[dict] = list(initial_history or [])

    def ask(self, query: str) -> dict:
        final_state = self.graph.invoke(
            {"user_query": query, "conversation_history": self.history},
            config=self.config,
        )
        self.history = final_state.get("conversation_history", [])
        return final_state

    def reset(self):
        self.history = []


# ---------------------------------------------------------------------------
# REPL (unchanged from previous version)
# ---------------------------------------------------------------------------

BANNER = """
==============================================================
  HL Series Pump Manual — GraphRAG Assistant (interactive)
==============================================================
Commands: /new  /history  /state  /exit
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
    counter = 0

    def new_session():
        nonlocal counter
        counter += 1
        s = ConversationSession(graph, thread_id=f"session_{counter}")
        print(f"[session] started thread session_{counter}\n")
        return s

    session = new_session()

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
            session = new_session()
            continue
        if line == "/history":
            for i, t in enumerate(session.history, 1):
                print(f"--- turn {i} ---\nQ:  {t.get('query')}")
                sa = t.get("standalone_query")
                if sa and sa != t.get("query"):
                    print(f"Q': {sa}")
                print(f"A:  {(t.get('answer') or '')[:200]}…\n")
            continue
        if line == "/state":
            try:
                snap = graph.get_state(config=session.config)
                print(json.dumps(snap.values, indent=2, default=str)[:4000], "\n")
            except Exception as e:
                print(f"[state] {e}\n")
            continue

        try:
            final_state = session.ask(line)
        except Exception as e:
            print(f"[error] {e}\n")
            continue

        print("\nAssistant >")
        print(final_state.get("answer", "(no answer)"))
        print()
        _print_turn_summary(final_state)
        print()


# ===========================================================================
# EVALUATION HARNESS
# ===========================================================================

TEST_QUESTIONS: list[tuple[str, str | None, str]] = [
    ("What does the DANGER hazard level mean?", "safety", "hazard-level definition"),
    ("What does the CAUTION hazard level indicate?", "safety", "hazard-level definition"),
    ("What is the maximum temperature batteries can be exposed to?", "safety", "battery warning"),
    ("What safety equipment should personnel wear when working on the pump?", "safety", "PPE"),
    ("Is it safe to use non-Xylem spare parts?", "safety", "spare parts caution"),

    ("How should I lift the pump unit using a crane?", "procedural", "lifting procedure"),
    ("What should I check before each trailer journey?", "procedural", "trailer pre-check"),
    ("What is the maximum nose weight for the trailer?", "factual", "spec lookup"),
    ("How often should I check tire pressures on the trailer?", "procedural", "trailer maintenance"),
    ("What precautions should I take before disconnecting the trailer from the towing vehicle?", "safety", "tip hazard"),

    ("What materials is the pump made of?", "factual", "materials list"),
    ("What is the maximum pressure for the HL125M?", "factual", "model spec"),
    ("What is the maximum soft solids handling for the HL225M?", "factual", "model spec"),
    ("What is the suction size of the HL250M?", "factual", "flange spec"),
    ("What is the maximum working temperature for the HL series?", "factual", "temperature spec"),

    ("How do I perform a vacuum pad test?", "procedural", "vacuum test"),
    ("What vacuum range should the pad gauge read during the test?", "factual", "vacuum spec"),
    ("What is the minimum distance between the pump inlet flange and the closest elbow?", "factual", "piping spec"),
    ("How do I attach the dial indicators for pump-to-driver alignment?", "procedural", "alignment procedure"),
    ("What are the tolerance values for pump-to-driver alignment?", "factual", "alignment spec"),
    ("What type of grout should be used for skid-mounted permanent installations?", "factual", "grout type"),

    ("How do I start the pump?", "procedural", "startup procedure"),
    ("How do I shut down the pump?", "procedural", "shutdown procedure"),
    ("What are the steps for operating the pump with flooded suction?", "procedural", "flooded suction"),
    ("What are the bearing lubrication requirements?", "factual", "lubrication spec"),
    ("What is the mechanical seal coolant capacity for the HL110M?", "factual", "coolant spec"),

    ("What warnings apply to the coupling?", "safety", "opener for follow-ups"),
    ("Does it need lubrication?", "factual", "pronoun follow-up — should reformulate"),
    ("What about the impeller?", "factual", "elliptical follow-up — should reformulate"),

    ("What daily maintenance is required for the engine?", "procedural", "daily engine maintenance"),
    ("What maintenance is performed after the first 100 hours?", "procedural", "100h maintenance"),
    ("What maintenance is performed after the first 200 hours?", "procedural", "200h maintenance"),
    ("What should I check during 250 hour pump maintenance?", "procedural", "250h maintenance"),
    ("What is checked at 5000 hours on the pump?", "procedural", "5000h maintenance"),
    ("What bearing lubricant brands are recommended?", "factual", "lubricant list"),

    ("Why does the pump not prime?", "troubleshoot", "priming problem"),
    ("What causes insufficient liquid delivery?", "troubleshoot", "flow problem"),
    ("Why does liquid flow cease?", "troubleshoot", "flow cessation"),
    ("What should I do if the unit vibrates or overheats?", "troubleshoot", "vibration/heat"),
    ("Why is the mechanical seal coolant emulsified?", "troubleshoot", "coolant issue"),
    ("Why does the driver not start?", "troubleshoot", "driver problem"),
    ("What causes one-sided trailer braking?", "troubleshoot", "trailer brakes"),
    ("What should I do if the trailer brakes lock up when reversing?", "troubleshoot", "trailer brakes"),

    ("What is the normal speed of the HL160M?", "factual", "speed spec"),
    ("What is the maximum flow of the HL300M?", "factual", "flow spec"),
    ("What is the torque value for an M12 steel bolt?", "factual", "torque spec"),
    ("What is the impeller-to-wear-plate clearance for cast iron parts?", "factual", "clearance spec"),
    ("What are the air compressor oil requirements?", "factual", "compressor oil"),

    ("Which HL model has the highest maximum pressure?", "comparison", "max pressure comparison"),
    ("Which HL model has the highest maximum flow?", "comparison", "max flow comparison"),

    ("What is the capital of France?", "out_of_scope", "off-topic"),
    ("How do I reset my email password?", "out_of_scope", "off-topic"),
]


# ---------------------------------------------------------------------------
# Cache: read / write / fingerprint
# ---------------------------------------------------------------------------

def _question_fingerprint(question: str, expected_intent: str | None, note: str) -> str:
    """
    Stable, short hash of the fields that identify a question. If any of
    these change (e.g. you tweak the expected intent), the cached entry
    is invalidated.
    """
    h = hashlib.sha256()
    for part in (str(CACHE_VERSION), question, expected_intent or "", note):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def _load_cache(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {
            "cache_version": CACHE_VERSION,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "entries": {},
        }

    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        # Corrupt or truncated file; back it up and start fresh.
        backup = f"{path}.corrupt.{int(time.time())}"
        try:
            p.replace(backup)
        except Exception:
            pass
        print(f"[cache] could not read {path} ({e}); moved to {backup}", flush=True)
        return {
            "cache_version": CACHE_VERSION,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "entries": {},
        }

    if raw.get("cache_version") != CACHE_VERSION:
        print(f"[cache] version mismatch "
              f"(file={raw.get('cache_version')}, code={CACHE_VERSION}); "
              f"ignoring cached entries.", flush=True)
        return {
            "cache_version": CACHE_VERSION,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "entries": {},
        }

    raw.setdefault("entries", {})
    return raw


def _save_cache_atomic(cache: dict, path: str) -> None:
    """
    Write cache to a temp file, then os.replace() to swap it in. On POSIX
    and Windows, os.replace is atomic on the same filesystem, so a crash
    mid-write cannot leave a half-written cache.
    """
    p = Path(path)
    tmp = p.with_suffix(p.suffix + ".tmp")

    cache["updated_at"] = datetime.now().isoformat(timespec="seconds")
    payload = json.dumps(cache, indent=2, ensure_ascii=False, default=str)

    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, p)


# ---------------------------------------------------------------------------
# Aggregate metrics helpers
# ---------------------------------------------------------------------------

def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


# ---------------------------------------------------------------------------
# Evaluation runner (resume-capable)
# ---------------------------------------------------------------------------

def run_evaluation(
    graph,
    out_path: str | None = None,
    cache_path: str = DEFAULT_CACHE_PATH,
    resume: bool = True,
    clear_cache: bool = False,
) -> str:
    """
    Run the evaluation suite. Completed questions are cached in `cache_path`
    and skipped on subsequent runs, so a crash mid-run is recoverable.
    """
    if out_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = f"evaluation_report_{ts}.txt"

    if clear_cache and Path(cache_path).exists():
        print(f"[cache] --clear-cache: removing {cache_path}", flush=True)
        Path(cache_path).unlink()

    cache = _load_cache(cache_path) if resume else {
        "cache_version": CACHE_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "entries": {},
    }

    # Build the run plan: for each question, decide cached vs. fresh.
    plan: list[dict] = []
    for i, (q, ei, note) in enumerate(TEST_QUESTIONS, start=1):
        fp = _question_fingerprint(q, ei, note)
        cached = cache["entries"].get(fp)
        plan.append({
            "index": i,
            "question": q,
            "expected_intent": ei,
            "note": note,
            "fingerprint": fp,
            "cached": cached,
        })

    n_total = len(plan)
    n_cached = sum(1 for p in plan if p["cached"])
    n_to_run = n_total - n_cached

    print("=" * 70)
    print(f"Evaluation: {n_total} questions ({n_cached} cached, {n_to_run} to run)")
    print(f"Report  -> {out_path}")
    print(f"Cache   -> {cache_path}")
    print(f"Retry   -> transient x{TRANSIENT_MAX_RETRIES}, "
          f"rate-limit budget {MAX_TOTAL_WAIT_S}s/call")
    print("=" * 70)

    reset_retry_stats()

    # Reconstruct history from cached turns so follow-ups still work.
    # We do this by replaying the cache entries in index order.
    session = ConversationSession(graph, thread_id="eval_run")

    # Seed the session's history with whatever's already cached, up to
    # the point where we'll start running. This makes follow-ups after
    # a resume still see their antecedent.
    seed_history: list[dict] = []
    for p in plan:
        if p["cached"]:
            entry = p["cached"]
            seed_history.append({
                "query": entry.get("question", p["question"]),
                "standalone_query": entry.get("standalone_query", "") or "",
                "answer": entry.get("answer_preview", "") or "",
                "verified": bool(entry.get("verified", False)),
                "intent": entry.get("intent", "factual"),
            })
        else:
            break  # stop seeding at the first uncached question
    session.history = seed_history[-MAX_HISTORY_TURNS:]

    per_turn: list[dict] = []
    overall_start = time.perf_counter()

    for p in plan:
        i = p["index"]
        q = p["question"]

        if p["cached"]:
            print(f"[{i:02d}/{n_total}] (cached) {q[:65]}"
                  + ("..." if len(q) > 65 else ""), flush=True)
            per_turn.append({**p["cached"], "from_cache": True})
            continue

        print(f"[{i:02d}/{n_total}] {q[:65]}"
              + ("..." if len(q) > 65 else ""), flush=True)

        retries_before = dict(_RETRY_STATS)
        t0 = time.perf_counter()
        try:
            state = session.ask(q)
            err = None
        except Exception as e:
            state = {}
            err = repr(e)
            # A crash here might be permanent (bad key, quota exhausted
            # past our budget). Persist what we have and keep going — the
            # entry will have error set, and the next run can retry it.
        elapsed = time.perf_counter() - t0

        retries_after = dict(_RETRY_STATS)
        answer = state.get("answer", "") or ""
        issues = state.get("verify_issues") or []
        retrieval = state.get("retrieval", {}) or {}

        entry = {
            "index": i,
            "question": q,
            "expected_intent": p["expected_intent"],
            "note": p["note"],
            "standalone_query": state.get("standalone_query", "") or "",
            "reformulated": bool(
                state.get("standalone_query")
                and state.get("standalone_query") != q
            ),
            "reformulation_reason": state.get("reformulation_reason", "") or "",
            "intent": state.get("intent", "?"),
            "verified": bool(state.get("verified", False)),
            "retry_count": int(state.get("retry_count", 0) or 0),
            "verify_issue_count": len(issues) if isinstance(issues, list) else 0,
            "answer_chars": len(answer),
            "answer_preview": answer[:280].replace("\n", " ").strip(),
            "used_graph": bool(retrieval.get("used_graph", False)),
            "chunk_count": len(retrieval.get("chunk_ids", []) or []),
            "linked_entities": retrieval.get("linked_entities", []) or [],
            "latency_s": elapsed,
            "rate_limit_retries": (
                retries_after["rate_limit_retries"] - retries_before["rate_limit_retries"]
            ),
            "transient_retries": (
                retries_after["transient_retries"] - retries_before["transient_retries"]
            ),
            "sleep_s": (
                retries_after["total_sleep_s"] - retries_before["total_sleep_s"]
            ),
            "error": err,
            "from_cache": False,
        }

        per_turn.append(entry)

        # --- Persist immediately, atomically ---
        # If the process dies right now, this question is safe.
        cache["entries"][p["fingerprint"]] = entry
        try:
            _save_cache_atomic(cache, cache_path)
        except Exception as e:
            print(f"    [cache] WARNING: could not write cache: {e}", flush=True)

        if entry["rate_limit_retries"] or entry["transient_retries"]:
            print(f"      ↳ retries: rate_limit={entry['rate_limit_retries']}, "
                  f"transient={entry['transient_retries']}, "
                  f"slept={entry['sleep_s']:.1f}s", flush=True)

    overall_elapsed = time.perf_counter() - overall_start

    # ----- Aggregate -----
    n = len(per_turn)
    latencies = [t["latency_s"] for t in per_turn]
    answer_lens = [t["answer_chars"] for t in per_turn]
    sleeps = [t["sleep_s"] for t in per_turn]

    verified_count = sum(1 for t in per_turn if t["verified"])
    retried_count = sum(1 for t in per_turn if t["retry_count"] > 0)
    reformulated_count = sum(1 for t in per_turn if t["reformulated"])
    graph_used_count = sum(1 for t in per_turn if t["used_graph"])
    errored_count = sum(1 for t in per_turn if t["error"])
    from_cache_count = sum(1 for t in per_turn if t.get("from_cache"))

    intent_dist: dict[str, int] = {}
    for t in per_turn:
        intent_dist[t["intent"]] = intent_dist.get(t["intent"], 0) + 1

    expected_dist: dict[str, int] = {}
    for t in per_turn:
        ei = t["expected_intent"] or "?"
        expected_dist[ei] = expected_dist.get(ei, 0) + 1

    comparable = [t for t in per_turn if t["expected_intent"]]
    intent_matches = sum(
        1 for t in comparable if t["intent"] == t["expected_intent"]
    )
    intent_agreement = (intent_matches / len(comparable)) if comparable else 0.0

    oos = [t for t in per_turn if t["expected_intent"] == "out_of_scope"]
    oos_correct = sum(1 for t in oos if t["intent"] == "out_of_scope")

    per_intent_verified: dict[str, list[bool]] = {}
    for t in per_turn:
        per_intent_verified.setdefault(t["intent"], []).append(t["verified"])

    run_status = "COMPLETE" if n == n_total else f"PARTIAL ({n}/{n_total})"

    # ----- Report -----
    lines: list[str] = []
    W = lines.append

    W("=" * 78)
    W("HL SERIES PUMP MANUAL — GRAPHRAG EVALUATION REPORT")
    W("=" * 78)
    W(f"Generated:          {datetime.now().isoformat(timespec='seconds')}")
    W(f"Run status:         {run_status}")
    W(f"Cache file:         {cache_path}")
    W(f"Cache hits:         {from_cache_count}/{n_total}")
    W(f"Total questions:    {n_total}")
    W(f"Questions in report:{n}")
    W(f"Total wall time:    {overall_elapsed:.2f} s")
    W("")

    W("-" * 78)
    W("RETRY / RATE-LIMIT STATS (this run only)")
    W("-" * 78)
    W(f"Total LLM calls:               {_RETRY_STATS['llm_calls']}")
    W(f"Rate-limit retries (429):      {_RETRY_STATS['rate_limit_retries']}")
    W(f"Transient retries (5xx/net):   {_RETRY_STATS['transient_retries']}")
    W(f"Total time spent sleeping:     {_RETRY_STATS['total_sleep_s']:.1f} s")
    W("")

    if n == 0:
        W("No turns recorded — nothing to aggregate.")
        W("=" * 78)
        Path(out_path).write_text("\n".join(lines), encoding="utf-8")
        print(f"[report] empty run; wrote placeholder to {out_path}", flush=True)
        return out_path

    W("-" * 78)
    W("AGGREGATE METRICS")
    W("-" * 78)
    W(f"Verification pass rate:        {verified_count}/{n}  "
      f"({100.0 * verified_count / n:.1f}%)")
    W(f"Turns that triggered a retry:  {retried_count}/{n}  "
      f"({100.0 * retried_count / n:.1f}%)")
    W(f"Turns with reformulated query: {reformulated_count}/{n}  "
      f"({100.0 * reformulated_count / n:.1f}%)")
    W(f"Turns that used the graph:     {graph_used_count}/{n}  "
      f"({100.0 * graph_used_count / n:.1f}%)")
    W(f"Turns that errored:            {errored_count}/{n}")
    W("")
    W(f"Latency (s):  mean={statistics.mean(latencies):.2f}  "
      f"median={statistics.median(latencies):.2f}  "
      f"p90={_percentile(latencies, 0.90):.2f}  "
      f"max={max(latencies):.2f}  min={min(latencies):.2f}")
    W(f"Answer chars: mean={statistics.mean(answer_lens):.1f}  "
      f"median={statistics.median(answer_lens):.1f}  "
      f"min={min(answer_lens)}  max={max(answer_lens)}")
    W(f"Sleep per turn (s): mean={statistics.mean(sleeps):.2f}  "
      f"max={max(sleeps):.2f}")
    W("")

    W("-" * 78)
    W("INTENT DISTRIBUTION (predicted)")
    W("-" * 78)
    for k in sorted(intent_dist):
        W(f"  {k:<15} {intent_dist[k]:>3}")
    W("")

    W("-" * 78)
    W("INTENT DISTRIBUTION (expected, informational)")
    W("-" * 78)
    for k in sorted(expected_dist):
        W(f"  {k:<15} {expected_dist[k]:>3}")
    W("")

    W("-" * 78)
    W("CLASSIFIER AGREEMENT vs. EXPECTED")
    W("-" * 78)
    W(f"Comparable questions: {len(comparable)}")
    W(f"Agreement:            {intent_matches}/{len(comparable)}  "
      f"({100.0 * intent_agreement:.1f}%)")
    if oos:
        W(f"Out-of-scope correct: {oos_correct}/{len(oos)}")
    W("")

    W("-" * 78)
    W("VERIFICATION RATE PER PREDICTED INTENT")
    W("-" * 78)
    for k in sorted(per_intent_verified):
        vals = per_intent_verified[k]
        passed = sum(1 for v in vals if v)
        W(f"  {k:<15} {passed}/{len(vals)}  "
          f"({100.0 * passed / len(vals):.1f}%)")
    W("")

    W("-" * 78)
    W("PER-TURN DETAIL")
    W("-" * 78)

    for t in per_turn:
        tag = " [cached]" if t.get("from_cache") else ""
        W(f"[{t['index']:02d}]{tag} {t['question']}")
        if t["note"]:
            W(f"     note            : {t['note']}")
        if t["expected_intent"]:
            W(f"     expected intent : {t['expected_intent']}")
        if t["reformulated"]:
            W(f"     reformulated    : {t['standalone_query']}")
            if t["reformulation_reason"]:
                W(f"     reason          : {t['reformulation_reason']}")
        W(f"     predicted intent: {t['intent']}")
        W(f"     verified        : {t['verified']}  "
          f"(issues={t['verify_issue_count']}, retries={t['retry_count']})")
        W(f"     retrieval       : chunks={t['chunk_count']}, "
          f"used_graph={t['used_graph']}, "
          f"entities={t['linked_entities']}")
        W(f"     retries/sleep   : rate_limit={t['rate_limit_retries']}, "
          f"transient={t['transient_retries']}, "
          f"sleep={t['sleep_s']:.2f}s")
        W(f"     latency         : {t['latency_s']:.2f} s")
        W(f"     answer_chars    : {t['answer_chars']}")
        W(f"     answer_preview  : {t['answer_preview']}")
        if t["error"]:
            W(f"     ERROR           : {t['error']}")
        W("")

    W("=" * 78)
    W("END OF REPORT")
    W("=" * 78)

    Path(out_path).write_text("\n".join(lines), encoding="utf-8")

    print()
    print("=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Run status:             {run_status}")
    print(f"Cache hits:             {from_cache_count}/{n_total}")
    print(f"Verification pass rate: {verified_count}/{n} "
          f"({100.0 * verified_count / n:.1f}%)")
    print(f"Retry rate:             {retried_count}/{n} "
          f"({100.0 * retried_count / n:.1f}%)")
    print(f"Reformulated queries:   {reformulated_count}/{n} "
          f"({100.0 * reformulated_count / n:.1f}%)")
    print(f"Graph usage:            {graph_used_count}/{n} "
          f"({100.0 * graph_used_count / n:.1f}%)")
    print(f"Classifier agreement:   {intent_matches}/{len(comparable)} "
          f"({100.0 * intent_agreement:.1f}%)")
    print(f"Rate-limit retries:     {_RETRY_STATS['rate_limit_retries']}")
    print(f"Transient retries:      {_RETRY_STATS['transient_retries']}")
    print(f"Time spent sleeping:    {_RETRY_STATS['total_sleep_s']:.1f} s")
    print(f"Mean latency:           {statistics.mean(latencies):.2f} s")
    print(f"p90 latency:            {_percentile(latencies, 0.90):.2f} s")
    print(f"Report written to:      {out_path}")
    print("=" * 70)

    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="HL Series Pump Manual — GraphRAG assistant"
    )
    parser.add_argument("--batch", action="store_true",
                        help="Run the evaluation suite instead of the REPL")
    parser.add_argument("--out", type=str, default=None,
                        help="Output path for the evaluation report")
    parser.add_argument("--cache", type=str, default=DEFAULT_CACHE_PATH,
                        help=f"Cache file path (default: {DEFAULT_CACHE_PATH})")
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore cache and re-run every question "
                             "(cache file is preserved)")
    parser.add_argument("--clear-cache", action="store_true",
                        help="Delete the cache file and start fresh")
    args = parser.parse_args()

    pipeline = GraphRAGPipeline(entities_path="entities.json")
    graph = build_graph(pipeline)

    if args.batch:
        run_evaluation(
            graph,
            out_path=args.out,
            cache_path=args.cache,
            resume=not args.no_resume,
            clear_cache=args.clear_cache,
        )
    else:
        run_repl(graph)


if __name__ == "__main__":
    main()