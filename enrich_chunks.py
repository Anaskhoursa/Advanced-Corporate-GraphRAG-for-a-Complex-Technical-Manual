#!/usr/bin/env python3
"""
Enrich chunks.json by asking an LLM to extract, from each chunk's `content`:
  - warnings : [{"level": "WARNING|CAUTION|DANGER|NOTICE", "text": "..."}]
  - steps    : [{"level": 0|1, "marker": "1."|"a)", "text": "..."}]
  - content  : the leftover prose after removing warnings and steps

Tables are NOT handled here — they will be processed separately.

Input :  chunks.json
Output:  all_chunks.json

Uses Groq (openai/gpt-oss-20b) with a fallback to openai/gpt-oss-120b.
"""

import json
import os
import re
import time
from pathlib import Path

from groq import Groq
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY environment variable is not set.")

IN_FILE = "chunks.json"
OUT_FILE = "all_chunks.json"

client = Groq(
    api_key=GROQ_API_KEY
)

PRIMARY_MODEL = "openai/gpt-oss-120b"
FALLBACK_MODEL = "openai/gpt-oss-20b"

MAX_RETRIES = 3
RETRY_SLEEP = 3

# Skip front-matter chunk (id "0") — it's just the TOC and cover
SKIP_IDS = {"0"}


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a structured-information extractor for industrial equipment manuals.

You will be given the raw text of ONE section of a manual. Extract from it:

1. warnings — every WARNING / CAUTION / DANGER / NOTICE block, verbatim.
   Each item: {"level": "WARNING"|"CAUTION"|"DANGER"|"NOTICE", "text": "..."}
   - Keep the full sentence(s). Strip the "WARNING:" label from `text`.
   - Do NOT invent warnings. Only extract what is explicitly present.

2. steps — every numbered or lettered procedure in the section.
   Each item: {"level": 0|1, "marker": "1."|"a)", "text": "..."}
   - level 0 = top-level numbered item ("1.", "2.", ...)
   - level 1 = lettered sub-step ("a)", "b)", ...)
   - Strip the marker from `text`.
   - Preserve the original order.
   - Do NOT extract bullet lists ("• ..."). Those are not steps.

3. content — the remaining prose, with all warnings and steps removed.
   Preserve paragraph structure and bullets that are NOT steps/warnings.
   Do NOT include any tabular data (pipe-delimited rows, aligned columns, etc.).
   If a line looks like a table, drop it entirely — tables are handled separately.

OUTPUT
------
Return ONLY a single JSON object, no markdown fences, no commentary:

{
  "warnings": [...],
  "steps": [...],
  "content": "..."
}

If a category is empty, return an empty list/string."""


def build_user_prompt(chunk: dict) -> str:
    breadcrumb = chunk.get("header", {}).get("breadcrumb", "")
    title = chunk.get("title", "")
    content = chunk.get("content", "")
    return (
        f"SECTION: {breadcrumb}\n"
        f"TITLE: {title}\n\n"
        f"RAW TEXT:\n\"\"\"\n{content}\n\"\"\""
    )


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------
def parse_json_response(text: str) -> dict:
    """Extract the first JSON object from the model output."""
    text = text.strip()
    # strip markdown fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    # find first { ... last }
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in response")
    return json.loads(text[start : end + 1])


def call_model(model: str, system: str, user: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content


def _normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def validate_warnings(warnings: list, raw_text: str, chunk_id: str) -> list[dict]:
    """Keep only warnings whose text and level are present in the raw source."""
    normalized_raw = _normalize_whitespace(raw_text).casefold()
    valid = []
    for warning in warnings:
        if not isinstance(warning, dict):
            print(f"    [warning validation] chunk {chunk_id}: invalid warning record")
            continue

        level = str(warning.get("level") or "").strip().upper()
        text = str(warning.get("text") or "").strip()
        normalized_warning = _normalize_whitespace(text).casefold()
        level_present = bool(
            level and re.search(
                rf"(?<![A-Z]){re.escape(level)}(?![A-Z])",
                raw_text,
                re.IGNORECASE,
            )
        )
        text_present = bool(normalized_warning and normalized_warning in normalized_raw)

        if not level_present or not text_present:
            reasons = []
            if not text_present:
                reasons.append("warning text not found verbatim after whitespace normalization")
            if not level_present:
                reasons.append(f"level keyword {level or '(missing)'} not found in raw chunk")
            print(
                f"    [warning validation] chunk {chunk_id}: rejected extracted warning "
                f"({'; '.join(reasons)}): {text[:160]!r}"
            )
            continue

        valid.append(warning)
    return valid


def enrich_with_llm(chunk: dict) -> dict:
    """Return {warnings, steps, content}. Falls back to empty on failure."""
    user_prompt = build_user_prompt(chunk)

    for attempt in range(1, MAX_RETRIES + 1):
        model = PRIMARY_MODEL if attempt == 1 else FALLBACK_MODEL
        try:
            raw = call_model(model, SYSTEM_PROMPT, user_prompt)
            parsed = parse_json_response(raw)
            warnings = validate_warnings(
                parsed.get("warnings", []) or [],
                chunk.get("content", ""),
                str(chunk.get("id", "?")),
            )
            return {
                "warnings": warnings,
                "steps": parsed.get("steps", []) or [],
                "content": parsed.get("content", "") or "",
            }
        except Exception as e:
            print(f"    attempt {attempt} with {model} failed: {e}")
            time.sleep(RETRY_SLEEP)

    # all attempts failed: return empty extras, keep original content
    return {
        "warnings": [],
        "steps": [],
        "content": chunk.get("content", ""),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    chunks = json.loads(Path(IN_FILE).read_text(encoding="utf-8"))
    print(f"Loaded {len(chunks)} chunks from {IN_FILE}")

    enriched = []
    for i, chunk in enumerate(chunks, start=1):
        cid = chunk.get("id", "?")
        print(f"[{i}/{len(chunks)}] enriching chunk id={cid} title={chunk.get('title', '')[:60]}")

        if cid in SKIP_IDS:
            enriched.append({
                **chunk,
                "warnings": [],
                "steps": [],
            })
            continue

        # skip truly empty chunks — nothing to send
        if not chunk.get("content", "").strip():
            enriched.append({
                **chunk,
                "warnings": [],
                "steps": [],
            })
            continue

        result = enrich_with_llm(chunk)

        # Drop any pre-existing `tables` key from the source chunk so it
        # does not leak into all_chunks.json.
        cleaned = {k: v for k, v in chunk.items() if k != "tables"}

        enriched.append({
            **cleaned,
            "content": result["content"],
            "warnings": result["warnings"],
            "steps": result["steps"],
        })

    Path(OUT_FILE).write_text(
        json.dumps(enriched, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    total_warnings = sum(len(c["warnings"]) for c in enriched)
    total_steps = sum(len(c["steps"]) for c in enriched)

    print(f"\nWrote {len(enriched)} chunks to {OUT_FILE}")
    print(f"total warnings = {total_warnings}")
    print(f"total steps    = {total_steps}")


if __name__ == "__main__":
    main()