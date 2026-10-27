#!/usr/bin/env python3
"""
Multi-pass Knowledge Graph Extraction Pipeline for HL Series Pump Manual.

Pipelines:
  1. TEXT    -> LLM extraction from chunk prose (NO steps, NO tables, NO warnings)
  2. STEP    -> Deterministic extraction from chunk["steps"]
  3. WARNING -> Deterministic extraction from chunk["warnings"]
  4. TABLE   -> LLM extraction from each table in chunk["tables"]
  5. RESOLVE -> Entity deduplication / merging across chunks
  6. LINK    -> Cross-linking steps/warnings <-> entities they mention
"""

import json
import os
import re
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import defaultdict

from groq import Groq

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

from dotenv import load_dotenv

load_dotenv()

CHUNKS_FILE = "all_chunks.json"
OUTPUT_ENTITIES = "entities.json"
OUTPUT_RELATIONS = "relations.json"
OUTPUT_WARNINGS = "warnings.json"
OUTPUT_STEPS = "steps.json"

CACHE_DIR = Path(os.getenv("LLM_CACHE_DIR", ".llm_cache"))
CACHE_ENABLED = os.getenv("LLM_CACHE", "1") != "0"

TEXT_PROMPT_VERSION = "text_v4"
TABLE_PROMPT_VERSION = "table_v4"

PRIMARY_MODEL = os.getenv("LLM_PRIMARY_MODEL", "openai/gpt-oss-120b")
FALLBACK_MODEL = os.getenv("LLM_FALLBACK_MODEL", "openai/gpt-oss-20b")

MIN_INTERVAL_SECONDS = float(os.getenv("LLM_MIN_INTERVAL", "3.0"))
MAX_ATTEMPTS_PER_MODEL = int(os.getenv("LLM_MAX_ATTEMPTS", "5"))
BASE_BACKOFF_SECONDS = float(os.getenv("LLM_BASE_BACKOFF", "15.0"))
MAX_BACKOFF_SECONDS = float(os.getenv("LLM_MAX_BACKOFF", "180.0"))

LIMIT_CHUNKS = int(os.getenv("LIMIT_CHUNKS", "0"))

# Allowed entity and relation vocabularies
ENTITY_TYPES = [
    "COMPONENT", "PART", "TOOL", "MATERIAL", "SPEC",
    "PROCEDURE", "CONCEPT", "DOCUMENT", "MODEL",
    "WARNING", "STEP", "TABLE",
]

RELATION_TYPES = [
    "PART_OF", "HAS_PART", "REQUIRES_TOOL", "USES_MATERIAL",
    "HAS_SPEC", "APPLIES_TO", "REFERENCES", "LOCATED_ON",
    "CONNECTED_TO", "SAME_AS", "CAUSES_FAULT", "RESOLVED_BY",
    "PRECEDES_STEP", "HAS_ROW", "HAS_STEP", "HAS_WARNING",
    "MENTIONS", "WARNS_ABOUT",
]

# Stop-words: skip generic single nouns that add noise to the graph
GENERIC_ENTITY_BLOCKLIST = {
    "heat", "dirt", "water", "oil", "information", "purpose",
    "vibrations", "vibration", "location", "product", "manual",
    "system", "process", "work", "area", "point", "value", "level",
    "item", "object", "matter", "amount", "time", "place", "thing",
}

# Relation types we never want a generic match on (circular / trivial)
TRIVIAL_RELATION_PATTERNS = [
    # (src_name_substr, relation, tgt_name_substr)
]



client = Groq(api_key="")

# --------------------------------------------------------------------------
# Rate Limiter & Caching
# --------------------------------------------------------------------------

_last_call_ts = 0.0

def _rate_limit():
    global _last_call_ts
    now = time.time()
    wait = MIN_INTERVAL_SECONDS - (now - _last_call_ts)
    if wait > 0:
        time.sleep(wait)
    _last_call_ts = time.time()

def _cache_key(version: str, chunk_id: str, extra: str) -> str:
    h = hashlib.sha256()
    h.update(version.encode("utf-8"))
    h.update(chunk_id.encode("utf-8"))
    h.update(extra.encode("utf-8"))
    return h.hexdigest()

def _cache_get(key: str) -> Optional[Dict[str, Any]]:
    if not CACHE_ENABLED:
        return None
    p = CACHE_DIR / f"{key}.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None

def _cache_put(key: str, value: Dict[str, Any]) -> None:
    if not CACHE_ENABLED:
        return
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / f"{key}.json").write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

# --------------------------------------------------------------------------
# Groq API Call Plumbing
# --------------------------------------------------------------------------

def _call_llm(model: str, system: str, user: str) -> str:
    _rate_limit()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
        max_completion_tokens=4096,
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content or ""

def _call_with_retry(system: str, user: str) -> str:
    models = [PRIMARY_MODEL, FALLBACK_MODEL]
    for model in models:
        backoff = BASE_BACKOFF_SECONDS
        for attempt in range(MAX_ATTEMPTS_PER_MODEL):
            try:
                return _call_llm(model, system, user)
            except Exception as e:
                msg = str(e)
                if any(x in msg.lower() for x in ["rate", "429", "500", "503"]):
                    sleep_for = min(backoff, MAX_BACKOFF_SECONDS)
                    print(f"   [{model}] attempt {attempt+1} failed: {msg[:80]}; sleeping {sleep_for:.0f}s")
                    time.sleep(sleep_for)
                    backoff *= 2
                    continue
                print(f"   [{model}] Non-retryable error: {msg[:120]}")
                break
    raise RuntimeError("All LLM models failed")

def _extract_json_object(s: str) -> Dict[str, Any]:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n", "", s)
        s = re.sub(r"\n```\s*$", "", s)
    start = s.find("{")
    if start == -1:
        raise ValueError("No JSON object found in response")
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return json.loads(s[start:i+1])
    raise ValueError("Unbalanced JSON object in output")

# --------------------------------------------------------------------------
# Text Normalization & Slug Helpers
# --------------------------------------------------------------------------

_SLUG_RE = re.compile(r"[^a-z0-9]+")

def slugify(name: str, max_len: int = 80) -> str:
    s = name.lower()
    s = _SLUG_RE.sub("_", s).strip("_")
    return s[:max_len] or "unnamed"

def make_canonical(chunk_id: str, name: str) -> str:
    """Legacy helper kept for compatibility (used by step/warning pipelines)."""
    return f"{slugify(name)}__{chunk_id}"

def _normalize_type(t: str, allowed: List[str], default: str) -> str:
    if not t:
        return default
    t = t.strip().upper().replace(" ", "_")
    if t in allowed:
        return t
    for a in allowed:
        if a in t or t in a:
            return a
    return default

def _is_generic_entity(name: str, description: str) -> bool:
    """Filter out low-signal entities that pollute the graph."""
    n = name.strip().lower()
    if n in GENERIC_ENTITY_BLOCKLIST:
        return True
    # Single-word generic nouns not in the blocklist but very short & non-specific
    if len(n) <= 3:
        return True
    return False

# --------------------------------------------------------------------------
# LLM Output Post-processing
# --------------------------------------------------------------------------

def _postprocess_llm_output(
    chunk_id: str,
    raw: Dict[str, Any],
    extra_entity_fields: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Convert raw LLM output into normalized entity/relation records.

    NOTE: Canonical names here are LOCAL to this chunk. Global deduplication
    happens later in the RESOLVE stage.
    """
    raw_entities = raw.get("entities", []) or []
    raw_relations = raw.get("relations", []) or []

    id_map: Dict[Any, str] = {}
    entities: List[Dict[str, Any]] = []
    seen_names: Dict[str, str] = {}
    used_canonicals: Set[str] = set()

    for ent in raw_entities:
        if not isinstance(ent, dict):
            continue
        name = (ent.get("name") or "").strip()
        if not name:
            continue
        description = (ent.get("description") or "").strip()

        # Skip generic low-signal entities
        if _is_generic_entity(name, description):
            continue

        key = name.lower()
        if key in seen_names:
            if "id" in ent:
                id_map[ent["id"]] = seen_names[key]
            continue

        canonical = make_canonical(chunk_id, name)
        if canonical in used_canonicals:
            canonical = f"{canonical}_{len(entities)}"
        used_canonicals.add(canonical)

        record = {
            "canonical_name": canonical,
            "name": name,
            "type": _normalize_type(ent.get("type", ""), ENTITY_TYPES, "CONCEPT"),
            "description": description,
            "source_chunk_id": chunk_id,
            "aliases": [],
        }
        if extra_entity_fields:
            record.update(extra_entity_fields)
        entities.append(record)
        seen_names[key] = canonical
        if "id" in ent:
            id_map[ent["id"]] = canonical

    valid = {e["canonical_name"] for e in entities}
    relations: List[Dict[str, Any]] = []
    for rel in raw_relations:
        if not isinstance(rel, dict):
            continue
        src = id_map.get(rel.get("source"))
        tgt = id_map.get(rel.get("target"))
        if not src or not tgt or src == tgt:
            continue
        if src not in valid or tgt not in valid:
            continue
        rtype = _normalize_type(rel.get("type", ""), RELATION_TYPES, "CONNECTED_TO")

        # Drop trivially-circular relations (e.g. "M6 steel spec APPLIES_TO steel")
        src_ent = next((e for e in entities if e["canonical_name"] == src), None)
        tgt_ent = next((e for e in entities if e["canonical_name"] == tgt), None)
        if src_ent and tgt_ent:
            src_l, tgt_l = src_ent["name"].lower(), tgt_ent["name"].lower()
            # If the target name is a substring of the source name, it's redundant
            if rtype == "APPLIES_TO" and tgt_l in src_l and len(tgt_l) >= 3:
                continue

        try:
            conf = float(rel.get("confidence", 0.8))
        except Exception:
            conf = 0.8

        rel_entry = {
            "source": src,
            "target": tgt,
            "type": rtype,
            "description": (rel.get("description") or "").strip(),
            "confidence": max(0.0, min(1.0, conf)),
            "source_chunk_id": chunk_id,
        }
        if extra_entity_fields and "table_id" in extra_entity_fields:
            rel_entry["table_id"] = extra_entity_fields["table_id"]

        relations.append(rel_entry)

    return entities, relations

# ==========================================================================
# PIPELINE 1: TEXT EXTRACTION
# ==========================================================================

TEXT_SYSTEM_PROMPT = """You are an entity and relation extraction engine for technical manuals.
Extract entities and relationships ONLY from the provided text string.

Return JSON in this format:
{
  "entities": [
    {
      "id": 1,
      "name": "Impeller",
      "type": "COMPONENT",
      "description": "Open impeller mounted to a shaft"
    }
  ],
  "relations": [
    {
      "source": 1,
      "target": 2,
      "type": "PART_OF",
      "description": "Impeller is attached to the shaft",
      "confidence": 0.95
    }
  ]
}

Allowed Types: COMPONENT, PART, TOOL, MATERIAL, SPEC, PROCEDURE, CONCEPT, DOCUMENT, MODEL.
Allowed Relations: PART_OF, HAS_PART, REQUIRES_TOOL, USES_MATERIAL, HAS_SPEC, APPLIES_TO, REFERENCES, LOCATED_ON, CONNECTED_TO, SAME_AS.

Rules:
- DO NOT extract warnings or numbered procedure steps (handled separately).
- DO NOT extract generic single nouns (e.g., "heat", "water", "oil", "purpose").
- Prefer specific, named entities (components, models, materials, tools).
- Names must be FULL and UNTRUNCATED.
- Use strictly valid JSON.
"""

def extract_text(chunk: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    chunk_id = str(chunk["id"])
    content = (chunk.get("content") or "").strip()
    content = content.replace("--- SURROUNDING TEXT ---", "").strip()

    if len(content) < 20:
        return [], []

    breadcrumb = chunk.get("header", {}).get("breadcrumb", "")
    title = chunk.get("title", "")

    key = _cache_key(TEXT_PROMPT_VERSION, chunk_id, content)
    cached = _cache_get(key)
    if cached is not None:
        raw = cached
    else:
        user_prompt = f"Chunk ID: {chunk_id}\nBreadcrumb: {breadcrumb}\nTitle: {title}\n\nContent:\n{content}"
        try:
            resp_str = _call_with_retry(TEXT_SYSTEM_PROMPT, user_prompt)
            raw = _extract_json_object(resp_str)
        except Exception as e:
            print(f"   [TEXT] Failed for chunk {chunk_id}: {e}")
            raw = {"entities": [], "relations": []}
        _cache_put(key, raw)

    ents, rels = _postprocess_llm_output(chunk_id, raw)
    print(f"   [TEXT] chunk {chunk_id:<8} -> {len(ents)} entities, {len(rels)} relations")
    return ents, rels

# ==========================================================================
# PIPELINE 2: STEP EXTRACTION (Deterministic)
# ==========================================================================

def extract_steps(chunk: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    chunk_id = str(chunk["id"])
    raw_steps = chunk.get("steps", []) or []
    if not raw_steps:
        return [], []

    steps: List[Dict[str, Any]] = []
    for i, s in enumerate(raw_steps):
        if not isinstance(s, dict):
            continue
        text = (s.get("text") or "").strip()
        marker = (s.get("marker") or "").strip()
        if not text:
            continue

        canonical = f"step_{i+1}__{chunk_id}"

        # Build a short human-readable name WITHOUT truncating mid-word.
        # Prefer the first sentence, capped at ~60 chars on a word boundary.
        first_sentence = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0]
        short_name = first_sentence
        if len(short_name) > 60:
            short_name = short_name[:60].rsplit(" ", 1)[0] + "…"

        steps.append({
            "canonical_name": canonical,
            "name": short_name,
            "full_text": text,
            "type": "STEP",
            "description": text,
            "marker": marker,
            "step_number": i + 1,
            "source_chunk_id": chunk_id,
            "aliases": [],
        })

    relations: List[Dict[str, Any]] = []
    for i in range(len(steps) - 1):
        relations.append({
            "source": steps[i]["canonical_name"],
            "target": steps[i+1]["canonical_name"],
            "type": "PRECEDES_STEP",
            "description": "Sequential step execution order",
            "confidence": 1.0,
            "source_chunk_id": chunk_id,
        })

    print(f"   [STEP] chunk {chunk_id:<8} -> {len(steps)} steps, {len(relations)} PRECEDES_STEP relations")
    return steps, relations

# ==========================================================================
# PIPELINE 3: WARNING EXTRACTION (Deterministic)
# ==========================================================================

def extract_warnings(chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
    chunk_id = str(chunk["id"])
    raw_warnings = chunk.get("warnings", []) or []
    if not raw_warnings:
        return []

    warnings: List[Dict[str, Any]] = []
    for i, w in enumerate(raw_warnings):
        if not isinstance(w, dict):
            continue
        level = (w.get("level") or "WARNING").strip().upper()
        text = (w.get("text") or "").strip()
        if not text:
            continue

        canonical = f"warning_{i+1}__{chunk_id}"
        # Same truncation-safe approach as steps.
        short_name = text
        if len(short_name) > 60:
            short_name = short_name[:60].rsplit(" ", 1)[0] + "…"

        warnings.append({
            "canonical_name": canonical,
            "name": f"[{level}] {short_name}",
            "full_text": text,
            "type": "WARNING",
            "level": level,
            "description": text,
            "source_chunk_id": chunk_id,
            "aliases": [],
        })

    print(f"   [WARNING] chunk {chunk_id:<5} -> {len(warnings)} warnings")
    return warnings

# ==========================================================================
# PIPELINE 4: TABLE EXTRACTION (LLM Per Table)
# ==========================================================================

TABLE_SYSTEM_PROMPT = """You convert structured table JSON into a knowledge graph.
Extract rows/column attributes into entities and relations.

Return JSON in this format:
{
  "entities": [
    {
      "id": 1,
      "name": "HL80M Maximum Pressure",
      "type": "SPEC",
      "description": "9.2 bar (133 psi)"
    }
  ],
  "relations": [
    {
      "source": 1,
      "target": 2,
      "type": "HAS_SPEC",
      "description": "Specification belongs to pump model HL80M",
      "confidence": 0.95
    }
  ]
}

Allowed Types: COMPONENT, PART, TOOL, MATERIAL, SPEC, PROCEDURE, CONCEPT, DOCUMENT, MODEL, TABLE.
Allowed Relations: PART_OF, HAS_PART, REQUIRES_TOOL, USES_MATERIAL, HAS_SPEC, APPLIES_TO, REFERENCES, LOCATED_ON, CONNECTED_TO, SAME_AS, CAUSES_FAULT, RESOLVED_BY.

Rules:
- Emit ONE entity per meaningful row-value pair, NOT one per cell.
- Model names (e.g., HL80M) should be a MODEL entity, referenced via HAS_SPEC.
- Do NOT create an entity for a column header alone; only for row-level facts.
- Names must be FULL and UNTRUNCATED.
"""

def extract_tables(chunk: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    chunk_id = str(chunk["id"])
    raw_tables = chunk.get("tables", []) or []
    if not raw_tables:
        return [], []

    all_ents: List[Dict[str, Any]] = []
    all_rels: List[Dict[str, Any]] = []

    for idx, tbl in enumerate(raw_tables):
        if not isinstance(tbl, dict):
            continue

        table_id = f"{chunk_id}_table_{idx}"
        table_json_str = json.dumps(tbl, ensure_ascii=False, indent=2)

        key = _cache_key(TABLE_PROMPT_VERSION, chunk_id, f"tbl_{idx}_{table_json_str}")
        cached = _cache_get(key)
        if cached is not None:
            raw = cached
        else:
            user_prompt = f"Chunk ID: {chunk_id}\nTable ID: {table_id}\n\nTable JSON:\n{table_json_str}"
            try:
                resp_str = _call_with_retry(TABLE_SYSTEM_PROMPT, user_prompt)
                raw = _extract_json_object(resp_str)
            except Exception as e:
                print(f"   [TABLE] Failed table {table_id}: {e}")
                raw = {"entities": [], "relations": []}
            _cache_put(key, raw)

        extra = {
            "source_chunk_id": chunk_id,
            "table_id": table_id,
        }
        ents, rels = _postprocess_llm_output(chunk_id, raw, extra_entity_fields=extra)

        # Tag with table_id but DON'T pollute the canonical_name — the RESOLVE
        # stage will merge duplicates across tables.
        for e in ents:
            e["table_id"] = table_id
        for r in rels:
            r["table_id"] = table_id

        all_ents.extend(ents)
        all_rels.extend(rels)

    print(f"   [TABLE] chunk {chunk_id:<8} -> {len(all_ents)} entities, {len(all_rels)} relations across {len(raw_tables)} tables")
    return all_ents, all_rels

# ==========================================================================
# PIPELINE 5: ENTITY RESOLUTION / DEDUPLICATION
# ==========================================================================

def _name_key(name: str) -> str:
    """Normalize a name for merge comparison (lowercase, strip punctuation)."""
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()

# Type precedence: if the same name appears with different types, prefer
# the more specific / structural one.
_TYPE_PRECEDENCE = {
    "COMPONENT": 100,
    "PART": 95,
    "MODEL": 90,
    "TOOL": 85,
    "MATERIAL": 80,
    "PROCEDURE": 75,
    "SPEC": 70,
    "WARNING": 65,
    "STEP": 60,
    "TABLE": 55,
    "DOCUMENT": 50,
    "CONCEPT": 10,
}

def resolve_entities(
    entities: List[Dict[str, Any]],
    relations: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Merge duplicate entities across chunks and remap relations.

    Merge key: (normalized name, normalized type family)
    - Same normalized name + compatible type -> merged into one node.
    - 'source_chunk_id' becomes a list under 'mention_chunks'.
    - 'aliases' collects the per-chunk canonical names.
    """
    # Group by normalized name
    by_name: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for e in entities:
        by_name[_name_key(e["name"])].append(e)

    canonical_remap: Dict[str, str] = {}  # old canonical -> new canonical
    merged_entities: List[Dict[str, Any]] = []

    for name_key, group in by_name.items():
        if not name_key:
            continue

        # Pick the primary type (highest precedence)
        primary = max(group, key=lambda e: _TYPE_PRECEDENCE.get(e["type"], 0))
        primary_type = primary["type"]

        # But if types are wholly incompatible (e.g., STEP vs COMPONENT), split
        # into buckets by type family.
        type_buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for e in group:
            type_buckets[e["type"]].append(e)

        # Structural types that should share a node: COMPONENT/PART/MODEL/TOOL/MATERIAL
        STRUCTURAL = {"COMPONENT", "PART", "MODEL", "TOOL", "MATERIAL"}
        structural_group = []
        for t, bucket in type_buckets.items():
            if t in STRUCTURAL:
                structural_group.extend(bucket)

        final_groups: List[List[Dict[str, Any]]] = []
        if structural_group:
            final_groups.append(structural_group)
        for t, bucket in type_buckets.items():
            if t not in STRUCTURAL:
                final_groups.append(bucket)

        for bucket in final_groups:
            chosen_type = max(bucket, key=lambda e: _TYPE_PRECEDENCE.get(e["type"], 0))["type"]
            # Build a globally unique canonical slug
            base_slug = slugify(bucket[0]["name"])
            new_canonical = base_slug

            mention_chunks: List[str] = []
            mention_tables: List[str] = []
            aliases: Set[str] = set()
            descriptions: List[str] = []

            for e in bucket:
                canonical_remap[e["canonical_name"]] = new_canonical
                cid = e.get("source_chunk_id")
                if cid and cid not in mention_chunks:
                    mention_chunks.append(cid)
                tid = e.get("table_id")
                if tid and tid not in mention_tables:
                    mention_tables.append(tid)
                if e["canonical_name"] != new_canonical:
                    aliases.add(e["canonical_name"])
                if e.get("description") and e["description"] not in descriptions:
                    descriptions.append(e["description"])

            merged = {
                "canonical_name": new_canonical,
                "name": bucket[0]["name"],
                "type": chosen_type,
                "description": descriptions[0] if descriptions else "",
                "mention_chunks": mention_chunks,
                "source_chunk_id": mention_chunks[0] if mention_chunks else None,
                "aliases": sorted(aliases),
            }
            if mention_tables:
                merged["mention_tables"] = mention_tables
            merged_entities.append(merged)

    # Remap relations
    remapped_relations: List[Dict[str, Any]] = []
    seen_rel_keys: Set[Tuple[str, str, str]] = set()
    for r in relations:
        src = canonical_remap.get(r["source"], r["source"])
        tgt = canonical_remap.get(r["target"], r["target"])
        if src == tgt:
            continue
        key = (src, tgt, r["type"])
        # Drop exact duplicate relations (from merged chunks)
        if key in seen_rel_keys:
            # But preserve the source_chunk_id as a list
            continue
        seen_rel_keys.add(key)
        new_r = dict(r)
        new_r["source"] = src
        new_r["target"] = tgt
        remapped_relations.append(new_r)

    print(f"   [RESOLVE] {len(entities)} -> {len(merged_entities)} entities; "
          f"{len(relations)} -> {len(remapped_relations)} relations")
    return merged_entities, remapped_relations

# ==========================================================================
# PIPELINE 6: CROSS-LINKING (steps/warnings <-> entities)
# ==========================================================================

# Words we ignore when looking for entity mentions inside step/warning text
_MENTION_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for",
    "with", "by", "at", "as", "is", "are", "be", "if", "then",
    "this", "that", "these", "those", "it", "its", "any", "all",
    "from", "into", "out", "up", "down", "over", "under", "not",
}

def _build_mention_index(
    entities: List[Dict[str, Any]],
) -> Dict[str, List[str]]:
    """
    Build a lookup from lowercase entity name -> list of canonical names.
    Used to find entity mentions inside step / warning descriptions.
    """
    index: Dict[str, List[str]] = defaultdict(list)
    for e in entities:
        n = e["name"].strip().lower()
        if not n:
            continue
        # Skip ultra-short names — they cause false positives
        if len(n) < 3 and n not in {"oil", "gas"}:
            continue
        if n in _MENTION_STOPWORDS:
            continue
        index[n].append(e["canonical_name"])
        # Also index slugified variants
        slug = slugify(n)
        if slug and slug != n:
            index[slug].append(e["canonical_name"])
    return index

_MENTION_RE_CACHE: Dict[str, re.Pattern] = {}

def _find_mentions(
    text: str,
    mention_index: Dict[str, List[str]],
) -> Set[str]:
    """
    Find entity canonical names mentioned in a piece of text.
    Uses word-boundary matching to avoid substring false positives.
    """
    found: Set[str] = set()
    text_l = text.lower()

    for name, canonicals in mention_index.items():
        # Cache compiled regex per name
        pattern = _MENTION_RE_CACHE.get(name)
        if pattern is None:
            # Escape and allow word boundaries around it
            pattern = re.compile(r"(?<![a-z0-9])" + re.escape(name) + r"(?![a-z0-9])")
            _MENTION_RE_CACHE[name] = pattern
        if pattern.search(text_l):
            found.update(canonicals)
    return found

def cross_link(
    entities: List[Dict[str, Any]],
    relations: List[Dict[str, Any]],
    steps: List[Dict[str, Any]],
    warnings: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Add MENTIONS relations from steps -> entities and WARNS_ABOUT relations
    from warnings -> entities. Also link warnings to steps that appear in the
    same chunk (WARNS_ABOUT is chunk-level proximity).
    """
    mention_index = _build_mention_index(entities)

    # Map canonical -> type so we can decide on relation semantics
    type_of = {e["canonical_name"]: e["type"] for e in entities}

    new_relations: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str, str]] = set(
        (r["source"], r["target"], r["type"]) for r in relations
    )

    # Steps -> entities
    for s in steps:
        src = s["canonical_name"]
        text = s.get("full_text") or s.get("description") or ""
        for tgt in _find_mentions(text, mention_index):
            if tgt == src:
                continue
            key = (src, tgt, "MENTIONS")
            if key in seen:
                continue
            seen.add(key)
            new_relations.append({
                "source": src,
                "target": tgt,
                "type": "MENTIONS",
                "description": f"Step text references {tgt.split('__')[0].replace('_',' ')}",
                "confidence": 0.75,
                "source_chunk_id": s.get("source_chunk_id"),
            })

    # Warnings -> entities
    for w in warnings:
        src = w["canonical_name"]
        text = w.get("full_text") or w.get("description") or ""
        for tgt in _find_mentions(text, mention_index):
            if tgt == src:
                continue
            key = (src, tgt, "WARNS_ABOUT")
            if key in seen:
                continue
            seen.add(key)
            new_relations.append({
                "source": src,
                "target": tgt,
                "type": "WARNS_ABOUT",
                "description": f"Warning concerns {tgt.split('__')[0].replace('_',' ')}",
                "confidence": 0.7,
                "source_chunk_id": w.get("source_chunk_id"),
            })

    # Warnings -> steps in the same chunk (proximity link)
    steps_by_chunk: Dict[str, List[str]] = defaultdict(list)
    for s in steps:
        steps_by_chunk[s.get("source_chunk_id")].append(s["canonical_name"])
    for w in warnings:
        cid = w.get("source_chunk_id")
        for s_canon in steps_by_chunk.get(cid, []):
            key = (w["canonical_name"], s_canon, "WARNS_ABOUT")
            if key in seen:
                continue
            seen.add(key)
            new_relations.append({
                "source": w["canonical_name"],
                "target": s_canon,
                "type": "WARNS_ABOUT",
                "description": "Warning applies to a step in the same section",
                "confidence": 0.6,
                "source_chunk_id": cid,
            })

    print(f"   [LINK] added {len(new_relations)} MENTIONS / WARNS_ABOUT relations")
    return relations + new_relations

# ==========================================================================
# Main Execution Loop
# ==========================================================================

def main():
    chunks_path = Path(CHUNKS_FILE)
    if not chunks_path.exists():
        raise FileNotFoundError(f"Missing input chunk file: {CHUNKS_FILE}")

    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    if LIMIT_CHUNKS > 0:
        chunks = chunks[:LIMIT_CHUNKS]

    print("=" * 65)
    print(f"RUNNING GRAPH EXTRACTION ({len(chunks)} Chunks)")
    print("=" * 65)

    text_entities, text_relations = [], []
    table_entities, table_relations = [], []
    step_entities, step_relations = [], []
    warning_entities = []

    for i, chunk in enumerate(chunks):
        cid = str(chunk.get("id", i))
        print(f"[{i+1}/{len(chunks)}] Processing Chunk: {cid}")

        # 1. Prose Text Pipeline
        e, r = extract_text(chunk)
        text_entities.extend(e)
        text_relations.extend(r)

        # 2. Step Pipeline
        e, r = extract_steps(chunk)
        step_entities.extend(e)
        step_relations.extend(r)

        # 3. Warning Pipeline
        e = extract_warnings(chunk)
        warning_entities.extend(e)

        # 4. Table Pipeline
        e, r = extract_tables(chunk)
        table_entities.extend(e)
        table_relations.extend(r)

    # Combine
    all_entities = text_entities + table_entities
    all_relations = text_relations + table_relations + step_relations

    # 5. Resolution / Deduplication
    print("-" * 65)
    print("Running entity resolution ...")
    all_entities, all_relations = resolve_entities(all_entities, all_relations)

    # 6. Cross-linking (steps + warnings into the graph)
    print("Running cross-linking ...")
    all_relations = cross_link(
        all_entities, all_relations, step_entities, warning_entities
    )

    # Also attach steps and warnings as first-class entities so downstream
    # graph construction sees them.
    all_entities_final = all_entities + step_entities + warning_entities

    # Save outputs
    Path(OUTPUT_ENTITIES).write_text(
        json.dumps(all_entities_final, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    Path(OUTPUT_RELATIONS).write_text(
        json.dumps(all_relations, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    Path(OUTPUT_WARNINGS).write_text(
        json.dumps(warning_entities, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    Path(OUTPUT_STEPS).write_text(
        json.dumps(step_entities, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n" + "=" * 65)
    print("EXTRACTION COMPLETE")
    print("=" * 65)
    print(f"Processed Chunks:        {len(chunks)}")
    print(f"Text Entities:           {len(text_entities):<6} | Text Relations:  {len(text_relations)}")
    print(f"Step Entities:           {len(step_entities):<6} | Step Relations:  {len(step_relations)}")
    print(f"Warning Entities:        {len(warning_entities):<6}")
    print(f"Table Entities:          {len(table_entities):<6} | Table Relations: {len(table_relations)}")
    print("-" * 65)
    print(f"Final unique entities:   {len(all_entities_final)}")
    print(f"Final relations:         {len(all_relations)}")
    print("-" * 65)
    print(f"Saved -> {OUTPUT_ENTITIES}")
    print(f"Saved -> {OUTPUT_RELATIONS}")
    print(f"Saved -> {OUTPUT_WARNINGS}")
    print(f"Saved -> {OUTPUT_STEPS}")

if __name__ == "__main__":
    main()