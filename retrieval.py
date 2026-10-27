"""
Phase 6: Query-time retrieval pipeline.

USER QUERY
  -> query understanding (entity linking, property entities filtered)
  -> [dense + sparse -> RRF #1 -> Top20]
  -> graph expansion (if any *concept* entities were found)
  -> collect warnings & steps attached to found entities + their communities
  -> re-score graph candidates (dense cosine on cached vectors)
  -> RRF #2 (Top20 + graph-only candidates, weight scaled by specificity)
       -> Top 5-10
  -> context construction (chunks WITH TABLES + ALL community warnings/steps)
  -> LLM -> grounded answer

Fallback: if entity linking finds nothing (or only property entities), graph
expansion is skipped, the Top20 hybrid list is trimmed to top_k, and context
= chunks only.
"""

import os
import re
import json
import numpy as np

from dataclasses import dataclass, field

from qdrant_client import QdrantClient, models
from neo4j import GraphDatabase
from fastembed import TextEmbedding, SparseTextEmbedding
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "pump_assembly")

NEO4J_URI  = os.environ.get("NEO4J_URI",  "bolt://localhost:7687")
NEO4J_USER     = os.environ.get("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")
if not NEO4J_PASSWORD:
    raise RuntimeError("NEO4J_PASSWORD environment variable is not set.")

DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SPARSE_MODEL = "Qdrant/bm25"

RRF_K = 60
TOP20 = 20
FINAL_TOP_K = 8
GRAPH_HOPS = 1

ALLOWED_RELATIONS = [
    "PART_OF", "HAS_PART", "REQUIRES_TOOL", "USES_MATERIAL",
    "HAS_SPEC", "APPLIES_TO", "REFERENCES", "LOCATED_ON",
    "CONNECTED_TO", "RESOLVED_BY",
    "PRECEDES_STEP", "MENTIONS", "WARNS_ABOUT",
]

# Warnings / steps rendered per-chunk. Caps prevent pathological fan-out
# but are set high enough that real content is not truncated.
MAX_WARNINGS_PER_CHUNK = 30
MAX_STEPS_PER_CHUNK = 40

MAX_TABLE_ROWS_IN_CONTEXT = 40
MAX_TABLES_PER_CHUNK = 4

GROQ_API_KEY = os.environ.get(
    "GROQ_API_KEY", ''

)
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY environment variable is not set.")
groq_client = Groq(api_key=GROQ_API_KEY)


# ---------------------------------------------------------------------------
# Entity classification constants
# ---------------------------------------------------------------------------

PROPERTY_ENTITY_PATTERNS = [
    re.compile(r"^maximum[_\s]", re.IGNORECASE),
    re.compile(r"^minimum[_\s]", re.IGNORECASE),
    re.compile(r"^normal[_\s]", re.IGNORECASE),
    re.compile(r"_capacity_\d", re.IGNORECASE),
    re.compile(r"_for_[a-z0-9_]+$", re.IGNORECASE),
    re.compile(r"_specification$", re.IGNORECASE),
]

GENERIC_ENTITY_NAMES = {
    "pump", "unit", "coupling", "valve", "bolt", "assembly",
    "engine", "component", "part", "driver", "motor",
    "skid", "trailer", "impeller", "volute",
}


def _is_property_entity(canonical_name: str) -> bool:
    return any(p.search(canonical_name) for p in PROPERTY_ENTITY_PATTERNS)


def _is_generic_entity(canonical_name: str) -> bool:
    return canonical_name.lower() in GENERIC_ENTITY_NAMES


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    query: str
    linked_entities: list[str]
    chunk_ids: list[str]
    chunks: list[dict] = field(default_factory=list)
    relations: list[dict] = field(default_factory=list)
    warnings: list[dict] = field(default_factory=list)
    steps: list[dict] = field(default_factory=list)
    used_graph: bool = False
    context: str = ""


# ---------------------------------------------------------------------------
# 1. Query understanding: entity linking
# ---------------------------------------------------------------------------

class EntityLinker:

    def __init__(self, entities_path: str):
        with open(entities_path, encoding="utf-8") as f:
            entities = json.load(f)

        self._surface_to_canonical: list[tuple[str, str]] = []
        for e in entities:
            canonical = e.get("canonical_name")
            if not canonical:
                continue
            forms = {
                e.get("name") or "",
                canonical,
                *(e.get("aliases") or []),
            }
            for form in forms:
                if form and len(form.strip()) > 2:
                    self._surface_to_canonical.append(
                        (form.lower().strip(), canonical)
                    )

        self._surface_to_canonical.sort(key=lambda x: -len(x[0]))

    def link(self, query: str, drop_properties: bool = True) -> list[str]:
        q_lower = query.lower()
        matched: list[str] = []
        consumed_spans: list[tuple[int, int]] = []

        for surface, canonical in self._surface_to_canonical:
            match = re.search(
                rf"(?<![a-z0-9]){re.escape(surface)}(?![a-z0-9])",
                q_lower,
            )
            if not match:
                continue
            span = (match.start(), match.end())
            if any(not (span[1] <= s or span[0] >= e) for s, e in consumed_spans):
                continue
            matched.append(canonical)
            consumed_spans.append(span)

        deduped = list(dict.fromkeys(matched))

        if drop_properties:
            deduped = [m for m in deduped if not _is_property_entity(m)]

        return deduped


# ---------------------------------------------------------------------------
# 2. Dense + sparse hybrid search, RRF #1
# ---------------------------------------------------------------------------

class HybridSearcher:
    def __init__(self, qdrant: QdrantClient):
        self.qdrant = qdrant
        self.dense_model = TextEmbedding(model_name=DENSE_MODEL)
        self.sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)

    def embed_query(self, query: str) -> tuple[list[float], models.SparseVector]:
        dense_vec = list(self.dense_model.embed([query]))[0].tolist()
        sparse_emb = list(self.sparse_model.query_embed([query]))[0]
        sparse_vec = models.SparseVector(
            indices=sparse_emb.indices.tolist(),
            values=sparse_emb.values.tolist(),
        )
        return dense_vec, sparse_vec

    def search(self, query: str, limit: int = TOP20) -> tuple[list[dict], list[float]]:
        dense_vec, sparse_vec = self.embed_query(query)

        result = self.qdrant.query_points(
            collection_name=QDRANT_COLLECTION,
            prefetch=[
                models.Prefetch(query=dense_vec, using="dense", limit=limit),
                models.Prefetch(query=sparse_vec, using="bm25", limit=limit),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )

        chunks = [
            {"chunk_id": p.payload["chunk_id"], **p.payload}
            for p in result.points
        ]
        return chunks, dense_vec


# ---------------------------------------------------------------------------
# 3. Graph expansion + warnings / steps collection
# ---------------------------------------------------------------------------

class GraphExpander:
    """
    Graph expansion has two jobs:

      1. Find related chunks the vector search missed (via relation hops).
      2. Fetch the warnings and steps that belong to the COMMUNITY of each
         retrieved chunk — so build_context can render every warning/step
         that lives in that chunk's section, without needing to guess
         which entity in the query pulled them in.

    The community-keyed step/warning maps are the important addition. They
    are keyed by chunk_id, so build_context can render "all warnings in
    this chunk" and "all steps in this chunk" without further graph work.
    """

    def __init__(self, driver):
        self.driver = driver

    def expand(
        self,
        canonical_names: list[str],
        retrieved_chunk_ids: list[str] | None = None,
        hops: int = GRAPH_HOPS,
    ) -> dict:
        if not canonical_names:
            return {
                "chunk_ids": set(),
                "relations": [],
                "warnings": [],
                "steps": [],
                "warnings_by_chunk": {},
                "steps_by_chunk": {},
            }

        safe_types = "|".join(
            re.sub(r"[^A-Z0-9_]+", "_", t) for t in ALLOWED_RELATIONS
        )

        relation_cypher = f"""
        MATCH (start:Entity)
        WHERE start.canonical_name IN $keys
        MATCH path = (start)-[:{safe_types}*1..{hops}]-(connected:Entity)
        RETURN start.canonical_name AS source,
               start.name           AS source_name,
               connected.canonical_name AS target,
               connected.name           AS target_name,
               connected.type           AS target_type,
               connected.source_chunk_id AS target_chunk,
               connected.mention_chunks  AS target_chunks,
               [rel IN relationships(path) | type(rel)] AS rel_types
        """

        chunk_ids: set[str] = set()
        relations: list[dict] = []
        warnings: list[dict] = []
        steps: list[dict] = []

        warnings_seen: set[str] = set()
        steps_seen: set[str] = set()

        with self.driver.session() as session:
            # -- Relation-hop expansion (unchanged) ------------------------
            for record in session.run(relation_cypher, keys=canonical_names):
                for cid in (record["target_chunks"] or []):
                    if cid:
                        chunk_ids.add(str(cid))
                if record["target_chunk"]:
                    chunk_ids.add(str(record["target_chunk"]))

                relations.append({
                    "source": record["source"],
                    "source_name": record["source_name"],
                    "target": record["target"],
                    "target_name": record["target_name"],
                    "target_type": record["target_type"],
                    "path": record["rel_types"],
                })

                ttype = (record["target_type"] or "").upper()
                tcanon = record["target"]
                if ttype == "WARNING" and tcanon not in warnings_seen:
                    warnings_seen.add(tcanon)
                    warnings.append({
                        "canonical_name": tcanon,
                        "name": record["target_name"],
                        "attached_to": record["source_name"],
                    })
                elif ttype == "STEP" and tcanon not in steps_seen:
                    steps_seen.add(tcanon)
                    steps.append({
                        "canonical_name": tcanon,
                        "name": record["target_name"],
                        "attached_to": record["source_name"],
                    })

            # -- Community-scoped collection, PER CHUNK --------------------
            # For each retrieved chunk, collect every warning and step
            # that lives in that chunk's community. This is the flat,
            # simple approach: if a warning/step belongs to the same
            # community as the chunk, it goes in the context.
            #
            # retrieved_chunk_ids is the union of the vector Top20 and the
            # relation-expanded graph chunks, so it represents every chunk
            # that might actually be used in the final answer.
            warnings_by_chunk: dict[str, list[dict]] = {}
            steps_by_chunk: dict[str, list[dict]] = {}

            community_chunk_ids = set(chunk_ids)
            if retrieved_chunk_ids:
                community_chunk_ids.update(str(c) for c in retrieved_chunk_ids)

            if community_chunk_ids:
                warnings_by_chunk = self._warnings_for_chunks(
                    session, list(community_chunk_ids)
                )
                steps_by_chunk = self._steps_for_chunks(
                    session, list(community_chunk_ids)
                )

                # Merge into the flat lists, deduplicated.
                for cid, ws in warnings_by_chunk.items():
                    for w in ws:
                        if w["canonical_name"] not in warnings_seen:
                            warnings_seen.add(w["canonical_name"])
                            warnings.append(w)
                for cid, ss in steps_by_chunk.items():
                    for s in ss:
                        if s["canonical_name"] not in steps_seen:
                            steps_seen.add(s["canonical_name"])
                            steps.append(s)

        return {
            "chunk_ids": chunk_ids,
            "relations": relations,
            "warnings": warnings,
            "steps": steps,
            "warnings_by_chunk": warnings_by_chunk,
            "steps_by_chunk": steps_by_chunk,
        }

    # -- Chunk-keyed collectors -------------------------------------------

    @staticmethod
    def _warnings_for_chunks(session, chunk_ids: list[str]) -> dict[str, list[dict]]:
        """
        For each chunk_id, return every WARNING entity that lives in the
        chunk's community.

        Uses collect() to avoid the row-multiplication problem: a warning
        shared by N seed entities previously produced N rows and exhausted
        the LIMIT before later chunks were visited.
        """
        cypher = """
        MATCH (c:Community {chunk_id: $chunk_id})-[:HAS_MEMBER]->(w:Entity)
        WHERE w.type = 'WARNING'
        RETURN w.canonical_name AS canonical_name,
               w.name           AS name,
               w.level          AS level,
               w.full_text      AS full_text
        LIMIT $limit
        """
        result: dict[str, list[dict]] = {}
        for cid in chunk_ids:
            warnings = []
            for record in session.run(
                cypher, chunk_id=cid, limit=MAX_WARNINGS_PER_CHUNK,
            ):
                warnings.append({
                    "canonical_name": record["canonical_name"],
                    "name": record["name"],
                    "level": record["level"],
                    "full_text": record["full_text"],
                    "source_chunk_id": cid,
                })
            if warnings:
                result[cid] = warnings
        return result

    @staticmethod
    def _steps_for_chunks(session, chunk_ids: list[str]) -> dict[str, list[dict]]:
        """
        For each chunk_id, return every STEP entity that lives in the
        chunk's community, ordered by step_number.
        """
        cypher = """
        MATCH (c:Community {chunk_id: $chunk_id})-[:HAS_MEMBER]->(s:Entity)
        WHERE s.type = 'STEP'
        RETURN s.canonical_name   AS canonical_name,
               s.name             AS name,
               s.step_number      AS step_number,
               s.marker           AS marker,
               s.full_text        AS full_text,
               s.source_chunk_id  AS source_chunk_id
        ORDER BY s.step_number
        LIMIT $limit
        """
        result: dict[str, list[dict]] = {}
        for cid in chunk_ids:
            steps = []
            for record in session.run(
                cypher, chunk_id=cid, limit=MAX_STEPS_PER_CHUNK,
            ):
                steps.append({
                    "canonical_name": record["canonical_name"],
                    "name": record["name"],
                    "step_number": record["step_number"],
                    "marker": record["marker"],
                    "full_text": record["full_text"],
                    # Use the STEP's own source_chunk_id, not the community
                    # chunk_id — a step from section 5.1.2 lives in the
                    # community keyed "5.1.2", so they agree, but be
                    # explicit to avoid drift.
                    "source_chunk_id": record["source_chunk_id"] or cid,
                })
            if steps:
                result[cid] = steps
        return result


# ---------------------------------------------------------------------------
# 4. Re-score graph candidates + RRF #2
# ---------------------------------------------------------------------------

def score_graph_candidates(
    qdrant: QdrantClient,
    chunk_ids: list[str],
    query_vec: list[float],
) -> list[dict]:
    if not chunk_ids:
        return []

    scroll_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="chunk_id",
                match=models.MatchAny(any=list(chunk_ids)),
            )
        ]
    )

    points, _ = qdrant.scroll(
        collection_name=QDRANT_COLLECTION,
        scroll_filter=scroll_filter,
        limit=len(chunk_ids),
        with_payload=True,
        with_vectors=True,
    )

    if not points:
        return []

    q = np.asarray(query_vec, dtype=np.float32)
    q_norm = q / (np.linalg.norm(q) + 1e-9)

    scored: list[dict] = []
    for p in points:
        vector = p.vector
        dense = vector.get("dense") if isinstance(vector, dict) else vector
        if dense is None:
            continue

        vec = np.asarray(dense, dtype=np.float32)
        vec_norm = vec / (np.linalg.norm(vec) + 1e-9)
        sim = float(np.dot(q_norm, vec_norm))

        scored.append({
            "chunk_id": p.payload["chunk_id"],
            "score": sim,
            **p.payload,
        })

    scored.sort(key=lambda x: -x["score"])
    return scored


def rrf_fuse(
    ranked_lists: list[list[dict]],
    id_key: str = "chunk_id",
    k: int = RRF_K,
    top_k: int = FINAL_TOP_K,
    weights: list[float] | None = None,
) -> list[dict]:
    if weights is None:
        weights = [1.0] * len(ranked_lists)

    if len(weights) != len(ranked_lists):
        raise ValueError("weights length must match ranked_lists length")

    scores: dict[str, float] = {}
    payloads: dict[str, dict] = {}

    for weight, ranked_list in zip(weights, ranked_lists):
        for rank, item in enumerate(ranked_list, start=1):
            cid = item[id_key]
            scores[cid] = scores.get(cid, 0.0) + weight * (1.0 / (k + rank))
            payloads.setdefault(cid, item)

    ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
    return [payloads[cid] for cid, _ in ranked]


def compute_graph_weight(linked_entities: list[str]) -> float:
    if not linked_entities:
        return 0.0

    n_generic = sum(1 for e in linked_entities if _is_generic_entity(e))
    n_specific = len(linked_entities) - n_generic

    if n_specific >= 1 and n_generic == 0:
        return 0.3
    if n_generic >= 1 and n_specific == 0:
        return 1.0
    return 0.6


# ---------------------------------------------------------------------------
# 5. TOC / front-matter filter
# ---------------------------------------------------------------------------

_TOC_INDICATORS = (
    "table of contents",
    "front matter",
)


def _is_toc_or_front_matter(chunk: dict) -> bool:
    haystack = " ".join([
        str(chunk.get("title") or ""),
        str(chunk.get("breadcrumb") or ""),
    ]).lower()
    return any(marker in haystack for marker in _TOC_INDICATORS)


# ---------------------------------------------------------------------------
# 6. Context construction + generation
# ---------------------------------------------------------------------------

def _truncate(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _render_chunk_tables(tables: list) -> str:
    if not tables:
        return ""

    rendered: list[str] = []
    tables_seen = 0

    for table_index, tbl in enumerate(tables, start=1):
        if not isinstance(tbl, dict):
            continue

        rows = tbl.get("table")
        if not isinstance(rows, list) or not rows:
            continue

        tables_seen += 1
        if tables_seen > MAX_TABLES_PER_CHUNK:
            rendered.append(f"(... more tables omitted ...)")
            break

        rendered.append(f"Table {table_index}:")

        for row_index, row in enumerate(rows):
            if row_index >= MAX_TABLE_ROWS_IN_CONTEXT:
                rendered.append("  | (…more rows omitted…) |")
                break
            if not isinstance(row, list):
                continue
            cells = [
                str(c).replace("\n", " ").strip()
                for c in row
                if c is not None and str(c).strip()
            ]
            if cells:
                rendered.append("  | " + " | ".join(cells) + " |")

        rendered.append("")

    return "\n".join(rendered)


def build_context(
    chunks,
    relations,
    warnings,
    steps,
    warnings_by_chunk=None,
    steps_by_chunk=None,
):
    """
    Render the full context.

    Layout:
      ## Retrieved manual excerpts
         [chunk_id] — breadcrumb
         content
         Steps in this excerpt:    <- every step in the chunk's community
         Tables in this excerpt:
         Warnings in this excerpt: <- every warning in the chunk's community

      ## Related facts from the knowledge graph
      ## Directly-linked safety warnings
      ## Section-level safety warnings (may apply)
    """

    warnings_by_chunk = warnings_by_chunk or {}
    steps_by_chunk = steps_by_chunk or {}

    # Fall back to a global index if per-chunk maps weren't provided.
    if not steps_by_chunk:
        for s in steps:
            cid = s.get("source_chunk_id")
            if cid:
                steps_by_chunk.setdefault(str(cid), []).append(s)
    if not warnings_by_chunk:
        for w in warnings:
            cid = w.get("source_chunk_id")
            if cid:
                warnings_by_chunk.setdefault(str(cid), []).append(w)

    # Sort steps per chunk.
    for cid in steps_by_chunk:
        steps_by_chunk[cid].sort(
            key=lambda s: (s.get("step_number") is None, s.get("step_number") or 0)
        )

    parts = ["## Retrieved manual excerpts\n"]

    for c in chunks:
        cid = str(c.get("chunk_id"))
        cite = f"[{cid} | p.{c.get('page_start')}-{c.get('page_end')}]"
        breadcrumb = c.get("breadcrumb") or c.get("title") or ""
        content = c.get("content") or ""

        if breadcrumb:
            parts.append(f"{cite} — {breadcrumb}")
        else:
            parts.append(cite)

        if content.strip():
            parts.append(content.strip())

        # Steps that live in this chunk's community
        chunk_steps = steps_by_chunk.get(cid) or c.get("steps") or []
        if chunk_steps:
            parts.append("")
            parts.append("Steps in this excerpt:")
            for s in chunk_steps:
                marker = s.get("marker") or f"{s.get('step_number', '?')}."
                text = (s.get("full_text") or s.get("text") or s.get("name") or "").replace("\n", " ")
                parts.append(f"  {marker} {text}")

        # Tables for this chunk
        tables_text = _render_chunk_tables(c.get("tables") or [])
        if tables_text:
            parts.append("")
            parts.append("Tables in this excerpt:")
            parts.append(tables_text)

        # Warnings that live in this chunk's community
        chunk_warnings = warnings_by_chunk.get(cid) or c.get("warnings") or []
        if chunk_warnings:
            parts.append("")
            parts.append("Warnings in this excerpt:")
            for w in chunk_warnings:
                level = w.get("level") or "WARNING"
                text = (w.get("full_text") or w.get("text") or w.get("name") or "").replace("\n", " ")
                parts.append(f"  [{level}] {text}")

        parts.append("")

    # -- Relations --------------------------------------------------------
    if relations:
        parts.append("## Related facts from the knowledge graph\n")
        seen_triples: set[tuple] = set()
        for r in relations[:40]:
            src = r.get("source_name") or r["source"]
            tgt = r.get("target_name") or r["target"]
            path = "/".join(r.get("path") or [])
            key = (src, path, tgt)
            if key in seen_triples:
                continue
            seen_triples.add(key)
            parts.append(f"- {src} --[{path}]--> {tgt}")

    return "\n".join(parts).strip()


def generate_answer(query: str, context: str) -> str:
    system = (
        "You are a technical assistant answering questions about a pump "
        "operation and maintenance manual. Answer ONLY using the provided "
        "context.\n"
        "\n"
        "Rules:\n"
        "1. Cite every factual claim using the [chunk_id | p.X-Y] markers "
        "   given in the context.\n"
        "2. Numbers, model numbers, and specifications must be quoted "
        "   verbatim from the context. Do not approximate, round, or "
        "   convert units unless the context provides both.\n"
        "3. If the answer is inside a table, read the table cells directly "
        "   and quote the exact value.\n"
        "4. Warnings: the context may contain many warnings, and most of "
        "   them will not be relevant to the user's specific question. "
        "   Surface ONLY the warnings whose text directly addresses the "
        "   component, procedure, or failure mode the user asked about. "
        "   A warning is relevant if its text names or clearly refers to "
        "   the same component, procedure, or system state as the query "
        "   — even if the warning was found inside a chunk whose section "
        "   title is about a broader topic. Do NOT surface warnings that "
        "   only mention the queried component in passing, that concern a "
        "   different component, or that are about general operating "
        "   safety. Do NOT invent, paraphrase, or generalize warnings. If "
        "   no warning in the context is relevant, say so and move on.\n"
        "5. Steps: the context may contain many procedural steps across "
        "   multiple sections. Present only the steps that belong to the "
        "   specific procedure the user asked about. Include conditional "
        "   or branching steps (steps that say 'depending on X, do A or B') "
        "   with their conditions clearly stated. Do NOT include steps from "
        "   unrelated procedures that merely share a component or section.\n"
        "6. If the context does not contain the answer, say so plainly. "
        "   Do NOT use prior knowledge about pumps, engines, or manuals "
        "   to fill in gaps.\n"
        "7. Do NOT add commentary, notes, or caveats that are not "
        "   supported by a citation.\n"
        "8. Filter by CONTENT, not by section title. A chunk whose "
        "   section title is about a broader topic (e.g. 'Pump-to-driver "
        "   alignment' for a query about the coupling) IS a valid source "
        "   if its content actually discusses the queried component. "
        "   Only reject a chunk if its content is about a different, "
        "   unrelated component."
    )
    resp = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": f"Question: {query}\n\nContext:\n{context}",
            },
        ],
        temperature=0.2,
        max_completion_tokens=4096,
        reasoning_effort="medium",
        reasoning_format="hidden",
    )
    return resp.choices[0].message.content


# ---------------------------------------------------------------------------
# 7. Orchestration
# ---------------------------------------------------------------------------

class GraphRAGPipeline:
    def __init__(self, entities_path: str):
        self.qdrant = QdrantClient(url=QDRANT_URL)
        self.driver = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
        self.linker = EntityLinker(entities_path)
        self.searcher = HybridSearcher(self.qdrant)
        self.expander = GraphExpander(self.driver)
        
    def retrieve(self, query: str, graph_weight: float | None = None) -> RetrievalResult:
        """Retrieval only — no LLM call. Optionally override the graph weight."""
        linked_entities = self.linker.link(query, drop_properties=True)
        top20, query_vec = self.searcher.search(query, limit=TOP20)
        top20 = [c for c in top20 if not _is_toc_or_front_matter(c)]

        relations, warnings, steps = [], [], []
        warnings_by_chunk, steps_by_chunk = {}, {}

        if not linked_entities:
            final_chunks = top20[:FINAL_TOP_K]
            used_graph = False
        else:
            vector_chunk_ids = [c["chunk_id"] for c in top20]
            expansion = self.expander.expand(
                linked_entities,
                retrieved_chunk_ids=vector_chunk_ids,
            )
            top20_ids = set(vector_chunk_ids)
            graph_candidate_ids = [
                cid for cid in expansion["chunk_ids"] if cid not in top20_ids
            ]
            graph_candidates = score_graph_candidates(
                self.qdrant, graph_candidate_ids, query_vec,
            )
            graph_candidates = [gc for gc in graph_candidates if not _is_toc_or_front_matter(gc)]

            if graph_weight is None:
                graph_weight = compute_graph_weight(linked_entities)

            n_graph = min(
                max(0, round(graph_weight * 3)),
                len(graph_candidates),
                FINAL_TOP_K,
            )
            graph_selected = graph_candidates[:n_graph]
            graph_selected_ids = {str(c["chunk_id"]) for c in graph_selected}
            final_chunks = top20[: FINAL_TOP_K - n_graph] + graph_selected
            selected_ids = {str(c["chunk_id"]) for c in final_chunks}
            if len(final_chunks) < FINAL_TOP_K:
                for candidate in top20:
                    candidate_id = str(candidate["chunk_id"])
                    if candidate_id not in selected_ids:
                        final_chunks.append(candidate)
                        selected_ids.add(candidate_id)
                    if len(final_chunks) >= FINAL_TOP_K:
                        break

            relations = expansion["relations"]
            warnings = expansion["warnings"]
            steps = expansion["steps"]
            warnings_by_chunk = expansion.get("warnings_by_chunk", {})
            steps_by_chunk = expansion.get("steps_by_chunk", {})
            used_graph = any(str(c["chunk_id"]) in graph_selected_ids for c in final_chunks)

        context = build_context(
            final_chunks, relations, warnings, steps,
            warnings_by_chunk=warnings_by_chunk,
            steps_by_chunk=steps_by_chunk,
        )

        return RetrievalResult(
            query=query,
            linked_entities=linked_entities,
            chunk_ids=[c["chunk_id"] for c in final_chunks],
            chunks=final_chunks,
            relations=relations,
            warnings=warnings,
            steps=steps,
            used_graph=used_graph,
        ), context


    def generate(self, query: str, context: str) -> str:
        return generate_answer(query, context)

    def answer(self, query: str):
        # -- query understanding --------------------------------------
        linked_entities = self.linker.link(query, drop_properties=True)

        # -- RRF #1: dense + sparse -> Top20 --------------------------
        top20, query_vec = self.searcher.search(query, limit=TOP20)
        top20 = [c for c in top20 if not _is_toc_or_front_matter(c)]

        relations: list[dict] = []
        warnings: list[dict] = []
        steps: list[dict] = []
        warnings_by_chunk: dict[str, list[dict]] = {}
        steps_by_chunk: dict[str, list[dict]] = {}

        # -- fallback: no *usable* linked entities = skip graph -------
        if not linked_entities:
            final_chunks = top20[:FINAL_TOP_K]
            used_graph = False
        else:
            # Pass the vector Top20 chunk ids so the expander knows which
            # chunk communities to harvest warnings/steps from.
            vector_chunk_ids = [c["chunk_id"] for c in top20]

            expansion = self.expander.expand(
                linked_entities,
                retrieved_chunk_ids=vector_chunk_ids,
            )

            top20_ids = set(vector_chunk_ids)
            graph_candidate_ids = [
                cid for cid in expansion["chunk_ids"] if cid not in top20_ids
            ]

            graph_candidates = score_graph_candidates(
                self.qdrant,
                graph_candidate_ids,
                query_vec,
            )

            graph_candidates = [
                gc for gc in graph_candidates
                if not _is_toc_or_front_matter(gc)
            ]

            graph_weight = compute_graph_weight(linked_entities)
            final_chunks = rrf_fuse(
                [top20, graph_candidates],
                top_k=FINAL_TOP_K,
                weights=[1.0, graph_weight],
            )

            final_chunks = [
                c for c in final_chunks if not _is_toc_or_front_matter(c)
            ]

            relations = expansion["relations"]
            warnings = expansion["warnings"]
            steps = expansion["steps"]
            warnings_by_chunk = expansion.get("warnings_by_chunk", {})
            steps_by_chunk = expansion.get("steps_by_chunk", {})

            used_graph = any(str(c["chunk_id"]) in graph_selected_ids for c in final_chunks)

        # -- context + generation -------------------------------------
        context = build_context(
            final_chunks,
            relations,
            warnings,
            steps,
            warnings_by_chunk=warnings_by_chunk,
            steps_by_chunk=steps_by_chunk,
        )
        answer_text = generate_answer(query, context)

        return (
            RetrievalResult(
                query=query,
                linked_entities=linked_entities,
                chunk_ids=[c["chunk_id"] for c in final_chunks],
                chunks=final_chunks,
                relations=relations,
                warnings=warnings,
                steps=steps,
                used_graph=used_graph,
            ),
            answer_text,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pipeline = GraphRAGPipeline(entities_path="entities.json")

    test_queries = [
        "How do I configure the pump for flooded suction operation, and what safety precautions apply?",
        "What maintenance does the air compressor need across all service intervals?",
        "What are all the safety warnings that apply to the coupling?",
    ]

    for q in test_queries:
        print("=" * 70)
        print(f"Q: {q}")
        print("=" * 70)

        result, answer_text = pipeline.answer(q)

        print(f"Linked entities : {result.linked_entities}")
        print(f"Used graph      : {result.used_graph}")
        print(f"Chunks used     : {result.chunk_ids}")
        print(f"Relations       : {len(result.relations)}")
        print(f"Warnings        : {len(result.warnings)}")
        print(f"Steps           : {len(result.steps)}")
        print()
        print("--- Answer ---")
        print(answer_text)
        print()