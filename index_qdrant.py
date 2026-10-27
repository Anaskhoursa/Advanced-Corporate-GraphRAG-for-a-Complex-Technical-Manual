"""
Hybrid Dense + BM25 Qdrant ingestion for the HL Series Pump Manual.

Pipeline:

    all_chunks.json
        ↓
    embedding text (breadcrumb + content + steps + warnings + tables)
        ├── Dense embedding
        │     └── SentenceTransformers
        │
        └── Sparse BM25 embedding
              └── FastEmbed Qdrant/bm25
        ↓
    Qdrant
        ├── dense named vector
        └── bm25 named sparse vector
        ↓
    Payload containing complete chunk metadata

Dense:
    sentence-transformers/all-MiniLM-L6-v2

Sparse:
    Qdrant/bm25

Install:

    pip install qdrant-client sentence-transformers fastembed tqdm

Local Qdrant:

    docker run -p 6333:6333 qdrant/qdrant

Run:

    python embed_to_qdrant.py
"""

# ============================================================================
# CONFIGURATION
# ============================================================================

from dotenv import load_dotenv

load_dotenv()

CHUNKS_PATH = "all_chunks.json"


# Qdrant Cloud:
#
# QDRANT_URL = "https://xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx.cloud.qdrant.io"
#
QDRANT_API_KEY = None

import os
QDRANT_URL      = os.environ.get("QDRANT_URL", "http://qdrant:6333")
COLLECTION_NAME = os.environ.get("QDRANT_COLLECTION", "pump_assembly")

# ---------------------------------------------------------------------------
# Dense embedding model
# ---------------------------------------------------------------------------

DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

DENSE_VECTOR_SIZE = 384

# ---------------------------------------------------------------------------
# Sparse BM25 model
# ---------------------------------------------------------------------------

SPARSE_MODEL = "Qdrant/bm25"

# ---------------------------------------------------------------------------
# Named vectors
# ---------------------------------------------------------------------------

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "bm25"

# ---------------------------------------------------------------------------
# Upload configuration
# ---------------------------------------------------------------------------

BATCH_SIZE = 32

# IMPORTANT:
#
# True  -> delete and rebuild collection
# False -> keep existing collection and upsert
#
# Use True for the first run.
# Once the collection is correct, switch to False.
#
RECREATE_COLLECTION = True

# ---------------------------------------------------------------------------
# Chunk filtering
# ---------------------------------------------------------------------------

# Skip chunks that carry no retrievable content. A chunk is "empty" if it has
# no content AND no steps AND no warnings AND no tables. These are usually
# section-heading stubs (e.g. "1 Introduction and Safety").
SKIP_EMPTY_CHUNKS = True

# When True, chunks with empty content but populated steps/warnings/tables are
# still embedded using their breadcrumb + structured content.
EMBED_STRUCTURED_ONLY_CHUNKS = True


# ============================================================================
# IMPORTS
# ============================================================================

import hashlib
import json

from pathlib import Path
from typing import Any

from tqdm import tqdm

from sentence_transformers import SentenceTransformer

from fastembed import SparseTextEmbedding

from qdrant_client import QdrantClient

from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
    Modifier,
    PointStruct,
    SparseVector,
)


# ============================================================================
# LOAD CHUNKS
# ============================================================================

def load_chunks(path: str) -> list[dict[str, Any]]:
    """
    Load and validate all_chunks.json.

    Expected schema (per chunk):
        {
            "id":         str,
            "title":      str,
            "header": {
                "section":     str | None,
                "subsection":  str | None,
                "parent":      str | None,
                "breadcrumb":  str,
            },
            "page_start": int,
            "page_end":   int,
            "content":    str,
            "warnings":   list,
            "steps":      list,
            "tables":     list,
        }
    """

    path_obj = Path(path)

    if not path_obj.exists():
        raise FileNotFoundError(f"Chunk file not found: {path}")

    with open(path_obj, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected chunk JSON to contain a list.")

    if not data:
        raise ValueError("Chunk JSON is empty.")

    required = {"id", "header"}

    for index, chunk in enumerate(data):
        if not isinstance(chunk, dict):
            raise ValueError(f"Chunk {index} is not an object.")

        missing = required - set(chunk.keys())
        if missing:
            raise ValueError(
                f"Chunk {index} is missing required keys: {sorted(missing)}"
            )

        header = chunk.get("header")
        if not isinstance(header, dict):
            raise ValueError(
                f"Chunk {index} has non-dict 'header'. Got {type(header).__name__}."
            )

    return data


# ============================================================================
# CHUNK FILTERING
# ============================================================================

def _has_structured_content(chunk: dict[str, Any]) -> bool:
    """Return True if the chunk contains steps, warnings, or tables."""

    return bool(
        (chunk.get("steps") or [])
        or (chunk.get("warnings") or [])
        or (chunk.get("tables") or [])
    )


def _has_text_content(chunk: dict[str, Any]) -> bool:
    """Return True if the chunk has any non-whitespace text content."""

    content = chunk.get("content") or ""
    return bool(str(content).strip())


def filter_chunks(
    chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Drop empty section-heading stubs that would otherwise produce
    zero-vector points and pollute retrieval.

    Keeps:
      - chunks with text content
      - chunks with structured content (steps/warnings/tables)
    """

    if not SKIP_EMPTY_CHUNKS:
        return chunks

    kept: list[dict[str, Any]] = []
    skipped: list[str] = []

    for chunk in chunks:
        has_text = _has_text_content(chunk)
        has_struct = _has_structured_content(chunk)

        if has_text:
            kept.append(chunk)
            continue

        if has_struct and EMBED_STRUCTURED_ONLY_CHUNKS:
            kept.append(chunk)
            continue

        skipped.append(str(chunk.get("id")))

    if skipped:
        print(f"[filter] skipped {len(skipped)} empty chunks: {skipped}")

    return kept


# ============================================================================
# EMBEDDING TEXT
# ============================================================================

def _get_breadcrumb(chunk: dict[str, Any]) -> str:
    """
    Safely extract the breadcrumb string from chunk['header']['breadcrumb'].

    The original script had a bug calling .breadcrumb on a dict. This helper
    never raises regardless of header shape.
    """

    header = chunk.get("header")
    if not isinstance(header, dict):
        return ""

    breadcrumb = header.get("breadcrumb")
    if breadcrumb is None:
        return ""

    return str(breadcrumb).strip()


def _render_steps(steps: list[Any]) -> str:
    """Flatten a chunk's steps list into a readable string for embedding."""

    if not steps:
        return ""

    lines: list[str] = []
    for i, step in enumerate(steps, start=1):
        if isinstance(step, dict):
            text = (step.get("text") or "").strip()
            marker = (step.get("marker") or f"{i}.").strip()
            if text:
                lines.append(f"{marker} {text}")
        elif isinstance(step, str):
            if step.strip():
                lines.append(step.strip())

    return "\n".join(lines)


def _render_warnings(warnings: list[Any]) -> str:
    """Flatten a chunk's warnings list into a readable string for embedding."""

    if not warnings:
        return ""

    lines: list[str] = []
    for warning in warnings:
        if isinstance(warning, dict):
            level = (warning.get("level") or "WARNING").strip().upper()
            text = (warning.get("text") or "").strip()
            if text:
                lines.append(f"[{level}] {text}")
        elif isinstance(warning, str):
            if warning.strip():
                lines.append(warning.strip())

    return "\n".join(lines)


def _render_tables(tables: list[Any]) -> str:
    """
    Flatten a chunk's tables list into a readable string for embedding.

    Tables in all_chunks.json follow the shape:
        {"page": ..., "ordinal": ..., "table": [[cell, cell], ...]}
    """

    if not tables:
        return ""

    rendered: list[str] = []

    for table_index, table_entry in enumerate(tables, start=1):
        if not isinstance(table_entry, dict):
            continue

        rows = table_entry.get("table")
        if not isinstance(rows, list) or not rows:
            continue

        rendered.append(f"Table {table_index}:")

        for row in rows:
            if not isinstance(row, list):
                continue

            cells = [
                str(cell).strip().replace("\n", " ")
                for cell in row
                if cell is not None and str(cell).strip()
            ]

            if cells:
                rendered.append(" | ".join(cells))

    return "\n".join(rendered)


def build_embedding_text(chunk: dict[str, Any]) -> str:
    """
    Build the string that BOTH dense and sparse models will embed.

    Layout (all optional parts are omitted if empty):

        <breadcrumb>
        <title>
        <content>
        <steps>
        <warnings>
        <tables>

    The breadcrumb and title give the model section-level context, which
    noticeably improves retrieval for short or generic chunks. Structured
    content (steps / warnings / tables) is included so that chunks whose
    'content' is blank but that carry procedural or safety information are
    still retrievable.
    """

    parts: list[str] = []

    breadcrumb = _get_breadcrumb(chunk)
    if breadcrumb:
        parts.append(breadcrumb)

    title = str(chunk.get("title") or "").strip()
    if title and title not in breadcrumb:
        parts.append(title)

    content = str(chunk.get("content") or "").strip()
    if content:
        parts.append(content)

    steps_text = _render_steps(chunk.get("steps") or [])
    if steps_text:
        parts.append("Steps:\n" + steps_text)

    warnings_text = _render_warnings(chunk.get("warnings") or [])
    if warnings_text:
        parts.append("Warnings:\n" + warnings_text)

    tables_text = _render_tables(chunk.get("tables") or [])
    if tables_text:
        parts.append("Tables:\n" + tables_text)

    return "\n\n".join(parts).strip()


# ============================================================================
# QDRANT POINT ID
# ============================================================================

def stable_qdrant_id(chunk_id: str) -> int:
    """
    Convert a string chunk_id into a deterministic positive integer
    accepted by Qdrant.

    Re-running ingestion updates the same point rather than creating
    duplicates.
    """

    digest = hashlib.sha1(chunk_id.encode("utf-8")).hexdigest()

    value = int(digest[:16], 16)

    return value & 0x7FFFFFFFFFFFFFFF


# ============================================================================
# PAYLOAD
# ============================================================================

def build_payload(chunk: dict[str, Any]) -> dict[str, Any]:
    """
    Build the Qdrant payload from a chunk.

    Field names match all_chunks.json exactly. Nothing is silently dropped.
    """

    header = chunk.get("header") or {}

    steps = chunk.get("steps") or []
    warnings = chunk.get("warnings") or []
    tables = chunk.get("tables") or []

    content = chunk.get("content") or ""

    return {
        # ------------------------------------------------------------------
        # Identity
        # ------------------------------------------------------------------
        "chunk_id": chunk.get("id"),
        "title": chunk.get("title"),

        # ------------------------------------------------------------------
        # Hierarchy (flattened from header for filtering convenience)
        # ------------------------------------------------------------------
        "section": header.get("section"),
        "subsection": header.get("subsection"),
        "parent": header.get("parent"),
        "breadcrumb": header.get("breadcrumb"),

        # ------------------------------------------------------------------
        # Location
        # ------------------------------------------------------------------
        "page_start": chunk.get("page_start"),
        "page_end": chunk.get("page_end"),

        # ------------------------------------------------------------------
        # Raw content (kept for display + provenance)
        # ------------------------------------------------------------------
        "content": content,

        # ------------------------------------------------------------------
        # Structured content (kept for downstream GraphRAG / reranking)
        # ------------------------------------------------------------------
        "steps": steps,
        "warnings": warnings,
        "tables": tables,

        # ------------------------------------------------------------------
        # Derived boolean flags (enable cheap Qdrant filters)
        # ------------------------------------------------------------------
        "has_content": bool(str(content).strip()),
        "has_steps": bool(steps),
        "has_warnings": bool(warnings),
        "has_tables": bool(tables),
        "step_count": len(steps),
        "warning_count": len(warnings),
        "table_count": len(tables),
    }


# ============================================================================
# QDRANT CLIENT
# ============================================================================

def create_qdrant_client() -> QdrantClient:
    if QDRANT_API_KEY:
        return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    return QdrantClient(url=QDRANT_URL)


# ============================================================================
# CREATE COLLECTION
# ============================================================================

def create_collection(client: QdrantClient) -> None:

    exists = client.collection_exists(collection_name=COLLECTION_NAME)

    if exists:
        if RECREATE_COLLECTION:
            print(
                f"[qdrant] deleting existing collection '{COLLECTION_NAME}'"
            )
            client.delete_collection(collection_name=COLLECTION_NAME)
        else:
            print(
                f"[qdrant] collection '{COLLECTION_NAME}' already exists"
            )
            return

    print(f"[qdrant] creating hybrid collection '{COLLECTION_NAME}'")

    dense_config = VectorParams(
        size=DENSE_VECTOR_SIZE,
        distance=Distance.COSINE,
    )

    sparse_config = SparseVectorParams(
        index=SparseIndexParams(on_disk=False),
        modifier=Modifier.IDF,
    )

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={DENSE_VECTOR_NAME: dense_config},
        sparse_vectors_config={SPARSE_VECTOR_NAME: sparse_config},
    )

    print("[qdrant] collection created")
    print(f"  dense : {DENSE_VECTOR_NAME}")
    print(f"  sparse: {SPARSE_VECTOR_NAME}")


# ============================================================================
# LOAD DENSE MODEL
# ============================================================================

def load_dense_model() -> SentenceTransformer:

    print()
    print(f"[dense] loading {DENSE_MODEL}")

    model = SentenceTransformer(DENSE_MODEL)

    test_vector = model.encode(
        ["test"],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]

    actual_size = len(test_vector)

    if actual_size != DENSE_VECTOR_SIZE:
        raise RuntimeError(
            "Dense embedding dimension mismatch: "
            f"configured={DENSE_VECTOR_SIZE}, actual={actual_size}"
        )

    print(f"[dense] dimension: {actual_size}")

    return model


# ============================================================================
# LOAD SPARSE MODEL
# ============================================================================

def load_sparse_model() -> SparseTextEmbedding:

    print()
    print(f"[sparse] loading {SPARSE_MODEL}")

    model = SparseTextEmbedding(model_name=SPARSE_MODEL)

    test_embeddings = list(model.embed(["test"]))

    if not test_embeddings:
        raise RuntimeError("BM25 model returned no embedding.")

    test_embedding = test_embeddings[0]

    if not hasattr(test_embedding, "indices") or not hasattr(
        test_embedding, "values"
    ):
        raise RuntimeError("Unexpected FastEmbed sparse embedding format.")

    print("[sparse] BM25 model loaded")
    print(f"[sparse] test non-zero terms: {len(test_embedding.indices)}")

    return model


# ============================================================================
# UPSERT
# ============================================================================

def upsert_chunks(
    client: QdrantClient,
    dense_model: SentenceTransformer,
    sparse_model: SparseTextEmbedding,
    chunks: list[dict[str, Any]],
) -> None:

    print()
    print(f"[embedding] processing {len(chunks)} chunks")

    for start in tqdm(
        range(0, len(chunks), BATCH_SIZE),
        desc="Dense + BM25 + Qdrant",
    ):
        batch = chunks[start : start + BATCH_SIZE]

        texts = [build_embedding_text(chunk) for chunk in batch]

        # Guard: drop any chunk whose embedding text is empty.
        # (Shouldn't happen after filter_chunks, but protects the pipeline.)
        valid_indices = [i for i, t in enumerate(texts) if t.strip()]
        if len(valid_indices) != len(texts):
            batch = [batch[i] for i in valid_indices]
            texts = [texts[i] for i in valid_indices]
            if not batch:
                continue

        dense_vectors = dense_model.encode(
            texts,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        sparse_vectors = list(sparse_model.embed(texts))

        if len(sparse_vectors) != len(batch):
            raise RuntimeError(
                "Dense/sparse embedding count mismatch: "
                f"dense={len(dense_vectors)}, "
                f"sparse={len(sparse_vectors)}, "
                f"batch={len(batch)}"
            )

        points: list[PointStruct] = []

        for chunk, dense_vector, sparse_vector in zip(
            batch, dense_vectors, sparse_vectors
        ):
            chunk_id = str(chunk["id"])

            point_id = stable_qdrant_id(chunk_id)

            payload = build_payload(chunk)

            point = PointStruct(
                id=point_id,
                vector={
                    DENSE_VECTOR_NAME: dense_vector.tolist(),
                    SPARSE_VECTOR_NAME: {
                        "indices": sparse_vector.indices.tolist(),
                        "values": sparse_vector.values.tolist(),
                    },
                },
                payload=payload,
            )

            points.append(point)

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True,
        )


# ============================================================================
# VERIFY COLLECTION
# ============================================================================

def verify_collection(client: QdrantClient) -> None:

    info = client.get_collection(COLLECTION_NAME)

    print()
    print("=" * 70)
    print("QDRANT HYBRID COLLECTION")
    print("=" * 70)
    print(f"Collection : {COLLECTION_NAME}")
    print(f"Points     : {info.points_count}")

    vectors_count = getattr(
        info,
        "indexed_vectors_count",
        getattr(info, "vectors_count", info.points_count),
    )
    print(f"Vectors    : {vectors_count}")
    print(f"Status     : {info.status}")
    print()
    print("Named vectors:")
    print(f"  dense  -> {DENSE_VECTOR_SIZE} dimensions")
    print(f"  bm25   -> sparse")
    print("=" * 70)


# ============================================================================
# TEST DENSE SEARCH
# ============================================================================

def test_dense_search(
    client: QdrantClient,
    dense_model: SentenceTransformer,
) -> None:

    query = (
        "What precautions should be followed before servicing the pump?"
    )

    print()
    print("[test] dense search")
    print(f"Query: {query}")

    vector = dense_model.encode(
        [query],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector.tolist(),
        using=DENSE_VECTOR_NAME,
        limit=5,
        with_payload=True,
    ).points

    for rank, result in enumerate(results, start=1):
        payload = result.payload or {}

        print()
        print(f"{rank}. score={result.score:.4f}")
        print(f"   chunk={payload.get('chunk_id')}")
        print(f"   page={payload.get('page_start')}")
        print(f"   {payload.get('breadcrumb')}")

        text = str(payload.get("content", "")).replace("\n", " ")
        print(f"   {text[:250]}")


# ============================================================================
# TEST BM25 SEARCH
# ============================================================================

def test_bm25_search(
    client: QdrantClient,
    sparse_model: SparseTextEmbedding,
) -> None:

    query = "pressure relief valve pump"

    print()
    print("[test] BM25 sparse search")
    print(f"Query: {query}")

    sparse_embedding = list(sparse_model.query_embed([query]))[0]

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=SparseVector(
            indices=sparse_embedding.indices.tolist(),
            values=sparse_embedding.values.tolist(),
        ),
        using=SPARSE_VECTOR_NAME,
        limit=5,
        with_payload=True,
    ).points

    for rank, result in enumerate(results, start=1):
        payload = result.payload or {}

        print()
        print(f"{rank}. score={result.score:.4f}")
        print(f"   chunk={payload.get('chunk_id')}")
        print(f"   page={payload.get('page_start')}")
        print(f"   {payload.get('breadcrumb')}")

        text = str(payload.get("content", "")).replace("\n", " ")
        print(f"   {text[:250]}")


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    print()
    print("=" * 70)
    print("HYBRID QDRANT INGESTION — HL SERIES PUMP MANUAL")
    print("=" * 70)
    print(f"Chunks       : {CHUNKS_PATH}")
    print(f"Dense model  : {DENSE_MODEL}")
    print(f"Sparse model : {SPARSE_MODEL}")
    print(f"Collection   : {COLLECTION_NAME}")
    print("=" * 70)

    # ------------------------------------------------------------------------
    # Load chunks
    # ------------------------------------------------------------------------

    chunks = load_chunks(CHUNKS_PATH)
    print()
    print(f"[chunks] loaded {len(chunks)} chunks")

    # ------------------------------------------------------------------------
    # Filter empty stubs
    # ------------------------------------------------------------------------

    chunks = filter_chunks(chunks)
    print(f"[chunks] {len(chunks)} chunks will be indexed")

    if not chunks:
        raise RuntimeError(
            "No chunks remain after filtering. "
            "Check SKIP_EMPTY_CHUNKS and EMBED_STRUCTURED_ONLY_CHUNKS."
        )

    # ------------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------------

    dense_model = load_dense_model()
    sparse_model = load_sparse_model()

    # ------------------------------------------------------------------------
    # Qdrant
    # ------------------------------------------------------------------------

    print()
    print(f"[qdrant] connecting to {QDRANT_URL}")

    client = create_qdrant_client()
    client.get_collections()
    print("[qdrant] connection OK")

    # ------------------------------------------------------------------------
    # Collection
    # ------------------------------------------------------------------------

    create_collection(client)

    # ------------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------------

    upsert_chunks(
        client=client,
        dense_model=dense_model,
        sparse_model=sparse_model,
        chunks=chunks,
    )

    # ------------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------------

    verify_collection(client)

    # ------------------------------------------------------------------------
    # Retrieval tests
    # ------------------------------------------------------------------------

    test_dense_search(client, dense_model)
    test_bm25_search(client, sparse_model)

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()