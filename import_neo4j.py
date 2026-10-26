#!/usr/bin/env python3
"""
Import pump manual entities, relations, communities, steps and warnings into Neo4j.

Schema facts this importer assumes (matching the corrected pipeline output):

  entities.json  – every node, including STEP and WARNING nodes, has:
      canonical_name   str      (globally unique)
      name             str
      type             str      (COMPONENT, STEP, WARNING, ...)
      description      str
      source_chunk_id  str|None (first chunk it appeared in)
      mention_chunks   list[str]|absent   (only on resolved text/table entities)
      aliases          list[str]
      mention_tables   list[str]|absent
      full_text        str|absent         (STEP and WARNING only)
      level            str|absent         (WARNING only)
      marker           str|absent         (STEP only)
      step_number      int|absent         (STEP only)

  relations.json – every relation has:
      source           str    (canonical_name)
      target           str    (canonical_name)
      type             str    (PART_OF, PRECEDES_STEP, MENTIONS, WARNS_ABOUT, ...)
      description      str
      confidence       float
      source_chunk_id  str|None
      table_id         str|absent

Community construction:

  For every chunk id that appears anywhere in the dataset, we create a
  (:Community {chunk_id}) node.

  Every entity whose source_chunk_id OR mention_chunks OR the source_chunk_id
  of a MENTIONS/WARNS_ABOUT relation pointing at it equals that chunk id, is
  attached to the community via:

      (c:Community)-[:HAS_MEMBER]->(e:Entity)

  Additionally, communities get derived edges:

      (c_warn:Community)-[:HAS_WARNING]->(c_entity:Community)

  when a chunk contains WARNING entities AND regular entities. The warning
  community is a *sub-community* of the same chunk id (we namespace it as
  "<chunk_id>::warnings"), and the link goes to the main entity community of
  the same chunk.
"""

import json
import re
from collections import defaultdict
from pathlib import Path

from neo4j import GraphDatabase


# ===================================================================
# CONFIGURATION
# ===================================================================

import os
from dotenv import load_dotenv

load_dotenv()

NEO4J_URI      = os.environ.get("NEO4J_URI",      "bolt://neo4j:7687")
NEO4J_USER     = os.environ.get("NEO4J_USER",     "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")

if not NEO4J_PASSWORD:
    raise RuntimeError("NEO4J_PASSWORD environment variable is not set.")

ENTITIES_FILE = "entities.json"
RELATIONS_FILE = "relations.json"

# Types that get grouped into the warning sub-community of a chunk.
WARNING_TYPE = "WARNING"
STEP_TYPE = "STEP"

# Relation types whose `source_chunk_id` should also count as evidence that
# an entity belongs to a chunk. This lets steps/warnings that were linked via
# MENTIONS/WARNS_ABOUT pull those entities into the same community.
CHUNK_ATTRIBUTING_RELATION_TYPES = {
    "MENTIONS",
    "WARNS_ABOUT",
    "PRECEDES_STEP",
}


# ===================================================================
# HELPERS
# ===================================================================

_CHUNK_ID_SAFE = re.compile(r"[^A-Za-z0-9_.\-]+")


def safe_id(text: str) -> str:
    """Make a string safe for use inside a Neo4j property value."""
    return _CHUNK_ID_SAFE.sub("_", str(text))


def entity_chunks(entity: dict) -> set[str]:
    """
    Return every chunk id this entity should be associated with.

    Sources of truth, in order of importance:
      - source_chunk_id   (always present)
      - mention_chunks    (present after entity resolution)
    """
    chunks: set[str] = set()

    scid = entity.get("source_chunk_id")
    if scid:
        chunks.add(str(scid))

    for cid in entity.get("mention_chunks") or []:
        if cid:
            chunks.add(str(cid))

    return chunks


def relations_by_type(relations: list[dict]) -> dict[str, list[dict]]:
    by_type: dict[str, list[dict]] = defaultdict(list)
    for rel in relations:
        rtype = str(rel.get("type") or "").upper()
        if rtype:
            by_type[rtype].append(rel)
    return by_type


# ===================================================================
# NEO4J IMPORTER
# ===================================================================

class Neo4jImporter:

    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    # --------------------------------------------------------------
    # Schema / constraints
    # --------------------------------------------------------------

    def create_constraints(self):
        with self.driver.session() as session:
            session.run(
                "CREATE CONSTRAINT IF NOT EXISTS "
                "FOR (e:Entity) REQUIRE e.canonical_name IS UNIQUE"
            )
            session.run(
                "CREATE CONSTRAINT IF NOT EXISTS "
                "FOR (c:Community) REQUIRE c.chunk_id IS UNIQUE"
            )
            session.run(
                "CREATE CONSTRAINT IF NOT EXISTS "
                "FOR (t:Type) REQUIRE t.name IS UNIQUE"
            )
            session.run(
                "CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.type)"
            )
            session.run(
                "CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.name)"
            )
            session.run(
                "CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.source_chunk_id)"
            )
        print("[neo4j] constraints + indexes ready")

    # --------------------------------------------------------------
    # Entities
    # --------------------------------------------------------------

    def import_entities(self, entities: list[dict]) -> None:
        print(f"[entities] importing {len(entities)} raw entities ...")

        # Deduplicate by canonical_name — the pipeline already resolved
        # entities globally, so this should mostly be a no-op, but it also
        # protects against a user re-importing an older file.
        merged: dict[str, dict] = {}
        for ent in entities:
            cname = str(ent.get("canonical_name") or "").strip()
            if not cname:
                continue

            if cname not in merged:
                merged[cname] = dict(ent)
                continue

            existing = merged[cname]

            # Merge chunk ids
            existing_chunks = set(entity_chunks(existing))
            existing_chunks |= entity_chunks(ent)
            existing["source_chunk_id"] = (
                existing.get("source_chunk_id")
                or ent.get("source_chunk_id")
            )
            existing["mention_chunks"] = sorted(existing_chunks)

            # Merge aliases
            aliases = set(existing.get("aliases") or [])
            aliases |= set(ent.get("aliases") or [])
            existing["aliases"] = sorted(aliases)

            # Merge mention_tables
            tables = set(existing.get("mention_tables") or [])
            tables |= set(ent.get("mention_tables") or [])
            if tables:
                existing["mention_tables"] = sorted(tables)

            # Prefer longer description
            if len(str(ent.get("description") or "")) > len(
                str(existing.get("description") or "")
            ):
                existing["description"] = ent["description"]

        entities_list = list(merged.values())
        print(f"[entities] deduplicated to {len(entities_list)} nodes")

        with self.driver.session() as session:
            batch_size = 100
            for i in range(0, len(entities_list), batch_size):
                batch = entities_list[i : i + batch_size]
                session.execute_write(self._write_entities_batch, batch)

        print(f"[entities] imported {len(entities_list)} nodes")

    @staticmethod
    def _write_entities_batch(tx, batch: list[dict]) -> None:
        for ent in batch:
            cname = str(ent.get("canonical_name") or "").strip()
            if not cname:
                continue

            etype = str(ent.get("type") or "UNKNOWN").upper()
            name = str(ent.get("name") or cname)

            description = str(ent.get("description") or "")
            aliases = list(ent.get("aliases") or [])
            mention_chunks = list(ent.get("mention_chunks") or [])
            mention_tables = list(ent.get("mention_tables") or [])
            source_chunk_id = ent.get("source_chunk_id")

            # Type-specific optional fields
            full_text = ent.get("full_text")
            level = ent.get("level")
            marker = ent.get("marker")
            step_number = ent.get("step_number")

            props = {
                "canonical_name": cname,
                "name": name,
                "type": etype,
                "description": description,
                "aliases": aliases,
                "mention_chunks": mention_chunks,
                "mention_tables": mention_tables,
                "source_chunk_id": (
                    str(source_chunk_id) if source_chunk_id else None
                ),
            }
            if full_text is not None:
                props["full_text"] = full_text
            if level is not None:
                props["level"] = level
            if marker is not None:
                props["marker"] = marker
            if step_number is not None:
                props["step_number"] = step_number

            tx.run(
                """
                MERGE (e:Entity {canonical_name: $canonical_name})
                SET   e += $props
                WITH  e
                MERGE (t:Type {name: $type})
                MERGE (e)-[:IS_TYPE]->(t)
                """,
                {"canonical_name": cname, "props": props, "type": etype},
            )

    # --------------------------------------------------------------
    # Relations
    # --------------------------------------------------------------

    def import_relations(self, relations: list[dict]) -> None:
        print(f"[relations] importing {len(relations)} raw relations ...")

        # Only keep relations whose endpoints exist.
        # (We could be defensive here — the pipeline is already consistent,
        #  but it costs nothing to verify.)
        known: set[str] = set()
        for rel in relations:
            s = rel.get("source")
            t = rel.get("target")
            if s:
                known.add(str(s))
            if t:
                known.add(str(t))

        # Merge duplicates (source, target, type)
        merged: dict[tuple[str, str, str], dict] = {}
        for rel in relations:
            src = str(rel.get("source") or "").strip()
            tgt = str(rel.get("target") or "").strip()
            rtype = str(rel.get("type") or "").upper().strip()

            if not src or not tgt or not rtype:
                continue

            key = (src, tgt, rtype)

            if key not in merged:
                merged[key] = {
                    "source": src,
                    "target": tgt,
                    "type": rtype,
                    "description": str(rel.get("description") or ""),
                    "confidence": float(rel.get("confidence") or 0.8),
                    "source_chunk_ids": (
                        [str(rel["source_chunk_id"])]
                        if rel.get("source_chunk_id")
                        else []
                    ),
                    "table_ids": (
                        [str(rel["table_id"])]
                        if rel.get("table_id")
                        else []
                    ),
                }
                continue

            existing = merged[key]

            if rel.get("source_chunk_id"):
                scid = str(rel["source_chunk_id"])
                if scid not in existing["source_chunk_ids"]:
                    existing["source_chunk_ids"].append(scid)

            if rel.get("table_id"):
                tid = str(rel["table_id"])
                if tid not in existing["table_ids"]:
                    existing["table_ids"].append(tid)

            try:
                conf = float(rel.get("confidence") or 0.0)
                if conf > existing["confidence"]:
                    existing["confidence"] = conf
            except (TypeError, ValueError):
                pass

            desc = str(rel.get("description") or "")
            if len(desc) > len(existing["description"]):
                existing["description"] = desc

        rels_list = list(merged.values())
        print(f"[relations] deduplicated to {len(rels_list)} edges")

        with self.driver.session() as session:
            batch_size = 200
            for i in range(0, len(rels_list), batch_size):
                batch = rels_list[i : i + batch_size]
                session.execute_write(self._write_relations_batch, batch)

        print(f"[relations] imported {len(rels_list)} edges")

    @staticmethod
    def _write_relations_batch(tx, batch: list[dict]) -> None:
        for rel in batch:
            # Use the relation type as a real Neo4j relationship type so
            # Cypher traversal is natural: (:Entity)-[:MENTIONS]->(:Entity)
            #
            # Neo4j relationship types cannot be parameterized, so we
            # interpolate after sanitizing to [A-Z_].
            safe_type = re.sub(r"[^A-Z0-9_]+", "_", rel["type"])

            query = f"""
                MATCH (s:Entity {{canonical_name: $source}})
                MATCH (t:Entity {{canonical_name: $target}})
                MERGE (s)-[r:{safe_type}]->(t)
                SET   r.description      = $description,
                      r.confidence       = $confidence,
                      r.source_chunk_ids = $source_chunk_ids,
                      r.table_ids        = $table_ids
            """

            tx.run(
                query,
                {
                    "source": rel["source"],
                    "target": rel["target"],
                    "description": rel["description"],
                    "confidence": rel["confidence"],
                    "source_chunk_ids": rel["source_chunk_ids"],
                    "table_ids": rel["table_ids"],
                },
            )

    # --------------------------------------------------------------
    # Communities
    # --------------------------------------------------------------

    def import_communities(
        self,
        entities: list[dict],
        relations: list[dict],
    ) -> None:
        """
        Build chunk-based communities:

          (:Community {chunk_id})-[:HAS_MEMBER]->(:Entity)

        A chunk can have two kinds of communities:

          - Main community: contains every entity attributed to the chunk
            (text entities, tables, steps, warnings).

          - Warning community: a sub-community that contains only the
            WARNING entities of the chunk. It is linked to the main
            community via (:Community)-[:HAS_WARNING]->(:Community).

        Additionally:
          - Each step that mentions an entity via MENTIONS is guaranteed to
            be in the same community as that entity.
          - Each warning that warns about an entity via WARNS_ABOUT is
            guaranteed to be in the same community as that entity.
        """

        print("[communities] computing chunk -> entity mapping ...")

        # chunk_id -> set(entity canonical_name)
        chunk_to_entities: dict[str, set[str]] = defaultdict(set)

        # Build a lookup from canonical_name -> entity dict
        entity_lookup: dict[str, dict] = {
            str(e["canonical_name"]): e for e in entities if e.get("canonical_name")
        }

        # --- 1. Direct attribution via source_chunk_id / mention_chunks ---
        for ent in entities:
            cname = ent.get("canonical_name")
            if not cname:
                continue
            for cid in entity_chunks(ent):
                chunk_to_entities[cid].add(cname)

        # --- 2. Attribution via relations that carry a source_chunk_id ---
        #     MENTIONS / WARNS_ABOUT / PRECEDES_STEP are chunk-local,
        #     so both endpoints belong to that chunk.
        for rel in relations:
            rtype = str(rel.get("type") or "").upper()
            if rtype not in CHUNK_ATTRIBUTING_RELATION_TYPES:
                continue

            cid = rel.get("source_chunk_id")
            if not cid:
                continue
            cid = str(cid)

            src = str(rel.get("source") or "")
            tgt = str(rel.get("target") or "")

            if src in entity_lookup:
                chunk_to_entities[cid].add(src)
            if tgt in entity_lookup:
                chunk_to_entities[cid].add(tgt)

        # --- 3. Warning sub-communities ---
        chunk_to_warnings: dict[str, set[str]] = defaultdict(set)
        for cid, members in chunk_to_entities.items():
            for cname in members:
                ent = entity_lookup.get(cname)
                if ent and str(ent.get("type") or "").upper() == WARNING_TYPE:
                    chunk_to_warnings[cid].add(cname)

        print(
            f"[communities] {len(chunk_to_entities)} chunks, "
            f"{sum(len(v) for v in chunk_to_entities.values())} member links, "
            f"{sum(len(v) for v in chunk_to_warnings.values())} warnings"
        )

        with self.driver.session() as session:
            # Create main communities + membership
            session.execute_write(
                self._write_main_communities, chunk_to_entities
            )
            # Create warning sub-communities + HAS_WARNING edges
            session.execute_write(
                self._write_warning_communities,
                chunk_to_warnings,
                chunk_to_entities,
            )

        print("[communities] import complete")

    @staticmethod
    def _write_main_communities(tx, chunk_to_entities: dict[str, set[str]]) -> None:
        for cid, members in chunk_to_entities.items():
            if not members:
                continue

            tx.run(
                """
                MERGE (c:Community {chunk_id: $chunk_id})
                SET   c.kind = 'chunk'
                WITH  c
                UNWIND $members AS cname
                MATCH (e:Entity {canonical_name: cname})
                MERGE (c)-[:HAS_MEMBER]->(e)
                """,
                {"chunk_id": cid, "members": sorted(members)},
            )

    @staticmethod
    def _write_warning_communities(
        tx,
        chunk_to_warnings: dict[str, set[str]],
        chunk_to_entities: dict[str, set[str]],
    ) -> None:
        for cid, warnings in chunk_to_warnings.items():
            if not warnings:
                continue

            warning_community_id = f"{cid}::warnings"

            # Create the warning sub-community and its HAS_MEMBER edges
            tx.run(
                """
                MERGE (cw:Community {chunk_id: $cw_id})
                SET   cw.kind = 'chunk_warnings',
                      cw.parent_chunk_id = $parent_chunk_id
                WITH  cw
                UNWIND $members AS cname
                MATCH (e:Entity {canonical_name: cname})
                MERGE (cw)-[:HAS_MEMBER]->(e)
                """,
                {
                    "cw_id": warning_community_id,
                    "parent_chunk_id": cid,
                    "members": sorted(warnings),
                },
            )

            # Link the warning sub-community to the main community of the
            # same chunk via HAS_WARNING.
            tx.run(
                """
                MATCH (cw:Community {chunk_id: $cw_id})
                MATCH (c :Community {chunk_id: $parent_chunk_id})
                MERGE (c)-[:HAS_WARNING]->(cw)
                """,
                {
                    "cw_id": warning_community_id,
                    "parent_chunk_id": cid,
                },
            )

    # --------------------------------------------------------------
    # Optional: explicit step ↔ community membership
    # --------------------------------------------------------------
    #
    # Steps already participate as entities (type STEP) and are attributed
    # to chunks via source_chunk_id, so they end up as members of their
    # chunk's community automatically. This method exists only if you want
    # to expose additional, explicit (:Community)-[:HAS_STEP]->(:Step)
    # edges for query convenience.

    def import_step_membership(
        self, entities: list[dict]
    ) -> None:
        steps_by_chunk: dict[str, list[str]] = defaultdict(list)
        for ent in entities:
            if str(ent.get("type") or "").upper() != STEP_TYPE:
                continue
            cname = ent.get("canonical_name")
            if not cname:
                continue
            for cid in entity_chunks(ent):
                steps_by_chunk[str(cid)].append(cname)

        if not steps_by_chunk:
            return

        print(
            f"[steps] linking {sum(len(v) for v in steps_by_chunk.values())} "
            f"step memberships"
        )

        with self.driver.session() as session:
            for cid, members in steps_by_chunk.items():
                session.run(
                    """
                    MATCH (c:Community {chunk_id: $chunk_id})
                    UNWIND $members AS cname
                    MATCH (e:Entity {canonical_name: cname})
                    MERGE (c)-[:HAS_STEP]->(e)
                    """,
                    {"chunk_id": cid, "members": sorted(set(members))},
                )

    # --------------------------------------------------------------
    # Stats
    # --------------------------------------------------------------

    def get_stats(self) -> None:
        with self.driver.session() as session:
            entity_count = session.run(
                "MATCH (e:Entity) RETURN count(e) AS c"
            ).single()["c"]

            rel_count = session.run(
                "MATCH ()-[r]->() WHERE type(r) <> 'IS_TYPE' "
                "AND type(r) <> 'HAS_MEMBER' RETURN count(r) AS c"
            ).single()["c"]

            community_count = session.run(
                "MATCH (c:Community) RETURN count(c) AS c"
            ).single()["c"]

            warning_community_count = session.run(
                "MATCH (c:Community {kind: 'chunk_warnings'}) "
                "RETURN count(c) AS c"
            ).single()["c"]

            has_warning_count = session.run(
                "MATCH (:Community)-[r:HAS_WARNING]->(:Community) "
                "RETURN count(r) AS c"
            ).single()["c"]

            type_count = session.run(
                "MATCH (t:Type) RETURN count(t) AS c"
            ).single()["c"]

            print("\nGraph statistics")
            print(f"  Entities                 : {entity_count}")
            print(f"  Entity types             : {type_count}")
            print(f"  Relations (non-structural): {rel_count}")
            print(f"  Communities              : {community_count}")
            print(f"  Warning sub-communities  : {warning_community_count}")
            print(f"  HAS_WARNING edges        : {has_warning_count}")


# ===================================================================
# MAIN
# ===================================================================

def main() -> None:
    print("=" * 64)
    print("HL SERIES PUMP MANUAL — NEO4J GRAPH IMPORTER")
    print("=" * 64)

    # Sanity: files exist
    for path in (ENTITIES_FILE, RELATIONS_FILE):
        if not Path(path).exists():
            raise FileNotFoundError(f"Missing input file: {path}")

    with open(ENTITIES_FILE, "r", encoding="utf-8") as f:
        entities = json.load(f)
    with open(RELATIONS_FILE, "r", encoding="utf-8") as f:
        relations = json.load(f)

    print(
        f"[input] {len(entities)} entities, "
        f"{len(relations)} relations"
    )

    importer = Neo4jImporter(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    try:
        importer.create_constraints()
        importer.import_entities(entities)
        importer.import_relations(relations)
        importer.import_communities(entities, relations)
        importer.import_step_membership(entities)
        importer.get_stats()
        print("\nGraph construction complete.")
    finally:
        importer.close()


if __name__ == "__main__":
    main()