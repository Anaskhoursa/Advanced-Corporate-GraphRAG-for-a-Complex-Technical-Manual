#!/usr/bin/env python3
"""
Attach tables from tables.json to the correct chunks in all_chunks.json.

Uses a STATIC mapping (hand-verified against the source PDF) instead of
the buggy heuristic that misassigned tables.

Mapping format:
  "page_N": [chunk_id_or_None, chunk_id_or_None, ...]
  - The list index corresponds to the ordinal position of the table on
    that page (as produced by pdfplumber.extract_tables()).
  - None means the table is empty or irrelevant and should be dropped.

Input:
  all_chunks.json   (chunks without tables, or with a broken `tables` field)
  tables.json       (raw tables keyed by page_N)

Output:
  all_chunks.json   (rewritten with correct `tables` field)
  tables_mapped.json (for inspection: page -> list of {chunk_id, table})
"""

import json
from pathlib import Path


CHUNKS_FILE = "all_chunks.json"
TABLES_FILE = "tables.json"
OUT_CHUNKS = "all_chunks.json"
OUT_MAPPED = "tables_mapped.json"


# ---------------------------------------------------------------------------
# Static mapping: page -> [chunk_id per table ordinal]
# None means drop the table (empty or irrelevant).
# ---------------------------------------------------------------------------
TABLE_MAP = {
    "page_6":  ["1.2"],
    "page_7":  ["1.2"],
    "page_11": [None],                                  # empty
    "page_14": ["3.1"],
    "page_15": ["3.2", "3.2", "3.2"],                   # liquid, pressures, flanges
    "page_18": ["4.1.1"],                               # pump location guidelines
    "page_20": ["4.2.1"],                               # general piping checklist
    "page_21": [None, "4.2.2"],                         # empty, suction-piping
    "page_22": ["4.2.2", "4.2.2", "4.2.3"],             # below/above pump, discharge
    "page_23": ["4.2.3", "4.2.4", None, None],          # secondary valve, final, empty, empty
    "page_25": [None],                                  # empty
    "page_26": [None, None],                            # empty
    "page_27": [None, None],                            # empty
    "page_28": ["4.4.1", "4.4.1", "4.4.2", "4.4.2", "4.4.2"],
    "page_29": ["4.4.4"],                               # reading values
    "page_30": ["4.4.4"],                               # condition/action
    "page_32": ["5.1.2"],                               # compressor/vacuum disconnection
    "page_44": ["7.2"],                                 # Table 1
    "page_45": ["7.2", "7.2"],                          # Tables 2, 3
    "page_46": ["7.2", "7.2"],                          # Tables 4, 5
    "page_47": ["7.2", "7.2"],                          # continuation
    "page_48": ["7.2", "7.3", "7.3", "7.4"],            # Table 5b, 6, 7, 8
    "page_49": ["7.4", "7.4", "7.4", "7.4"],            # Tables 9, 10, 11, 12
    "page_50": ["7.4", "7.4", "7.4", "7.4"],            # Table 12b, 13, 14, 15
    "page_51": ["8.1"],                                 # designations
    "page_52": ["8.2", "8.2", "8.2"],                   # compressor oil, coolant, vacuum oil
    "page_53": ["8.2", "8.3"],                          # vacuum oil (above 0), metric torque
    "page_54": ["8.3", "8.3"],                          # UNC torque, plating factors
    "page_55": ["8.4.3", "8.5"],                        # belt tension, impeller clearance
    "page_56": ["8.5"],                                 # impeller clearance (cont.)
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_empty(table):
    """True if every cell in every row is None or blank."""
    for row in table or []:
        for cell in row or []:
            if cell is not None and str(cell).strip() != "":
                return False
    return True


def load():
    chunks = json.loads(Path(CHUNKS_FILE).read_text(encoding="utf-8"))
    tables = json.loads(Path(TABLES_FILE).read_text(encoding="utf-8"))
    return chunks, tables


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    chunks, tables = load()
    chunks_by_id = {c["id"]: c for c in chunks}

    # Strip any pre-existing broken `tables` field from every chunk
    for c in chunks:
        c.pop("tables", None)

    # ---- Map every table to its chunk ----
    mapped = {}          # page -> [{chunk_id, table}]
    per_chunk = {}       # chunk_id -> [table_entry, ...]
    dropped = 0
    orphans = []

    for page_key, page_tables in tables.items():
        assignment = TABLE_MAP.get(page_key)
        if assignment is None:
            orphans.append(page_key)
            continue

        if len(assignment) != len(page_tables):
            print(
                f"WARN: {page_key}: mapping has {len(assignment)} slots "
                f"but tables.json has {len(page_tables)} tables. "
                f"Skipping this page."
            )
            orphans.append(page_key)
            continue

        mapped[page_key] = []
        for idx, (chunk_id, table) in enumerate(zip(assignment, page_tables)):
            if chunk_id is None or is_empty(table):
                dropped += 1
                mapped[page_key].append({"chunk_id": None, "table": table})
                continue

            if chunk_id not in chunks_by_id:
                print(f"WARN: {page_key}[{idx}] -> unknown chunk '{chunk_id}'")
                dropped += 1
                continue

            entry = {"page": page_key, "ordinal": idx, "table": table}
            per_chunk.setdefault(chunk_id, []).append(entry)
            mapped[page_key].append({"chunk_id": chunk_id, "table": table})

    # ---- Attach tables to their chunks ----
    for c in chunks:
        c["tables"] = per_chunk.get(c["id"], [])

    # ---- Write outputs ----
    Path(OUT_CHUNKS).write_text(
        json.dumps(chunks, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    Path(OUT_MAPPED).write_text(
        json.dumps(mapped, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # ---- Summary ----
    print(f"\nTables processed: {sum(len(v) for v in mapped.values())}")
    print(f"Tables dropped:   {dropped}")
    if orphans:
        print(f"Unmapped pages:   {orphans}")
    print()
    print("Chunks with tables:")
    for cid in sorted(per_chunk.keys(), key=_chunk_sort_key):
        print(f"  {cid:8s} -> {len(per_chunk[cid])} table(s)")
    print()
    print(f"Wrote {OUT_CHUNKS} and {OUT_MAPPED}")


def _chunk_sort_key(cid):
    """Sort '1.2' before '3.10' before '4' before '4.1'."""
    parts = []
    for p in cid.split("."):
        try:
            parts.append((0, int(p)))
        except ValueError:
            parts.append((1, p))
    return parts


if __name__ == "__main__":
    main()