#!/usr/bin/env python3
"""
Split a pdfplumber text dump into chunks using TOC headers as boundaries.

Rule
----
- Walk the body text.
- When you see a line matching a TOC header:
    - If it's the SAME as the currently open chunk's header -> skip (running header).
    - If it's DIFFERENT -> close the current chunk, open a new one.

This kills the duplicate-chunk problem caused by running page headers, because
a running header repeats the same "N Title" string; only a *new* header string
triggers a boundary.
"""

import json
import re
from pathlib import Path

SRC = "output.txt"
OUT = "chunks.json"

TOC_LINE_RE = re.compile(r"^(\d+(?:\.\d+)*)\s+(.*?)\.{3,}\s*(\d+)\s*$")
PAGE_MARKER_RE = re.compile(r"^--- PAGE (\d+) ---\s*$")
FOOTER_RE = re.compile(
    r"^(HL series Installation, Operation, and Maintenance Manual\s*\d*|"
    r"\d+\s*HL series Installation, Operation, and Maintenance Manual)\s*$"
)
def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def load_lines():
    with open(SRC, encoding="utf-8") as f:
        return f.read().split("\n")


# ---------------------------------------------------------------------------
# 1. Parse TOC
# ---------------------------------------------------------------------------
def parse_toc(lines):
    entries = []
    for line in lines:
        m = TOC_LINE_RE.match(line.rstrip("\r"))
        if m:
            entries.append({
                "number": m.group(1),
                "title": m.group(2).strip(),
                "page": int(m.group(3)),
            })
    seen, ordered = set(), []
    for e in entries:
        if e["number"] in seen:
            continue
        seen.add(e["number"])
        ordered.append(e)
    return ordered


def build_header_index(toc):
    """normalized '<number> <title>' -> TOC entry"""
    return {normalize(f"{e['number']} {e['title']}"): e for e in toc}


# ---------------------------------------------------------------------------
# 2. Split the body: only split when header CHANGES
# ---------------------------------------------------------------------------
def split_by_headers(lines, header_index):
    chunks = []
    current = {"number": None, "title": "Front Matter", "raw_lines": []}
    chunks.append(current)

    for raw in lines:
        line = raw.rstrip("\r")
        stripped = line.strip()

        # skip page markers and footers, they aren't content
        if PAGE_MARKER_RE.match(stripped):
            continue
        if FOOTER_RE.match(stripped):
            continue

        entry = header_index.get(normalize(stripped))
        if entry is not None:
            # FIX: only split if this header is DIFFERENT from the current one.
            def _is_running_header(current, candidate):
                if current is None or candidate is None:
                    return False
                return current == candidate or current.startswith(candidate + ".")

            if _is_running_header(current["number"], entry["number"]):
                continue

            # new section
            current = {
                "number": entry["number"],
                "title": entry["title"],
                "raw_lines": [],
            }
            chunks.append(current)
            continue

        current["raw_lines"].append(line)

    return chunks


# ---------------------------------------------------------------------------
# 3. Assign page ranges from the TOC
# ---------------------------------------------------------------------------
def assign_pages(chunks, toc):
    toc_by_number = {e["number"]: e for e in toc}
    ordered_numbers = [e["number"] for e in toc]

    next_number = {}
    for i, n in enumerate(ordered_numbers):
        next_number[n] = ordered_numbers[i + 1] if i + 1 < len(ordered_numbers) else None

    last_page = max(e["page"] for e in toc) if toc else 1

    for c in chunks:
        num = c.get("number")
        if num is None:
            first_page = toc[0]["page"] if toc else 1
            c["page_start"] = 1
            c["page_end"] = max(1, first_page - 1)
            continue

        start = toc_by_number[num]["page"]
        nxt = next_number[num]
        if nxt is None:
            end = last_page
        else:
            end = toc_by_number[nxt]["page"] - 1
            if end < start:
                end = start

        c["page_start"] = start
        c["page_end"] = end


# ---------------------------------------------------------------------------
# 4. Header struct + breadcrumb
# ---------------------------------------------------------------------------
def build_breadcrumb(toc_by_number, number):
    parts = number.split(".")
    chain = []
    for k in range(1, len(parts) + 1):
        sub = ".".join(parts[:k])
        if sub in toc_by_number:
            chain.append(f"{sub} {toc_by_number[sub]['title']}")
    return " -> ".join(chain)


def header_struct(toc_by_number, number):
    if number is None:
        return {"section": None, "subsection": None, "parent": None,
                "breadcrumb": "Front Matter"}
    parts = number.split(".")
    return {
        "section": parts[0],
        "subsection": number if len(parts) > 1 else None,
        "parent": ".".join(parts[:-1]) if len(parts) > 1 else None,
        "breadcrumb": build_breadcrumb(toc_by_number, number),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    lines = load_lines()
    toc = parse_toc(lines)
    if not toc:
        print("No TOC entries found.")
        return

    header_index = build_header_index(toc)
    print(f"Parsed {len(toc)} TOC entries.")

    raw_chunks = split_by_headers(lines, header_index)
    print(f"Split into {len(raw_chunks)} chunks (including front matter).")

    assign_pages(raw_chunks, toc)

    toc_by_number = {e["number"]: e for e in toc}
    final = []
    for c in raw_chunks:
        content = "\n".join(l for l in c["raw_lines"] if l.strip())
        if not content and c["number"] is None:
            continue
        final.append({
            "id": c["number"] if c["number"] is not None else "0",
            "title": c["title"],
            "header": header_struct(toc_by_number, c["number"]),
            "page_start": c["page_start"],
            "page_end": c["page_end"],
            "content": content,
        })

    Path(OUT).write_text(
        json.dumps(final, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    total = sum(len(c["content"]) for c in final)
    print(f"Wrote {len(final)} chunks to {OUT}")
    print(f"total content chars = {total}")


if __name__ == "__main__":
    main()