# Advanced Corporate GraphRAG for Technical Manuals

A GraphRAG question-answering system for complex industrial manuals. It combines hybrid vector retrieval with a Neo4j knowledge graph and uses a LangGraph workflow to produce answers grounded in retrieved manual evidence.

## What it does

- Extracts text and tables from a PDF manual.
- Splits manual content into section-aware chunks using table-of-contents headings.
- Uses an LLM to extract warnings, cautions, procedural steps, entities, and relationships.
- Indexes chunk content in Qdrant using dense and sparse retrieval.
- Stores entities and relationships in Neo4j for graph expansion.
- Retrieves related chunks and their associated steps, warnings, and tables to construct answer context.
- Runs a conversational LangGraph agent and includes a batch evaluation harness.

## Architecture

```text
Authorized technical manual (PDF)
              |
              v
extract_pdf_text_and_tables.py
              |
       output.txt + tables.json
              |
              v
chunk_manual.py -> chunks.json
              |
              v
enrich_chunks.py -> all_chunks.json
              |
              v
attach_tables_to_chunks.py
              |
              v
extract_entities_and_relations.py
       |                         |
       v                         v
entities.json / relations.json  steps.json / warnings.json
       |                         |
       +------------+------------+
                    |
          +---------+---------+
          |                   |
          v                   v
      Neo4j graph          Qdrant index
          |                   |
          +---------+---------+
                    |
                    v
          retrieval.py + run_assistant.py
                    |
                    v
          Grounded answer via LangGraph
```

## Tech stack

- Python
- Groq API for LLM inference
- LangGraph for orchestration and conversation state
- Qdrant for dense + sparse hybrid retrieval
- Neo4j for entity and relationship traversal
- Sentence Transformers and FastEmbed for embeddings
- pdfplumber for PDF text and table extraction
- Docker Compose for local services

## Repository layout

| File | Purpose |
| --- | --- |
| `run_assistant.py` | Interactive GraphRAG assistant |
| `retrieval.py` | Hybrid retrieval, graph expansion, and context construction |
| `index_qdrant.py` | Indexes chunk data in Qdrant |
| `import_neo4j.py` | Imports entities and relationships into Neo4j |
| `extract_entities_and_relations.py` | Builds entity and relationship data |
| `extract_pdf_text_and_tables.py` | Extracts PDF text and tables |
| `chunk_manual.py` | Creates section-aware chunks |
| `enrich_chunks.py` | Extracts warnings and steps from chunks |
| `attach_tables_to_chunks.py` | Attaches extracted tables to chunks |
| `evaluation.py` | Batch evaluation and interactive evaluation workflow |
| `all_chunks.json`, `entities.json`, etc. | Structured example data used by the demo |

## Quick start with Docker Compose

### 1. Configure environment variables

Copy the example file and add your own credentials:

```bash
cp .env.example .env
```

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Edit `.env` and set a valid `GROQ_API_KEY` and a strong `NEO4J_PASSWORD`. Never commit `.env`.

### 2. Start the services

```bash
docker compose up -d qdrant neo4j
```

### 3. Build the indexes

This is a one-time step, or repeat it after changing the data:

```bash
docker compose --profile bootstrap run --rm bootstrap
```

### 4. Run the assistant

```bash
docker compose run --rm app
```

The services use persistent Docker volumes for Qdrant and Neo4j data.

## Run Python scripts locally

Use Python 3.11 or newer.

```bash
python -m venv .venv
```

Activate the environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Create `.env` from `.env.example` and start local Qdrant and Neo4j services. The scripts load environment variables from `.env`.

For the interactive assistant:

```bash
python run_assistant.py
```

For batch evaluation:

```bash
python evaluation.py --batch
```

Check `python evaluation.py --help` for supported evaluation options.

## Rebuild the data from a manual

The source PDF is intentionally not included. Place the HL Series pump manual you are authorized to process at `manuals/hl_series_iom.pdf`, then run:

**Important:** `attach_tables_to_chunks.py` currently uses a hardcoded `TABLE_MAP` tailored to this HL Series pump manual. The rebuild sequence is not fully generic for arbitrary manuals; when processing a different manual, update and validate that mapping (or replace it with automatic table-to-section association) before trusting the resulting chunks.

```bash
python extract_pdf_text_and_tables.py
python chunk_manual.py
python enrich_chunks.py
python attach_tables_to_chunks.py
python extract_entities_and_relations.py
python index_qdrant.py
python import_neo4j.py
```

LLM-based extraction requires a valid Groq API key. The indexing/import commands require Qdrant and Neo4j to be running.

## Evaluation

The evaluation harness supports batch questions and caches completed results locally to make interrupted runs resumable. Evaluation outputs and cache files are ignored by Git because they are generated artifacts.

Report retrieval and grounding metrics separately from answer correctness. A verifier's approval rate is not, by itself, a measure of factual accuracy; correctness should be assessed against human-reviewed reference answers and evidence.

## Security

- Keep API keys, database passwords, and local `.env` files out of Git.
- Rotate any credential that was previously committed publicly.
- Use a strong, unique Neo4j password.
- Local caches, generated evaluation reports, PDF manuals, and intermediate extraction output are excluded from version control.

## Data and licensing

The source manual is not bundled with this repository. Obtain manuals from their rights holder and comply with their terms before processing or redistributing content. The committed JSON files are derived from technical-manual content; verify that you have permission to redistribute them before publishing this repository for public use.
