# Skein — Scientific Context Engine

A context engineering system for scientific material. Skein ingests documents
(PDFs, handwritten notes, web articles), extracts scientific concepts,
equations, and entities, links them into a knowledge graph, and answers
questions grounded in the user's own sources — with citations back to where
each answer came from.

## Status

**All 7 phases are done** — ingestion through evaluation, end-to-end. See
the Roadmap and API Reference below for what each phase shipped, and
[documentations/design_choices.md](documentations/design_choices.md) for
the reasoning behind the current architecture.

## Roadmap

| Phase | Description | Status |
|---|---|---|
| 1 | Source registry & identity (upload, dedup, retrieve) | ✅ Done |
| 2 | Multi-format ingestion (handwriting OCR, web articles) & versioning | ✅ Done |
| 3 | Content understanding — concept/equation/entity extraction | ✅ Done |
| 4 | Knowledge graph — linking concepts (prerequisite, related, derivation, application) | ✅ Done |
| 5 | Context engineering & retrieval | ✅ Done |
| 6 | Chat interface, grounded in sources with citations | ✅ Done |
| 7 | Evaluation — groundedness and retrieval quality checks | ✅ Done |

## Tech Stack

- **API**: FastAPI + Uvicorn
- **Database**: SQLAlchemy ORM (SQLite by default, configurable via `DATABASE_URL`)
- **Validation/config**: Pydantic / pydantic-settings
- **OCR**: Tesseract (via `pytesseract`) for handwritten/image sources
- **PDF text extraction**: `pypdfium2`
- **Web ingestion**: `requests` + `trafilatura` for fetching and extracting article text
- **Concept extraction**: local LLM via [Ollama](https://ollama.com) (default model: Qwen2.5-7B-Instruct), behind a provider-agnostic `LLMClient` interface
- **Knowledge graph linking**: local embedding model (nomic-embed-text) for similarity prefiltering + the same local LLM for relationship classification
- **Retrieval & context assembly**: embedding search (nomic-embed-text, query/document-prefixed) over chunk text, enriched with already-extracted concepts
- **Chat**: streaming, citation-grounded Q&A over the assembled context, using the same local LLM as extraction/linking
- **Evaluation**: structural citation checks, LLM-judged groundedness, and batch retrieval-quality metrics (hit rate, MRR)
- **Testing**: pytest

## Project Structure

```
src/
  api/routes/          FastAPI route handlers
  core/                Settings and logging configuration
  database/            Connection/session management, ORM models, repositories
  ingestion/            Pipelines that turn raw uploads into registered sources
  processing/           Chunking and LLM-based concept/equation/entity extraction
  llm/                   Provider-agnostic LLM client interface (Ollama by default)
  linking/                 Embedding-prefiltered, LLM-classified concept linking
  retrieval/                 Embedding-based search over chunk text
  context_engineering/         Assembles citable context blocks for Q&A
  chat/                           Grounded, streaming question-answering
  evaluation/                        Citation, groundedness, and retrieval-quality checks
frontend/              graph.html — live knowledge graph viewer, served at GET /graph
data/                  Local storage: raw sources, processed output, metadata DB
```

## Getting Started

### Prerequisites

- Python 3.11+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) engine, for
  handwritten/image source ingestion:
  ```bash
  brew install tesseract   # macOS
  ```
- [Ollama](https://ollama.com), for local LLM-based concept extraction:
  ```bash
  brew install ollama                  # macOS
  brew services start ollama
  ollama pull qwen2.5:7b-instruct      # default extraction model, ~4.7GB
  ```

### Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run

```bash
uvicorn src.main:app --reload
```

The API is available at `http://127.0.0.1:8000`, with interactive docs at
`http://127.0.0.1:8000/docs`.

## API Reference

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/sources` | Upload a source (PDF, or PNG/JPEG for handwritten notes — OCR runs automatically). Returns the registered source; `duplicate: true` if the same content was already registered, now or previously. |
| `POST` | `/api/sources/web` | Register or re-check a web article by URL (JSON body: `{"url": "..."}`). Fetches the page, extracts clean article text, and de-dupes by content — re-posting a known URL is how you check it for updates. |
| `GET` | `/api/sources` | List all registered sources, newest first. |
| `GET` | `/api/sources/{source_id}` | Fetch a single source by its public ID (current version). |
| `POST` | `/api/sources/{source_id}/versions` | Upload a new version of an existing PDF/image source. Must match the source's original file type. Archives the previous version and bumps `version`. |
| `GET` | `/api/sources/{source_id}/versions` | List the archived (superseded) versions of a source. |
| `POST` | `/api/sources/{source_id}/extract` | Chunk the source's extracted text and run local-LLM extraction to produce concepts, equations, and entities. Cached — repeat calls are a no-op unless the source changed or `?force=true` is passed. |
| `GET` | `/api/sources/{source_id}/concepts` | List all concepts/equations/entities extracted from a source. |
| `POST` | `/api/knowledge/link` | Link concepts across all sources into the knowledge graph (prerequisite/related/derivation/application). Cached — only new or previously-unlinked concept pairs are processed unless `?force=true` is passed. |
| `GET` | `/api/knowledge/graph` | Fetch the full knowledge graph: every concept as a node, every link as an edge. |
| `GET` | `/api/knowledge/concepts/{concept_id}/links` | List the links involving a single concept. |
| `GET` | `/api/retrieval/search` | Embedding search over chunk text for a query (`?q=...`, optional `?top_k=...`). Returns scored matches with source/chunk citations, no concept enrichment. |
| `GET` | `/api/context` | The retrieval result for `?q=...`, assembled into a structured context block — each matched chunk annotated with the concepts already extracted from it. The hand-off point for the chat endpoint below. |
| `POST` | `/api/chat` | Ask a question grounded in the user's sources. Body: `{"question": "...", "history": [{"role": "user"\|"assistant", "content": "..."}], "top_k": 5}` (`history`/`top_k` optional — history is caller-supplied, not persisted). Streams the answer as plain text; citations (source/chunk/similarity) are returned in the `X-Citations` response header as JSON. |
| `POST` | `/api/evaluation/chat` | Runs a question through the same chat logic as `/api/chat` (non-streamed), then checks the answer: a structural citation check (do `[1]`/`[2]` markers refer to real context items) and an LLM-judged groundedness check (is every claim actually supported by the context). Same body shape as `/api/chat`. |
| `POST` | `/api/evaluation/retrieval` | Batch-evaluates retrieval quality. Body: `{"cases": [{"query": "...", "expected_source_id": "...", "expected_chunk_index": 0}], "top_k": 20}` (`expected_chunk_index`/`top_k` optional). Returns hit rate, mean reciprocal rank, and per-case rank/results. |
| `GET` | `/graph` | A live, interactive visualization of the knowledge graph (not part of `/api` — a static page served directly, fetching from `/api/knowledge/graph` client-side). Filter by kind/relationship/source, search, and inspect a concept's connections. |
