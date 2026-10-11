# Design Choices

This document records the architectural decisions behind Skein and the
reasoning for each, so future changes can be evaluated against the original
intent rather than guessed at. It is updated as each phase lands.

## Architecture Overview

The codebase is layered, and each layer only talks to the one directly below
it:

```
API routes (FastAPI)
      │  thin — HTTP concerns only (status codes, request/response shape)
      ▼
Service / pipeline layer (e.g. IngestionPipeline)
      │  owns the use case logic
      ▼
Repository layer (e.g. SourceRepository)
      │  owns all data access — no raw queries outside this layer
      ▼
ORM models (SQLAlchemy) + filesystem storage
```

Business logic is expressed as classes (`Database`, `IngestionPipeline`,
`ConceptExtractor`, `SourceRepository`, `LoggingConfigurator`, `Settings`)
rather than free functions and module-level state. Route handlers remain
plain FastAPI functions — that's the framework's idiomatic
dependency-injection entry point — but they delegate immediately to a class
and contain no business logic themselves.

## Decisions

### FastAPI + SQLAlchemy

**Decision**: FastAPI for the API layer, SQLAlchemy (ORM) for persistence,
Pydantic/pydantic-settings for validation and configuration.

**Rationale**: FastAPI's dependency-injection model maps cleanly onto the
layered architecture above (a request-scoped DB session is just another
dependency). SQLAlchemy keeps persistence swappable — development uses
SQLite, and moving to Postgres later is a one-line `DATABASE_URL` change with
no code changes elsewhere.

### SQLite for development, swappable via configuration

**Decision**: Default `database_url` is a local SQLite file
(`sqlite:///./data/metadata/context_engine.db`), overridable via environment
variable / `.env`.

**Rationale**: Zero setup for local development. `Database` (in
`src/database/connection.py`) only special-cases SQLite for the
`check_same_thread` connection argument; nothing else in the codebase assumes
SQLite, so switching the backend later doesn't require touching the
repository or service layers.

### Public `source_id` decoupled from the database primary key

**Decision**: `Source` has both an internal integer `id` (auto-increment
primary key, never exposed) and a `source_id` (UUID, unique, used in the API
and in file paths).

**Rationale**: The UUID is the identity that matters to callers and to
provenance tracking — it stays stable across database rebuilds/migrations
and doesn't leak information about insertion order or row count. The integer
PK exists purely for efficient indexing/joins internal to the database.

### Content-hash based deduplication

**Decision**: Every source stores a SHA-256 hash of its raw bytes
(`content_hash`, unique). Ingestion checks this hash before creating a new
record; an exact match short-circuits with the existing `Source` and a
`duplicate` flag, writing nothing new to disk or the database.

**Rationale**: This is the mechanism behind "remember what it has already
seen" — re-uploading the same file is a cheap no-op instead of a duplicate
row and a duplicate file on disk. Hashing raw bytes (rather than, say,
filename) means the check is correct even if the file is renamed.

### Repository pattern for data access

**Decision**: All queries and writes for a given model go through a single
repository class (e.g. `SourceRepository`). No other module issues raw
SQLAlchemy queries against `Source`.

**Rationale**: Keeps persistence details (query shape, session handling) out
of route handlers and service classes, and gives future phases (which will
add more models — concepts, links, chunks) a consistent, testable pattern to
follow.

### Class-based service layer

**Decision**: Cross-cutting concerns that were previously module-level
functions/globals — database connection setup, session management, ingestion
logic, logging configuration — are now encapsulated in classes (`Database`,
`IngestionPipeline`, `LoggingConfigurator`) with a single instantiated
object where module-level singletons are appropriate (e.g. `database`).

**Rationale**: Groups related state and behavior together (e.g. the engine,
session factory, and session lifecycle all live on `Database`), and makes
the service layer easier to extend or substitute (e.g. a test could
construct a `Database` pointed at an in-memory SQLite instance) without
relying on module-level side effects.

### Versioning field on `Source`

**Decision**: `Source.version` exists (defaulting to `1`) even though phase 1
never changes it.

**Rationale**: Reserved for phase 2, where re-ingesting an updated version of
a known document needs to be distinguished from an unrelated new source.
Adding the column now avoids a migration later.

### Versioning as archive-on-supersede, not an append-only log

**Decision**: A `Source` row always holds the *current* version's data
(`content_hash`, `file_path`, `original_filename`, `version`,
`extracted_text`). A separate `SourceVersion` row is written only at the
moment a version is superseded, as a snapshot of what the `Source` row
looked like immediately before being overwritten.

**Rationale**: Every other part of the system (dedup checks, retrieval,
eventually the knowledge graph) wants to ask "what does this source look
like *now*" far more often than "what did it look like before" — keeping the
current state on the primary row means that's a direct read with no extra
join. History exists (satisfying "remember where knowledge came from" for
superseded content) without making the common case pay for it. The
alternative — treating every upload as a new row in one versions table with
no distinguished "current" row — would require a `MAX(version)` or
`is_current` query on every read.

**Alternatives considered**: A single append-only `SourceVersion` table with
no separate `Source` "current" row was rejected for the reason above. Storing
full version history as a JSON blob on `Source` was rejected because it
doesn't support efficient hash lookups across historical content.

### Deduplication scoped across current AND historical content

**Decision**: `SourceRepository.get_by_hash_anywhere()` checks both
`Source.content_hash` (current versions) and `SourceVersion.content_hash`
(archived versions) before accepting any new upload — as a brand-new source
or as a new version of an existing one.

**Rationale**: Without this, re-uploading bytes that were once (but are no
longer) a source's current version would be treated as genuinely new
content, silently duplicating storage. Scoping the check across both tables
keeps "remember what it has already seen" true for the full lifetime of a
source, not just its latest snapshot.

### File storage: type-scoped subfolders, version-suffixed filenames

**Decision**: Raw files live at
`data/sources/{pdf|handwriting|web}/{source_id}_v{n}{ext}`. Every version of
a source gets its own file; nothing is ever overwritten or deleted on disk.

**Rationale**: The version-suffixed filename makes `create_version()`
trivial (write a new file, never touch the old one) and means a superseded
`SourceVersion.file_path` always still resolves — provenance isn't just a
database record, the original bytes remain retrievable. The type-scoped
subfolders (already present in the initial `data/` layout but unused by
phase 1's flat PDF-only storage) keep the directory browsable by source type
as ingestion formats grow.

### OCR via Tesseract, run synchronously at ingestion time

**Decision**: Handwritten-note sources (`image/png`, `image/jpeg`) run
through `pytesseract` (a thin wrapper over the `tesseract` CLI, installed as
a system dependency via Homebrew) synchronously inside the ingestion
request; the extracted text is stored on `Source.extracted_text` and
returned in the same response as the upload.

**Rationale**: Tesseract is lightweight (~60MB, no GPU, fast per-image) and
mature, versus pure-Python deep-learning OCR alternatives (EasyOCR, docTR)
which pull in a full PyTorch/TensorFlow stack (1-2GB) for marginal accuracy
gain on this kind of source material. Running OCR synchronously keeps phase
2 simple — a background job queue is unwarranted at current volume and would
add operational complexity (worker process, job status polling) with no
current requirement driving it. Revisit if ingestion volume or image size
makes request latency a problem.

**Alternatives considered**: EasyOCR/docTR (pure-pip, no system dependency,
but much larger install and slower per-image) — rejected for this project's
scale. Async/background OCR processing — deferred until synchronous latency
is actually observed to be a problem.

### `extracted_text` as a single column, not a chunked text model yet

**Decision**: OCR output is stored as one `Text` column on `Source` rather
than being split into paragraphs/chunks or given its own table.

**Rationale**: Phase 2's job is registration and identity, not understanding
— a single blob is sufficient for "did OCR work, can I see the text" today.
Phase 3 (content understanding) will need proper chunking/structure to do
extraction well, and is expected to introduce the right model for that
rather than extending this column. Treated as a deliberately temporary
shape, not a long-term schema decision.

### Web sources hash extracted text, not raw HTML

**Decision**: A web source's `content_hash` is computed over the
`trafilatura`-extracted article text, not the raw HTML response.

**Rationale**: Raw HTML changes on nearly every fetch — ads, tracking
parameters, timestamps, unrelated sidebar content — even when the article
itself is unchanged. Hashing the cleaned text means dedup and "did this
article change" checks reflect the content that actually matters, not page
noise. The tradeoff is that this ties dedup correctness to `trafilatura`'s
extraction being stable; a change in the extraction library's output for the
same input could register as a false "new version".

### Web sources identified by URL, not by upload

**Decision**: There's no separate "upload a new version of a web source"
endpoint. `POST /api/sources/web` is idempotent by URL:
`SourceRepository.get_web_source_by_url()` looks up any existing source with
`source_type="web"` and that exact URL in `original_filename`; if found, the
same content-hash-based logic used for file versioning (unchanged → no-op,
changed → archive-and-supersede) runs automatically instead of creating a
duplicate registration.

**Rationale**: A URL is a stable identity in a way a re-uploaded file isn't
— re-fetching a known URL to check for updates is a natural, repeatable
operation, whereas asking the user to separately track "which source_id did
I register this URL under" would be pure friction for no benefit. This also
means `original_filename` is reused across source types to mean "where this
came from" (filename for uploads, URL for web) rather than adding a
type-specific column.

### Generic `text_extracted` status instead of a format-specific one

**Decision**: Both handwriting (OCR) and web (scraped) sources use
`status="text_extracted"` once ingested; PDFs remain `status="registered"`.
An earlier iteration used `"ocr_complete"`, specific to image sources.

**Rationale**: The distinction that matters is "does this source already
have usable extracted text, or does phase 3 still need to produce it from
raw content" — that's true for both OCR and web scraping alike. A
format-specific status name would need a new value for every ingestion
method added later; the generic name describes the actual state instead of
how it was reached.

### Local LLM (Ollama) for concept extraction, not a hosted API

**Decision**: Concept/equation/entity extraction runs against a local model
served by [Ollama](https://ollama.com) (default: Qwen2.5-7B-Instruct),
rather than a hosted API such as Anthropic's or OpenAI's.

**Rationale**: Hosted LLM APIs bill per call/token, separately from any
chat-product subscription — an ongoing cost that doesn't fit this project.
A local model running on Ollama has zero marginal cost per extraction call,
no API key to manage, and no network dependency once the model is pulled.
The tradeoff is extraction quality: a local 7B model is noticeably weaker
than a frontier hosted model at nuanced scientific reasoning. Given the
project's current single-user, non-time-critical scope, that tradeoff is
acceptable; the `LLMClient` abstraction (below) keeps the door open to a
hosted provider later without touching the extraction pipeline.

**Alternatives considered**: A hosted API (Anthropic, OpenAI, Gemini,
Groq) — rejected (or deferred) on cost/ToS grounds, though the free-tier
hosted options remain a viable future `LLMClient` implementation if quality
becomes a priority over cost. Google Colab running a larger open model —
rejected because Colab sessions are ephemeral (idle disconnects, hard
session caps) and would require tunneling a local FastAPI app to a
notebook, adding a fragile dependency that a local Ollama service doesn't
have.

### Provider-agnostic `LLMClient` interface

**Decision**: All LLM calls go through an abstract `LLMClient`
(`src/llm/base.py`) with one method, `extract_concepts(text) -> list[dict]`.
`OllamaClient` is the only implementation today.

**Rationale**: Given the local-model choice above was driven by cost rather
than a permanent architectural commitment, extraction logic
(`ConceptExtractor`) is written against the interface, not the
implementation — adding a hosted-API-backed client later (if cost
constraints change) means writing one new class, with no changes to
chunking, validation, persistence, or the API routes.

### Single `Concept` table with a `kind` field

**Decision**: Concepts, equations, and named entities are stored in one
`Concept` table, distinguished by a `kind` column (`concept` / `equation`
/ `entity`), rather than three separate tables.

**Rationale**: All three share the same shape — a name, a description, and
a supporting quote from the source text — and the common query is "what was
extracted from this source", across all three kinds at once. A shared table
also matches the pattern already established by `Source`/`SourceVersion`:
extend with a field rather than fork the schema unless the types genuinely
diverge in structure.

**Alternatives considered**: Separate `Concept`/`Equation`/`Entity` tables —
rejected for now since none of the three currently need type-specific
columns; revisit if, e.g., equations need a distinct LaTeX/rendering field
that doesn't apply to the other kinds.

### Chunking as a separate, source-version-pinned table

**Decision**: A new `Chunk` model splits a source's `extracted_text` into
LLM-sized pieces, each one recording which `Source.version` it was derived
from.

**Rationale**: `extracted_text` was already flagged (in phase 2) as a
temporary single-blob shape. Extraction needs to operate on right-sized
pieces of text, and provenance needs to point at more than just "this
source" — concepts trace back to the specific chunk they came from. Pinning
each chunk to a `source_version` gives re-extraction a cheap staleness
check: if a source has been revised since its chunks were made, they're
recognized as stale and rebuilt automatically, without a separate
"is_current" flag to keep in sync by hand.

### Extraction is cached and idempotent, like ingestion dedup

**Decision**: `POST /api/sources/{id}/extract` only calls the LLM if the
source has no chunks yet, its chunks are stale (source revised since), or
`force=true` is explicitly passed. Otherwise it returns the existing
concepts with no LLM call.

**Rationale**: Mirrors the dedup philosophy from phases 1-2 — "remember
what's already been done" applies to extraction too. Since local-model
inference isn't free of latency (even if free of cost), making repeated
`/extract` calls on an unchanged source a no-op keeps the endpoint safe to
call defensively (e.g. from a UI on every page load) without redoing work.

### Per-chunk failure isolation, not all-or-nothing

**Decision**: Each chunk's extraction is wrapped individually. A chunk whose
LLM call fails (e.g. the model returns malformed JSON on a low-content
chunk like a references list) is recorded on `Chunk.last_extraction_error`
and skipped; the other chunks' concepts are still extracted and persisted,
and the request still returns `201` (with a `failed_chunks` count in the
response) rather than failing the whole call. A plain retry (`force` not
set) re-runs only chunks that still have an error recorded — not the whole
source — so fixing a transient failure (Ollama restarted, a flaky
response) doesn't mean redoing already-successful work.

**Rationale**: A 7B local model doesn't perfectly follow the requested JSON
schema on every call, especially on chunks with little or no extractable
content. Treating one such chunk as fatal to the entire multi-chunk
extraction discarded real, already-computed results from every other
chunk in the same request — a correctness bug, not just a quality issue.
`OllamaClient` was also made more lenient at the parsing layer itself (a
dict response with no `"items"` key is treated as "nothing found" and
logged, not raised as an error), since that's an expected shape for a
local model to produce when it has genuinely found nothing — the
chunk-level isolation is the backstop for failures that are still real
errors (bad JSON entirely, connection issues, etc).

### `embed()` chunks large batches internally, inside the provider, not the caller

**Decision**: `OllamaClient.embed()` splits its input into batches of
`settings.embedding_batch_size` (default 100) and calls Ollama once per
batch, concatenating the results — rather than sending every text in one
request, and rather than making callers (`ConceptLinker`, `Retriever`) do
the chunking themselves.

**Rationale**: Discovered empirically on real data that one oversized
`/api/embed` call can crash `nomic-embed-text`'s model-runner subprocess
(confirmed by bisection against the real failing batch — see Project
Summary for exact numbers). This is a transport/provider-level limit
specific to Ollama's current implementation, not a property of
embeddings in general — a hosted provider might have a different limit
or none. Hiding the chunking inside `OllamaClient` keeps that detail out
of `ConceptLinker` and `Retriever`, both of which still just call
`self.llm.embed(texts)` once with everything that needs embedding,
exactly as before.

### Embedding-similarity prefilter before LLM relationship classification

**Decision**: Phase 4 linking doesn't ask the LLM to judge every pair of
concepts. It first computes a cheap embedding per concept (a separate,
much smaller local model, `nomic-embed-text`) and uses plain in-memory
cosine similarity to find each concept's top-K nearest neighbors. Only
those candidate pairs are sent to the LLM to classify the actual
relationship type.

**Rationale**: Pairwise LLM judgment is O(n²) in the number of concepts —
observed extraction latency (10-90s per call) makes that impractical past
a handful of concepts. Embedding similarity is local, deterministic, and
effectively free by comparison (pure dot products), and reliably narrows
the field to plausible candidates before paying for an LLM call. This
mirrors the general shape of retrieval-augmented systems: cheap retrieval
first, expensive reasoning only on the shortlist.

**Alternatives considered**: Pure pairwise LLM judgment — rejected, does
not scale. Rule-based co-occurrence (e.g. "concepts from the same chunk
are related") — rejected as the primary mechanism since it can't identify
specific relationship types (prerequisite/derivation/application) or
connect concepts that never appeared near each other, which is the actual
point of a cross-document knowledge graph.

### One `LLMClient` interface for extraction, embedding, and classification

**Decision**: `embed()` and `classify_relationship()` were added to the
same `LLMClient` abstract base as `extract_concepts()`, rather than
introducing a separate `EmbeddingClient` abstraction.

**Rationale**: In this architecture all three capabilities come from the
same provider at once (Ollama today) — a future swap to a hosted provider
would need to implement all three together anyway. One interface per
provider is simpler than two parallel ones that would need independent
factories and settings for provider selection, for no current benefit.

### Links are directed and global (cross-source), concept pairs are
deduplicated regardless of direction

**Decision**: `ConceptLink` always stores a directed edge
(`from_concept_row_id` → `to_concept_row_id`), and linking runs over every
concept in the database regardless of which source it came from.
`exists_between()`/`delete_between()` treat A→B and B→A as the same pair
for idempotency purposes, even though "related" is conceptually symmetric
and the other three relationship types are not.

**Rationale**: Prerequisite/derivation/application are inherently
directional (A is a prerequisite FOR B, not the reverse), so the storage
shape has to support direction regardless — special-casing "related" as
unordered would complicate the schema for one relationship type out of
four. Cross-source scope is the actual point of this phase per the
original vision (an Obsidian-style graph over the whole accumulated
knowledge base, not per-document silos).

### No separate `src/knowledge/` module

**Decision**: The original phase map proposed both `src/knowledge/`
(graph storage) and `src/linking/` (edge-building) modules. Only
`src/linking/` was implemented; the graph itself lives as `ConceptLink` in
`src/database/models/` and `ConceptLinkRepository` in
`src/database/repositories/`, with the empty `src/knowledge/` package
removed.

**Rationale**: The established repository pattern already is the
"knowledge storage" layer — adding a second module for the same
responsibility would just be a second place to look for graph persistence
code, with no behavioral difference from putting it alongside `Concept`
and `SourceVersion` in the same layer those already live in.

### Linking is idempotent and incremental, like extraction

**Decision**: `POST /api/knowledge/link` only computes embeddings for
concepts that don't have one yet (or whose `embedding_model` doesn't match
the current config), and only classifies candidate pairs that aren't
already linked — unless `force=true`.

**Rationale**: Same "remember what's already been done" philosophy as
ingestion dedup and extraction caching. Concepts accumulate over time as
more sources are ingested; re-running `/link` after adding one new source
shouldn't re-embed or re-classify everything that was already linked.

### Retrieval searches chunk text, not `Concept` embeddings

**Decision**: Phase 5 added a separate `embedding`/`embedding_model` pair
of columns on `Chunk`, distinct from `Concept`'s (phase 4). `Retriever`
searches over chunk embeddings only.

**Rationale**: `Concept` embeddings capture already-summarized,
LLM-paraphrased knowledge-graph nodes — useful for concept-to-concept
similarity (phase 4), but a grounded answer needs to quote or point back
to the source's actual words, not a paraphrase of them. Keeping the two
embedding spaces on separate columns, even though both happen to use the
same underlying model today, means a future change to either the
extraction-concept embedding strategy or the retrieval-chunk one doesn't
risk silently comparing vectors from different intents.

### Asymmetric query/document prefixes for retrieval embeddings

**Decision**: `Retriever` prefixes chunk text with `"search_document: "`
and the user's query with `"search_query: "` before calling `embed()`.
Phase 4's concept-linking embeddings use no prefix at all.

**Rationale**: `nomic-embed-text` is trained for asymmetric retrieval —
short queries and long passages are expected to carry different prefixes,
and omitting them measurably weakens similarity scores. This only applies
where there's an actual query/document asymmetry: phase 5's search (a
question against prose) has one; phase 4's linking (comparing two concept
descriptions as peers) does not, so it correctly stays unprefixed.

### `retrieval_min_similarity` set lower than `link_min_similarity`

**Decision**: Phase 5 defaults `retrieval_min_similarity` to `0.3`, well
below phase 4's `link_min_similarity` of `0.5`.

**Rationale**: The two thresholds compare different kinds of pairs. Phase
4 compares two concept descriptions, which tend to be similarly shaped
and dense; phase 5 compares a short, informally-phrased question against
a long, dense paragraph of source prose, which structurally scores lower
on raw cosine similarity even when the match is the right one. Verified
against real `nomic-embed-text` output: genuinely correct top matches
scored in the 0.5-0.76 range, but second/third-place unrelated sources
still scored 0.48-0.56 — a single shared threshold tuned for phase 4
would have been too strict here, or too loose there.

### `ContextAssembler` enriches with existing concepts, generates no answer

**Decision**: `ContextAssembler.assemble()` retrieves chunks and attaches
each one's already-extracted concepts (`ConceptRepository.list_for_chunk`)
but does not call the LLM to generate any answer text.

**Rationale**: Keeps phase 5 scoped to "find and cite the relevant
material" — a clean, independently testable responsibility — and leaves
"generate a grounded answer from this material" entirely to phase 6,
where the actual chat prompt shape (how much context, what citation
format, conversation history) can be designed without phase 5 having
pre-committed to assumptions about it.

### Graph-adjacent expansion deferred, not built into phase 5

**Decision**: `/api/context` returns only the top-k chunks by direct
embedding similarity to the query. It does not walk `ConceptLink` edges
from a retrieved chunk's concepts to pull in prerequisite/related
concepts whose own chunks didn't directly match.

**Rationale**: The original vision explicitly wants "not just relevant
text but also the related concepts and supporting knowledge" — this is a
real gap, not an oversight — but doing it well depends on decisions phase
6 hasn't made yet (how much extra context a chat prompt can absorb before
it hurts more than helps, how to weight direct matches against graph
neighbors). Shipping plain top-k retrieval first keeps this phase
testable in isolation; graph expansion is a layer on top of the exact
same `AssembledContext` shape, not a rewrite of it.

### Chat is stateless today, designed to accept persisted history later without a rewrite

**Decision**: `ChatService.answer(question, history, top_k)` takes `history`
as a plain list of `{"role", "content"}` dicts, supplied by the caller on
every request. No `Conversation`/`Message` database model exists yet.

**Rationale**: Shipping the simplest version first — no new tables, no
server-side state — while keeping the service's actual contract
(messages in, token stream out) stable. If persisted conversations are
added later, only the *source* of the `history` list changes (loaded from
the database by a `conversation_id` instead of read from the request
body); nothing about prompt-building or streaming needs to change.

### `Client.chat()` with a message list, not `Client.generate()` with a flat prompt

**Decision**: `OllamaClient.stream_chat()` uses the `ollama` package's
`Client.chat(messages=[...], stream=True)`, unlike `extract_concepts()`
and `classify_relationship()`, which both use `Client.generate(system=,
prompt=, ...)`.

**Rationale**: Chat inherently needs a list of turns (system instructions,
prior exchanges, the new question) with roles preserved — `chat()` takes
that shape directly, matching what `ChatService` already needed to build.
Using `generate()` instead would mean hand-concatenating roles into a
single string and losing the model's native understanding of turn
boundaries, for no benefit.

### Citations travel in a response header, not inline in the streamed body

**Decision**: `POST /api/chat` returns a `StreamingResponse` whose body is
pure answer text; which chunks were used is sent separately as a JSON
`X-Citations` response header, set before streaming begins.

**Rationale**: The citation list is fully known before the LLM call even
starts (it's the synchronous output of `ContextAssembler.assemble()`).
Mixing structured citation data into a plain-text token stream would need
either a delimiter convention or a real event protocol (SSE) — unwarranted
complexity for a first version with no frontend yet to consume it. A
header is simple, correct, and arrives atomically rather than needing to
be parsed out of a stream.

### Chat grounds on phase 5's plain top-k retrieval, no graph expansion

**Decision**: `ChatService` calls `ContextAssembler.assemble()` unchanged
— it does not additionally walk `ConceptLink` edges to pull in
prerequisite/related concepts of a matched chunk.

**Rationale**: Same reasoning as phase 5's own deferral of this: ships a
testable, working chat loop first. Graph-adjacent expansion is additive on
top of the same `AssembledContext` shape (more items, not a different
shape), so it can be layered in later without touching `ChatService`'s
prompt-building or the API's request/response contract.

### Citation validity and groundedness are two separate checks

**Decision**: `CitationChecker` (pure regex/index arithmetic, no LLM) and
`GroundednessJudge` (LLM-as-judge) are kept as two independent classes
rather than merged into one "answer quality" check.

**Rationale**: They check genuinely different things. A citation marker
can point at a real context item (structurally valid) while the sentence
beside it still misstates what that item says (not grounded) — and the
reverse is possible too (an answer with no citations at all could still
be fully accurate). Running both and reporting them separately avoids
conflating "cited correctly" with "told the truth" into one ambiguous
score, and lets the cheap, 100%-reliable structural check run even when
the slower LLM-judged one isn't needed.

### Retrieval evaluation searches wider and unfiltered, not with production defaults

**Decision**: `RetrievalEvaluator` calls `Retriever.search()` with
`min_similarity=0.0` and a wide `top_k` (`settings.eval_retrieval_top_k`,
default 20), ignoring phase 5's production defaults (`top_k=5`,
`min_similarity=0.3`).

**Rationale**: The evaluator's purpose is to measure the embedding
ranking itself, specifically so `retrieval_top_k`/`retrieval_min_similarity`
can be tuned from real results. If evaluation used the same cutoffs as
production, a correct result ranked 6th (just outside `top_k=5`) would
report identically to one ranked 1000th — a flat miss — destroying the
exact signal needed to distinguish "close, raise `top_k` a little" from
"the embedding doesn't understand this query at all."

### Evaluation reuses `ChatService` directly rather than reimplementing chat logic

**Decision**: `POST /api/evaluation/chat` calls the exact same
`ChatService.answer()` that `POST /api/chat` uses, just collecting the
generator into a complete string instead of streaming it.

**Rationale**: An evaluation endpoint that ran a separate, parallel
implementation of "ask a question, get a grounded answer" could drift
from the real chat path and end up testing something that isn't actually
what users hit. Reusing the same service guarantees the thing being
evaluated is the thing actually shipped.

### No linking-quality evaluation (scope decision)

**Decision**: Phase 7 added evaluation for retrieval quality and
chat groundedness/citations, but not for `ConceptLinker`'s relationship
classification accuracy, even though it's the same kind of
LLM-classification task and could be evaluated the same way.

**Rationale**: The original roadmap commitment for this phase was
specifically "groundedness and retrieval quality checks" — linking
accuracy wasn't part of that promise, and adding it wasn't requested.
Left as a known gap rather than scope creep; it would follow the exact
same shape as `RetrievalEvaluator` if built later (a labeled set of
concept pairs with expected relationship types, run through
`classify_relationship()` directly, accuracy/confusion computed from the
results).

### Concept deduplication and graph cleanup run as a one-time migration script, not an API endpoint

**Decision**: Real data turned out far more granular than useful — a large
share of concepts were bibliographic scaffolding (citations, figure/table
labels, bare years, venue names) misfiled as `entity`-kind, and a
meaningful fraction were near-duplicates of each other, extracted
separately per chunk with no cross-chunk awareness. Addressed with: a
regex junk filter (`is_bibliographic_junk()`) run at extraction time going
forward; a `quote_verified` flag on `Concept` (does the quote actually
appear verbatim in its chunk); a two-pass dedup merge (exact normalized
name, then high-threshold embedding similarity within the same `kind`)
that preserves every merged concept's source chunks in a new
`ConceptProvenance` table; `kind="entity"` excluded from linking entirely;
the relationship taxonomy extended (`same_as`/`is_a`/`part_of`) with an
explicit LLM-chosen edge direction instead of the old arbitrary
`from=concept_a, to=concept_b` ordering; and a per-node cap on final
stored edges (`ConceptLinkRepository.prune_to_top_n_per_node()`).

**Rationale**: This was a one-time cleanup of data extracted before these
rules existed, not a policy a repeated API call should re-apply — so it's
a standalone script (`scripts/migrate_concept_cleanup.py`) run once against
the live `.db` file with the server stopped, not a new endpoint. The
filter/verification/taxonomy/cap pieces are also now enforced going
forward by the normal extraction/linking path, so this kind of drift
shouldn't reaccumulate at the same rate. Full before/after numbers from
running this against real data are in Project Summary, not here — they're
a one-time data point, not part of the lasting design.

## Known Gaps (tracked, not yet addressed)

- Every `POST /api/sources/web` call re-fetches and re-extracts over the
  network — no HTTP caching (ETag/Last-Modified) — so checking a known URL
  for updates always costs a full fetch + parse. Acceptable at current
  volume; would need caching before any scheduled/frequent re-checking.
- `trafilatura`'s plain-text output can retain markdown-style table
  formatting from source HTML (observed on a Wikipedia infobox) — not
  post-processed further; extraction chunks this as-is rather than assuming
  `extracted_text` is clean prose.
- No database-level uniqueness constraint on `(source_row_id, version)` in
  `SourceVersion` — correctness currently relies on `create_version()` being
  the only write path. Worth adding if multi-writer access becomes possible.
- OCR uses Tesseract's English data only, with no image preprocessing
  (deskew, binarization, contrast normalization) — accuracy on real
  handwriting (vs. cleanly rendered text) has not yet been evaluated.
- PDFs with no extractable text layer (e.g. scanned/image-only PDFs) still
  get no `extracted_text` — `pypdfium2` only reads an existing text layer;
  OCR-on-rendered-PDF-pages would be needed to close this.
- Extraction quality is bounded by the local model (Qwen2.5-7B-Instruct) —
  no fact-checking or groundedness verification exists yet beyond
  structural validation (required fields present, `kind` in the allowed
  set). A hallucinated but well-formed item would pass through.
- Chunking is a plain character-based sliding window with whitespace-aware
  boundaries — no sentence/paragraph/semantic awareness.
- No automatic retry/backoff around a failed Ollama call — a failed chunk
  is recorded and skipped (see "Per-chunk failure isolation" above) and
  only retried on the next plain `/extract` call, not immediately.
- Concept provenance points at a chunk, not a page/character offset within
  the source — page-level citation isn't available yet.
- No authentication/ownership model — acceptable for the current
  single-user scope, will need revisiting if that changes.
- No automated test suite yet; `pytest` is a dependency but unused so far —
  verification has been ad hoc smoke testing via `TestClient` and a real
  running server.
- `link_min_similarity` (0.5), `link_top_k` (5), and `link_max_per_node`
  (3) are still untuned beyond one real data point each — no sweep has
  been done to see whether a different cap or threshold changes answer
  quality downstream, only graph tidiness.
- The exact-name/embedding-similarity merge in
  `scripts/migrate_concept_cleanup.py` is a heuristic, not a verified-correct
  dedup — the 0.97 cosine threshold and "longer description wins" canonical
  pick were chosen by inspection of one real dataset, not validated against
  a labeled set of true/false duplicate pairs.
- `is_bibliographic_junk()`'s regex patterns are English-only and were
  written against what actually showed up in 6 real ML papers — likely
  misses venue/citation styles from other fields or languages.
- The new `same_as`/`is_a`/`part_of` relationship types are stored like any
  other edge but nothing yet *consumes* them specially (e.g. auto-merging
  concepts the LLM itself flags as `same_as` but the embedding pass missed).
- `embedding_batch_size` (100) is an empirically-chosen safe margin below
  a confirmed real crash threshold on this specific machine/Ollama
  version — not a documented Ollama limit, so it may need to move on
  different hardware or a future Ollama version.
- `GET /graph` (a static page served by `src/main.py`) visualizes the
  knowledge graph live against `/api/knowledge/graph` — not yet shareable
  outside localhost.
- `_top_k_candidates()` computes cosine similarity over every concept pair
  in memory (O(n²)) — fine at hundreds of concepts, would need a proper
  vector index at much larger scale.
- No constraint preventing two different `relationship_type` edges between
  the same concept pair across separate `force=true` runs that produce
  different LLM judgments — `exists_between()` only checks pair identity.
- Concept embeddings are stored as JSON text, not a dedicated vector
  column/index — the same "deliberately temporary" shape `extracted_text`
  had before chunking existed.
- No migration tool (no Alembic) — schema changes rely on
  `Base.metadata.create_all()`, which creates missing tables but never
  alters existing ones. This has already required one hand-written,
  idempotent `ALTER TABLE` against real data (for `Concept.quote_verified`)
  because `create_all()` silently does nothing for a new column on an
  already-populated table. Worth adopting Alembic before the next schema
  change that touches populated data.
- `retrieval_min_similarity` (0.3) and `retrieval_top_k` (5) are
  reasoned-but-untuned defaults, validated only against a 3-source
  synthetic smoke test — real retrieval quality against a large, varied
  set of chunks hasn't been evaluated yet.
- `/api/context` does no graph-adjacent expansion via `ConceptLink` —
  see "Graph-adjacent expansion deferred" above.
- Every `/api/retrieval/search` or `/api/context` call re-scans every
  chunk's embedding in memory (same O(n) shape as phase 4's O(n²)
  linking) — fine at current scale, would need a real vector index
  later. No caching of a fully assembled context across repeated
  identical questions, either — only chunk embeddings are cached, not
  query embeddings or assembled results.
- Chat has no persisted conversation history — the caller must resend
  the full history every call, with no truncation/summarization of long
  conversations. See "Chat is stateless today" above for why this is a
  deliberate, reversible choice rather than an oversight.
- No citation-accuracy verification — the model is instructed to cite
  `[1]`, `[2]` etc. matching the numbered context items, but nothing
  checks it actually did, or that a given citation is the right one for
  the claim beside it. Groundedness/citation checking is explicitly
  phase 7's job.
- If the Ollama call fails after streaming has already started (first
  token sent, then a failure), the client sees an abruptly truncated 200
  response rather than a clean error status — HTTP headers are already
  committed by the time any mid-stream failure could be detected. Only a
  failure before the first token surfaces as a real error today.
- `POST /api/chat` has no frontend yet — it's exercised via
  TestClient/curl/Swagger, all of which buffer the full streamed
  response rather than rendering it incrementally like a real chat UI
  would.
- No persisted evaluation test sets or run history — labeled cases are
  supplied in the request body every call to `/api/evaluation/retrieval`;
  nothing is saved to compare quality across code/prompt/model changes
  over time.
- No real labeled evaluation data yet — verification so far used small
  synthetic test sets, since the real project database currently has 0
  rows (the still-deferred data-loss recovery from phase 3). Building an
  actually useful labeled set depends on that recovery happening first.
- No linking-quality evaluation — see "No linking-quality evaluation
  (scope decision)" above.
- `GroundednessJudge` uses the same local model that generated the
  answer to check its own output — not independent verification by a
  separate, stronger model. A confidently-worded hallucination could
  plausibly fool both the generator and the judge the same way.
- `CitationChecker` only validates that a marker points at a real
  context item, not that the specific sentence beside it is supported by
  that item's content — that finer-grained check isn't implemented;
  `GroundednessJudge` only judges the answer holistically, not
  marker-by-marker.
- No CI, so evaluation endpoints are tools run by hand today, not an
  automatic regression gate on changes.
