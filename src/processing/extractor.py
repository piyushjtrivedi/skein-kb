import logging
import re
from dataclasses import dataclass

from sqlalchemy.orm import Session

from src.core.config import settings
from src.database.models.chunk import Chunk
from src.database.models.concept import ALLOWED_KINDS, Concept
from src.database.models.source import Source
from src.database.repositories.chunk_repository import ChunkRepository
from src.database.repositories.concept_repository import ConceptRepository
from src.llm import get_llm_client
from src.llm.base import LLMClient

logger = logging.getLogger(__name__)

_JUNK_ENTITY_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"^(19|20)\d{2}$",                          # bare year
        r"^figure\s*\d+[a-z]?$",
        r"^table\s*\d+[a-z]?$",
        r"^section\s*\d+(\.\d+)*$",
        r"^(appendix|equation|eq\.?)\s*[a-z0-9.]+$",
        r"et\s*al\.?$",
        r"^arxiv[:\s]",
        r"^(pp?|pg)\.?\s*\d+(-\d+)?$",               # page numbers
        r"^(iclr|neurips|nips|icml|cvpr|acl|emnlp)\s*(20\d{2})?$",
        r"^\[\d+\]$",                                 # bracketed citation marker
    ]
]


def is_bibliographic_junk(name: str) -> bool:
    """True for entity names that are bibliographic scaffolding (citations,
    figure/table/section labels, bare years, venue names) rather than
    actual scientific content -- these made up roughly half of all
    extracted concepts on real data and added only noise to the graph."""
    normalized = name.strip()
    return any(pattern.match(normalized) for pattern in _JUNK_ENTITY_PATTERNS)


def verify_quote(quote: str | None, chunk_text: str) -> bool:
    """Whether `quote` appears verbatim in `chunk_text`, modulo case and
    whitespace collapsing -- the extraction model sometimes paraphrases
    instead of quoting exactly."""
    if not quote:
        return False
    normalize = lambda s: re.sub(r"\s+", " ", s).strip().lower()
    return normalize(quote) in normalize(chunk_text)


@dataclass
class ExtractionResult:
    source: Source
    chunks: list[Chunk]
    concepts: list[Concept]
    newly_extracted: bool
    failed_chunks: int = 0


class ConceptExtractor:
    def __init__(self, db: Session, llm_client: LLMClient | None = None):
        self.chunk_repo = ChunkRepository(db)
        self.concept_repo = ConceptRepository(db)
        self.llm = llm_client or get_llm_client()

    def extract(self, source: Source, *, force: bool = False) -> ExtractionResult:
        if not source.extracted_text:
            raise ValueError(
                "Source has no extracted text yet; nothing to extract concepts from."
            )

        existing_chunks = self.chunk_repo.list_for_source(source)
        chunks_current = bool(existing_chunks) and existing_chunks[0].source_version == source.version

        if chunks_current and not force:
            chunks = existing_chunks
            chunks_to_run = [
                c for c in chunks
                if c.last_extraction_attempt_at is None or c.last_extraction_error is not None
            ]
            if not chunks_to_run:
                return ExtractionResult(
                    source=source,
                    chunks=chunks,
                    concepts=self.concept_repo.list_for_source(source),
                    newly_extracted=False,
                )
        else:
            if existing_chunks:
                self.chunk_repo.delete_for_source(source)
            texts = self._chunk_text(
                source.extracted_text,
                size=settings.chunk_size,
                overlap=settings.chunk_overlap,
            )
            chunks = self.chunk_repo.create_many(source, texts)
            chunks_to_run = chunks
            self.concept_repo.delete_for_source(source)

        failed_chunks = 0
        total = len(chunks_to_run)
        for index, chunk in enumerate(chunks_to_run, start=1):
            try:
                items = self.llm.extract_concepts(chunk.text)
            except ValueError as exc:
                logger.warning(
                    "Chunk %s (source %s) failed extraction, will retry on next "
                    "non-forced call: %s",
                    chunk.chunk_id, source.source_id, exc,
                )
                self.chunk_repo.mark_extraction_result(chunk, error=str(exc))
                failed_chunks += 1
                self._log_progress(source, index, total)
                continue

            self.chunk_repo.mark_extraction_result(chunk, error=None)
            valid_items = [
                item for item in (self._validate_item(i, chunk.text) for i in items) if item
            ]
            if valid_items:
                self.concept_repo.create_many(source, chunk, valid_items)
            self._log_progress(source, index, total)

        return ExtractionResult(
            source=source,
            chunks=chunks,
            concepts=self.concept_repo.list_for_source(source),
            newly_extracted=True,
            failed_chunks=failed_chunks,
        )

    @staticmethod
    def _log_progress(source: Source, index: int, total: int) -> None:
        percent_done = (index / total) * 100 if total else 100
        logger.info(
            "Extraction progress | source=%s | chunk %d/%d done | %.0f%% "
            "complete, %.0f%% remaining",
            source.source_id, index, total, percent_done, 100 - percent_done,
        )

    @staticmethod
    def _validate_item(item: dict, chunk_text: str) -> dict | None:
        if not isinstance(item, dict):
            logger.warning("Skipping non-dict extracted item: %r", item)
            return None

        kind = item.get("kind")
        name = item.get("name")
        description = item.get("description")

        if kind not in ALLOWED_KINDS or not name or not description:
            logger.warning("Skipping malformed extracted item: %r", item)
            return None

        name = str(name).strip()
        if kind == "entity" and is_bibliographic_junk(name):
            logger.info("Dropping bibliographic/entity junk: %r", name)
            return None

        quote = str(item["quote"]).strip() if item.get("quote") else None

        return {
            "kind": kind,
            "name": name,
            "description": str(description).strip(),
            "quote": quote,
            "quote_verified": verify_quote(quote, chunk_text),
        }

    @staticmethod
    def _chunk_text(text: str, *, size: int, overlap: int) -> list[str]:
        text = text.strip()
        if not text:
            return []
        if len(text) <= size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            if end < len(text):
                boundary = text.rfind(" ", start, end)
                if boundary > start:
                    end = boundary
            piece = text[start:end].strip()
            if piece:
                chunks.append(piece)
            if end >= len(text):
                break
            start = max(end - overlap, start + 1)
        return chunks
