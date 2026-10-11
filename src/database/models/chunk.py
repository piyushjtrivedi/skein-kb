from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.connection import Base


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Chunk(Base):
    """A slice of a Source's extracted_text, sized for LLM extraction.

    `source_version` pins the chunk to the Source.version it was split from,
    so re-extraction can detect when a source has moved on and old chunks
    (and the concepts derived from them) are stale.
    """

    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    chunk_id: Mapped[str] = mapped_column(
        String(36),
        unique=True,
        index=True,
        nullable=False,
    )

    source_row_id: Mapped[int] = mapped_column(
        ForeignKey("sources.id"),
        nullable=False,
        index=True,
    )

    source_version: Mapped[int] = mapped_column(Integer, nullable=False)

    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)

    text: Mapped[str] = mapped_column(Text, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    last_extraction_error: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )
    """Set when the last extraction attempt on this chunk failed (e.g. the
    LLM returned malformed JSON); cleared on success. Lets a plain retry
    (no `force`) redo only the chunks that actually failed, instead of
    either skipping them forever or re-running every chunk."""

    last_extraction_attempt_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    """Set on every extraction attempt, success or failure. Distinguishes
    "never attempted" (e.g. the server was restarted mid-extraction, before
    this chunk's turn) from "succeeded with zero concepts found" — both
    leave `last_extraction_error` as None, so this timestamp is what a
    plain retry uses to tell the two apart."""

    embedding: Mapped[str | None] = mapped_column(Text, nullable=True)
    """JSON-encoded list of floats, used for phase 5 retrieval (similarity
    search over chunk text). Stored as JSON text for the same reason as
    `Concept.embedding` — no native vector column in SQLite at this scale."""

    embedding_model: Mapped[str | None] = mapped_column(String(100), nullable=True)
    """Which model produced `embedding` — lets a changed embedding model be
    detected and recomputed, instead of comparing vectors from two spaces."""

    source: Mapped["Source"] = relationship()
