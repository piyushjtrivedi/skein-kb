from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.connection import Base

ALLOWED_KINDS = ("concept", "equation", "entity")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Concept(Base):
    """A concept, equation, or named entity extracted from one Chunk.

    `kind` distinguishes the three within a single table rather than three
    separate ones, since they share the same shape (name, description,
    supporting quote) and callers usually want to query across all of them
    for a source at once.
    """

    __tablename__ = "concepts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    concept_id: Mapped[str] = mapped_column(
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

    chunk_row_id: Mapped[int] = mapped_column(
        ForeignKey("chunks.id"),
        nullable=False,
        index=True,
    )

    kind: Mapped[str] = mapped_column(String(20), nullable=False)

    name: Mapped[str] = mapped_column(String(255), nullable=False)

    description: Mapped[str] = mapped_column(Text, nullable=False)

    quote: Mapped[str | None] = mapped_column(Text, nullable=True)

    embedding: Mapped[str | None] = mapped_column(Text, nullable=True)
    """JSON-encoded list of floats (e.g. from nomic-embed-text), used only
    for the phase 4 linking step's similarity search. Stored as JSON text
    rather than a dedicated vector type since SQLite has none and the
    concept count here doesn't warrant a vector extension/separate store."""

    embedding_model: Mapped[str | None] = mapped_column(String(100), nullable=True)
    """Which model produced `embedding` — lets re-linking detect that the
    configured embedding model changed and recompute, instead of silently
    comparing vectors from two different embedding spaces."""

    quote_verified: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    """Whether `quote` was found verbatim (case/whitespace-normalized) in its
    source chunk's text. The extraction model occasionally paraphrases
    instead of quoting; this flags which quotes are safe to cite as-is."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    source: Mapped["Source"] = relationship()
    chunk: Mapped["Chunk"] = relationship()
    provenance: Mapped[list["ConceptProvenance"]] = relationship(
        cascade="all, delete-orphan"
    )
