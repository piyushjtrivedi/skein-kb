from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.connection import Base

ALLOWED_RELATIONSHIPS = (
    "prerequisite",
    "related",
    "derivation",
    "application",
    "same_as",
    "is_a",
    "part_of",
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ConceptLink(Base):
    """A directed edge between two concepts, possibly from different
    sources — the actual "knowledge graph" of phase 4.

    Stored as a directed pair (from_concept_row_id -> to_concept_row_id)
    even for the symmetric "related" relationship, since prerequisite/
    derivation/application are inherently directional (A is a prerequisite
    FOR B, not the reverse) and one shape is simpler than special-casing
    "related" as unordered. Queries for "all links touching concept X"
    check both columns.
    """

    __tablename__ = "concept_links"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    link_id: Mapped[str] = mapped_column(
        String(36),
        unique=True,
        index=True,
        nullable=False,
    )

    from_concept_row_id: Mapped[int] = mapped_column(
        ForeignKey("concepts.id"),
        nullable=False,
        index=True,
    )

    to_concept_row_id: Mapped[int] = mapped_column(
        ForeignKey("concepts.id"),
        nullable=False,
        index=True,
    )

    relationship_type: Mapped[str] = mapped_column(String(20), nullable=False)

    rationale: Mapped[str | None] = mapped_column(Text, nullable=True)

    similarity: Mapped[float] = mapped_column(Float, nullable=False)
    """The embedding cosine similarity that surfaced this pair as a
    candidate in the first place — kept for provenance/debugging, not
    used after the LLM has classified the relationship."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utc_now,
        nullable=False,
    )

    from_concept: Mapped["Concept"] = relationship(foreign_keys=[from_concept_row_id])
    to_concept: Mapped["Concept"] = relationship(foreign_keys=[to_concept_row_id])
