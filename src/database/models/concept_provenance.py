from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, Integer, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from src.database.connection import Base


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ConceptProvenance(Base):
    """Records every chunk a Concept was mentioned in.

    A Concept keeps a single `chunk_row_id` as its "primary" origin, but
    deduplication merges several near-identical Concept rows (extracted
    separately per chunk) into one survivor — this table is where the
    chunks of the ones that got merged away are preserved, so a merged
    concept doesn't silently lose where its other mentions came from.
    """

    __tablename__ = "concept_provenance"
    __table_args__ = (UniqueConstraint("concept_row_id", "chunk_row_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    concept_row_id: Mapped[int] = mapped_column(
        ForeignKey("concepts.id"), nullable=False, index=True
    )

    chunk_row_id: Mapped[int] = mapped_column(
        ForeignKey("chunks.id"), nullable=False, index=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
