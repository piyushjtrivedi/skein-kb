from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.connection import Base


class SourceVersion(Base):
    """A superseded version of a Source, kept for provenance."""

    __tablename__ = "source_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    source_row_id: Mapped[int] = mapped_column(
        ForeignKey("sources.id"),
        nullable=False,
        index=True,
    )

    version: Mapped[int] = mapped_column(Integer, nullable=False)

    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    file_path: Mapped[str] = mapped_column(Text, nullable=False)

    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    source: Mapped["Source"] = relationship(back_populates="versions")
