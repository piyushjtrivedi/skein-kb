import json
from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.database.models.chunk import Chunk
from src.database.models.source import Source


class ChunkRepository:
    def __init__(self, db: Session):
        self.db = db

    def list_for_source(self, source: Source) -> list[Chunk]:
        return list(
            self.db.scalars(
                select(Chunk)
                .where(Chunk.source_row_id == source.id)
                .order_by(Chunk.chunk_index)
            ).all()
        )

    def list_all(self) -> list[Chunk]:
        return list(self.db.scalars(select(Chunk).order_by(Chunk.id)).all())

    def update_embedding(self, chunk: Chunk, vector: list[float], model: str) -> None:
        chunk.embedding = json.dumps(vector)
        chunk.embedding_model = model
        self.db.commit()

    def create_many(self, source: Source, texts: list[str]) -> list[Chunk]:
        chunks = [
            Chunk(
                chunk_id=str(uuid4()),
                source_row_id=source.id,
                source_version=source.version,
                chunk_index=index,
                text=text,
            )
            for index, text in enumerate(texts)
        ]
        self.db.add_all(chunks)
        self.db.commit()
        for chunk in chunks:
            self.db.refresh(chunk)
        return chunks

    def delete_for_source(self, source: Source) -> None:
        for chunk in self.list_for_source(source):
            self.db.delete(chunk)
        self.db.commit()

    def mark_extraction_result(self, chunk: Chunk, *, error: str | None) -> None:
        chunk.last_extraction_error = error
        chunk.last_extraction_attempt_at = datetime.now(timezone.utc)
        self.db.commit()

    @staticmethod
    def to_dict(chunk: Chunk) -> dict:
        return {
            "chunk_id": chunk.chunk_id,
            "source_version": chunk.source_version,
            "chunk_index": chunk.chunk_index,
            "text": chunk.text,
            "created_at": chunk.created_at,
        }
