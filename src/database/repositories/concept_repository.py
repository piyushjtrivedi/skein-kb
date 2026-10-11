import json
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.database.models.chunk import Chunk
from src.database.models.concept import Concept
from src.database.models.source import Source


class ConceptRepository:
    def __init__(self, db: Session):
        self.db = db

    def list_for_source(self, source: Source) -> list[Concept]:
        return list(
            self.db.scalars(
                select(Concept)
                .where(Concept.source_row_id == source.id)
                .order_by(Concept.id)
            ).all()
        )

    def list_all(self) -> list[Concept]:
        return list(self.db.scalars(select(Concept).order_by(Concept.id)).all())

    def list_for_chunk(self, chunk: Chunk) -> list[Concept]:
        return list(
            self.db.scalars(
                select(Concept)
                .where(Concept.chunk_row_id == chunk.id)
                .order_by(Concept.id)
            ).all()
        )

    def get_by_concept_id(self, concept_id: str) -> Concept | None:
        return self.db.scalar(
            select(Concept).where(Concept.concept_id == concept_id)
        )

    def update_embedding(self, concept: Concept, vector: list[float], model: str) -> None:
        concept.embedding = json.dumps(vector)
        concept.embedding_model = model
        self.db.commit()

    def create_many(self, source: Source, chunk: Chunk, items: list[dict]) -> list[Concept]:
        concepts = [
            Concept(
                concept_id=str(uuid4()),
                source_row_id=source.id,
                chunk_row_id=chunk.id,
                kind=item["kind"],
                name=item["name"],
                description=item["description"],
                quote=item.get("quote"),
                quote_verified=item.get("quote_verified", False),
            )
            for item in items
        ]
        self.db.add_all(concepts)
        self.db.commit()
        for concept in concepts:
            self.db.refresh(concept)
        return concepts

    def delete_for_source(self, source: Source) -> None:
        for concept in self.list_for_source(source):
            self.db.delete(concept)
        self.db.commit()

    @staticmethod
    def to_dict(concept: Concept) -> dict:
        return {
            "concept_id": concept.concept_id,
            "kind": concept.kind,
            "name": concept.name,
            "description": concept.description,
            "quote": concept.quote,
            "quote_verified": concept.quote_verified,
            "source_id": concept.source.source_id,
            "source_name": concept.source.name,
            "created_at": concept.created_at,
        }
