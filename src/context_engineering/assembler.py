from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from src.database.models.chunk import Chunk
from src.database.repositories.concept_repository import ConceptRepository
from src.llm.base import LLMClient
from src.retrieval.retriever import Retriever


@dataclass
class ContextItem:
    source_id: str
    source_name: str
    chunk_index: int
    text: str
    similarity: float
    concepts: list[dict] = field(default_factory=list)


@dataclass
class AssembledContext:
    query: str
    items: list[ContextItem]

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "items": [
                {
                    "source_id": item.source_id,
                    "source_name": item.source_name,
                    "chunk_index": item.chunk_index,
                    "text": item.text,
                    "similarity": item.similarity,
                    "concepts": item.concepts,
                }
                for item in self.items
            ],
        }


class ContextAssembler:
    """Turns a retrieval result into a structured, citable context block.

    This is the hand-off point for phase 6's chat: it finds the relevant
    chunks and annotates each with the concepts already extracted from it
    (phase 3), but generates no answer itself — that's the next phase.
    """

    def __init__(self, db: Session, llm_client: LLMClient | None = None):
        self.retriever = Retriever(db, llm_client=llm_client)
        self.concept_repo = ConceptRepository(db)

    def assemble(
        self,
        query: str,
        *,
        top_k: int | None = None,
        min_similarity: float | None = None,
    ) -> AssembledContext:
        retrieved = self.retriever.search(query, top_k=top_k, min_similarity=min_similarity)

        items = [
            ContextItem(
                source_id=r.chunk.source.source_id,
                source_name=r.chunk.source.name,
                chunk_index=r.chunk.chunk_index,
                text=r.chunk.text,
                similarity=r.similarity,
                concepts=[
                    {"kind": c.kind, "name": c.name, "description": c.description}
                    for c in self._concepts_for_chunk(r.chunk)
                ],
            )
            for r in retrieved
        ]
        return AssembledContext(query=query, items=items)

    def _concepts_for_chunk(self, chunk: Chunk) -> list:
        return self.concept_repo.list_for_chunk(chunk)
