import json
import logging
import math
from dataclasses import dataclass

from sqlalchemy.orm import Session

from src.core.config import settings
from src.database.models.chunk import Chunk
from src.database.repositories.chunk_repository import ChunkRepository
from src.llm import get_llm_client
from src.llm.base import LLMClient

logger = logging.getLogger(__name__)

# nomic-embed-text is trained for asymmetric retrieval: queries and the
# passages they're matched against need different prefixes, or similarity
# scores come out noticeably weaker. Phase 4's concept-to-concept linking
# has no query/document distinction (it's symmetric), so this doesn't apply
# there — it's specific to this query-against-chunk-text search.
QUERY_PREFIX = "search_query: "
DOCUMENT_PREFIX = "search_document: "


@dataclass
class RetrievedChunk:
    chunk: Chunk
    similarity: float


class Retriever:
    """Embedding-similarity search over chunk text, for grounding answers in
    the user's own sources. Reuses the same LLMClient/embedding model as
    phase 4's linking, applied here to raw chunk text instead of concepts."""

    def __init__(self, db: Session, llm_client: LLMClient | None = None):
        self.chunk_repo = ChunkRepository(db)
        self.llm = llm_client or get_llm_client()

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        min_similarity: float | None = None,
    ) -> list[RetrievedChunk]:
        top_k = settings.retrieval_top_k if top_k is None else top_k
        min_similarity = (
            settings.retrieval_min_similarity if min_similarity is None else min_similarity
        )

        chunks = self.chunk_repo.list_all()
        if not chunks:
            return []

        self._ensure_embeddings(chunks)

        query_vector = self.llm.embed([QUERY_PREFIX + query])[0]

        scored: list[RetrievedChunk] = []
        for chunk in chunks:
            if not chunk.embedding:
                continue
            similarity = self._cosine(query_vector, json.loads(chunk.embedding))
            if similarity >= min_similarity:
                scored.append(RetrievedChunk(chunk=chunk, similarity=similarity))

        scored.sort(key=lambda r: r.similarity, reverse=True)
        return scored[:top_k]

    def _ensure_embeddings(self, chunks: list[Chunk]) -> int:
        missing = [
            c for c in chunks
            if not c.embedding or c.embedding_model != settings.embedding_model
        ]
        if not missing:
            return 0

        texts = [DOCUMENT_PREFIX + c.text for c in missing]
        vectors = self.llm.embed(texts)
        for chunk, vector in zip(missing, vectors):
            self.chunk_repo.update_embedding(chunk, vector, settings.embedding_model)
        logger.info("Retrieval: computed embeddings for %d chunk(s)", len(missing))
        return len(missing)

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
