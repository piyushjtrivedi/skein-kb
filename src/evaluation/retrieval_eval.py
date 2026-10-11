from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from src.core.config import settings
from src.llm.base import LLMClient
from src.retrieval.retriever import Retriever


@dataclass
class RetrievalCaseResult:
    query: str
    expected_source_id: str
    expected_chunk_index: int | None
    hit: bool
    rank: int | None
    top_results: list[dict] = field(default_factory=list)


@dataclass
class RetrievalEvalResult:
    cases: list[RetrievalCaseResult]
    hit_rate: float
    mean_reciprocal_rank: float


class RetrievalEvaluator:
    """Batch-checks Retriever quality against a labeled set of
    (query, expected source/chunk) pairs — the thing actually needed to
    tune `retrieval_top_k`/`retrieval_min_similarity` instead of guessing.

    Searches with `min_similarity=0.0` and a wide `top_k` by default
    (`settings.eval_retrieval_top_k`, not the production
    `settings.retrieval_top_k`) regardless of what the caller passes in
    for display purposes, so a "miss" in the result reflects the
    embedding ranking itself, not an artifact of the production cutoff.
    """

    def __init__(self, db: Session, llm_client: LLMClient | None = None):
        self.retriever = Retriever(db, llm_client=llm_client)

    def evaluate(self, cases: list[dict], *, top_k: int | None = None) -> RetrievalEvalResult:
        effective_top_k = top_k or settings.eval_retrieval_top_k
        results = [self._evaluate_case(case, top_k=effective_top_k) for case in cases]

        hits = [r for r in results if r.hit]
        hit_rate = len(hits) / len(results) if results else 0.0
        mrr = (
            sum(1 / r.rank for r in results if r.rank) / len(results)
            if results else 0.0
        )
        return RetrievalEvalResult(cases=results, hit_rate=hit_rate, mean_reciprocal_rank=mrr)

    def _evaluate_case(self, case: dict, *, top_k: int) -> RetrievalCaseResult:
        query = case["query"]
        expected_source_id = case["expected_source_id"]
        expected_chunk_index = case.get("expected_chunk_index")

        retrieved = self.retriever.search(query, top_k=top_k, min_similarity=0.0)

        rank = None
        for i, r in enumerate(retrieved, start=1):
            if r.chunk.source.source_id == expected_source_id and (
                expected_chunk_index is None or r.chunk.chunk_index == expected_chunk_index
            ):
                rank = i
                break

        return RetrievalCaseResult(
            query=query,
            expected_source_id=expected_source_id,
            expected_chunk_index=expected_chunk_index,
            hit=rank is not None,
            rank=rank,
            top_results=[
                {
                    "source_id": r.chunk.source.source_id,
                    "chunk_index": r.chunk.chunk_index,
                    "similarity": r.similarity,
                }
                for r in retrieved
            ],
        )
