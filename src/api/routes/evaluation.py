from typing import Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.chat.service import ChatService
from src.database.connection import database
from src.evaluation.citation_checker import CitationChecker
from src.evaluation.groundedness import GroundednessJudge
from src.evaluation.retrieval_eval import RetrievalEvaluator

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


class EvalChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class EvalChatRequest(BaseModel):
    question: str
    history: list[EvalChatMessage] = []
    top_k: int | None = None


@router.post("/chat")
def evaluate_chat(request: EvalChatRequest, db: Session = Depends(database.get_session)):
    service = ChatService(db)
    context, stream = service.answer(
        request.question,
        history=[m.model_dump() for m in request.history],
        top_k=request.top_k,
    )
    answer = "".join(stream)

    citation_result = CitationChecker.check(answer, context)
    groundedness_result = GroundednessJudge(llm_client=service.llm).judge(
        request.question, answer, context
    )

    return {
        "question": request.question,
        "answer": answer,
        "context_items": [
            {
                "source_id": item.source_id,
                "source_name": item.source_name,
                "chunk_index": item.chunk_index,
                "similarity": item.similarity,
            }
            for item in context.items
        ],
        "citation_check": {
            "cited_indices": citation_result.cited_indices,
            "invalid_citations": citation_result.invalid_citations,
            "uncited_context_items": citation_result.uncited_context_items,
            "all_valid": citation_result.all_valid,
        },
        "groundedness": {
            "grounded": groundedness_result.grounded,
            "unsupported_claims": groundedness_result.unsupported_claims,
        },
    }


class RetrievalEvalCase(BaseModel):
    query: str
    expected_source_id: str
    expected_chunk_index: int | None = None


class RetrievalEvalRequest(BaseModel):
    cases: list[RetrievalEvalCase]
    top_k: int | None = None


@router.post("/retrieval")
def evaluate_retrieval(request: RetrievalEvalRequest, db: Session = Depends(database.get_session)):
    evaluator = RetrievalEvaluator(db)
    result = evaluator.evaluate(
        [c.model_dump() for c in request.cases], top_k=request.top_k
    )
    return {
        "hit_rate": result.hit_rate,
        "mean_reciprocal_rank": result.mean_reciprocal_rank,
        "cases": [
            {
                "query": c.query,
                "expected_source_id": c.expected_source_id,
                "expected_chunk_index": c.expected_chunk_index,
                "hit": c.hit,
                "rank": c.rank,
                "top_results": c.top_results,
            }
            for c in result.cases
        ],
    }
