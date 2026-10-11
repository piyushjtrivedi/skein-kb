from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from src.database.connection import database
from src.retrieval.retriever import Retriever

router = APIRouter(prefix="/retrieval", tags=["retrieval"])


@router.get("/search")
def search(
    q: str,
    top_k: int | None = None,
    db: Session = Depends(database.get_session),
):
    retriever = Retriever(db)
    results = retriever.search(q, top_k=top_k)
    return [
        {
            "source_id": r.chunk.source.source_id,
            "source_name": r.chunk.source.name,
            "chunk_index": r.chunk.chunk_index,
            "text": r.chunk.text,
            "similarity": r.similarity,
        }
        for r in results
    ]
