import json
from typing import Literal

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.chat.service import ChatService
from src.database.connection import database

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    question: str
    history: list[ChatMessage] = []
    top_k: int | None = None


@router.post("")
def chat(request: ChatRequest, db: Session = Depends(database.get_session)):
    service = ChatService(db)
    context, stream = service.answer(
        request.question,
        history=[m.model_dump() for m in request.history],
        top_k=request.top_k,
    )

    citations = [
        {
            "source_id": item.source_id,
            "source_name": item.source_name,
            "chunk_index": item.chunk_index,
            "similarity": item.similarity,
        }
        for item in context.items
    ]

    return StreamingResponse(
        stream,
        media_type="text/plain",
        headers={"X-Citations": json.dumps(citations)},
    )
