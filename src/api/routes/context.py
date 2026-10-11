from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from src.context_engineering.assembler import ContextAssembler
from src.database.connection import database

router = APIRouter(prefix="/context", tags=["context"])


@router.get("")
def get_context(
    q: str,
    top_k: int | None = None,
    db: Session = Depends(database.get_session),
):
    assembler = ContextAssembler(db)
    return assembler.assemble(q, top_k=top_k).to_dict()
