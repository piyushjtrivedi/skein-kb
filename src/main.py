from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, RedirectResponse

from src.api.routes.chat import router as chat_router
from src.api.routes.context import router as context_router
from src.api.routes.evaluation import router as evaluation_router
from src.api.routes.knowledge import router as knowledge_router
from src.api.routes.retrieval import router as retrieval_router
from src.api.routes.sources import router as sources_router
from src.core.logging import LoggingConfigurator
from src.database.connection import database

LoggingConfigurator.configure()

app = FastAPI(
    title="Scientific Context Engine",
    version="0.1.0",
    description=(
        "Multi-format source ingestion (PDF, handwriting OCR, web articles) with "
        "versioning, local-LLM concept/equation/entity extraction, a "
        "concept-to-concept knowledge graph, embedding-based retrieval with "
        "citable context assembly, a streaming chat endpoint grounded in "
        "the user's own sources, and evaluation checks for citation "
        "accuracy, groundedness, and retrieval quality."
    ),
)


@app.on_event("startup")
def startup() -> None:
    database.init_db()


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


@app.get("/graph", include_in_schema=False)
def graph_page() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "graph.html")


app.include_router(sources_router, prefix="/api")
app.include_router(knowledge_router, prefix="/api")
app.include_router(retrieval_router, prefix="/api")
app.include_router(context_router, prefix="/api")
app.include_router(chat_router, prefix="/api")
app.include_router(evaluation_router, prefix="/api")