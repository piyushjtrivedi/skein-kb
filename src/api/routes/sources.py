from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel, HttpUrl
from sqlalchemy.orm import Session

from src.database.connection import database
from src.database.repositories.concept_repository import ConceptRepository
from src.database.repositories.source_repository import SourceRepository
from src.ingestion.pipeline import CONTENT_TYPE_SOURCE_TYPES, IngestionPipeline
from src.processing.extractor import ConceptExtractor

router = APIRouter(prefix="/sources", tags=["sources"])

VERSION_CONTENT_TYPES = {
    "pdf": {"application/pdf"},
    "handwriting": {"image/png", "image/jpeg", "image/jpg"},
}


class WebSourceRequest(BaseModel):
    url: HttpUrl


@router.post("", status_code=status.HTTP_201_CREATED)
async def upload_source(
    file: UploadFile = File(...),
    db: Session = Depends(database.get_session),
):
    if file.content_type not in CONTENT_TYPE_SOURCE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported file type. Supported: PDF, PNG, JPEG.",
        )

    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    pipeline = IngestionPipeline(db)
    try:
        result = pipeline.ingest(
            filename=file.filename,
            content=content,
            content_type=file.content_type,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    response = SourceRepository.to_dict(result.source)
    response["duplicate"] = result.is_duplicate
    return response


@router.post("/web", status_code=status.HTTP_201_CREATED)
def upload_web_source(
    payload: WebSourceRequest,
    db: Session = Depends(database.get_session),
):
    pipeline = IngestionPipeline(db)
    try:
        result = pipeline.ingest_web(url=str(payload.url))
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    response = SourceRepository.to_dict(result.source)
    response["duplicate"] = result.is_duplicate
    response["is_new_version"] = result.is_new_version
    response["previous_version"] = result.previous_version
    return response


@router.get("")
def list_sources(db: Session = Depends(database.get_session)):
    repo = SourceRepository(db)
    return [SourceRepository.to_dict(source) for source in repo.list_all()]


@router.get("/{source_id}")
def get_source(source_id: str, db: Session = Depends(database.get_session)):
    repo = SourceRepository(db)
    source = repo.get_by_id(source_id)
    if source is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Source not found.",
        )
    return SourceRepository.to_dict(source)


@router.post("/{source_id}/versions", status_code=status.HTTP_201_CREATED)
async def upload_new_version(
    source_id: str,
    file: UploadFile = File(...),
    db: Session = Depends(database.get_session),
):
    repo = SourceRepository(db)
    source = repo.get_by_id(source_id)
    if source is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Source not found.",
        )

    allowed_content_types = VERSION_CONTENT_TYPES.get(source.source_type, set())
    if file.content_type not in allowed_content_types:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Source '{source_id}' is type '{source.source_type}'; "
                f"uploaded file type '{file.content_type}' does not match."
            ),
        )

    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    pipeline = IngestionPipeline(db)
    try:
        result = pipeline.ingest_new_version(source, filename=file.filename, content=content)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    response = SourceRepository.to_dict(result.source)
    response["duplicate"] = result.is_duplicate
    response["is_new_version"] = result.is_new_version
    response["previous_version"] = result.previous_version
    return response


@router.get("/{source_id}/versions")
def list_versions(source_id: str, db: Session = Depends(database.get_session)):
    repo = SourceRepository(db)
    source = repo.get_by_id(source_id)
    if source is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Source not found.",
        )
    return [SourceRepository.version_to_dict(v) for v in repo.list_versions(source)]


@router.post("/{source_id}/extract", status_code=status.HTTP_201_CREATED)
def extract_concepts(
    source_id: str,
    force: bool = False,
    db: Session = Depends(database.get_session),
):
    repo = SourceRepository(db)
    source = repo.get_by_id(source_id)
    if source is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Source not found.",
        )

    extractor = ConceptExtractor(db)
    try:
        result = extractor.extract(source, force=force)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    return {
        "source_id": source.source_id,
        "chunk_count": len(result.chunks),
        "concept_count": len(result.concepts),
        "newly_extracted": result.newly_extracted,
        "failed_chunks": result.failed_chunks,
        "concepts": [ConceptRepository.to_dict(c) for c in result.concepts],
    }


@router.get("/{source_id}/concepts")
def list_concepts(source_id: str, db: Session = Depends(database.get_session)):
    repo = SourceRepository(db)
    source = repo.get_by_id(source_id)
    if source is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Source not found.",
        )

    concept_repo = ConceptRepository(db)
    return [ConceptRepository.to_dict(c) for c in concept_repo.list_for_source(source)]
