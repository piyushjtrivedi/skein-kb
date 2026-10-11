from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from src.database.connection import database
from src.database.repositories.concept_link_repository import ConceptLinkRepository
from src.database.repositories.concept_repository import ConceptRepository
from src.linking.linker import ConceptLinker

router = APIRouter(prefix="/knowledge", tags=["knowledge"])


@router.post("/link", status_code=status.HTTP_201_CREATED)
def link_concepts(force: bool = False, db: Session = Depends(database.get_session)):
    linker = ConceptLinker(db)
    result = linker.link_all(force=force)
    return {
        "concepts_considered": result.concepts_considered,
        "embeddings_computed": result.embeddings_computed,
        "candidate_pairs": result.candidate_pairs,
        "links_created": result.links_created,
        "pairs_skipped_existing": result.pairs_skipped_existing,
        "pairs_skipped_no_relationship": result.pairs_skipped_no_relationship,
        "failed_pairs": result.failed_pairs,
    }


@router.get("/graph")
def get_graph(db: Session = Depends(database.get_session)):
    concept_repo = ConceptRepository(db)
    link_repo = ConceptLinkRepository(db)
    return {
        "nodes": [ConceptRepository.to_dict(c) for c in concept_repo.list_all()],
        "edges": [ConceptLinkRepository.to_dict(link) for link in link_repo.list_all()],
    }


@router.get("/concepts/{concept_id}/links")
def get_concept_links(concept_id: str, db: Session = Depends(database.get_session)):
    concept_repo = ConceptRepository(db)
    concept = concept_repo.get_by_concept_id(concept_id)
    if concept is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Concept not found.",
        )

    link_repo = ConceptLinkRepository(db)
    return [ConceptLinkRepository.to_dict(link) for link in link_repo.list_for_concept(concept)]
