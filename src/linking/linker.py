import json
import logging
import math
from dataclasses import dataclass

from sqlalchemy.orm import Session

from src.core.config import settings
from src.database.models.concept import Concept
from src.database.models.concept_link import ALLOWED_RELATIONSHIPS
from src.database.repositories.concept_link_repository import ConceptLinkRepository
from src.database.repositories.concept_repository import ConceptRepository
from src.llm import get_llm_client
from src.llm.base import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class LinkingResult:
    concepts_considered: int
    embeddings_computed: int
    candidate_pairs: int
    links_created: int
    pairs_skipped_existing: int
    pairs_skipped_no_relationship: int
    failed_pairs: int


class ConceptLinker:
    def __init__(self, db: Session, llm_client: LLMClient | None = None):
        self.concept_repo = ConceptRepository(db)
        self.link_repo = ConceptLinkRepository(db)
        self.llm = llm_client or get_llm_client()

    def link_all(self, *, force: bool = False) -> LinkingResult:
        concepts = [
            c for c in self.concept_repo.list_all()
            if c.kind not in settings.link_exclude_kinds
        ]
        if len(concepts) < 2:
            return LinkingResult(len(concepts), 0, 0, 0, 0, 0, 0)

        embeddings_computed = self._ensure_embeddings(concepts, force=force)

        vectors = {
            c.id: json.loads(c.embedding) for c in concepts if c.embedding
        }
        candidates = self._top_k_candidates(
            concepts, vectors,
            k=settings.link_top_k,
            min_similarity=settings.link_min_similarity,
        )

        links_created = 0
        skipped_existing = 0
        skipped_none = 0
        failed_pairs = 0
        total = len(candidates)

        for index, (concept_a, concept_b, similarity) in enumerate(candidates, start=1):
            if not force and self.link_repo.exists_between(concept_a.id, concept_b.id):
                skipped_existing += 1
                self._log_progress(index, total)
                continue

            if force:
                self.link_repo.delete_between(concept_a.id, concept_b.id)

            try:
                result = self.llm.classify_relationship(
                    {"name": concept_a.name, "description": concept_a.description},
                    {"name": concept_b.name, "description": concept_b.description},
                )
            except ValueError as exc:
                logger.warning(
                    "Relationship classification failed for %s <-> %s: %s",
                    concept_a.concept_id, concept_b.concept_id, exc,
                )
                failed_pairs += 1
                self._log_progress(index, total)
                continue

            if result is None or result.get("relationship") not in ALLOWED_RELATIONSHIPS:
                skipped_none += 1
                self._log_progress(index, total)
                continue

            from_concept, to_concept = (
                (concept_b, concept_a)
                if result.get("direction") == "b_to_a"
                else (concept_a, concept_b)
            )
            self.link_repo.create(
                from_concept=from_concept,
                to_concept=to_concept,
                relationship_type=result["relationship"],
                rationale=result.get("rationale"),
                similarity=similarity,
            )
            links_created += 1
            self._log_progress(index, total)

        self.link_repo.prune_to_top_n_per_node(settings.link_max_per_node)

        return LinkingResult(
            concepts_considered=len(concepts),
            embeddings_computed=embeddings_computed,
            candidate_pairs=total,
            links_created=links_created,
            pairs_skipped_existing=skipped_existing,
            pairs_skipped_no_relationship=skipped_none,
            failed_pairs=failed_pairs,
        )

    def _ensure_embeddings(self, concepts: list[Concept], *, force: bool) -> int:
        missing = [
            c for c in concepts
            if force or not c.embedding or c.embedding_model != settings.embedding_model
        ]
        if not missing:
            return 0

        texts = [f"{c.name}: {c.description}" for c in missing]
        vectors = self.llm.embed(texts)
        for concept, vector in zip(missing, vectors):
            self.concept_repo.update_embedding(concept, vector, settings.embedding_model)
        return len(missing)

    @staticmethod
    def _top_k_candidates(
        concepts: list[Concept], vectors: dict[int, list[float]], *, k: int, min_similarity: float
    ) -> list[tuple[Concept, Concept, float]]:
        by_id = {c.id: c for c in concepts}
        ids = [cid for cid in by_id if cid in vectors]

        neighbors: dict[int, list[tuple[int, float]]] = {cid: [] for cid in ids}
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a_id, b_id = ids[i], ids[j]
                similarity = ConceptLinker._cosine(vectors[a_id], vectors[b_id])
                if similarity >= min_similarity:
                    neighbors[a_id].append((b_id, similarity))
                    neighbors[b_id].append((a_id, similarity))

        seen_pairs: set[tuple[int, int]] = set()
        candidates: list[tuple[Concept, Concept, float]] = []
        for concept_id, sims in neighbors.items():
            sims.sort(key=lambda pair: pair[1], reverse=True)
            for other_id, similarity in sims[:k]:
                key = (min(concept_id, other_id), max(concept_id, other_id))
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
                candidates.append((by_id[key[0]], by_id[key[1]], similarity))

        candidates.sort(key=lambda item: item[2], reverse=True)
        return candidates

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    @staticmethod
    def _log_progress(index: int, total: int) -> None:
        percent_done = (index / total) * 100 if total else 100
        logger.info(
            "Linking progress | pair %d/%d done | %.0f%% complete, %.0f%% remaining",
            index, total, percent_done, 100 - percent_done,
        )
