from uuid import uuid4

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from src.database.models.concept import Concept
from src.database.models.concept_link import ConceptLink


class ConceptLinkRepository:
    def __init__(self, db: Session):
        self.db = db

    def exists_between(self, concept_a_row_id: int, concept_b_row_id: int) -> bool:
        return (
            self.db.scalar(
                select(ConceptLink).where(
                    or_(
                        (ConceptLink.from_concept_row_id == concept_a_row_id)
                        & (ConceptLink.to_concept_row_id == concept_b_row_id),
                        (ConceptLink.from_concept_row_id == concept_b_row_id)
                        & (ConceptLink.to_concept_row_id == concept_a_row_id),
                    )
                )
            )
            is not None
        )

    def delete_between(self, concept_a_row_id: int, concept_b_row_id: int) -> None:
        existing = self.db.scalars(
            select(ConceptLink).where(
                or_(
                    (ConceptLink.from_concept_row_id == concept_a_row_id)
                    & (ConceptLink.to_concept_row_id == concept_b_row_id),
                    (ConceptLink.from_concept_row_id == concept_b_row_id)
                    & (ConceptLink.to_concept_row_id == concept_a_row_id),
                )
            )
        ).all()
        for link in existing:
            self.db.delete(link)
        self.db.commit()

    def create(
        self,
        *,
        from_concept: Concept,
        to_concept: Concept,
        relationship_type: str,
        similarity: float,
        rationale: str | None = None,
    ) -> ConceptLink:
        link = ConceptLink(
            link_id=str(uuid4()),
            from_concept_row_id=from_concept.id,
            to_concept_row_id=to_concept.id,
            relationship_type=relationship_type,
            similarity=similarity,
            rationale=rationale,
        )
        self.db.add(link)
        self.db.commit()
        self.db.refresh(link)
        return link

    def prune_to_top_n_per_node(self, max_per_node: int) -> int:
        """Keeps only each concept's `max_per_node` strongest links (by
        similarity), dropping weaker ones -- run after a full linking pass
        so a node that was a top-k candidate for many others doesn't end up
        with dozens of edges and turn the graph into a hairball. A link
        counts against both endpoints, so a link already kept for one side
        can still be re-deleted if it falls outside the other side's top N;
        iterate until stable instead of a single pass.
        """
        deleted_total = 0
        while True:
            links = self.list_all()
            by_node: dict[int, list[ConceptLink]] = {}
            for link in links:
                by_node.setdefault(link.from_concept_row_id, []).append(link)
                by_node.setdefault(link.to_concept_row_id, []).append(link)

            to_delete: set[int] = set()
            for node_links in by_node.values():
                if len(node_links) <= max_per_node:
                    continue
                node_links.sort(key=lambda l: l.similarity, reverse=True)
                for weak_link in node_links[max_per_node:]:
                    to_delete.add(weak_link.id)

            if not to_delete:
                break
            for link in links:
                if link.id in to_delete:
                    self.db.delete(link)
            self.db.commit()
            deleted_total += len(to_delete)

        return deleted_total

    def list_all(self) -> list[ConceptLink]:
        return list(self.db.scalars(select(ConceptLink).order_by(ConceptLink.id)).all())

    def list_for_concept(self, concept: Concept) -> list[ConceptLink]:
        return list(
            self.db.scalars(
                select(ConceptLink)
                .where(
                    or_(
                        ConceptLink.from_concept_row_id == concept.id,
                        ConceptLink.to_concept_row_id == concept.id,
                    )
                )
                .order_by(ConceptLink.id)
            ).all()
        )

    @staticmethod
    def to_dict(link: ConceptLink) -> dict:
        return {
            "link_id": link.link_id,
            "from_concept_id": link.from_concept.concept_id,
            "to_concept_id": link.to_concept.concept_id,
            "relationship_type": link.relationship_type,
            "rationale": link.rationale,
            "similarity": link.similarity,
            "created_at": link.created_at,
        }
