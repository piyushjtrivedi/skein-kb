"""One-time data migration: drop bibliographic-junk entities, merge
near-duplicate concepts, backfill quote_verified, and clear stale links.

Run with the server stopped (no --reload process holding the same sqlite
file): `python3 -m scripts.migrate_concept_cleanup` from the project root.

After this finishes, re-run `POST /api/knowledge/link` against the
survivors to regenerate the graph under the new taxonomy/direction/
entity-exclusion/per-node-cap rules -- this script does NOT call the LLM
or re-link anything itself, it only cleans up existing rows.
"""

import json
import math
import re
import sys

from sqlalchemy import select, text
from sqlalchemy.orm import Session

sys.path.insert(0, ".")

from src.database.connection import database  # noqa: E402
from src.database.models.concept import Concept  # noqa: E402
from src.database.models.concept_link import ConceptLink  # noqa: E402
from src.database.models.concept_provenance import ConceptProvenance  # noqa: E402
from src.processing.extractor import is_bibliographic_junk, verify_quote  # noqa: E402


def normalize_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower()).strip(".,:;")


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def repoint_links(db: Session, old_id: int, new_id: int) -> None:
    links = list(
        db.scalars(
            select(ConceptLink).where(
                (ConceptLink.from_concept_row_id == old_id)
                | (ConceptLink.to_concept_row_id == old_id)
            )
        ).all()
    )
    for link in links:
        new_from = new_id if link.from_concept_row_id == old_id else link.from_concept_row_id
        new_to = new_id if link.to_concept_row_id == old_id else link.to_concept_row_id

        if new_from == new_to:
            db.delete(link)
            continue

        duplicate = db.scalar(
            select(ConceptLink).where(
                ConceptLink.id != link.id,
                (
                    ((ConceptLink.from_concept_row_id == new_from) & (ConceptLink.to_concept_row_id == new_to))
                    | ((ConceptLink.from_concept_row_id == new_to) & (ConceptLink.to_concept_row_id == new_from))
                ),
            )
        )
        if duplicate is not None:
            db.delete(link)
        else:
            link.from_concept_row_id = new_from
            link.to_concept_row_id = new_to
    db.commit()


def merge_into(db: Session, canonical: Concept, duplicate: Concept) -> None:
    existing_provenance = {
        p.chunk_row_id
        for p in db.scalars(
            select(ConceptProvenance).where(ConceptProvenance.concept_row_id == canonical.id)
        ).all()
    }
    for chunk_row_id in {canonical.chunk_row_id, duplicate.chunk_row_id} - existing_provenance:
        db.add(ConceptProvenance(concept_row_id=canonical.id, chunk_row_id=chunk_row_id))
    db.commit()

    repoint_links(db, duplicate.id, canonical.id)
    db.delete(duplicate)
    db.commit()


def pick_canonical(group: list[Concept]) -> Concept:
    return sorted(
        group,
        key=lambda c: (c.quote_verified, len(c.description or "")),
        reverse=True,
    )[0]


def ensure_quote_verified_column() -> None:
    """create_all() only creates missing TABLES, not missing columns on an
    existing one -- quote_verified is new on an already-populated
    `concepts` table, so it needs an explicit ALTER TABLE, once."""
    with database.engine.connect() as conn:
        columns = {row[1] for row in conn.execute(text("PRAGMA table_info(concepts)"))}
        if "quote_verified" not in columns:
            conn.execute(text("ALTER TABLE concepts ADD COLUMN quote_verified BOOLEAN NOT NULL DEFAULT 0"))
            conn.commit()
            print("Added quote_verified column to concepts table")


def main() -> None:
    ensure_quote_verified_column()
    database.init_db()
    db = next(database.get_session())

    all_concepts = list(db.scalars(select(Concept).order_by(Concept.id)).all())
    print(f"Starting concepts: {len(all_concepts)}")

    # 1. Backfill quote_verified for everything extracted before this flag existed.
    verified_count = 0
    for concept in all_concepts:
        verified = verify_quote(concept.quote, concept.chunk.text)
        if verified != concept.quote_verified:
            concept.quote_verified = verified
            verified_count += 1
    db.commit()
    print(f"quote_verified backfilled/changed on {verified_count} concepts")

    # 2. Drop bibliographic-junk entities.
    junk = [c for c in all_concepts if c.kind == "entity" and is_bibliographic_junk(c.name)]
    print(f"Deleting {len(junk)} bibliographic-junk entity concepts")
    for concept in junk:
        for link in db.scalars(
            select(ConceptLink).where(
                (ConceptLink.from_concept_row_id == concept.id)
                | (ConceptLink.to_concept_row_id == concept.id)
            )
        ).all():
            db.delete(link)
        db.delete(concept)
    db.commit()

    survivors = list(db.scalars(select(Concept).order_by(Concept.id)).all())
    print(f"Concepts after junk removal: {len(survivors)}")

    # 3. Merge exact normalized-name duplicates (same kind).
    by_key: dict[tuple[str, str], list[Concept]] = {}
    for concept in survivors:
        key = (concept.kind, normalize_name(concept.name))
        by_key.setdefault(key, []).append(concept)

    exact_merges = 0
    for group in by_key.values():
        if len(group) < 2:
            continue
        canonical = pick_canonical(group)
        for duplicate in group:
            if duplicate.id == canonical.id:
                continue
            merge_into(db, canonical, duplicate)
            exact_merges += 1
    print(f"Merged {exact_merges} exact-normalized-name duplicates")

    # 4. Merge cross-name near-duplicates via embedding similarity, within kind.
    survivors = list(db.scalars(select(Concept).order_by(Concept.id)).all())
    by_kind: dict[str, list[Concept]] = {}
    for concept in survivors:
        if concept.embedding:
            by_kind.setdefault(concept.kind, []).append(concept)

    threshold = 0.97
    embedding_merges = 0
    for kind, group in by_kind.items():
        vectors = {c.id: json.loads(c.embedding) for c in group}
        merged_away: set[int] = set()
        ids = [c.id for c in group]
        by_id = {c.id: c for c in group}
        for i in range(len(ids)):
            a_id = ids[i]
            if a_id in merged_away:
                continue
            for j in range(i + 1, len(ids)):
                b_id = ids[j]
                if b_id in merged_away:
                    continue
                if cosine(vectors[a_id], vectors[b_id]) >= threshold:
                    pair = [by_id[a_id], by_id[b_id]]
                    canonical = pick_canonical(pair)
                    duplicate = pair[0] if pair[1].id == canonical.id else pair[1]
                    merge_into(db, canonical, duplicate)
                    merged_away.add(duplicate.id)
                    embedding_merges += 1
    print(f"Merged {embedding_merges} cross-name near-duplicates (kind-scoped, similarity >= {threshold})")

    # 5. Clear all links -- taxonomy/direction/entity-exclusion/cap rules all
    # changed, and the survivor set has shifted; re-link from scratch via the
    # API afterward rather than trying to patch stale edges in place.
    all_links = list(db.scalars(select(ConceptLink)).all())
    for link in all_links:
        db.delete(link)
    db.commit()
    print(f"Cleared {len(all_links)} links (re-run POST /api/knowledge/link to regenerate)")

    final_concepts = list(db.scalars(select(Concept)).all())
    print(f"Final concepts: {len(final_concepts)}")
    db.close()


if __name__ == "__main__":
    main()
