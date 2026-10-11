from pathlib import Path
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.core.config import settings
from src.database.models.source import Source
from src.database.models.source_version import SourceVersion

SOURCE_TYPE_SUBDIR = {
    "pdf": "pdf",
    "handwriting": "handwriting",
    "web": "web",
}


class SourceRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_by_hash(self, content_hash: str) -> Source | None:
        return self.db.scalar(
            select(Source).where(Source.content_hash == content_hash)
        )

    def get_by_hash_anywhere(self, content_hash: str) -> Source | None:
        """Find a source whose CURRENT or any PAST version matches this hash."""
        current = self.get_by_hash(content_hash)
        if current is not None:
            return current

        version = self.db.scalar(
            select(SourceVersion).where(SourceVersion.content_hash == content_hash)
        )
        if version is None:
            return None

        return self.db.get(Source, version.source_row_id)

    def get_by_id(self, source_id: str) -> Source | None:
        return self.db.scalar(
            select(Source).where(Source.source_id == source_id)
        )

    def get_web_source_by_url(self, url: str) -> Source | None:
        return self.db.scalar(
            select(Source).where(
                Source.source_type == "web",
                Source.original_filename == url,
            )
        )

    def list_all(self) -> list[Source]:
        return list(
            self.db.scalars(
                select(Source).order_by(Source.created_at.desc())
            ).all()
        )

    def list_versions(self, source: Source) -> list[SourceVersion]:
        return list(
            self.db.scalars(
                select(SourceVersion)
                .where(SourceVersion.source_row_id == source.id)
                .order_by(SourceVersion.version)
            ).all()
        )

    def create(
        self,
        *,
        source_type: str,
        name: str,
        original_filename: str,
        extension: str,
        content: bytes,
        content_hash: str,
        extracted_text: str | None = None,
        status: str = "registered",
    ) -> Source:
        source_id = str(uuid4())
        destination = self._version_path(
            source_type=source_type,
            source_id=source_id,
            version=1,
            extension=extension,
        )
        destination.write_bytes(content)

        source = Source(
            source_id=source_id,
            name=name,
            source_type=source_type,
            original_filename=original_filename,
            content_hash=content_hash,
            file_path=str(destination),
            version=1,
            status=status,
            extracted_text=extracted_text,
        )

        self.db.add(source)
        self.db.commit()
        self.db.refresh(source)

        return source

    def create_version(
        self,
        source: Source,
        *,
        content: bytes,
        content_hash: str,
        extension: str,
        original_filename: str | None = None,
        extracted_text: str | None = None,
    ) -> Source:
        """Supersede `source`'s current content with a new version, archiving
        the previous content_hash/file_path as a SourceVersion row.

        `original_filename` is only passed for re-uploaded files (pdf/image),
        where a new upload may carry a different filename. Web sources keep
        their original URL across versions, so it's left unchanged when
        omitted.
        """
        snapshot = SourceVersion(
            source_row_id=source.id,
            version=source.version,
            content_hash=source.content_hash,
            file_path=source.file_path,
            original_filename=source.original_filename,
            created_at=source.updated_at,
        )
        self.db.add(snapshot)

        new_version = source.version + 1
        destination = self._version_path(
            source_type=source.source_type,
            source_id=source.source_id,
            version=new_version,
            extension=extension,
        )
        destination.write_bytes(content)

        source.content_hash = content_hash
        source.file_path = str(destination)
        if original_filename is not None:
            source.original_filename = original_filename
        source.version = new_version
        if extracted_text is not None:
            source.extracted_text = extracted_text

        self.db.commit()
        self.db.refresh(source)

        return source

    @staticmethod
    def _version_path(*, source_type: str, source_id: str, version: int, extension: str) -> Path:
        subdir = SOURCE_TYPE_SUBDIR[source_type]
        directory = settings.source_dir / subdir
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{source_id}_v{version}{extension}"

    @staticmethod
    def to_dict(source: Source) -> dict:
        return {
            "source_id": source.source_id,
            "name": source.name,
            "source_type": source.source_type,
            "original_filename": source.original_filename,
            "content_hash": source.content_hash,
            "extracted_text": source.extracted_text,
            "version": source.version,
            "status": source.status,
            "file_path": source.file_path,
            "created_at": source.created_at,
            "updated_at": source.updated_at,
        }

    @staticmethod
    def version_to_dict(version: SourceVersion) -> dict:
        return {
            "version": version.version,
            "content_hash": version.content_hash,
            "original_filename": version.original_filename,
            "file_path": version.file_path,
            "created_at": version.created_at,
        }
