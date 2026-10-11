import hashlib
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import pypdfium2 as pdfium
import pytesseract
import requests
import trafilatura
from PIL import Image, UnidentifiedImageError
from sqlalchemy.orm import Session

from src.database.models.source import Source
from src.database.repositories.source_repository import SourceRepository

CONTENT_TYPE_SOURCE_TYPES = {
    "application/pdf": "pdf",
    "image/png": "handwriting",
    "image/jpeg": "handwriting",
    "image/jpg": "handwriting",
}


@dataclass
class IngestResult:
    source: Source
    is_duplicate: bool


@dataclass
class VersionResult:
    source: Source
    is_duplicate: bool
    is_new_version: bool
    previous_version: int


@dataclass
class WebIngestResult:
    source: Source
    is_duplicate: bool
    is_new_version: bool
    previous_version: int | None


class IngestionPipeline:
    def __init__(self, db: Session):
        self.repo = SourceRepository(db)

    @staticmethod
    def compute_content_hash(content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

    def ingest(self, *, filename: str, content: bytes, content_type: str) -> IngestResult:
        source_type = CONTENT_TYPE_SOURCE_TYPES[content_type]
        content_hash = self.compute_content_hash(content)

        existing = self.repo.get_by_hash_anywhere(content_hash)
        if existing is not None:
            return IngestResult(source=existing, is_duplicate=True)

        extracted_text = self._extract_text(source_type, content)
        status = "text_extracted" if extracted_text else "registered"

        source = self.repo.create(
            source_type=source_type,
            name=Path(filename).stem,
            original_filename=filename,
            extension=Path(filename).suffix,
            content=content,
            content_hash=content_hash,
            extracted_text=extracted_text,
            status=status,
        )
        return IngestResult(source=source, is_duplicate=False)

    def ingest_new_version(self, source: Source, *, filename: str, content: bytes) -> VersionResult:
        content_hash = self.compute_content_hash(content)
        previous_version = source.version

        if content_hash == source.content_hash:
            return VersionResult(
                source=source,
                is_duplicate=True,
                is_new_version=False,
                previous_version=previous_version,
            )

        existing_elsewhere = self.repo.get_by_hash_anywhere(content_hash)
        if existing_elsewhere is not None:
            return VersionResult(
                source=existing_elsewhere,
                is_duplicate=True,
                is_new_version=False,
                previous_version=previous_version,
            )

        extracted_text = self._extract_text(source.source_type, content)

        updated = self.repo.create_version(
            source,
            content=content,
            content_hash=content_hash,
            extension=Path(filename).suffix,
            original_filename=filename,
            extracted_text=extracted_text,
        )
        return VersionResult(
            source=updated,
            is_duplicate=False,
            is_new_version=True,
            previous_version=previous_version,
        )

    def ingest_web(self, *, url: str) -> WebIngestResult:
        text, title = self._fetch_and_extract(url)
        content_hash = self.compute_content_hash(text.encode("utf-8"))

        current = self.repo.get_web_source_by_url(url)

        if current is not None:
            previous_version = current.version

            if content_hash == current.content_hash:
                return WebIngestResult(
                    source=current,
                    is_duplicate=True,
                    is_new_version=False,
                    previous_version=previous_version,
                )

            existing_elsewhere = self.repo.get_by_hash_anywhere(content_hash)
            if existing_elsewhere is not None:
                return WebIngestResult(
                    source=existing_elsewhere,
                    is_duplicate=True,
                    is_new_version=False,
                    previous_version=previous_version,
                )

            updated = self.repo.create_version(
                current,
                content=text.encode("utf-8"),
                content_hash=content_hash,
                extension=".txt",
                extracted_text=text,
            )
            return WebIngestResult(
                source=updated,
                is_duplicate=False,
                is_new_version=True,
                previous_version=previous_version,
            )

        existing_elsewhere = self.repo.get_by_hash_anywhere(content_hash)
        if existing_elsewhere is not None:
            return WebIngestResult(
                source=existing_elsewhere,
                is_duplicate=True,
                is_new_version=False,
                previous_version=None,
            )

        source = self.repo.create(
            source_type="web",
            name=title or url,
            original_filename=url,
            extension=".txt",
            content=text.encode("utf-8"),
            content_hash=content_hash,
            extracted_text=text,
            status="text_extracted",
        )
        return WebIngestResult(
            source=source,
            is_duplicate=False,
            is_new_version=False,
            previous_version=None,
        )

    @classmethod
    def _extract_text(cls, source_type: str, content: bytes) -> str | None:
        if source_type == "handwriting":
            return cls._run_ocr(content)
        if source_type == "pdf":
            return cls._extract_pdf_text(content)
        return None

    @staticmethod
    def _run_ocr(content: bytes) -> str:
        try:
            image = Image.open(BytesIO(content))
        except UnidentifiedImageError as exc:
            raise ValueError("Uploaded file is not a valid image.") from exc

        return pytesseract.image_to_string(image).strip()

    @staticmethod
    def _extract_pdf_text(content: bytes) -> str | None:
        try:
            pdf = pdfium.PdfDocument(content)
        except pdfium.PdfiumError as exc:
            raise ValueError("Uploaded file is not a valid PDF.") from exc

        pages_text = []
        for page in pdf:
            text_page = page.get_textpage()
            pages_text.append(text_page.get_text_range())
            text_page.close()
            page.close()
        pdf.close()

        text = "\n\n".join(pages_text).strip()
        return text or None

    @staticmethod
    def _fetch_and_extract(url: str) -> tuple[str, str | None]:
        try:
            response = requests.get(
                url,
                timeout=15,
                headers={"User-Agent": "skein-kb/0.1 (personal research tool)"},
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise ValueError(f"Failed to fetch URL: {exc}") from exc

        text = trafilatura.extract(
            response.text,
            url=url,
            include_comments=False,
            include_tables=False,
        )
        if not text:
            raise ValueError("Could not extract article content from this URL.")

        metadata = trafilatura.extract_metadata(response.text, default_url=url)
        title = metadata.title if metadata else None

        return text.strip(), title
