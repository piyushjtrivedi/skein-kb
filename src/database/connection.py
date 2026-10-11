from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from src.core.config import settings


class Base(DeclarativeBase):
    pass


class Database:
    def __init__(self, database_url: str):
        connect_args = (
            {"check_same_thread": False}
            if database_url.startswith("sqlite")
            else {}
        )

        self.engine = create_engine(database_url, connect_args=connect_args)
        self.session_factory = sessionmaker(
            bind=self.engine,
            autoflush=False,
            autocommit=False,
        )

    def get_session(self) -> Session:
        db = self.session_factory()
        try:
            yield db
        finally:
            db.close()

    def init_db(self) -> None:
        from src.database.models.chunk import Chunk  # noqa: F401
        from src.database.models.concept import Concept  # noqa: F401
        from src.database.models.concept_link import ConceptLink  # noqa: F401
        from src.database.models.concept_provenance import ConceptProvenance  # noqa: F401
        from src.database.models.source import Source  # noqa: F401
        from src.database.models.source_version import SourceVersion  # noqa: F401

        Base.metadata.create_all(bind=self.engine)


database = Database(settings.database_url)
