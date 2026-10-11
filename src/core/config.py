from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "sqlite:///./data/metadata/context_engine.db"
    source_dir: Path = Path("./data/sources")
    processed_dir: Path = Path("./data/processed")
    metadata_dir: Path = Path("./data/metadata")

    llm_provider: str = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b-instruct"

    chunk_size: int = 1500
    chunk_overlap: int = 200

    embedding_model: str = "nomic-embed-text"
    embedding_batch_size: int = 100
    link_top_k: int = 5
    link_min_similarity: float = 0.5
    link_max_per_node: int = 3
    link_exclude_kinds: tuple[str, ...] = ("entity",)
    merge_duplicate_name_similarity: float = 0.97

    retrieval_top_k: int = 5
    retrieval_min_similarity: float = 0.3

    eval_retrieval_top_k: int = 20

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def ensure_directories(self) -> None:
        for directory in (self.source_dir, self.processed_dir, self.metadata_dir):
            directory.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_directories()
