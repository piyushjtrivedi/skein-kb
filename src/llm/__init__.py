from src.core.config import settings
from src.llm.base import LLMClient
from src.llm.ollama_client import OllamaClient


def get_llm_client() -> LLMClient:
    if settings.llm_provider == "ollama":
        return OllamaClient()
    raise ValueError(f"Unknown LLM provider: {settings.llm_provider!r}")


__all__ = ["LLMClient", "OllamaClient", "get_llm_client"]
