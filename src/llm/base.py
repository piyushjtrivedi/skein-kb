from abc import ABC, abstractmethod
from collections.abc import Iterator


class LLMClient(ABC):
    """Provider-agnostic interface for extraction, embedding, and
    relationship classification.

    Implementations translate text into extracted items, vectors, or
    relationship judgments. Swapping providers (local model, hosted API)
    means writing one new subclass — nothing else in the extraction or
    linking pipelines depends on how the text is actually processed.
    """

    @abstractmethod
    def extract_concepts(self, text: str) -> list[dict]:
        """Return extracted items as dicts with keys: kind, name,
        description, and optionally quote."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text, same order."""

    @abstractmethod
    def classify_relationship(self, concept_a: dict, concept_b: dict) -> dict | None:
        """Given two concepts (each a dict with name/description), return
        {"relationship": ..., "rationale": ...} if a meaningful directed
        relationship exists from concept_a to concept_b, else None."""

    @abstractmethod
    def stream_chat(self, messages: list[dict[str, str]]) -> Iterator[str]:
        """Given a list of {"role": "system"|"user"|"assistant", "content":
        ...} messages, yield the assistant's reply as a sequence of text
        chunks, in the order generated."""

    @abstractmethod
    def judge_groundedness(
        self, question: str, answer: str, context_items: list[dict]
    ) -> dict:
        """Given a question, a generated answer, and the numbered context
        items (each a dict with index/source_name/text) it was supposedly
        grounded in, return {"grounded": bool, "unsupported_claims":
        list[str]}."""
