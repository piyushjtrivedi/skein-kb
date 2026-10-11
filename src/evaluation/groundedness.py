from dataclasses import dataclass, field

from src.context_engineering.assembler import AssembledContext
from src.llm import get_llm_client
from src.llm.base import LLMClient


@dataclass
class GroundednessResult:
    grounded: bool
    unsupported_claims: list[str] = field(default_factory=list)


class GroundednessJudge:
    """LLM-as-judge: is a generated answer actually supported by the
    context it was given? Complements CitationChecker, which only checks
    that citation markers are structurally valid, not that the claim next
    to one is actually true. Uses the same local LLM as everything else —
    this is explicitly a "does this model's own output hold up" check,
    not independent verification against a stronger model.
    """

    def __init__(self, llm_client: LLMClient | None = None):
        self.llm = llm_client or get_llm_client()

    def judge(self, question: str, answer: str, context: AssembledContext) -> GroundednessResult:
        context_items = [
            {"index": i, "source_name": item.source_name, "text": item.text}
            for i, item in enumerate(context.items, start=1)
        ]
        result = self.llm.judge_groundedness(question, answer, context_items)
        return GroundednessResult(
            grounded=bool(result.get("grounded")),
            unsupported_claims=result.get("unsupported_claims") or [],
        )
