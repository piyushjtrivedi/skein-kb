import re
from dataclasses import dataclass

from src.context_engineering.assembler import AssembledContext

CITATION_PATTERN = re.compile(r"\[(\d+)\]")


@dataclass
class CitationCheckResult:
    cited_indices: list[int]
    invalid_citations: list[int]
    uncited_context_items: list[int]
    all_valid: bool


class CitationChecker:
    """Structural check: does every citation marker in an answer (`[1]`,
    `[2]`, ...) actually refer to a context item that was retrieved?

    Deliberately has no LLM involved — it checks the mechanics of citing
    (is this a real reference), not whether the claim beside it is true
    (that's GroundednessJudge's job). Cheap and deterministic, so it's
    worth running on every answer regardless of whether the slower
    LLM-judged groundedness check also runs.
    """

    @staticmethod
    def check(answer: str, context: AssembledContext) -> CitationCheckResult:
        cited = sorted({int(n) for n in CITATION_PATTERN.findall(answer)})
        valid_range = range(1, len(context.items) + 1)
        invalid = [n for n in cited if n not in valid_range]
        uncited = [n for n in valid_range if n not in cited]
        return CitationCheckResult(
            cited_indices=cited,
            invalid_citations=invalid,
            uncited_context_items=uncited,
            all_valid=not invalid,
        )
