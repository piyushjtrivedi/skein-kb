from collections.abc import Iterator

from sqlalchemy.orm import Session

from src.context_engineering.assembler import AssembledContext, ContextAssembler
from src.llm import get_llm_client
from src.llm.base import LLMClient

SYSTEM_PROMPT_TEMPLATE = """You are a research assistant that answers \
questions using ONLY the context below, retrieved from the user's own \
uploaded sources.

Rules:
- Base your answer strictly on the provided context. Do not use outside \
knowledge, even if you're confident in it.
- If the context doesn't contain enough information to answer, say so \
plainly instead of guessing.
- When you state something drawn from the context, cite it with the \
matching bracketed number, e.g. [1].
- Be concise and direct.

Context:
{context}
"""


class ChatService:
    """Grounded question-answering over the user's sources.

    Deliberately stateless for now: `history` is whatever the caller
    passes in on each call, never read from or written to the database.
    The core contract (messages in, token stream out) doesn't change if a
    later phase adds persisted conversations — only where `history` comes
    from would change, from "passed by the caller" to "loaded by
    conversation_id".
    """

    def __init__(self, db: Session, llm_client: LLMClient | None = None):
        self.assembler = ContextAssembler(db, llm_client=llm_client)
        self.llm = llm_client or get_llm_client()

    def answer(
        self,
        question: str,
        *,
        history: list[dict[str, str]] | None = None,
        top_k: int | None = None,
    ) -> tuple[AssembledContext, Iterator[str]]:
        context = self.assembler.assemble(question, top_k=top_k)
        messages = self._build_messages(question, context, history or [])
        return context, self.llm.stream_chat(messages)

    @staticmethod
    def _build_messages(
        question: str, context: AssembledContext, history: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        system = SYSTEM_PROMPT_TEMPLATE.format(
            context=ChatService._format_context(context)
        )
        messages = [{"role": "system", "content": system}]
        messages.extend(history)
        messages.append({"role": "user", "content": question})
        return messages

    @staticmethod
    def _format_context(context: AssembledContext) -> str:
        if not context.items:
            return "(No relevant material was found in the user's sources.)"
        return "\n\n".join(
            f"[{i}] Source: {item.source_name}\n{item.text}"
            for i, item in enumerate(context.items, start=1)
        )
