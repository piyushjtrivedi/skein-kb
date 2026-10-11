import json
import logging
import time
from collections.abc import Iterator

import ollama

from src.core.config import settings
from src.llm.base import LLMClient

logger = logging.getLogger(__name__)

EXTRACTION_SYSTEM_PROMPT = """You are a scientific document analyst. Given a \
chunk of text from a scientific source, extract the distinct concepts, \
equations, and named entities it substantively discusses.

For each item found, output an object with:
- "kind": one of "concept", "equation", or "entity"
  - "concept": a scientific idea, principle, method, or definition
  - "equation": a mathematical formula or relationship (put the formula \
itself, in plain text/LaTeX-like notation, in "name", e.g. "E = mc^2")
  - "entity": a named person, organization, dataset, or specific \
software/hardware system that the text substantively discusses or depends \
on (e.g. "Google", "ImageNet", "TPU v3") -- NOT a passing mention
- "name": a short name or label (a few words, or the formula for kind="equation")
- "description": one to two sentences explaining it, grounded only in this text
- "quote": a short verbatim excerpt (at most ~200 characters) from the text \
supporting this item

Do NOT extract: paper/author citations ("Vaswani et al., 2017"), reference \
list entries, figure/table/section labels ("Figure 2", "Table 3", "Section \
4.1"), bare years or dates, conference/journal names ("NeurIPS 2017", \
"ICLR"), or page numbers. These are bibliographic scaffolding, not content.

Only extract items that are actually substantively discussed, not passing \
mentions. If the text has no extractable scientific content, return an \
empty list.

Respond with ONLY a JSON object of the form {"items": [...]}. No other \
text, no markdown code fences.
"""

RELATIONSHIP_SYSTEM_PROMPT = """You are a scientific knowledge-graph \
curator. Given two concepts, A and B (each with a name and description, \
possibly from different documents), decide whether there is a meaningful \
relationship between them, and if so, which one and which direction it \
points.

Relationship types (pick exactly one, or none):
- "same_as": A and B refer to the same real-world thing, just named or \
phrased differently (this is symmetric -- direction doesn't matter)
- "is_a": one is a specific type/instance of the other (e.g. "ResNet-50" \
is_a "ResNet")
- "part_of": one is a component or sub-part of the other
- "prerequisite": understanding one is necessary before the other makes sense
- "derivation": one is mathematically/logically derived from the other
- "application": one is a practical application of the other
- "related": they are meaningfully connected but none of the above fits \
precisely

For "is_a", "part_of", "prerequisite", "derivation", and "application", \
also decide the direction: does it point from A to B, or from B to A? \
(e.g. for prerequisite, which one must be understood FIRST is the \
"from" side.) For "same_as" and "related", direction doesn't matter -- \
always use "a_to_b".

If A and B are not meaningfully related at all (e.g. just superficially \
similar wording, or unrelated topics), say so.

Respond with ONLY a JSON object of one of these two forms:
{"relationship": "same_as" | "is_a" | "part_of" | "prerequisite" | \
"derivation" | "application" | "related", "direction": "a_to_b" | \
"b_to_a", "rationale": "<one sentence>"}
or, if there is no meaningful relationship:
{"relationship": null}
No other text, no markdown code fences.
"""

GROUNDEDNESS_SYSTEM_PROMPT = """You are a strict fact-checker. You will \
be given a QUESTION, a set of numbered CONTEXT passages, and an ANSWER \
that was supposedly generated using only that context.

Decide whether the ANSWER is fully supported by the CONTEXT alone.

Respond with ONLY a JSON object of the form:
{"grounded": true | false, "unsupported_claims": ["<claim>", ...]}

Rules:
- "grounded" is true only if every factual claim in the ANSWER is \
directly supported by the CONTEXT.
- If the ANSWER says the context doesn't contain enough information \
(and that's correct), that counts as grounded=true.
- List each unsupported claim (verbatim or closely paraphrased) in \
"unsupported_claims". Use an empty list if grounded is true.
No other text, no markdown code fences.
"""


class OllamaClient(LLMClient):
    def __init__(self, base_url: str | None = None, model: str | None = None):
        self.model = model or settings.ollama_model
        self.base_url = base_url or settings.ollama_base_url
        self._client = ollama.Client(host=self.base_url)

    def extract_concepts(self, text: str) -> list[dict]:
        wall_start = time.monotonic()
        try:
            response = self._client.generate(
                model=self.model,
                system=EXTRACTION_SYSTEM_PROMPT,
                prompt=text,
                format="json",
                options={"temperature": 0.1},
            )
        except ollama.ResponseError as exc:
            logger.warning(
                "Ollama call failed | model=%s | input_chars=%d | error=%s",
                self.model, len(text), exc,
            )
            raise ValueError(
                f"Ollama request failed (is `ollama serve` running and is "
                f"'{self.model}' pulled?): {exc}"
            ) from exc
        except OSError as exc:
            logger.warning(
                "Could not reach Ollama | model=%s | base_url=%s | error=%s",
                self.model, self.base_url, exc,
            )
            raise ValueError(
                f"Could not reach Ollama at {self.base_url}: {exc}"
            ) from exc
        wall_time = time.monotonic() - wall_start

        payload = response if isinstance(response, dict) else response.model_dump()
        self._log_usage(
            "extract_concepts", payload, input_chars=len(text), wall_time=wall_time
        )

        raw = payload["response"]

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Model did not return valid JSON: {exc}") from exc

        if isinstance(parsed, list):
            items = parsed
        elif isinstance(parsed, dict):
            items = parsed.get("items")
            if items is None:
                logger.warning(
                    "Model response had no 'items' key, treating as empty: %r",
                    parsed,
                )
                items = []
        else:
            raise ValueError(f"Unexpected JSON shape from model: {parsed!r}")

        if not isinstance(items, list):
            raise ValueError("Expected a JSON list of extracted items under 'items'.")

        return items

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embeds in batches of settings.embedding_batch_size, not all at
        once. A single large Ollama /api/embed call (observed: 824 real
        concept strings, ~81KB of text) can crash the model's runner
        subprocess (EOF from its internal /tokenize call) even though the
        same texts split into two halves of 412 each succeed individually
        -- this keeps each request comfortably under that cliff."""
        if not texts:
            return []

        batch_size = settings.embedding_batch_size
        if len(texts) <= batch_size:
            return self._embed_batch(texts)

        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            vectors.extend(self._embed_batch(texts[start:start + batch_size]))
        return vectors

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        wall_start = time.monotonic()
        try:
            response = self._client.embed(model=settings.embedding_model, input=texts)
        except ollama.ResponseError as exc:
            logger.warning(
                "Ollama embed call failed | model=%s | count=%d | error=%s",
                settings.embedding_model, len(texts), exc,
            )
            raise ValueError(
                f"Ollama embedding request failed (is '{settings.embedding_model}' "
                f"pulled?): {exc}"
            ) from exc
        except OSError as exc:
            logger.warning(
                "Could not reach Ollama | model=%s | base_url=%s | error=%s",
                settings.embedding_model, self.base_url, exc,
            )
            raise ValueError(
                f"Could not reach Ollama at {self.base_url}: {exc}"
            ) from exc
        wall_time = time.monotonic() - wall_start

        payload = response if isinstance(response, dict) else response.model_dump()
        self._log_usage(
            "embed", payload,
            input_chars=sum(len(t) for t in texts),
            wall_time=wall_time,
            count=len(texts),
        )

        return payload["embeddings"]

    def classify_relationship(self, concept_a: dict, concept_b: dict) -> dict | None:
        prompt = (
            f"Concept A: {concept_a['name']}\n{concept_a['description']}\n\n"
            f"Concept B: {concept_b['name']}\n{concept_b['description']}"
        )
        wall_start = time.monotonic()
        try:
            response = self._client.generate(
                model=self.model,
                system=RELATIONSHIP_SYSTEM_PROMPT,
                prompt=prompt,
                format="json",
                options={"temperature": 0.1},
            )
        except ollama.ResponseError as exc:
            logger.warning(
                "Ollama call failed | model=%s | error=%s", self.model, exc,
            )
            raise ValueError(
                f"Ollama request failed (is `ollama serve` running and is "
                f"'{self.model}' pulled?): {exc}"
            ) from exc
        except OSError as exc:
            logger.warning(
                "Could not reach Ollama | model=%s | base_url=%s | error=%s",
                self.model, self.base_url, exc,
            )
            raise ValueError(
                f"Could not reach Ollama at {self.base_url}: {exc}"
            ) from exc
        wall_time = time.monotonic() - wall_start

        payload = response if isinstance(response, dict) else response.model_dump()
        self._log_usage(
            "classify_relationship", payload, input_chars=len(prompt), wall_time=wall_time
        )

        raw = payload["response"]
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Model did not return valid JSON: {exc}") from exc

        if not isinstance(parsed, dict):
            raise ValueError(f"Unexpected JSON shape from model: {parsed!r}")

        relationship = parsed.get("relationship")
        if not relationship:
            return None

        return {
            "relationship": relationship,
            "direction": parsed.get("direction") if parsed.get("direction") == "b_to_a" else "a_to_b",
            "rationale": parsed.get("rationale"),
        }

    def stream_chat(self, messages: list[dict[str, str]]) -> Iterator[str]:
        wall_start = time.monotonic()
        input_chars = sum(len(m.get("content", "")) for m in messages)
        final_payload: dict | None = None
        try:
            stream = self._client.chat(
                model=self.model,
                messages=messages,
                stream=True,
                options={"temperature": 0.2},
            )
            for chunk in stream:
                payload = chunk if isinstance(chunk, dict) else chunk.model_dump()
                content = payload.get("message", {}).get("content", "")
                if content:
                    yield content
                if payload.get("done"):
                    final_payload = payload
        except ollama.ResponseError as exc:
            logger.warning(
                "Ollama chat call failed | model=%s | error=%s", self.model, exc,
            )
            raise ValueError(
                f"Ollama request failed (is `ollama serve` running and is "
                f"'{self.model}' pulled?): {exc}"
            ) from exc
        except OSError as exc:
            logger.warning(
                "Could not reach Ollama | model=%s | base_url=%s | error=%s",
                self.model, self.base_url, exc,
            )
            raise ValueError(
                f"Could not reach Ollama at {self.base_url}: {exc}"
            ) from exc

        wall_time = time.monotonic() - wall_start
        if final_payload:
            self._log_usage(
                "stream_chat", final_payload, input_chars=input_chars, wall_time=wall_time
            )

    def judge_groundedness(
        self, question: str, answer: str, context_items: list[dict]
    ) -> dict:
        context_text = "\n\n".join(
            f"[{item['index']}] {item['source_name']}: {item['text']}"
            for item in context_items
        ) or "(no context was provided)"
        prompt = f"QUESTION:\n{question}\n\nCONTEXT:\n{context_text}\n\nANSWER:\n{answer}"

        wall_start = time.monotonic()
        try:
            response = self._client.generate(
                model=self.model,
                system=GROUNDEDNESS_SYSTEM_PROMPT,
                prompt=prompt,
                format="json",
                options={"temperature": 0.1},
            )
        except ollama.ResponseError as exc:
            logger.warning(
                "Ollama call failed | model=%s | error=%s", self.model, exc,
            )
            raise ValueError(
                f"Ollama request failed (is `ollama serve` running and is "
                f"'{self.model}' pulled?): {exc}"
            ) from exc
        except OSError as exc:
            logger.warning(
                "Could not reach Ollama | model=%s | base_url=%s | error=%s",
                self.model, self.base_url, exc,
            )
            raise ValueError(
                f"Could not reach Ollama at {self.base_url}: {exc}"
            ) from exc
        wall_time = time.monotonic() - wall_start

        payload = response if isinstance(response, dict) else response.model_dump()
        self._log_usage(
            "judge_groundedness", payload, input_chars=len(prompt), wall_time=wall_time
        )

        raw = payload["response"]
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Model did not return valid JSON: {exc}") from exc

        if not isinstance(parsed, dict):
            raise ValueError(f"Unexpected JSON shape from model: {parsed!r}")

        return {
            "grounded": bool(parsed.get("grounded", False)),
            "unsupported_claims": parsed.get("unsupported_claims") or [],
        }

    @staticmethod
    def _log_usage(
        call: str, payload: dict, *, input_chars: int, wall_time: float, count: int = 1
    ) -> None:
        prompt_tokens = payload.get("prompt_eval_count")
        completion_tokens = payload.get("eval_count")
        total_duration_s = (payload.get("total_duration") or 0) / 1e9
        load_duration_s = (payload.get("load_duration") or 0) / 1e9
        prompt_eval_duration_s = (payload.get("prompt_eval_duration") or 0) / 1e9
        eval_duration_s = (payload.get("eval_duration") or 0) / 1e9
        tokens_per_sec = (
            completion_tokens / eval_duration_s
            if completion_tokens and eval_duration_s
            else None
        )

        logger.info(
            "Ollama %s | model=%s | input_chars=%d | count=%d | "
            "prompt_tokens=%s | completion_tokens=%s | load=%.2fs | "
            "prompt_eval=%.2fs | eval=%.2fs | total=%.2fs | wall=%.2fs | "
            "tokens/sec=%s",
            call,
            payload.get("model"),
            input_chars,
            count,
            prompt_tokens,
            completion_tokens,
            load_duration_s,
            prompt_eval_duration_s,
            eval_duration_s,
            total_duration_s,
            wall_time,
            f"{tokens_per_sec:.1f}" if tokens_per_sec else "n/a",
        )
