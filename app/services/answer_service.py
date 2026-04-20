from __future__ import annotations

from functools import lru_cache
from typing import Protocol

from transformers import pipeline

from app.core.config import get_settings


class AnswerServiceError(Exception):
    pass


class AnswerProvider(Protocol):
    def generate_answer(
        self,
        question: str,
        vector_results: list[dict],
        graph_results: list[dict],
    ) -> str:
        ...


class DummyAnswerProvider:
    def generate_answer(
        self,
        question: str,
        vector_results: list[dict],
        graph_results: list[dict],
    ) -> str:
        if not vector_results and not graph_results:
            return "Non ho trovato risultati rilevanti nella base di conoscenza per questa domanda."

        top_payload = vector_results[0].get("payload", {}) if vector_results else {}
        top_text = (top_payload.get("text") or "").strip()

        parts = []
        if top_text:
            parts.append(f"Dal contesto recuperato emerge che: {top_text}")
        if graph_results:
            parts.append(
                f"Ho trovato anche {len(graph_results)} elementi di contesto collegati nel grafo."
            )
        return " ".join(parts) if parts else "Ho trovato risultati, ma il contesto è limitato."


@lru_cache(maxsize=1)
def load_text2text_pipeline(model_name: str):
    return pipeline(
        "text2text-generation",
        model=model_name,
        tokenizer=model_name,
    )


class LocalTransformersAnswerProvider:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def _build_prompt(
        self,
        question: str,
        vector_results: list[dict],
        graph_results: list[dict],
    ) -> str:
        vector_lines: list[str] = []
        for index, item in enumerate(vector_results, start=1):
            payload = item.get("payload", {})
            vector_lines.append(
                f"[{index}] score={item.get('score')} "
                f"document_id={payload.get('document_id')} "
                f"chunk_id={payload.get('chunk_id')} "
                f"text={payload.get('text')}"
            )

        graph_lines: list[str] = []
        for index, item in enumerate(graph_results, start=1):
            graph_lines.append(
                f"[{index}] document_id={item.get('document_id')} "
                f"chunk_id={item.get('chunk_id')} "
                f"text={item.get('text')}"
            )

        return (
            "Rispondi in italiano in modo breve, fedele al contesto e senza inventare.\n\n"
            f"Domanda: {question}\n\n"
            "Contesto vettoriale:\n"
            + ("\n".join(vector_lines) if vector_lines else "nessuno")
            + "\n\nContesto grafo:\n"
            + ("\n".join(graph_lines) if graph_lines else "nessuno")
            + "\n\nRisposta:"
        )

    def generate_answer(
        self,
        question: str,
        vector_results: list[dict],
        graph_results: list[dict],
    ) -> str:
        try:
            pipe = load_text2text_pipeline(self.model_name)
            prompt = self._build_prompt(question, vector_results, graph_results)
            output = pipe(
                prompt,
                max_new_tokens=160,
                do_sample=False,
            )
            answer = output[0]["generated_text"].strip()
            if not answer:
                raise AnswerServiceError("Local model returned an empty answer")
            return answer
        except Exception as exc:
            raise AnswerServiceError(f"Local answer generation failed: {exc}") from exc


def get_answer_provider() -> AnswerProvider:
    settings = get_settings()
    llm_provider = getattr(settings, "llm_provider", "dummy")

    if llm_provider == "local":
        return LocalTransformersAnswerProvider(
            model_name=settings.local_chat_model,
        )

    return DummyAnswerProvider()


def generate_answer(
    question: str,
    vector_results: list[dict],
    graph_results: list[dict],
    provider: AnswerProvider | None = None,
) -> str:
    active_provider = provider or get_answer_provider()
    answer = active_provider.generate_answer(question, vector_results, graph_results)

    if not answer.strip():
        raise AnswerServiceError("Answer provider returned an empty answer")

    return answer.strip()