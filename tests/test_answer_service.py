from __future__ import annotations

import pytest

import app.services.answer_service as answer_module


class FakeProvider:
    def __init__(self, answer: str):
        self.answer = answer

    def generate_answer(self, question: str, vector_results: list[dict], graph_results: list[dict]) -> str:
        return self.answer


def test_dummy_answer_provider_handles_empty_results():
    provider = answer_module.DummyAnswerProvider()

    answer = provider.generate_answer("Che cosa sai?", [], [])

    assert "Non ho trovato risultati rilevanti" in answer


def test_dummy_answer_provider_summarizes_results():
    provider = answer_module.DummyAnswerProvider()

    answer = provider.generate_answer(
        "Che cosa sai?",
        [{"payload": {"text": "deployment del servizio X"}}],
        [{"text": "chunk nel grafo"}],
    )

    assert "Risultati vettoriali trovati: 1" in answer
    assert "Risultati grafo trovati: 1" in answer
    assert "deployment del servizio X" in answer
    assert "chunk nel grafo" in answer


def test_generate_answer_uses_custom_provider():
    answer = answer_module.generate_answer(
        "Domanda",
        vector_results=[],
        graph_results=[],
        provider=FakeProvider("risposta finale"),
    )

    assert answer == "risposta finale"


def test_generate_answer_raises_on_empty_output():
    with pytest.raises(
        answer_module.AnswerServiceError,
        match="Answer provider returned an empty answer",
    ):
        answer_module.generate_answer(
            "Domanda",
            vector_results=[],
            graph_results=[],
            provider=FakeProvider("   "),
        )


def test_get_answer_provider_returns_dummy_without_api_key(monkeypatch):
    class FakeSettings:
        openai_api_key = None
        chat_model = "gpt-4.1-mini"

    monkeypatch.setattr(answer_module, "get_settings", lambda: FakeSettings())

    provider = answer_module.get_answer_provider()

    assert isinstance(provider, answer_module.DummyAnswerProvider)


def test_get_answer_provider_returns_openai_provider_with_api_key(monkeypatch):
    class FakeSettings:
        openai_api_key = "secret"
        chat_model = "gpt-4.1-mini"

    monkeypatch.setattr(answer_module, "get_settings", lambda: FakeSettings())

    provider = answer_module.get_answer_provider()

    assert isinstance(provider, answer_module.OpenAICompatibleAnswerProvider)
    assert provider.api_key == "secret"
    assert provider.model == "gpt-4.1-mini"