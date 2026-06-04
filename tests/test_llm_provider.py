"""Unit tests for LLM provider diagnostics."""

from __future__ import annotations

import logging
import sys
from types import ModuleType
from types import SimpleNamespace

import pytest

from voice_loop.providers.llm import GeminiLLMProvider


def test_gemini_response_diagnostics_logs_finish_reason_and_candidate_count(caplog) -> None:
    """Gemini diagnostics should expose finish metadata without changing behavior."""

    response = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                finish_reason="STOP",
                safety_ratings=["safe"],
                content=SimpleNamespace(parts=[SimpleNamespace(text="A complete answer.")]),
            )
        ],
        prompt_feedback="ok",
    )

    caplog.set_level(logging.INFO)
    GeminiLLMProvider._log_response_diagnostics(response, "A complete answer.")

    messages = [record.getMessage() for record in caplog.records]
    assert any("candidate_count=1" in message for message in messages)
    assert any("finish_reasons=STOP" in message for message in messages)
    assert any("terminal_punctuation=True" in message for message in messages)


def test_gemini_response_diagnostics_logs_partial_answer_without_punctuation(caplog) -> None:
    """Partial text should be visible as no terminal punctuation in diagnostics."""

    response = SimpleNamespace(candidates=[], prompt_feedback=None)

    caplog.set_level(logging.INFO)
    GeminiLLMProvider._log_response_diagnostics(response, "Manchester United da thang tran gan nhat")

    assert any("terminal_punctuation=False" in record.getMessage() for record in caplog.records)


def test_gemini_response_diagnostics_logs_candidate_text_lengths(caplog) -> None:
    """Diagnostics should reveal when candidate content is longer than response.text."""

    response = SimpleNamespace(
        candidates=[
            SimpleNamespace(
                finish_reason="MAX_TOKENS",
                safety_ratings=[],
                content=SimpleNamespace(parts=[SimpleNamespace(text="Longer candidate text than response text.")]),
            )
        ],
        prompt_feedback=None,
    )

    caplog.set_level(logging.INFO)
    GeminiLLMProvider._log_response_diagnostics(response, "Short.")

    assert any("candidate_text_chars=41" in record.getMessage() for record in caplog.records)


def test_gemini_finalizes_stop_response() -> None:
    """STOP text should be returned from the first response."""

    response = SimpleNamespace(candidates=[SimpleNamespace(finish_reason="STOP")])

    assert GeminiLLMProvider._finalize_first_response(response, "A complete answer.", 12.0) == "A complete answer."


def test_gemini_finalizes_usable_max_token_response() -> None:
    """Usable MAX_TOKENS text should be returned from the first response."""

    response = SimpleNamespace(candidates=[SimpleNamespace(finish_reason="MAX_TOKENS")])

    assert (
        GeminiLLMProvider._finalize_first_response(
            response,
            "OpenAI released new tools, while Google expanded Gemini support",
            12.0,
        )
        == "OpenAI released new tools, while Google expanded Gemini support"
    )


def test_gemini_rejects_unusable_max_token_response() -> None:
    """Unusable MAX_TOKENS text should fail immediately."""

    response = SimpleNamespace(candidates=[SimpleNamespace(finish_reason="MAX_TOKENS")])

    with pytest.raises(RuntimeError, match="unusable truncated"):
        GeminiLLMProvider._finalize_first_response(response, "This week, Anthropic secured a new deal with", 12.0)


def test_gemini_generate_answer_makes_one_generation_call(monkeypatch) -> None:
    """Live Gemini generation should not make a hidden second call."""

    google_module = ModuleType("google")
    genai_module = ModuleType("google.genai")
    types_module = ModuleType("google.genai.types")

    class _Config:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _Models:
        async def generate_content(self, **kwargs):
            return SimpleNamespace(text="A complete answer.", candidates=[SimpleNamespace(finish_reason="STOP")])

    class _Client:
        def __init__(self, **kwargs):
            self.aio = SimpleNamespace(models=_Models())

    genai_module.Client = _Client
    types_module.GenerateContentConfig = _Config
    types_module.GoogleSearch = lambda: SimpleNamespace()
    types_module.Tool = lambda **kwargs: SimpleNamespace(**kwargs)
    genai_module.types = types_module
    google_module.genai = genai_module
    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.genai", genai_module)
    monkeypatch.setitem(sys.modules, "google.genai.types", types_module)

    calls = 0

    async def _fake_generate_content(*args, **kwargs):
        nonlocal calls
        calls += 1
        return SimpleNamespace(text="A complete answer.", candidates=[SimpleNamespace(finish_reason="STOP")])

    monkeypatch.setattr(GeminiLLMProvider, "_generate_content", staticmethod(_fake_generate_content))
    provider = GeminiLLMProvider(project="project")

    import asyncio

    assert asyncio.run(provider.generate_answer("en", "question")) == "A complete answer."
    assert calls == 1


def test_gemini_salvages_usable_max_token_text() -> None:
    """Usable MAX_TOKENS text should be salvageable without retrying."""

    assert GeminiLLMProvider._is_usable_truncated_response(
        "OpenAI released new tools, while Google expanded Gemini support"
    )
    assert GeminiLLMProvider._is_usable_truncated_response("A complete answer.")
    assert not GeminiLLMProvider._is_usable_truncated_response("This week, Anthropic secured a new deal with")
