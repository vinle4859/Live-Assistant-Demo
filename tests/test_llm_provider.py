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

    class _ThinkingConfig:
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
    types_module.ThinkingConfig = _ThinkingConfig
    types_module.GoogleSearch = lambda: SimpleNamespace()
    types_module.Tool = lambda **kwargs: SimpleNamespace(**kwargs)
    genai_module.types = types_module
    google_module.genai = genai_module
    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.genai", genai_module)
    monkeypatch.setitem(sys.modules, "google.genai.types", types_module)

    calls = 0
    observed_max_tokens = 0

    async def _fake_generate_content(*args, **kwargs):
        nonlocal calls, observed_max_tokens
        calls += 1
        observed_max_tokens = args[6]
        return SimpleNamespace(text="A complete answer.", candidates=[SimpleNamespace(finish_reason="STOP")])

    monkeypatch.setattr(GeminiLLMProvider, "_generate_content", staticmethod(_fake_generate_content))
    provider = GeminiLLMProvider(project="project")

    import asyncio

    assert asyncio.run(provider.generate_answer("en", "question")) == "A complete answer."
    assert calls == 1
    assert observed_max_tokens == 1024


def test_gemini_search_is_disabled_for_greenwich_demo_prompt() -> None:
    """Greenwich demo prompts should not pay Google Search latency."""

    assert not GeminiLLMProvider._should_enable_search("Why should students choose Greenwich Vietnam?")


def test_gemini_search_is_enabled_for_current_prompts() -> None:
    """Current-information prompts should still enable Google Search."""

    assert GeminiLLMProvider._should_enable_search("What is the latest weather in Ha Noi?")
    assert GeminiLLMProvider._should_enable_search("Tell me traffic on Cong Hoa street")


def test_gemini_three_uses_minimal_thinking_level() -> None:
    """Gemini 3 requests should use minimal thinking for low live latency."""

    class _ThinkingConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    provider = GeminiLLMProvider(project="project", model="gemini-3.1-flash-lite", thinking_level="minimal")

    thinking_config = provider._build_thinking_config(SimpleNamespace(ThinkingConfig=_ThinkingConfig), provider.model)

    assert thinking_config.kwargs == {"thinking_level": "minimal"}


def test_gemini_two_point_five_fallback_uses_zero_thinking_budget() -> None:
    """Gemini 2.5 fallback should disable thinking for voice latency."""

    class _ThinkingConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    provider = GeminiLLMProvider(project="project", model="gemini-2.5-flash", thinking_budget=0)

    thinking_config = provider._build_thinking_config(SimpleNamespace(ThinkingConfig=_ThinkingConfig), provider.model)

    assert thinking_config.kwargs == {"thinking_budget": 0}


def test_gemini_model_unavailable_error_detection_is_narrow() -> None:
    """Fallback should be reserved for model availability/configuration failures."""

    assert GeminiLLMProvider._is_model_unavailable_error(RuntimeError("model not found"))
    assert not GeminiLLMProvider._is_model_unavailable_error(RuntimeError("Gemini returned an empty response"))


def test_gemini_unavailable_primary_retries_once_with_fallback(monkeypatch) -> None:
    """Unavailable primary models should retry once with the configured fallback model."""

    google_module = ModuleType("google")
    genai_module = ModuleType("google.genai")
    types_module = ModuleType("google.genai.types")

    class _Config:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _ThinkingConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _Client:
        def __init__(self, **kwargs):
            self.aio = SimpleNamespace(models=SimpleNamespace())

    genai_module.Client = _Client
    types_module.GenerateContentConfig = _Config
    types_module.ThinkingConfig = _ThinkingConfig
    genai_module.types = types_module
    google_module.genai = genai_module
    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.genai", genai_module)
    monkeypatch.setitem(sys.modules, "google.genai.types", types_module)

    requested_models: list[str] = []

    async def _fake_generate_content(*args, **kwargs):
        requested_models.append(args[2])
        if args[2] == "gemini-3.1-flash-lite":
            raise RuntimeError("model not found")
        return SimpleNamespace(text="A complete answer.", candidates=[SimpleNamespace(finish_reason="STOP")])

    monkeypatch.setattr(GeminiLLMProvider, "_generate_content", staticmethod(_fake_generate_content))
    provider = GeminiLLMProvider(
        project="project",
        model="gemini-3.1-flash-lite",
        fallback_model="gemini-2.5-flash",
    )

    import asyncio

    assert asyncio.run(provider.generate_answer("en", "question")) == "A complete answer."
    assert requested_models == ["gemini-3.1-flash-lite", "gemini-2.5-flash"]


def test_gemini_greenwich_prompt_uses_admissions_advisor_persona() -> None:
    """Greenwich questions should get demo-friendly admissions guidance."""

    system_prompt = GeminiLLMProvider._build_system_prompt("en", "Why should students choose Greenwich Vietnam?")
    user_prompt = GeminiLLMProvider._build_user_prompt("en", "Why should students choose Greenwich Vietnam?")

    assert "experienced admissions advisor" in system_prompt
    assert "FPT Education" in system_prompt
    assert "Answer the user's specific concern first" in system_prompt
    assert "Do not lead with partnership" in system_prompt
    assert "Greenwich fact guardrails" in user_prompt


def test_gemini_non_greenwich_prompt_does_not_use_admissions_persona() -> None:
    """General questions should not inherit Greenwich-specific demo tone."""

    system_prompt = GeminiLLMProvider._build_system_prompt("en", "What is the weather in Ha Noi?")
    user_prompt = GeminiLLMProvider._build_user_prompt("en", "What is the weather in Ha Noi?")

    assert "experienced admissions advisor" not in system_prompt
    assert "Greenwich fact guardrails" not in user_prompt


def test_gemini_vietnamese_greenwich_prompt_uses_fpt_education_vietnamese_name() -> None:
    """Vietnamese Greenwich prompts should use the Vietnamese FPT Education name."""

    system_prompt = GeminiLLMProvider._build_system_prompt("vi", "Greenwich Việt Nam khác gì?")
    user_prompt = GeminiLLMProvider._build_user_prompt("vi", "Greenwich Việt Nam khác gì?")

    assert "Tổ chức Giáo dục FPT" in system_prompt
    assert "Tổ chức Giáo dục FPT" in user_prompt
    assert "Trả lời đúng trọng tâm" in system_prompt


def test_gemini_salvages_usable_max_token_text() -> None:
    """Usable MAX_TOKENS text should be salvageable without retrying."""

    assert GeminiLLMProvider._is_usable_truncated_response(
        "OpenAI released new tools, while Google expanded Gemini support"
    )
    assert GeminiLLMProvider._is_usable_truncated_response("A complete answer.")
    assert not GeminiLLMProvider._is_usable_truncated_response("This week, Anthropic secured a new deal with")
