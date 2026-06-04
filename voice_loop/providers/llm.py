"""Answer synthesis providers used after local database misses."""

from __future__ import annotations

import asyncio
import logging
import time

from ..types import LanguageCode
from .base import LanguageModelProvider

LOGGER = logging.getLogger(__name__)
_MAX_OUTPUT_TOKENS = 768


class HeuristicLLMProvider(LanguageModelProvider):
    """Produce a concise answer without an external API key."""

    async def generate_answer(self, language: LanguageCode, question: str) -> str:
        """Synthesize a short answer from the user question."""

        return await asyncio.to_thread(self._generate_sync, language, question)

    def _generate_sync(self, language: LanguageCode, question: str) -> str:
        """Assemble a language-appropriate answer from the question."""

        if language == "vi":
            return f"Câu trả lời ngắn gọn cho câu hỏi của bạn là: {question.strip()}"
        return f"A concise answer to your question is: {question.strip()}"


class GeminiLLMProvider(LanguageModelProvider):
    """Use Gemini on Vertex AI to synthesize a short answer."""

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        project: str = "",
        location: str = "us-central1",
        enable_google_search: bool = True,
        timeout_seconds: float | None = None,
    ) -> None:
        """Store the model name and Vertex AI placement used for completions."""

        self.model = model
        self.project = project
        self.location = location
        self.enable_google_search = enable_google_search
        self.timeout_seconds = timeout_seconds
        self._client = None

    async def generate_answer(self, language: LanguageCode, question: str) -> str:
        """Call the Gemini API asynchronously and return the generated answer."""

        if not self.project:
            raise RuntimeError("GOOGLE_CLOUD_PROJECT is not configured")

        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:  # pragma: no cover - runtime dependency guard
            raise RuntimeError("google-genai is not installed") from exc

        if self._client is None:
            self._client = genai.Client(vertexai=True, project=self.project, location=self.location)
        client = self._client
        language_name = "Vietnamese" if language == "vi" else "English"
        system_prompt = (
            "You are a concise assistant. Answer in exactly the requested language using general knowledge. "
            "Use Google Search when current information is needed, such as weather, news, or recent events. "
            "Produce a complete spoken answer in this first response. If a location is required and missing, ask for the city. "
            "Answer in 1-3 short sentences, under about 60 words. For list requests, use up to 2 short bullets. "
            "For poems, use at most 4 short lines."
        )
        user_prompt = f"Language: {language_name}\nQuestion: {question}"
        tools = [types.Tool(googleSearch=types.GoogleSearch())] if self.enable_google_search else None
        LOGGER.info(
            "Gemini request diagnostics: model=%s language=%s question_chars=%d timeout=%s max_output_tokens=%d search=%s",
            self.model,
            language,
            len(question),
            self.timeout_seconds if self.timeout_seconds is not None else "n/a",
            _MAX_OUTPUT_TOKENS,
            self.enable_google_search,
        )
        started_at = time.perf_counter()
        response = await self._generate_content(client, types, self.model, user_prompt, system_prompt, tools, _MAX_OUTPUT_TOKENS)
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        content = getattr(response, "text", None)
        self._log_response_diagnostics(response, content)
        return self._finalize_first_response(response, content, elapsed_ms)

    @classmethod
    def _finalize_first_response(cls, response, content: str | None, elapsed_ms: float) -> str:
        """Validate and return the first Gemini response without retrying."""

        if not content:
            raise RuntimeError("Gemini returned an empty response")
        if cls._is_max_tokens_response(response):
            salvaged_partial = cls._is_usable_truncated_response(content)
            LOGGER.info(
                "Gemini truncation handling: finish_reasons=%s text_chars=%d salvaged_partial=%s elapsed_ms=%.0f",
                cls._candidate_finish_reasons(response),
                len(content.strip()),
                salvaged_partial,
                elapsed_ms,
            )
            if not salvaged_partial:
                raise RuntimeError("Gemini returned an unusable truncated response")
        return content.strip()

    @staticmethod
    async def _generate_content(client, types, model: str, user_prompt: str, system_prompt: str, tools, max_output_tokens: int):
        """Call Gemini with the requested token cap."""

        return await client.aio.models.generate_content(
            model=model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.2,
                max_output_tokens=max_output_tokens,
                tools=tools,
            ),
        )

    @classmethod
    def _is_max_tokens_response(cls, response) -> bool:
        """Return whether Gemini reported a token-limit finish reason."""

        return any(reason.endswith("MAX_TOKENS") for reason in cls._candidate_finish_reasons(response).split(","))

    @classmethod
    def _is_usable_truncated_response(cls, content: str | None) -> bool:
        """Return whether a MAX_TOKENS response is complete enough to speak."""

        text = (content or "").strip()
        if not text:
            return False
        if text.endswith((".", "!", "?")):
            return True
        tokens = text.split()
        return len(tokens) >= 8 and tokens[-1].strip(".,;:!?").lower() not in {
            "a",
            "an",
            "and",
            "as",
            "at",
            "by",
            "for",
            "from",
            "in",
            "include",
            "including",
            "of",
            "on",
            "or",
            "the",
            "to",
            "with",
        }

    @classmethod
    def _log_response_diagnostics(cls, response, content: str | None) -> None:
        """Log Gemini response metadata without changing provider behavior."""

        text = (content or "").strip()
        LOGGER.info(
            "Gemini response diagnostics: text_chars=%d terminal_punctuation=%s candidate_count=%d candidate_text_chars=%s finish_reasons=%s safety=%s prompt_feedback=%s",
            len(text),
            bool(text.endswith((".", "!", "?"))),
            len(getattr(response, "candidates", None) or []),
            cls._candidate_text_lengths(response),
            cls._candidate_finish_reasons(response),
            cls._candidate_safety(response),
            cls._safe_repr(getattr(response, "prompt_feedback", None)),
        )
        if cls._debug_llm_text_enabled() and text:
            LOGGER.info(
                'Gemini raw text excerpt: first="%s" last="%s"',
                cls._excerpt(text[:160]),
                cls._excerpt(text[-160:]),
            )

    @staticmethod
    def _candidate_finish_reasons(response) -> str:
        """Return candidate finish reasons as a compact diagnostic string."""

        reasons: list[str] = []
        for candidate in getattr(response, "candidates", None) or []:
            reason = getattr(candidate, "finish_reason", None)
            reasons.append(str(reason) if reason is not None else "n/a")
        return ",".join(reasons) or "n/a"

    @classmethod
    def _candidate_text_lengths(cls, response) -> str:
        """Return text lengths found inside candidate content parts."""

        lengths: list[str] = []
        for candidate in getattr(response, "candidates", None) or []:
            lengths.append(str(len(cls._candidate_text(candidate))))
        return ",".join(lengths) or "n/a"

    @staticmethod
    def _candidate_text(candidate) -> str:
        """Extract candidate content text defensively for diagnostics."""

        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        text_parts = [str(getattr(part, "text", "")) for part in parts if getattr(part, "text", "")]
        return "".join(text_parts)

    @classmethod
    def _candidate_safety(cls, response) -> str:
        """Return safety metadata as a compact diagnostic string."""

        safety_entries: list[str] = []
        for candidate in getattr(response, "candidates", None) or []:
            safety = getattr(candidate, "safety_ratings", None)
            if safety:
                safety_entries.append(cls._safe_repr(safety))
        return " | ".join(safety_entries) or "n/a"

    @staticmethod
    def _debug_llm_text_enabled() -> bool:
        """Return whether raw LLM text excerpts should be logged."""

        import os

        return os.getenv("VOICE_LOOP_DEBUG_LLM_TEXT", "").strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _safe_repr(value) -> str:
        """Return a bounded representation for provider metadata."""

        if value is None:
            return "n/a"
        return " ".join(str(value).split())[:240]

    @staticmethod
    def _excerpt(value: str) -> str:
        """Return a log-safe single-line text excerpt."""

        return " ".join(value.split())
