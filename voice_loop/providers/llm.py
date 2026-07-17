"""Answer synthesis providers used after local database misses."""

from __future__ import annotations

import asyncio
import logging
import re
import time
import unicodedata

from ..types import LanguageCode
from .base import LanguageModelProvider

LOGGER = logging.getLogger(__name__)
_MAX_OUTPUT_TOKENS = 1024
_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


class HeuristicLLMProvider(LanguageModelProvider):
    """Produce a concise answer without an external API key."""

    async def generate_answer(self, language: LanguageCode, question: str) -> str:
        """Synthesize a short answer from the user question."""

        return await asyncio.to_thread(self._generate_sync, language, question)

    async def generate_answer_stream(self, language: LanguageCode, question: str):
        """Synthesize a short answer stream from the user question."""
        yield self._generate_sync(language, question)

    def _generate_sync(self, language: LanguageCode, question: str) -> str:
        """Assemble a language-appropriate answer from the question."""

        if language == "vi":
            return f"Câu trả lời ngắn gọn cho câu hỏi của bạn là: {question.strip()}"
        return f"A concise answer to your question is: {question.strip()}"


class GeminiLLMProvider(LanguageModelProvider):
    """Use Gemini on Vertex AI to synthesize a short answer."""

    def __init__(
        self,
        model: str = "gemini-3.1-flash-lite",
        fallback_model: str = "",
        project: str = "",
        location: str = "global",
        enable_google_search: bool = True,
        timeout_seconds: float | None = None,
        thinking_level: str = "minimal",
        thinking_budget: int = 0,
    ) -> None:
        """Store the model name and Vertex AI placement used for completions."""

        self.model = model
        self.fallback_model = fallback_model
        self.project = project
        self.location = location
        self.enable_google_search = enable_google_search
        self.timeout_seconds = timeout_seconds
        self.thinking_level = thinking_level
        self.thinking_budget = thinking_budget
        self._client = None
        self._primary_model_unavailable = False

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
        system_prompt = self._build_system_prompt(language, question)
        user_prompt = self._build_user_prompt(language, question)
        search_enabled_for_request = self.enable_google_search and self._should_enable_search(question)
        tools = [types.Tool(googleSearch=types.GoogleSearch())] if search_enabled_for_request else None
        active_model = self.fallback_model if self._primary_model_unavailable and self.fallback_model else self.model
        thinking_config = self._build_thinking_config(types, active_model)
        LOGGER.info(
            "Gemini request diagnostics: model=%s fallback_model=%s active_model=%s language=%s question_chars=%d timeout=%s max_output_tokens=%d search=%s thinking_level=%s thinking_budget=%s",
            self.model,
            self.fallback_model or "n/a",
            active_model,
            language,
            len(question),
            self.timeout_seconds if self.timeout_seconds is not None else "n/a",
            _MAX_OUTPUT_TOKENS,
            search_enabled_for_request,
            self.thinking_level or "n/a",
            self.thinking_budget,
        )
        started_at = time.perf_counter()
        try:
            response = await self._generate_content(
                client,
                types,
                active_model,
                user_prompt,
                system_prompt,
                tools,
                _MAX_OUTPUT_TOKENS,
                thinking_config,
            )
        except Exception as exc:
            if not self._is_model_unavailable_error(exc) or active_model == self.fallback_model or not self.fallback_model:
                raise
            LOGGER.warning(
                "Gemini primary model unavailable; retrying once with fallback model. primary_model=%s fallback_model=%s error_type=%s reason=%s",
                self.model,
                self.fallback_model,
                type(exc).__name__,
                str(exc).strip() or "<no error message>",
            )
            self._primary_model_unavailable = True
            active_model = self.fallback_model
            thinking_config = self._build_thinking_config(types, active_model)
            response = await self._generate_content(
                client,
                types,
                active_model,
                user_prompt,
                system_prompt,
                tools,
                _MAX_OUTPUT_TOKENS,
                thinking_config,
            )
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        content = getattr(response, "text", None)
        self._log_response_diagnostics(response, content)
        return self._finalize_first_response(response, content, elapsed_ms)

    async def generate_answer_stream(self, language: LanguageCode, question: str):
        """Call the Gemini API asynchronously and yield generated text chunks in real-time."""

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
        system_prompt = self._build_system_prompt(language, question)
        user_prompt = self._build_user_prompt(language, question)
        search_enabled_for_request = self.enable_google_search and self._should_enable_search(question)
        tools = [types.Tool(googleSearch=types.GoogleSearch())] if search_enabled_for_request else None
        active_model = self.fallback_model if self._primary_model_unavailable and self.fallback_model else self.model
        thinking_config = self._build_thinking_config(types, active_model)
        LOGGER.info(
            "Gemini stream request diagnostics: model=%s active_model=%s language=%s search=%s",
            self.model,
            active_model,
            language,
            search_enabled_for_request,
        )
        try:
            config_kwargs = {
                "system_instruction": system_prompt,
                "temperature": 0.2,
                "max_output_tokens": _MAX_OUTPUT_TOKENS,
            }
            if tools is not None:
                config_kwargs["tools"] = tools
            if thinking_config is not None:
                config_kwargs["thinking_config"] = thinking_config

            async for response in await client.aio.models.generate_content_stream(
                model=active_model,
                contents=user_prompt,
                config=types.GenerateContentConfig(**config_kwargs),
            ):
                content = getattr(response, "text", None)
                if content:
                    yield content
        except Exception as exc:
            if not self._is_model_unavailable_error(exc) or active_model == self.fallback_model or not self.fallback_model:
                raise
            LOGGER.warning(
                "Gemini primary model stream unavailable; retrying with fallback model %s.",
                self.fallback_model,
            )
            self._primary_model_unavailable = True
            active_model = self.fallback_model
            thinking_config = self._build_thinking_config(types, active_model)
            config_kwargs["thinking_config"] = thinking_config

            async for response in await client.aio.models.generate_content_stream(
                model=active_model,
                contents=user_prompt,
                config=types.GenerateContentConfig(**config_kwargs),
            ):
                content = getattr(response, "text", None)
                if content:
                    yield content

    @classmethod
    def _build_system_prompt(cls, language: LanguageCode, question: str) -> str:
        """Build the system prompt, adding demo tone only for Greenwich questions."""

        base_prompt = (
            "You are the Greenwich Admissions Voice Assistant. Answer in exactly the requested language. "
            "First, analyze the user's question. If the question is a garbled, incoherent jumble of words, repetitive noise, or background chatter, "
            "reply exactly with the sentinel token: [NOISE_REPROMPT]. "
            "You should prioritize questions about Greenwich Vietnam admissions, student life, and basic pleasantries. "
            "However, you can also answer general everyday questions (like the weather, basic facts, time) concisely. "
            "For highly inappropriate, spam, or completely meaningless questions, politely decline and restate your purpose. "
            "Answer naturally in 1-2 short, complete sentences. Do not use any markdown formatting like bold (**), italics (*), or headers (#)."
        )
        if not cls._is_greenwich_context_question(question):
            return base_prompt
        if language == "vi":
            return (
                base_prompt
                + " Khi câu hỏi nói về Greenwich Việt Nam, hãy trả lời như một chuyên viên tư vấn tuyển sinh có kinh nghiệm: "
                "ấm áp, tự tin, thực tế và hướng đến nhu cầu của học sinh/phụ huynh. "
                "Trả lời đúng trọng tâm câu hỏi trước. Với câu hỏi về lý do chọn, điểm khác biệt, độ phù hợp hoặc ngành yêu thích, "
                "hãy mở đầu bằng giá trị thực tế cho người học như phong cách học, hỗ trợ sinh viên, dự án thực hành, môi trường học, định hướng nghề nghiệp hoặc mức độ phù hợp. "
                "Không mở đầu bằng thông tin liên kết, bằng cấp hoặc chuẩn Anh Quốc trừ khi người dùng hỏi về danh tính, uy tín, bằng cấp hoặc đối tác. "
                "Không bịa xếp hạng, cam kết việc làm, cam kết trúng tuyển, học bổng hoặc kết quả đầu ra. "
                "Nếu cần nhắc tên, dùng đúng: Đại học Greenwich và Tổ chức Giáo dục FPT."
            )
        return (
            base_prompt
            + " When the question is about Greenwich Vietnam, answer like an experienced admissions advisor: "
            "warm, confident, practical, and focused on student or parent needs. "
            "Answer the user's specific concern first. For why, difference, suitability, or interest-based questions, "
            "lead with practical student value such as learning style, student support, project work, campus experience, career orientation, or fit. "
            "Do not lead with partnership, degree, or UK-standard facts unless the user asks about identity, credibility, degree, or partnership. "
            "Do not invent rankings, guaranteed jobs, guaranteed admission, scholarships, or outcomes. "
            "If needed, use the correct names: University of Greenwich and FPT Education."
        )

    @classmethod
    def _build_user_prompt(cls, language: LanguageCode, question: str) -> str:
        """Build the user prompt with a compact Greenwich fact note when relevant."""

        language_name = "Vietnamese" if language == "vi" else "English"
        prompt = f"Language: {language_name}\nQuestion: {question}"
        if not cls._is_greenwich_context_question(question):
            return prompt
        if language == "vi":
            return (
                prompt
                + "\nGreenwich fact guardrails, use only when directly relevant: Greenwich Việt Nam liên kết với Đại học Greenwich "
                "và Tổ chức Giáo dục FPT; chương trình đào tạo theo chuẩn Anh Quốc tại Việt Nam."
            )
        return (
            prompt
            + "\nGreenwich fact guardrails, use only when directly relevant: Greenwich Vietnam is linked with the "
            "University of Greenwich and FPT Education; it offers UK-standard study in Vietnam."
        )

    @classmethod
    def _is_greenwich_context_question(cls, question: str) -> bool:
        """Return whether a prompt clearly refers to Greenwich Vietnam."""

        tokens = set(_TOKEN_RE.findall(cls._normalize_for_context(question)))
        return "greenwich" in tokens or {"green", "which"} <= tokens

    @classmethod
    def _should_enable_search(cls, question: str) -> bool:
        """Return whether the request likely needs current Google Search grounding."""

        tokens = set(_TOKEN_RE.findall(cls._normalize_for_context(question)))
        return bool(
            tokens
            & {
                "current",
                "gan",
                "giao",
                "hom",
                "latest",
                "match",
                "news",
                "score",
                "sport",
                "sports",
                "thang",
                "thong",
                "tin",
                "today",
                "traffic",
                "tran",
                "weather",
                "won",
            }
        )

    def _build_thinking_config(self, types, model: str):
        """Build model-family specific thinking controls when the SDK exposes them."""

        thinking_config_cls = getattr(types, "ThinkingConfig", None)
        if thinking_config_cls is None:
            return None
        normalized_model = model.lower()
        if normalized_model.startswith("gemini-3"):
            return thinking_config_cls(thinking_level=self.thinking_level or "minimal")
        if normalized_model.startswith("gemini-2.5"):
            return thinking_config_cls(thinking_budget=self.thinking_budget)
        return None

    @staticmethod
    def _is_model_unavailable_error(exc: Exception) -> bool:
        """Return whether an exception is a model availability/configuration failure."""

        exc_type = type(exc).__name__.lower()
        error_text = str(exc).lower()
        return exc_type in {"notfound", "invalidargument", "failedprecondition"} or any(
            marker in error_text
            for marker in (
                "model not found",
                "not found",
                "not supported",
                "unsupported model",
                "invalid model",
                "permission denied",
            )
        )

    @staticmethod
    def _normalize_for_context(text: str) -> str:
        """Normalize text for lightweight entity detection."""

        lowered = text.lower().replace("Ä‘", "d")
        ascii_like = unicodedata.normalize("NFD", lowered).encode("ascii", "ignore").decode("ascii")
        return " ".join(_TOKEN_RE.findall(ascii_like))

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
    async def _generate_content(
        client,
        types,
        model: str,
        user_prompt: str,
        system_prompt: str,
        tools,
        max_output_tokens: int,
        thinking_config,
    ):
        """Call Gemini with the requested token cap."""

        config_kwargs = {
            "system_instruction": system_prompt,
            "temperature": 0.2,
            "max_output_tokens": max_output_tokens,
        }
        if tools is not None:
            config_kwargs["tools"] = tools
        if thinking_config is not None:
            config_kwargs["thinking_config"] = thinking_config
        return await client.aio.models.generate_content(
            model=model,
            contents=user_prompt,
            config=types.GenerateContentConfig(**config_kwargs),
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
