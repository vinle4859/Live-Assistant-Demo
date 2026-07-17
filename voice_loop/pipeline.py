"""Pipeline orchestration for the voice-to-voice workflow."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable
from uuid import uuid4

from .db import KnowledgeBase, KnowledgeMatch
from .transcript_cheats import TranscriptCheatRule, apply_transcript_cheats
from .types import LanguageCode, PipelineResponse
from .providers.base import LanguageModelProvider, SpeechToTextProvider, TextToSpeechProvider

LOGGER = logging.getLogger(__name__)
_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


@dataclass
class _ContextAnchor:
    """Short-lived topic anchor derived from high-confidence local Q&A matches."""

    source_id: str | None
    topic_phrase: str
    turns_remaining: int


@dataclass(frozen=True)
class _LLMDirectResult:
    answer: str
    status: str
    failure_type: str | None = None


@dataclass(frozen=True)
class _LocalDbRoutingDecision:
    action: str
    reason: str = ""


class VoicePipeline:
    """Run transcription, retrieval, synthesis, and TTS as a single unit."""

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        stt_provider: SpeechToTextProvider,
        llm_provider: LanguageModelProvider,
        fallback_llm_provider: LanguageModelProvider | None,
        primary_tts_provider: TextToSpeechProvider,
        fallback_tts_provider: TextToSpeechProvider | None,
        output_dir: Path,
        timeout_seconds: float = 3.0,
        llm_timeout_seconds: float | None = None,
        llm_direct_min_query_tokens: int = 1,
        context_link_enabled: bool = True,
        context_link_max_turn_gap: int = 2,
        context_link_short_query_max_tokens: int = 6,
        context_link_min_score_delta: float = 0.08,
        transcript_cheats: tuple[TranscriptCheatRule, ...] = (),
    ) -> None:
        """Store the providers and runtime settings for the pipeline."""

        self.knowledge_base = knowledge_base
        self.stt_provider = stt_provider
        self.llm_provider = llm_provider
        self.fallback_llm_provider = fallback_llm_provider
        self.primary_tts_provider = primary_tts_provider
        self.fallback_tts_provider = fallback_tts_provider
        self.output_dir = output_dir
        self.timeout_seconds = timeout_seconds
        self.llm_timeout_seconds = llm_timeout_seconds or timeout_seconds
        self.llm_direct_min_query_tokens = llm_direct_min_query_tokens
        self.context_link_enabled = context_link_enabled
        self.context_link_max_turn_gap = max(1, context_link_max_turn_gap)
        self.context_link_short_query_max_tokens = max(1, context_link_short_query_max_tokens)
        self.context_link_min_score_delta = max(0.0, min(1.0, context_link_min_score_delta))
        self.transcript_cheats = transcript_cheats
        self._context_anchor: _ContextAnchor | None = None
        self._greenwich_llm_context_turns_remaining = 0

    async def process_transcription(self, transcription: str, language: LanguageCode) -> PipelineResponse:
        """Resolve a precomputed transcript through DB/LLM routing and synthesize output audio."""

        return await self.process_transcription_with_metrics(transcription, language, stt_elapsed_ms=None)

    async def process_transcription_with_metrics(
        self,
        transcription: str,
        language: LanguageCode,
        stt_elapsed_ms: float | None,
        language_confidence: float | None = None,
        language_reason: str | None = None,
        thinking_cue_delay_seconds: float = 0.0,
        thinking_cue_callback: Callable[[], Awaitable[None]] | None = None,
    ) -> PipelineResponse:
        """Resolve transcript and emit per-stage timing telemetry for the turn."""

        started_at = time.perf_counter()
        self.knowledge_base.ensure_schema()
        transcription = transcription.strip()
        transcription, applied_cheats = apply_transcript_cheats(transcription, self.transcript_cheats)
        if applied_cheats:
            LOGGER.info(
                "Transcript cheats applied (%s): %s",
                ", ".join(applied_cheats),
                transcription,
            )
        transcription, guarded_corrections = self._correct_guarded_greenwich_misrecognitions(transcription)
        if guarded_corrections:
            LOGGER.info(
                "Guarded Greenwich corrections applied (%s): %s",
                ", ".join(guarded_corrections),
                transcription,
            )
        transcription, context_corrections = self._apply_greenwich_llm_context(transcription)
        if context_corrections:
            LOGGER.info(
                "Greenwich LLM context applied (%s): %s",
                ", ".join(context_corrections),
                transcription,
            )
        self._advance_context_anchor()
        clarification_text = self._clarification_phrase_for_query(transcription, language)
        if clarification_text:
            db_started_at = time.perf_counter()
            db_match = None
            routed_query = transcription
            used_context_link = False
        else:
            db_started_at = time.perf_counter()
            db_match, routed_query, used_context_link = self._lookup_response_with_context(
                transcription,
                language,
                language_confidence=language_confidence,
                language_reason=language_reason,
            )
        response_text = None
        db_elapsed_ms = (time.perf_counter() - db_started_at) * 1000.0
        resolved_source: str
        llm_elapsed_ms: float | None = None
        db_score: float | None = None
        db_mode: str | None = None
        routed_query_token_count = self._query_token_count(routed_query)
        llm_status = "not_needed"
        llm_failure_type: str | None = None
        pre_rendered_audio_path: Path | None = None

        if db_match is not None:
            db_score = db_match.score
            db_mode = db_match.retrieval_mode
        local_db_decision = (
            self._decide_local_db_routing(
                db_match,
                routed_query,
                language_confidence,
                language_reason,
            )
            if db_match
            else _LocalDbRoutingDecision("reject_to_llm")
        )

        if clarification_text:
            response_text = clarification_text
            resolved_source = "fallback"
            llm_status = "clarification"
        elif local_db_decision.action == "reprompt_noisy":
            response_text = self._repeat_request_phrase(language)
            resolved_source = "fallback"
            llm_status = "skipped_noisy_query"
        elif db_match and local_db_decision.action == "accept_local_db":
            response_text = self._compact_local_db_response(db_match.response, routed_query, db_match, language)
            resolved_source = "local_db"
            self._update_context_anchor(db_match)
            if getattr(db_match, "audio_path", None):
                candidate_path = Path(db_match.audio_path)
                if candidate_path.is_file():
                    pre_rendered_audio_path = candidate_path
                else:
                    db_relative = Path(self.knowledge_base.db_path).parent / candidate_path
                    if db_relative.is_file():
                        pre_rendered_audio_path = db_relative
        else:
            if self._should_skip_direct_llm_for_noisy_query(routed_query, language_confidence, language_reason):
                response_text = self._repeat_request_phrase(language)
                resolved_source = "fallback"
                llm_status = "skipped_noisy_query"
            elif routed_query_token_count >= max(1, self.llm_direct_min_query_tokens):
                llm_status = "attempted"
                llm_started_at = time.perf_counter()
                llm_result = await self._answer_direct_with_thinking_cue(
                    language,
                    routed_query,
                    thinking_cue_delay_seconds,
                    thinking_cue_callback,
                )
                raw_answer = llm_result.answer or ""
                if "[NOISE_REPROMPT]" in raw_answer:
                    response_text = self._noise_reprompt_phrase(language)
                    llm_status = "noise_detected"
                    llm_failure_type = "noise_reprompt"
                    llm_elapsed_ms = (time.perf_counter() - llm_started_at) * 1000.0
                    resolved_source = "fallback"
                else:
                    response_text = self._compact_direct_llm_response(raw_answer, routed_query) if raw_answer else ""
                    llm_status = llm_result.status
                    llm_failure_type = llm_result.failure_type
                    llm_elapsed_ms = (time.perf_counter() - llm_started_at) * 1000.0
                    resolved_source = "llm_direct" if response_text else "fallback"
                if response_text and self._is_greenwich_context_query(routed_query) and llm_status != "noise_detected":
                    self._update_greenwich_llm_context()
                if not response_text:
                    if self._should_repeat_after_failed_direct_llm(
                        routed_query,
                        language_confidence,
                        language_reason,
                    ):
                        response_text = self._repeat_request_phrase(language)
                    elif self._is_greenwich_context_query(routed_query):
                        response_text = self._greenwich_failed_llm_fallback_phrase(language)
                    else:
                        response_text = self._fallback_phrase(language)
            else:
                response_text = self._fallback_phrase(language)
                resolved_source = "fallback"
                llm_status = "skipped_token_guard"

        if pre_rendered_audio_path is not None:
            output_path = pre_rendered_audio_path
            tts_elapsed_ms = 0.0
            LOGGER.info("Using pre-rendered cached audio from database: %s", output_path)
        else:
            output_path = self.output_dir / f"response_{uuid4().hex}.mp3"
            tts_started_at = time.perf_counter()
            await self._render_audio_with_fallback(response_text, language, output_path)
            tts_elapsed_ms = (time.perf_counter() - tts_started_at) * 1000.0
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        if stt_elapsed_ms is not None:
            stt_label = f"{stt_elapsed_ms:.0f}"
        else:
            stt_label = "n/a"
        llm_label = f"{llm_elapsed_ms:.0f}" if llm_elapsed_ms is not None else "n/a"
        LOGGER.info(
            "Stage timings (ms): stt=%s db=%.0f llm=%s tts=%.0f total=%.0f source=%s db_score=%s db_mode=%s",
            stt_label,
            db_elapsed_ms,
            llm_label,
            tts_elapsed_ms,
            elapsed_ms,
            resolved_source,
            f"{db_score:.2f}" if db_score is not None else "n/a",
            db_mode or "n/a",
        )
        LOGGER.info(
            "Resolved transcript via %s in %.0fms (chars=%d).",
            resolved_source,
            elapsed_ms,
            len(transcription),
        )
        if used_context_link and routed_query != transcription:
            LOGGER.info('Context-linked routing query: "%s"', routed_query)
        if resolved_source == "fallback":
            LOGGER.info(
                "Fallback routing diagnostics: llm_status=%s llm_failure_type=%s "
                "tokens=%d min_tokens=%d language=%s language_confidence=%s "
                "language_reason=%s db_score=%s routed_query=%r",
                llm_status,
                llm_failure_type or "n/a",
                routed_query_token_count,
                max(1, self.llm_direct_min_query_tokens),
                language,
                f"{language_confidence:.2f}" if language_confidence is not None else "n/a",
                language_reason or "n/a",
                f"{db_score:.2f}" if db_score is not None else "n/a",
                routed_query,
            )
        return {
            "detected_language": language,
            "transcription": transcription,
            "resolved_source": resolved_source,  # type: ignore[typeddict-item]
            "response_text": response_text,
            "audio_output_path": str(output_path),
        }

    async def process_transcription_stream(
        self,
        transcription: str,
        language: LanguageCode,
        stt_elapsed_ms: float | None = None,
        language_confidence: float | None = None,
        language_reason: str | None = None,
    ):
        """Resolve transcript and yield output audio paths as they are synthesized chunk-by-chunk."""

        started_at = time.perf_counter()
        self.knowledge_base.ensure_schema()
        transcription = transcription.strip()
        transcription, applied_cheats = apply_transcript_cheats(transcription, self.transcript_cheats)
        if applied_cheats:
            LOGGER.info("Transcript cheats applied (%s): %s", ", ".join(applied_cheats), transcription)
        transcription, guarded_corrections = self._correct_guarded_greenwich_misrecognitions(transcription)
        if guarded_corrections:
            LOGGER.info("Guarded Greenwich corrections applied (%s): %s", ", ".join(guarded_corrections), transcription)
        transcription, context_corrections = self._apply_greenwich_llm_context(transcription)
        if context_corrections:
            LOGGER.info("Greenwich LLM context applied (%s): %s", ", ".join(context_corrections), transcription)
        self._advance_context_anchor()
        clarification_text = self._clarification_phrase_for_query(transcription, language)
        if clarification_text:
            db_match = None
            routed_query = transcription
            used_context_link = False
        else:
            db_match, routed_query, used_context_link = self._lookup_response_with_context(
                transcription,
                language,
                language_confidence=language_confidence,
                language_reason=language_reason,
            )

        routed_query_token_count = self._query_token_count(routed_query)
        local_db_decision = (
            self._decide_local_db_routing(db_match, routed_query, language_confidence, language_reason)
            if db_match
            else _LocalDbRoutingDecision("reject_to_llm")
        )

        if clarification_text:
            output_path = self.output_dir / f"response_stream_{uuid4().hex}.mp3"
            await self._render_audio_with_fallback(clarification_text, language, output_path)
            yield {
                "detected_language": language,
                "transcription": transcription,
                "resolved_source": "fallback",
                "response_text": clarification_text,
                "audio_output_path": str(output_path),
                "is_final_segment": True,
            }
            return
        elif local_db_decision.action == "reprompt_noisy":
            phrase = self._repeat_request_phrase(language)
            output_path = self.output_dir / f"response_stream_{uuid4().hex}.mp3"
            await self._render_audio_with_fallback(phrase, language, output_path)
            yield {
                "detected_language": language,
                "transcription": transcription,
                "resolved_source": "fallback",
                "response_text": phrase,
                "audio_output_path": str(output_path),
                "is_final_segment": True,
            }
            return
        elif db_match and local_db_decision.action == "accept_local_db":
            response_text = self._compact_local_db_response(db_match.response, routed_query, db_match, language)
            self._update_context_anchor(db_match)
            pre_rendered_audio_path = None
            if getattr(db_match, "audio_path", None):
                candidate_path = Path(db_match.audio_path)
                if candidate_path.is_file():
                    pre_rendered_audio_path = candidate_path
                else:
                    db_relative = Path(self.knowledge_base.db_path).parent / candidate_path
                    if db_relative.is_file():
                        pre_rendered_audio_path = db_relative
            if pre_rendered_audio_path is not None:
                output_path = pre_rendered_audio_path
            else:
                output_path = self.output_dir / f"response_stream_{uuid4().hex}.mp3"
                await self._render_audio_with_fallback(response_text, language, output_path)
            yield {
                "detected_language": language,
                "transcription": transcription,
                "resolved_source": "local_db",
                "response_text": response_text,
                "audio_output_path": str(output_path),
                "is_final_segment": True,
            }
            return

        if self._should_skip_direct_llm_for_noisy_query(routed_query, language_confidence, language_reason):
            phrase = self._repeat_request_phrase(language)
            output_path = self.output_dir / f"response_stream_{uuid4().hex}.mp3"
            await self._render_audio_with_fallback(phrase, language, output_path)
            yield {
                "detected_language": language,
                "transcription": transcription,
                "resolved_source": "fallback",
                "response_text": phrase,
                "audio_output_path": str(output_path),
                "is_final_segment": True,
            }
            return

        if routed_query_token_count < max(1, self.llm_direct_min_query_tokens):
            phrase = self._fallback_phrase(language)
            output_path = self.output_dir / f"response_stream_{uuid4().hex}.mp3"
            await self._render_audio_with_fallback(phrase, language, output_path)
            yield {
                "detected_language": language,
                "transcription": transcription,
                "resolved_source": "fallback",
                "response_text": phrase,
                "audio_output_path": str(output_path),
                "is_final_segment": True,
            }
            return

        text_generator = self.llm_provider.generate_answer_stream(language, routed_query)
        buffer = ""
        sentence_index = 0
        noise_reprompt_detected = False
        pending_segment = None

        async for chunk in text_generator:
            buffer += chunk
            sentences = self._split_spoken_sentences(buffer)
            if not sentences:
                continue

            last_complete = bool(re.search(r"[.!?]\s*$", sentences[-1]))
            if last_complete:
                target_sentences = sentences
                buffer = ""
            else:
                target_sentences = sentences[:-1]
                buffer = sentences[-1]

            for sentence in target_sentences:
                sentence_text = sentence.strip()
                if not sentence_text:
                    continue

                if sentence_index == 0 and "[NOISE_REPROMPT]" in sentence_text:
                    noise_reprompt_detected = True
                    break

                if sentence_index >= 2:
                    break

                sentence_index += 1
                cleaned_sentence = self._compact_direct_llm_response(sentence_text, routed_query)
                if not cleaned_sentence:
                    continue

                output_path = self.output_dir / f"response_stream_{uuid4().hex}.mp3"
                await self._render_audio_with_fallback(cleaned_sentence, language, output_path)

                if pending_segment:
                    yield {**pending_segment, "is_final_segment": False}

                pending_segment = {
                    "detected_language": language,
                    "transcription": transcription,
                    "resolved_source": "llm_direct",
                    "response_text": cleaned_sentence,
                    "audio_output_path": str(output_path),
                }

            if noise_reprompt_detected or sentence_index >= 2:
                break

        if not noise_reprompt_detected and sentence_index < 2 and buffer.strip():
            sentence_text = buffer.strip()
            if sentence_index == 0 and "[NOISE_REPROMPT]" in sentence_text:
                noise_reprompt_detected = True
            else:
                cleaned_sentence = self._compact_direct_llm_response(sentence_text, routed_query)
                if cleaned_sentence:
                    sentence_index += 1
                    output_path = self.output_dir / f"response_stream_{uuid4().hex}.mp3"
                    await self._render_audio_with_fallback(cleaned_sentence, language, output_path)

                    if pending_segment:
                        yield {**pending_segment, "is_final_segment": False}

                    pending_segment = {
                        "detected_language": language,
                        "transcription": transcription,
                        "resolved_source": "llm_direct",
                        "response_text": cleaned_sentence,
                        "audio_output_path": str(output_path),
                    }

        if noise_reprompt_detected:
            phrase = self._noise_reprompt_phrase(language)
            output_path = self.output_dir / f"response_stream_{uuid4().hex}.mp3"
            await self._render_audio_with_fallback(phrase, language, output_path)
            yield {
                "detected_language": language,
                "transcription": transcription,
                "resolved_source": "fallback",
                "response_text": phrase,
                "audio_output_path": str(output_path),
                "is_final_segment": True,
            }
            return

        if pending_segment:
            yield {**pending_segment, "is_final_segment": True}
        else:
            phrase = self._fallback_phrase(language)
            output_path = self.output_dir / f"response_stream_{uuid4().hex}.mp3"
            await self._render_audio_with_fallback(phrase, language, output_path)
            yield {
                "detected_language": language,
                "transcription": transcription,
                "resolved_source": "fallback",
                "response_text": phrase,
                "audio_output_path": str(output_path),
                "is_final_segment": True,
            }

    def _compact_local_db_response(
        self,
        response_text: str,
        query: str = "",
        db_match: KnowledgeMatch | None = None,
        language: LanguageCode | None = None,
    ) -> str:
        """Shorten verbose local answers while keeping the leading factual content."""

        if db_match is not None:
            override = self._spoken_answer_override(db_match.source_id, db_match.response, db_match, language, query)
            if override:
                return override

        normalized = self._normalize_spoken_response(response_text)
        if not normalized:
            return normalized

        sentences = self._split_spoken_sentences(normalized)
        if len(sentences) > 1:
            normalized = self._select_spoken_answer_sentence(sentences, query)

        if len(normalized) <= 180:
            return normalized

        complete_sentence = self._first_complete_sentence(normalized)
        if complete_sentence:
            return complete_sentence
        trimmed = normalized[:180].rsplit(" ", 1)[0].rstrip(".,;:-")
        if self._has_incomplete_spoken_ending(trimmed):
            return complete_sentence or normalized
        return trimmed or normalized[:180].rstrip()

    @staticmethod
    def _spoken_answer_override(
        source_id: str | None,
        response_text: str,
        db_match: KnowledgeMatch,
        language: LanguageCode | None,
        query: str = "",
    ) -> str:
        """Return curated spoken summaries for rows that are too long for voice UX."""

        if not source_id:
            return ""
        question_text = db_match.question or ""
        response_has_vietnamese = bool(re.search(r"[À-ỹĐđ]", response_text))
        is_vi = language == "vi" or response_has_vietnamese or bool(re.search(r"[À-ỹĐđ]", question_text))
        normalized_query = VoicePipeline._normalize_for_context(query)
        query_tokens = set(_TOKEN_RE.findall(normalized_query))
        asks_graduation = "graduation" in query_tokens or {"chuan", "dau", "ra"} <= query_tokens
        asks_ielts = "ielts" in query_tokens or "mandatory" in query_tokens or {"bat", "buoc"} <= query_tokens
        overrides = {
            ("web-004", False): (
                "Greenwich Vietnam offers Information Technology, Graphic and Digital Design, Business Administration, "
                "Marketing Management, Event Management, Communication Management, International Business, Logistics "
                "and Supply Chain Management, and Multimedia Communication."
            ),
            ("web-004", True): (
                "Greenwich Việt Nam đào tạo Công nghệ thông tin, Thiết kế đồ họa và kỹ thuật số, Quản trị kinh doanh, "
                "Marketing, Sự kiện, Truyền thông, Kinh doanh quốc tế, Logistics và Truyền thông đa phương tiện."
            ),
            ("web-006", False): "Bachelor's programs last 3 years, and master's programs last 1.5 years.",
            ("web-006", True): "Chuong trinh cu nhan keo dai 3 nam, va chuong trinh thac si keo dai 1.5 nam.",
            ("web-009", False): (
                "Graduation requirements include specialized knowledge, professional skills, and graduation English proficiency."
            ),
            ("web-009-combined", False): (
                "Graduation requirements include specialized knowledge, professional skills, and graduation English proficiency. "
                "IELTS is not mandatory at admission; students may submit IELTS or use the integrated English pathway."
            ),
            ("web-009-ielts", False): (
                "IELTS is not mandatory at admission; students may submit IELTS if available or use the integrated English pathway."
            ),
            ("web-009", True): (
                "Chuẩn tốt nghiệp gồm kiến thức chuyên môn, kỹ năng nghề nghiệp và năng lực tiếng Anh tốt nghiệp."
            ),
            ("web-009-combined", True): (
                "Chuẩn tốt nghiệp gồm kiến thức chuyên môn, kỹ năng nghề nghiệp và năng lực tiếng Anh tốt nghiệp. "
                "IELTS không bắt buộc khi đầu vào; sinh viên có thể nộp IELTS hoặc học lộ trình tiếng Anh tích hợp."
            ),
            ("web-009-ielts", True): (
                "IELTS không bắt buộc khi đầu vào; sinh viên có thể nộp IELTS nếu có hoặc học lộ trình tiếng Anh tích hợp."
            ),
            ("web-012", False): "Applicants need IELTS 6.0 or an equivalent Level 4/6 English certificate.",
            ("web-012", True): "Thí sinh cần IELTS 6.0 trở lên hoặc chứng chỉ tiếng Anh tương đương Level 4/6.",
            ("web-014", False): "The total tuition is 450,000,000 VND, based on 50,000,000 VND per semester for 9 semesters.",
            ("web-014", True): "Học phí toàn khóa là 450.000.000 VNĐ, tính theo 50.000.000 VNĐ mỗi học kỳ trong 9 học kỳ.",
            ("web-017", False): "Greenwich Vietnam has 4 campuses: Hanoi, Da Nang, Ho Chi Minh City, and Can Tho.",
            ("web-017", True): "Greenwich Việt Nam có 4 cơ sở tại Hà Nội, Đà Nẵng, Thành phố Hồ Chí Minh và Cần Thơ.",
            ("web-033", False): "Tuition is paid per semester. Detailed fee links are available in the log.",
            ("web-033", True): "Học phí được đóng theo từng học kỳ. Đường dẫn chi tiết có trong log.",
        }
        if source_id == "web-009" and asks_graduation and asks_ielts:
            return overrides.get(("web-009-combined", is_vi), "")
        return overrides.get((source_id, is_vi), "")

    @staticmethod
    def _normalize_spoken_response(response_text: str) -> str:
        """Normalize DB text for spoken output without changing stored DB rows."""

        protected = response_text.replace("TP.HCM", "TP<DOT>HCM").replace("e.g.", "e<DOT>g<DOT>")
        protected = re.sub(r"(\d)\.(\d)", r"\1<DECIMAL_DOT>\2", protected)
        without_urls = re.sub(r"https?://\S+", "", protected)
        fixed_spacing = re.sub(r"(?<=[.!?])(?=[^\s])", " ", without_urls)
        without_markdown_bullets = re.sub(r"(^|\s)[*\u2022-]\s+", r"\1", fixed_spacing)
        without_bold = re.sub(r"\*\*|__", "", without_markdown_bullets)
        without_italic = re.sub(r"\*|_", "", without_bold)
        without_headers = re.sub(r"#+\s+", "", without_italic)
        flattened = re.sub(r"[\r\n]+", " ", without_headers)
        flattened = re.sub(r"\s+", " ", flattened).strip(" -")
        return flattened.replace("<DECIMAL_DOT>", ".").replace("<DOT>", ".")

    def _first_complete_sentence(self, text: str) -> str:
        """Return the first sentence ending cleanly with punctuation."""

        sentences = self._split_spoken_sentences(text)
        return sentences[0] if sentences else ""

    @staticmethod
    def _split_spoken_sentences(text: str) -> list[str]:
        """Split spoken text while protecting decimals and common abbreviations."""

        protected = text.replace("TP.HCM", "TP<DOT>HCM").replace("e.g.", "e<DOT>g<DOT>")
        protected = re.sub(r"(\d)\.(\d)", r"\1<DECIMAL_DOT>\2", protected)
        sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", protected) if sentence.strip()]
        return [sentence.replace("<DECIMAL_DOT>", ".").replace("<DOT>", ".") for sentence in sentences]

    def _select_spoken_answer_sentence(self, sentences: list[str], query: str) -> str:
        """Prefer the sentence that directly answers the user's intent."""

        query_tokens = set(_TOKEN_RE.findall(self._normalize_for_context(query)))
        if not query_tokens:
            return sentences[0].strip()

        answer_signal_tokens = {
            "ielts",
            "level",
            "proficiency",
            "requirement",
            "required",
            "campus",
            "campuses",
            "co",
            "so",
            "hoc",
            "bong",
            "scholarship",
            "fee",
            "tuition",
            "vnd",
        }
        scored: list[tuple[float, int, str]] = []
        for index, sentence in enumerate(sentences):
            sentence_tokens = set(_TOKEN_RE.findall(self._normalize_for_context(sentence)))
            overlap = len(query_tokens & sentence_tokens)
            signal_overlap = len(answer_signal_tokens & query_tokens & sentence_tokens)
            numeric_bonus = 1 if re.search(r"\d", sentence) else 0
            score = overlap + (2.0 * signal_overlap) + numeric_bonus - (0.15 * index)
            scored.append((score, -index, sentence.strip()))
        return max(scored, key=lambda item: (item[0], item[1]))[2]

    def _lookup_response_with_context(
        self,
        query: str,
        language: LanguageCode,
        language_confidence: float | None = None,
        language_reason: str | None = None,
    ) -> tuple[KnowledgeMatch | None, str, bool]:
        """Run primary retrieval and optionally a context-linked expansion pass."""

        primary_match = self.knowledge_base.lookup_response_details(query, language)
        intent_source_id = self._domain_intent_source_id(query, language)
        if intent_source_id and not self._should_disable_broad_intent_shortcut(
            query,
            language_confidence,
            language_reason,
        ):
            intent_source_match = self.knowledge_base.lookup_response_by_source_details(intent_source_id, language)
            if intent_source_match is not None:
                return intent_source_match, query, False

        intent_query = self._expand_domain_intent_query(query, language)
        if intent_query != query:
            intent_match = self.knowledge_base.lookup_response_details(intent_query, language)
            if intent_match is not None and (
                primary_match is None or intent_match.score >= primary_match.score
            ):
                primary_match = intent_match
                query = intent_query
        if not self.context_link_enabled:
            return primary_match, query, False
        if self._context_anchor is None:
            return primary_match, query, False
        if not self._is_context_link_allowed(language_confidence, language_reason):
            LOGGER.info(
                "Context link skipped: language_confidence=%s language_reason=%s query=%r",
                f"{language_confidence:.2f}" if language_confidence is not None else "n/a",
                language_reason or "n/a",
                query,
            )
            return primary_match, query, False
        if not self._is_followup_candidate(query):
            return primary_match, query, False

        context_phrase = self._context_phrase_for_language(language)
        if not context_phrase:
            return primary_match, query, False

        expanded_query = f"{query.strip()} {context_phrase}".strip()
        if expanded_query == query:
            return primary_match, query, False

        expanded_match = self.knowledge_base.lookup_response_details(expanded_query, language)
        if expanded_match is None:
            return primary_match, query, False

        primary_score = primary_match.score if primary_match is not None else 0.0
        if expanded_match.score < (primary_score + self.context_link_min_score_delta):
            return primary_match, query, False

        return expanded_match, expanded_query, True

    def _expand_domain_intent_query(self, query: str, language: LanguageCode) -> str:
        """Recover common Greenwich intents when STT clips the opening words."""

        normalized = self._normalize_for_context(query)
        tokens = set(_TOKEN_RE.findall(normalized))
        if not tokens:
            return query

        expansions: list[str] = []
        if language == "vi":
            if {"co", "so"} <= tokens or "campus" in tokens:
                expansions.append("greenwich viet nam co bao nhieu co so")
            if {"hoc", "bong"} <= tokens or {"chinh", "sach", "bong"} <= tokens:
                expansions.append("greenwich viet nam co chinh sach hoc bong khong")
            if {"tieng", "anh"} <= tokens and ({"trinh", "do"} & tokens or {"yeu", "cau"} <= tokens):
                expansions.append("trinh do tieng anh dau vao yeu cau la bao nhieu")
        else:
            if "campus" in tokens or "campuses" in tokens:
                expansions.append("how many campuses does greenwich vietnam have")
            if "scholarship" in tokens or "scholarships" in tokens:
                expansions.append("does greenwich vietnam have a scholarship policy")
            if {"english", "proficiency"} <= tokens or {"ielts", "level"} & tokens:
                expansions.append("what is the required english proficiency level for admission")

        if not expansions:
            return query
        return f"{query.strip()} {' '.join(expansions)}".strip()

    def _domain_intent_source_id(self, query: str, language: LanguageCode) -> str | None:
        """Pin common FAQ intents so generic Greenwich keywords cannot win."""

        normalized = self._normalize_for_context(query)
        tokens = set(_TOKEN_RE.findall(normalized))
        if not tokens:
            return None
        if self._is_greenwich_conversational_intent(tokens):
            return None

        if language == "vi":
            asks_graduation = {"chuan", "dau", "ra"} <= tokens or "tot" in tokens and "nghiep" in tokens
            asks_ielts = "ielts" in tokens or {"bat", "buoc"} <= tokens
            if {"hoc", "phi"} <= tokens or {"toan", "khoa"} <= tokens:
                return "web-014"
            if {"hoc", "bong"} <= tokens or {"chinh", "sach", "bong"} <= tokens:
                return "web-015"
            if {"co", "so"} <= tokens or "campus" in tokens:
                return "web-017"
            if {"du", "hoc"} <= tokens or {"trao", "doi"} <= tokens:
                return "web-020"
            if {"dieu", "kien", "xet", "tuyen"} <= tokens or {"xet", "tuyen"} <= tokens:
                return "web-011"
            if asks_ielts and not asks_graduation:
                return "web-009-ielts"
            if asks_graduation:
                return "web-009"
            if {"tieng", "anh"} <= tokens and ({"dau", "vao"} & tokens or {"trinh", "do"} & tokens):
                return "web-012"
            if {"chuyen", "nganh"} <= tokens or {"dao", "tao", "nganh"} <= tokens:
                return "web-004"
            if {"nganh", "hoc"} <= tokens or ("nganh" in tokens and {"nao", "nhung", "co"} & tokens):
                return "web-004"
            if {"thoi", "gian"} <= tokens or {"bao", "lau"} <= tokens:
                return "web-006"
            if self._is_greenwich_identity_query(tokens):
                return "web-001"
            return None

        if {"how", "much", "semester"} <= tokens:
            return "web-014"
        if {"tuition", "fee"} & tokens or {"total", "cost"} <= tokens:
            return "web-014"
        if "scholarship" in tokens or "scholarships" in tokens:
            return "web-015"
        if "campus" in tokens or "campuses" in tokens:
            return "web-017"
        if {"study", "abroad"} <= tokens or "exchange" in tokens or {"studying", "abroad"} <= tokens:
            return "web-020"
        asks_graduation = "graduation" in tokens
        asks_ielts = "ielts" in tokens and ("mandatory" in tokens or "required" in tokens)
        if asks_ielts and not asks_graduation:
            return "web-009-ielts"
        if asks_graduation:
            return "web-009"
        if {"admission", "requirement"} <= tokens or {"admission", "requirements"} <= tokens:
            return "web-011"
        if {"english", "proficiency"} <= tokens or {"ielts", "level"} & tokens:
            return "web-012"
        if "majors" in tokens or {"what", "major"} <= tokens:
            return "web-004"
        if "duration" in tokens or {"how", "long"} <= tokens:
            return "web-006"
        if self._is_greenwich_identity_query(tokens):
            return "web-001"
        return None

    @staticmethod
    def _is_context_link_allowed(language_confidence: float | None, language_reason: str | None) -> bool:
        """Avoid carrying topic anchors through uncertain language-routing turns."""

        if language_confidence is None:
            return True
        if language_confidence < 0.60:
            return False
        reason_base = (language_reason or "").split("+", 1)[0]
        return reason_base not in {"no_marker_hit", "marker_tie"}

    def _advance_context_anchor(self) -> None:
        """Decay short-lived anchor state once per turn."""

        if self._greenwich_llm_context_turns_remaining > 0:
            self._greenwich_llm_context_turns_remaining -= 1
        if self._context_anchor is not None:
            self._context_anchor.turns_remaining -= 1
            if self._context_anchor.turns_remaining <= 0:
                self._context_anchor = None

    def _update_greenwich_llm_context(self) -> None:
        """Keep a short Greenwich anchor from successful conversational LLM turns."""

        self._greenwich_llm_context_turns_remaining = self.context_link_max_turn_gap

    def _update_context_anchor(self, db_match: KnowledgeMatch) -> None:
        """Persist a short topic anchor from high-confidence local matches."""

        topic_phrase = self._derive_topic_phrase(db_match.matched_keyword or db_match.question or "")
        if not topic_phrase:
            return
        self._context_anchor = _ContextAnchor(
            source_id=db_match.source_id,
            topic_phrase=topic_phrase,
            turns_remaining=self.context_link_max_turn_gap,
        )

    def _context_phrase_for_language(self, language: LanguageCode) -> str:
        """Return anchor phrase for the current language, using source ID when available."""

        if self._context_anchor is None:
            return ""

        if self._context_anchor.source_id:
            question_text = self.knowledge_base.lookup_question_by_source(self._context_anchor.source_id, language)
            if question_text:
                from_source_question = self._derive_topic_phrase(question_text)
                if from_source_question:
                    return from_source_question

        return self._context_anchor.topic_phrase

    def _is_followup_candidate(self, query: str) -> bool:
        """Return whether the turn likely depends on prior context."""

        tokens = _TOKEN_RE.findall(self._normalize_for_context(query))
        if not tokens:
            return False
        if len(tokens) > self.context_link_short_query_max_tokens:
            return False
        blocked_context_terms = {
            "ai",
            "america",
            "bank",
            "bong",
            "entanglement",
            "football",
            "latest",
            "manchester",
            "news",
            "quantum",
            "thang",
            "tran",
            "united",
            "weather",
        }
        if set(tokens) & blocked_context_terms:
            return False

        followup_markers = {
            "cost",
            "price",
            "fee",
            "tuition",
            "this",
            "that",
            "it",
            "also",
            "chi",
            "phi",
            "hoc",
            "bao",
            "nhieu",
            "them",
            "nay",
            "kia",
        }
        return any(token in followup_markers for token in tokens)

    def _apply_greenwich_llm_context(self, query: str) -> tuple[str, tuple[str, ...]]:
        """Recover short Greenwich demo follow-ups after a successful Greenwich LLM turn."""

        if self._greenwich_llm_context_turns_remaining <= 0:
            return query, ()
        tokens = set(_TOKEN_RE.findall(self._normalize_for_context(query)))
        if not tokens or self._is_greenwich_context_query(query):
            return query, ()
        if self._is_current_information_query(query):
            return query, ()
        if not self._is_greenwich_conversational_intent(tokens):
            return query, ()
        if not ({"viet", "nam"} <= tokens or "vietnam" in tokens):
            return query, ()
        return f"Greenwich {query}".strip(), ("recent_greenwich_llm_context",)

    @staticmethod
    def _is_greenwich_conversational_intent(tokens: set[str]) -> bool:
        """Return whether a Greenwich question should stay live/advisory instead of FAQ-pinned."""

        return bool(
            {"why", "choose"} <= tokens
            or {"khac", "gi"} <= tokens
            or {"diem", "manh"} <= tokens
            or {"phu", "hop"} <= tokens
            or tokens
            & {
                "advisor",
                "different",
                "difference",
                "fit",
                "recommend",
                "strong",
                "strength",
                "strengths",
                "suitable",
            }
        )

    def _should_accept_local_db_match(
        self,
        db_match: KnowledgeMatch,
        query: str,
        language_confidence: float | None = None,
        language_reason: str | None = None,
    ) -> bool:
        """Return whether a DB match is strong enough for direct spoken routing."""

        return (
            self._decide_local_db_routing(db_match, query, language_confidence, language_reason).action
            == "accept_local_db"
        )

    def _decide_local_db_routing(
        self,
        db_match: KnowledgeMatch,
        query: str,
        language_confidence: float | None = None,
        language_reason: str | None = None,
    ) -> _LocalDbRoutingDecision:
        """Return the local-DB routing decision for a retrieval candidate."""

        query_tokens = set(_TOKEN_RE.findall(self._normalize_for_context(query)))
        if self._is_greenwich_context_query(query) and self._is_greenwich_conversational_intent(query_tokens):
            self._log_rejected_local_db_match(db_match, query, "greenwich_conversational_intent")
            return _LocalDbRoutingDecision("reject_to_llm", "greenwich_conversational_intent")
        if db_match.score < self.knowledge_base.confidence_low:
            return _LocalDbRoutingDecision("reject_to_llm", "below_confidence")
        if db_match.retrieval_mode.endswith("_intent"):
            if self._should_disable_broad_intent_shortcut(query, language_confidence, language_reason):
                self._log_rejected_local_db_match(db_match, query, "suppressed_noisy_domain_intent")
                return _LocalDbRoutingDecision("reprompt_noisy", "suppressed_noisy_domain_intent")
            return _LocalDbRoutingDecision("accept_local_db", "canonical_intent")
        domain_intent_source_id = self._domain_intent_source_id(query, "en") or self._domain_intent_source_id(query, "vi")
        if domain_intent_source_id and not self._should_disable_broad_intent_shortcut(
            query,
            language_confidence,
            language_reason,
        ):
            return _LocalDbRoutingDecision("accept_local_db", "domain_intent")
        if db_match.source_id == "web-001" and not self._is_greenwich_identity_query(
            set(_TOKEN_RE.findall(self._normalize_for_context(query)))
        ):
            self._log_rejected_local_db_match(db_match, query, "non_identity_overview_match")
            return _LocalDbRoutingDecision("reject_to_llm", "non_identity_overview_match")

        if self._should_reprompt_long_noisy_db_capture(
            db_match,
            query,
            language_confidence,
            language_reason,
        ):
            return _LocalDbRoutingDecision("reprompt_noisy", "long_noisy_capture")

        if db_match.whole_phrase_match and db_match.matched_keyword_token_count >= 2:
            return _LocalDbRoutingDecision("accept_local_db", "whole_phrase")

        is_strong_phrase = db_match.whole_phrase_match and db_match.matched_keyword_token_count >= 2
        if not is_strong_phrase:
            required_coverage = 0.35 if not db_match.whole_phrase_match else 0.20
            if db_match.query_coverage < required_coverage:
                self._log_rejected_local_db_match(db_match, query, "low_query_coverage")
                return _LocalDbRoutingDecision("reject_to_llm", "low_query_coverage")

        if db_match.fuzzy_hit_count > db_match.exact_hit_count:
            self._log_rejected_local_db_match(db_match, query, "mostly_fuzzy_evidence")
            return _LocalDbRoutingDecision("reject_to_llm", "mostly_fuzzy_evidence")

        if db_match.exact_hit_count >= 2 and db_match.keyword_coverage >= 0.60:
            if db_match.score_margin <= 0.01 and not db_match.whole_phrase_match:
                self._log_rejected_local_db_match(db_match, query, "ambiguous_score_tie")
                return _LocalDbRoutingDecision("reject_to_llm", "ambiguous_score_tie")
            return _LocalDbRoutingDecision("accept_local_db", "token_evidence")

        self._log_rejected_local_db_match(db_match, query, "weak_token_evidence")
        return _LocalDbRoutingDecision("reject_to_llm", "weak_token_evidence")

    def _should_reprompt_long_noisy_db_capture(
        self,
        db_match: KnowledgeMatch,
        query: str,
        language_confidence: float | None,
        language_reason: str | None,
    ) -> bool:
        """Return whether a long transcript is too noisy for DB acceptance."""

        tokens = _TOKEN_RE.findall(self._normalize_for_context(query))
        if len(tokens) < 8:
            return False
        if db_match.retrieval_mode.endswith("_intent"):
            return False
        if db_match.whole_phrase_match and db_match.matched_keyword_token_count >= 3 and db_match.query_coverage >= 0.25:
            return False
        if db_match.exact_hit_count >= 3 and db_match.keyword_coverage >= 0.75 and db_match.query_coverage >= 0.25:
            return False

        reason_base = (language_reason or "").split("+", 1)[0]
        uncertain_language = language_confidence is not None and language_confidence < 0.62
        uncertain_language = uncertain_language or reason_base in {"marker_tie", "no_marker_hit"}
        weak_match_inside_long_query = (
            db_match.query_coverage < 0.35
            or db_match.exact_hit_count <= 2
            or db_match.matched_keyword_token_count <= 2
            or db_match.fuzzy_hit_count > 0
        )
        if uncertain_language and weak_match_inside_long_query:
            self._log_rejected_local_db_match(db_match, query, "long_noisy_capture")
            return True
        return False

    @classmethod
    def _should_disable_broad_intent_shortcut(
        cls,
        query: str,
        language_confidence: float | None,
        language_reason: str | None,
    ) -> bool:
        """Return whether a broad intent shortcut should yield to retrieval evidence."""

        tokens = _TOKEN_RE.findall(cls._normalize_for_context(query))
        if len(tokens) < 8:
            return False
        reason_base = (language_reason or "").split("+", 1)[0]
        uncertain_language = language_confidence is not None and language_confidence < 0.62
        return uncertain_language or reason_base in {"marker_tie", "no_marker_hit"}

    @staticmethod
    def _log_rejected_local_db_match(db_match: KnowledgeMatch, query: str, reason: str) -> None:
        """Log why a candidate local DB match was withheld from spoken routing."""

        LOGGER.info(
            'Local DB match rejected: reason=%s score=%.2f source_id=%s exact=%d fuzzy=%d '
            'keyword_coverage=%.2f query_coverage=%.2f margin=%.2f keyword="%s" query="%s"',
            reason,
            db_match.score,
            db_match.source_id or "",
            db_match.exact_hit_count,
            db_match.fuzzy_hit_count,
            db_match.keyword_coverage,
            db_match.query_coverage,
            db_match.score_margin,
            db_match.matched_keyword,
            query,
        )

    @staticmethod
    def _has_explicit_greenwich_mention(tokens: set[str]) -> bool:
        """Return whether tokens explicitly name Greenwich despite common STT splits."""

        return "greenwich" in tokens or {"green", "which"} <= tokens

    @classmethod
    def _is_greenwich_identity_query(cls, tokens: set[str]) -> bool:
        """Return whether tokens ask for the Greenwich Vietnam overview row."""

        if not cls._has_explicit_greenwich_mention(tokens):
            return False
        if "vietnam" not in tokens and not {"viet", "nam"} <= tokens:
            return False

        subject_tokens = {"greenwich", "green", "which", "vietnam", "viet", "nam"}
        extra_tokens = tokens - subject_tokens
        if not extra_tokens:
            return False
        return extra_tokens <= {"what", "is"} or extra_tokens <= {"tell", "me", "about"} or extra_tokens <= {
            "la",
            "gi",
        }

    def _derive_topic_phrase(self, text: str) -> str:
        """Extract a compact topic phrase from matched keyword/question text."""

        normalized = self._normalize_for_context(text)
        tokens = _TOKEN_RE.findall(normalized)
        if not tokens:
            return ""

        token_set = set(tokens)
        if "greenwich" in token_set and ("vietnam" in token_set or ("viet" in token_set and "nam" in token_set)):
            return "greenwich vietnam"

        stopwords = {
            "what",
            "is",
            "the",
            "a",
            "an",
            "for",
            "of",
            "to",
            "in",
            "at",
            "do",
            "does",
            "how",
            "when",
            "where",
            "who",
            "why",
            "la",
            "gi",
            "co",
            "khong",
            "ve",
            "cho",
            "toi",
            "bao",
            "nhieu",
            "nao",
            "duoc",
        }
        content_tokens = [token for token in tokens if token not in stopwords]
        if len(content_tokens) >= 2:
            return " ".join(content_tokens[:3])
        return content_tokens[0] if content_tokens else ""

    @staticmethod
    def _normalize_for_context(text: str) -> str:
        """Normalize text for robust follow-up heuristics and topic extraction."""

        lowered = text.lower().replace("đ", "d")
        ascii_like = unicodedata.normalize("NFD", lowered).encode("ascii", "ignore").decode("ascii")
        return " ".join(_TOKEN_RE.findall(ascii_like))

    async def _answer_direct_with_timeout(self, language: LanguageCode, question: str) -> _LLMDirectResult:
        """Generate an answer directly from the LLM when search cannot provide snippets."""

        timeout_seconds = self._llm_timeout_for_query(question)
        LOGGER.info(
            "Direct LLM timeout budget: provider=Direct timeout_seconds=%.1f current_query=%s",
            timeout_seconds,
            self._is_current_information_query(question),
        )
        direct_answer = await self._generate_answer_with_provider(
            provider=self.llm_provider,
            provider_label="Direct",
            language=language,
            question=question,
            timeout_seconds=timeout_seconds,
        )
        if direct_answer.answer:
            return direct_answer

        if self.fallback_llm_provider is None:
            return direct_answer

        secondary_answer = await self._generate_answer_with_provider(
            provider=self.fallback_llm_provider,
            provider_label="Secondary",
            language=language,
            question=question,
            timeout_seconds=timeout_seconds,
        )
        if secondary_answer.answer:
            return secondary_answer
        return _LLMDirectResult(
            "",
            secondary_answer.status if secondary_answer.status != "empty_response" else direct_answer.status,
            secondary_answer.failure_type or direct_answer.failure_type,
        )

    async def _answer_direct_with_thinking_cue(
        self,
        language: LanguageCode,
        question: str,
        thinking_cue_delay_seconds: float,
        thinking_cue_callback: Callable[[], Awaitable[None]] | None,
    ) -> _LLMDirectResult:
        """Generate direct LLM answer and optionally play one transition cue while waiting."""

        if thinking_cue_callback is None or thinking_cue_delay_seconds <= 0:
            return await self._answer_direct_with_timeout(language, question)

        answer_task = asyncio.create_task(self._answer_direct_with_timeout(language, question))
        done, _ = await asyncio.wait({answer_task}, timeout=thinking_cue_delay_seconds)
        if answer_task in done:
            return answer_task.result()

        cue_started_at = time.perf_counter()
        try:
            await thinking_cue_callback()
            LOGGER.info(
                "Thinking cue diagnostics: played=true elapsed_ms=%.0f",
                (time.perf_counter() - cue_started_at) * 1000.0,
            )
        except (RuntimeError, OSError, ValueError):
            LOGGER.exception("Thinking cue playback failed; continuing LLM direct wait.")
        return await answer_task

    async def _generate_answer_with_provider(
        self,
        provider: LanguageModelProvider,
        provider_label: str,
        language: LanguageCode,
        question: str,
        timeout_seconds: float,
    ) -> _LLMDirectResult:
        """Run one LLM provider call with timeout and normalized result handling."""

        try:
            answer = await asyncio.wait_for(
                provider.generate_answer(language, question),
                timeout=timeout_seconds,
            )
            normalized_answer = answer.strip()
            if (
                not normalized_answer
                or normalized_answer == self._fallback_phrase(language)
                or self._looks_like_incomplete_llm_fragment(normalized_answer)
            ):
                return _LLMDirectResult("", "empty_response")
            return _LLMDirectResult(normalized_answer, "succeeded")
        except Exception as exc:
            exc_type = type(exc).__name__
            reason = str(exc).strip() or "<no error message>"
            if provider_label == "Direct":
                LOGGER.warning(
                    "Direct LLM fallback failed (type=%s, reason=%s) for question '%s'.",
                    exc_type,
                    reason,
                    question,
                )
            else:
                LOGGER.warning(
                    "Secondary LLM fallback failed (type=%s, reason=%s) for question '%s'.",
                    exc_type,
                    reason,
                    question,
                )
            return _LLMDirectResult("", "failed", exc_type)

    def _llm_timeout_for_query(self, query: str) -> float:
        """Return the live direct-answer budget for the current query shape."""

        requested_budget = 10.0 if self._is_current_information_query(query) or self._is_greenwich_context_query(query) else 8.0
        return max(0.001, min(self.llm_timeout_seconds, requested_budget))

    @classmethod
    def _is_current_information_query(cls, query: str) -> bool:
        """Return whether a query likely needs search/current retrieval."""

        tokens = set(_TOKEN_RE.findall(cls._normalize_for_context(query)))
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

    @staticmethod
    def _query_token_count(query: str) -> int:
        """Return normalized token count used by direct LLM guard diagnostics."""

        return len(_TOKEN_RE.findall(query.lower()))

    def _should_skip_direct_llm_for_noisy_query(
        self,
        query: str,
        language_confidence: float | None,
        language_reason: str | None,
    ) -> bool:
        """Avoid giving authoritative answers to short likely-misheard fragments."""

        tokens = _TOKEN_RE.findall(self._normalize_for_context(query))
        if len(tokens) <= 1:
            return True
        reason_base = (language_reason or "").split("+", 1)[0]
        if language_confidence is not None and language_confidence < 0.60 and len(tokens) <= 3:
            return True
        if reason_base in {"no_marker_hit", "marker_tie"} and len(tokens) <= 3:
            return True

        malformed_price_terms = {"pose", "simulator", "simeta", "persimexer", "pasimetas"}
        if {"how", "much"} <= set(tokens) and malformed_price_terms & set(tokens):
            return True
        return False

    @classmethod
    def _clarification_phrase_for_query(cls, query: str, language: LanguageCode) -> str:
        """Return a targeted clarification for ambiguous entity-only captures."""

        tokens = _TOKEN_RE.findall(cls._normalize_for_context(query))
        token_set = set(tokens)
        if cls._is_bare_greenwich_entity_query(tokens):
            return (
                "Ban muon hoi ve chu de nao cua Greenwich Viet Nam?"
                if language == "vi"
                else "Which Greenwich Vietnam topic do you mean?"
            )
        if token_set == {"what", "is", "english", "vietnam"} or token_set == {"what", "is", "english", "viet", "nam"}:
            return (
                "Ban muon hoi ve Greenwich Viet Nam, hay tieng Anh o Viet Nam?"
                if language == "vi"
                else "Did you mean Greenwich Vietnam, or English in Vietnam?"
            )
        return ""

    @classmethod
    def _correct_guarded_greenwich_misrecognitions(cls, query: str) -> tuple[str, tuple[str, ...]]:
        """Correct long-form Greenwich-like entity misses without overriding clean ambiguity."""

        tokens = _TOKEN_RE.findall(cls._normalize_for_context(query))
        if not tokens:
            return query, ()
        token_set = set(tokens)
        if token_set == {"what", "is", "english", "vietnam"} or token_set == {"what", "is", "english", "viet", "nam"}:
            return query, ()
        if not cls._has_greenwich_domain_context(token_set):
            return query, ()

        corrected = query
        labels: list[str] = []
        corrected, replacements = re.subn(
            r"(?<![A-Za-z0-9])English\s+(?:Vietnam|Viet\s+Nam|Việt\s+Nam)(?![A-Za-z0-9])",
            "Greenwich Vietnam",
            corrected,
            flags=re.IGNORECASE,
        )
        if replacements:
            labels.append("english vietnam=>greenwich vietnam")
        return corrected, tuple(labels)

    @staticmethod
    def _has_greenwich_domain_context(token_set: set[str]) -> bool:
        """Return whether a query has enough study/admissions context for guarded correction."""

        return bool(
            token_set
            & {
                "admission",
                "dai",
                "diem",
                "difference",
                "hoc",
                "information",
                "it",
                "khac",
                "major",
                "majors",
                "manh",
                "nganh",
                "phu",
                "student",
                "students",
                "study",
                "suitable",
                "technology",
                "university",
            }
        )

    @classmethod
    def _is_bare_greenwich_entity_query(cls, tokens: list[str]) -> bool:
        """Return whether the transcript only names Greenwich Vietnam."""

        token_set = set(tokens)
        if "vietnam" not in token_set and not {"viet", "nam"} <= token_set:
            return False
        allowed = {"greenwich", "green", "which", "vietnam", "viet", "nam"}
        return cls._has_explicit_greenwich_mention(token_set) and token_set <= allowed

    def _should_repeat_after_failed_direct_llm(
        self,
        query: str,
        language_confidence: float | None,
        language_reason: str | None,
    ) -> bool:
        """Return whether failed direct LLM should reprompt instead of hard fallback."""

        tokens = _TOKEN_RE.findall(self._normalize_for_context(query))
        if self._query_token_count(query) <= 3:
            return True
        reason_base = (language_reason or "").split("+", 1)[0]
        if reason_base in {"no_marker_hit", "marker_tie"} and len(tokens) <= 5:
            return True
        if language_confidence is not None and language_confidence < 0.60 and len(tokens) <= 5:
            return True
        return False

    def _compact_direct_llm_response(self, response_text: str, query: str = "") -> str:
        """Keep direct LLM answers short enough for spoken interaction."""

        normalized = self._normalize_spoken_response(response_text)
        raw_length = len(response_text)
        length_limit = 420 if self._is_greenwich_context_query(query) else 220
        if not normalized:
            self._log_direct_llm_compaction(raw_length, 0, "empty_normalized", False, response_text, "")
            return normalized
        sentences = self._split_spoken_sentences(normalized)
        decision = "unchanged"
        if sentences:
            normalized = " ".join(sentences[:2]).strip()
            decision = "first_two_sentences" if len(sentences) > 1 else "first_sentence"
        incomplete_ending = self._has_incomplete_spoken_ending(normalized)
        if incomplete_ending:
            self._log_direct_llm_compaction(raw_length, 0, "rejected_incomplete", True, response_text, normalized)
            return ""
        if len(normalized) <= length_limit:
            self._log_direct_llm_compaction(raw_length, len(normalized), decision, False, response_text, normalized)
            return normalized
        complete_sentence = self._first_complete_sentence(normalized)
        if complete_sentence:
            self._log_direct_llm_compaction(raw_length, len(complete_sentence), "first_complete_sentence", False, response_text, complete_sentence)
            return complete_sentence
        truncated = normalized[:length_limit].rsplit(" ", 1)[0].rstrip(".,;:-")
        incomplete_ending = self._has_incomplete_spoken_ending(truncated)
        if incomplete_ending:
            self._log_direct_llm_compaction(raw_length, 0, "rejected_truncated_incomplete", True, response_text, truncated)
            return ""
        compacted = truncated or normalized[:length_limit].rstrip()
        self._log_direct_llm_compaction(raw_length, len(compacted), "truncated", False, response_text, compacted)
        return compacted

    @staticmethod
    def _log_direct_llm_compaction(
        raw_length: int,
        compacted_length: int,
        decision: str,
        incomplete_ending: bool,
        raw_text: str,
        compacted_text: str,
    ) -> None:
        """Log direct LLM compaction decisions for live truncation diagnostics."""

        LOGGER.info(
            "Direct LLM compaction diagnostics: raw_chars=%d compacted_chars=%d decision=%s incomplete_ending=%s",
            raw_length,
            compacted_length,
            decision,
            incomplete_ending,
        )
        if os.getenv("VOICE_LOOP_DEBUG_LLM_TEXT", "").strip().lower() in {"1", "true", "yes", "on"}:
            LOGGER.info(
                'Direct LLM text diagnostics: raw_first="%s" raw_last="%s" compacted="%s"',
                VoicePipeline._debug_excerpt(raw_text[:160]),
                VoicePipeline._debug_excerpt(raw_text[-160:]),
                VoicePipeline._debug_excerpt(compacted_text),
            )

    @staticmethod
    def _debug_excerpt(text: str) -> str:
        """Return a single-line excerpt safe for debug logs."""

        return " ".join(text.split())

    @staticmethod
    def _looks_like_incomplete_llm_fragment(response_text: str) -> bool:
        """Reject short generated fragments that should not be spoken."""

        normalized = " ".join(response_text.split()).strip()
        if not normalized:
            return True
        tokens = _TOKEN_RE.findall(normalized)
        if len(tokens) <= 3 and not re.search(r"[.!?]$", normalized):
            return True
        if VoicePipeline._has_incomplete_spoken_ending(normalized):
            return True
        incomplete_prefixes = (
            "tôi xin lỗi nếu",
            "toi xin loi neu",
            "để trả lời câu",
            "de tra loi cau",
        )
        lowered = normalized.lower()
        return any(lowered.startswith(prefix) and not re.search(r"[.!?]$", normalized) for prefix in incomplete_prefixes)

    @staticmethod
    def _has_incomplete_spoken_ending(response_text: str) -> bool:
        """Return whether text ends with a fragment that should not be spoken."""

        normalized = " ".join(response_text.split()).strip()
        if not normalized:
            return True
        lowered = normalized.lower().rstrip()
        if lowered.endswith(('"', "'", "“", "”")):
            return True
        if re.search(r"\b(?:tp|e)\.$", lowered):
            return True
        if re.search(r"^\d{1,2}\.$", lowered):
            return True
        if re.search(
            r"\b(?:in|on|at|by|for|with|from|of|to|about|include|includes|including|announce|announced)$",
            lowered,
        ):
            return True
        return False

    async def _render_audio_with_fallback(self, text: str, language: LanguageCode, output_path: Path) -> None:
        """Try the primary TTS provider first, then the fallback provider if available."""

        tts_text = self._prepare_text_for_tts(text)
        primary_timeout_seconds = min(self.timeout_seconds, 5.0) if self.fallback_tts_provider is not None else self.timeout_seconds
        primary_output_path = (
            output_path.with_name(f"{output_path.stem}.primary{output_path.suffix}")
            if self.fallback_tts_provider is not None
            else output_path
        )
        if primary_output_path != output_path:
            self._remove_stale_primary_tts_output(primary_output_path)
        LOGGER.info(
            "TTS synthesis diagnostics: provider=primary text_chars=%d language=%s output=%s timeout_seconds=%.1f",
            len(tts_text),
            language,
            primary_output_path.name,
            primary_timeout_seconds,
        )
        started_at = time.perf_counter()
        try:
            await asyncio.wait_for(
                self.primary_tts_provider.synthesize(tts_text, language, primary_output_path),
                timeout=primary_timeout_seconds,
            )
            if primary_output_path != output_path:
                primary_output_path.replace(output_path)
            LOGGER.info(
                "TTS synthesis result: provider=primary elapsed_ms=%.0f output=%s bytes=%d",
                (time.perf_counter() - started_at) * 1000.0,
                output_path.name,
                output_path.stat().st_size if output_path.exists() else 0,
            )
            return
        except Exception as exc:
            LOGGER.warning(
                "TTS synthesis failed: provider=primary elapsed_ms=%.0f type=%s reason=%s",
                (time.perf_counter() - started_at) * 1000.0,
                type(exc).__name__,
                str(exc).strip() or "<no error message>",
            )
            if self.fallback_tts_provider is None:
                raise
        fallback_timeout_seconds = min(self.timeout_seconds, 5.0)
        LOGGER.info(
            "TTS synthesis diagnostics: provider=fallback text_chars=%d language=%s output=%s timeout_seconds=%.1f",
            len(tts_text),
            language,
            output_path.name,
            fallback_timeout_seconds,
        )
        started_at = time.perf_counter()
        try:
            await asyncio.wait_for(
                self.fallback_tts_provider.synthesize(tts_text, language, output_path),
                timeout=fallback_timeout_seconds,
            )
            LOGGER.info(
                "TTS synthesis result: provider=fallback elapsed_ms=%.0f output=%s bytes=%d",
                (time.perf_counter() - started_at) * 1000.0,
                output_path.name,
                output_path.stat().st_size if output_path.exists() else 0,
            )
        except Exception as exc:
            LOGGER.warning(
                "TTS synthesis failed: provider=fallback elapsed_ms=%.0f type=%s reason=%s",
                (time.perf_counter() - started_at) * 1000.0,
                type(exc).__name__,
                str(exc).strip() or "<no error message>",
            )
            raise
        finally:
            if primary_output_path != output_path:
                self._remove_stale_primary_tts_output(primary_output_path)

    @staticmethod
    def _remove_stale_primary_tts_output(primary_output_path: Path) -> None:
        """Remove a leftover primary TTS temp file when fallback audio is used."""

        try:
            primary_output_path.unlink(missing_ok=True)
        except OSError:
            LOGGER.debug("Could not remove stale primary TTS temp output: %s", primary_output_path)

    @staticmethod
    def _prepare_text_for_tts(text: str) -> str:
        """Flatten generated text so TTS does not stumble over multiline output."""

        flattened = re.sub(r"[\r\n]+", " ", text)
        flattened = re.sub(r"(^|\s)[*\u2022-]\s+", r"\1", flattened)
        flattened = re.sub(r"\*\*|__", "", flattened)
        flattened = re.sub(r"\*|_", "", flattened)
        flattened = re.sub(r"#+\s+", "", flattened)
        flattened = re.sub(r"\s+", " ", flattened).strip()
        return flattened or text

    @staticmethod
    def _fallback_phrase(language: LanguageCode) -> str:
        """Return the hard fallback response for the selected language."""

        return (
            "Xin lỗi, tôi không có câu trả lời cho vấn đề này hiện tại."
            if language == "vi"
            else "I am sorry, I do not have an answer for that right now."
        )

    @staticmethod
    def _greenwich_failed_llm_fallback_phrase(language: LanguageCode) -> str:
        """Return a Greenwich-specific rescue answer when LLM direct fails."""

        return (
            "Greenwich Việt Nam đào tạo theo chuẩn Anh Quốc thông qua liên kết giữa Đại học Greenwich và Tổ chức Giáo dục FPT. "
            "Bạn có thể hỏi thêm về ngành học, học phí, cơ sở, tuyển sinh hoặc đời sống sinh viên."
            if language == "vi"
            else "Greenwich Vietnam offers UK-standard study in Vietnam through the University of Greenwich and FPT Education partnership. "
            "You can ask me about majors, tuition, campuses, admissions, or student life."
        )

    @classmethod
    def _is_greenwich_context_query(cls, query: str) -> bool:
        """Return whether a transcript clearly refers to Greenwich."""

        tokens = set(_TOKEN_RE.findall(cls._normalize_for_context(query)))
        return "greenwich" in tokens or {"green", "which"} <= tokens

    @staticmethod
    def _repeat_request_phrase(language: LanguageCode) -> str:
        """Return a short prompt when the transcript is too noisy to answer."""

        return "Tôi nghe chưa rõ, bạn nói lại câu hỏi được không?" if language == "vi" else "I may have misheard that. Please repeat the question."

    @staticmethod
    def _noise_reprompt_phrase(language: LanguageCode) -> str:
        """Return a random noise reprompt phrase in the given language."""
        import random
        en_phrases = [
            "I couldn't hear your question clearly. Could you please repeat?",
            "It is a bit noisy here. Please say that again.",
            "I didn't catch that. Could you repeat your question?"
        ]
        vi_phrases = [
            "Tôi nghe chưa rõ do tiếng ồn. Bạn nói lại giúp tôi nhé.",
            "Không gian hơi ồn, bạn có thể lặp lại câu hỏi được không?",
            "Tôi chưa nghe rõ câu hỏi. Bạn vui lòng lặp lại nhé."
        ]
        phrases = vi_phrases if language == "vi" else en_phrases
        return random.choice(phrases)
