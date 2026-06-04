"""Configuration objects and environment loading helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .transcript_cheats import TranscriptCheatRule
from .types import LanguageCode


def _load_dotenv_if_present() -> None:
    """Load runtime env files from the repository root.

    Precedence is:
    1) Existing process environment
    2) .env values
    3) .env.example values only when .env is absent
    """

    root_dir = Path(__file__).resolve().parent.parent
    env_path = root_dir / ".env"
    _load_env_file_if_present(env_path)
    if not env_path.exists():
        _load_env_file_if_present(root_dir / ".env.example")


def _load_env_file_if_present(env_path: Path) -> None:
    """Load simple KEY=VALUE pairs from a file using setdefault semantics."""

    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv_if_present()


@dataclass(frozen=True)
class AppConfig:
    """Immutable runtime configuration for the pipeline."""

    language: LanguageCode
    db_path: Path
    output_dir: Path
    provider_timeout_seconds: float = 3.0
    stt_provider: str = "google"
    stt_model: str = ""
    stt_location: str = "global"
    stt_hint_phrases: tuple[str, ...] = ()
    transcript_cheats: tuple[TranscriptCheatRule, ...] = ()
    tts_provider: str = "google"
    llm_timeout_seconds: float = 10.0
    llm_direct_min_query_tokens: int = 1
    llm_enable_google_search: bool = True
    language_mode: str = "adaptive"
    output_language_mode: str = "auto"
    output_language_fixed: LanguageCode = "en"
    enable_bilingual_output: bool = False
    language_switch_min_confidence: float = 0.75
    language_switch_sticky_turns: int = 2
    language_override_commands: bool = True
    qa_retrieval_mode: str = "lexical"
    qa_lexical_top_k: int = 5
    qa_vector_top_k: int = 5
    qa_confidence_low: float = 0.55
    qa_seed_json_path: Path = Path("data/qa_seed_vi_en.json")
    qa_seed_auto_sync: bool = False
    context_link_enabled: bool = True
    context_link_max_turn_gap: int = 2
    context_link_short_query_max_tokens: int = 6
    context_link_min_score_delta: float = 0.08
    seed_demo_data: bool = True
    gemini_model: str = "gemini-3.1-flash-lite"
    gemini_fallback_model: str = "gemini-2.5-flash"
    llm_thinking_level: str = "minimal"
    llm_thinking_budget: int = 0
    google_cloud_project: str = ""
    google_cloud_location: str = "global"
    log_level: str = "INFO"
    wake_word: str = "hey lemon"
    wake_aliases: tuple[str, ...] = ()
    wake_ack_mode: str = "adaptive"
    wake_ack_beep_frequency: int = 880
    wake_ack_beep_duration_ms: int = 120
    wake_ack_prompt_text_en: str = "How can I help you?"
    wake_ack_prompt_text_vi: str = "Tôi có thể giúp gì cho bạn?"
    wake_ack_adaptive_speak_on_wake: bool = True
    request_ready_cue_mode: str = "speech"
    request_ready_first_text_en: str = "Go ahead."
    request_ready_first_text_vi: str = "Bạn cứ hỏi nhé."
    request_ready_texts_en: tuple[str, ...] = ("What else would you like to know?", "I'm listening.", "Go ahead.")
    request_ready_texts_vi: tuple[str, ...] = ("Bạn hỏi tiếp nhé.", "Mình nghe đây.", "Bạn cần hỏi thêm gì?")
    request_ready_cache: bool = True
    thinking_cue_enabled: bool = True
    thinking_cue_delay_seconds: float = 1.2
    thinking_texts_en: tuple[str, ...] = ("Let me check.", "One moment.")
    thinking_texts_vi: tuple[str, ...] = ("Để tôi kiểm tra.", "Chờ tôi một chút.")
    wake_window_seconds: float = 4.0
    utterance_seconds: float = 8.0
    utterance_min_rms: float = 30.0
    utterance_min_peak: float = 250.0
    request_post_tts_guard_seconds: float = 0.15
    minimum_transcript_characters: int = 4
    enable_streaming_stt: bool = True
    streaming_chunk_duration_ms: int = 100
    streaming_speech_start_timeout_seconds: float = 8.0
    streaming_speech_end_timeout_seconds: float = 1.8
    streaming_local_speech_end_ms: int = 2000
    streaming_max_active_seconds: float = 10.0
    streaming_no_progress_seconds: float = 4.5
    streaming_weak_progress_seconds: float = 6.0
    streaming_weak_progress_min_tokens: int = 3
    preroll_enabled: bool = True
    preroll_ms: int = 750
    startup_calibration_enabled: bool = True
    startup_calibration_seconds: float = 2.0
    request_max_ignored_turns: int = 4
    request_max_turns: int = 0
    request_idle_timeout_seconds: float = 90.0
    request_max_session_seconds: float = 180.0
    barge_in_mode: str = "off"
    barge_in_listen_seconds: float = 0.7
    barge_in_grace_seconds: float = 1.2
    barge_in_min_rms: float = 45.0
    barge_in_min_peak: float = 400.0
    sample_rate: int = 44100
    input_device_index: int | None = None
    debug_audio_io: bool = False
    debug_stt_stream: bool = False

    @classmethod
    def from_env(
        cls,
        language: LanguageCode,
        db_path: str | None = None,
        output_dir: str | None = None,
        seed_demo_data: bool = True,
    ) -> "AppConfig":
        """Build configuration from environment variables and CLI overrides."""

        wake_word = os.getenv("VOICE_LOOP_WAKE_WORD", "hey lemon")
        configured_hints = _parse_phrase_list(os.getenv("VOICE_LOOP_STT_HINT_PHRASES"))
        stt_hint_phrases = configured_hints or _default_stt_hint_phrases(wake_word)
        configured_wake_aliases = _parse_phrase_list(os.getenv("VOICE_LOOP_WAKE_ALIASES"))
        wake_aliases = configured_wake_aliases or _derive_wake_aliases(wake_word, stt_hint_phrases)
        language_mode = _parse_language_mode(os.getenv("VOICE_LOOP_LANGUAGE_MODE"), "adaptive")
        output_language_mode = _parse_output_language_mode(os.getenv("VOICE_LOOP_OUTPUT_LANGUAGE_MODE"), "auto")
        output_language_fixed = _parse_language_code(os.getenv("VOICE_LOOP_OUTPUT_LANGUAGE_FIXED"), language)
        qa_confidence_low = _parse_probability(os.getenv("VOICE_LOOP_QA_CONFIDENCE_LOW"), 0.55)

        return cls(
            language=language,
            db_path=Path(db_path or os.getenv("VOICE_LOOP_DB_PATH", "data/knowledge_base.sqlite3")),
            output_dir=Path(output_dir or os.getenv("VOICE_LOOP_OUTPUT_DIR", "output")),
            provider_timeout_seconds=float(os.getenv("VOICE_LOOP_PROVIDER_TIMEOUT_SECONDS", "3.0")),
            stt_provider=os.getenv("VOICE_LOOP_STT_PROVIDER", "google").lower(),
            stt_model=os.getenv("VOICE_LOOP_STT_MODEL", "").strip(),
            stt_location=os.getenv("VOICE_LOOP_STT_LOCATION", "global").strip() or "global",
            stt_hint_phrases=stt_hint_phrases,
            transcript_cheats=_parse_transcript_cheats(os.getenv("VOICE_LOOP_TRANSCRIPT_CHEATS")),
            tts_provider=os.getenv("VOICE_LOOP_TTS_PROVIDER", "google").lower(),
            llm_timeout_seconds=_parse_float(os.getenv("VOICE_LOOP_LLM_TIMEOUT_SECONDS"), 10.0),
            llm_direct_min_query_tokens=_parse_int(os.getenv("VOICE_LOOP_LLM_DIRECT_MIN_QUERY_TOKENS"), 1),
            llm_enable_google_search=_parse_bool(os.getenv("VOICE_LOOP_LLM_ENABLE_GOOGLE_SEARCH"), True),
            language_mode=language_mode,
            output_language_mode=output_language_mode,
            output_language_fixed=output_language_fixed,
            enable_bilingual_output=_parse_bool(os.getenv("VOICE_LOOP_ENABLE_BILINGUAL_OUTPUT"), False),
            language_switch_min_confidence=_parse_probability(
                os.getenv("VOICE_LOOP_LANGUAGE_SWITCH_MIN_CONFIDENCE"),
                0.75,
            ),
            language_switch_sticky_turns=_parse_int(os.getenv("VOICE_LOOP_LANGUAGE_SWITCH_STICKY_TURNS"), 2),
            language_override_commands=_parse_bool(os.getenv("VOICE_LOOP_LANGUAGE_OVERRIDE_COMMANDS"), True),
            qa_retrieval_mode=_parse_qa_retrieval_mode(os.getenv("VOICE_LOOP_QA_RETRIEVAL_MODE"), "lexical"),
            qa_lexical_top_k=_parse_int(os.getenv("VOICE_LOOP_QA_LEXICAL_TOP_K"), 5),
            qa_vector_top_k=_parse_int(os.getenv("VOICE_LOOP_QA_VECTOR_TOP_K"), 5),
            qa_confidence_low=qa_confidence_low,
            qa_seed_json_path=Path(os.getenv("VOICE_LOOP_QA_SEED_JSON_PATH", "data/qa_seed_vi_en.json")),
            qa_seed_auto_sync=_parse_bool(os.getenv("VOICE_LOOP_QA_SEED_AUTO_SYNC"), False),
            context_link_enabled=_parse_bool(os.getenv("VOICE_LOOP_CONTEXT_LINK_ENABLED"), True),
            context_link_max_turn_gap=_parse_int(os.getenv("VOICE_LOOP_CONTEXT_LINK_MAX_TURN_GAP"), 2),
            context_link_short_query_max_tokens=_parse_int(
                os.getenv("VOICE_LOOP_CONTEXT_LINK_SHORT_QUERY_MAX_TOKENS"),
                6,
            ),
            context_link_min_score_delta=_parse_probability(
                os.getenv("VOICE_LOOP_CONTEXT_LINK_MIN_SCORE_DELTA"),
                0.08,
            ),
            seed_demo_data=seed_demo_data,
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite"),
            gemini_fallback_model=os.getenv("GEMINI_FALLBACK_MODEL", "gemini-2.5-flash").strip(),
            llm_thinking_level=os.getenv("VOICE_LOOP_LLM_THINKING_LEVEL", "minimal").strip(),
            llm_thinking_budget=_parse_int(os.getenv("VOICE_LOOP_LLM_THINKING_BUDGET"), 0),
            google_cloud_project=os.getenv("GOOGLE_CLOUD_PROJECT", ""),
            google_cloud_location=os.getenv("GOOGLE_CLOUD_LOCATION", "global"),
            log_level=os.getenv("VOICE_LOOP_LOG_LEVEL", "INFO"),
            wake_word=wake_word,
            wake_aliases=wake_aliases,
            wake_ack_mode=_parse_wake_ack_mode(os.getenv("VOICE_LOOP_WAKE_ACK_MODE"), "adaptive"),
            wake_ack_beep_frequency=_parse_int(os.getenv("VOICE_LOOP_WAKE_ACK_BEEP_FREQUENCY"), 880),
            wake_ack_beep_duration_ms=_parse_int(os.getenv("VOICE_LOOP_WAKE_ACK_BEEP_DURATION_MS"), 120),
            wake_ack_prompt_text_en=os.getenv("VOICE_LOOP_WAKE_ACK_PROMPT_TEXT_EN", "How can I help you?"),
            wake_ack_prompt_text_vi=os.getenv("VOICE_LOOP_WAKE_ACK_PROMPT_TEXT_VI", "Tôi có thể giúp gì cho bạn?"),
            wake_ack_adaptive_speak_on_wake=_parse_bool(
                os.getenv("VOICE_LOOP_WAKE_ACK_ADAPTIVE_SPEAK_ON_WAKE"),
                True,
            ),
            request_ready_cue_mode=_parse_cue_mode(os.getenv("VOICE_LOOP_REQUEST_READY_CUE_MODE"), "speech"),
            request_ready_first_text_en=os.getenv("VOICE_LOOP_REQUEST_READY_FIRST_TEXT_EN", "Go ahead."),
            request_ready_first_text_vi=os.getenv("VOICE_LOOP_REQUEST_READY_FIRST_TEXT_VI", "Bạn cứ hỏi nhé."),
            request_ready_texts_en=_parse_phrase_list(os.getenv("VOICE_LOOP_REQUEST_READY_TEXTS_EN"))
            or ("What else would you like to know?", "I'm listening.", "Go ahead."),
            request_ready_texts_vi=_parse_phrase_list(os.getenv("VOICE_LOOP_REQUEST_READY_TEXTS_VI"))
            or ("Bạn hỏi tiếp nhé.", "Mình nghe đây.", "Bạn cần hỏi thêm gì?"),
            request_ready_cache=_parse_bool(os.getenv("VOICE_LOOP_REQUEST_READY_CACHE"), True),
            thinking_cue_enabled=_parse_bool(os.getenv("VOICE_LOOP_THINKING_CUE_ENABLED"), True),
            thinking_cue_delay_seconds=_parse_float(os.getenv("VOICE_LOOP_THINKING_CUE_DELAY_SECONDS"), 1.2),
            thinking_texts_en=_parse_phrase_list(os.getenv("VOICE_LOOP_THINKING_TEXTS_EN"))
            or ("Let me check.", "One moment."),
            thinking_texts_vi=_parse_phrase_list(os.getenv("VOICE_LOOP_THINKING_TEXTS_VI"))
            or ("Để tôi kiểm tra.", "Chờ tôi một chút."),
            wake_window_seconds=_parse_float(os.getenv("VOICE_LOOP_WAKE_WINDOW_SECONDS"), 4.0),
            utterance_seconds=_parse_float(os.getenv("VOICE_LOOP_UTTERANCE_SECONDS"), 8.0),
            utterance_min_rms=_parse_float(os.getenv("VOICE_LOOP_UTTERANCE_MIN_RMS"), 30.0),
            utterance_min_peak=_parse_float(os.getenv("VOICE_LOOP_UTTERANCE_MIN_PEAK"), 250.0),
            request_post_tts_guard_seconds=_parse_float(
                os.getenv("VOICE_LOOP_REQUEST_POST_TTS_GUARD_SECONDS"),
                0.15,
            ),
            minimum_transcript_characters=_parse_int(os.getenv("VOICE_LOOP_MIN_TRANSCRIPT_CHARACTERS"), 4),
            enable_streaming_stt=_parse_bool(os.getenv("VOICE_LOOP_ENABLE_STREAMING_STT"), True),
            streaming_chunk_duration_ms=_parse_int(os.getenv("VOICE_LOOP_STREAMING_CHUNK_DURATION_MS"), 100),
            streaming_speech_start_timeout_seconds=_parse_float(
                os.getenv("VOICE_LOOP_STREAMING_SPEECH_START_TIMEOUT_SECONDS"),
                8.0,
            ),
            streaming_speech_end_timeout_seconds=_parse_float(
                os.getenv("VOICE_LOOP_STREAMING_SPEECH_END_TIMEOUT_SECONDS"),
                1.8,
            ),
            streaming_local_speech_end_ms=_parse_int(os.getenv("VOICE_LOOP_STREAMING_LOCAL_SPEECH_END_MS"), 2000),
            streaming_max_active_seconds=_parse_float(
                os.getenv("VOICE_LOOP_STREAMING_MAX_ACTIVE_SECONDS"),
                10.0,
            ),
            streaming_no_progress_seconds=_parse_float(
                os.getenv("VOICE_LOOP_STREAMING_NO_PROGRESS_SECONDS"),
                4.5,
            ),
            streaming_weak_progress_seconds=_parse_float(
                os.getenv("VOICE_LOOP_STREAMING_WEAK_PROGRESS_SECONDS"),
                6.0,
            ),
            streaming_weak_progress_min_tokens=_parse_int(
                os.getenv("VOICE_LOOP_STREAMING_WEAK_PROGRESS_MIN_TOKENS"),
                3,
            ),
            preroll_enabled=_parse_bool(os.getenv("VOICE_LOOP_PREROLL_ENABLED"), True),
            preroll_ms=_parse_int(os.getenv("VOICE_LOOP_PREROLL_MS"), 750),
            startup_calibration_enabled=_parse_bool(
                os.getenv("VOICE_LOOP_STARTUP_CALIBRATION_ENABLED"),
                True,
            ),
            startup_calibration_seconds=_parse_float(
                os.getenv("VOICE_LOOP_STARTUP_CALIBRATION_SECONDS"),
                2.0,
            ),
            request_max_ignored_turns=_parse_int(os.getenv("VOICE_LOOP_REQUEST_MAX_IGNORED_TURNS"), 4),
            request_max_turns=_parse_int(os.getenv("VOICE_LOOP_REQUEST_MAX_TURNS"), 0),
            request_idle_timeout_seconds=_parse_float(
                os.getenv("VOICE_LOOP_REQUEST_IDLE_TIMEOUT_SECONDS"),
                90.0,
            ),
            request_max_session_seconds=_parse_float(os.getenv("VOICE_LOOP_REQUEST_MAX_SESSION_SECONDS"), 180.0),
            barge_in_mode=_parse_barge_in_mode(os.getenv("VOICE_LOOP_BARGE_IN_MODE"), "off"),
            barge_in_listen_seconds=_parse_float(os.getenv("VOICE_LOOP_BARGE_IN_LISTEN_SECONDS"), 0.7),
            barge_in_grace_seconds=_parse_float(os.getenv("VOICE_LOOP_BARGE_IN_GRACE_SECONDS"), 1.2),
            barge_in_min_rms=_parse_float(os.getenv("VOICE_LOOP_BARGE_IN_MIN_RMS"), 45.0),
            barge_in_min_peak=_parse_float(os.getenv("VOICE_LOOP_BARGE_IN_MIN_PEAK"), 400.0),
            sample_rate=_parse_int(os.getenv("VOICE_LOOP_SAMPLE_RATE"), 44100),
            input_device_index=_parse_optional_int(os.getenv("VOICE_LOOP_INPUT_DEVICE_INDEX")),
            debug_audio_io=_parse_bool(os.getenv("VOICE_LOOP_DEBUG_AUDIO_IO"), False),
            debug_stt_stream=_parse_bool(os.getenv("VOICE_LOOP_DEBUG_STT_STREAM"), False),
        )


def _default_stt_hint_phrases(wake_word: str) -> tuple[str, ...]:
    """Derive default hint phrases from the configured wake word."""

    cleaned_wake_word = wake_word.strip()
    if not cleaned_wake_word:
        return ()
    hints: list[str] = [cleaned_wake_word]
    wake_tokens = [token for token in cleaned_wake_word.split() if token]
    if len(wake_tokens) >= 2:
        hints.append(wake_tokens[-1])
    return tuple(dict.fromkeys(hints))


def _derive_wake_aliases(wake_word: str, stt_hint_phrases: tuple[str, ...]) -> tuple[str, ...]:
    """Derive wake aliases from STT hint phrases when explicit aliases are not set."""

    normalized_wake_word = wake_word.strip().lower()
    aliases: list[str] = []
    for phrase in stt_hint_phrases:
        cleaned_phrase = phrase.strip()
        if not cleaned_phrase:
            continue
        if cleaned_phrase.lower() == normalized_wake_word:
            continue
        aliases.append(cleaned_phrase)
    return tuple(dict.fromkeys(aliases))


def _parse_optional_int(value: str | None) -> int | None:
    """Parse an optional integer string, returning None for empty values."""

    if value is None or not value.strip():
        return None
    return int(value.strip())


def _parse_float(value: str | None, default: float) -> float:
    """Parse a float value from environment or return a default."""

    if value is None or not value.strip():
        return default
    return float(value.strip())


def _parse_int(value: str | None, default: int) -> int:
    """Parse an integer value from environment or return a default."""

    if value is None or not value.strip():
        return default
    return int(value.strip())


def _parse_bool(value: str | None, default: bool) -> bool:
    """Parse a boolean value from environment or return a default."""

    if value is None or not value.strip():
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _parse_phrase_list(value: str | None) -> tuple[str, ...]:
    """Parse a comma-separated phrase list from environment."""

    if value is None or not value.strip():
        return ()
    phrases = [item.strip() for item in value.split(",")]
    return tuple(item for item in phrases if item)


def _parse_transcript_cheats(value: str | None) -> tuple[TranscriptCheatRule, ...]:
    """Parse transcript correction pairs from environment.

    Format:
    VOICE_LOOP_TRANSCRIPT_CHEATS=wrong phrase=correct phrase|context1,context2;other wrong=other correct
    """

    if value is None or not value.strip():
        return ()

    parsed_pairs: list[TranscriptCheatRule] = []
    for raw_pair in value.split(";"):
        pair = raw_pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            continue
        wrong, corrected_with_context = pair.split("=", 1)
        corrected_text, _, context_part = corrected_with_context.partition("|")
        normalized_wrong = wrong.strip()
        normalized_correct = corrected_text.strip()
        if not normalized_wrong or not normalized_correct:
            continue
        context_terms = _parse_phrase_list(context_part) if context_part.strip() else ()
        parsed_pairs.append(
            TranscriptCheatRule(
                wrong_phrase=normalized_wrong,
                corrected_phrase=normalized_correct,
                required_context_terms=context_terms,
            )
        )

    return tuple(dict.fromkeys(parsed_pairs))


def _parse_wake_ack_mode(value: str | None, default: str) -> str:
    """Parse wake acknowledgement mode with safe defaults."""

    if value is None or not value.strip():
        return default
    normalized = value.strip().lower()
    if normalized in {"none", "beep", "speech", "adaptive"}:
        return normalized
    return default


def _parse_cue_mode(value: str | None, default: str) -> str:
    """Parse short cue mode with safe defaults."""

    if value is None or not value.strip():
        return default
    normalized = value.strip().lower()
    if normalized in {"none", "beep", "speech"}:
        return normalized
    return default


def _parse_barge_in_mode(value: str | None, default: str) -> str:
    """Parse barge-in mode with an office-safe default."""

    if value is None or not value.strip():
        return default
    normalized = value.strip().lower()
    if normalized in {"off", "wake_phrase"}:
        return normalized
    return default


def _parse_language_mode(value: str | None, default: str) -> str:
    """Parse input language mode with a safe default."""

    if value is None or not value.strip():
        return default
    normalized = value.strip().lower()
    if normalized in {"adaptive", "fixed"}:
        return normalized
    return default


def _parse_output_language_mode(value: str | None, default: str) -> str:
    """Parse output language mode with a safe default."""

    if value is None or not value.strip():
        return default
    normalized = value.strip().lower()
    if normalized in {"auto", "fixed"}:
        return normalized
    return default


def _parse_qa_retrieval_mode(value: str | None, default: str) -> str:
    """Parse Q&A retrieval mode with allowed values only."""

    if value is None or not value.strip():
        return default
    normalized = value.strip().lower()
    if normalized in {"lexical", "vector", "hybrid"}:
        return normalized
    return default


def _parse_language_code(value: str | None, default: LanguageCode) -> LanguageCode:
    """Parse language code while preserving known language values only."""

    if value is None or not value.strip():
        return default
    normalized = value.strip().lower()
    if normalized in {"en", "vi"}:
        return normalized  # type: ignore[return-value]
    return default


def _parse_probability(value: str | None, default: float) -> float:
    """Parse and clamp a probability-like float into [0.0, 1.0]."""

    parsed = _parse_float(value, default)
    if parsed < 0.0:
        return 0.0
    if parsed > 1.0:
        return 1.0
    return parsed
