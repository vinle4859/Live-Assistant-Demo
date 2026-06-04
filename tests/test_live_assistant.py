"""Unit tests for wake-word and sleep-command text matching logic."""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace

from voice_loop.live_assistant import (
    AudioPlayer,
    LiveAssistantConfig,
    LiveVoiceAssistant,
    contains_wake_word,
    describe_matched_wake_phrase,
    detect_input_language,
    detect_matched_wake_phrase,
    is_command_like_sleep_phrase,
    is_sleep_command,
    parse_language_override_command,
)


class _RecordingPlayer:
    """Capture played paths and optionally raise to exercise error paths."""

    def __init__(self, *, runtime_error: bool = False, generic_error: bool = False) -> None:
        """Store configured failure mode for playback simulation."""

        self.runtime_error = runtime_error
        self.generic_error = generic_error
        self.played_paths: list[str] = []

    def play(self, audio_path) -> None:
        """Record playback call and raise configured errors when requested."""

        self.played_paths.append(str(audio_path))
        if self.runtime_error:
            raise RuntimeError("playback configuration error")
        if self.generic_error:
            raise ValueError("playback transient error")

    def play_interruptible(self, audio_path, should_interrupt, grace_seconds: float) -> bool:
        """Record interruptible playback calls without doing real audio work."""

        self.play(audio_path)
        return should_interrupt()


class _StreamingWakeProvider:
    """Fake STT provider that records streaming wake usage."""

    def __init__(self) -> None:
        self.listen_calls = 0

    def supports_streaming_wake(self) -> bool:
        return True

    async def listen_for_wake_phrase(self, **kwargs):
        self.listen_calls += 1
        return {
            "transcript": "hello lemon",
            "phrase": "hello lemon",
            "match_type": "alias",
            "restart_reason": "wake_detected",
        }


class _StreamingRequestProvider:
    """Fake streaming STT provider that returns configured request transcripts."""

    def __init__(self, transcripts: list[str]) -> None:
        self.transcripts = transcripts
        self.last_live_capture_stats = {"end_reason": "speech_start_timeout"}

    def supports_live_streaming(self) -> bool:
        return True

    async def transcribe_live_utterance(self, **kwargs) -> str:
        if not self.transcripts:
            return "Goodbye"
        return self.transcripts.pop(0)


class _WakePipeline:
    """Minimal pipeline wrapper for wake tests."""

    def __init__(self, stt_provider) -> None:
        self.stt_provider = stt_provider


class _FakeTTSProvider:
    """Create fake cue audio and record synthesis calls."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    async def synthesize(self, text, language, output_path):
        self.calls.append((text, language, str(output_path)))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake")
        return output_path


class _CuePipeline:
    """Minimal pipeline wrapper for cue tests."""

    def __init__(self) -> None:
        self.primary_tts_provider = _FakeTTSProvider()
        self.timeout_seconds = 1.0
        self.stt_provider = _StreamingWakeProvider()


class _RequestPipeline:
    """Minimal pipeline wrapper for request-loop tests."""

    def __init__(self, transcripts: list[str]) -> None:
        self.primary_tts_provider = _FakeTTSProvider()
        self.timeout_seconds = 1.0
        self.stt_provider = _StreamingRequestProvider(transcripts)


def test_contains_wake_word_matches_case_and_punctuation() -> None:
    """Wake-word matching should ignore case and punctuation noise."""

    assert contains_wake_word("...Hey, Lemon! Are you there?", "hey lemon")


def test_contains_wake_word_rejects_other_phrases() -> None:
    """Wake-word matching should not trigger for unrelated transcripts."""

    assert not contains_wake_word("hello assistant", "hey lemon")


def test_contains_wake_word_accepts_minor_stt_variants() -> None:
    """Wake-word matching should tolerate minor STT spelling drift."""

    assert contains_wake_word("hey leman", "hey lemon")


def test_contains_wake_word_accepts_keyword_only_short_phrase() -> None:
    """Wake-word matching should allow the assistant keyword in short transcripts."""

    assert contains_wake_word("lemon", "hey lemon")


def test_detect_matched_wake_phrase_uses_aliases() -> None:
    """Wake matching should accept configured aliases when primary phrase is absent."""

    matched = detect_matched_wake_phrase("hello lemon", ("hey lemon", "hello lemon"))

    assert matched == "hello lemon"


def test_detect_matched_wake_phrase_rejects_plain_hello() -> None:
    """Plain hello should not wake the assistant without the Lemon keyword."""

    matched = detect_matched_wake_phrase("hello", ("hey lemon", "hello lemon", "hey leman", "le minh"))

    assert matched is None


def test_detect_matched_wake_phrase_accepts_live_stt_variant() -> None:
    """Wake matching should accept the observed Lemon transcription variant."""

    matched = detect_matched_wake_phrase("Hello Lê Minh", ("hey lemon", "hello lemon", "hey leman", "le minh"))

    assert matched == "le minh"


def test_streaming_wake_path_uses_provider_without_fixed_capture() -> None:
    """Wake mode should use continuous provider streaming when available."""

    provider = _StreamingWakeProvider()
    assistant = LiveVoiceAssistant(
        pipeline=_WakePipeline(provider),
        config=LiveAssistantConfig(language="en"),
    )  # type: ignore[arg-type]

    asyncio.run(assistant._wait_for_wake_word(("hey lemon", "hello lemon")))

    assert provider.listen_calls == 1


def test_describe_matched_wake_phrase_reports_match_type() -> None:
    """Wake diagnostics should identify fuzzy phrase matches."""

    matched = describe_matched_wake_phrase("hey leman", ("hey lemon", "hello lemon"))

    assert matched is not None
    assert matched.phrase == "hey lemon"
    assert matched.match_type == "fuzzy"


def test_is_sleep_command_english() -> None:
    """English sleep commands should be recognized."""

    assert is_sleep_command("Please stop listening now", "en")
    assert is_sleep_command("Top listening.", "en")
    assert is_command_like_sleep_phrase("bye")
    assert is_command_like_sleep_phrase("goodbye")


def test_command_like_sleep_phrase_blocks_stt_slips() -> None:
    """Command-like transcripts should be blocked before normal answer routing."""

    assert is_command_like_sleep_phrase("Top listening.")
    assert is_command_like_sleep_phrase("quit")
    assert is_command_like_sleep_phrase("goodbye")
    assert not is_command_like_sleep_phrase("top universities in Vietnam")


def test_is_sleep_command_vietnamese() -> None:
    """Vietnamese sleep commands should be recognized."""

    assert is_sleep_command("Bạn có thể ngủ đi", "vi")
    assert is_sleep_command("Chào tạm biệt.", "vi")
    assert is_sleep_command("Dừng nghe.", "vi")
    assert is_command_like_sleep_phrase("thoát")


def test_sleep_ack_text_localized_by_language() -> None:
    """Stop acknowledgement text should be localized for EN and VI turns."""

    assert LiveVoiceAssistant._sleep_ack_text("en").startswith("Okay, I will stop listening")
    assert "Say Hey Lemon" in LiveVoiceAssistant._sleep_ack_text("en")
    assert LiveVoiceAssistant._sleep_ack_text("vi").startswith("Đã dừng lắng nghe")


def test_english_stop_command_ack_language_ignores_current_vietnamese_state() -> None:
    """English stop commands should acknowledge in English even after Vietnamese turns."""

    assistant = LiveVoiceAssistant(pipeline=object(), config=LiveAssistantConfig(language="vi"))  # type: ignore[arg-type]
    assistant._current_output_language = "vi"

    assert assistant._command_ack_language("Stop listening.") == "en"


def test_vietnamese_stop_command_ack_language_stays_vietnamese() -> None:
    """Vietnamese stop commands should still acknowledge in Vietnamese."""

    assistant = LiveVoiceAssistant(pipeline=object(), config=LiveAssistantConfig(language="en"))  # type: ignore[arg-type]

    assert assistant._command_ack_language("ngủ đi") == "vi"


def test_detect_input_language_prefers_vietnamese_on_diacritics() -> None:
    """Vietnamese diacritics plus Vietnamese token evidence should route VI."""

    language, confidence, reason = detect_input_language("Tôi muốn hỏi học phí", "en")

    assert language == "vi"
    assert confidence >= 0.9
    assert reason == "vi_diacritic_weighted"


def test_detect_input_language_prefers_english_lead_marker_before_diacritics() -> None:
    """English question lead words should beat Vietnamese location/name diacritics."""

    language, confidence, reason = detect_input_language("What is Greenwich Việt Nam?", "vi")

    assert language == "en"
    assert confidence >= 0.9
    assert reason == "en_lead_marker"


def test_detect_input_language_routes_english_weather_with_vietnamese_location_to_english() -> None:
    """Weather is not enough alone, but What at the start should route English."""

    language, confidence, reason = detect_input_language("What is the weather in Hà Nội right now?", "vi")

    assert language == "en"
    assert confidence >= 0.9
    assert reason == "en_lead_marker"


def test_detect_input_language_keeps_vietnamese_weather_question_vietnamese() -> None:
    """Vietnamese-native weather phrasing should remain Vietnamese."""

    language, confidence, reason = detect_input_language("Hôm nay thời tiết ở Hà Nội thế nào?", "en")

    assert language == "vi"
    assert confidence >= 0.9
    assert reason == "vi_diacritic_weighted"


def test_detect_input_language_routes_english_traffic_with_vietnamese_place_to_english() -> None:
    """English majority and structure should beat Vietnamese place-name diacritics."""

    language, confidence, reason = detect_input_language("Tell me if traffic is bad near Cộng Hòa street", "vi")

    assert language == "en"
    assert confidence >= 0.7
    assert reason in {"en_lead_marker", "en_weighted_majority"}


def test_detect_input_language_routes_english_traffic_without_diacritics_to_english() -> None:
    """English traffic prompt with romanized place name should not stay in VI context."""

    language, confidence, reason = detect_input_language("Tell me traffic on Cong Hoa street", "vi")

    assert language == "en"
    assert confidence >= 0.7
    assert reason in {"en_lead_marker", "en_weighted_majority"}


def test_detect_input_language_routes_english_cost_with_vietnamese_name_to_english() -> None:
    """English majority should route EN even when Greenwich Vietnam has Vietnamese diacritics."""

    language, confidence, reason = detect_input_language(
        "How much does the whole program cost at Greenwich Việt Nam?",
        "vi",
    )

    assert language == "en"
    assert confidence >= 0.9
    assert reason == "en_lead_marker"


def test_detect_input_language_falls_back_when_no_markers() -> None:
    """Unknown text without markers should fall back to current language."""

    language, confidence, reason = detect_input_language("abc xyz", "en")

    assert language == "en"
    assert 0.0 <= confidence <= 1.0
    assert reason == "no_marker_hit"


def test_detect_input_language_recognizes_english_after_vietnamese_context() -> None:
    """Correct English STT text should not stay Vietnamese because of previous output state."""

    language, confidence, reason = detect_input_language(
        "Do students have opportunities for studying abroad or exchange programs?",
        "vi",
    )

    assert language == "en"
    assert confidence >= 0.6
    assert reason in {"en_marker_majority", "en_lead_marker", "en_weighted_majority"}


def test_detect_input_language_recognizes_quantum_question_after_vietnamese_context() -> None:
    """Out-of-DB English questions should switch output away from prior Vietnamese state."""

    language, confidence, reason = detect_input_language(
        "Explain quantum entanglement in simple terms.",
        "vi",
    )

    assert language == "en"
    assert confidence >= 0.6
    assert reason in {"en_marker_majority", "en_lead_marker", "en_weighted_majority"}


def test_detect_input_language_breaks_ai_news_tie_to_english() -> None:
    """English AI/news questions should switch back from a Vietnamese session."""

    language, confidence, reason = detect_input_language("latest AI news this week", "vi")

    assert language == "en"
    assert confidence >= 0.6
    assert reason in {"en_marker_majority", "en_marker_tie_break", "en_lead_marker"}


def test_parse_language_override_command_detects_english_only() -> None:
    """Command parser should detect English-only override phrases."""

    assert parse_language_override_command("Please speak English only") == "force_en"


def test_parse_language_override_command_detects_auto_mode() -> None:
    """Command parser should detect auto language override phrases."""

    assert parse_language_override_command("Bật chế độ tự động ngôn ngữ") == "auto"


def test_ignore_low_information_turn() -> None:
    """Very short transcripts should always be ignored as low-information noise."""

    assistant = LiveVoiceAssistant(pipeline=object(), config=LiveAssistantConfig(language="en"))  # type: ignore[arg-type]

    assert assistant._should_ignore_low_information_turn("uh")
    assert assistant._should_ignore_low_information_turn("weather")
    assert assistant._should_ignore_low_information_turn("What is?")
    assert not assistant._should_ignore_low_information_turn("where is london")


def test_low_confidence_reprompt_blocks_short_no_marker_turns() -> None:
    """Short no-marker transcripts should reprompt instead of reaching LLM direct."""

    assistant = LiveVoiceAssistant(pipeline=object(), config=LiveAssistantConfig(language="en"))  # type: ignore[arg-type]

    assert assistant._low_confidence_reprompt_reason("Pasimetas", 0.45, "no_marker_hit+same_output")
    assert assistant._low_confidence_reprompt_reason("Top listening", 0.45, "no_marker_hit+same_output") == "command_like"
    assert assistant._low_confidence_reprompt_reason("Explain quantum entanglement", 0.95, "en_marker_majority") is None


def test_unclear_reprompt_text_is_short() -> None:
    """Low-confidence reprompts should stay short for voice UX."""

    assert LiveVoiceAssistant._unclear_reprompt_text("en") == "I didn't catch that. Please repeat."
    assert len(LiveVoiceAssistant._unclear_reprompt_text("vi")) < 45


def test_looks_like_silence_uses_configured_thresholds() -> None:
    """Silence gating should use both RMS and peak thresholds."""

    assistant = LiveVoiceAssistant(pipeline=object(), config=LiveAssistantConfig(language="en"))  # type: ignore[arg-type]

    assert assistant._looks_like_silence({"rms": 10.0, "peak": 100.0})
    assert not assistant._looks_like_silence({"rms": 35.0, "peak": 100.0})


def test_log_calibration_result_warns_for_silent_input(caplog) -> None:
    """Startup calibration should surface likely wrong-device captures."""

    assistant = LiveVoiceAssistant(pipeline=object(), config=LiveAssistantConfig(language="en"))  # type: ignore[arg-type]

    import logging

    caplog.set_level(logging.WARNING)
    assistant._log_calibration_result({"duration_seconds": 2.0, "rms": 1.0, "peak": 2.0, "clipping_ratio": 0.0})

    assert any("near-silent input" in record.getMessage() for record in caplog.records)


def test_log_command_card_includes_wake_and_exit_phrases(caplog) -> None:
    """Startup command card should make wake and exit phrases operator-visible."""

    assistant = LiveVoiceAssistant(pipeline=object(), config=LiveAssistantConfig(language="en"))  # type: ignore[arg-type]

    import logging

    caplog.set_level(logging.INFO)
    assistant._log_command_card(
        {"index": 1},
        ("hey lemon", "hello"),
    )

    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "wake=\"hey lemon\"" in message and "stop listening" in message and "tạm biệt" in message
        for message in messages
    )


def test_wake_ack_mode_defaults_to_adaptive_for_invalid_values() -> None:
    """Unsupported wake-ack modes should normalize to adaptive."""

    assistant = LiveVoiceAssistant(
        pipeline=object(),
        config=LiveAssistantConfig(language="en", wake_ack_mode="unexpected"),
    )  # type: ignore[arg-type]

    assert assistant._wake_ack_mode() == "adaptive"


def test_barge_in_defaults_off() -> None:
    """Barge-in should be opt-in for office use."""

    assistant = LiveVoiceAssistant(pipeline=object(), config=LiveAssistantConfig(language="en"))  # type: ignore[arg-type]

    assert not assistant._barge_in_enabled()


def test_confirmed_barge_in_requires_wake_or_interrupt_phrase() -> None:
    """Ambient speech should not interrupt playback unless it addresses the assistant."""

    assistant = LiveVoiceAssistant(
        pipeline=object(),
        config=LiveAssistantConfig(language="en", wake_word="hey lemon"),
    )  # type: ignore[arg-type]

    assert assistant._is_confirmed_barge_in_phrase("hey lemon stop")
    assert assistant._is_confirmed_barge_in_phrase("lemon pause")
    assert not assistant._is_confirmed_barge_in_phrase("can you repeat that")


def test_effective_streaming_speech_end_timeout_enforces_floor() -> None:
    """Streaming speech-end timeout should use a safe minimum floor."""

    assistant = LiveVoiceAssistant(
        pipeline=object(),
        config=LiveAssistantConfig(language="en", streaming_speech_end_timeout_seconds=0.8),
    )  # type: ignore[arg-type]

    assert assistant._effective_streaming_speech_end_timeout_seconds() == 1.2


def test_effective_streaming_speech_end_timeout_keeps_larger_values() -> None:
    """Streaming speech-end timeout should keep caller values above the floor."""

    assistant = LiveVoiceAssistant(
        pipeline=object(),
        config=LiveAssistantConfig(language="en", streaming_speech_end_timeout_seconds=2.2),
    )  # type: ignore[arg-type]

    assert assistant._effective_streaming_speech_end_timeout_seconds() == 2.2


def test_play_answer_and_cleanup_deletes_audio_file_on_success(tmp_path, caplog) -> None:
    """Answer audio files should be removed after normal playback."""

    player = _RecordingPlayer()
    assistant = LiveVoiceAssistant(
        pipeline=object(),
        config=LiveAssistantConfig(language="en"),
        player=player,
    )  # type: ignore[arg-type]
    audio_file = tmp_path / "answer.mp3"
    audio_file.write_bytes(b"fake")

    import logging

    caplog.set_level(logging.INFO)
    keep_running = asyncio.run(assistant._play_answer_and_cleanup(audio_file))

    assert keep_running
    assert player.played_paths == [str(audio_file)]
    assert not audio_file.exists()


def test_audio_player_logs_playback_start_and_finish(tmp_path, monkeypatch, caplog) -> None:
    """Playback diagnostics should include file size and completion."""

    audio_file = tmp_path / "answer.mp3"
    audio_file.write_bytes(b"fake")
    played_paths: list[str] = []

    def fake_playsound(path: str, block: bool = True) -> None:
        played_paths.append(path)

    monkeypatch.setitem(sys.modules, "playsound", SimpleNamespace(playsound=fake_playsound))

    import logging

    caplog.set_level(logging.INFO)
    monkeypatch.setenv("VOICE_LOOP_DEBUG_AUDIO_IO", "true")
    AudioPlayer().play(audio_file)

    assert played_paths == [str(audio_file)]
    assert any("Playback diagnostics: started" in record.getMessage() for record in caplog.records)
    assert any("Playback diagnostics: finished" in record.getMessage() for record in caplog.records)


def test_audio_player_hides_playback_diagnostics_by_default(tmp_path, monkeypatch, caplog) -> None:
    """Playback byte-level diagnostics should be debug-gated."""

    audio_file = tmp_path / "answer.mp3"
    audio_file.write_bytes(b"fake")

    def fake_playsound(path: str, block: bool = True) -> None:
        return None

    monkeypatch.setitem(sys.modules, "playsound", SimpleNamespace(playsound=fake_playsound))
    monkeypatch.delenv("VOICE_LOOP_DEBUG_AUDIO_IO", raising=False)

    import logging

    caplog.set_level(logging.INFO)
    AudioPlayer().play(audio_file)

    assert not any("Playback diagnostics:" in record.getMessage() for record in caplog.records)


def test_play_answer_and_cleanup_deletes_audio_file_on_runtime_error(tmp_path) -> None:
    """Answer audio files should still be removed when playback raises runtime errors."""

    player = _RecordingPlayer(runtime_error=True)
    assistant = LiveVoiceAssistant(
        pipeline=object(),
        config=LiveAssistantConfig(language="en"),
        player=player,
    )  # type: ignore[arg-type]
    audio_file = tmp_path / "answer_runtime_error.mp3"
    audio_file.write_bytes(b"fake")

    keep_running = asyncio.run(assistant._play_answer_and_cleanup(audio_file))

    assert not keep_running
    assert not audio_file.exists()


def test_play_answer_and_cleanup_deletes_audio_file_on_generic_error(tmp_path) -> None:
    """Answer audio files should be removed when playback has non-fatal transient errors."""

    player = _RecordingPlayer(generic_error=True)
    assistant = LiveVoiceAssistant(
        pipeline=object(),
        config=LiveAssistantConfig(language="en"),
        player=player,
    )  # type: ignore[arg-type]
    audio_file = tmp_path / "answer_generic_error.mp3"
    audio_file.write_bytes(b"fake")

    keep_running = asyncio.run(assistant._play_answer_and_cleanup(audio_file))

    assert keep_running
    assert not audio_file.exists()


def test_should_speak_reprompt_once_in_adaptive_mode() -> None:
    """Adaptive mode should reprompt once on early ignored turns."""

    assistant = LiveVoiceAssistant(
        pipeline=object(),
        config=LiveAssistantConfig(
            language="en",
            wake_ack_mode="adaptive",
            wake_ack_adaptive_speak_on_wake=False,
        ),
    )  # type: ignore[arg-type]

    assert assistant._should_speak_reprompt(turn=1, reprompted=False)
    assert not assistant._should_speak_reprompt(turn=3, reprompted=False)
    assert not assistant._should_speak_reprompt(turn=1, reprompted=True)


def test_should_not_reprompt_when_adaptive_speaks_immediately() -> None:
    """Adaptive mode with immediate wake speech should not reprompt again."""

    assistant = LiveVoiceAssistant(
        pipeline=object(),
        config=LiveAssistantConfig(
            language="en",
            wake_ack_mode="adaptive",
            wake_ack_adaptive_speak_on_wake=True,
        ),
    )  # type: ignore[arg-type]

    assert not assistant._should_speak_reprompt(turn=1, reprompted=False)


def test_request_ready_cue_uses_cached_speech_rotation(tmp_path) -> None:
    """Request-ready cue should synthesize once per cue text when caching is enabled."""

    pipeline = _CuePipeline()
    player = _RecordingPlayer()
    assistant = LiveVoiceAssistant(
        pipeline=pipeline,
        config=LiveAssistantConfig(
            language="en",
            request_ready_texts_en=("Anything else?",),
            temp_audio_dir=tmp_path,
        ),
        player=player,
    )

    asyncio.run(assistant._play_request_ready_cue())
    asyncio.run(assistant._play_request_ready_cue())

    assert len(pipeline.primary_tts_provider.calls) == 1
    assert pipeline.primary_tts_provider.calls[0][0] == "Anything else?"
    assert len(player.played_paths) == 2
    assert all("request_ready_en_0_" in path for path in player.played_paths)


def test_first_request_ready_cue_uses_direct_prompt(tmp_path) -> None:
    """The first post-wake request should not say a follow-up phrase."""

    pipeline = _CuePipeline()
    player = _RecordingPlayer()
    assistant = LiveVoiceAssistant(
        pipeline=pipeline,
        config=LiveAssistantConfig(
            language="en",
            request_ready_first_text_en="Go ahead.",
            request_ready_texts_en=("Anything else?",),
            temp_audio_dir=tmp_path,
        ),
        player=player,
    )

    asyncio.run(assistant._play_request_ready_cue("first_request"))

    assert pipeline.primary_tts_provider.calls[0][0] == "Go ahead."
    assert "request_ready_first_en_0_" in player.played_paths[0]


def test_request_cue_defaults_use_vietnamese_diacritics() -> None:
    """Vietnamese cue defaults should be native text, not romanized ASCII."""

    config = LiveAssistantConfig(language="vi")

    assert config.request_ready_first_text_vi == "Bạn cứ hỏi nhé."
    assert "Bạn hỏi tiếp nhé." in config.request_ready_texts_vi
    assert "Để tôi kiểm tra." in config.thinking_texts_vi
    assert not LiveVoiceAssistant._looks_like_romanized_vietnamese(config.request_ready_first_text_vi)


def test_request_loop_skips_ready_cue_after_empty_retry(tmp_path) -> None:
    """An empty capture retry should listen again without another ready cue."""

    pipeline = _RequestPipeline(["", "Goodbye"])
    player = _RecordingPlayer()
    assistant = LiveVoiceAssistant(
        pipeline=pipeline,
        config=LiveAssistantConfig(
            language="en",
            request_ready_first_text_en="Go ahead.",
            request_ready_texts_en=("Go ahead.",),
            request_ready_cache=False,
            temp_audio_dir=tmp_path,
        ),
        player=player,
    )

    asyncio.run(assistant._conversation_loop())

    ready_cue_calls = [call for call in pipeline.primary_tts_provider.calls if call[0] == "Go ahead."]
    assert len(ready_cue_calls) == 1


def test_prime_request_cue_cache_prepares_first_prompt(tmp_path) -> None:
    """Startup cache priming should synthesize the first request cue before live use."""

    pipeline = _CuePipeline()
    assistant = LiveVoiceAssistant(
        pipeline=pipeline,
        config=LiveAssistantConfig(
            language="en",
            request_ready_first_text_en="Go ahead.",
            request_ready_texts_en=("Anything else?",),
            thinking_texts_en=("One moment.",),
            temp_audio_dir=tmp_path,
        ),
    )

    asyncio.run(assistant._prime_request_cue_cache())

    assert ("request_ready_first", "en", 0) in assistant._cue_cache_paths
    assert ("request_ready", "en", 0) in assistant._cue_cache_paths
    assert ("thinking", "en", 0) in assistant._cue_cache_paths


def test_startup_cache_priming_runs_in_background(tmp_path) -> None:
    """Startup cue-cache priming should not block wake-listening startup."""

    pipeline = _CuePipeline()
    assistant = LiveVoiceAssistant(
        pipeline=pipeline,
        config=LiveAssistantConfig(
            language="en",
            wake_ack_mode="none",
            request_ready_cache=True,
            temp_audio_dir=tmp_path,
        ),
    )
    started = False
    finished = False

    async def slow_prime_request_cue_cache() -> None:
        nonlocal started, finished
        started = True
        await asyncio.sleep(0.01)
        finished = True

    assistant._prime_request_cue_cache = slow_prime_request_cue_cache  # type: ignore[method-assign]

    async def scenario() -> None:
        assistant._start_startup_cache_priming()
        assert assistant._startup_cache_tasks
        assert not finished
        await asyncio.sleep(0)
        assert started
        assert not finished
        await asyncio.gather(*assistant._startup_cache_tasks)
        assert finished

    asyncio.run(scenario())


def test_initial_output_language_follows_session_language_in_auto_mode() -> None:
    """The first request cue should not default to English in a Vietnamese session."""

    assistant = LiveVoiceAssistant(
        pipeline=object(),
        config=LiveAssistantConfig(language="vi", output_language_mode="auto", output_language_fixed="en"),
    )  # type: ignore[arg-type]

    assert assistant._current_output_language == "vi"


def test_prime_request_cue_cache_skips_disabled_thinking_prompts(tmp_path) -> None:
    """Disabled thinking cues should not add startup synthesis work."""

    pipeline = _CuePipeline()
    assistant = LiveVoiceAssistant(
        pipeline=pipeline,
        config=LiveAssistantConfig(
            language="en",
            thinking_cue_enabled=False,
            temp_audio_dir=tmp_path,
        ),
    )

    asyncio.run(assistant._prime_request_cue_cache())

    assert not any(cache_key[0] == "thinking" for cache_key in assistant._cue_cache_paths)


def test_thinking_cue_uses_configured_language_text(tmp_path) -> None:
    """Slow LLM cue should use the output language prompt set."""

    pipeline = _CuePipeline()
    player = _RecordingPlayer()
    assistant = LiveVoiceAssistant(
        pipeline=pipeline,
        config=LiveAssistantConfig(
            language="vi",
            thinking_texts_vi=("Cho toi mot chut.",),
            temp_audio_dir=tmp_path,
        ),
        player=player,
    )

    asyncio.run(assistant._play_thinking_cue("vi"))

    assert pipeline.primary_tts_provider.calls[0][0] == "Cho toi mot chut."
    assert pipeline.primary_tts_provider.calls[0][1] == "vi"


def test_capture_quality_reprompt_blocks_weak_question_fragment() -> None:
    """Weak-progress captures with only a question stem should reprompt before routing."""

    assistant = LiveVoiceAssistant(pipeline=object(), config=LiveAssistantConfig(language="en"))  # type: ignore[arg-type]

    reason = assistant._capture_quality_reprompt_reason(
        "What is?",
        {"end_reason": "weak_stt_progress", "interim_tokens": 2.0},
    )

    assert reason == "unfinished_question_stem"


def test_capture_quality_accepts_valid_short_followup() -> None:
    """Short but meaningful follow-ups should still be allowed."""

    assistant = LiveVoiceAssistant(pipeline=object(), config=LiveAssistantConfig(language="en"))  # type: ignore[arg-type]

    assert assistant._capture_quality_reprompt_reason("How much per semester?", {"end_reason": "local_silence"}) is None


def test_log_command_card_highlights_fixed_english_profile(caplog) -> None:
    """Startup diagnostics should make the English-only fallback profile obvious."""

    assistant = LiveVoiceAssistant(
        pipeline=object(),
        config=LiveAssistantConfig(language="en", language_mode="fixed", output_language_mode="fixed", output_language_fixed="en"),
    )  # type: ignore[arg-type]

    import logging

    caplog.set_level(logging.INFO)
    assistant._log_command_card({"index": 1}, ("hey lemon",))

    messages = [record.getMessage() for record in caplog.records]
    assert any("language_mode=fixed output_mode=fixed output=en" in message for message in messages)


def test_should_end_conversation_by_ignored_turn_limit() -> None:
    """Conversation loop should end when ignored-turn limit is reached."""

    assistant = LiveVoiceAssistant(
        pipeline=object(),
        config=LiveAssistantConfig(language="en", request_max_ignored_turns=2),
    )  # type: ignore[arg-type]

    assert assistant._should_end_conversation(turn=1, ignored_turns=2, elapsed_seconds=5.0)


def test_should_end_conversation_by_turn_limit() -> None:
    """Conversation loop should end when turn limit is reached."""

    assistant = LiveVoiceAssistant(
        pipeline=object(),
        config=LiveAssistantConfig(language="en", request_max_turns=3),
    )  # type: ignore[arg-type]

    assert assistant._should_end_conversation(turn=3, ignored_turns=0, elapsed_seconds=5.0)


def test_should_end_conversation_by_session_time_limit() -> None:
    """Conversation loop should end when session duration limit is reached."""

    assistant = LiveVoiceAssistant(
        pipeline=object(),
        config=LiveAssistantConfig(language="en", request_max_session_seconds=10.0),
    )  # type: ignore[arg-type]

    assert assistant._should_end_conversation(turn=1, ignored_turns=0, elapsed_seconds=10.0)


def test_should_end_conversation_idle_timeout_renews_after_activity() -> None:
    """Conversation loop should keep running when recent activity refreshed the idle timer."""

    assistant = LiveVoiceAssistant(
        pipeline=object(),
        config=LiveAssistantConfig(
            language="en",
            request_idle_timeout_seconds=10.0,
            request_max_session_seconds=180.0,
        ),
    )  # type: ignore[arg-type]

    assert not assistant._should_end_conversation(
        turn=3,
        ignored_turns=0,
        elapsed_seconds=95.0,
        idle_elapsed_seconds=2.0,
    )
    assert assistant._should_end_conversation(
        turn=3,
        ignored_turns=0,
        elapsed_seconds=95.0,
        idle_elapsed_seconds=10.0,
    )


def test_resolve_turn_languages_switches_to_detected_language() -> None:
    """Adaptive auto mode should switch output language on confident detection."""

    assistant = LiveVoiceAssistant(
        pipeline=object(),
        config=LiveAssistantConfig(language="en"),
    )  # type: ignore[arg-type]

    input_language, output_language, confidence, reason = assistant._resolve_turn_languages("Tôi muốn hỏi học phí")

    assert input_language == "vi"
    assert output_language == "vi"
    assert confidence > 0.5
    assert "switched" in reason


def test_resolve_turn_languages_respects_sticky_turns() -> None:
    """Sticky turns should hold output language briefly after a recent switch."""

    assistant = LiveVoiceAssistant(
        pipeline=object(),
        config=LiveAssistantConfig(language="en"),
    )  # type: ignore[arg-type]

    assistant._current_output_language = "vi"
    assistant._language_sticky_remaining = 1

    input_language, output_language, _, reason = assistant._resolve_turn_languages("please help me")

    assert input_language == "en"
    assert output_language == "vi"
    assert "sticky_hold" in reason
