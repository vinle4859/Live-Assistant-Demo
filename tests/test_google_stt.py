"""Unit tests for Google STT provider helpers."""

from __future__ import annotations

import struct
from collections import deque

from voice_loop.providers.google_stt import GoogleSpeechToTextProvider


def test_pcm_stats_and_speech_gate_distinguish_noise_from_speech() -> None:
    """Local gating should separate low-energy chunks from speech-like chunks."""

    quiet_chunk = struct.pack("<hhhh", 5, -5, 10, -10)
    speech_chunk = struct.pack("<hhhh", 1000, -1200, 800, -900)

    quiet_stats = GoogleSpeechToTextProvider._pcm_stats(quiet_chunk)
    speech_stats = GoogleSpeechToTextProvider._pcm_stats(speech_chunk)

    assert not GoogleSpeechToTextProvider._is_speech_chunk(quiet_stats, 200.0, 500.0)
    assert GoogleSpeechToTextProvider._is_speech_chunk(speech_stats, 200.0, 500.0)


def test_vad_thresholds_cap_high_ambient_noise() -> None:
    """High ambient calibration should not make normal speech impossible to start."""

    thresholds = GoogleSpeechToTextProvider._build_vad_thresholds(
        ambient_rms=349.4,
        ambient_peak=2682.0,
        min_speech_rms=30.0,
        min_speech_peak=250.0,
    )

    assert thresholds["start_rms"] == 180.0
    assert thresholds["start_peak"] == 1400.0
    assert thresholds["tail_rms"] == 100.0
    assert thresholds["tail_peak"] == 800.0


def test_vad_thresholds_apply_low_ambient_floors() -> None:
    """Quiet calibration should not make one-off device noise start a turn."""

    thresholds = GoogleSpeechToTextProvider._build_vad_thresholds(
        ambient_rms=40.0,
        ambient_peak=140.0,
        min_speech_rms=30.0,
        min_speech_peak=250.0,
    )

    assert thresholds["start_rms"] == 80.0
    assert thresholds["start_peak"] == 500.0
    assert thresholds["tail_rms"] == 52.0
    assert thresholds["tail_peak"] == 350.0


def test_vad_debounce_requires_multiple_speech_like_chunks() -> None:
    """Speech start should require repeated evidence, not one noisy chunk."""

    weak_hits: deque[float] = deque()
    strong_hits: deque[float] = deque()

    GoogleSpeechToTextProvider._record_debounce_hit(weak_hits, strong_hits, 1.0, False, 0.5)
    assert not GoogleSpeechToTextProvider._has_confirmed_speech_start(weak_hits, strong_hits)

    GoogleSpeechToTextProvider._record_debounce_hit(weak_hits, strong_hits, 1.2, False, 0.5)
    assert not GoogleSpeechToTextProvider._has_confirmed_speech_start(weak_hits, strong_hits)

    GoogleSpeechToTextProvider._record_debounce_hit(weak_hits, strong_hits, 1.4, False, 0.5)
    assert GoogleSpeechToTextProvider._has_confirmed_speech_start(weak_hits, strong_hits)


def test_vad_debounce_accepts_two_strong_chunks() -> None:
    """Strong speech chunks should start faster than weak speech-like chunks."""

    weak_hits: deque[float] = deque()
    strong_hits: deque[float] = deque()

    GoogleSpeechToTextProvider._record_debounce_hit(weak_hits, strong_hits, 1.0, True, 0.5)
    assert not GoogleSpeechToTextProvider._has_confirmed_speech_start(weak_hits, strong_hits)

    GoogleSpeechToTextProvider._record_debounce_hit(weak_hits, strong_hits, 1.2, True, 0.5)
    assert GoogleSpeechToTextProvider._has_confirmed_speech_start(weak_hits, strong_hits)


def test_streaming_interim_fallback_requires_substantial_text() -> None:
    """Tiny interim fragments should not become user questions."""

    assert GoogleSpeechToTextProvider._is_substantial_interim("What is weather")
    assert not GoogleSpeechToTextProvider._is_substantial_interim("What is")


def test_provider_wake_match_accepts_alias_and_rejects_plain_hello() -> None:
    """Provider-local wake matching should mirror live wake aliases."""

    wake_phrases = ("hey lemon", "hello lemon", "hey leman", "le minh")

    assert GoogleSpeechToTextProvider._match_wake_phrase("Hello, lemon.", wake_phrases)
    assert GoogleSpeechToTextProvider._match_wake_phrase("hey leman", wake_phrases)
    assert GoogleSpeechToTextProvider._match_wake_phrase("hello", wake_phrases) is None


def test_vad_tail_allows_short_natural_pause() -> None:
    """A 1.5 second pause should not end a 2 second local tail window."""

    assert not GoogleSpeechToTextProvider._should_end_for_local_silence(1.5, 2.0)
    assert GoogleSpeechToTextProvider._should_end_for_local_silence(2.1, 2.0)


def test_vad_hard_active_cap_still_ends_noisy_turn() -> None:
    """Sustained active capture should still respect the hard cap."""

    assert not GoogleSpeechToTextProvider._should_end_for_max_active(9.5, 10.0)
    assert GoogleSpeechToTextProvider._should_end_for_max_active(10.0, 10.0)


def test_no_stt_progress_stop_requires_elapsed_time_and_no_text() -> None:
    """No-progress stop should only trigger after active capture has produced no STT text."""

    stats = {"final_tokens": 0.0, "interim_tokens": 0.0}

    assert not GoogleSpeechToTextProvider._should_end_for_no_stt_progress(4.4, 4.5, stats)
    assert GoogleSpeechToTextProvider._should_end_for_no_stt_progress(4.5, 4.5, stats)

    stats["interim_tokens"] = 1.0
    assert not GoogleSpeechToTextProvider._should_end_for_no_stt_progress(5.0, 4.5, stats)


def test_weak_stt_progress_stop_allows_meaningful_interim() -> None:
    """Weak-progress stop should preserve captures with enough interim tokens."""

    weak_stats = {"final_tokens": 0.0, "interim_tokens": 2.0}
    good_stats = {"final_tokens": 0.0, "interim_tokens": 3.0}

    assert not GoogleSpeechToTextProvider._should_end_for_weak_stt_progress(5.9, 6.0, 3, weak_stats)
    assert GoogleSpeechToTextProvider._should_end_for_weak_stt_progress(6.0, 6.0, 3, weak_stats)
    assert not GoogleSpeechToTextProvider._should_end_for_weak_stt_progress(6.0, 6.0, 3, good_stats)


def test_progress_end_reason_normalizes_after_late_text() -> None:
    """Stale no-progress reasons should not survive after usable text arrives."""

    stats = {"end_reason": "no_stt_progress", "final_tokens": 9.0, "interim_tokens": 2.0}

    assert GoogleSpeechToTextProvider._normalize_progress_end_reason(stats) == "stt_progress_received"


def test_tail_speech_gate_rejects_isolated_peak_noise() -> None:
    """Post-start tail gate should not extend capture for peak-only background noise."""

    peak_only = {"rms": 30.0, "peak": 900.0}
    speech_like = {"rms": 60.0, "peak": 900.0}

    assert not GoogleSpeechToTextProvider._is_tail_speech_chunk(peak_only, 100.0, 800.0)
    assert GoogleSpeechToTextProvider._is_tail_speech_chunk(speech_like, 100.0, 800.0)
