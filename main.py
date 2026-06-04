"""Command-line entry point for the voice-to-voice pipeline."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
import logging
from pathlib import Path
import sys
from datetime import datetime

from voice_loop.config import AppConfig
from voice_loop.factory import build_default_pipeline
from voice_loop.live_assistant import LiveAssistantConfig, LiveVoiceAssistant, detect_input_language


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for live assistant mode."""

    parser = argparse.ArgumentParser(description="Run the voice-to-voice pipeline")
    parser.add_argument(
        "--language",
        choices=["adaptive", "en", "vi"],
        default=None,
        help="Language profile: adaptive detects EN/VI per turn; en and vi force fixed input/output.",
    )
    parser.add_argument("--db-path", default=None, help="Override the SQLite database path")
    parser.add_argument("--output-dir", default=None, help="Override the synthesized audio output directory")
    parser.add_argument("--no-demo-seed", action="store_true", help="Do not seed demo Q&A rows when the database is empty")
    parser.add_argument(
        "--diagnose-transcript",
        default=None,
        help="Run one transcript through DB/LLM/TTS diagnostics without microphone capture.",
    )
    parser.add_argument(
        "--diagnose-language",
        choices=["en", "vi"],
        default=None,
        help="Language to use with --diagnose-transcript. Defaults to the active language profile.",
    )
    return parser.parse_args(argv)


async def _run_async() -> None:
    """Execute the live wake-word assistant."""

    args = parse_args()
    selected_language = "en" if args.language == "adaptive" else args.language or "en"
    config = AppConfig.from_env(
        language=selected_language,
        db_path=args.db_path,
        output_dir=args.output_dir,
        seed_demo_data=not args.no_demo_seed,
    )
    config = apply_language_profile(config, args.language)

    configure_logging(config.log_level)
    pipeline = build_default_pipeline(config)
    if args.diagnose_transcript is not None:
        if args.diagnose_language is not None:
            diagnose_language = args.diagnose_language
        elif config.language_mode == "adaptive":
            diagnose_language = detect_input_language(args.diagnose_transcript, config.language)[0]
        else:
            diagnose_language = config.language
        result = await pipeline.process_transcription(args.diagnose_transcript, diagnose_language)
        print(f"diagnose_language={diagnose_language}")
        print(f"resolved_source={result['resolved_source']}")
        print(f"response_text={result['response_text']}")
        print(f"audio_output_path={result['audio_output_path']}")
        return

    assistant = LiveVoiceAssistant(
        pipeline=pipeline,
        config=LiveAssistantConfig(
            language=config.language,
            language_mode=config.language_mode,
            output_language_mode=config.output_language_mode,
            output_language_fixed=config.output_language_fixed,
            enable_bilingual_output=config.enable_bilingual_output,
            language_switch_min_confidence=config.language_switch_min_confidence,
            language_switch_sticky_turns=config.language_switch_sticky_turns,
            language_override_commands=config.language_override_commands,
            wake_word=config.wake_word,
            wake_aliases=config.wake_aliases,
            wake_ack_mode=config.wake_ack_mode,
            wake_ack_beep_frequency=config.wake_ack_beep_frequency,
            wake_ack_beep_duration_ms=config.wake_ack_beep_duration_ms,
            wake_ack_prompt_text_en=config.wake_ack_prompt_text_en,
            wake_ack_prompt_text_vi=config.wake_ack_prompt_text_vi,
            wake_ack_adaptive_speak_on_wake=config.wake_ack_adaptive_speak_on_wake,
            request_ready_cue_mode=config.request_ready_cue_mode,
            request_ready_first_text_en=config.request_ready_first_text_en,
            request_ready_first_text_vi=config.request_ready_first_text_vi,
            request_ready_texts_en=config.request_ready_texts_en,
            request_ready_texts_vi=config.request_ready_texts_vi,
            request_ready_cache=config.request_ready_cache,
            thinking_cue_enabled=config.thinking_cue_enabled,
            thinking_cue_delay_seconds=config.thinking_cue_delay_seconds,
            thinking_texts_en=config.thinking_texts_en,
            thinking_texts_vi=config.thinking_texts_vi,
            wake_window_seconds=config.wake_window_seconds,
            utterance_seconds=config.utterance_seconds,
            utterance_min_rms=config.utterance_min_rms,
            utterance_min_peak=config.utterance_min_peak,
            request_post_tts_guard_seconds=config.request_post_tts_guard_seconds,
            minimum_transcript_characters=config.minimum_transcript_characters,
            enable_streaming_stt=config.enable_streaming_stt,
            streaming_chunk_duration_ms=config.streaming_chunk_duration_ms,
            streaming_speech_start_timeout_seconds=config.streaming_speech_start_timeout_seconds,
            streaming_speech_end_timeout_seconds=config.streaming_speech_end_timeout_seconds,
            streaming_local_speech_end_ms=config.streaming_local_speech_end_ms,
            streaming_max_active_seconds=config.streaming_max_active_seconds,
            streaming_no_progress_seconds=config.streaming_no_progress_seconds,
            streaming_weak_progress_seconds=config.streaming_weak_progress_seconds,
            streaming_weak_progress_min_tokens=config.streaming_weak_progress_min_tokens,
            preroll_enabled=config.preroll_enabled,
            preroll_ms=config.preroll_ms,
            startup_calibration_enabled=config.startup_calibration_enabled,
            startup_calibration_seconds=config.startup_calibration_seconds,
            request_max_ignored_turns=config.request_max_ignored_turns,
            request_max_turns=config.request_max_turns,
            request_idle_timeout_seconds=config.request_idle_timeout_seconds,
            request_max_session_seconds=config.request_max_session_seconds,
            barge_in_mode=config.barge_in_mode,
            barge_in_listen_seconds=config.barge_in_listen_seconds,
            barge_in_grace_seconds=config.barge_in_grace_seconds,
            barge_in_min_rms=config.barge_in_min_rms,
            barge_in_min_peak=config.barge_in_min_peak,
            sample_rate=config.sample_rate,
            input_device_index=config.input_device_index,
            debug_audio_io=config.debug_audio_io,
            debug_stt_stream=config.debug_stt_stream,
        ),
    )
    await assistant.run()


def apply_language_profile(config: AppConfig, language_profile: str | None) -> AppConfig:
    """Apply the public CLI language profile to the internal language policy fields."""

    if language_profile == "adaptive":
        return replace(
            config,
            language_mode="adaptive",
            output_language_mode="auto",
        )
    if language_profile in {"en", "vi"}:
        return replace(
            config,
            language=language_profile,
            language_mode="fixed",
            output_language_mode="fixed",
            output_language_fixed=language_profile,
        )
    return config


def configure_logging(log_level: str) -> None:
    """Configure process-wide logging with a consistent, timestamped format."""

    _configure_stream_encoding(sys.stdout)
    _configure_stream_encoding(sys.stderr)
    level_name = (log_level or "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    log_dir = Path("output/live_sessions")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"live_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logging.basicConfig(
        level=level,
        handlers=[stream_handler, file_handler],
        force=True,
    )
    logging.getLogger(__name__).info("Live session log: %s", log_path)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("google.auth").setLevel(logging.WARNING)
    logging.getLogger("google.api_core").setLevel(logging.WARNING)


def _configure_stream_encoding(stream) -> None:
    """Force UTF-8 console encoding when the active terminal defaults to a legacy code page."""

    reconfigure = getattr(stream, "reconfigure", None)
    if callable(reconfigure):
        try:
            reconfigure(encoding="utf-8", errors="backslashreplace")
        except (ValueError, OSError):
            pass


def main() -> None:
    """Launch the CLI entry point."""

    try:
        asyncio.run(_run_async())
    except KeyboardInterrupt:
        print("Stopped by user.")
    except RuntimeError as exc:
        print(f"Runtime error: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
