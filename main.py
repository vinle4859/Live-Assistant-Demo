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
from voice_loop.factory import build_default_pipeline, build_tts_provider
from voice_loop.live_assistant import AudioPlayer, LiveAssistantConfig, LiveVoiceAssistant, detect_input_language
from voice_loop.scripted_speech import ScriptedSpeechRunner, parse_script_lines


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for live assistant mode."""

    parser = argparse.ArgumentParser(description="Run the voice-to-voice pipeline")
    parser.add_argument(
        "--mode",
        choices=["live", "diagnose", "script"],
        default=None,
        help="Runtime mode. Defaults to live, or diagnose when --diagnose-transcript is supplied.",
    )
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
    parser.add_argument(
        "--script-text",
        action="append",
        default=[],
        help="Script line to synthesize in script mode. Can be passed more than once.",
    )
    parser.add_argument(
        "--script-file",
        default=None,
        help="UTF-8 text file containing one script line per line for script mode.",
    )
    parser.add_argument(
        "--script-no-play",
        action="store_true",
        help="Render script audio without playing it.",
    )
    parser.add_argument(
        "--script-start-at",
        type=int,
        default=1,
        help="First one-based script line number to render or play.",
    )
    parser.add_argument(
        "--script-delay-seconds",
        type=float,
        default=0.0,
        help="Delay after each played script line.",
    )
    parser.add_argument(
        "--script-manual-next",
        action="store_true",
        help="Wait for Enter before each rendered script line is played.",
    )
    parser.add_argument(
        "--script-validate",
        action="store_true",
        help="Validate script mode inputs without synthesizing or playing audio.",
    )
    return parser.parse_args(argv)


async def _run_async() -> None:
    """Execute the selected assistant mode."""

    args = parse_args()
    mode = resolve_mode(args)
    selected_language = "en" if args.language == "adaptive" else args.language or "en"
    config = AppConfig.from_env(
        language=selected_language,
        db_path=args.db_path,
        output_dir=args.output_dir,
        seed_demo_data=not args.no_demo_seed,
    )
    config = apply_language_profile(config, args.language)

    configure_logging(config.log_level, mode)

    if mode == "diagnose":
        await run_diagnose_mode(args, config)
        return
    if mode == "script":
        await run_script_mode(args, config)
        return
    await run_live_mode(config)


def resolve_mode(args: argparse.Namespace) -> str:
    """Return the selected runtime mode with backward-compatible diagnose inference."""

    if args.mode is not None:
        return args.mode
    if args.diagnose_transcript is not None:
        return "diagnose"
    return "live"


async def run_diagnose_mode(args: argparse.Namespace, config: AppConfig) -> None:
    """Run one transcript through routing diagnostics without microphone capture."""

    pipeline = build_default_pipeline(config)
    if args.diagnose_transcript is None:
        raise ValueError("Diagnose mode requires --diagnose-transcript.")
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


async def run_script_mode(args: argparse.Namespace, config: AppConfig) -> None:
    """Render and optionally play fixed event script lines."""

    script_lines = load_script_lines(args)
    if args.script_validate:
        validate_script_mode(args, config, script_lines)
        return
    runner = ScriptedSpeechRunner(
        tts_provider=build_tts_provider(config),
        output_dir=config.output_dir,
        timeout_seconds=config.provider_timeout_seconds,
        player=AudioPlayer(),
    )
    output_paths = await runner.run(
        lines=script_lines,
        language=config.output_language_fixed if config.output_language_mode == "fixed" else config.language,
        play_audio=not args.script_no_play,
        start_at=args.script_start_at,
        delay_seconds=args.script_delay_seconds,
        manual_next=args.script_manual_next,
    )
    for output_path in output_paths:
        print(f"script_audio_output_path={output_path}")


def validate_script_mode(args: argparse.Namespace, config: AppConfig, lines: tuple[str, ...]) -> None:
    """Validate script inputs and print event-run configuration."""

    parsed_lines = parse_script_lines(lines, start_at=args.script_start_at)
    if not parsed_lines:
        raise ValueError("Script mode requires at least one non-empty line.")
    language = config.output_language_fixed if config.output_language_mode == "fixed" else config.language
    print(f"script_line_count={len(parsed_lines)}")
    print(f"script_first_line={parsed_lines[0].index}")
    print(f"script_last_line={parsed_lines[-1].index}")
    print(f"script_language={language}")
    print(f"script_tts_provider={config.tts_provider}")
    print(f"script_tts_voice_en={config.tts_voice_en or '<provider default>'}")
    print(f"script_tts_voice_vi={config.tts_voice_vi or '<provider default>'}")
    print(f"script_output_root={config.output_dir / 'script_sessions'}")


def load_script_lines(args: argparse.Namespace) -> tuple[str, ...]:
    """Load script lines from CLI text and/or a UTF-8 script file."""

    lines: list[str] = []
    if args.script_file:
        script_path = Path(args.script_file)
        if not script_path.exists():
            raise FileNotFoundError(f"Script file does not exist: {script_path}")
        if not script_path.is_file():
            raise ValueError(f"Script path is not a file: {script_path}")
        lines.extend(script_path.read_text(encoding="utf-8").splitlines())
    lines.extend(args.script_text)
    return tuple(lines)


async def run_live_mode(config: AppConfig) -> None:
    """Execute the live wake-word assistant."""

    pipeline = build_default_pipeline(config)
    assistant = LiveVoiceAssistant(
        pipeline=pipeline,
        config=build_live_assistant_config(config),
    )
    await assistant.run()


def build_live_assistant_config(config: AppConfig) -> LiveAssistantConfig:
    """Map app configuration into live assistant configuration."""

    return LiveAssistantConfig(
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
    )


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


def configure_logging(log_level: str, mode: str = "live") -> None:
    """Configure process-wide logging with a consistent, timestamped format."""

    _configure_stream_encoding(sys.stdout)
    _configure_stream_encoding(sys.stderr)
    level_name = (log_level or "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    log_dir = Path("output") / f"{mode}_sessions"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{mode}_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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
    logging.getLogger(__name__).info("Session log: mode=%s path=%s", mode, log_path)
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

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        asyncio.run(_run_async())
    except KeyboardInterrupt:
        print("Stopped by user.")
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Runtime error: {exc}")
        raise SystemExit(1)
if __name__ == "__main__":
    main()
