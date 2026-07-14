"""Tests for CLI language profile handling."""

from __future__ import annotations

from pathlib import Path

from main import apply_language_profile, load_script_lines, parse_args, resolve_mode, validate_script_mode
from voice_loop.config import AppConfig


def _base_config() -> AppConfig:
    return AppConfig(
        language="en",
        db_path=Path("data/knowledge_base.sqlite3"),
        output_dir=Path("output"),
    )


def test_parse_args_accepts_single_adaptive_language_profile() -> None:
    args = parse_args(["--language", "adaptive"])

    assert args.language == "adaptive"
    assert not hasattr(args, "language_mode")
    assert not hasattr(args, "output_language_mode")
    assert not hasattr(args, "output_language_fixed")


def test_parse_args_accepts_explicit_modes() -> None:
    assert parse_args(["--mode", "live"]).mode == "live"
    assert parse_args(["--mode", "diagnose", "--diagnose-transcript", "hello"]).mode == "diagnose"
    assert parse_args(["--mode", "script", "--script-text", "Welcome."]).mode == "script"


def test_parse_args_accepts_script_operator_controls() -> None:
    args = parse_args(
        [
            "--mode",
            "script",
            "--script-text",
            "Welcome.",
            "--script-start-at",
            "2",
            "--script-delay-seconds",
            "1.5",
            "--script-manual-next",
            "--script-validate",
        ]
    )

    assert args.script_start_at == 2
    assert args.script_delay_seconds == 1.5
    assert args.script_manual_next
    assert args.script_validate


def test_resolve_mode_defaults_to_live() -> None:
    assert resolve_mode(parse_args([])) == "live"


def test_resolve_mode_infers_diagnose_from_legacy_argument() -> None:
    assert resolve_mode(parse_args(["--diagnose-transcript", "What is Greenwich Vietnam?"])) == "diagnose"


def test_load_script_lines_combines_file_and_cli_text(tmp_path: Path) -> None:
    script_file = tmp_path / "script.txt"
    script_file.write_text("Welcome.\n\nPlease take your seats.\n", encoding="utf-8")
    args = parse_args(["--mode", "script", "--script-file", str(script_file), "--script-text", "Thank you."])

    assert load_script_lines(args) == ("Welcome.", "", "Please take your seats.", "Thank you.")


def test_validate_script_mode_prints_operator_summary(capsys) -> None:
    args = parse_args(["--mode", "script", "--script-text", "Welcome.", "--script-text", "Begin."])
    config = _base_config()

    validate_script_mode(args, config, ("Welcome.", "Begin."))

    output = capsys.readouterr().out
    assert "script_line_count=2" in output
    assert "script_tts_provider=google" in output
    assert "script_output_root=output" in output


def test_apply_language_profile_adaptive_uses_detected_input_for_output() -> None:
    config = apply_language_profile(_base_config(), "adaptive")

    assert config.language_mode == "adaptive"
    assert config.output_language_mode == "auto"


def test_apply_language_profile_english_forces_fixed_input_and_output() -> None:
    config = apply_language_profile(_base_config(), "en")

    assert config.language == "en"
    assert config.language_mode == "fixed"
    assert config.output_language_mode == "fixed"
    assert config.output_language_fixed == "en"


def test_apply_language_profile_vietnamese_forces_fixed_input_and_output() -> None:
    config = apply_language_profile(_base_config(), "vi")

    assert config.language == "vi"
    assert config.language_mode == "fixed"
    assert config.output_language_mode == "fixed"
    assert config.output_language_fixed == "vi"
