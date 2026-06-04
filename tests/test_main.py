"""Tests for CLI language profile handling."""

from __future__ import annotations

from pathlib import Path

from main import apply_language_profile, parse_args
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

