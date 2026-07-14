"""Tests for environment-backed app configuration."""

from __future__ import annotations

from voice_loop.config import AppConfig


def test_from_env_adds_scoped_greenwich_domain_hints(monkeypatch) -> None:
    monkeypatch.setenv("VOICE_LOOP_STT_HINT_PHRASES", "hey lemon")

    config = AppConfig.from_env(language="en", seed_demo_data=False)

    assert "hey lemon" in config.stt_hint_phrases
    assert "Greenwich Vietnam" in config.stt_hint_phrases
    assert "Greenwich Việt Nam" in config.stt_hint_phrases
    assert "University of Greenwich" in config.stt_hint_phrases
    assert "Đại học Greenwich" in config.stt_hint_phrases
    assert config.domain_profile == "greenwich"


def test_from_env_can_disable_domain_profile_hints(monkeypatch) -> None:
    monkeypatch.setenv("VOICE_LOOP_DOMAIN_PROFILE", "none")
    monkeypatch.setenv("VOICE_LOOP_STT_HINT_PHRASES", "hey lemon")

    config = AppConfig.from_env(language="en", seed_demo_data=False)

    assert config.stt_hint_phrases == ("hey lemon",)
    assert config.domain_profile == "none"
