"""Tests for TTS provider configuration."""

from __future__ import annotations

from pathlib import Path

from voice_loop.config import AppConfig
from voice_loop.factory import build_tts_provider
from voice_loop.providers.edge_tts import EdgeTTSProvider
from voice_loop.providers.google_tts import GoogleTextToSpeechProvider, _voice_language_code


def _base_config(tts_provider: str, voice_en: str, voice_vi: str) -> AppConfig:
    return AppConfig(
        language="en",
        db_path=Path("data/knowledge_base.sqlite3"),
        output_dir=Path("output"),
        tts_provider=tts_provider,
        tts_voice_en=voice_en,
        tts_voice_vi=voice_vi,
    )


def test_google_tts_provider_uses_configured_voices() -> None:
    provider = build_tts_provider(_base_config("google", "en-GB-Neural2-F", "vi-VN-Neural2-A"))

    assert isinstance(provider, GoogleTextToSpeechProvider)
    assert provider.voice_en == "en-GB-Neural2-F"
    assert provider.voice_vi == "vi-VN-Neural2-A"


def test_edge_tts_provider_uses_configured_voices() -> None:
    provider = build_tts_provider(_base_config("edge", "en-GB-SoniaNeural", "vi-VN-HoaiMyNeural"))

    assert isinstance(provider, EdgeTTSProvider)
    assert provider.voice_en == "en-GB-SoniaNeural"
    assert provider.voice_vi == "vi-VN-HoaiMyNeural"


def test_google_voice_language_code_uses_voice_prefix() -> None:
    assert _voice_language_code("en-GB-Neural2-F", "en-US") == "en-GB"
    assert _voice_language_code("vi-VN-Neural2-A", "vi-VN") == "vi-VN"
