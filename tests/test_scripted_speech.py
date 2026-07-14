"""Tests for scripted event speech mode."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from voice_loop.scripted_speech import ScriptedSpeechRunner, parse_script_lines


class _RecordingTTSProvider:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, Path]] = []

    async def synthesize(self, text: str, language: str, output_path: Path) -> Path:
        self.calls.append((text, language, output_path))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(text.encode("utf-8"))
        return output_path


class _RecordingPlayer:
    def __init__(self, provider: _RecordingTTSProvider | None = None) -> None:
        self.provider = provider
        self.played_paths: list[Path] = []
        self.provider_call_counts_at_play: list[int] = []

    def play(self, audio_path: Path) -> None:
        if self.provider is not None:
            self.provider_call_counts_at_play.append(len(self.provider.calls))
        self.played_paths.append(audio_path)


def test_parse_script_lines_skips_empty_lines_and_preserves_positions() -> None:
    lines = parse_script_lines(("Welcome.", "", "  ", "Please sit."))

    assert [line.index for line in lines] == [1, 4]
    assert [line.text for line in lines] == ["Welcome.", "Please sit."]


def test_parse_script_lines_starts_at_requested_line() -> None:
    lines = parse_script_lines(("Welcome.", "", "Please sit.", "Begin."), start_at=3)

    assert [line.index for line in lines] == [3, 4]
    assert [line.text for line in lines] == ["Please sit.", "Begin."]


def test_scripted_speech_synthesizes_lines_in_order_without_playback(tmp_path: Path) -> None:
    provider = _RecordingTTSProvider()
    runner = ScriptedSpeechRunner(provider, tmp_path, timeout_seconds=1.0)

    output_paths = asyncio.run(runner.run(("Welcome.", "", "Please sit."), "en", play_audio=False))

    assert [call[0] for call in provider.calls] == ["Welcome.", "Please sit."]
    assert [call[1] for call in provider.calls] == ["en", "en"]
    assert len(output_paths) == 2
    assert all(path.exists() for path in output_paths)
    assert all(path.parent.parent.name == "script_sessions" for path in output_paths)


def test_scripted_speech_prerenders_before_playback(tmp_path: Path) -> None:
    provider = _RecordingTTSProvider()
    player = _RecordingPlayer(provider)
    runner = ScriptedSpeechRunner(provider, tmp_path, timeout_seconds=1.0, player=player)

    output_paths = asyncio.run(runner.run(("Xin chào.", "Cảm ơn."), "vi"))

    assert tuple(player.played_paths) == output_paths
    assert player.provider_call_counts_at_play == [2, 2]
    assert [call[1] for call in provider.calls] == ["vi", "vi"]


def test_scripted_speech_rejects_empty_script(tmp_path: Path) -> None:
    runner = ScriptedSpeechRunner(_RecordingTTSProvider(), tmp_path, timeout_seconds=1.0)

    with pytest.raises(ValueError, match="at least one non-empty line"):
        asyncio.run(runner.run(("", "  "), "en", play_audio=False))
