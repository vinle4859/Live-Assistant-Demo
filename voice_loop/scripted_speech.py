"""Scripted speech mode for event playback."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from .providers.base import TextToSpeechProvider
from .types import LanguageCode

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScriptSpeechLine:
    """One non-empty event script line."""

    index: int
    text: str


class ScriptedSpeechRunner:
    """Render and optionally play fixed event script lines."""

    def __init__(
        self,
        tts_provider: TextToSpeechProvider,
        output_dir: Path,
        timeout_seconds: float,
        player=None,
    ) -> None:
        self.tts_provider = tts_provider
        self.output_dir = output_dir
        self.timeout_seconds = timeout_seconds
        self.player = player

    async def run(
        self,
        lines: tuple[str, ...],
        language: LanguageCode,
        play_audio: bool = True,
        start_at: int = 1,
        delay_seconds: float = 0.0,
        manual_next: bool = False,
    ) -> tuple[Path, ...]:
        """Render script lines in order and optionally play each output."""

        if delay_seconds < 0:
            raise ValueError("Script delay must be 0 or greater.")
        script_lines = parse_script_lines(lines, start_at=start_at)
        if not script_lines:
            raise ValueError("Script mode requires at least one non-empty line.")

        session_dir = self.output_dir / "script_sessions" / time.strftime("%Y%m%d_%H%M%S")
        session_dir.mkdir(parents=True, exist_ok=True)
        output_paths: list[Path] = []
        for line in script_lines:
            output_path = session_dir / f"line_{line.index:03d}_{uuid4().hex}.mp3"
            await asyncio.wait_for(
                self.tts_provider.synthesize(line.text, language, output_path),
                timeout=self.timeout_seconds,
            )
            output_paths.append(output_path)
            LOGGER.info("Script line rendered: index=%d language=%s output=%s", line.index, language, output_path)
        if play_audio:
            if self.player is None:
                raise RuntimeError("Script playback requires an audio player.")
            for output_path in output_paths:
                if manual_next:
                    await asyncio.to_thread(input, f"Press Enter to play {output_path.name}...")
                await asyncio.to_thread(self.player.play, output_path)
                if delay_seconds > 0:
                    await asyncio.sleep(delay_seconds)
        return tuple(output_paths)


def parse_script_lines(lines: tuple[str, ...], start_at: int = 1) -> tuple[ScriptSpeechLine, ...]:
    """Return non-empty script lines with stable one-based positions."""

    if start_at < 1:
        raise ValueError("Script start line must be 1 or greater.")

    parsed: list[ScriptSpeechLine] = []
    for index, line in enumerate(lines, start=1):
        if index < start_at:
            continue
        stripped = line.strip()
        if stripped:
            parsed.append(ScriptSpeechLine(index=index, text=stripped))
    return tuple(parsed)
