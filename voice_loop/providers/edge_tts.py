"""edge-tts fallback provider implementation."""

from __future__ import annotations

import asyncio
from pathlib import Path

from ..types import LanguageCode
from .base import TextToSpeechProvider


class EdgeTTSProvider(TextToSpeechProvider):
    """Render speech using Microsoft's online edge-tts service."""

    def __init__(self, voice_en: str = "", voice_vi: str = "") -> None:
        """Store the fallback voice names used per language."""

        self.voice_en = voice_en or "en-GB-SoniaNeural"
        self.voice_vi = voice_vi or "vi-VN-HoaiMyNeural"

    async def synthesize(self, text: str, language: LanguageCode, output_path: Path) -> Path:
        """Generate audio using the async edge-tts client and validate the file."""

        try:
            import edge_tts
        except ImportError as exc:  # pragma: no cover - runtime dependency guard
            raise RuntimeError("edge-tts is not installed") from exc

        output_path.parent.mkdir(parents=True, exist_ok=True)
        voice = self.voice_en if language == "en" else self.voice_vi
        communicate = edge_tts.Communicate(text=text, voice=voice)
        await communicate.save(str(output_path))
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError("edge-tts did not produce a valid audio file")
        return output_path
