"""Local demo providers for running the app without cloud credentials."""

from __future__ import annotations

import os
from pathlib import Path

from ..types import LanguageCode
from .base import SpeechToTextProvider, TextToSpeechProvider


class DemoSpeechToTextProvider(SpeechToTextProvider):
    """Return a fixed transcript for local smoke testing."""

    async def transcribe(self, audio_path: Path, language: LanguageCode) -> str:
        """Ignore the audio file and return a configured demo transcript."""

        default_transcript = "how do i reset my password" if language == "en" else "tôi quên mật khẩu"
        return os.getenv("VOICE_LOOP_DEMO_TRANSCRIPT", default_transcript)


class DemoTextToSpeechProvider(TextToSpeechProvider):
    """Write a tiny placeholder audio file for local smoke testing."""

    async def synthesize(self, text: str, language: LanguageCode, output_path: Path) -> Path:
        """Create a non-empty output file and return its path."""

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"demo-audio")
        return output_path
