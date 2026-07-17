"""Google Cloud Text-to-Speech provider implementation."""

from __future__ import annotations

import asyncio
from pathlib import Path

from ..lang_detect import detect_language
from ..types import LanguageCode
from .base import TextToSpeechProvider


class GoogleTextToSpeechProvider(TextToSpeechProvider):
    """Render speech audio using Google Cloud Text-to-Speech."""

    def __init__(
        self,
        timeout_seconds: float = 3.0,
        voice_en: str = "",
        voice_vi: str = "",
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.voice_en = voice_en or "en-GB-Neural2-F"
        self.voice_vi = voice_vi or "vi-VN-Neural2-A"

    async def synthesize(self, text: str, language: LanguageCode, output_path: Path) -> Path:
        """Generate audio in a worker thread and validate the resulting file."""

        # Auto-detect overrides the caller — text content is the ground truth for voice selection.
        resolved_language: LanguageCode = detect_language(text)  # type: ignore[assignment]
        return await asyncio.to_thread(self._synthesize_sync, text, resolved_language, output_path)

    def _synthesize_sync(self, text: str, language: LanguageCode, output_path: Path) -> Path:
        """Call the Google TTS client and write an MP3 file to disk."""

        try:
            from google.cloud import texttospeech
        except ImportError as exc:  # pragma: no cover - runtime dependency guard
            raise RuntimeError("google-cloud-texttospeech is not installed") from exc
        try:
            from google.auth.exceptions import DefaultCredentialsError
        except ImportError:
            DefaultCredentialsError = Exception  # type: ignore[assignment]

        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            client = texttospeech.TextToSpeechClient()
        except DefaultCredentialsError as exc:
            raise RuntimeError(
                "Google ADC credentials are missing. Configure Application Default Credentials "
                "before running speech synthesis, or switch to edge/demo TTS provider."
            ) from exc
        voice_name = self.voice_en if language == "en" else self.voice_vi
        voice_language = _voice_language_code(voice_name, "en-GB" if language == "en" else "vi-VN")
        request = texttospeech.SynthesizeSpeechRequest(
            input=texttospeech.SynthesisInput(text=text),
            voice=texttospeech.VoiceSelectionParams(
                language_code=voice_language,
                name=voice_name,
                ssml_gender=texttospeech.SsmlVoiceGender.FEMALE,
            ),
            audio_config=texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3),
        )
        response = client.synthesize_speech(request=request, timeout=self.timeout_seconds)
        output_path.write_bytes(response.audio_content)
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError("Google TTS did not produce a valid audio file")
        return output_path


def _voice_language_code(voice_name: str, default: str) -> str:
    """Return the Google language code prefix for a concrete voice name."""

    parts = [part for part in voice_name.split("-") if part]
    if len(parts) >= 2:
        return "-".join(parts[:2])
    return default
