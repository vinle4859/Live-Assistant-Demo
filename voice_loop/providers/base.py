"""Abstract provider contracts used by the pipeline orchestration layer."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ..types import LanguageCode


class SpeechToTextProvider(ABC):
    """Convert an audio file into text."""

    @abstractmethod
    async def transcribe(self, audio_path: Path, language: LanguageCode) -> str:
        """Transcribe the given audio file into text."""

    def supports_live_streaming(self) -> bool:
        """Return whether this provider supports live microphone streaming STT."""

        return False

    def supports_streaming_wake(self) -> bool:
        """Return whether this provider can keep the microphone open for wake detection."""

        return False

    async def transcribe_live_utterance(
        self,
        language: LanguageCode,
        sample_rate: int,
        input_device_index: int | None,
        max_utterance_seconds: float,
        chunk_duration_ms: int,
        speech_start_timeout_seconds: float,
        speech_end_timeout_seconds: float,
        preroll_ms: int = 0,
        ambient_rms: float = 0.0,
        ambient_peak: float = 0.0,
        min_speech_rms: float = 30.0,
        min_speech_peak: float = 250.0,
        local_speech_end_ms: int = 1100,
        max_active_seconds: float = 8.0,
        no_progress_seconds: float = 4.5,
        weak_progress_seconds: float = 6.0,
        weak_progress_min_tokens: int = 3,
        mic_gain: float = 1.0,
    ) -> str:
        """Transcribe one live utterance from microphone audio using provider endpointing."""

        raise NotImplementedError("Live streaming STT is not supported by this provider")

    async def listen_for_wake_phrase(
        self,
        language: LanguageCode,
        sample_rate: int,
        input_device_index: int | None,
        chunk_duration_ms: int,
        wake_phrases: tuple[str, ...],
        max_stream_seconds: float = 45.0,
        mic_gain: float = 1.0,
        should_interrupt: Callable[[], bool] | None = None,
    ) -> dict[str, str]:
        """Continuously listen for a wake phrase and return match diagnostics."""

        raise NotImplementedError("Streaming wake detection is not supported by this provider")


class LanguageModelProvider(ABC):
    """Generate a concise conversational answer from a user question."""

    @abstractmethod
    async def generate_answer(
        self,
        language: LanguageCode,
        question: str,
    ) -> str:
        """Synthesize the final response text from the question."""

    @abstractmethod
    async def generate_answer_stream(
        self,
        language: LanguageCode,
        question: str,
    ):
        """Synthesize the response text as a stream of chunks."""


class TextToSpeechProvider(ABC):
    """Convert response text into a playable audio file."""

    @abstractmethod
    async def synthesize(self, text: str, language: LanguageCode, output_path: Path) -> Path:
        """Render audio for the supplied text and return the created file path."""
