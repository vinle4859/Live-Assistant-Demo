"""WAV audio helpers used by the pipeline and tests."""

from __future__ import annotations

import audioop
import wave
from pathlib import Path


def read_wav_as_mono_pcm(audio_path: Path) -> tuple[bytes, int]:
    """Load a WAV file and return mono 16-bit PCM plus the sample rate."""

    with wave.open(str(audio_path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()
        if sample_width != 2:
            raise ValueError("Only 16-bit PCM WAV files are supported")
        pcm_data = wav_file.readframes(wav_file.getnframes())

    if channels == 1:
        return pcm_data, sample_rate
    if channels == 2:
        return audioop.tomono(pcm_data, 2, 0.5, 0.5), sample_rate
    raise ValueError("Only mono or stereo WAV files are supported")


def iter_pcm_chunks(pcm_data: bytes, sample_rate: int, chunk_duration_ms: int = 600) -> list[bytes]:
    """Split PCM bytes into chunk-sized blocks for downstream streaming work."""

    bytes_per_second = sample_rate * 2
    chunk_size = max(1, int(bytes_per_second * (chunk_duration_ms / 1000.0)))
    return [pcm_data[index : index + chunk_size] for index in range(0, len(pcm_data), chunk_size)]
