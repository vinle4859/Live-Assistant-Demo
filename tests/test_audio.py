"""Tests for WAV audio helpers."""

from __future__ import annotations

import struct
import wave
from pathlib import Path

import pytest

from voice_loop.audio import _stereo_16bit_pcm_to_mono, read_wav_as_mono_pcm


def _write_wav(path: Path, channels: int, samples: tuple[int, ...], sample_rate: int = 16000) -> None:
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(struct.pack(f"<{len(samples)}h", *samples))


def test_read_wav_as_mono_pcm_keeps_mono_samples(tmp_path: Path) -> None:
    wav_path = tmp_path / "mono.wav"
    _write_wav(wav_path, channels=1, samples=(100, -100))

    pcm_data, sample_rate = read_wav_as_mono_pcm(wav_path)

    assert sample_rate == 16000
    assert struct.unpack("<2h", pcm_data) == (100, -100)


def test_read_wav_as_mono_pcm_averages_stereo_samples(tmp_path: Path) -> None:
    wav_path = tmp_path / "stereo.wav"
    _write_wav(wav_path, channels=2, samples=(1000, -1000, 300, 700))

    pcm_data, _ = read_wav_as_mono_pcm(wav_path)

    assert struct.unpack("<2h", pcm_data) == (0, 500)


def test_stereo_16bit_pcm_to_mono_rejects_incomplete_frame() -> None:
    with pytest.raises(ValueError, match="complete frames"):
        _stereo_16bit_pcm_to_mono(b"\x00\x00")
