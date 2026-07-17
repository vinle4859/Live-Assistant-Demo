"""WAV audio helpers used by the pipeline and tests."""

from __future__ import annotations

import struct
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
        return _stereo_16bit_pcm_to_mono(pcm_data), sample_rate
    raise ValueError("Only mono or stereo WAV files are supported")


def _stereo_16bit_pcm_to_mono(pcm_data: bytes) -> bytes:
    """Average little-endian stereo 16-bit PCM frames into mono frames."""

    if len(pcm_data) % 4 != 0:
        raise ValueError("Stereo 16-bit PCM data must contain complete frames")
    mono_samples = bytearray(len(pcm_data) // 2)
    output_offset = 0
    for left, right in struct.iter_unpack("<hh", pcm_data):
        mono_sample = int((left + right) / 2)
        struct.pack_into("<h", mono_samples, output_offset, mono_sample)
        output_offset += 2
    return bytes(mono_samples)


def iter_pcm_chunks(pcm_data: bytes, sample_rate: int, chunk_duration_ms: int = 600) -> list[bytes]:
    """Split PCM bytes into chunk-sized blocks for downstream streaming work."""

    bytes_per_second = sample_rate * 2
    chunk_size = max(1, int(bytes_per_second * (chunk_duration_ms / 1000.0)))
    return [pcm_data[index : index + chunk_size] for index in range(0, len(pcm_data), chunk_size)]


def scale_pcm16_volume(pcm_data: bytes, gain: float) -> bytes:
    """Scale 16-bit signed PCM samples by a gain factor, clamping to signed 16-bit limits."""

    if gain == 1.0 or not pcm_data:
        return pcm_data
    import array
    samples = array.array("h")
    samples.frombytes(pcm_data)
    for i in range(len(samples)):
        val = int(samples[i] * gain)
        samples[i] = max(-32768, min(32767, val))
    return samples.tobytes()
