"""Google Cloud Speech-to-Text provider implementation."""

from __future__ import annotations

import asyncio
import logging
import math
import os
import queue
import re
import struct
import threading
import time
import unicodedata
from collections import deque
from pathlib import Path

from ..audio import read_wav_as_mono_pcm
from ..types import LanguageCode
from .base import SpeechToTextProvider

LOGGER = logging.getLogger(__name__)


class GoogleSpeechToTextProvider(SpeechToTextProvider):
    """Transcribe WAV files using Google Cloud Speech-to-Text."""

    def __init__(
        self,
        timeout_seconds: float = 3.0,
        model: str | None = None,
        hint_phrases: tuple[str, ...] = (),
        project: str = "",
        location: str = "global",
        language_mode: str = "fixed",
    ) -> None:
        """Store the request timeout used for Google API calls."""

        self.timeout_seconds = timeout_seconds
        self.model = model
        self.hint_phrases = hint_phrases
        self.project = project
        self.location = location
        self.language_mode = language_mode
        self.last_live_capture_stats: dict[str, object] = {}

    def supports_live_streaming(self) -> bool:
        """Return whether this provider can stream live microphone audio."""

        return True

    def supports_streaming_wake(self) -> bool:
        """Return whether this provider can keep a streaming wake recognizer open."""

        return True

    async def listen_for_wake_phrase(
        self,
        language: LanguageCode,
        sample_rate: int,
        input_device_index: int | None,
        chunk_duration_ms: int,
        wake_phrases: tuple[str, ...],
        max_stream_seconds: float = 45.0,
    ) -> dict[str, str]:
        """Listen continuously until a wake phrase appears in interim or final STT."""

        return await asyncio.to_thread(
            self._listen_for_wake_phrase_sync,
            language,
            sample_rate,
            input_device_index,
            chunk_duration_ms,
            wake_phrases,
            max_stream_seconds,
        )

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
    ) -> str:
        """Stream microphone audio to Google STT until endpointing signals utterance end."""

        return await asyncio.to_thread(
            self._transcribe_live_utterance_sync,
            language,
            sample_rate,
            input_device_index,
            max_utterance_seconds,
            chunk_duration_ms,
            speech_start_timeout_seconds,
            speech_end_timeout_seconds,
            preroll_ms,
            ambient_rms,
            ambient_peak,
            min_speech_rms,
            min_speech_peak,
            local_speech_end_ms,
            max_active_seconds,
            no_progress_seconds,
            weak_progress_seconds,
            weak_progress_min_tokens,
        )

    async def transcribe(self, audio_path: Path, language: LanguageCode) -> str:
        """Transcribe a WAV file by delegating to the Google client in a worker thread."""

        return await asyncio.to_thread(self._transcribe_sync, audio_path, language)

    def _transcribe_sync(self, audio_path: Path, language: LanguageCode) -> str:
        """Call the Google client synchronously and return the final transcript."""

        try:
            from google.cloud import speech
        except ImportError as exc:  # pragma: no cover - runtime dependency guard
            raise RuntimeError("google-cloud-speech is not installed") from exc
        try:
            from google.auth.exceptions import DefaultCredentialsError
        except ImportError:
            DefaultCredentialsError = Exception  # type: ignore[assignment]

        pcm_data, sample_rate = read_wav_as_mono_pcm(audio_path)
        try:
            client = speech.SpeechClient()
        except DefaultCredentialsError as exc:
            raise RuntimeError(
                "Google ADC credentials are missing. Configure Application Default Credentials "
                "before running live mode, or switch to demo STT provider."
            ) from exc

        language_codes = self._build_language_codes(language)
        language_code = language_codes[0]
        alternative_language_codes = language_codes[1:]
        audio = speech.RecognitionAudio(content=pcm_data)
        config_kwargs: dict[str, object] = {
            "encoding": speech.RecognitionConfig.AudioEncoding.LINEAR16,
            "sample_rate_hertz": sample_rate,
            "language_code": language_code,
            "enable_automatic_punctuation": True,
        }
        if alternative_language_codes:
            config_kwargs["alternative_language_codes"] = alternative_language_codes
        if self.model:
            config_kwargs["model"] = self.model
        if self.hint_phrases:
            config_kwargs["speech_contexts"] = [
                speech.SpeechContext(phrases=list(self.hint_phrases), boost=15.0)
            ]
        config = speech.RecognitionConfig(
            **config_kwargs,
        )
        response = client.recognize(config=config, audio=audio, timeout=self.timeout_seconds)
        transcripts = [result.alternatives[0].transcript for result in response.results if result.alternatives]
        return " ".join(transcripts).strip()

    def _transcribe_live_utterance_sync(
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
    ) -> str:
        """Use STT v2 streaming + voice activity endpointing to transcribe one utterance."""

        if not self.project:
            raise RuntimeError(
                "GOOGLE_CLOUD_PROJECT is required for live streaming STT endpointing. "
                "Set GOOGLE_CLOUD_PROJECT in .env or disable VOICE_LOOP_ENABLE_STREAMING_STT."
            )

        try:
            import pyaudio
        except ImportError as exc:  # pragma: no cover - runtime dependency guard
            raise RuntimeError("PyAudio is required for live microphone mode") from exc

        try:
            from google.cloud.speech_v2 import SpeechClient
            from google.cloud.speech_v2.types import cloud_speech
            from google.protobuf.duration_pb2 import Duration
            from google.api_core.exceptions import InvalidArgument
        except ImportError as exc:  # pragma: no cover - runtime dependency guard
            raise RuntimeError("google-cloud-speech with speech_v2 support is required for streaming STT") from exc

        try:
            from google.auth.exceptions import DefaultCredentialsError
        except ImportError:
            DefaultCredentialsError = Exception  # type: ignore[assignment]

        try:
            client = SpeechClient()
        except DefaultCredentialsError as exc:
            raise RuntimeError(
                "Google ADC credentials are missing. Configure Application Default Credentials "
                "before running live mode, or switch to demo STT provider."
            ) from exc

        language_codes = self._build_language_codes(language)
        model_name = self.model or "latest_short"
        phrase_entries = [
            cloud_speech.PhraseSet.Phrase(value=phrase, boost=15.0)
            for phrase in self.hint_phrases
            if phrase.strip()
        ]
        LOGGER.info(
            "Request streaming STT config: model=%s location=%s language_codes=%s hint_count=%d",
            model_name,
            self.location,
            ",".join(language_codes),
            len(phrase_entries),
        )
        adaptation = None
        if phrase_entries:
            adaptation = cloud_speech.SpeechAdaptation(
                phrase_sets=[
                    cloud_speech.SpeechAdaptation.AdaptationPhraseSet(
                        inline_phrase_set=cloud_speech.PhraseSet(phrases=phrase_entries)
                    )
                ]
            )

        recognition_config_kwargs: dict[str, object] = {
            "explicit_decoding_config": cloud_speech.ExplicitDecodingConfig(
                encoding=cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=sample_rate,
                audio_channel_count=1,
            ),
            "language_codes": language_codes,
            "model": model_name,
            "features": cloud_speech.RecognitionFeatures(enable_automatic_punctuation=True),
            "adaptation": adaptation,
        }
        recognition_config = cloud_speech.RecognitionConfig(
            **recognition_config_kwargs,
        )
        streaming_features = cloud_speech.StreamingRecognitionFeatures(
            enable_voice_activity_events=True,
            interim_results=True,
            voice_activity_timeout=cloud_speech.StreamingRecognitionFeatures.VoiceActivityTimeout(
                speech_start_timeout=self._duration_proto(Duration, speech_start_timeout_seconds),
                speech_end_timeout=self._duration_proto(Duration, speech_end_timeout_seconds),
            ),
        )
        streaming_config = cloud_speech.StreamingRecognitionConfig(
            config=recognition_config,
            streaming_features=streaming_features,
        )
        recognizer_name = f"projects/{self.project}/locations/{self.location}/recognizers/_"
        initial_request = cloud_speech.StreamingRecognizeRequest(
            recognizer=recognizer_name,
            streaming_config=streaming_config,
        )

        safe_chunk_duration_ms = max(20, int(chunk_duration_ms))
        chunk_frames = max(1, int(sample_rate * safe_chunk_duration_ms / 1000.0))
        safe_max_seconds = max(1.0, float(max_utterance_seconds))
        preroll_chunk_count = max(0, int(max(0, preroll_ms) / safe_chunk_duration_ms))
        local_speech_end_seconds = max(0.3, float(local_speech_end_ms) / 1000.0)
        safe_max_active_seconds = min(safe_max_seconds, max(1.0, float(max_active_seconds)))
        vad_thresholds = self._build_vad_thresholds(
            ambient_rms,
            ambient_peak,
            min_speech_rms,
            min_speech_peak,
        )

        audio = pyaudio.PyAudio()
        stream_kwargs: dict[str, object] = {
            "format": pyaudio.paInt16,
            "channels": 1,
            "rate": sample_rate,
            "input": True,
            "frames_per_buffer": chunk_frames,
        }
        if input_device_index is not None:
            stream_kwargs["input_device_index"] = input_device_index
        stream = audio.open(**stream_kwargs)

        stop_capture = threading.Event()
        audio_queue: queue.Queue[bytes | None] = queue.Queue(maxsize=24)
        started_at = time.monotonic()
        capture_stats: dict[str, float | str] = {
            "speech_start_delay_seconds": -1.0,
            "active_seconds": 0.0,
            "voiced_chunks": 0.0,
            "weak_speech_chunks": 0.0,
            "strong_speech_chunks": 0.0,
            "preroll_chunks": float(preroll_chunk_count),
            "google_endpoint_events": 0.0,
            "final_chars": 0.0,
            "final_tokens": 0.0,
            "interim_chars": 0.0,
            "interim_tokens": 0.0,
            "used_interim": "false",
            "end_reason": "unknown",
        }

        def capture_worker() -> None:
            """Capture microphone chunks and push them into the streaming request queue."""

            speech_started = False
            speech_started_at = 0.0
            last_speech_at = 0.0
            weak_speech_times: deque[float] = deque()
            strong_speech_times: deque[float] = deque()
            preroll_chunks: deque[bytes] = deque(maxlen=preroll_chunk_count)
            try:
                while not stop_capture.is_set():
                    now = time.monotonic()
                    if (now - started_at) >= safe_max_seconds:
                        capture_stats["end_reason"] = "max_utterance"
                        break
                    if not speech_started and (now - started_at) >= max(0.5, speech_start_timeout_seconds):
                        capture_stats["end_reason"] = "speech_start_timeout"
                        break
                    chunk = stream.read(chunk_frames, exception_on_overflow=False)
                    if not chunk:
                        continue
                    chunk_stats = self._pcm_stats(chunk)
                    if speech_started:
                        is_speech = self._is_tail_speech_chunk(
                            chunk_stats,
                            vad_thresholds["tail_rms"],
                            vad_thresholds["tail_peak"],
                        )
                    else:
                        is_speech = self._is_speech_chunk(
                            chunk_stats,
                            vad_thresholds["start_rms"],
                            vad_thresholds["start_peak"],
                        )
                    if not speech_started:
                        if not is_speech:
                            if preroll_chunk_count > 0:
                                preroll_chunks.append(chunk)
                            continue
                        strong_speech = self._is_strong_start_chunk(
                            chunk_stats,
                            vad_thresholds["start_rms"],
                            vad_thresholds["start_peak"],
                        )
                        self._record_debounce_hit(
                            weak_speech_times,
                            strong_speech_times,
                            now,
                            strong_speech,
                            0.5,
                        )
                        capture_stats["weak_speech_chunks"] = float(capture_stats["weak_speech_chunks"]) + 1.0
                        if strong_speech:
                            capture_stats["strong_speech_chunks"] = float(capture_stats["strong_speech_chunks"]) + 1.0
                        if not self._has_confirmed_speech_start(weak_speech_times, strong_speech_times):
                            if preroll_chunk_count > 0:
                                preroll_chunks.append(chunk)
                            continue
                        speech_started = True
                        speech_started_at = now
                        last_speech_at = now
                        capture_stats["speech_start_delay_seconds"] = now - started_at
                        for preroll_chunk in preroll_chunks:
                            if not self._put_audio_chunk(audio_queue, preroll_chunk):
                                break
                    elif is_speech:
                        last_speech_at = now
                    if is_speech:
                        capture_stats["voiced_chunks"] = float(capture_stats["voiced_chunks"]) + 1.0
                    elif self._should_end_for_local_silence(now - last_speech_at, local_speech_end_seconds):
                        capture_stats["end_reason"] = "local_silence"
                        break
                    if speech_started and self._should_end_for_max_active(now - speech_started_at, safe_max_active_seconds):
                        capture_stats["end_reason"] = "max_active"
                        break
                    if not speech_started:
                        continue
                    capture_stats["active_seconds"] = now - speech_started_at
                    if not self._put_audio_chunk(audio_queue, chunk):
                        continue
            finally:
                if capture_stats["end_reason"] == "unknown":
                    capture_stats["end_reason"] = "stopped"
                try:
                    audio_queue.put_nowait(None)
                except queue.Full:
                    pass

        capture_thread = threading.Thread(target=capture_worker, daemon=True)
        capture_thread.start()

        def request_iterator():
            """Yield the initial streaming config then microphone audio chunks."""

            yield initial_request
            while not stop_capture.is_set():
                try:
                    item = audio_queue.get(timeout=0.3)
                except queue.Empty:
                    if not capture_thread.is_alive():
                        break
                    continue
                if item is None:
                    break
                yield cloud_speech.StreamingRecognizeRequest(audio=item)

        final_segments: list[str] = []
        last_interim = ""

        try:
            responses = client.streaming_recognize(requests=request_iterator())
            for response in responses:
                event_type = response.speech_event_type
                if event_type in (
                    cloud_speech.StreamingRecognizeResponse.SpeechEventType.END_OF_SINGLE_UTTERANCE,
                    cloud_speech.StreamingRecognizeResponse.SpeechEventType.SPEECH_ACTIVITY_END,
                ):
                    capture_stats["google_endpoint_events"] = float(capture_stats["google_endpoint_events"]) + 1.0
                    if final_segments:
                        stop_capture.set()

                for result in response.results:
                    if not result.alternatives:
                        continue
                    transcript = result.alternatives[0].transcript.strip()
                    if not transcript:
                        continue
                    if result.is_final:
                        final_segments.append(transcript)
                        capture_stats["final_chars"] = float(sum(len(segment) for segment in final_segments))
                        capture_stats["final_tokens"] = float(
                            sum(len(segment.split()) for segment in final_segments)
                        )
                        stop_capture.set()
                    else:
                        last_interim = transcript
                        capture_stats["interim_chars"] = float(len(last_interim))
                        capture_stats["interim_tokens"] = float(len(last_interim.split()))
        except InvalidArgument:
            raise
        finally:
            stop_capture.set()
            capture_thread.join(timeout=1.0)
            stream.stop_stream()
            stream.close()
            audio.terminate()
            capture_stats["end_reason"] = self._normalize_progress_end_reason(capture_stats)
            LOGGER.info(
                "Streaming VAD telemetry: start_rms=%.1f start_peak=%.1f tail_rms=%.1f tail_peak=%.1f "
                "speech_start_delay=%.2fs active=%.2fs voiced_chunks=%.0f weak_chunks=%.0f "
                "strong_chunks=%.0f preroll_chunks=%.0f google_events=%.0f final_tokens=%.0f "
                "interim_tokens=%.0f used_interim=%s end_reason=%s",
                vad_thresholds["start_rms"],
                vad_thresholds["start_peak"],
                vad_thresholds["tail_rms"],
                vad_thresholds["tail_peak"],
                capture_stats["speech_start_delay_seconds"],
                capture_stats["active_seconds"],
                capture_stats["voiced_chunks"],
                capture_stats["weak_speech_chunks"],
                capture_stats["strong_speech_chunks"],
                capture_stats["preroll_chunks"],
                capture_stats["google_endpoint_events"],
                capture_stats["final_tokens"],
                capture_stats["interim_tokens"],
                capture_stats["used_interim"],
                capture_stats["end_reason"],
            )

        final_text = " ".join(final_segments).strip()
        if final_text:
            self.last_live_capture_stats = dict(capture_stats)
            LOGGER.info(
                "Streaming STT result: final_tokens=%.0f interim_tokens=%.0f used_interim=false end_reason=%s",
                capture_stats["final_tokens"],
                capture_stats["interim_tokens"],
                capture_stats["end_reason"],
            )
            return final_text
        if self._is_substantial_interim(last_interim):
            LOGGER.info('Streaming STT using substantial interim transcript: "%s"', last_interim.strip())
            capture_stats["used_interim"] = "true"
            self.last_live_capture_stats = dict(capture_stats)
            LOGGER.info(
                "Streaming STT result: final_tokens=0 interim_tokens=%.0f used_interim=true end_reason=%s",
                capture_stats["interim_tokens"],
                capture_stats["end_reason"],
            )
            return last_interim.strip()
        self.last_live_capture_stats = dict(capture_stats)
        LOGGER.info(
            "Streaming STT result: final_tokens=0 interim_tokens=%.0f used_interim=false end_reason=%s",
            capture_stats["interim_tokens"],
            capture_stats["end_reason"],
        )
        return ""

    def _listen_for_wake_phrase_sync(
        self,
        language: LanguageCode,
        sample_rate: int,
        input_device_index: int | None,
        chunk_duration_ms: int,
        wake_phrases: tuple[str, ...],
        max_stream_seconds: float = 45.0,
    ) -> dict[str, str]:
        """Use Google v2 streaming recognition for continuous wake detection."""

        if not self.project:
            raise RuntimeError(
                "GOOGLE_CLOUD_PROJECT is required for streaming wake detection. "
                "Set GOOGLE_CLOUD_PROJECT in .env or use fixed-window wake fallback."
            )

        try:
            import pyaudio
        except ImportError as exc:  # pragma: no cover - runtime dependency guard
            raise RuntimeError("PyAudio is required for live microphone mode") from exc

        try:
            from google.cloud.speech_v2 import SpeechClient
            from google.cloud.speech_v2.types import cloud_speech
        except ImportError as exc:  # pragma: no cover - runtime dependency guard
            raise RuntimeError("google-cloud-speech with speech_v2 support is required for streaming wake STT") from exc

        try:
            from google.auth.exceptions import DefaultCredentialsError
        except ImportError:
            DefaultCredentialsError = Exception  # type: ignore[assignment]

        try:
            client = SpeechClient()
        except DefaultCredentialsError as exc:
            raise RuntimeError(
                "Google ADC credentials are missing. Configure Application Default Credentials "
                "before running live mode, or switch to demo STT provider."
            ) from exc

        language_codes = self._build_language_codes(language)
        phrases = tuple(dict.fromkeys([*wake_phrases, *self.hint_phrases]))
        phrase_entries = [
            cloud_speech.PhraseSet.Phrase(value=phrase, boost=20.0)
            for phrase in phrases
            if phrase.strip()
        ]
        adaptation = None
        if phrase_entries:
            adaptation = cloud_speech.SpeechAdaptation(
                phrase_sets=[
                    cloud_speech.SpeechAdaptation.AdaptationPhraseSet(
                        inline_phrase_set=cloud_speech.PhraseSet(phrases=phrase_entries)
                    )
                ]
            )

        recognition_config = cloud_speech.RecognitionConfig(
            explicit_decoding_config=cloud_speech.ExplicitDecodingConfig(
                encoding=cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=sample_rate,
                audio_channel_count=1,
            ),
            language_codes=language_codes,
            model=self.model or "latest_short",
            features=cloud_speech.RecognitionFeatures(enable_automatic_punctuation=True),
            adaptation=adaptation,
        )
        streaming_config = cloud_speech.StreamingRecognitionConfig(
            config=recognition_config,
            streaming_features=cloud_speech.StreamingRecognitionFeatures(
                interim_results=True,
                enable_voice_activity_events=False,
            ),
        )
        recognizer_name = f"projects/{self.project}/locations/{self.location}/recognizers/_"
        initial_request = cloud_speech.StreamingRecognizeRequest(
            recognizer=recognizer_name,
            streaming_config=streaming_config,
        )

        safe_chunk_duration_ms = max(20, int(chunk_duration_ms))
        chunk_frames = max(1, int(sample_rate * safe_chunk_duration_ms / 1000.0))
        safe_max_stream_seconds = max(5.0, float(max_stream_seconds))

        audio = pyaudio.PyAudio()
        stream_kwargs: dict[str, object] = {
            "format": pyaudio.paInt16,
            "channels": 1,
            "rate": sample_rate,
            "input": True,
            "frames_per_buffer": chunk_frames,
        }
        if input_device_index is not None:
            stream_kwargs["input_device_index"] = input_device_index
        stream = audio.open(**stream_kwargs)

        stop_capture = threading.Event()
        audio_queue: queue.Queue[bytes | None] = queue.Queue(maxsize=32)
        started_at = time.monotonic()
        LOGGER.info(
            "Wake streaming started: chunk=%dms max_stream=%.1fs phrases=%s",
            safe_chunk_duration_ms,
            safe_max_stream_seconds,
            ", ".join(wake_phrases),
        )

        def capture_worker() -> None:
            try:
                while not stop_capture.is_set():
                    if time.monotonic() - started_at >= safe_max_stream_seconds:
                        break
                    chunk = stream.read(chunk_frames, exception_on_overflow=False)
                    if chunk:
                        self._put_audio_chunk(audio_queue, chunk)
            finally:
                try:
                    audio_queue.put_nowait(None)
                except queue.Full:
                    pass

        capture_thread = threading.Thread(target=capture_worker, daemon=True)
        capture_thread.start()

        def request_iterator():
            yield initial_request
            while not stop_capture.is_set():
                try:
                    item = audio_queue.get(timeout=0.3)
                except queue.Empty:
                    if not capture_thread.is_alive():
                        break
                    continue
                if item is None:
                    break
                yield cloud_speech.StreamingRecognizeRequest(audio=item)

        last_transcript = ""
        debug_stt_stream = os.getenv("VOICE_LOOP_DEBUG_STT_STREAM", "").strip().lower() in {"1", "true", "yes", "on"}
        try:
            responses = client.streaming_recognize(requests=request_iterator())
            for response in responses:
                for result in response.results:
                    if not result.alternatives:
                        continue
                    transcript = result.alternatives[0].transcript.strip()
                    if not transcript:
                        continue
                    last_transcript = transcript
                    wake_match = self._match_wake_phrase(transcript, wake_phrases)
                    if debug_stt_stream or result.is_final or wake_match is not None:
                        LOGGER.info(
                            'Wake stream transcript: final=%s raw="%s" detected=%s phrase="%s" match_type="%s"',
                            result.is_final,
                            transcript,
                            wake_match is not None,
                            wake_match["phrase"] if wake_match is not None else "",
                            wake_match["match_type"] if wake_match is not None else "",
                        )
                    if wake_match is not None:
                        stop_capture.set()
                        return {
                            "transcript": transcript,
                            "phrase": wake_match["phrase"],
                            "match_type": wake_match["match_type"],
                            "restart_reason": "wake_detected",
                        }
        finally:
            stop_capture.set()
            capture_thread.join(timeout=1.0)
            stream.stop_stream()
            stream.close()
            audio.terminate()

        return {
            "transcript": last_transcript,
            "phrase": "",
            "match_type": "",
            "restart_reason": "stream_limit",
        }

    def _build_language_codes(self, requested_language: LanguageCode) -> list[str]:
        """Build STT language-code preference list for fixed or adaptive mode."""

        primary = "en-US" if requested_language == "en" else "vi-VN"
        if self.language_mode != "adaptive":
            return [primary]
        secondary = "vi-VN" if primary == "en-US" else "en-US"
        return [primary, secondary]

    @staticmethod
    def _build_vad_thresholds(
        ambient_rms: float,
        ambient_peak: float,
        min_speech_rms: float,
        min_speech_peak: float,
    ) -> dict[str, float]:
        """Build capped speech thresholds for noisy rooms without silencing normal speech."""

        start_rms = min(180.0, max(80.0, float(min_speech_rms), float(ambient_rms) * 1.25))
        start_peak = min(1400.0, max(500.0, float(min_speech_peak), float(ambient_peak) * 1.05))
        tail_rms = min(100.0, max(45.0, float(min_speech_rms), start_rms * 0.65))
        tail_peak = min(800.0, max(350.0, float(min_speech_peak), start_peak * 0.60))
        return {
            "start_rms": start_rms,
            "start_peak": start_peak,
            "tail_rms": tail_rms,
            "tail_peak": tail_peak,
        }

    @staticmethod
    def _should_end_for_local_silence(silence_seconds: float, local_speech_end_seconds: float) -> bool:
        """Return whether local silence is long enough to end the utterance."""

        return float(silence_seconds) >= float(local_speech_end_seconds)

    @staticmethod
    def _should_end_for_max_active(active_seconds: float, max_active_seconds: float) -> bool:
        """Return whether active speech capture has reached the hard cap."""

        return float(active_seconds) >= float(max_active_seconds)

    @staticmethod
    def _should_end_for_no_stt_progress(
        active_seconds: float,
        no_progress_seconds: float,
        capture_stats: dict[str, object],
    ) -> bool:
        """Return whether capture should stop because Google has not produced text."""

        if float(active_seconds) < float(no_progress_seconds):
            return False
        return (
            float(capture_stats.get("final_tokens", 0.0)) <= 0.0
            and float(capture_stats.get("interim_tokens", 0.0)) <= 0.0
        )

    @staticmethod
    def _should_end_for_weak_stt_progress(
        active_seconds: float,
        weak_progress_seconds: float,
        weak_progress_min_tokens: int,
        capture_stats: dict[str, object],
    ) -> bool:
        """Return whether capture should stop because STT text remains too weak."""

        if float(active_seconds) < float(weak_progress_seconds):
            return False
        strongest_token_count = max(
            float(capture_stats.get("final_tokens", 0.0)),
            float(capture_stats.get("interim_tokens", 0.0)),
        )
        return strongest_token_count < float(weak_progress_min_tokens)

    @staticmethod
    def _normalize_progress_end_reason(capture_stats: dict[str, object]) -> str:
        """Avoid stale progress-stop reasons after Google returns usable text."""

        end_reason = str(capture_stats.get("end_reason", "unknown"))
        final_tokens = float(capture_stats.get("final_tokens", 0.0))
        interim_tokens = float(capture_stats.get("interim_tokens", 0.0))
        if end_reason in {"no_stt_progress", "weak_stt_progress"} and max(final_tokens, interim_tokens) > 0:
            return "stt_progress_received"
        return end_reason

    @staticmethod
    def _put_audio_chunk(audio_queue: queue.Queue[bytes | None], chunk: bytes) -> bool:
        """Push one chunk into the streaming queue without blocking capture indefinitely."""

        try:
            audio_queue.put(chunk, timeout=0.2)
            return True
        except queue.Full:
            return False

    @staticmethod
    def _is_strong_start_chunk(
        chunk_stats: dict[str, float],
        speech_rms_threshold: float,
        speech_peak_threshold: float,
    ) -> bool:
        """Return whether a chunk is strong enough to accelerate speech-start debounce."""

        return (
            chunk_stats.get("rms", 0.0) >= speech_rms_threshold * 1.5
            or chunk_stats.get("peak", 0.0) >= speech_peak_threshold * 1.5
        )

    @staticmethod
    def _record_debounce_hit(
        weak_speech_times: deque[float],
        strong_speech_times: deque[float],
        now: float,
        strong_speech: bool,
        window_seconds: float,
    ) -> None:
        """Record recent speech-like chunks used to confirm speech start."""

        weak_speech_times.append(now)
        if strong_speech:
            strong_speech_times.append(now)
        while weak_speech_times and now - weak_speech_times[0] > window_seconds:
            weak_speech_times.popleft()
        while strong_speech_times and now - strong_speech_times[0] > window_seconds:
            strong_speech_times.popleft()

    @staticmethod
    def _has_confirmed_speech_start(
        weak_speech_times: deque[float],
        strong_speech_times: deque[float],
    ) -> bool:
        """Return whether debounce has enough recent evidence to start a request."""

        return len(weak_speech_times) >= 3 or len(strong_speech_times) >= 2

    @staticmethod
    def _is_substantial_interim(transcript: str) -> bool:
        """Return whether an interim transcript is useful enough as fallback text."""

        tokens = [token for token in transcript.strip().split() if token]
        return len(tokens) >= 3

    @staticmethod
    def _pcm_stats(chunk: bytes) -> dict[str, float]:
        """Compute simple RMS and peak stats for 16-bit PCM audio chunks."""

        if len(chunk) < 2:
            return {"rms": 0.0, "peak": 0.0}
        sample_count = len(chunk) // 2
        samples = struct.iter_unpack("<h", chunk[: sample_count * 2])
        sum_squares = 0
        peak = 0
        count = 0
        for (sample,) in samples:
            count += 1
            abs_sample = abs(sample)
            peak = max(peak, abs_sample)
            sum_squares += sample * sample
        if count == 0:
            return {"rms": 0.0, "peak": 0.0}
        return {"rms": math.sqrt(sum_squares / count), "peak": float(peak)}

    @staticmethod
    def _is_speech_chunk(
        chunk_stats: dict[str, float],
        speech_rms_threshold: float,
        speech_peak_threshold: float,
    ) -> bool:
        """Return whether a chunk has enough energy to be treated as speech."""

        return (
            chunk_stats.get("rms", 0.0) >= speech_rms_threshold
            or chunk_stats.get("peak", 0.0) >= speech_peak_threshold
        )

    @staticmethod
    def _is_tail_speech_chunk(
        chunk_stats: dict[str, float],
        speech_rms_threshold: float,
        speech_peak_threshold: float,
    ) -> bool:
        """Return whether a post-start chunk is strong enough to extend capture."""

        rms = chunk_stats.get("rms", 0.0)
        peak = chunk_stats.get("peak", 0.0)
        if rms >= speech_rms_threshold:
            return True
        return peak >= speech_peak_threshold and rms >= speech_rms_threshold * 0.55

    @classmethod
    def _match_wake_phrase(cls, transcript: str, wake_phrases: tuple[str, ...]) -> dict[str, str] | None:
        """Return wake match details without importing the live assistant module."""

        normalized_transcript = cls._normalize_text(transcript)
        if not normalized_transcript:
            return None
        transcript_tokens = normalized_transcript.split()
        for phrase in wake_phrases:
            normalized_phrase = cls._normalize_text(phrase)
            if not normalized_phrase:
                continue
            if normalized_phrase in normalized_transcript:
                return {"phrase": phrase, "match_type": "primary" if phrase == wake_phrases[0] else "alias"}
            phrase_tokens = normalized_phrase.split()
            if len(phrase_tokens) > 1:
                phrase_length = len(phrase_tokens)
                windows = (
                    " ".join(transcript_tokens[index : index + phrase_length])
                    for index in range(0, max(0, len(transcript_tokens) - phrase_length + 1))
                )
                if any(cls._similarity(window, normalized_phrase) >= 0.82 for window in windows):
                    return {"phrase": phrase, "match_type": "fuzzy"}
        return None

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize text for provider-local wake matching."""

        lowered = text.lower().replace("đ", "d")
        ascii_like = unicodedata.normalize("NFD", lowered).encode("ascii", "ignore").decode("ascii")
        return " ".join(re.findall(r"[a-z0-9]+", ascii_like))

    @staticmethod
    def _similarity(left: str, right: str) -> float:
        """Return a lightweight similarity ratio for wake phrase variants."""

        from difflib import SequenceMatcher

        return SequenceMatcher(None, left, right).ratio()

    @staticmethod
    def _duration_proto(duration_class, total_seconds: float):
        """Build a protobuf Duration from floating-point seconds."""

        safe_seconds = max(0.0, float(total_seconds))
        whole_seconds = int(safe_seconds)
        nanos = int((safe_seconds - whole_seconds) * 1_000_000_000)
        duration = duration_class()
        duration.seconds = whole_seconds
        duration.nanos = nanos
        return duration
