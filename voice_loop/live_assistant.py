"""Live wake-word assistant loop built on top of the voice pipeline."""

from __future__ import annotations

import asyncio
import logging
import math
import os
import re
import struct
import subprocess
import sys
import time
import unicodedata
import wave
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from uuid import uuid4

from .pipeline import VoicePipeline
from .types import LanguageCode

_WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_VI_DIACRITIC_RE = re.compile(r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]", re.IGNORECASE)
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class WakePhraseMatch:
    """Wake phrase match details for diagnostics."""

    phrase: str
    match_type: str


@dataclass(frozen=True)
class LiveAssistantConfig:
    """Runtime configuration for the microphone wake-word assistant."""

    language: LanguageCode
    language_mode: str = "adaptive"
    output_language_mode: str = "auto"
    output_language_fixed: LanguageCode = "en"
    enable_bilingual_output: bool = False
    language_switch_min_confidence: float = 0.75
    language_switch_sticky_turns: int = 2
    language_override_commands: bool = True
    wake_word: str = "hey lemon"
    wake_aliases: tuple[str, ...] = ()
    wake_ack_mode: str = "adaptive"
    wake_ack_beep_frequency: int = 880
    wake_ack_beep_duration_ms: int = 120
    wake_ack_prompt_text_en: str = "How can I help you?"
    wake_ack_prompt_text_vi: str = "Tôi có thể giúp gì cho bạn?"
    wake_ack_adaptive_speak_on_wake: bool = True
    request_ready_cue_mode: str = "speech"
    request_ready_first_text_en: str = "Go ahead."
    request_ready_first_text_vi: str = "Bạn cứ hỏi nhé."
    request_ready_texts_en: tuple[str, ...] = ("What else would you like to know?", "I'm listening.", "Go ahead.")
    request_ready_texts_vi: tuple[str, ...] = ("Bạn hỏi tiếp nhé.", "Mình nghe đây.", "Bạn cần hỏi thêm gì?")
    request_ready_cache: bool = True
    thinking_cue_enabled: bool = True
    thinking_cue_delay_seconds: float = 1.2
    thinking_texts_en: tuple[str, ...] = ("Let me check.", "One moment.")
    thinking_texts_vi: tuple[str, ...] = ("Để tôi kiểm tra.", "Chờ tôi một chút.")
    wake_window_seconds: float = 2.5
    utterance_seconds: float = 8.0
    utterance_min_rms: float = 30.0
    utterance_min_peak: float = 250.0
    request_post_tts_guard_seconds: float = 0.15
    minimum_transcript_characters: int = 4
    enable_streaming_stt: bool = True
    streaming_chunk_duration_ms: int = 100
    streaming_speech_start_timeout_seconds: float = 8.0
    streaming_speech_end_timeout_seconds: float = 1.8
    streaming_local_speech_end_ms: int = 2000
    streaming_max_active_seconds: float = 10.0
    streaming_no_progress_seconds: float = 4.5
    streaming_weak_progress_seconds: float = 6.0
    streaming_weak_progress_min_tokens: int = 3
    preroll_enabled: bool = True
    preroll_ms: int = 750
    startup_calibration_enabled: bool = True
    startup_calibration_seconds: float = 2.0
    request_max_ignored_turns: int = 4
    request_max_turns: int = 0
    request_idle_timeout_seconds: float = 90.0
    request_max_session_seconds: float = 180.0
    barge_in_mode: str = "off"
    barge_in_listen_seconds: float = 0.7
    barge_in_grace_seconds: float = 1.2
    barge_in_min_rms: float = 45.0
    barge_in_min_peak: float = 400.0
    sample_rate: int = 16000
    input_device_index: int | None = None
    temp_audio_dir: Path = Path("data/live_audio")
    debug_audio_io: bool = False
    debug_stt_stream: bool = False


def normalize_text(text: str) -> str:
    """Normalize text to lowercase alphanumeric tokens separated by spaces."""

    # Normalize Unicode first so Vietnamese diacritics map consistently for keyword checks.
    lowered = text.lower().replace("đ", "d")
    ascii_like = (
        unicodedata.normalize("NFD", lowered)
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    return " ".join(_WORD_RE.findall(ascii_like))


def contains_wake_word(transcript: str, wake_word: str) -> bool:
    """Return whether the transcript contains the normalized wake phrase."""

    return describe_wake_phrase_match(transcript, wake_word) is not None


def describe_wake_phrase_match(transcript: str, wake_word: str) -> WakePhraseMatch | None:
    """Return wake phrase match diagnostics when a transcript addresses the assistant."""

    normalized_transcript = normalize_text(transcript)
    normalized_wake = normalize_text(wake_word)
    if not normalized_wake:
        return None
    if normalized_wake in normalized_transcript:
        return WakePhraseMatch(phrase=wake_word, match_type="exact")

    wake_tokens = normalized_wake.split()
    transcript_tokens = normalized_transcript.split()
    if not wake_tokens:
        return None

    if len(transcript_tokens) < len(wake_tokens):
        if (
            len(wake_tokens) >= 2
            and len(transcript_tokens) <= len(wake_tokens) + 1
            and wake_tokens[-1] in transcript_tokens
        ):
            return WakePhraseMatch(phrase=wake_word, match_type="keyword_only")
        return None

    # Use fuzzy n-gram matching so minor STT spelling drift still triggers wake-up.
    window_size = len(wake_tokens)
    for start in range(0, len(transcript_tokens) - window_size + 1):
        candidate = " ".join(transcript_tokens[start : start + window_size])
        similarity = SequenceMatcher(None, candidate, normalized_wake).ratio()
        if similarity >= 0.82:
            return WakePhraseMatch(phrase=wake_word, match_type="fuzzy")

    # Allow short transcripts that include only the assistant keyword (e.g., "lemon").
    if (
        len(wake_tokens) >= 2
        and len(transcript_tokens) <= len(wake_tokens) + 1
        and wake_tokens[-1] in transcript_tokens
    ):
        return WakePhraseMatch(phrase=wake_word, match_type="keyword_only")
    return None


def detect_matched_wake_phrase(transcript: str, wake_phrases: tuple[str, ...]) -> str | None:
    """Return the wake phrase variant that matched the transcript, if any."""

    match = describe_matched_wake_phrase(transcript, wake_phrases)
    return match.phrase if match is not None else None


def describe_matched_wake_phrase(transcript: str, wake_phrases: tuple[str, ...]) -> WakePhraseMatch | None:
    """Return wake phrase diagnostics for the first configured phrase match."""

    normalized_transcript = normalize_text(transcript)
    for wake_phrase in wake_phrases:
        normalized_wake = normalize_text(wake_phrase)
        if normalized_wake and normalized_wake in normalized_transcript:
            return WakePhraseMatch(phrase=wake_phrase, match_type="exact")
    for wake_phrase in wake_phrases:
        match = describe_wake_phrase_match(transcript, wake_phrase)
        if match is not None:
            return match
    return None


def is_sleep_command(transcript: str, language: LanguageCode) -> bool:
    """Return whether the user is asking the assistant to stop listening."""

    normalized_transcript = normalize_text(transcript)
    tokens = normalized_transcript.split()
    if language == "vi":
        return any(
            phrase in normalized_transcript
            for phrase in (
                "dung nghe",
                "tam dung",
                "dung lai",
                "ngu di",
                "ket thuc",
                "thoat",
                "tam biet",
                "chao tam biet",
            )
        )
    if "listening" in tokens and tokens:
        first_token = tokens[0]
        if first_token in {"stop", "top", "stopped", "stopping"}:
            return True
    return any(
        phrase in normalized_transcript
        for phrase in (
            "stop listening",
            "go to sleep",
            "sleep now",
            "exit",
            "quit",
        )
    )


def is_command_like_sleep_phrase(transcript: str) -> bool:
    """Return whether a transcript is close enough to an exit command to avoid LLM routing."""

    normalized_transcript = normalize_text(transcript)
    tokens = normalized_transcript.split()
    if not tokens:
        return False
    if len(tokens) == 1 and tokens[0] in {"exit", "quit", "goodbye", "bye", "thoat"}:
        return True
    if "listening" in tokens and tokens[0] in {"stop", "top", "stopped", "stopping"}:
        return True
    command_phrases = (
        "stop listening",
        "go to sleep",
        "sleep now",
        "goodbye",
        "bye",
        "exit",
        "quit",
        "dung nghe",
        "tam dung",
        "dung lai",
        "ngu di",
        "ket thuc",
        "thoat",
        "tam biet",
        "chao tam biet",
    )
    return any(SequenceMatcher(None, normalized_transcript, phrase).ratio() >= 0.84 for phrase in command_phrases)


def detect_input_language(transcript: str, fallback_language: LanguageCode) -> tuple[LanguageCode, float, str]:
    """Detect whether transcript is Vietnamese or English with a simple confidence score."""

    if not transcript.strip():
        return fallback_language, 0.0, "empty_transcript"

    normalized = normalize_text(transcript)
    token_list = normalized.split()
    token_set = set(token_list)
    if not token_set:
        return fallback_language, 0.0, "no_tokens"

    if _has_english_lead_marker(normalized):
        return "en", 0.95, "en_lead_marker"

    vi_markers = {
        "anh",
        "bao",
        "biet",
        "co",
        "cong",
        "duong",
        "toi",
        "ban",
        "khong",
        "duoc",
        "la",
        "gi",
        "hoc",
        "phi",
        "truong",
        "sinh",
        "vien",
        "xin",
        "chao",
        "cam",
        "on",
        "cho",
        "thoi",
        "tiet",
        "hom",
        "nay",
        "ha",
        "noi",
        "ai",
        "thang",
        "tran",
        "bong",
        "da",
        "gan",
        "nhat",
        "cua",
        "meo",
        "trung",
        "phut",
        "ca",
        "phe",
        "viet",
        "mot",
        "doan",
        "tho",
        "ngan",
        "hoi",
        "noi",
        "lai",
        "giup",
        "duong",
        "ket",
        "xe",
    }
    en_markers = {
        "a",
        "about",
        "at",
        "bad",
        "if",
        "in",
        "me",
        "near",
        "on",
        "street",
        "tell",
        "traffic",
        "the",
        "is",
        "are",
        "what",
        "how",
        "where",
        "when",
        "why",
        "please",
        "help",
        "my",
        "you",
        "can",
        "do",
        "cost",
        "tuition",
        "fee",
        "price",
        "admission",
        "alternative",
        "available",
        "abroad",
        "duration",
        "english",
        "exchange",
        "graduation",
        "ielts",
        "mandatory",
        "majors",
        "offer",
        "opportunities",
        "program",
        "programs",
        "proficiency",
        "required",
        "requirements",
        "students",
        "studying",
        "there",
        "level",
        "ai",
        "breakfast",
        "bullets",
        "entanglement",
        "explain",
        "healthy",
        "ideas",
        "latest",
        "news",
        "quantum",
        "simple",
        "summarize",
        "terms",
        "weather",
        "week",
    }
    neutral_vi_place_tokens = {"cong", "hoa", "ha", "noi", "viet", "nam"}
    en_count = sum(1 for token in token_list if token in en_markers)
    vi_count = sum(
        1
        for token in token_list
        if token in vi_markers and not (_has_english_structure(token_set) and token in neutral_vi_place_tokens)
    )
    token_count = len(token_list)
    en_ratio = en_count / token_count
    vi_ratio = vi_count / token_count
    has_vi_diacritic = bool(_VI_DIACRITIC_RE.search(transcript))

    LOGGER.info(
        "Language evidence: en_tokens=%d vi_tokens=%d token_count=%d en_ratio=%.2f vi_ratio=%.2f vi_diacritic=%s",
        en_count,
        vi_count,
        token_count,
        en_ratio,
        vi_ratio,
        has_vi_diacritic,
    )

    if _has_english_structure(token_set) and en_count >= max(2, vi_count + 1):
        confidence = min(0.95, 0.70 + max(0, en_count - vi_count) * 0.08)
        return "en", confidence, "en_weighted_majority"

    if has_vi_diacritic and vi_count >= max(2, en_count):
        return "vi", 0.95, "vi_diacritic_weighted"

    if vi_count == 0 and en_count == 0:
        if len(token_set) >= 4 and _looks_like_latin_english(normalized):
            return "en", 0.62, "latin_multiword_default_en"
        return fallback_language, 0.45, "no_marker_hit"
    if vi_count > en_count:
        confidence = min(0.95, 0.55 + (vi_count - en_count) * 0.15)
        return "vi", confidence, "vi_marker_majority"
    if en_count > vi_count:
        confidence = min(0.95, 0.55 + (en_count - vi_count) * 0.15)
        return "en", confidence, "en_marker_majority"

    strong_en_markers = {
        "what",
        "how",
        "where",
        "tell",
        "latest",
        "explain",
        "summarize",
    }
    if token_set & strong_en_markers:
        return "en", 0.65, "en_marker_tie_break"

    if has_vi_diacritic:
        return "vi", 0.70, "vi_diacritic_tie_break"

    return fallback_language, 0.5, "marker_tie"


def _has_english_lead_marker(normalized_text: str) -> bool:
    """Return whether the utterance starts with an English question/request marker."""

    tokens = normalized_text.split()
    return bool(tokens) and tokens[0] in {"what", "how", "where", "latest", "explain", "summarize", "tell"}


def _has_english_structure(tokens: set[str]) -> bool:
    """Return whether tokens contain an English request/question frame."""

    return bool(
        tokens & {"what", "how", "where", "when", "why", "tell", "explain", "summarize", "please"}
        or {"can", "you"} <= tokens
        or {"do", "you"} <= tokens
    )


def _looks_like_latin_english(normalized_text: str) -> bool:
    """Return whether marker-free Latin text is more likely English than Vietnamese."""

    tokens = normalized_text.split()
    if len(tokens) < 4:
        return False
    common_english_endings = ("ing", "tion", "ment", "ly", "ed")
    return any(token.endswith(common_english_endings) for token in tokens)


def parse_language_override_command(transcript: str) -> str | None:
    """Parse runtime language-policy commands from normalized transcript text."""

    normalized = normalize_text(transcript)
    if not normalized:
        return None

    if any(
        phrase in normalized
        for phrase in (
            "english only",
            "speak english",
            "use english",
            "reply in english",
            "chi tieng anh",
        )
    ):
        return "force_en"
    if any(
        phrase in normalized
        for phrase in (
            "vietnamese only",
            "speak vietnamese",
            "use vietnamese",
            "reply in vietnamese",
            "chi tieng viet",
        )
    ):
        return "force_vi"
    if any(
        phrase in normalized
        for phrase in (
            "auto language",
            "automatic language",
            "language auto",
            "tu dong ngon ngu",
            "che do tu dong",
        )
    ):
        return "auto"
    if any(
        phrase in normalized
        for phrase in (
            "both languages",
            "bilingual mode",
            "song ngu",
        )
    ):
        return "bilingual"
    return None


class MicrophoneRecorder:
    """Record fixed-length microphone audio into temporary WAV files."""

    def __init__(self, temp_audio_dir: Path, input_device_index: int | None = None) -> None:
        """Store and prepare the directory used for temporary microphone captures."""

        self.temp_audio_dir = temp_audio_dir
        self.input_device_index = input_device_index
        self.temp_audio_dir.mkdir(parents=True, exist_ok=True)

    def capture_wav(self, duration_seconds: float, sample_rate: int, prefix: str) -> Path:
        """Capture mono microphone audio and save it as 16-bit PCM WAV."""

        try:
            import pyaudio
        except ImportError as exc:  # pragma: no cover - runtime dependency guard
            raise RuntimeError("PyAudio is required for live microphone mode") from exc

        chunk_size = 1024
        total_chunks = max(1, int((sample_rate * duration_seconds) / chunk_size))
        frames: list[bytes] = []
        audio = pyaudio.PyAudio()
        stream_kwargs: dict[str, object] = {
            "format": pyaudio.paInt16,
            "channels": 1,
            "rate": sample_rate,
            "input": True,
            "frames_per_buffer": chunk_size,
        }
        if self.input_device_index is not None:
            stream_kwargs["input_device_index"] = self.input_device_index
        stream = audio.open(**stream_kwargs)
        try:
            for _ in range(total_chunks):
                frames.append(stream.read(chunk_size, exception_on_overflow=False))
        finally:
            stream.stop_stream()
            stream.close()

        output_path = self.temp_audio_dir / f"{prefix}_{uuid4().hex}.wav"
        with wave.open(str(output_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(b"".join(frames))
        audio.terminate()
        return output_path

    def describe_input_device(self, sample_rate: int) -> dict[str, object]:
        """Return selected input-device metadata for startup diagnostics."""

        try:
            import pyaudio
        except ImportError as exc:  # pragma: no cover - runtime dependency guard
            raise RuntimeError("PyAudio is required for live microphone mode") from exc

        audio = pyaudio.PyAudio()
        try:
            if self.input_device_index is None:
                device_info = audio.get_default_input_device_info()
            else:
                device_info = audio.get_device_info_by_index(self.input_device_index)
            return {
                "index": int(device_info["index"]),
                "name": str(device_info["name"]),
                "default_sample_rate": int(device_info.get("defaultSampleRate", 0)),
                "max_input_channels": int(device_info.get("maxInputChannels", 0)),
                "requested_sample_rate": int(sample_rate),
            }
        except (OSError, KeyError, TypeError, ValueError) as exc:
            available_devices: list[str] = []
            for device_index in range(audio.get_device_count()):
                device = audio.get_device_info_by_index(device_index)
                if int(device.get("maxInputChannels", 0)) > 0:
                    available_devices.append(f'[{device_index}] {device.get("name", "unknown")}')
            raise RuntimeError(
                "Unable to access the configured microphone input device. "
                f"Configured index: {self.input_device_index!r}. "
                f"Available input devices: {', '.join(available_devices) or 'none'}."
            ) from exc
        finally:
            audio.terminate()

    @staticmethod
    def analyze_wav(audio_path: Path) -> dict[str, float]:
        """Compute simple diagnostics from a recorded WAV file for debugging."""

        with wave.open(str(audio_path), "rb") as wav_file:
            frame_rate = wav_file.getframerate()
            sample_width = wav_file.getsampwidth()
            frame_count = wav_file.getnframes()
            raw_frames = wav_file.readframes(frame_count)
        if not raw_frames:
            return {"duration_seconds": 0.0, "rms": 0.0, "peak": 0.0}
        duration_seconds = frame_count / float(frame_rate) if frame_rate > 0 else 0.0

        # Live captures are 16-bit PCM; if format differs unexpectedly, return duration only.
        if sample_width != 2 or len(raw_frames) < 2:
            return {"duration_seconds": duration_seconds, "rms": 0.0, "peak": 0.0}

        samples = [sample[0] for sample in struct.iter_unpack("<h", raw_frames)]
        if not samples:
            return {"duration_seconds": duration_seconds, "rms": 0.0, "peak": 0.0}
        sum_squares = sum(sample * sample for sample in samples)
        rms = math.sqrt(sum_squares / len(samples))
        peak = max(abs(sample) for sample in samples)
        clipping_count = sum(1 for sample in samples if abs(sample) >= 32700)
        clipping_ratio = clipping_count / len(samples)
        return {
            "duration_seconds": duration_seconds,
            "rms": rms,
            "peak": peak,
            "clipping_ratio": clipping_ratio,
        }


class AudioPlayer:
    """Play synthesized assistant audio files in a blocking manner."""

    def play(self, audio_path: Path) -> None:
        """Play the given audio file using the default system backend."""

        try:
            from playsound import playsound
        except ImportError as exc:  # pragma: no cover - runtime dependency guard
            raise RuntimeError("playsound is required for live audio playback") from exc
        file_size = audio_path.stat().st_size if audio_path.exists() else 0
        debug_audio = os.getenv("VOICE_LOOP_DEBUG_AUDIO_IO", "").strip().lower() in {"1", "true", "yes", "on"}
        if debug_audio:
            LOGGER.info("Playback diagnostics: started file=%s bytes=%d", audio_path.name, file_size)
        started_at = time.perf_counter()
        playsound(str(audio_path), block=True)
        if debug_audio:
            LOGGER.info(
                "Playback diagnostics: finished file=%s elapsed_ms=%.0f",
                audio_path.name,
                (time.perf_counter() - started_at) * 1000.0,
            )

    def play_interruptible(
        self,
        audio_path: Path,
        should_interrupt,
        grace_seconds: float,
        poll_seconds: float = 0.2,
    ) -> bool:
        """Play audio in a subprocess so confirmed barge-in can stop playback."""

        command = [
            sys.executable,
            "-c",
            "from playsound import playsound; import sys; playsound(sys.argv[1], block=True)",
            str(audio_path),
        ]
        process = subprocess.Popen(command)
        started_at = time.monotonic()
        file_size = audio_path.stat().st_size if audio_path.exists() else 0
        LOGGER.info("Playback diagnostics: interruptible_started file=%s bytes=%d", audio_path.name, file_size)
        try:
            while process.poll() is None:
                if time.monotonic() - started_at >= grace_seconds and should_interrupt():
                    process.terminate()
                    try:
                        process.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=1.0)
                    LOGGER.info(
                        "Playback diagnostics: interrupted file=%s elapsed_ms=%.0f",
                        audio_path.name,
                        (time.monotonic() - started_at) * 1000.0,
                    )
                    return True
                time.sleep(poll_seconds)
            if process.returncode != 0:
                raise RuntimeError(f"audio playback subprocess exited with code {process.returncode}")
            LOGGER.info(
                "Playback diagnostics: interruptible_finished file=%s elapsed_ms=%.0f",
                audio_path.name,
                (time.monotonic() - started_at) * 1000.0,
            )
            return False
        except Exception:
            if process.poll() is None:
                process.terminate()
            raise


class LiveVoiceAssistant:
    """Run a wake-word gated voice interaction loop on top of VoicePipeline."""

    def __init__(
        self,
        pipeline: VoicePipeline,
        config: LiveAssistantConfig,
        recorder: MicrophoneRecorder | None = None,
        player: AudioPlayer | None = None,
    ) -> None:
        """Initialize assistant dependencies and runtime settings."""

        self.pipeline = pipeline
        self.config = config
        self.recorder = recorder or MicrophoneRecorder(config.temp_audio_dir, config.input_device_index)
        self.player = player or AudioPlayer()
        self._cached_wake_prompt_path: Path | None = None
        self._cue_cache_paths: dict[tuple[str, LanguageCode, int], Path] = {}
        self._request_ready_cue_indexes: dict[LanguageCode, int] = {"en": 0, "vi": 0}
        self._thinking_cue_indexes: dict[LanguageCode, int] = {"en": 0, "vi": 0}
        self._last_request_capture_meta: dict[str, object] = {}
        self._startup_cache_tasks: set[asyncio.Task[None]] = set()
        self._output_language_mode = config.output_language_mode
        self._output_language_fixed: LanguageCode = config.output_language_fixed
        self._bilingual_output_enabled = config.enable_bilingual_output
        self._current_output_language: LanguageCode = (
            config.output_language_fixed if config.output_language_mode == "fixed" else config.language
        )
        self._language_sticky_remaining = 0
        self._ambient_rms = 0.0
        self._ambient_peak = 0.0
        if hasattr(self.pipeline, "output_dir"):
            self.pipeline.output_dir = self.config.temp_audio_dir

    async def run(self) -> None:
        """Run the assistant forever: wake listening followed by conversation turns."""

        wake_phrases = self._wake_phrases()
        device_info = await asyncio.to_thread(self.recorder.describe_input_device, self.config.sample_rate)
        LOGGER.info(
            'Microphone device: index=%d name="%s" max_channels=%d sample_rate=%d (requested=%d)',
            device_info["index"],
            device_info["name"],
            device_info["max_input_channels"],
            device_info["default_sample_rate"],
            device_info["requested_sample_rate"],
        )
        if int(device_info["default_sample_rate"]) != int(device_info["requested_sample_rate"]):
            LOGGER.warning(
                "Requested sample rate %d differs from device default %d; if wake misses continue, "
                "try setting VOICE_LOOP_SAMPLE_RATE to the device default in .env.",
                device_info["requested_sample_rate"],
                device_info["default_sample_rate"],
            )
        if len(wake_phrases) > 1:
            LOGGER.info('Wake phrases enabled: "%s"', '", "'.join(wake_phrases))
        self._log_command_card(device_info, wake_phrases)
        self._cleanup_stale_runtime_audio()
        if self.config.startup_calibration_enabled:
            await self._run_startup_calibration()
        if self._supports_streaming_stt():
            effective_speech_end_timeout = self._effective_streaming_speech_end_timeout_seconds()
            if effective_speech_end_timeout > self.config.streaming_speech_end_timeout_seconds:
                LOGGER.warning(
                    "Configured streaming speech-end timeout %.2fs is too low for natural pauses; "
                    "using %.2fs for this run.",
                    self.config.streaming_speech_end_timeout_seconds,
                    effective_speech_end_timeout,
                )
            LOGGER.info(
                "Live streaming STT enabled (chunk=%dms, speech_start_timeout=%.2fs, "
                "speech_end_timeout=%.2fs, no_progress=%.2fs, weak_progress=%.2fs/%d tokens).",
                self.config.streaming_chunk_duration_ms,
                self.config.streaming_speech_start_timeout_seconds,
                effective_speech_end_timeout,
                self.config.streaming_no_progress_seconds,
                self.config.streaming_weak_progress_seconds,
                self.config.streaming_weak_progress_min_tokens,
            )
        else:
            LOGGER.info("Live streaming STT disabled; using fixed %.1fs capture windows.", self.config.utterance_seconds)
        LOGGER.info(
            "Conversation limits: max_turns=%d, max_ignored_turns=%d, max_session_seconds=%.1f",
            self.config.request_max_turns,
            self.config.request_max_ignored_turns,
            self.config.request_max_session_seconds,
        )
        self._start_startup_cache_priming()

        while True:
            LOGGER.info('Waiting for wake word "%s"...', self.config.wake_word)
            await self._wait_for_wake_word(wake_phrases)
            await self._acknowledge_wake()
            LOGGER.info("Wake word detected. Start speaking your request.")
            await self._conversation_loop()
            LOGGER.info("Returning to wake-word listening mode.")

    async def _wait_for_wake_word(self, wake_phrases: tuple[str, ...]) -> None:
        """Listen continuously for wake phrases, falling back to fixed chunks if needed."""

        if self.pipeline.stt_provider.supports_streaming_wake():
            while True:
                try:
                    started_at = time.perf_counter()
                    result = await self.pipeline.stt_provider.listen_for_wake_phrase(
                        language=self._stt_request_language(),
                        sample_rate=self.config.sample_rate,
                        input_device_index=self.config.input_device_index,
                        chunk_duration_ms=self.config.streaming_chunk_duration_ms,
                        wake_phrases=wake_phrases,
                    )
                    elapsed_ms = (time.perf_counter() - started_at) * 1000.0
                except RuntimeError:
                    LOGGER.exception("Streaming wake detection failed due to runtime configuration error.")
                    raise
                except Exception:
                    LOGGER.exception("Streaming wake detection failed unexpectedly; falling back to fixed wake capture.")
                    break
                if result.get("phrase"):
                    LOGGER.info(
                        'Wake stream detected: elapsed=%.0fms raw="%s" phrase="%s" match_type="%s"',
                        elapsed_ms,
                        result.get("transcript", ""),
                        result.get("phrase", ""),
                        result.get("match_type", ""),
                    )
                    return
                LOGGER.info(
                    'Wake stream restarted: reason=%s last_transcript="%s"',
                    result.get("restart_reason", "unknown"),
                    result.get("transcript", ""),
                )

        await self._wait_for_wake_word_fixed_window(wake_phrases)

    async def _wait_for_wake_word_fixed_window(self, wake_phrases: tuple[str, ...]) -> None:
        """Listen in short chunks until the configured wake phrase is heard."""

        attempt = 0
        empty_transcript_streak = 0
        while True:
            attempt += 1
            capture_path = await asyncio.to_thread(
                self.recorder.capture_wav,
                self.config.wake_window_seconds,
                self.config.sample_rate,
                "wake",
            )
            try:
                audio_stats = await asyncio.to_thread(self.recorder.analyze_wav, capture_path)
                LOGGER.debug(
                    "Wake capture #%d | file=%s | duration=%.2fs | rms=%.1f | peak=%.1f",
                    attempt,
                    capture_path.name,
                    audio_stats["duration_seconds"],
                    audio_stats["rms"],
                    audio_stats["peak"],
                )
                if audio_stats["rms"] < 20.0 and audio_stats["peak"] < 120.0:
                    LOGGER.warning(
                        "Wake capture #%d looks nearly silent (rms=%.1f, peak=%.1f). "
                        "Likely wrong/muted microphone input device.",
                        attempt,
                        audio_stats["rms"],
                        audio_stats["peak"],
                    )
                try:
                    stt_started_at = time.perf_counter()
                    transcript = (await self.pipeline.stt_provider.transcribe(capture_path, self._stt_request_language())).strip()
                    stt_elapsed_ms = (time.perf_counter() - stt_started_at) * 1000.0
                except RuntimeError:
                    LOGGER.exception("Wake-word transcription failed due to runtime configuration error.")
                    raise
                except Exception:
                    LOGGER.exception("Wake-word transcription failed unexpectedly; continuing to next wake chunk.")
                    continue
            finally:
                capture_path.unlink(missing_ok=True)
            normalized_transcript = normalize_text(transcript)
            wake_match = describe_matched_wake_phrase(transcript, wake_phrases) if transcript else None
            wake_detected = wake_match is not None
            LOGGER.info(
                'Wake transcript #%d: capture=%.2fs stt=%.0fms raw="%s" normalized="%s" detected=%s phrase="%s" match_type="%s"',
                attempt,
                audio_stats["duration_seconds"],
                stt_elapsed_ms,
                transcript,
                normalized_transcript,
                wake_detected,
                wake_match.phrase if wake_match is not None else "",
                wake_match.match_type if wake_match is not None else "",
            )
            if transcript:
                empty_transcript_streak = 0
            else:
                empty_transcript_streak += 1
                if empty_transcript_streak >= 3 and empty_transcript_streak % 3 == 0:
                    LOGGER.warning(
                        "Wake STT has returned empty transcript %d times in a row. "
                        "Check microphone input gain/selection and speak clearly within %.1f seconds.",
                        empty_transcript_streak,
                        self.config.wake_window_seconds,
                    )
            if wake_detected:
                return

    async def _conversation_loop(self) -> None:
        """Process user turns until the user asks the assistant to sleep."""

        loop = asyncio.get_running_loop()
        session_started_at = loop.time()
        request_last_activity_at = session_started_at
        turn = 0
        ignored_turns = 0
        reprompted = False
        use_streaming_stt = self._supports_streaming_stt()
        consecutive_empty_streaming_turns = 0
        skip_next_ready_cue = False

        def mark_request_activity() -> None:
            nonlocal request_last_activity_at
            request_last_activity_at = loop.time()

        while True:
            elapsed_seconds = loop.time() - session_started_at
            idle_elapsed_seconds = loop.time() - request_last_activity_at
            limit_reason = self._conversation_end_reason(
                turn,
                ignored_turns,
                elapsed_seconds,
                idle_elapsed_seconds,
            )
            if limit_reason is not None:
                LOGGER.info("Exiting request mode: %s", limit_reason)
                return

            turn += 1
            stt_elapsed_ms: float | None = None
            self._last_request_capture_meta = {}
            try:
                if use_streaming_stt:
                    if skip_next_ready_cue:
                        LOGGER.info("Skipping request-ready cue before retry turn %d.", turn)
                        skip_next_ready_cue = False
                    else:
                        await self._play_request_ready_cue("first_request" if turn == 1 else "followup")
                    LOGGER.info("Listening for request (turn %d, streaming STT+VAD)...", turn)
                    try:
                        stt_started_at = time.perf_counter()
                        transcript = await self._transcribe_live_request_streaming()
                        stt_elapsed_ms = (time.perf_counter() - stt_started_at) * 1000.0
                    except Exception as exc:
                        if self._is_streaming_stt_config_error(exc):
                            LOGGER.warning(
                                "Streaming STT unavailable due configuration error (%s). "
                                "Falling back to fixed-window STT for this session.",
                                exc,
                            )
                            use_streaming_stt = False
                            continue
                        raise
                    if transcript:
                        LOGGER.debug('Streaming transcript (turn %d): "%s"', turn, transcript)
                else:
                    if skip_next_ready_cue:
                        LOGGER.info("Skipping request-ready cue before retry turn %d.", turn)
                        skip_next_ready_cue = False
                    else:
                        await self._play_request_ready_cue("first_request" if turn == 1 else "followup")
                    LOGGER.info("Listening for request (turn %d)...", turn)
                    stt_started_at = time.perf_counter()
                    capture_path = await asyncio.to_thread(
                        self.recorder.capture_wav,
                        self.config.utterance_seconds,
                        self.config.sample_rate,
                        "utterance",
                    )
                    try:
                        audio_stats = await asyncio.to_thread(self.recorder.analyze_wav, capture_path)
                        LOGGER.debug(
                            "Utterance capture #%d | file=%s | duration=%.2fs | rms=%.1f | peak=%.1f",
                            turn,
                            capture_path.name,
                            audio_stats["duration_seconds"],
                            audio_stats["rms"],
                            audio_stats["peak"],
                        )
                        if self._looks_like_silence(audio_stats):
                            LOGGER.info(
                                "Ignoring turn %d as likely silence/no speech (rms=%.1f peak=%.1f).",
                                turn,
                                audio_stats["rms"],
                                audio_stats["peak"],
                            )
                            ignored_turns += 1
                            if self._should_speak_reprompt(turn, reprompted):
                                await self._play_wake_prompt()
                                mark_request_activity()
                                reprompted = True
                            skip_next_ready_cue = True
                            continue
                        transcript = await self.pipeline.stt_provider.transcribe(
                            capture_path,
                            self._stt_request_language(),
                        )
                    finally:
                        capture_path.unlink(missing_ok=True)
                    transcript = transcript.strip()
                    stt_elapsed_ms = (time.perf_counter() - stt_started_at) * 1000.0
            except RuntimeError:
                LOGGER.exception("Pipeline processing failed due to runtime configuration error.")
                return
            except Exception:
                LOGGER.exception("Pipeline processing failed unexpectedly; listening for the next request.")
                continue

            if not transcript:
                LOGGER.warning("No transcript produced for turn %d. Please try again.", turn)
                if use_streaming_stt:
                    consecutive_empty_streaming_turns += 1
                    if consecutive_empty_streaming_turns >= 2:
                        LOGGER.warning(
                            "Streaming STT returned empty transcripts for %d consecutive turns; "
                            "staying in streaming mode to avoid long noisy fixed-window captures.",
                            consecutive_empty_streaming_turns,
                        )
                ignored_turns += 1
                if consecutive_empty_streaming_turns >= 2 or self._should_speak_reprompt(turn, reprompted):
                    await self._speak_status(self._unclear_reprompt_text(self._current_output_language), self._current_output_language)
                    mark_request_activity()
                    reprompted = True
                skip_next_ready_cue = True
                turn -= 1
                continue
            consecutive_empty_streaming_turns = 0
            capture_quality_reason = self._capture_quality_reprompt_reason(transcript, self._last_request_capture_meta)
            if capture_quality_reason is not None:
                LOGGER.info(
                    'Reprompting capture-quality turn %d before routing (reason=%s transcript="%s" meta=%s).',
                    turn,
                    capture_quality_reason,
                    transcript,
                    self._last_request_capture_meta,
                )
                ignored_turns += 1
                await self._speak_status(self._unclear_reprompt_text(self._current_output_language), self._current_output_language)
                mark_request_activity()
                skip_next_ready_cue = True
                continue
            if is_command_like_sleep_phrase(transcript):
                command_language = self._command_ack_language(transcript)
                LOGGER.info(
                    'Sleep command detected before normal routing on turn %d: "%s"',
                    turn,
                    transcript,
                )
                await self._speak_status(self._sleep_ack_text(command_language), command_language)
                return
            if self._should_ignore_low_information_turn(transcript):
                LOGGER.info(
                    'Ignoring low-information turn %d transcript: "%s"',
                    turn,
                    transcript,
                )
                ignored_turns += 1
                if self._should_speak_reprompt(turn, reprompted):
                    await self._play_wake_prompt()
                    mark_request_activity()
                    reprompted = True
                skip_next_ready_cue = True
                continue
            LOGGER.info('User turn %d transcript: "%s"', turn, transcript)
            ignored_turns = 0

            input_language, output_language, language_confidence, language_reason = self._resolve_turn_languages(transcript)
            LOGGER.info(
                "Language policy (turn %d): input=%s output=%s confidence=%.2f reason=%s mode=%s",
                turn,
                input_language,
                output_language,
                language_confidence,
                language_reason,
                self._output_language_mode,
            )

            if self.config.language_override_commands:
                command = parse_language_override_command(transcript)
                if command is not None:
                    await self._apply_language_override(command, input_language)
                    mark_request_activity()
                    continue

            if is_sleep_command(transcript, input_language) or is_command_like_sleep_phrase(transcript):
                command_language = self._command_ack_language(transcript)
                LOGGER.info(
                    "Sleep command detected on turn %d; acknowledging and returning to wake mode.",
                    turn,
                )
                await self._speak_status(self._sleep_ack_text(command_language), command_language)
                return

            low_confidence_reason = self._low_confidence_reprompt_reason(
                transcript,
                language_confidence,
                language_reason,
            )
            if low_confidence_reason is not None:
                LOGGER.info(
                    'Reprompting low-confidence turn %d before LLM routing (reason=%s transcript="%s").',
                    turn,
                    low_confidence_reason,
                    transcript,
                )
                ignored_turns += 1
                await self._speak_status(self._unclear_reprompt_text(output_language), output_language)
                mark_request_activity()
                skip_next_ready_cue = True
                continue

            try:
                result = await self.pipeline.process_transcription_with_metrics(
                    transcript,
                    output_language,
                    stt_elapsed_ms=stt_elapsed_ms,
                    language_confidence=language_confidence,
                    language_reason=language_reason.split("+", 1)[0],
                    thinking_cue_delay_seconds=(
                        self.config.thinking_cue_delay_seconds if self.config.thinking_cue_enabled else 0.0
                    ),
                    thinking_cue_callback=lambda: self._play_thinking_cue(output_language),
                )
            except RuntimeError:
                LOGGER.exception("Pipeline processing failed due to runtime configuration error.")
                return
            except Exception:
                LOGGER.exception("Pipeline processing failed unexpectedly; listening for the next request.")
                continue

            response_text = result["response_text"].strip()
            audio_output_path = Path(result["audio_output_path"])
            LOGGER.info('Assistant response (turn %d): "%s"', turn, response_text)
            keep_running = await self._play_answer_and_cleanup(audio_output_path)
            if not keep_running:
                return
            if self.config.request_post_tts_guard_seconds > 0:
                await asyncio.sleep(self.config.request_post_tts_guard_seconds)
            mark_request_activity()

    def _supports_streaming_stt(self) -> bool:
        """Return whether live mode can use provider streaming STT endpointing."""

        if not self.config.enable_streaming_stt:
            return False
        return self.pipeline.stt_provider.supports_live_streaming()

    async def _transcribe_live_request_streaming(self) -> str:
        """Transcribe one microphone request with provider-managed streaming endpointing."""

        transcript = await self.pipeline.stt_provider.transcribe_live_utterance(
            language=self._stt_request_language(),
            sample_rate=self.config.sample_rate,
            input_device_index=self.config.input_device_index,
            max_utterance_seconds=self.config.utterance_seconds,
            chunk_duration_ms=self.config.streaming_chunk_duration_ms,
            speech_start_timeout_seconds=self.config.streaming_speech_start_timeout_seconds,
            speech_end_timeout_seconds=self._effective_streaming_speech_end_timeout_seconds(),
            preroll_ms=self.config.preroll_ms if self.config.preroll_enabled else 0,
            ambient_rms=self._ambient_rms,
            ambient_peak=self._ambient_peak,
            min_speech_rms=self.config.utterance_min_rms,
            min_speech_peak=self.config.utterance_min_peak,
            local_speech_end_ms=self.config.streaming_local_speech_end_ms,
            max_active_seconds=self.config.streaming_max_active_seconds,
            no_progress_seconds=self.config.streaming_no_progress_seconds,
            weak_progress_seconds=self.config.streaming_weak_progress_seconds,
            weak_progress_min_tokens=self.config.streaming_weak_progress_min_tokens,
        )
        self._last_request_capture_meta = dict(getattr(self.pipeline.stt_provider, "last_live_capture_stats", {}) or {})
        return transcript

    def _effective_streaming_speech_end_timeout_seconds(self) -> float:
        """Return a speech-end timeout floor that avoids ending turns too aggressively."""

        return max(1.2, self.config.streaming_speech_end_timeout_seconds)

    def _stt_request_language(self) -> LanguageCode:
        """Return preferred STT request language while adaptive mode uses provider multi-language config."""

        if self.config.language_mode == "adaptive":
            return self._current_output_language
        return self.config.language

    def _resolve_turn_languages(self, transcript: str) -> tuple[LanguageCode, LanguageCode, float, str]:
        """Resolve input and output languages for the current turn using configured policy."""

        if self.config.language_mode == "fixed":
            input_language: LanguageCode = self.config.language
            input_confidence = 1.0
            input_reason = "fixed_input_language"
        else:
            input_language, input_confidence, input_reason = detect_input_language(
                transcript,
                fallback_language=self._current_output_language,
            )

        if self._output_language_mode == "fixed":
            output_language = self._output_language_fixed
            self._current_output_language = output_language
            self._language_sticky_remaining = 0
            return input_language, output_language, input_confidence, f"{input_reason}+fixed_output"

        # Auto output mode: follow detected input language unless switching is suppressed.
        if input_language == self._current_output_language:
            return input_language, self._current_output_language, input_confidence, f"{input_reason}+same_output"

        if input_confidence < self.config.language_switch_min_confidence:
            return input_language, self._current_output_language, input_confidence, f"{input_reason}+low_confidence_hold"

        if self._language_sticky_remaining > 0:
            self._language_sticky_remaining -= 1
            return input_language, self._current_output_language, input_confidence, f"{input_reason}+sticky_hold"

        self._current_output_language = input_language
        self._language_sticky_remaining = max(0, self.config.language_switch_sticky_turns)
        return input_language, self._current_output_language, input_confidence, f"{input_reason}+switched"

    async def _apply_language_override(self, command: str, input_language: LanguageCode) -> None:
        """Apply runtime language policy override command and acknowledge it to the user."""

        if command == "force_en":
            self._output_language_mode = "fixed"
            self._output_language_fixed = "en"
            self._current_output_language = "en"
            self._bilingual_output_enabled = False
            await self._speak_status("English-only mode enabled.", "en")
            LOGGER.info("Language override applied: English-only output mode.")
            return

        if command == "force_vi":
            self._output_language_mode = "fixed"
            self._output_language_fixed = "vi"
            self._current_output_language = "vi"
            self._bilingual_output_enabled = False
            await self._speak_status("Da bat che do chi tra loi bang tieng Viet.", "vi")
            LOGGER.info("Language override applied: Vietnamese-only output mode.")
            return

        if command == "auto":
            self._output_language_mode = "auto"
            self._bilingual_output_enabled = False
            message = "Auto language mode enabled." if input_language == "en" else "Da bat che do ngon ngu tu dong."
            await self._speak_status(message, input_language)
            LOGGER.info("Language override applied: auto output mode.")
            return

        if command == "bilingual":
            self._bilingual_output_enabled = True
            message = (
                "Bilingual mode request acknowledged. Single-output mode remains default for low latency."
                if input_language == "en"
                else "Da ghi nhan yeu cau che do song ngu. Che do mot ngon ngu van la mac dinh de giam do tre."
            )
            await self._speak_status(message, input_language)
            LOGGER.info("Language override applied: bilingual requested (single-output default retained).")

    async def _speak_status(self, text: str, language: LanguageCode) -> None:
        """Synthesize and play a short status acknowledgement line."""

        status_path = self.config.temp_audio_dir / f"status_{uuid4().hex}.mp3"
        try:
            if language == "vi" and self._looks_like_romanized_vietnamese(text):
                LOGGER.warning("Vietnamese cue/status text has no diacritics and may sound unnatural: %r", text)
            await asyncio.wait_for(
                self.pipeline.primary_tts_provider.synthesize(text, language, status_path),
                timeout=self.pipeline.timeout_seconds,
            )
            await asyncio.to_thread(self.player.play, status_path)
        except Exception:
            LOGGER.debug("Status acknowledgement playback skipped due to synthesis/playback failure.")
        finally:
            status_path.unlink(missing_ok=True)

    async def _play_answer_and_cleanup(self, audio_output_path: Path) -> bool:
        """Play one answer clip and delete the generated MP3 to avoid persistent output artifacts."""

        try:
            if self._barge_in_enabled():
                interrupted = await asyncio.to_thread(
                    self.player.play_interruptible,
                    audio_output_path,
                    self._detect_barge_in_sync,
                    self.config.barge_in_grace_seconds,
                )
                if interrupted:
                    LOGGER.info("Answer playback interrupted by confirmed barge-in phrase.")
            else:
                await asyncio.to_thread(self.player.play, audio_output_path)
        except RuntimeError:
            LOGGER.exception("Audio playback failed due to runtime configuration error.")
            return False
        except Exception:
            LOGGER.exception("Audio playback failed unexpectedly; keeping conversation loop alive.")
            return True
        finally:
            audio_output_path.unlink(missing_ok=True)
        return True

    def _cleanup_stale_runtime_audio(self) -> None:
        """Remove stale runtime audio from interrupted prior sessions."""

        patterns = (
            "response_*.mp3",
            "wake_*.wav",
            "utterance_*.wav",
            "status_*.mp3",
            "barge_in_*.wav",
            "wake_ack_*.mp3",
            "request_ready_*.mp3",
            "thinking_*.mp3",
            "calibration_*.wav",
        )
        removed_count = 0
        for directory in (self.config.temp_audio_dir, Path("output")):
            if not directory.exists():
                continue
            for pattern in patterns:
                for audio_path in directory.glob(pattern):
                    try:
                        audio_path.unlink()
                        removed_count += 1
                    except OSError:
                        LOGGER.debug("Unable to remove stale runtime audio: %s", audio_path)
        if removed_count:
            LOGGER.info("Removed %d stale runtime audio files from previous sessions.", removed_count)

    def _log_command_card(self, device_info: dict[str, object], wake_phrases: tuple[str, ...]) -> None:
        """Log the operator-facing commands and live-mode state."""

        LOGGER.info(
            'Command card: wake="%s" aliases="%s" exit_en="goodbye | stop listening | go to sleep | sleep now | exit | quit" exit_vi="tạm biệt | chào tạm biệt | dừng nghe | ngủ đi | thoát" '
            "language_mode=%s output_mode=%s output=%s barge_in=%s input_device=%s",
            self.config.wake_word,
            ", ".join(phrase for phrase in wake_phrases if phrase != self.config.wake_word) or "none",
            self.config.language_mode,
            self._output_language_mode,
            self._output_language_fixed if self._output_language_mode == "fixed" else "auto",
            self.config.barge_in_mode,
            device_info.get("index", "unknown"),
        )
        if self.config.language_mode == "fixed" and self._output_language_mode == "fixed":
            LOGGER.info(
                "Fixed language profile active: language_mode=fixed output_mode=fixed output=%s",
                self._output_language_fixed,
            )

    async def _run_startup_calibration(self) -> None:
        """Capture brief ambient audio and log deployment tuning diagnostics."""

        try:
            capture_path = await asyncio.to_thread(
                self.recorder.capture_wav,
                self.config.startup_calibration_seconds,
                self.config.sample_rate,
                "calibration",
            )
        except RuntimeError:
            LOGGER.exception("Startup mic calibration skipped due to runtime configuration error.")
            return
        except OSError:
            LOGGER.exception("Startup mic calibration skipped because microphone capture failed.")
            return
        try:
            audio_stats = await asyncio.to_thread(self.recorder.analyze_wav, capture_path)
            self._log_calibration_result(audio_stats)
        finally:
            capture_path.unlink(missing_ok=True)

    def _log_calibration_result(self, audio_stats: dict[str, float]) -> None:
        """Log mic/environment health without mutating runtime thresholds."""

        rms = audio_stats.get("rms", 0.0)
        peak = audio_stats.get("peak", 0.0)
        clipping_ratio = audio_stats.get("clipping_ratio", 0.0)
        self._ambient_rms = rms
        self._ambient_peak = peak
        LOGGER.info(
            "Startup mic calibration: duration=%.2fs rms=%.1f peak=%.1f clipping=%.4f",
            audio_stats.get("duration_seconds", 0.0),
            rms,
            peak,
            clipping_ratio,
        )
        if self._looks_like_silence(audio_stats):
            LOGGER.warning(
                "Startup mic calibration detected near-silent input; check VOICE_LOOP_INPUT_DEVICE_INDEX or mic gain."
            )
        if rms >= max(self.config.utterance_min_rms * 4.0, 120.0):
            LOGGER.warning(
                "Startup mic calibration detected high ambient noise; consider raising VOICE_LOOP_UTTERANCE_MIN_RMS."
            )
        if clipping_ratio >= 0.001:
            LOGGER.warning("Startup mic calibration detected clipping; reduce microphone input gain.")

    def _barge_in_enabled(self) -> bool:
        """Return whether conservative wake-phrase barge-in is enabled."""

        return self.config.barge_in_mode.strip().lower() == "wake_phrase"

    def _detect_barge_in_sync(self) -> bool:
        """Confirm interruption with microphone energy plus explicit wake/interruption speech."""

        capture_path = self.recorder.capture_wav(
            self.config.barge_in_listen_seconds,
            self.config.sample_rate,
            "barge_in",
        )
        try:
            audio_stats = self.recorder.analyze_wav(capture_path)
            if (
                audio_stats.get("rms", 0.0) < self.config.barge_in_min_rms
                and audio_stats.get("peak", 0.0) < self.config.barge_in_min_peak
            ):
                LOGGER.debug(
                    "Barge-in candidate ignored as low energy (rms=%.1f peak=%.1f).",
                    audio_stats.get("rms", 0.0),
                    audio_stats.get("peak", 0.0),
                )
                return False
            transcript = asyncio.run(
                self.pipeline.stt_provider.transcribe(capture_path, self._stt_request_language())
            ).strip()
            if self._is_confirmed_barge_in_phrase(transcript):
                LOGGER.info('Confirmed barge-in transcript: "%s"', transcript)
                return True
            LOGGER.info('Barge-in candidate ignored; transcript did not contain wake/interruption phrase: "%s"', transcript)
            return False
        except RuntimeError:
            LOGGER.exception("Barge-in detection failed due to runtime configuration error.")
            return False
        except Exception:
            LOGGER.exception("Barge-in detection failed unexpectedly.")
            return False
        finally:
            capture_path.unlink(missing_ok=True)

    def _is_confirmed_barge_in_phrase(self, transcript: str) -> bool:
        """Require explicit assistant address before interrupting playback."""

        normalized = normalize_text(transcript)
        if not normalized:
            return False
        if detect_matched_wake_phrase(transcript, self._wake_phrases()) is not None:
            return True
        return any(
            phrase in normalized
            for phrase in (
                "stop lemon",
                "lemon stop",
                "pause lemon",
                "lemon pause",
                "interrupt lemon",
                "lemon interrupt",
            )
        )

    @staticmethod
    def _is_streaming_stt_config_error(exc: Exception) -> bool:
        """Return whether an exception indicates streaming STT configuration mismatch."""

        error_text = str(exc).lower()
        return (
            "expected resource location to be global" in error_text
            or "invalidargument: 400" in error_text
            or "google.api_core.exceptions.invalidargument" in error_text
        )

    async def _acknowledge_wake(self) -> None:
        """Emit a low-latency wake acknowledgement based on configured mode."""

        mode = self._wake_ack_mode()
        if mode == "beep":
            await asyncio.to_thread(self._play_beep)
            return
        if mode == "adaptive":
            await asyncio.to_thread(self._play_beep)
            if self.config.wake_ack_adaptive_speak_on_wake:
                await self._play_wake_prompt()
            return
        if mode == "speech":
            await self._play_wake_prompt()

    async def _play_request_ready_cue(self, phase: str = "followup") -> None:
        """Emit the short cue immediately before request capture opens."""

        mode = self.config.request_ready_cue_mode.strip().lower()
        if mode == "none":
            return
        if mode == "beep":
            await asyncio.to_thread(self._play_beep)
            return
        language = self._current_output_language
        cue_type = "request_ready_first" if phase == "first_request" else "request_ready"
        cue_index, cue_text = self._next_rotating_cue(cue_type, language)
        played = await self._play_cached_short_cue(
            cue_type=cue_type,
            language=language,
            cue_index=cue_index,
            text=cue_text,
            cache_enabled=self.config.request_ready_cache,
            log_phase=phase,
        )
        if not played:
            await asyncio.to_thread(self._play_beep)

    async def _play_thinking_cue(self, language: LanguageCode) -> None:
        """Play one short transition cue while a slow direct LLM answer is pending."""

        cue_index, cue_text = self._next_rotating_cue("thinking", language)
        await self._play_cached_short_cue(
            cue_type="thinking",
            language=language,
            cue_index=cue_index,
            text=cue_text,
            cache_enabled=True,
            log_phase="thinking",
        )

    def _next_rotating_cue(self, cue_type: str, language: LanguageCode) -> tuple[int, str]:
        """Return the next deterministic cue text for a cue type/language pair."""

        if cue_type == "request_ready_first":
            text = self.config.request_ready_first_text_vi if language == "vi" else self.config.request_ready_first_text_en
            return 0, text
        if cue_type == "thinking":
            texts = self.config.thinking_texts_vi if language == "vi" else self.config.thinking_texts_en
            index = self._thinking_cue_indexes[language]
            self._thinking_cue_indexes[language] = (index + 1) % max(1, len(texts))
        else:
            texts = self.config.request_ready_texts_vi if language == "vi" else self.config.request_ready_texts_en
            index = self._request_ready_cue_indexes[language]
            self._request_ready_cue_indexes[language] = (index + 1) % max(1, len(texts))
        if not texts:
            return 0, ""
        return index, texts[index % len(texts)]

    async def _play_cached_short_cue(
        self,
        cue_type: str,
        language: LanguageCode,
        cue_index: int,
        text: str,
        cache_enabled: bool,
        log_phase: str,
    ) -> bool:
        """Synthesize and play a short cue, caching stable prompts when enabled."""

        if not text.strip():
            return False
        if language == "vi" and self._looks_like_romanized_vietnamese(text):
            LOGGER.warning("Vietnamese cue/status text has no diacritics and may sound unnatural: %r", text)

        cache_key = (cue_type, language, cue_index)
        cached_path = self._cue_cache_paths.get(cache_key)
        if cache_enabled and cached_path is not None and cached_path.exists():
            try:
                await asyncio.to_thread(self.player.play, cached_path)
                LOGGER.info(
                    "Cue playback: type=%s phase=%s language=%s cache=hit text=%r",
                    cue_type,
                    log_phase,
                    language,
                    text,
                )
                return True
            except (RuntimeError, OSError, ValueError):
                LOGGER.exception("Cached cue playback failed: type=%s language=%s", cue_type, language)

        cue_path = self.config.temp_audio_dir / f"{cue_type}_{language}_{cue_index}_{uuid4().hex}.mp3"
        try:
            started_at = time.perf_counter()
            await asyncio.wait_for(
                self.pipeline.primary_tts_provider.synthesize(text, language, cue_path),
                timeout=self.pipeline.timeout_seconds,
            )
            await asyncio.to_thread(self.player.play, cue_path)
            elapsed_ms = (time.perf_counter() - started_at) * 1000.0
            LOGGER.info(
                "Cue playback: type=%s phase=%s language=%s cache=%s elapsed_ms=%.0f text=%r",
                cue_type,
                log_phase,
                language,
                "miss",
                elapsed_ms,
                text,
            )
            if cache_enabled:
                self._cue_cache_paths[cache_key] = cue_path
            return True
        except (RuntimeError, asyncio.TimeoutError, OSError, ValueError):
            LOGGER.exception("Cue synthesis/playback failed: type=%s language=%s", cue_type, language)
            cue_path.unlink(missing_ok=True)
            return False
        finally:
            if not cache_enabled:
                cue_path.unlink(missing_ok=True)

    async def _play_wake_prompt(self) -> None:
        """Synthesize and play a short acknowledgement prompt."""

        cached_path = self._cached_wake_prompt_path
        if cached_path is not None and cached_path.exists():
            try:
                await asyncio.to_thread(self.player.play, cached_path)
                return
            except Exception:
                LOGGER.debug("Cached wake prompt playback failed; falling back to fresh synthesis.")

        prompt_path: Path | None = None
        try:
            prompt_path = self.config.temp_audio_dir / f"wake_ack_{uuid4().hex}.mp3"
            prompt_text = self._wake_prompt_text()
            await asyncio.wait_for(
                self.pipeline.primary_tts_provider.synthesize(prompt_text, self.config.language, prompt_path),
                timeout=self.pipeline.timeout_seconds,
            )
            await asyncio.to_thread(self.player.play, prompt_path)
        except Exception:
            LOGGER.debug("Wake prompt playback skipped due to synthesis/playback failure.")
        finally:
            if prompt_path is not None:
                prompt_path.unlink(missing_ok=True)

    def _play_beep(self) -> None:
        """Play a short wake beep when supported on the host OS."""

        try:
            import winsound

            winsound.Beep(self.config.wake_ack_beep_frequency, self.config.wake_ack_beep_duration_ms)
        except Exception:
            LOGGER.debug("Wake beep unavailable on this platform.")

    def _wake_phrases(self) -> tuple[str, ...]:
        """Build wake phrase variants used by the detector."""

        phrases = [self.config.wake_word, *self.config.wake_aliases]
        cleaned = [phrase.strip() for phrase in phrases if phrase.strip()]
        if not cleaned:
            return ("hey lemon",)
        return tuple(dict.fromkeys(cleaned))

    def _looks_like_silence(self, audio_stats: dict[str, float]) -> bool:
        """Return whether a captured utterance likely does not contain speech."""

        return (
            audio_stats.get("rms", 0.0) < self.config.utterance_min_rms
            and audio_stats.get("peak", 0.0) < self.config.utterance_min_peak
        )

    def _low_confidence_reprompt_reason(
        self,
        transcript: str,
        language_confidence: float,
        language_reason: str,
    ) -> str | None:
        """Return why a turn should be reprompted instead of sent to LLM direct."""

        normalized = normalize_text(transcript)
        tokens = normalized.split()
        base_language_reason = language_reason.split("+", 1)[0]
        if is_command_like_sleep_phrase(transcript):
            return "command_like"
        if base_language_reason == "no_marker_hit" and language_confidence < 0.60 and len(tokens) <= 4:
            return "low_confidence_no_marker"
        if language_confidence < 0.55 and len(tokens) <= 3:
            return "low_confidence_short"
        return None

    def _capture_quality_reprompt_reason(
        self,
        transcript: str,
        capture_meta: dict[str, object],
    ) -> str | None:
        """Return whether capture quality is too weak to route safely."""

        normalized = normalize_text(transcript)
        tokens = normalized.split()
        if self._is_unfinished_question_stem(tokens):
            return "unfinished_question_stem"
        end_reason = str(capture_meta.get("end_reason", ""))
        if end_reason in {"no_stt_progress", "weak_stt_progress"} and len(tokens) <= 3:
            return end_reason
        if end_reason == "max_active" and len(tokens) <= 3:
            return "max_active_low_information"
        if len(tokens) <= 2 and any(token in {"what", "how", "where", "latest", "explain", "summarize"} for token in tokens):
            return "malformed_short_query"
        return None

    @staticmethod
    def _unclear_reprompt_text(language: LanguageCode) -> str:
        """Return a short reprompt for transcripts that should not be answered."""

        if language == "vi":
            return "Tôi chưa nghe rõ. Bạn nói lại giúp tôi."
        return "I didn't catch that. Please repeat."

    def _should_ignore_low_information_turn(self, transcript: str) -> bool:
        """Skip very short transcripts that are likely noise."""

        normalized = normalize_text(transcript)
        tokens = normalized.split()
        if len(normalized) < self.config.minimum_transcript_characters:
            return True
        if self._is_unfinished_question_stem(tokens):
            return True
        return len(tokens) <= 1

    @staticmethod
    def _is_unfinished_question_stem(tokens: list[str]) -> bool:
        """Return whether a transcript is only a question lead without content."""

        if len(tokens) > 3:
            return False
        return set(tokens) <= {"what", "is", "are", "the", "how", "where", "explain", "summarize", "latest"}

    def _wake_ack_mode(self) -> str:
        """Return normalized wake acknowledgement mode."""

        mode = self.config.wake_ack_mode.strip().lower()
        if mode in {"none", "beep", "speech", "adaptive"}:
            return mode
        return "adaptive"

    def _command_ack_language(self, transcript: str) -> LanguageCode:
        """Return acknowledgement language for sleep/exit commands."""

        normalized = normalize_text(transcript)
        tokens = normalized.split()
        if tokens and tokens[0] in {"stop", "top", "quit", "exit", "goodbye"}:
            return "en"
        if any(phrase in normalized for phrase in ("go to sleep", "sleep now", "stop listening")):
            return "en"
        if any(phrase in normalized for phrase in ("tam dung", "dung lai", "ngu di", "ket thuc", "thoat")):
            return "vi"
        detected_language, _, _ = detect_input_language(transcript, self._current_output_language)
        return detected_language

    @staticmethod
    def _sleep_ack_text(language: LanguageCode) -> str:
        """Return localized acknowledgement spoken when user asks assistant to stop."""

        if language == "vi":
            return "Đã dừng lắng nghe. Hãy gọi Hey Lemon khi bạn cần tôi."
        return "Okay, I will stop listening. Say Hey Lemon when you need me."

    @staticmethod
    def _looks_like_romanized_vietnamese(text: str) -> bool:
        """Return whether Vietnamese cue text is likely unaccented romanized Vietnamese."""

        if _VI_DIACRITIC_RE.search(text):
            return False
        normalized = normalize_text(text)
        tokens = set(normalized.split())
        romanized_markers = {
            "ban",
            "can",
            "hoi",
            "gi",
            "them",
            "toi",
            "nghe",
            "day",
            "noi",
            "tiep",
            "di",
            "de",
            "kiem",
            "tra",
            "cho",
            "mot",
            "chut",
        }
        return len(tokens & romanized_markers) >= 2

    def _wake_prompt_text(self) -> str:
        """Return localized wake acknowledgement prompt text."""

        if self.config.language == "vi":
            return self.config.wake_ack_prompt_text_vi
        return self.config.wake_ack_prompt_text_en

    def _should_speak_reprompt(self, turn: int, reprompted: bool) -> bool:
        """Return whether adaptive mode should speak the post-wake prompt."""

        return (
            self._wake_ack_mode() == "adaptive"
            and not self.config.wake_ack_adaptive_speak_on_wake
            and not reprompted
            and turn <= 2
        )

    def _should_prime_wake_prompt_cache(self) -> bool:
        """Return whether wake-prompt caching should be prepared at startup."""

        mode = self._wake_ack_mode()
        return mode == "speech" or (mode == "adaptive" and self.config.wake_ack_adaptive_speak_on_wake)

    def _should_prime_request_cue_cache(self) -> bool:
        """Return whether request and thinking cue audio should be prepared at startup."""

        return self.config.request_ready_cue_mode.strip().lower() == "speech" and self.config.request_ready_cache

    def _start_startup_cache_priming(self) -> None:
        """Start cue-cache priming without delaying wake listening."""

        if self._should_prime_wake_prompt_cache():
            self._start_background_cache_task(self._prime_wake_prompt_cache(), "wake_prompt")
        if self._should_prime_request_cue_cache():
            self._start_background_cache_task(self._prime_request_cue_cache(), "request_cues")

    def _start_background_cache_task(self, coroutine, label: str) -> None:
        """Track a background cache task and log failures when it finishes."""

        task = asyncio.create_task(coroutine)
        self._startup_cache_tasks.add(task)
        LOGGER.info("Startup cache priming started in background: %s", label)
        task.add_done_callback(lambda completed_task: self._handle_background_cache_task_done(completed_task, label))

    def _handle_background_cache_task_done(self, task: asyncio.Task[None], label: str) -> None:
        """Remove a completed cache task and surface unexpected failures."""

        self._startup_cache_tasks.discard(task)
        if task.cancelled():
            LOGGER.debug("Startup cache priming cancelled: %s", label)
            return
        try:
            task.result()
            LOGGER.debug("Startup cache priming finished: %s", label)
        except (RuntimeError, asyncio.TimeoutError, OSError, ValueError):
            LOGGER.exception("Startup cache priming failed: %s", label)

    async def _prime_request_cue_cache(self) -> None:
        """Pre-generate short cue audio so first request capture starts without TTS delay."""

        cue_specs: list[tuple[str, LanguageCode, int, str]] = []
        for language in ("en", "vi"):
            first_text = self.config.request_ready_first_text_vi if language == "vi" else self.config.request_ready_first_text_en
            cue_specs.append(("request_ready_first", language, 0, first_text))
            ready_texts = self.config.request_ready_texts_vi if language == "vi" else self.config.request_ready_texts_en
            for index, text in enumerate(ready_texts):
                cue_specs.append(("request_ready", language, index, text))
            if self.config.thinking_cue_enabled:
                thinking_texts = self.config.thinking_texts_vi if language == "vi" else self.config.thinking_texts_en
                for index, text in enumerate(thinking_texts):
                    cue_specs.append(("thinking", language, index, text))

        for cue_type, language, cue_index, text in cue_specs:
            if not text.strip():
                continue
            cache_key = (cue_type, language, cue_index)
            if cache_key in self._cue_cache_paths:
                continue
            cue_path = self.config.temp_audio_dir / f"{cue_type}_{language}_{cue_index}_cached.mp3"
            try:
                if language == "vi" and self._looks_like_romanized_vietnamese(text):
                    LOGGER.warning("Vietnamese cue/status text has no diacritics and may sound unnatural: %r", text)
                await asyncio.wait_for(
                    self.pipeline.primary_tts_provider.synthesize(text, language, cue_path),
                    timeout=self.pipeline.timeout_seconds,
                )
                self._cue_cache_paths[cache_key] = cue_path
            except (RuntimeError, asyncio.TimeoutError, OSError, ValueError):
                cue_path.unlink(missing_ok=True)
                LOGGER.exception("Unable to pre-generate cue audio: type=%s language=%s", cue_type, language)

    async def _prime_wake_prompt_cache(self) -> None:
        """Pre-generate wake prompt audio once so acknowledgement speech starts faster."""

        prompt_path = self.config.temp_audio_dir / f"wake_ack_cached_{self.config.language}.mp3"
        try:
            await asyncio.wait_for(
                self.pipeline.primary_tts_provider.synthesize(
                    self._wake_prompt_text(),
                    self.config.language,
                    prompt_path,
                ),
                timeout=self.pipeline.timeout_seconds,
            )
            self._cached_wake_prompt_path = prompt_path
            LOGGER.debug("Prepared cached wake prompt audio: %s", prompt_path.name)
        except Exception:
            prompt_path.unlink(missing_ok=True)
            LOGGER.debug("Unable to pre-generate cached wake prompt audio.")

    def _conversation_end_reason(
        self,
        turn: int,
        ignored_turns: int,
        elapsed_seconds: float,
        idle_elapsed_seconds: float | None = None,
    ) -> str | None:
        """Return a detailed reason when conversation exits due to configured limits."""

        if idle_elapsed_seconds is None:
            idle_elapsed_seconds = elapsed_seconds
        if self.config.request_max_turns > 0 and turn >= self.config.request_max_turns:
            return f"turn limit reached ({turn}/{self.config.request_max_turns})"
        if self.config.request_max_ignored_turns > 0 and ignored_turns >= self.config.request_max_ignored_turns:
            return (
                f"ignored-turn limit reached "
                f"({ignored_turns}/{self.config.request_max_ignored_turns})"
            )
        if self.config.request_idle_timeout_seconds > 0 and idle_elapsed_seconds >= self.config.request_idle_timeout_seconds:
            return (
                f"idle-timeout reached "
                f"({idle_elapsed_seconds:.1f}s/{self.config.request_idle_timeout_seconds:.1f}s)"
            )
        if self.config.request_max_session_seconds > 0 and elapsed_seconds >= self.config.request_max_session_seconds:
            return (
                f"session-duration limit reached "
                f"({elapsed_seconds:.1f}s/{self.config.request_max_session_seconds:.1f}s)"
            )
        return None

    def _should_end_conversation(
        self,
        turn: int,
        ignored_turns: int,
        elapsed_seconds: float,
        idle_elapsed_seconds: float | None = None,
    ) -> bool:
        """Return whether the request stage should end and return to wake mode."""

        return self._conversation_end_reason(turn, ignored_turns, elapsed_seconds, idle_elapsed_seconds) is not None
