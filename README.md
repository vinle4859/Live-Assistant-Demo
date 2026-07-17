# Live Voice Assistant

Headless Python voice-to-voice assistant for English and Vietnamese interactions, designed for deployment on autonomous robots (Raspberry Pi 4).

## Architecture

- **STT**: Google Cloud Speech-to-Text (streaming, boost factor `20.0` for domain context)
- **TTS**: Google Cloud Text-to-Speech (edge-tts fallback)
- **LLM**: Gemini on Vertex AI for direct-answer generation
- **Q&A**: Local SQLite knowledge base with `lexical`, `vector`, or `hybrid` retrieval
- **Audio**: In-process ctypes WinMM on Windows; native CLI player (`gst-play-1.0`, `mpg123`, etc.) on Linux
- **VAD**: Dynamic calibration with RMS/peak clamping to prevent lockup in noisy environments
- **Language**: Bilingual adaptive (VI/EN per turn) with exclusion-guarded transcript correction
- **Response routing**: `local_db → direct_LLM → hard fallback`

## Quick Start

### Prerequisites
- Python 3.12+
- Google Cloud credentials (STT, TTS, Vertex AI)
- Microphone and speaker access

### Setup

```bash
python -m venv .venv

# Windows
.\.venv\Scripts\python -m pip install -r requirements.txt

# Linux / Raspberry Pi
.venv/bin/pip install -r requirements.txt
```

### Google Cloud Authentication

```bash
# Local development
gcloud auth application-default login

# Headless / Raspberry Pi deployment
# Use a service account key:
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
```

### Run

```bash
python main.py --language adaptive   # auto-detect EN/VI per turn
python main.py --language en          # English only
python main.py --language vi          # Vietnamese only
```

### Smoke Test (no microphone)

```bash
python main.py --diagnose-transcript "What is Greenwich Vietnam?" --diagnose-language en
```

## Configuration

Copy `.env.example` to `.env`. The app loads `.env` automatically; falls back to `.env.example` if missing.

See [.env.example](.env.example) for all available configuration options with descriptions.

### Key Settings

| Setting | Purpose |
|---|---|
| `VOICE_LOOP_WAKE_WORD` | Wake phrase (default: `hey lemon`) |
| `VOICE_LOOP_WAKE_ALIASES` | Alternative wake phrases |
| `VOICE_LOOP_STT_HINT_PHRASES` | STT phrase boost hints |
| `VOICE_LOOP_TRANSCRIPT_CHEATS` | Domain-specific STT corrections with context/exclusion guards |
| `VOICE_LOOP_ENABLE_STREAMING_STT` | Live streaming STT with server endpointing |
| `VOICE_LOOP_SAMPLE_RATE` | Microphone sample rate |
| `VOICE_LOOP_INPUT_DEVICE_INDEX` | PyAudio input device override |
| `VOICE_LOOP_LOG_LEVEL` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |

### Voice Commands

| Command | Action |
|---|---|
| `Hey Lemon` / configured alias | Enter request mode |
| `stop listening` / `go to sleep` / `exit` / `quit` | Return to wake mode |

## Docker

```bash
docker build -t live-assistant .
```

See [DOCKER_HANDOFF.md](DOCKER_HANDOFF.md) for the full container contract (volumes, credentials, device passthrough).

## Google Cloud IAM

Required roles for the service account:

| Role | Purpose |
|---|---|
| `roles/speech.client` | Speech-to-Text API |
| `roles/texttospeech.user` | Text-to-Speech API |
| `roles/aiplatform.user` | Gemini on Vertex AI |
| `roles/serviceusage.serviceUsageConsumer` | API consumption |

## Deployment Launcher & Diagnostics (Windows Edge Device)

### Edge Deployment Launcher
The [`run_assistant_edge.bat`](file:///e:/ARL%20Projects/Live%20Assistant/run_assistant_edge.bat) is the single deployment script designed to be run on the edge machine. It automatically:
- Validates `.env` variables for critical setup (using `tools/check_env.py`).
- Activates the virtual environment and installs/caches dependencies.
- Runs audio diagnostics (using `tools/verify_audio.py`) to report input/output devices and master volume levels (pausing if muted or error-level).
- Spins up a background Cloudflare Quick Tunnel, extracts the ephemeral public URL, and prints it in the console.

### Offline Q&A Batch Diagnostics
You can validate the routing pipeline (local DB match vs LLM fallback, noise filtering, and bilingual voice routing) offline using:
```bash
python tools/diagnose_batch.py
```
This runs the Q&A battery defined in [`tools/diagnose_cases.json`](file:///e:/ARL%20Projects/Live%20Assistant/tools/diagnose_cases.json) in milliseconds by mocking network-heavy TTS calls, and prints a final routing accuracy report.

## Tests

```bash
python -m pytest tests/
```

## Project Structure

```
main.py                  # CLI entrypoint
voice_loop/
  live_assistant.py      # Main voice loop orchestrator
  audio.py               # In-process audio playback
  config.py              # Environment config loader
  factory.py             # Provider factory
  transcript_cheats.py   # STT transcript corrections
  domain_profile.py      # Domain-specific STT profiles
  scripted_speech.py     # Scripted speech sequences
  providers/
    google_stt.py        # Google Cloud STT provider
    google_tts.py        # Google Cloud TTS provider
    edge_tts.py          # Edge TTS fallback provider
data/
  knowledge_base.sqlite3 # Local Q&A knowledge base
```
