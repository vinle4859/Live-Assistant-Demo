# Docker Handoff

This project is a live microphone CLI assistant. It is not an HTTP service and does not expose a health port.

## Runtime Contract

- Live entrypoint: `python main.py --language adaptive`
- No-microphone smoke command: `python main.py --diagnose-transcript "What is Greenwich Vietnam?" --diagnose-language en`
- Required data file: `data/knowledge_base.sqlite3`
- Writable runtime paths:
  - `output/`
  - `data/live_audio/`
- Runtime logs:
  - stdout/stderr
  - `output/<mode>_sessions/<mode>_session_*.log`

## Image Contents

The repo includes a starter `Dockerfile` for the runtime image. The Docker team should treat it as the base deployment image and add host-specific compose/Kubernetes wiring outside this app repo.

Include:
- `Dockerfile`
- `main.py`
- `voice_loop/`
- `requirements.txt`
- `.env.example`
- `implementation.md`
- `data/knowledge_base.sqlite3`


Exclude:
- `.env`
- `.venv/`
- `output/`
- `.pytest_cache/`
- `__pycache__/`
- `tests/`
- `tools/`
- `data/live_audio/*`
- Google credential files
- local debug or scratch files

## System Dependencies

The image needs Python 3.12 plus system packages required by:
- `PyAudio` / PortAudio for microphone capture
- `playsound` and host audio playback support
- Google client libraries from `requirements.txt`

The exact package names depend on the base image. For Debian/Ubuntu images, expect PortAudio and ALSA-related packages to be required.

## Google Cloud Access

The assistant uses:
- Google Cloud Speech-to-Text
- Google Cloud Text-to-Speech
- Vertex AI Gemini

Local Docker options:
- Mount an ADC or service-account credential JSON.
- Set `GOOGLE_APPLICATION_CREDENTIALS` to the mounted file path.
- Set `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`, and `GEMINI_MODEL`.

GKE option:
- Prefer Workload Identity.
- Do not mount credential JSON when Workload Identity is configured.

The runtime identity needs access to Speech-to-Text, Text-to-Speech, Vertex AI, and service usage.

## Audio Device Access

Live mode requires host microphone and speaker access from inside the container.

Linux hosts commonly need:
- `/dev/snd` passthrough
- audio group permissions
- host audio stack configured for the container user

Windows Docker audio passthrough is less reliable. If live microphone mode must run on Windows, native Python deployment may be simpler than Docker.

If audio passthrough is unavailable, use diagnostic transcript mode only.

## Environment

Use `.env.example` as the deployment template. For Docker, these path values are recommended:

```env
VOICE_LOOP_DB_PATH=/app/data/knowledge_base.sqlite3
VOICE_LOOP_OUTPUT_DIR=/app/output
VOICE_LOOP_QA_SEED_AUTO_SYNC=false
VOICE_LOOP_STT_PROVIDER=google
VOICE_LOOP_TTS_PROVIDER=google
VOICE_LOOP_DOMAIN_PROFILE=greenwich
VOICE_LOOP_STT_MODEL=
VOICE_LOOP_STT_LOCATION=global
GOOGLE_CLOUD_LOCATION=global
GEMINI_MODEL=gemini-3.1-flash-lite
GEMINI_FALLBACK_MODEL=gemini-2.5-flash
```

Host-specific values:
- `VOICE_LOOP_INPUT_DEVICE_INDEX`
- `VOICE_LOOP_SAMPLE_RATE`
- `GOOGLE_CLOUD_PROJECT`
- `GOOGLE_CLOUD_LOCATION=global`
- `GEMINI_MODEL=gemini-3.1-flash-lite`
- `GEMINI_FALLBACK_MODEL=gemini-2.5-flash`

## Validation

Source or CI validation before building the runtime image:

```bash
python -m pytest
```

Runtime container smoke validation:

```bash
docker build -t live-assistant-demo .
docker run --rm \
  --env-file .env \
  -v /path/to/google-credentials.json:/run/secrets/google-credentials.json:ro \
  -e GOOGLE_APPLICATION_CREDENTIALS=/run/secrets/google-credentials.json \
  live-assistant-demo \
  python main.py --diagnose-transcript "What is Greenwich Vietnam?" --diagnose-language en
```

Scripted event validation:

```bash
docker run --rm \
  --env-file .env \
  live-assistant-demo \
  python main.py --mode script --script-file event_script.txt --script-validate
```

Live host validation:
1. Start `python main.py --language adaptive`.
2. Confirm startup mic calibration logs a non-silent input device.
3. Say the wake word.
4. Ask one local DB question.
5. Ask one direct LLM question.
6. Confirm audio playback and stage timing logs.

Stage timing logs should include `stt`, `db`, `llm`, `tts`, `total`, `source`, `db_score`, and `db_mode`.
