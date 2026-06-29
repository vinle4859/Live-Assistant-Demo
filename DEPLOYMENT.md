# Deployment

This project is ready to hand off for live Docker deployment. The repo includes a starter `Dockerfile`; the Docker team still owns host-specific device passthrough, compose/Kubernetes manifests, image registry, and production rollout.

## Runtime Shape

- Process type: live microphone CLI assistant.
- Live command: `python main.py --language adaptive`
- Diagnostic command without microphone: `python main.py --diagnose-transcript "What is Greenwich Vietnam?" --diagnose-language en`
- Script validation without microphone: `python main.py --mode script --script-file event_script.txt --script-validate`
- No HTTP server, API port, or health endpoint exists today.
- Cloud providers remain required for STT, TTS, and Gemini.

## Required Runtime Inputs

- `data/knowledge_base.sqlite3`
- Google Cloud access for:
  - Speech-to-Text
  - Text-to-Speech
  - Vertex AI Gemini
- A writable output directory.
- Microphone and speaker access for live mode.
- Speaker access for scripted mode; microphone, STT, DB routing, and LLM are not used in scripted mode.

## Docker Handoff

See `DOCKER_HANDOFF.md` for:
- the starter Dockerfile
- image content boundaries
- cloud credential options
- audio passthrough notes
- required environment variables
- Docker validation commands

Recommended Docker env path values:

```env
VOICE_LOOP_DB_PATH=/app/data/knowledge_base.sqlite3
VOICE_LOOP_OUTPUT_DIR=/app/output
VOICE_LOOP_QA_SEED_AUTO_SYNC=false
VOICE_LOOP_STT_MODEL=
VOICE_LOOP_STT_LOCATION=global
VOICE_LOOP_DOMAIN_PROFILE=greenwich
```

## Deployment Package

Include:
- `Dockerfile`
- `main.py`
- `voice_loop/`
- `requirements.txt`
- `.env.example`
- `.dockerignore`
- deployment docs
- `IMPLEMENTATION_NOTES.md`
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
- local debug files

## Validation

Before handoff:

```powershell
.\.venv\Scripts\python.exe -m pytest --basetemp .pytest_tmp
.\.venv\Scripts\python.exe main.py --diagnose-transcript "What is Greenwich Vietnam?" --diagnose-language en
```

Source or CI validation before building the runtime image:

```bash
python -m pytest
```

Runtime container smoke validation:

```bash
python main.py --diagnose-transcript "What is Greenwich Vietnam?" --diagnose-language en
```

Live validation on the deployment host:
1. Start the live command.
2. Confirm startup mic calibration logs device health.
3. Say the wake word.
4. Ask one local DB question.
5. Ask one direct LLM question.
6. Confirm audio playback and stage timing logs.

## Logging

The app writes logs to stdout and to `output/<mode>_sessions/<mode>_session_*.log`.

Keep `VOICE_LOOP_LOG_LEVEL=INFO` for normal deployment. Use `DEBUG` only during short troubleshooting windows.

## Scripted event mode

Use scripted mode when the assistant only needs to speak planned lines.

1. Validate the file:
   `python main.py --mode script --script-file event_script.txt --script-validate`
2. Pre-render all selected lines:
   `python main.py --mode script --script-file event_script.txt --script-no-play`
3. Rehearse controlled playback:
   `python main.py --mode script --script-file event_script.txt --script-manual-next`

Generated audio is written under `output/script_sessions/<timestamp>/` so the operator can inspect or replay the exact files used for the event.

## Google Cloud Auth

Local Docker:
- Mount ADC or service-account JSON.
- Set `GOOGLE_APPLICATION_CREDENTIALS` inside the container.

GKE:
- Prefer Workload Identity.
- Do not mount credential JSON when Workload Identity is configured.

Required IAM capabilities:
- Speech-to-Text client access
- Text-to-Speech user access
- Vertex AI user access
- service usage consumer access
