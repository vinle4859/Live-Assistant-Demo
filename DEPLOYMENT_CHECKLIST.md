# Deployment Checklist

## A. Repository Package
- [ ] Include `main.py`, `voice_loop/`, `requirements.txt`, `.env.example`, `.dockerignore`, deployment docs, and `data/knowledge_base.sqlite3`.
- [ ] Exclude `.env`, `.venv/`, `output/`, `.pytest_cache/`, `__pycache__/`, `tests/`, `tools/`, runtime audio, credentials, and local scratch files from the Docker image.
- [ ] Confirm `data/live_audio/` contains only `.gitkeep` before handoff.
- [ ] Confirm no real Google credential files are present.

## B. Environment
- [ ] Create deployment `.env` from `.env.example`.
- [ ] Set `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`, and `GEMINI_MODEL`.
- [ ] Set `VOICE_LOOP_DB_PATH=/app/data/knowledge_base.sqlite3` for Docker.
- [ ] Set `VOICE_LOOP_OUTPUT_DIR=/app/output` for Docker.
- [ ] Keep `VOICE_LOOP_QA_SEED_AUTO_SYNC=false`.
- [ ] Set host-specific `VOICE_LOOP_INPUT_DEVICE_INDEX` and `VOICE_LOOP_SAMPLE_RATE`.

## C. Docker Runtime
- [ ] Install Python 3.12 runtime.
- [ ] Install PortAudio/PyAudio system dependencies.
- [ ] Provide speaker playback support for `playsound`.
- [ ] Mount or grant Google Cloud credentials.
- [ ] Pass through microphone and speaker devices for live mode.
- [ ] Provide writable mounts for output and runtime audio if the container filesystem is read-only.

## D. Validation
- [ ] Run `python -m pytest`.
- [ ] Run `python main.py --diagnose-transcript "What is Greenwich Vietnam?" --diagnose-language en`.
- [ ] Start live mode with `python main.py --language adaptive`.
- [ ] Confirm startup mic calibration logs non-silent device health.
- [ ] Complete one wake -> request -> response cycle.
- [ ] Confirm logs contain stage timing entries.

## E. GitHub
- [ ] Initialize or connect this folder to `https://github.com/vinle4859/Live-Assistant-Demo.git`.
- [ ] Configure local Git identity as `vinle4859 <178093380+vinle4859@users.noreply.github.com>`.
- [ ] Inspect staged files before commit.
- [ ] Commit deployment handoff changes.
- [ ] Push branch `deployment-handoff`.
