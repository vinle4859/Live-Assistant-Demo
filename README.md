# Voice-to-Voice Pipeline

Headless Python service for English and Vietnamese voice-to-voice interactions.

## Docker deployment handoff

This project is a live microphone CLI assistant, not an HTTP service. The Docker runtime entrypoint is:

```bash
python main.py --language adaptive
```

For container smoke validation without microphone access:

```bash
python main.py --diagnose-transcript "What is Greenwich Vietnam?" --diagnose-language en
```

Docker integration requirements:
- Include `main.py`, `voice_loop/`, `requirements.txt`, `.env.example`, deployment docs, and `data/knowledge_base.sqlite3`.
- Exclude `.env`, `.venv/`, `output/`, tests/tools from the runtime image, runtime audio, caches, and credential files.
- Provide Google Cloud credentials through mounted ADC/service-account JSON or Workload Identity.
- Provide microphone and speaker passthrough for live mode.
- Provide writable `VOICE_LOOP_OUTPUT_DIR` and `data/live_audio/`.

See [DOCKER_HANDOFF.md](DOCKER_HANDOFF.md) and [DEPLOYMENT.md](DEPLOYMENT.md) for the full handoff contract.

## Windows runtime setup (required)

Use CPython 3.12 via the Python launcher. In this workspace, plain `python` may resolve to MSYS Python and break dependency installation.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python -m pip install -r requirements.txt
```

Run commands with the venv interpreter:

```powershell
.\.venv\Scripts\python main.py --language adaptive
.\.venv\Scripts\python main.py --language en
.\.venv\Scripts\python main.py --language vi
$env:VOICE_LOOP_LOG_LEVEL='DEBUG'; .\.venv\Scripts\python main.py --language adaptive
```

Default architecture:
- Google Cloud Speech-to-Text for STT
- Q&A retrieval supports `lexical`, `vector` (semantic proxy), and `hybrid` confidence-gated local matching
- Gemini-backed synthesis on Vertex AI when available, otherwise heuristic synthesis
- Default response routing: `local_db -> direct_LLM -> hard fallback`
- Default language policy: dual-input adaptive (VI/EN input per turn) with single-output replies
- Google Cloud Text-to-Speech for output, with edge-tts as fallback

Current response-source labels in logs:
- `local_db`: local Q&A match score is at or above `VOICE_LOOP_QA_CONFIDENCE_LOW`.
- `llm_direct`: local match misses confidence gate, but direct LLM generation returns a valid answer.
- `fallback`: local match misses confidence gate and direct LLM is skipped or returns no valid answer.

Token guard (`VOICE_LOOP_LLM_DIRECT_MIN_QUERY_TOKENS`):
- Purpose: skip direct LLM calls for very short/noisy transcripts.
- Rule: if normalized token count is below this value, the turn routes directly to `fallback`.
- With default value `1`, any non-empty tokenized query can attempt `llm_direct` after a local miss.
- Typical hard-fallback cases:
	- Transcript is empty/noisy after normalization.
	- Token count is below guard threshold.
	- LLM call times out/throws an exception.
	- LLM returns empty text or the same hard fallback phrase.

## Run

```bash
python main.py --language adaptive
```

Live wake-word mode is the default and only runtime mode.

```bash
python main.py --language en
python main.py --language vi
```

For Google STT/TTS, you must configure Application Default Credentials (ADC):

```powershell
gcloud auth application-default login
```

Useful environment variables:
- `VOICE_LOOP_DB_PATH` - SQLite database path
- `VOICE_LOOP_OUTPUT_DIR` - synthesized audio output directory
- `VOICE_LOOP_PROVIDER_TIMEOUT_SECONDS` - provider timeout in seconds
- `VOICE_LOOP_LLM_TIMEOUT_SECONDS` - LLM-only timeout in seconds for direct-answer generation
- `VOICE_LOOP_LOG_LEVEL` - logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`)
- `VOICE_LOOP_WAKE_WORD` - wake phrase for live mode
- `VOICE_LOOP_WAKE_ALIASES` - optional comma-separated full wake aliases (for example `hello lemon, hey leman`)
- `VOICE_LOOP_WAKE_ACK_MODE` - wake confirmation mode: `none`, `beep`, `speech`, or `adaptive`
- `VOICE_LOOP_WAKE_ACK_ADAPTIVE_SPEAK_ON_WAKE` - in adaptive mode, speak the wake prompt immediately after beep
- `VOICE_LOOP_WAKE_ACK_BEEP_FREQUENCY` - wake beep frequency in Hz
- `VOICE_LOOP_WAKE_ACK_BEEP_DURATION_MS` - wake beep duration in milliseconds
- `VOICE_LOOP_WAKE_ACK_PROMPT_TEXT_EN` - spoken wake prompt text for English
- `VOICE_LOOP_WAKE_ACK_PROMPT_TEXT_VI` - spoken wake prompt text for Vietnamese
- `VOICE_LOOP_REQUEST_READY_CUE_MODE` - request-listening cue mode: `none`, `beep`, or `speech`
- `VOICE_LOOP_REQUEST_READY_FIRST_TEXT_EN` - first post-wake English cue before request capture
- `VOICE_LOOP_REQUEST_READY_FIRST_TEXT_VI` - first post-wake Vietnamese cue before request capture
- `VOICE_LOOP_REQUEST_READY_TEXTS_EN` - comma-separated English request-ready prompts
- `VOICE_LOOP_REQUEST_READY_TEXTS_VI` - comma-separated Vietnamese request-ready prompts
- `VOICE_LOOP_REQUEST_READY_CACHE` - cache short request-ready TTS prompts for the current process
- `VOICE_LOOP_THINKING_CUE_ENABLED` - play one transition cue for slow direct-LLM turns
- `VOICE_LOOP_THINKING_CUE_DELAY_SECONDS` - delay before the direct-LLM thinking cue is played
- `VOICE_LOOP_THINKING_TEXTS_EN` - comma-separated English thinking prompts
- `VOICE_LOOP_THINKING_TEXTS_VI` - comma-separated Vietnamese thinking prompts
- `VOICE_LOOP_WAKE_WINDOW_SECONDS` - wake capture chunk length (seconds)
- `VOICE_LOOP_UTTERANCE_SECONDS` - conversation capture chunk length (seconds)
- `VOICE_LOOP_UTTERANCE_MIN_RMS` - minimum RMS energy to treat a turn as speech
- `VOICE_LOOP_UTTERANCE_MIN_PEAK` - minimum peak energy to treat a turn as speech
- `VOICE_LOOP_MIN_TRANSCRIPT_CHARACTERS` - minimum normalized transcript length before accepting fallback replies
- `VOICE_LOOP_ENABLE_STREAMING_STT` - enable live streaming STT with server endpointing for request turns
- `VOICE_LOOP_STREAMING_CHUNK_DURATION_MS` - microphone chunk size sent to streaming STT (milliseconds)
- `VOICE_LOOP_STREAMING_SPEECH_START_TIMEOUT_SECONDS` - streaming timeout before speech starts
- `VOICE_LOOP_STREAMING_SPEECH_END_TIMEOUT_SECONDS` - streaming timeout after speech ends
- `VOICE_LOOP_STREAMING_LOCAL_SPEECH_END_MS` - local silence window that can end streaming before surrounding noise stretches a turn
- `VOICE_LOOP_STREAMING_MAX_ACTIVE_SECONDS` - max active speech duration after local speech start
- `VOICE_LOOP_STREAMING_NO_PROGRESS_SECONDS` - stop active capture when STT produces no interim/final text by this time
- `VOICE_LOOP_STREAMING_WEAK_PROGRESS_SECONDS` - stop active capture when STT text is still too weak by this time
- `VOICE_LOOP_STREAMING_WEAK_PROGRESS_MIN_TOKENS` - minimum stable token count for weak-progress capture to continue
- `VOICE_LOOP_PREROLL_ENABLED` - prepend a short initial mic buffer to live request streaming
- `VOICE_LOOP_PREROLL_MS` - request pre-roll duration in milliseconds
- `VOICE_LOOP_STARTUP_CALIBRATION_ENABLED` - run a short startup mic/environment diagnostic
- `VOICE_LOOP_STARTUP_CALIBRATION_SECONDS` - startup calibration capture duration
- `VOICE_LOOP_REQUEST_MAX_IGNORED_TURNS` - max ignored turns in request mode before auto-return to wake mode
- `VOICE_LOOP_REQUEST_MAX_TURNS` - max request turns per wake session before auto-return to wake mode
- `VOICE_LOOP_REQUEST_IDLE_TIMEOUT_SECONDS` - inactivity timeout in request mode; resets after each accepted turn or assistant response
- `VOICE_LOOP_REQUEST_MAX_SESSION_SECONDS` - absolute safety ceiling in request mode before auto-return to wake mode
- `VOICE_LOOP_SAMPLE_RATE` - microphone capture sample rate
- `VOICE_LOOP_INPUT_DEVICE_INDEX` - optional PyAudio input device index override
- `VOICE_LOOP_STT_PROVIDER` - `google`
- `VOICE_LOOP_STT_MODEL` - optional Google STT model override (leave empty for auto)
- `VOICE_LOOP_STT_LOCATION` - Google STT location for streaming recognizer path (use `global` by default)
- `VOICE_LOOP_STT_HINT_PHRASES` - comma-separated STT phrase hints (for wake words, commands)
- `VOICE_LOOP_TRANSCRIPT_CHEATS` - semicolon-separated transcript corrections, format: `wrong=correct|context1,context2` (context optional)
- `VOICE_LOOP_TTS_PROVIDER` - primary TTS provider (`google` or `demo`); edge-tts is used as internal fallback
- `VOICE_LOOP_LLM_DIRECT_MIN_QUERY_TOKENS` - minimum token count required before direct LLM fallback runs
- `VOICE_LOOP_DEBUG_LLM_TEXT` - log raw/compacted LLM excerpts for diagnosis
- `VOICE_LOOP_DEBUG_AUDIO_IO` - log detailed TTS/playback file diagnostics
- `VOICE_LOOP_DEBUG_STT_STREAM` - reserve verbose STT stream diagnostics for provider-level tuning
- `--language adaptive|en|vi` - public CLI language profile; `adaptive` detects EN/VI per turn and replies in the detected language, while `en` and `vi` force fixed input/output
- `VOICE_LOOP_LANGUAGE_MODE` - input-language policy (`adaptive` or `fixed`)
- `VOICE_LOOP_OUTPUT_LANGUAGE_MODE` - output-language policy (`auto` or `fixed`)
- `VOICE_LOOP_OUTPUT_LANGUAGE_FIXED` - fixed output language (`en` or `vi`) when output mode is fixed
- `VOICE_LOOP_ENABLE_BILINGUAL_OUTPUT` - allow dual-language responses when bilingual mode is requested
- `VOICE_LOOP_LANGUAGE_SWITCH_MIN_CONFIDENCE` - minimum confidence required before adaptive language switching
- `VOICE_LOOP_LANGUAGE_SWITCH_STICKY_TURNS` - number of turns to keep output language after a switch
- `VOICE_LOOP_LANGUAGE_OVERRIDE_COMMANDS` - enable runtime commands for en-only/vi-only/auto/bilingual modes
- `VOICE_LOOP_QA_RETRIEVAL_MODE` - Q&A retrieval mode (`lexical`, `vector`, `hybrid`)
- `VOICE_LOOP_QA_LEXICAL_TOP_K` - lexical candidate shortlist size for Q&A retrieval
- `VOICE_LOOP_QA_VECTOR_TOP_K` - vector candidate size for Q&A retrieval
- `VOICE_LOOP_QA_CONFIDENCE_LOW` - Q&A low-confidence threshold (route to direct LLM fallback)
- `VOICE_LOOP_QA_SEED_AUTO_SYNC` - optional import of curated QA JSON into SQLite when curated rows are missing (recommended `false` for deployment)
- `VOICE_LOOP_QA_SEED_JSON_PATH` - curated QA JSON path used only when auto-sync is enabled (default `data/qa_seed_vi_en.json`)
- `VOICE_LOOP_CONTEXT_LINK_ENABLED` - enable deterministic short follow-up context linking
- `VOICE_LOOP_CONTEXT_LINK_MAX_TURN_GAP` - number of turns to keep a topic anchor alive
- `VOICE_LOOP_CONTEXT_LINK_SHORT_QUERY_MAX_TOKENS` - max tokens considered a short follow-up query
- `VOICE_LOOP_CONTEXT_LINK_MIN_SCORE_DELTA` - minimum retrieval score improvement required to accept context expansion
- `GOOGLE_CLOUD_PROJECT` - Google Cloud project ID for Vertex AI Gemini
- `GOOGLE_CLOUD_LOCATION` - Vertex AI region, defaults to `us-central1`
- `GEMINI_MODEL` - optional, defaults to `gemini-2.0-flash`

## Live Mixed Test Workflow

Prepare a mixed test pack (in-DB, context follow-up, out-of-DB) and annotation sheet:

```powershell
.\.venv\Scripts\python tools\prepare_live_test_pack.py
```

Generated files:
- `output/live_test_topics.json` (ordered prompts with expected routing bucket)
- `output/live_test_sheet.csv` (manual annotation template)

Run live mode with log capture:

```powershell
.\.venv\Scripts\python main.py 2>&1 | Tee-Object -FilePath output/live_test_session.log
```

Analyze routing and latency after the session:

```powershell
.\.venv\Scripts\python tools\analyze_live_test_log.py --log output/live_test_session.log --plan output/live_test_topics.json --out output/live_test_report.json
```

Report outputs include:
- source distribution (`local_db`, `llm_direct`, `fallback`)
- fallback/local-db rates
- stage latency summaries (mean, p50, p95 for STT/DB/LLM/TTS/total)
- expected-vs-observed alignment by ordered turn

## Env file

Create a root-level `.env` file from [`.env.example`](.env.example). The app loads `.env` automatically at startup. If `.env` is missing, it falls back to `.env.example`.

For live-mode tuning, prefer editing `.env` (wake phrase, mic device index, sample rate, wake/utterance durations) instead of passing many CLI flags.

If `VOICE_LOOP_STT_HINT_PHRASES` is not set, the app derives wake-word hints automatically from `VOICE_LOOP_WAKE_WORD` (for example, `hey lemon` and `lemon`).
If `VOICE_LOOP_WAKE_ALIASES` is not set, the app reuses `VOICE_LOOP_STT_HINT_PHRASES` values (except the primary wake phrase) as wake aliases.
For repeated STT mixups on domain names (for example `remix` when user means `greenwich`), use `VOICE_LOOP_TRANSCRIPT_CHEATS` with context terms instead of broad STT hint tuning.
Operator guide: see [`TRANSCRIPT_CHEATS.md`](TRANSCRIPT_CHEATS.md).
When `VOICE_LOOP_ENABLE_STREAMING_STT=true`, live request turns use Google streaming STT endpointing so capture can end when speech ends instead of always waiting for the full utterance window.
Live mode now writes a timestamped log automatically under `output/live_sessions/`, so `Tee-Object` is optional for manual test capture.
For model-quality experiments, prefer changing `VOICE_LOOP_STT_MODEL` and `VOICE_LOOP_STT_LOCATION` over adding local audio denoise. Current default leaves `VOICE_LOOP_STT_MODEL=` empty, which uses the provider default `latest_short`.

### Live command phrases

The assistant is wake-gated. Say `Hey Lemon` or a configured full alias such as `Hello Lemon` to enter request mode.

To return to wake-word listening mode, say one of:
- `stop listening`
- `go to sleep`
- `sleep now`
- `exit`
- `quit`

The app prints the active wake phrase, aliases, exit phrases, language mode, selected input device, and barge-in mode at startup.

### English-only fallback profile

If mixed VI/EN routing is not stable enough for a deployment test, force English input/output from the CLI:

```powershell
.\.venv\Scripts\python main.py --language en
```

Equivalent `.env` profile:

```env
VOICE_LOOP_LANGUAGE_MODE=fixed
VOICE_LOOP_OUTPUT_LANGUAGE_MODE=fixed
VOICE_LOOP_OUTPUT_LANGUAGE_FIXED=en
```

Fixed input mode removes the secondary VI recognition language from Google STT requests, and fixed output mode keeps every answer and cue in English.

### Live capture and routing notes

Request mode now speaks a short ready cue before opening the microphone. Start speaking after that cue, not during assistant answer playback.
Keep `VOICE_LOOP_PREROLL_ENABLED=true` for live tests so request streaming includes the configured initial buffer.

Streaming STT uses local VAD plus STT progress stops. If logs show `no_stt_progress` or `weak_stt_progress`, treat the turn as a capture/timing issue rather than a knowledge-base miss.
Noisy or partially recognized text should reprompt before DB/LLM routing and should not update the context anchor.

### Live wake-word troubleshooting

If wake-word detection fails, set `VOICE_LOOP_LOG_LEVEL=DEBUG` and check:
- `rms`/`peak` near zero (e.g., `rms=0.5`, `peak=1.0`) means the mic stream is effectively silent -> wrong/muted input device is the top issue.
- Repeated empty wake transcripts with non-silent audio usually means phrase timing/noise issues; increase `VOICE_LOOP_WAKE_WINDOW_SECONDS` and set `VOICE_LOOP_STT_HINT_PHRASES`.

For production on GKE, use Workload Identity so the app gets Application Default Credentials from the pod identity. Do not set `GOOGLE_APPLICATION_CREDENTIALS` in that deployment path. For Gemini on Vertex AI, set `GOOGLE_CLOUD_PROJECT` and optionally `GOOGLE_CLOUD_LOCATION` and `GEMINI_MODEL`.

## Build Sample QA Dataset (VN/EN)

This section is for data-prep workflows only and is not part of deployment runtime.

Generate a reviewable sample Q/A dataset from the Greenwich FAQ seed and optional PDF extraction/generation from `data/Q&A`:

```powershell
.\.venv\Scripts\python tools\build_sample_qa_dataset.py --include-pdf --translate-en
```

Output file:
- `output/qa_sample_vi_en.json`

For deployment, use a prebuilt `data/knowledge_base.sqlite3` and keep `VOICE_LOOP_QA_SEED_AUTO_SYNC=false`.

Web FAQ seed:
- `data/web_faq_qa_vi.json` (Vietnamese question+answer pairs exported from the live accordion page)

Useful flags:
- `--max-pdf-questions 12` to increase per-PDF candidate count.
- `--disable-pdf-llm-fallback` to keep PDF extraction strictly regex-based (no LLM question generation from PDF text).
- `--disable-pdf-answer-generation` to skip LLM answer generation for PDF-derived questions.
- `--translation-batch-size 4` to reduce per-call translation payload size when `--translate-en` is enabled.
- `--output output/custom_file.json` to change destination.

Output schema fields include both Vietnamese and English Q/A values:
- `question_vi`, `answer_vi`, `question_en`, `answer_en`

## Google Cloud setup with Workload Identity

Use this when deploying to GKE.

1. Enable the required APIs in the Google Cloud project: Speech-to-Text, Text-to-Speech, and Vertex AI.
2. Create a Google service account for the app, for example `voice-loop-sa`.
3. Grant that service account the app-level roles it actually needs. A practical starting set is:
	- `roles/speech.client`
	- `roles/texttospeech.user`
	- `roles/aiplatform.user`
	- `roles/serviceusage.serviceUsageConsumer`

   In the IAM console, search for the exact labels `Cloud Speech Client`, `Cloud Text-to-Speech User`, and `Vertex AI User`. Do not grant `Vertex AI Service Agent` or `Cloud Speech-to-Text Service Agent` to the application service account. Those are Google-managed service-agent identities for the platform itself, not for your workload. `Aiplatform Editor` is broader than needed for this app and should be avoided unless you have a separate reason to use it.
4. Create or use a GKE cluster with Workload Identity enabled. The cluster must use a workload pool like `PROJECT_ID.svc.id.goog`.
5. Create a Kubernetes service account for the workload, then bind it to the Google service account with the `iam.gke.io/gcp-service-account` annotation and the `roles/iam.workloadIdentityUser` IAM binding.
6. Deploy the app using that Kubernetes service account.
7. Leave `GOOGLE_APPLICATION_CREDENTIALS` unset. The Google client libraries will use ADC from Workload Identity automatically.

### Detailed GKE / Workload Identity steps

Below is the full flow with the important objects called out.

1. Set your project variables locally:

```bash
export PROJECT_ID="your-project-id"
export PROJECT_NUMBER="$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')"
gcloud config set project "$PROJECT_ID"
```

2. Enable the APIs:

```bash
gcloud services enable speech.googleapis.com texttospeech.googleapis.com aiplatform.googleapis.com
```

3. Create the Google service account for the app:

```bash
gcloud iam service-accounts create voice-loop-sa \
	--project "$PROJECT_ID" \
	--display-name "Voice loop application service account"
```

4. Grant the app service account the minimal IAM roles:

```bash
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
	--member="serviceAccount:voice-loop-sa@$PROJECT_ID.iam.gserviceaccount.com" \
	--role="roles/speech.client"

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
	--member="serviceAccount:voice-loop-sa@$PROJECT_ID.iam.gserviceaccount.com" \
	--role="roles/texttospeech.user"

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
	--member="serviceAccount:voice-loop-sa@$PROJECT_ID.iam.gserviceaccount.com" \
	--role="roles/aiplatform.user"

gcloud projects add-iam-policy-binding "$PROJECT_ID" \
	--member="serviceAccount:voice-loop-sa@$PROJECT_ID.iam.gserviceaccount.com" \
	--role="roles/serviceusage.serviceUsageConsumer"
```

5. Create a GKE cluster with Workload Identity enabled if you do not already have one:

```bash
gcloud container clusters create-auto voice-loop-cluster \
	--project "$PROJECT_ID" \
	--region us-central1 \
	--workload-pool="$PROJECT_ID.svc.id.goog"
```

If you are using an existing cluster, make sure Workload Identity is enabled on that cluster and that the workload pool matches `PROJECT_ID.svc.id.goog`.

6. Get cluster credentials so kubectl points at the right cluster:

```bash
gcloud container clusters get-credentials voice-loop-cluster \
	--project "$PROJECT_ID" \
	--region us-central1
```

7. Create a Kubernetes service account in the namespace where the app will run:

```bash
kubectl create serviceaccount voice-loop-ksa -n default
```

8. Bind the Kubernetes service account to the Google service account:

```bash
gcloud iam service-accounts add-iam-policy-binding \
	voice-loop-sa@$PROJECT_ID.iam.gserviceaccount.com \
	--project "$PROJECT_ID" \
	--role="roles/iam.workloadIdentityUser" \
	--member="serviceAccount:$PROJECT_ID.svc.id.goog[default/voice-loop-ksa]"
```

9. Annotate the Kubernetes service account so GKE knows which Google service account to use:

```bash
kubectl annotate serviceaccount voice-loop-ksa -n default \
	iam.gke.io/gcp-service-account=voice-loop-sa@$PROJECT_ID.iam.gserviceaccount.com
```

10. Deploy the app using that Kubernetes service account. In your pod spec, set:

```yaml
serviceAccountName: voice-loop-ksa
```

11. Leave `GOOGLE_APPLICATION_CREDENTIALS` unset inside the container. ADC will come from Workload Identity automatically.

12. Test the identity from inside the pod:

```bash
python -c "from google.auth import default; creds, project = default(); print(project); print(type(creds).__name__)"
```

If that prints a project and credential type instead of raising an error, ADC is wired correctly.

13. Keep using `gcloud` and `kubectl` for the entire setup. If you need to inspect the assigned roles later, use IAM & Admin -> IAM, not the Roles page. The Roles page is only for browsing predefined roles.

### What each role does

- `roles/speech.client` / `Cloud Speech Client`: lets the app call Speech-to-Text recognition APIs.
- `roles/texttospeech.user` / `Cloud Text-to-Speech User`: lets the app synthesize audio from text.
- `roles/aiplatform.user` / `Vertex AI User`: lets the app call Gemini on Vertex AI.
- `roles/serviceusage.serviceUsageConsumer`: lets the workload consume enabled Google APIs from the project.

Those four roles are the application permissions. They are separate from the Google-managed service-agent identities that you may see in the console.

Important: the screenshot is the IAM & Admin -> Roles page. That page is for listing predefined roles and creating custom roles. To actually grant a role to your app service account, use `gcloud projects add-iam-policy-binding`.

If the console search does not show `Cloud Text-to-Speech User`, use the exact role id `roles/texttospeech.user` from `gcloud`. The role is the application permission you want; do not substitute service-agent roles.

If you are not on GKE, the safer alternative is a Google-managed runtime identity such as Cloud Run or a VM with a service account attached. In that case, ADC still works, but it is not Workload Identity in the GKE sense.
