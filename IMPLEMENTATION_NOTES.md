# Implementation Notes and Next Steps

## Current Implementation

The project is a headless Python CLI assistant with three runtime modes:

- `live`: wake-word conversational mode with STT, DB/LLM routing, and TTS playback.
- `diagnose`: transcript-only routing diagnosis without microphone capture.
- `script`: TTS-only event playback for prepared lines.

The runtime package is under `voice_loop/`. `main.py` now dispatches into mode-specific runners so future modes do not keep expanding one live-only path.

## Important Decisions

- Scripted speech stays in this repo because it shares TTS providers, output paths, logging, deployment config, and event rehearsal workflow with live mode.
- Script mode intentionally does not use STT, the Q&A database, or LLM generation. It is deterministic event playback, not a conversational assistant.
- Script mode pre-renders all selected lines before playback. This makes TTS/network failures happen before the live moment instead of mid-script.
- Script output is written under `output/script_sessions/<timestamp>/` so operators can inspect and replay the exact generated MP3 files.
- Live mode remains autonomous. No push-to-talk/operator-controlled interview mode has been added because that would change the assistant from an interviewable entity into a controlled voice command tool.
- Domain-specific Greenwich STT hints now live behind `VOICE_LOOP_DOMAIN_PROFILE=greenwich|none`. This is only the first boundary; deeper Greenwich routing rules still exist in `VoicePipeline`.
- `audioop` has been removed. Stereo 16-bit PCM WAV input is now converted to mono with a local stdlib-only averaging helper.

## Tradeoffs and Options

- `audioop-lts` was not added because the project only needed one simple `audioop.tomono(..., 2, 0.5, 0.5)` behavior. Local code avoids another dependency and supports Python 3.13.
- Script files are plain one-line-per-utterance text for now. A richer script format could add `[en]`, `[vi]`, `[pause 2]`, or cue labels later, but that would add parsing and operator complexity.
- `--script-manual-next` is useful for rehearsals and stage operation, but pre-rendered `--script-no-play` is still the safest first check.
- Persona work for interviews is still optional. A persona could improve interview tone, but it should be scoped as LLM prompt/profile behavior, not as operator-controlled capture.
- Domain profiles currently affect provider hints only. Moving all Greenwich-specific routing and fallback rules into profiles is a larger refactor and should be done with regression tests from live logs.

## Next Tasks

1. Run a full event rehearsal with the exact laptop, speaker, mic, internet, and room noise.
2. Collect live logs for failed transcripts, especially missed or distorted “Greenwich” captures.
3. Move remaining Greenwich-specific routing/fallback logic out of `VoicePipeline` into a domain profile layer.
4. Decide whether interview personas are needed:
   - keep current neutral assistant persona if the interviewer asks factual questions;
   - add a Greenwich admissions persona if tone and persuasion matter;
   - avoid persona changes if consistency and low risk matter more.
5. Add optional script metadata only if real scripts need bilingual line switching or pauses.
6. Improve live observability with a compact session summary: transcript, language, source, response, and latency.
7. Revisit audio playback dependency risk. `playsound` is simple but host-dependent; a more actively maintained playback backend may be needed after more deployment testing.

## Verification Baseline

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Expected current result: `207 passed`.
