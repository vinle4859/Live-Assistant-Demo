# Current Live Quality Plan

## Task Order
1. Keep deployment automation deferred until live capture and answer quality are stable.
2. Keep request-ready cues short, cached, and suppressed after reprompts or weak captures.
3. Keep direct LLM first-pass success as the priority; do not rely on hidden second generation.
4. Keep local DB routing evidence-gated and conservative for long noisy captures.
5. Keep docs and `.env` comments aligned with current runtime defaults.

## Acceptance Checks
- Normal live runs leave no `response_*.mp3`, `wake_*.wav`, `utterance_*.wav`, `status_*.mp3`, or `barge_in_*.wav` artifacts behind.
- The user hears `Go ahead.` or `Bạn hỏi đi.` before the first post-wake request, then short follow-up cues for later turns.
- Slow direct LLM turns can play one thinking cue; local DB answers stay fast and cue-free.
- Bare `Greenwich Vietnam` / `Greenwich Việt Nam` asks for clarification instead of routing to the overview row.
- Weak or long noisy captures reprompt instead of forcing DB, context, or direct LLM answers.
- Streaming STT logs capture end reason, token progress, active duration, and interim/final selection.
- Stage timing logs include STT, DB, LLM, TTS, total latency, source, DB score, and DB mode.

## Live Test Checklist
1. Run `.\.venv\Scripts\python.exe main.py --language adaptive`.
2. Use the shortened prompts in `output/live_test_topics.json`.
3. Confirm ready cues are audible before speaking.
4. Track empty captures, weak-progress captures, wrong DB/context routing, LLM-direct latency, and answer truncation.
5. Review the timestamped log under `output/live_sessions/`.

## Current Defaults
- Request ready cue: speech mode with rotating EN/VI prompts.
- First request cue: `Go ahead.` / `Bạn hỏi đi.`
- Thinking cue: enabled after `1.2s` for direct LLM only.
- Direct LLM live budget: `8s` normally, `10s` for current/search-style prompts; `.env` LLM timeout is only the ceiling.
- Primary TTS live fallback path: primary capped at `5s`; fallback capped at `5s`.
- Streaming request tail: `2000ms` local silence.
- STT no-progress stop: `4.5s`.
- STT weak-progress stop: `6.0s` with fewer than 3 stable tokens.
- Barge-in: off.
- No local or Google standalone denoise path is active; model quality experiments should use STT model/location changes.

## Possible STT Model Increment
- Current default: `VOICE_LOOP_STT_MODEL=` empty, so Google STT uses the provider default `latest_short`.
- First VI/EN live-recognition candidate remains `chirp_2`, but it is not a direct `.env` swap in the current pipeline: regional v2 streaming may need explicit regional endpoint support, and fixed-window v1 rejects Chirp model names.
- Accuracy-first candidate remains `chirp_3`, but test it only after the provider path is updated for v2/regional model compatibility.
- Do not add local denoise or heavy Python audio processing unless profiling proves the current lightweight VAD is insufficient.
