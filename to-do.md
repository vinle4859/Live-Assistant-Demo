# To Do

## Event Readiness
- Run a full scripted-mode rehearsal on the exact event laptop, speaker, room, and network.
- Validate the event script with `python main.py --mode script --script-file event_script.txt --script-validate`.
- Pre-render script audio with `python main.py --mode script --script-file event_script.txt --script-no-play`.
- Rehearse controlled playback with `python main.py --mode script --script-file event_script.txt --script-manual-next`.

## Live Interview Mode
- Collect live session logs for missed transcripts, wrong language detection, clipped questions, and slow answers.
- Decide whether an interview persona is needed, and if so keep it as LLM prompt/profile behavior rather than operator-controlled capture.
- Add a compact session summary showing transcript, detected language, routing source, response text, and latency.

## Domain Cleanup
- Move remaining Greenwich-specific routing and fallback logic out of `VoicePipeline` into the domain profile layer.
- Add regression tests from real failed live transcripts before adding new correction rules.
- Keep transcript cheats context-guarded; avoid broad fuzzy replacements.

## Deployment Quality
- Re-test Google TTS, Edge TTS, and Vietnamese/UK English voices through event speakers.
- Evaluate replacing `playsound` if deployment hosts show playback instability.
- Keep `VOICE_LOOP_DOMAIN_PROFILE=none` available for non-Greenwich events.
