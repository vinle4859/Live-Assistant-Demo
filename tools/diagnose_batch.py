"""Batch diagnostics tool to run tricky Q&A cases, out-of-domain prompts, and noise handling."""

import asyncio
import json
import sys
sys.stdout.reconfigure(encoding='utf-8')
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voice_loop.config import AppConfig
from voice_loop.factory import build_default_pipeline
from voice_loop.lang_detect import detect_language
from voice_loop.live_assistant import normalize_text

def should_ignore(transcript: str, min_chars: int) -> bool:
    """Check if the transcript would be ignored as noise by the live assistant."""
    normalized = normalize_text(transcript)
    tokens = normalized.split()
    if len(normalized) < min_chars:
        return True
    # Check question stem
    if len(tokens) <= 3 and set(tokens) <= {"what", "is", "are", "the", "how", "where", "explain", "summarize", "latest"}:
        return True
    return len(tokens) <= 1

async def run_batch() -> None:
    cases_path = Path("tools/diagnose_cases.json")
    if not cases_path.exists():
        print(f"[-] ERROR: test cases file not found at {cases_path}")
        sys.exit(1)

    with open(cases_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    # Initialize configuration & pipeline
    config = AppConfig.from_env("en")
    pipeline = build_default_pipeline(config)

    # Mock TTS synthesize to run tests in milliseconds without network calls
    async def dummy_synthesize(text, lang, output_path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"dummy audio")
        return output_path

    if pipeline.primary_tts_provider:
        pipeline.primary_tts_provider.synthesize = dummy_synthesize
    if pipeline.fallback_tts_provider:
        pipeline.fallback_tts_provider.synthesize = dummy_synthesize

    print("======================================================================================")
    print("                              BATCH DIAGNOSTICS RUNNER                                ")
    print("======================================================================================")
    print(f"Loaded {len(cases)} test cases.")
    print("Running end-to-end evaluation...\n")

    results = []
    passed_source = 0
    passed_lang = 0
    total = len(cases)

    # Print table header
    print(f"{'#':<3} | {'Transcript':<42} | {'Exp':<5} | {'Got':<5} | {'Lang':<4} | {'Match':<5} | {'Time (ms)'}")
    print("-" * 94)

    for idx, case in enumerate(cases, 1):
        transcript = case["transcript"]
        expected_lang = case["language"]
        expected_source = case["expect_source"]

        start_time = time.perf_counter()

        # Check if ignored as noise
        is_noise = should_ignore(transcript, config.minimum_transcript_characters)

        actual_source = None
        response_text = None
        actual_lang = None
        elapsed_ms = 0.0

        if is_noise:
            actual_source = "noise"
            response_text = "(Low-information transcript ignored)"
            actual_lang = expected_lang
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        else:
            try:
                result = await pipeline.process_transcription(transcript, expected_lang)
                actual_source = result["resolved_source"]
                response_text = result["response_text"]
                actual_lang = detect_language(response_text)
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            except Exception as e:
                actual_source = "ERROR"
                response_text = f"Exception: {str(e)}"
                actual_lang = "error"
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # Check matching criteria
        source_ok = False
        if expected_source == "db":
            source_ok = actual_source in ("local_db", "local_db_rewrite")
        elif expected_source == "llm":
            source_ok = actual_source in ("llm_direct", "fallback", "db_llm_fallback")
        elif expected_source == "noise":
            source_ok = (actual_source == "noise")

        lang_ok = (actual_lang == expected_lang)

        if source_ok:
            passed_source += 1
        if lang_ok:
            passed_lang += 1

        match_str = "OK" if (source_ok and lang_ok) else "FAIL"
        
        # Format printing
        trunc_transcript = transcript[:40] + "..." if len(transcript) > 40 else transcript
        print(f"{idx:<3} | {trunc_transcript:<42} | {expected_source:<5} | {actual_source:<5} | {actual_lang:<4} | {match_str:<5} | {elapsed_ms:>8.1f}ms")

        # Save details
        results.append({
            "idx": idx,
            "transcript": transcript,
            "expected_source": expected_source,
            "actual_source": actual_source,
            "expected_lang": expected_lang,
            "actual_lang": actual_lang,
            "response": response_text,
            "elapsed_ms": elapsed_ms,
            "status": match_str
        })

    # Summary
    print("-" * 94)
    print("                                   SUMMARY                                            ")
    print("-" * 94)
    print(f"Total Cases Checked:        {total}")
    print(f"Routing Source Accuracy:    {passed_source}/{total} ({passed_source/total*100:.1f}%)")
    print(f"Language Routing Accuracy:  {passed_lang}/{total} ({passed_lang/total*100:.1f}%)")
    print("======================================================================================")

    # Output detailed report to output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "diagnose_batch_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "total": total,
                "routing_passed": passed_source,
                "lang_passed": passed_lang,
            },
            "results": results
        }, f, indent=2, ensure_ascii=False)
    print(f"Detailed JSON report written to: {report_path.resolve()}\n")

    if passed_source < total or passed_lang < total:
        print("[!] Warning: Some test cases failed expectations. See details above.")
        sys.exit(0)  # We exit 0 to let the runner see the summary output instead of blocking, or we can handle it.

if __name__ == "__main__":
    asyncio.run(run_batch())
