"""Build a mixed live-test prompt pack for DB and out-of-DB routing checks.

The generated pack is intended for manual voice testing. It contains:
- A JSON plan with ordered prompts and expected routing buckets.
- A CSV sheet template for quick turn-by-turn annotations during a live run.
It can operate with or without the separately generated variant dataset.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PromptCase:
    """One live-test prompt with routing expectation metadata."""

    turn_id: int
    category: str
    language: str
    prompt: str
    expected_source: str
    purpose: str


def _load_variant_records(variants_path: Path) -> dict[str, dict[str, Any]]:
    """Load variants JSON and return a source-id index."""

    payload = json.loads(variants_path.read_text(encoding="utf-8"))
    records = payload.get("records", []) if isinstance(payload, dict) else []
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        source_id = str(record.get("id", "")).strip()
        if source_id:
            indexed[source_id] = record
    return indexed


def _build_cases(records: dict[str, dict[str, Any]]) -> list[PromptCase]:
    """Create an ordered mixed prompt list for the live test run."""

    ordered_specs: list[tuple[str, str, str, str, str, str]] = [
        ("web-001", "en", "in_db", "local_db", "identity question baseline", "What is Greenwich Vietnam?"),
        ("web-004", "en", "in_db", "local_db", "majors retrieval baseline", "What majors does Greenwich Vietnam offer?"),
        ("web-006", "en", "in_db", "local_db", "duration retrieval baseline", "What is the duration of the program?"),
        ("web-014", "vi", "in_db", "local_db", "tuition retrieval baseline", "Học phí toàn khóa của Greenwich Việt Nam là bao nhiêu?"),
        ("web-017", "vi", "in_db", "local_db", "campus count retrieval baseline", "Greenwich Việt Nam có bao nhiêu cơ sở?"),
        ("web-009", "en", "in_db", "local_db", "IELTS requirement retrieval", "What are the graduation requirements? Is IELTS mandatory?"),
        ("web-015", "vi", "in_db", "local_db", "scholarship policy retrieval", "Greenwich Việt Nam có chính sách học bổng không?"),
        ("web-020", "en", "in_db", "local_db", "international opportunity retrieval", "Do students have opportunities for studying abroad or exchange programs?"),
        ("web-011", "vi", "in_db", "local_db", "admission requirements retrieval", "Điều kiện xét tuyển vào Greenwich Việt Nam là gì?"),
        ("web-012", "en", "in_db", "local_db", "English proficiency retrieval", "What is the required English proficiency level for admission?"),
        ("web-014", "en", "context_followup", "local_db", "ask before short follow-up on cost", "How much does the whole program cost at Greenwich Vietnam?"),
    ]

    cases: list[PromptCase] = []
    turn_id = 1
    for source_id, language, category, expected_source, purpose, prompt_override in ordered_specs:
        record = records.get(source_id)
        prompt = prompt_override.strip()
        if not prompt and record is not None:
            variant_key = f"variants_{language}"
            variants = record.get(variant_key, [])
            if isinstance(variants, list) and variants:
                for candidate in variants:
                    text = str(candidate).strip()
                    if text:
                        prompt = text
                        break
            if not prompt:
                prompt = str(record.get(f"question_{language}", "")).strip()
        if not prompt:
            continue
        cases.append(
            PromptCase(
                turn_id=turn_id,
                category=category,
                language=language,
                prompt=prompt,
                expected_source=expected_source,
                purpose=purpose,
            )
        )
        turn_id += 1

    # Follow-up prompts to probe deterministic context-linking behavior.
    follow_ups = [
        PromptCase(
            turn_id=turn_id,
            category="context_followup",
            language="en",
            prompt="How much per semester?",
            expected_source="local_db",
            purpose="short follow-up should inherit tuition topic anchor",
        ),
        PromptCase(
            turn_id=turn_id + 1,
            category="context_followup",
            language="en",
            prompt="How many majors?",
            expected_source="local_db",
            purpose="short follow-up should map to majors topic",
        ),
    ]
    cases.extend(follow_ups)
    turn_id += len(follow_ups)

    # Out-of-DB prompts to observe llm_direct vs fallback behavior.
    out_of_db_prompts: list[tuple[str, str]] = [
        ("en", "What is the weather in Hanoi right now?"),
        ("vi", "Hôm nay thời tiết ở Hà Nội thế nào?"),
        ("en", "Summarize the latest AI news this week in 2 bullets."),
        ("vi", "Viết một đoạn thơ ngắn về cà phê."),
        ("en", "Explain quantum entanglement in simple terms."),
        ("vi", "Ai thắng trận bóng gần nhất của Manchester United?"),
        ("en", "What are 3 healthy breakfast ideas?"),
        ("vi", "Cho tôi mẹo học tập trung trong 30 phút."),
    ]
    for language, prompt in out_of_db_prompts:
        cases.append(
            PromptCase(
                turn_id=turn_id,
                category="out_of_db",
                language=language,
                prompt=prompt,
                expected_source="llm_direct_or_fallback",
                purpose="outside curated Q&A should not route to local_db",
            )
        )
        turn_id += 1

    return cases


def _write_json_plan(cases: list[PromptCase], output_path: Path) -> None:
    """Write the JSON plan used as the source of truth for the session."""

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_prompts": len(cases),
        "notes": [
            "Run prompts in order to align per-turn analysis.",
            "expected_source=llm_direct_or_fallback means either source is acceptable.",
            "For context_followup rows, ask immediately after the previous related prompt.",
        ],
        "prompts": [
            {
                "turn_id": case.turn_id,
                "category": case.category,
                "language": case.language,
                "prompt": case.prompt,
                "expected_source": case.expected_source,
                "purpose": case.purpose,
            }
            for case in cases
        ],
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv_sheet(cases: list[PromptCase], output_path: Path) -> None:
    """Write a reviewer-friendly CSV for manual annotations during testing."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "turn_id",
                "category",
                "language",
                "prompt",
                "expected_source",
                "actual_source",
                "stt_quality",
                "notes",
            ]
        )
        for case in cases:
            writer.writerow(
                [
                    case.turn_id,
                    case.category,
                    case.language,
                    case.prompt,
                    case.expected_source,
                    "",
                    "",
                    "",
                ]
            )


def main() -> None:
    """Build the live-test prompt pack from generated variant records."""

    root = Path(__file__).resolve().parent.parent
    variants_path = root / "output" / "qa_variants_auto.json"
    records = _load_variant_records(variants_path) if variants_path.exists() else {}
    cases = _build_cases(records)
    if not cases:
        raise SystemExit("No prompt cases were built. Check prompt definitions in prepare_live_test_pack.py.")

    json_plan_path = root / "output" / "live_test_topics.json"
    csv_sheet_path = root / "output" / "live_test_sheet.csv"
    _write_json_plan(cases, json_plan_path)
    _write_csv_sheet(cases, csv_sheet_path)

    print(f"Generated {len(cases)} prompts at {json_plan_path}")
    print(f"Generated annotation sheet at {csv_sheet_path}")


if __name__ == "__main__":
    main()
