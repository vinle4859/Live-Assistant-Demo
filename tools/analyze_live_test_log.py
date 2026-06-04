"""Analyze live assistant logs for source routing and latency metrics.

The script parses pipeline timing lines, summarizes answer-source distribution,
and optionally compares observed source labels against an ordered prompt plan.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from statistics import median
from typing import Any

_STAGE_RE = re.compile(
    r"Stage timings \(ms\): "
    r"stt=(?P<stt>\S+) "
    r"db=(?P<db>\S+) "
    r"llm=(?P<llm>\S+) "
    r"tts=(?P<tts>\S+) "
    r"total=(?P<total>\S+) "
    r"source=(?P<source>\S+) "
    r"db_score=(?P<db_score>\S+) "
    r"db_mode=(?P<db_mode>\S+)"
)


def _parse_float(value: str) -> float | None:
    """Parse a numeric string into float, returning None on placeholders."""

    normalized = value.strip().lower()
    if normalized in {"n/a", "na", "none", "-"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _percentile(values: list[float], percentile: float) -> float:
    """Compute percentile with linear interpolation for small samples."""

    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]

    ordered = sorted(values)
    rank = (len(ordered) - 1) * max(0.0, min(100.0, percentile)) / 100.0
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return ordered[low]
    fraction = rank - low
    return ordered[low] + (ordered[high] - ordered[low]) * fraction


def _parse_turns(log_text: str) -> list[dict[str, Any]]:
    """Extract turn telemetry rows from log text."""

    turns: list[dict[str, Any]] = []
    for line in log_text.splitlines():
        match = _STAGE_RE.search(line)
        if not match:
            continue
        groups = match.groupdict()
        turns.append(
            {
                "turn_index": len(turns) + 1,
                "source": groups["source"],
                "stt_ms": _parse_float(groups["stt"]),
                "db_ms": _parse_float(groups["db"]),
                "llm_ms": _parse_float(groups["llm"]),
                "tts_ms": _parse_float(groups["tts"]),
                "total_ms": _parse_float(groups["total"]),
                "db_score": _parse_float(groups["db_score"]),
                "db_mode": groups["db_mode"],
            }
        )
    return turns


def _read_text_auto(path: Path) -> str:
    """Read text while handling the UTF-16 logs produced by PowerShell teeing."""

    raw_bytes = path.read_bytes()
    if raw_bytes.startswith(b"\xff\xfe"):
        return raw_bytes.decode("utf-16")
    if raw_bytes.startswith(b"\xfe\xff"):
        return raw_bytes.decode("utf-16")
    if raw_bytes.startswith(b"\xef\xbb\xbf"):
        return raw_bytes.decode("utf-8-sig")
    return raw_bytes.decode("utf-8", errors="ignore")


def _source_summary(turns: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a compact source distribution summary."""

    counts: dict[str, int] = {}
    for turn in turns:
        source = str(turn.get("source", "unknown"))
        counts[source] = counts.get(source, 0) + 1

    total = len(turns)
    percentages = {
        key: round((value / total) * 100.0, 2) if total else 0.0
        for key, value in counts.items()
    }
    return {
        "counts": counts,
        "percentages": percentages,
        "fallback_rate_percent": round((counts.get("fallback", 0) / total) * 100.0, 2) if total else 0.0,
        "local_db_rate_percent": round((counts.get("local_db", 0) / total) * 100.0, 2) if total else 0.0,
    }


def _latency_summary(turns: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute p50/p95 and mean metrics for each stage."""

    def summarize_field(field: str) -> dict[str, float]:
        values = [float(value) for value in (turn.get(field) for turn in turns) if isinstance(value, (int, float))]
        if not values:
            return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0}
        return {
            "count": float(len(values)),
            "mean": round(sum(values) / len(values), 2),
            "p50": round(median(values), 2),
            "p95": round(_percentile(values, 95), 2),
        }

    return {
        "stt_ms": summarize_field("stt_ms"),
        "db_ms": summarize_field("db_ms"),
        "llm_ms": summarize_field("llm_ms"),
        "tts_ms": summarize_field("tts_ms"),
        "total_ms": summarize_field("total_ms"),
    }


def _expected_alignment(turns: list[dict[str, Any]], plan_payload: dict[str, Any]) -> dict[str, Any]:
    """Compare expected and observed sources by ordered turn index."""

    prompts = plan_payload.get("prompts", [])
    if not isinstance(prompts, list):
        return {"evaluated_turns": 0, "matches": 0, "accuracy_percent": 0.0, "details": []}

    evaluated_turns = min(len(prompts), len(turns))
    matches = 0
    details: list[dict[str, Any]] = []

    for index in range(evaluated_turns):
        expected = str(prompts[index].get("expected_source", "")).strip()
        observed = str(turns[index].get("source", "")).strip()
        if expected == "llm_direct_or_fallback":
            matched = observed in {"llm_direct", "fallback"}
        else:
            matched = expected == observed
        if matched:
            matches += 1
        details.append(
            {
                "turn_index": index + 1,
                "prompt": str(prompts[index].get("prompt", "")),
                "expected_source": expected,
                "observed_source": observed,
                "matched": matched,
            }
        )

    accuracy = round((matches / evaluated_turns) * 100.0, 2) if evaluated_turns else 0.0
    return {
        "evaluated_turns": evaluated_turns,
        "matches": matches,
        "accuracy_percent": accuracy,
        "details": details,
    }


def main() -> None:
    """Parse arguments, analyze logs, and write a JSON report."""

    parser = argparse.ArgumentParser(description="Analyze live test logs for source and latency metrics")
    parser.add_argument("--log", required=True, help="Path to captured runtime log file")
    parser.add_argument(
        "--plan",
        default="output/live_test_topics.json",
        help="Optional path to ordered prompt plan JSON",
    )
    parser.add_argument(
        "--out",
        default="output/live_test_report.json",
        help="Output path for machine-readable summary JSON",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    log_path = (root / args.log).resolve() if not Path(args.log).is_absolute() else Path(args.log)
    plan_path = (root / args.plan).resolve() if not Path(args.plan).is_absolute() else Path(args.plan)
    out_path = (root / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)

    if not log_path.exists():
        raise SystemExit(f"Log file not found: {log_path}")

    turns = _parse_turns(_read_text_auto(log_path))
    source = _source_summary(turns)
    latency = _latency_summary(turns)
    report: dict[str, Any] = {
        "log_path": str(log_path),
        "turns_parsed": len(turns),
        "source_summary": source,
        "latency_summary": latency,
        "turns": turns,
    }

    if plan_path.exists():
        plan_payload = json.loads(plan_path.read_text(encoding="utf-8"))
        report["plan_path"] = str(plan_path)
        report["expected_alignment"] = _expected_alignment(turns, plan_payload)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Parsed turns: {len(turns)}")
    print(f"Source counts: {source['counts']}")
    print(f"Fallback rate: {source['fallback_rate_percent']}%")
    print(f"Total latency p50/p95: {latency['total_ms']['p50']}ms / {latency['total_ms']['p95']}ms")
    print(f"Wrote report: {out_path}")


if __name__ == "__main__":
    main()
