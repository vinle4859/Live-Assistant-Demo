"""Generate auto-approved question variants for curated Q&A records.

This tool expands the canonical Q&A set with speech-style question variants that
preserve source IDs. It favors semantic coverage (added/removed function words,
spoken phrasing, concise forms) while applying strict post-filters to reduce
hallucination risk and near-duplicate churn.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import unicodedata
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


def load_env_file(env_path: Path) -> None:
    """Load .env key-value pairs into process env if keys are not already set."""

    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def normalize_text(text: str) -> str:
    """Normalize text to lowercase ASCII token stream for robust dedupe checks."""

    lowered = text.lower().replace("đ", "d")
    ascii_like = unicodedata.normalize("NFD", lowered).encode("ascii", "ignore").decode("ascii")
    return " ".join(re.findall(r"[a-z0-9]+", ascii_like))


def similarity(a: str, b: str) -> float:
    """Return normalized similarity ratio between two strings."""

    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


def strip_question_suffix(question: str) -> str:
    """Remove common terminal punctuation before templated rewrites."""

    return question.strip().rstrip(" ?.!:;")


def enforce_question_shape(candidate: str, original: str, language: str) -> str:
    """Normalize punctuation so variants stay question-like and concise."""

    text = candidate.strip()
    if not text:
        return text

    if language == "vi":
        if not text.endswith("?"):
            text = text.rstrip(".!;:") + "?"
    else:
        if not text.endswith("?"):
            text = text.rstrip(".!;:") + "?"

    # Keep capitalization natural without forcing title-casing.
    if original and original[0].isupper() and text:
        text = text[0].upper() + text[1:]
    return text


def semantic_anchor_variants(question: str, language: str) -> list[str]:
    """Create domain-safe enrichment variants for high-frequency entity patterns."""

    normalized = normalize_text(question)
    anchors: list[str] = []

    if "greenwich" in normalized and ("vietnam" in normalized or "viet nam" in normalized):
        if language == "vi":
            if any(phrase in normalized for phrase in ("la gi", "gioi thieu", "thong tin", "ve greenwich")):
                anchors.extend(
                    [
                        "Cho toi biet ve truong dai hoc Greenwich Viet Nam?",
                        "Truong dai hoc Greenwich Viet Nam la gi?",
                    ]
                )
        else:
            if any(phrase in normalized for phrase in ("what is", "tell me about", "about greenwich", "overview")):
                anchors.extend(
                    [
                        "Tell me about Greenwich Vietnam university?",
                        "What can you share about Greenwich Vietnam university?",
                    ]
                )

    if any(token in normalized for token in ("hoc phi", "tuition", "fee", "cost")):
        if language == "vi":
            anchors.extend(
                [
                    "Hoc phi la bao nhieu?",
                    "Chi phi chuong trinh la bao nhieu?",
                ]
            )
        else:
            anchors.extend(
                [
                    "How much is the tuition fee?",
                    "What is the program cost?",
                ]
            )

    return anchors


def deterministic_variants(question: str, language: str, target_count: int) -> list[str]:
    """Create deterministic paraphrase candidates without external model calls."""

    stem = strip_question_suffix(question)
    if not stem:
        return []

    if language == "vi":
        templates = (
            "Cho toi biet {stem}?",
            "Toi muon hoi {stem}?",
            "Ban co the cho biet {stem}?",
            "Cho hoi {stem}?",
            "Thong tin ve {stem} la gi?",
            "Noi ro hon ve {stem} duoc khong?",
        )
    else:
        lower_stem = stem[0:1].lower() + stem[1:] if stem else stem
        templates = (
            "Could you tell me {stem}?",
            "Can you tell me about {stem}?",
            "I would like to know {stem}.",
            "Please share details about {stem}.",
            "What should I know about {stem}?",
            "Could you explain {stem}?",
        )
        stem = lower_stem

    original_normalized = normalize_text(question)
    variants: list[str] = []
    seen: set[str] = {original_normalized}

    for template in templates:
        candidate = enforce_question_shape(template.format(stem=stem).strip(), question, language)
        normalized = normalize_text(candidate)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        variants.append(candidate)
        if len(variants) >= target_count:
            break

    return variants


def parse_json_response_array(text: str) -> list[dict[str, Any]]:
    """Parse a model response that may include fenced JSON markup."""

    cleaned = (text or "").strip()
    if not cleaned:
        return []

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

    payload = json.loads(cleaned)
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def select_variants(
    original_question: str,
    language: str,
    candidates: list[str],
    target_count: int,
) -> list[str]:
    """Apply strict safety and diversity filters to candidate variants."""

    normalized_original = normalize_text(original_question)
    original_has_digits = bool(re.search(r"\d", original_question))
    selected: list[str] = []
    seen: set[str] = {normalized_original}
    original_tokens = set(normalized_original.split())
    focus_terms = _focus_terms(original_tokens, language)

    for raw in candidates:
        candidate = enforce_question_shape(str(raw), original_question, language)
        normalized = normalize_text(candidate)
        candidate_tokens = set(normalized.split())
        if not normalized or normalized in seen:
            continue
        if len(normalized.split()) < 2:
            continue
        if not original_has_digits and re.search(r"\d", candidate):
            # Guard against invented numbers.
            continue

        sim = similarity(candidate, original_question)
        if sim > 0.95:
            # Too close to canonical phrasing.
            continue
        if sim < 0.40:
            # Too far semantically for safe auto-approval.
            continue
        if focus_terms and not (candidate_tokens & focus_terms):
            # Keep row-specific intent (e.g., majors, tuition, duration) in each variant.
            continue

        seen.add(normalized)
        selected.append(candidate)
        if len(selected) >= target_count:
            break

    if len(selected) >= target_count:
        return selected[:target_count]

    # Fill with deterministic semantic anchors, then fallback templates.
    filler_candidates = semantic_anchor_variants(original_question, language)
    filler_candidates.extend(deterministic_variants(original_question, language, target_count * 2))
    for filler in filler_candidates:
        candidate = enforce_question_shape(filler, original_question, language)
        normalized = normalize_text(candidate)
        candidate_tokens = set(normalized.split())
        if not normalized or normalized in seen:
            continue
        sim = similarity(candidate, original_question)
        if sim < 0.35:
            continue
        if focus_terms and not (candidate_tokens & focus_terms):
            continue
        seen.add(normalized)
        selected.append(candidate)
        if len(selected) >= target_count:
            break

    return selected[:target_count]


def _focus_terms(tokens: set[str], language: str) -> set[str]:
    """Return intent-bearing tokens that variants should preserve to avoid topic drift."""

    if language == "vi":
        vocabulary = {
            "hoc",
            "phi",
            "nganh",
            "chuyen",
            "nganh",
            "thoi",
            "gian",
            "chuong",
            "trinh",
            "bang",
            "cap",
            "hoc",
            "bong",
            "co",
            "so",
            "tuyen",
            "sinh",
            "ielts",
            "thuc",
            "hanh",
            "ly",
            "thuyet",
            "thuc",
            "tap",
            "giang",
            "vien",
            "ky",
            "tuc",
            "xa",
            "chuyen",
            "tiep",
        }
    else:
        vocabulary = {
            "tuition",
            "fee",
            "cost",
            "major",
            "majors",
            "program",
            "duration",
            "degree",
            "scholarship",
            "campus",
            "campuses",
            "admission",
            "requirements",
            "ielts",
            "practical",
            "theory",
            "internship",
            "faculty",
            "transfer",
            "dormitory",
            "documents",
        }
    return {token for token in tokens if token in vocabulary}


def generate_variants_with_gemini(
    records: list[dict[str, str]],
    language: str,
    variants_per_question: int,
    batch_size: int,
) -> dict[str, list[str]]:
    """Generate variants with Gemini in small batches and return id->variants map."""

    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip()
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1").strip() or "us-central1"
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"
    if not project:
        return {}

    try:
        from google import genai
        from google.genai import types
    except Exception:
        return {}

    client = genai.Client(vertexai=True, project=project, location=location)
    all_results: dict[str, list[str]] = {}
    language_label = "Vietnamese" if language == "vi" else "English"
    total_batches = max(1, (len(records) + max(1, batch_size) - 1) // max(1, batch_size))

    for start in range(0, len(records), max(1, batch_size)):
        batch_index = (start // max(1, batch_size)) + 1
        batch = records[start : start + max(1, batch_size)]
        print(
            f"[{language}] batch {batch_index}/{total_batches}: generating variants for {len(batch)} questions...",
            flush=True,
        )
        lines = [f"- {item['id']} | section={item.get('section_vi', '')}: {item['question']}" for item in batch]
        prompt = (
            "Create speech-style question variants for FAQ retrieval. "
            "Keep intent and scope exactly the same, keep language unchanged, do not answer, "
            "and do not add new facts, names, numbers, dates, or claims. "
            "You may add or remove function words, change politeness tone, shorten/expand phrasing, "
            "and use natural spoken alternatives. "
            "For questions mentioning Greenwich Vietnam, include at least one variant that explicitly mentions "
            "university (EN) or truong dai hoc (VI) without adding new factual claims. "
            "For two-clause questions, include at least one concise single-clause variant. "
            f"Return JSON array only with objects: {{\"id\": string, \"variants\": string[]}}. "
            f"Each object must contain exactly {variants_per_question} variants. "
            f"Language is {language_label}.\n\nQuestions:\n"
            + "\n".join(lines)
        )

        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                system_instruction=(
                    "You are a precise question-rewrite assistant for multilingual FAQ retrieval. "
                    "Output strict JSON only."
                ),
            ),
        )

        parsed = parse_json_response_array(getattr(response, "text", "") or "")
        for item in parsed:
            item_id = str(item.get("id", "")).strip()
            variants_raw = item.get("variants", [])
            if not item_id or not isinstance(variants_raw, list):
                continue
            question_lookup = {q["id"]: q["question"] for q in batch}
            cleaned = select_variants(
                original_question=question_lookup.get(item_id, ""),
                language=language,
                candidates=[str(variant).strip() for variant in variants_raw],
                target_count=variants_per_question,
            )
            if cleaned:
                all_results[item_id] = cleaned

    return all_results


def build_variant_records(
    qa_records: list[dict[str, Any]],
    variants_per_language: int,
    batch_size: int,
    enable_llm: bool,
) -> list[dict[str, Any]]:
    """Build auto-approved variant records while preserving canonical source IDs."""

    vi_questions: list[dict[str, str]] = []
    en_questions: list[dict[str, str]] = []
    for record in qa_records:
        source_id = str(record.get("id", "")).strip()
        if not source_id:
            continue
        question_vi = str(record.get("question_vi", "")).strip()
        question_en = str(record.get("question_en", "")).strip()
        if question_vi:
            vi_questions.append(
                {
                    "id": source_id,
                    "question": question_vi,
                    "section_vi": str(record.get("section_vi", "")).strip(),
                }
            )
        if question_en:
            en_questions.append(
                {
                    "id": source_id,
                    "question": question_en,
                    "section_vi": str(record.get("section_vi", "")).strip(),
                }
            )

    vi_variants_by_id: dict[str, list[str]] = {}
    en_variants_by_id: dict[str, list[str]] = {}

    if enable_llm:
        vi_variants_by_id = generate_variants_with_gemini(
            records=vi_questions,
            language="vi",
            variants_per_question=variants_per_language,
            batch_size=batch_size,
        )
        en_variants_by_id = generate_variants_with_gemini(
            records=en_questions,
            language="en",
            variants_per_question=variants_per_language,
            batch_size=batch_size,
        )

    result: list[dict[str, Any]] = []
    for record in qa_records:
        source_id = str(record.get("id", "")).strip()
        if not source_id:
            continue
        question_vi = str(record.get("question_vi", "")).strip()
        question_en = str(record.get("question_en", "")).strip()

        variants_vi = vi_variants_by_id.get(source_id) if enable_llm else None
        variants_en = en_variants_by_id.get(source_id) if enable_llm else None

        if not variants_vi and question_vi:
            variants_vi = select_variants(
                original_question=question_vi,
                language="vi",
                candidates=semantic_anchor_variants(question_vi, "vi")
                + deterministic_variants(question_vi, "vi", variants_per_language * 2),
                target_count=variants_per_language,
            )
        if not variants_en and question_en:
            variants_en = select_variants(
                original_question=question_en,
                language="en",
                candidates=semantic_anchor_variants(question_en, "en")
                + deterministic_variants(question_en, "en", variants_per_language * 2),
                target_count=variants_per_language,
            )

        if variants_vi and question_vi:
            variants_vi = select_variants(question_vi, "vi", variants_vi, variants_per_language)
        if variants_en and question_en:
            variants_en = select_variants(question_en, "en", variants_en, variants_per_language)

        result.append(
            {
                "id": source_id,
                "source": record.get("source"),
                "source_detail": record.get("source_detail"),
                "section_vi": record.get("section_vi"),
                "question_vi": question_vi,
                "question_en": question_en,
                "variants_vi": variants_vi or [],
                "variants_en": variants_en or [],
                "generation_status": "auto_approved",
            }
        )

    return result


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for variant generation."""

    parser = argparse.ArgumentParser(description="Generate auto-approved semantic question variants for curated Q&A")
    parser.add_argument(
        "--input",
        default="output/qa_sample_vi_en.json",
        help="Path to canonical Q&A JSON file",
    )
    parser.add_argument(
        "--output",
        default="output/qa_variants_auto.json",
        help="Destination path for generated variant JSON",
    )
    parser.add_argument(
        "--variants-per-language",
        type=int,
        default=3,
        help="Number of variant candidates to generate for each language question",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="LLM batch size when model-based generation is enabled",
    )
    parser.add_argument(
        "--disable-llm",
        action="store_true",
        help="Force deterministic generation without Gemini calls",
    )
    return parser.parse_args()


def main() -> None:
    """Generate auto-approved variants and write them to disk."""

    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    load_env_file(Path(".env"))

    records_raw = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(records_raw, list):
        raise RuntimeError("Input JSON must be an array of Q&A records")

    variants = build_variant_records(
        qa_records=[item for item in records_raw if isinstance(item, dict)],
        variants_per_language=max(1, int(args.variants_per_language)),
        batch_size=max(1, int(args.batch_size)),
        enable_llm=not bool(args.disable_llm),
    )

    output_payload = {
        "source_file": str(input_path),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "variants_per_language": max(1, int(args.variants_per_language)),
        "generation_mode": "auto_approved",
        "human_review_required": False,
        "records": variants,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Generated {len(variants)} variant records at {output_path}")


if __name__ == "__main__":
    main()
