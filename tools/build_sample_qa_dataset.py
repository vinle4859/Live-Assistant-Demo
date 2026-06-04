"""Build a VN/EN sample Q/A dataset from Greenwich FAQ and local PDFs.

This utility creates a practical MVP dataset for assistant bootstrapping:
1. Loads web FAQ question+answer pairs from a local seed exported from the FAQ accordion.
2. Extracts additional question-like candidates from local PDFs in ``data/Q&A``.
3. Optionally generates PDF answers from PDF content using Gemini on Vertex AI.
4. Optionally translates Vietnamese questions/answers to English using Gemini.
5. Writes JSON output ready for review or ingestion.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pypdf import PdfReader


FAQ_URL = "https://greenwich.edu.vn/cau-hoi-thuong-gap/"
DEFAULT_WEB_QA_SEED_PATH = Path("data/web_faq_qa_vi.json")

WEB_FAQ_QUESTIONS_VI: list[dict[str, str]] = [
    {"section_vi": "Về Chương Trình Đào Tạo và Bằng Cấp", "question_vi": "Greenwich Việt Nam là gì?"},
    {
        "section_vi": "Về Chương Trình Đào Tạo và Bằng Cấp",
        "question_vi": "Bằng cấp do ai cấp? Có phải bằng quốc tế không?",
    },
    {
        "section_vi": "Về Chương Trình Đào Tạo và Bằng Cấp",
        "question_vi": "Chương trình đào tạo tại Greenwich Việt Nam có gì đặc biệt?",
    },
    {
        "section_vi": "Về Chương Trình Đào Tạo và Bằng Cấp",
        "question_vi": "Greenwich Việt Nam đào tạo những chuyên ngành nào?",
    },
    {
        "section_vi": "Về Chương Trình Đào Tạo và Bằng Cấp",
        "question_vi": "Bằng cấp nhận được ở Greenwich Việt Nam là gì và có giá trị không?",
    },
    {"section_vi": "Về Chương Trình Đào Tạo và Bằng Cấp", "question_vi": "Thời gian đào tạo của chương trình là bao lâu?"},
    {
        "section_vi": "Về Chương Trình Đào Tạo và Bằng Cấp",
        "question_vi": "Giáo trình và tài liệu học tập được sử dụng như thế nào?",
    },
    {
        "section_vi": "Về Chương Trình Đào Tạo và Bằng Cấp",
        "question_vi": "Khác biệt gì với học ở Anh / Singapore / các trường quốc tế khác?",
    },
    {
        "section_vi": "Về Chương Trình Đào Tạo và Bằng Cấp",
        "question_vi": "Chuẩn đầu ra thế nào? Có IELTS bắt buộc không?",
    },
    {
        "section_vi": "Về Chương Trình Đào Tạo và Bằng Cấp",
        "question_vi": "Có phải học nhiều lý thuyết không hay thiên về thực hành?",
    },
    {"section_vi": "Về Điều Kiện Tuyển Sinh và Học Phí", "question_vi": "Điều kiện xét tuyển vào Greenwich Việt Nam là gì?"},
    {"section_vi": "Về Điều Kiện Tuyển Sinh và Học Phí", "question_vi": "Trình độ tiếng Anh đầu vào yêu cầu là bao nhiêu?"},
    {
        "section_vi": "Về Điều Kiện Tuyển Sinh và Học Phí",
        "question_vi": "Nếu chưa đáp ứng yêu cầu tiếng Anh đầu vào thì phải làm sao?",
    },
    {"section_vi": "Về Điều Kiện Tuyển Sinh và Học Phí", "question_vi": "Học phí toàn khóa của Greenwich Việt Nam là bao nhiêu?"},
    {"section_vi": "Về Điều Kiện Tuyển Sinh và Học Phí", "question_vi": "Greenwich Việt Nam có chính sách học bổng không?"},
    {"section_vi": "Về Điều Kiện Tuyển Sinh và Học Phí", "question_vi": "Có phỏng vấn đầu vào không?"},
    {"section_vi": "Về Môi Trường Học Tập và Cơ Hội Quốc Tế", "question_vi": "Greenwich Việt Nam có bao nhiêu cơ sở?"},
    {
        "section_vi": "Về Môi Trường Học Tập và Cơ Hội Quốc Tế",
        "question_vi": "Đội ngũ giảng viên có đạt chuẩn quốc tế không?",
    },
    {
        "section_vi": "Về Môi Trường Học Tập và Cơ Hội Quốc Tế",
        "question_vi": "Sinh viên có được học với giảng viên nước ngoài không?",
    },
    {
        "section_vi": "Về Môi Trường Học Tập và Cơ Hội Quốc Tế",
        "question_vi": "Sinh viên có cơ hội đi du học hoặc trao đổi không?",
    },
    {
        "section_vi": "Về Môi Trường Học Tập và Cơ Hội Quốc Tế",
        "question_vi": "Cần chuẩn bị gì khi chuyển tiếp sang học tại Anh?",
    },
    {
        "section_vi": "Về Môi Trường Học Tập và Cơ Hội Quốc Tế",
        "question_vi": "Kỹ năng tiếng Anh chưa tốt có học được không?",
    },
    {
        "section_vi": "Về Môi Trường Học Tập và Cơ Hội Quốc Tế",
        "question_vi": "Hoạt động ngoại khóa / câu lạc bộ có nhiều không?",
    },
    {"section_vi": "Về Môi Trường Học Tập và Cơ Hội Quốc Tế", "question_vi": "Quy mô lớp học có đông không?"},
    {"section_vi": "Về Triển Vọng Nghề Nghiệp", "question_vi": "Cơ hội việc làm sau khi tốt nghiệp có cao không?"},
    {
        "section_vi": "Về Triển Vọng Nghề Nghiệp",
        "question_vi": "Sinh viên tốt nghiệp có thể học lên Thạc sĩ, Tiến sĩ không?",
    },
    {
        "section_vi": "Về Triển Vọng Nghề Nghiệp",
        "question_vi": "Học các ngành như Công nghệ thông tin (CNTT) hay Thiết kế Đồ họa có cần năng khiếu/giỏi Toán/vẽ đẹp không?",
    },
    {
        "section_vi": "Về Triển Vọng Nghề Nghiệp",
        "question_vi": "Trường có hỗ trợ kết nối doanh nghiệp, thực tập không?",
    },
    {
        "section_vi": "Về Triển Vọng Nghề Nghiệp",
        "question_vi": "Học xong có thể ra nước ngoài làm việc được không?",
    },
    {"section_vi": "Về Triển Vọng Nghề Nghiệp", "question_vi": "Sinh viên năm mấy bắt đầu được thực tập?"},
    {"section_vi": "Các Câu Hỏi Khác", "question_vi": "Hồ sơ xét tuyển vào chương trình gồm những gì?"},
    {"section_vi": "Các Câu Hỏi Khác", "question_vi": "Nhà trường có kí túc xá cho sinh viên không?"},
    {"section_vi": "Các Câu Hỏi Khác", "question_vi": "Học phí đóng theo tháng, kì hay theo năm học?"},
    {
        "section_vi": "Các Câu Hỏi Khác",
        "question_vi": "Thí sinh theo học các chương trình tự học tại nhà (Homeschool) có đủ điều kiện xét tuyển vào trường không?",
    },
]


@dataclass
class QuestionRecord:
    """A normalized Q/A record for downstream ingestion."""

    id: str
    source: str
    source_detail: str
    section_vi: str
    question_vi: str
    answer_vi: str | None = None
    question_en: str | None = None
    answer_en: str | None = None


def load_env_file(env_path: Path) -> None:
    """Load key-value entries from ``.env`` into ``os.environ`` if missing."""

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


def normalize_space(text: str) -> str:
    """Collapse repeated whitespace and trim boundaries."""

    return re.sub(r"\s+", " ", text).strip()


def get_vertex_settings() -> tuple[str, str, str]:
    """Return Gemini runtime settings from environment variables."""

    project = os.environ.get("GOOGLE_CLOUD_PROJECT", "").strip()
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1").strip() or "us-central1"
    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"
    return project, location, model


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract and join text from all pages of a PDF file."""

    reader = PdfReader(str(pdf_path))
    chunks: list[str] = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        if page_text:
            chunks.append(page_text)

    return "\n".join(chunks)


def extract_pdf_question_candidates(text: str, max_questions: int) -> list[str]:
    """Extract question-like sentence candidates from PDF text."""

    if not text:
        return []

    # Keep question-like fragments and filter obvious noise/too-short fragments.
    matches = re.findall(r"[^\n?.!]{12,220}\?", text)
    cleaned: list[str] = []
    seen: set[str] = set()
    for match in matches:
        candidate = normalize_space(match)
        lower = candidate.lower()
        if not candidate.endswith("?"):
            continue
        if lower in seen:
            continue
        if any(token in lower for token in ("http://", "https://", "@", "www.")):
            continue
        seen.add(lower)
        cleaned.append(candidate)
        if len(cleaned) >= max_questions:
            break

    return cleaned


def generate_pdf_questions_with_gemini(pdf_path: Path, text: str, max_questions: int) -> list[str]:
    """Generate Vietnamese question candidates from PDF text with Gemini."""

    if not text:
        return []

    project, location, model = get_vertex_settings()
    if not project:
        return []

    from google import genai
    from google.genai import types

    prompt = (
        "Tạo danh sách câu hỏi FAQ bằng tiếng Việt dựa trên nội dung tài liệu sau. "
        f"Chỉ trả về JSON array các chuỗi, tối đa {max_questions} câu hỏi, mỗi câu kết thúc bằng dấu hỏi. "
        "Không thêm giải thích ngoài JSON.\n\n"
        f"Tài liệu: {pdf_path.name}\n"
        f"Nội dung trích xuất:\n{text[:14000]}"
    )

    client = genai.Client(vertexai=True, project=project, location=location)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            system_instruction="Bạn là biên tập viên học vụ, tạo câu hỏi đúng ngữ cảnh tài liệu.",
        ),
    )

    content = (getattr(response, "text", "") or "").strip()
    if not content:
        return []

    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?", "", content).strip()
        content = re.sub(r"```$", "", content).strip()

    try:
        items = json.loads(content)
    except json.JSONDecodeError:
        return []

    if not isinstance(items, list):
        return []

    cleaned: list[str] = []
    seen: set[str] = set()
    for item in items:
        question = normalize_space(str(item))
        if not question:
            continue
        if not question.endswith("?"):
            question = f"{question}?"
        lower = question.lower()
        if lower in seen:
            continue
        seen.add(lower)
        cleaned.append(question)
        if len(cleaned) >= max_questions:
            break

    return cleaned


def generate_pdf_answers_with_gemini(pdf_path: Path, text: str, questions: list[str]) -> list[str | None]:
    """Generate Vietnamese answers for PDF-derived questions using Gemini."""

    if not text or not questions:
        return [None for _ in questions]

    project, location, model = get_vertex_settings()
    if not project:
        return [None for _ in questions]

    from google import genai
    from google.genai import types

    payload = [{"id": idx + 1, "question_vi": question} for idx, question in enumerate(questions)]
    prompt = (
        "Trả lời ngắn gọn, chính xác bằng tiếng Việt cho từng câu hỏi dựa trên tài liệu sau. "
        "Không bịa thông tin ngoài tài liệu. "
        "Chỉ trả về JSON array gồm các object: "
        '{"id": number, "answer_vi": string}.\n\n'
        f"Tài liệu: {pdf_path.name}\n"
        f"Câu hỏi JSON:\n{json.dumps(payload, ensure_ascii=False)}\n\n"
        f"Nội dung trích xuất:\n{text[:16000]}"
    )

    client = genai.Client(vertexai=True, project=project, location=location)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            system_instruction="Bạn là cố vấn tuyển sinh, chỉ dùng thông tin có trong tài liệu được cung cấp.",
        ),
    )

    content = (getattr(response, "text", "") or "").strip()
    if not content:
        return [None for _ in questions]

    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?", "", content).strip()
        content = re.sub(r"```$", "", content).strip()

    try:
        items = json.loads(content)
    except json.JSONDecodeError:
        return [None for _ in questions]

    if not isinstance(items, list):
        return [None for _ in questions]

    answers_by_id: dict[int, str] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        item_id = item.get("id")
        answer_vi = normalize_space(str(item.get("answer_vi", "")))
        if isinstance(item_id, int) and answer_vi:
            answers_by_id[item_id] = answer_vi

    return [answers_by_id.get(idx + 1) for idx in range(len(questions))]


def build_pdf_records(
    pdf_dir: Path,
    max_per_pdf: int,
    use_llm_fallback: bool,
    generate_answers: bool,
) -> list[QuestionRecord]:
    """Create Q/A records from all PDFs in a directory."""

    records: list[QuestionRecord] = []
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        text = extract_pdf_text(pdf_path)
        candidates = extract_pdf_question_candidates(text, max_per_pdf)

        # Some handbook-style PDFs rarely use question marks. Generate candidates with LLM as fallback.
        if use_llm_fallback and not candidates:
            candidates = generate_pdf_questions_with_gemini(pdf_path, text, max_per_pdf)

        answers: list[str | None] = [None for _ in candidates]
        if generate_answers and candidates:
            answers = generate_pdf_answers_with_gemini(pdf_path, text, candidates)

        for idx, question in enumerate(candidates, start=1):
            records.append(
                QuestionRecord(
                    id=f"pdf-{pdf_path.stem}-{idx:03d}",
                    source="pdf",
                    source_detail=str(pdf_path.as_posix()),
                    section_vi=f"PDF::{pdf_path.stem}",
                    question_vi=question,
                    answer_vi=answers[idx - 1],
                )
            )
    return records


def build_web_records(web_seed_path: Path) -> list[QuestionRecord]:
    """Create normalized records from a web FAQ Q/A seed file."""

    records: list[QuestionRecord] = []
    source_items = WEB_FAQ_QUESTIONS_VI
    if web_seed_path.exists():
        loaded = json.loads(web_seed_path.read_text(encoding="utf-8"))
        if isinstance(loaded, list):
            source_items = loaded

    for idx, item in enumerate(source_items, start=1):
        question_vi = normalize_space(str(item.get("question_vi", ""))) if isinstance(item, dict) else ""
        if not question_vi:
            continue
        answer_vi_raw = str(item.get("answer_vi", "")) if isinstance(item, dict) else ""
        answer_vi = normalize_space(answer_vi_raw)
        records.append(
            QuestionRecord(
                id=f"web-{idx:03d}",
                source="web",
                source_detail=FAQ_URL,
                section_vi=normalize_space(str(item.get("section_vi", ""))) if isinstance(item, dict) else "",
                question_vi=question_vi,
                answer_vi=answer_vi or None,
            )
        )
    return records


def deduplicate_records(records: list[QuestionRecord]) -> list[QuestionRecord]:
    """Drop duplicates by Vietnamese question text while preserving order."""

    unique: list[QuestionRecord] = []
    seen: set[str] = set()
    for record in records:
        key = normalize_space(record.question_vi).lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(record)
    return unique


def translate_records_with_gemini(records: list[QuestionRecord], batch_size: int) -> None:
    """Translate Vietnamese questions/answers to English in-place using Gemini batches."""

    project, location, model = get_vertex_settings()

    if not project:
        raise RuntimeError("GOOGLE_CLOUD_PROJECT is not configured for Gemini translation")

    from google import genai
    from google.genai import types

    client = genai.Client(vertexai=True, project=project, location=location)

    batch_size = max(1, batch_size)
    total_batches = (len(records) + batch_size - 1) // batch_size

    for batch_index, start in enumerate(range(0, len(records), batch_size), start=1):
        print(f"Translating batch {batch_index}/{total_batches}...")
        batch = records[start : start + batch_size]
        payload = [
            {
                "id": record.id,
                "question_vi": record.question_vi,
                "answer_vi": record.answer_vi,
            }
            for record in batch
        ]

        prompt = (
            "Translate each Vietnamese university FAQ question and answer into natural English. "
            "Keep the meaning exact, keep proper nouns (like Greenwich Vietnam) intact, "
            "and return JSON array only with objects: "
            '{"id": string, "question_en": string, "answer_en": string|null}.\n\n'
            f"Input JSON:\n{json.dumps(payload, ensure_ascii=False)}"
        )

        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                system_instruction="You are a precise professional translator.",
            ),
        )

        content = (getattr(response, "text", "") or "").strip()
        if not content:
            raise RuntimeError("Gemini translation returned empty output")

        # Handle optional markdown fences in model output.
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?", "", content).strip()
            content = re.sub(r"```$", "", content).strip()

        translated_items = json.loads(content)
        if not isinstance(translated_items, list):
            raise RuntimeError("Unexpected translation payload shape")

        en_by_id: dict[str, dict[str, str | None]] = {}
        for item in translated_items:
            if not isinstance(item, dict):
                continue
            item_id = str(item.get("id", "")).strip()
            question_en = str(item.get("question_en", "")).strip()
            answer_en_raw = item.get("answer_en")
            answer_en = normalize_space(str(answer_en_raw)) if answer_en_raw else None
            if item_id and question_en:
                en_by_id[item_id] = {
                    "question_en": question_en,
                    "answer_en": answer_en,
                }

        for record in batch:
            translated = en_by_id.get(record.id)
            if not translated:
                continue
            record.question_en = translated.get("question_en")
            record.answer_en = translated.get("answer_en")


def to_json_serializable(records: list[QuestionRecord]) -> list[dict[str, Any]]:
    """Convert dataclass records to JSON serializable dictionaries."""

    return [
        {
            "id": record.id,
            "source": record.source,
            "source_detail": record.source_detail,
            "section_vi": record.section_vi,
            "question_vi": record.question_vi,
            "answer_vi": record.answer_vi,
            "question_en": record.question_en,
            "answer_en": record.answer_en,
        }
        for record in records
    ]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for dataset build options."""

    parser = argparse.ArgumentParser(description="Build VN/EN sample Q/A dataset from web FAQ and PDFs.")
    parser.add_argument(
        "--web-seed",
        type=Path,
        default=DEFAULT_WEB_QA_SEED_PATH,
        help="Local JSON seed with web FAQ records (section_vi/question_vi/answer_vi).",
    )
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=Path("data/Q&A"),
        help="Directory containing local PDF files.",
    )
    parser.add_argument(
        "--max-pdf-questions",
        type=int,
        default=8,
        help="Maximum number of question candidates extracted per PDF.",
    )
    parser.add_argument(
        "--include-pdf",
        action="store_true",
        help="Include question candidates extracted from PDFs.",
    )
    parser.add_argument(
        "--disable-pdf-llm-fallback",
        action="store_true",
        help="Disable Gemini fallback generation when PDFs contain no explicit question lines.",
    )
    parser.add_argument(
        "--disable-pdf-answer-generation",
        action="store_true",
        help="Disable Gemini answer generation for PDF-derived questions.",
    )
    parser.add_argument(
        "--translate-en",
        action="store_true",
        help="Translate Vietnamese questions and answers to English using Gemini.",
    )
    parser.add_argument(
        "--translation-batch-size",
        type=int,
        default=8,
        help="Number of records to translate per Gemini request.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/qa_sample_vi_en.json"),
        help="Destination JSON file path.",
    )
    return parser.parse_args()


def main() -> None:
    """Execute the end-to-end sample dataset build."""

    args = parse_args()
    load_env_file(Path(".env"))

    records = build_web_records(args.web_seed)

    if args.include_pdf and args.pdf_dir.exists():
        records.extend(
            build_pdf_records(
                args.pdf_dir,
                max_per_pdf=max(1, args.max_pdf_questions),
                use_llm_fallback=not args.disable_pdf_llm_fallback,
                generate_answers=not args.disable_pdf_answer_generation,
            )
        )

    records = deduplicate_records(records)

    if args.translate_en:
        translate_records_with_gemini(records, batch_size=max(1, args.translation_batch_size))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(to_json_serializable(records), ensure_ascii=False, indent=2), encoding="utf-8")

    web_count = sum(1 for record in records if record.source == "web")
    pdf_count = sum(1 for record in records if record.source == "pdf")
    translated_question_count = sum(1 for record in records if record.question_en)
    translated_answer_count = sum(1 for record in records if record.answer_en)
    vi_answer_count = sum(1 for record in records if record.answer_vi)

    print(f"Wrote {len(records)} records to {args.output.as_posix()}")
    print(f"  web: {web_count}")
    print(f"  pdf: {pdf_count}")
    print(f"  answered_vi: {vi_answer_count}")
    print(f"  translated_questions_en: {translated_question_count}")
    print(f"  translated_answers_en: {translated_answer_count}")


if __name__ == "__main__":
    main()
