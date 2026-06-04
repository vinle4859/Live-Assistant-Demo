"""Unit tests for the SQLite knowledge base matcher."""

from __future__ import annotations

import json
from pathlib import Path

from voice_loop.db import KnowledgeBase


def test_keyword_matching_returns_expected_english_answer(tmp_path: Path) -> None:
    """The matcher should return the best local English answer for a close query."""

    kb = KnowledgeBase(tmp_path / "knowledge.sqlite3")
    kb.ensure_schema()
    kb.seed_demo_rows()

    response = kb.lookup_response("How do I reset my password?", "en")

    assert response is not None
    assert "reset your password" in response.lower()


def test_keyword_matching_returns_expected_vietnamese_answer(tmp_path: Path) -> None:
    """The matcher should return the best local Vietnamese answer for a close query."""

    kb = KnowledgeBase(tmp_path / "knowledge.sqlite3")
    kb.ensure_schema()
    kb.seed_demo_rows()

    response = kb.lookup_response("Tôi quên mật khẩu rồi", "vi")

    assert response is not None
    assert "mật khẩu" in response.lower()


def test_keyword_matching_tolerates_minor_spelling_drift(tmp_path: Path) -> None:
    """Fuzzy token overlap should still match local answers on small STT misspellings."""

    kb = KnowledgeBase(tmp_path / "knowledge.sqlite3")
    kb.ensure_schema()

    import sqlite3

    with sqlite3.connect(kb.db_path) as connection:
        connection.execute(
            "INSERT INTO knowledge_base(language, keywords, response) VALUES (?, ?, ?)",
            (
                "en",
                "university of greenwich,greenwich campus",
                "The University of Greenwich is in London and Kent.",
            ),
        )
        connection.commit()

    response = kb.lookup_response("Tell me about university of greenwhich", "en")

    assert response is not None
    assert "greenwich" in response.lower()


def test_lookup_response_details_returns_mode_and_score(tmp_path: Path) -> None:
    """Detailed lookup should return match metadata for confidence-gated routing."""

    kb = KnowledgeBase(tmp_path / "knowledge.sqlite3")
    kb.ensure_schema()
    kb.seed_demo_rows()

    match = kb.lookup_response_details("How do I reset my password?", "en")

    assert match is not None
    assert match.retrieval_mode == "lexical"
    assert 0.0 <= match.score <= 1.0
    assert "password" in match.response.lower()


def test_lookup_response_details_supports_hybrid_mode(tmp_path: Path) -> None:
    """Hybrid retrieval mode should still produce ranked local matches."""

    kb = KnowledgeBase(tmp_path / "knowledge.sqlite3", retrieval_mode="hybrid", lexical_top_k=3)
    kb.ensure_schema()
    kb.seed_demo_rows()

    match = kb.lookup_response_details("office working hours", "en")

    assert match is not None
    assert match.retrieval_mode == "hybrid"
    assert match.score >= 0.0


def test_seed_from_qa_json_imports_bilingual_rows_with_source_metadata(tmp_path: Path) -> None:
    """Curated QA JSON import should create VI/EN rows and preserve source IDs."""

    kb = KnowledgeBase(tmp_path / "knowledge.sqlite3", retrieval_mode="hybrid")
    kb.ensure_schema()

    qa_path = tmp_path / "qa_seed.json"
    qa_seed = [
        {
            "id": "web-014",
            "section_vi": "Về Điều Kiện Tuyển Sinh và Học Phí",
            "question_vi": "Học phí toàn khóa của Greenwich Việt Nam là bao nhiêu?",
            "answer_vi": "Học phí toàn khóa là 450.000.000 VNĐ.",
            "question_en": "What is the total tuition fee for the entire program at Greenwich Vietnam?",
            "answer_en": "The total tuition fee is 450,000,000 VND.",
        }
    ]
    qa_path.write_text(json.dumps(qa_seed, ensure_ascii=False), encoding="utf-8")

    inserted_rows = kb.seed_from_qa_json(qa_path)

    assert inserted_rows == 2
    assert kb.has_curated_rows()
    assert kb.lookup_question_by_source("web-014", "en") == qa_seed[0]["question_en"]

    match = kb.lookup_response_details("What is tuition fee cost?", "en")

    assert match is not None
    assert match.source_id == "web-014"
    assert "450,000,000" in match.response


def test_lookup_response_by_source_details_returns_exact_curated_row(tmp_path: Path) -> None:
    """Exact source lookups should bypass ambiguous keyword ranking."""

    kb = KnowledgeBase(tmp_path / "knowledge.sqlite3", retrieval_mode="hybrid")
    kb.ensure_schema()

    qa_path = tmp_path / "qa_seed.json"
    qa_seed = [
        {
            "id": "web-017",
            "question_vi": "Greenwich Việt Nam có bao nhiêu cơ sở?",
            "answer_vi": "Greenwich Việt Nam có 4 cơ sở.",
            "question_en": "How many campuses does Greenwich Vietnam have?",
            "answer_en": "Greenwich Vietnam has 4 campuses.",
        }
    ]
    qa_path.write_text(json.dumps(qa_seed, ensure_ascii=False), encoding="utf-8")
    kb.seed_from_qa_json(qa_path)

    match = kb.lookup_response_by_source_details("web-017", "vi")

    assert match is not None
    assert match.source_id == "web-017"
    assert match.score == 1.0
    assert "4 cơ sở" in match.response
