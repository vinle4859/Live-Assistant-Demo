"""Unit tests for live interview enhancements and fallback logic."""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from voice_loop.db import KnowledgeBase
from voice_loop.pipeline import VoicePipeline


class FakeSTTProvider:
    async def transcribe(self, audio_path: Path, language: str) -> str:
        return "what is your name"


class FakeLLMProvider:
    async def generate_answer(self, language: str, question: str) -> str:
        return "I am the voice assistant pipeline."


class FakeTTSProvider:
    async def synthesize(self, text: str, language: str, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake-audio")
        return output_path


def test_markdown_stripping_logic() -> None:
    """VoicePipeline should strip markdown bold, italic, and header formatting."""

    # Test _normalize_spoken_response
    raw_response = "### Greenwich \n**University** has *9* semesters."
    normalized = VoicePipeline._normalize_spoken_response(raw_response)
    assert "#" not in normalized
    assert "*" not in normalized
    assert "_" not in normalized
    assert "Greenwich University has 9 semesters." in normalized

    # Test _prepare_text_for_tts
    raw_tts = "## FPT Education\n**Greenwich** *Vietnam*"
    prepared = VoicePipeline._prepare_text_for_tts(raw_tts)
    assert prepared == "FPT Education Greenwich Vietnam"


def test_audio_path_schema_migration(tmp_path: Path) -> None:
    """KnowledgeBase should add audio_path column automatically and retrieve it."""

    db_file = tmp_path / "knowledge_migrated.sqlite3"
    
    # 1. Create a DB schema with the old layout (no audio_path)
    with sqlite3.connect(db_file) as conn:
        conn.execute(
            """
            CREATE TABLE knowledge_base (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                language TEXT NOT NULL,
                keywords TEXT NOT NULL,
                response TEXT NOT NULL,
                source_id TEXT,
                section TEXT,
                question TEXT
            )
            """
        )
        conn.execute(
            "INSERT INTO knowledge_base (language, keywords, response, source_id) VALUES (?, ?, ?, ?)",
            ("en", "test tuition", "Tuition is 100.", "web-014"),
        )
        conn.commit()

    # 2. Instantiate KnowledgeBase (calls ensure_schema -> _ensure_optional_columns)
    kb = KnowledgeBase(db_file)
    kb.ensure_schema()

    # Check that audio_path exists now
    with sqlite3.connect(db_file) as conn:
        cursor = conn.execute("PRAGMA table_info(knowledge_base)")
        cols = {row[1] for row in cursor.fetchall()}
        assert "audio_path" in cols

    # Update row to set audio_path
    with sqlite3.connect(db_file) as conn:
        conn.execute("UPDATE knowledge_base SET audio_path = ? WHERE source_id = ?", ("data/cached_fee.mp3", "web-014"))
        conn.commit()

    # Retrieve match details
    match = kb.lookup_response_by_source_details("web-014", "en")
    assert match is not None
    assert match.audio_path == "data/cached_fee.mp3"


def test_adaptive_query_coverage_rejection(tmp_path: Path) -> None:
    """VoicePipeline should reject DB matches with low query coverage if not whole-phrase or high score."""

    # Set up DB
    db_file = tmp_path / "knowledge.sqlite3"
    kb = KnowledgeBase(db_file)
    kb.ensure_schema()
    
    # Add a row with keywords "online registration"
    with sqlite3.connect(db_file) as conn:
        conn.execute(
            "INSERT INTO knowledge_base (language, keywords, response, source_id) VALUES (?, ?, ?, ?)",
            ("en", "online registration", "Please register online.", "web-999"),
        )
        conn.commit()

    pipeline = VoicePipeline(
        knowledge_base=kb,
        stt_provider=FakeSTTProvider(),
        llm_provider=FakeLLMProvider(),
        fallback_llm_provider=None,
        primary_tts_provider=FakeTTSProvider(),
        fallback_tts_provider=None,
        output_dir=tmp_path / "output",
        timeout_seconds=3.0,
    )

    # 1. Medium query with high query coverage (disjoint match but coverage = 2/4 = 50% >= 20%)
    medium_query = "online help for registration"
    match_medium = pipeline.knowledge_base.lookup_response_details(medium_query, "en")
    assert match_medium is not None
    
    decision_medium = pipeline._decide_local_db_routing(match_medium, medium_query)
    assert decision_medium.action == "accept_local_db"

    # 2. Long query with low query coverage
    long_query = "yesterday when walking outside I was thinking about the weather and my friend but then I wanted to do it online and find info about registration later on the computer."
    match_long = pipeline.knowledge_base.lookup_response_details(long_query, "en")
    assert match_long is not None
    assert match_long.query_coverage < 0.20

    decision_long = pipeline._decide_local_db_routing(match_long, long_query)
    # Should reject to LLM due to low query coverage
    assert decision_long.action == "reject_to_llm"
    assert decision_long.reason == "low_query_coverage"


def test_noise_reprompt_interception(tmp_path: Path) -> None:
    """VoicePipeline should intercept [NOISE_REPROMPT] and return a polite noise fallback phrase."""

    db_file = tmp_path / "knowledge.sqlite3"
    kb = KnowledgeBase(db_file)
    kb.ensure_schema()

    pipeline = VoicePipeline(
        knowledge_base=kb,
        stt_provider=FakeSTTProvider(),
        llm_provider=FakeLLMProvider(),
        fallback_llm_provider=None,
        primary_tts_provider=FakeTTSProvider(),
        fallback_tts_provider=None,
        output_dir=tmp_path / "output",
        timeout_seconds=3.0,
    )

    # Mock direct LLM response to return sentinel
    pipeline._answer_direct_with_thinking_cue = AsyncMock(
        return_value=MagicMock(answer="[NOISE_REPROMPT]", status="attempted", failure_type=None)
    )

    # Call pipeline transcription process
    result = asyncio.run(pipeline.process_transcription_with_metrics("some garbled noise", "en", stt_elapsed_ms=100))

    # Should fall back and use a noise reprompt phrase
    assert result["resolved_source"] == "fallback"
    assert "reprompt" in result["response_text"].lower() or "repeat" in result["response_text"].lower() or "say that again" in result["response_text"].lower()
