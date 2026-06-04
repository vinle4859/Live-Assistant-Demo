"""Regression tests for robust local DB routing guards."""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

from voice_loop.db import KnowledgeBase
from voice_loop.pipeline import VoicePipeline
from voice_loop.transcript_cheats import TranscriptCheatRule


class _FakeSTTProvider:
    async def transcribe(self, audio_path: Path, language: str) -> str:
        return ""


class _EchoLLMProvider:
    async def generate_answer(self, language: str, question: str) -> str:
        return question


class _FakeTTSProvider:
    async def synthesize(self, text: str, language: str, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake-audio")
        return output_path


def _greenwich_force_rules() -> tuple[TranscriptCheatRule, ...]:
    context_terms = (
        "university",
        "truong",
        "trường",
        "dai hoc",
        "đại học",
        "hoc",
        "học",
        "sinh vien",
        "vietnam",
        "viet nam",
        "việt nam",
        "khac biet",
        "khác biệt",
        "so voi",
        "so với",
        "tuition",
        "hoc phi",
        "campus",
    )
    return tuple(
        TranscriptCheatRule(wrong_phrase=wrong_phrase, corrected_phrase="greenwich", required_context_terms=context_terms)
        for wrong_phrase in ("rmit", "huflit", "huflix", "huflex", "remix", "re mix")
    )


def _build_pipeline(tmp_path: Path, rows: list[tuple[str, str, str, str, str, str]]) -> VoicePipeline:
    knowledge_base = KnowledgeBase(tmp_path / "knowledge.sqlite3", retrieval_mode="hybrid", confidence_low=0.55)
    knowledge_base.ensure_schema()
    with sqlite3.connect(knowledge_base.db_path) as connection:
        connection.executemany(
            """
            INSERT INTO knowledge_base(language, keywords, response, source_id, section, question)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        connection.commit()
    return VoicePipeline(
        knowledge_base=knowledge_base,
        stt_provider=_FakeSTTProvider(),
        llm_provider=_EchoLLMProvider(),
        fallback_llm_provider=None,
        primary_tts_provider=_FakeTTSProvider(),
        fallback_tts_provider=None,
        output_dir=tmp_path / "out",
        timeout_seconds=3.0,
        llm_direct_min_query_tokens=1,
    )


def test_keyword_scoring_requires_whole_token_sequence_for_phrase_match(tmp_path: Path) -> None:
    kb = KnowledgeBase(tmp_path / "knowledge.sqlite3")
    query = "Karaoke Le Minh hom nay minh di duoi len di"

    evidence = kb._score_match(kb._tokenize_list(query), kb._normalize(query), "di du")

    assert not evidence.whole_phrase_match
    assert evidence.exact_hit_count == 1
    assert evidence.score < 1.0


def test_keyword_scoring_does_not_fuzzy_match_short_vietnamese_noise(tmp_path: Path) -> None:
    kb = KnowledgeBase(tmp_path / "knowledge.sqlite3")
    query = "cho toi biet tinh hinh giao thong tai duong Cong Hoa"

    evidence = kb._score_match(kb._tokenize_list(query), kb._normalize(query), "giao trinh")

    assert evidence.exact_hit_count == 1
    assert evidence.fuzzy_hit_count == 0
    assert evidence.keyword_coverage < 1.0


def test_generic_english_openers_do_not_create_high_confidence_match(tmp_path: Path) -> None:
    kb = KnowledgeBase(tmp_path / "knowledge.sqlite3", retrieval_mode="hybrid")
    kb.ensure_schema()
    with sqlite3.connect(kb.db_path) as connection:
        connection.execute(
            """
            INSERT INTO knowledge_base(language, keywords, response, source_id, section, question)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "en",
                "tell me about greenwich vietnam",
                "Greenwich Vietnam is an international joint program.",
                "web-001",
                "Overview",
                "What is Greenwich Vietnam?",
            ),
        )
        connection.commit()

    match = kb.lookup_response_details("Tell me traffic on Cong Hoa street", "en")

    assert match is not None
    assert match.score < kb.confidence_low
    assert match.exact_hit_count == 0


def test_vietnamese_karaoke_transcript_does_not_force_local_db(tmp_path: Path) -> None:
    pipeline = _build_pipeline(
        tmp_path,
        [
            (
                "vi",
                "di du",
                "Study abroad answer should not be used.",
                "web-020",
                "Exchange",
                "Sinh vien co co hoi di du hoc hoac trao doi khong?",
            ),
        ],
    )

    response = asyncio.run(pipeline.process_transcription("Karaoke Le Minh hom nay minh di duoi len di", "vi"))

    assert response["resolved_source"] == "llm_direct"
    assert "Study abroad" not in response["response_text"]


def test_vietnamese_traffic_transcript_does_not_force_local_db(tmp_path: Path) -> None:
    pipeline = _build_pipeline(
        tmp_path,
        [
            (
                "vi",
                "giao trinh,thong tin",
                "Curriculum answer should not be used.",
                "web-007",
                "Materials",
                "Giao trinh va tai lieu hoc tap duoc su dung nhu the nao?",
            ),
        ],
    )

    response = asyncio.run(
        pipeline.process_transcription("cho toi biet tinh hinh giao thong tai duong Cong Hoa", "vi")
    )

    assert response["resolved_source"] == "llm_direct"
    assert "Curriculum" not in response["response_text"]


def test_english_traffic_transcript_does_not_force_local_db(tmp_path: Path) -> None:
    pipeline = _build_pipeline(
        tmp_path,
        [
            (
                "en",
                "tell me about greenwich vietnam",
                "Overview answer should not be used.",
                "web-001",
                "Overview",
                "What is Greenwich Vietnam?",
            ),
        ],
    )

    response = asyncio.run(pipeline.process_transcription("Tell me traffic on Cong Hoa street", "en"))

    assert response["resolved_source"] == "llm_direct"
    assert "Overview" not in response["response_text"]


def test_vietnamese_sports_transcript_does_not_force_local_db(tmp_path: Path) -> None:
    pipeline = _build_pipeline(
        tmp_path,
        [
            (
                "vi",
                "sac nhat,giai nhat",
                "Award answer should not be used.",
                "pdf-awards_25-005",
                "Awards",
                "Gia tri giai thuong cua Giai Xuat sac nhat ky la bao nhieu?",
            ),
        ],
    )

    response = asyncio.run(pipeline.process_transcription("Ai thang tran gan nhat cua Manchester United?", "vi"))

    assert response["resolved_source"] == "llm_direct"
    assert "Award" not in response["response_text"]


def test_valid_local_db_matches_are_preserved(tmp_path: Path) -> None:
    pipeline = _build_pipeline(
        tmp_path,
        [
            (
                "en",
                "what majors does greenwich vietnam offer,greenwich vietnam",
                "Greenwich Vietnam offers several majors.",
                "web-004",
                "Programs",
                "What majors does Greenwich Vietnam offer?",
            ),
            (
                "vi",
                "hoc phi toan khoa greenwich viet nam,hoc phi",
                "Hoc phi toan khoa la 450.000.000 VND.",
                "web-014",
                "Tuition",
                "Hoc phi toan khoa cua Greenwich Viet Nam la bao nhieu?",
            ),
            (
                "vi",
                "greenwich viet nam co bao nhieu co so,co so",
                "Greenwich Viet Nam co 4 co so.",
                "web-017",
                "Campus",
                "Greenwich Viet Nam co bao nhieu co so?",
            ),
        ],
    )

    majors = asyncio.run(pipeline.process_transcription("What majors does Greenwich Vietnam offer?", "en"))
    tuition = asyncio.run(pipeline.process_transcription("Hoc phi cua Greenwich Viet Nam la bao nhieu?", "vi"))
    campus = asyncio.run(pipeline.process_transcription("Truong co bao nhieu co so?", "vi"))

    assert majors["resolved_source"] == "local_db"
    assert tuition["resolved_source"] == "local_db"
    assert campus["resolved_source"] == "local_db"


def test_bare_greenwich_vietnam_asks_for_clarification(tmp_path: Path) -> None:
    pipeline = _build_pipeline(
        tmp_path,
        [
            (
                "vi",
                "greenwich viet nam,greenwich viet nam la gi",
                "Generic Greenwich overview.",
                "web-001",
                "Overview",
                "Greenwich Viet Nam la gi?",
            ),
            (
                "vi",
                "greenwich viet nam co bao nhieu co so",
                "Campus answer should not win.",
                "web-017",
                "Campus",
                "Greenwich Viet Nam co bao nhieu co so?",
            ),
        ],
    )

    response = asyncio.run(pipeline.process_transcription("greenwich Việt Nam", "vi"))

    assert response["resolved_source"] == "fallback"
    assert "Generic Greenwich overview" not in response["response_text"]
    assert "chu de" in response["response_text"]


def test_corrected_remix_bare_greenwich_asks_for_clarification(tmp_path: Path) -> None:
    pipeline = _build_pipeline(
        tmp_path,
        [
            (
                "vi",
                "greenwich viet nam,greenwich viet nam la gi",
                "Generic Greenwich overview.",
                "web-001",
                "Overview",
                "Greenwich Viet Nam la gi?",
            ),
        ],
    )
    pipeline.transcript_cheats = (
        TranscriptCheatRule(
            wrong_phrase="remix",
            corrected_phrase="greenwich",
            required_context_terms=("vietnam", "viet nam", "việt nam"),
        ),
    )

    response = asyncio.run(pipeline.process_transcription("remix, Việt Nam", "vi"))

    assert response["resolved_source"] == "fallback"
    assert "Generic Greenwich overview" not in response["response_text"]
    assert "chu de" in response["response_text"]


def test_english_greenwich_identity_forms_route_to_overview(tmp_path: Path) -> None:
    pipeline = _build_pipeline(
        tmp_path,
        [
            (
                "en",
                "greenwich vietnam",
                "Generic Greenwich overview.",
                "web-001",
                "Overview",
                "What is Greenwich Vietnam?",
            ),
        ],
    )

    what_is = asyncio.run(pipeline.process_transcription("What is Greenwich Vietnam?", "en"))
    tell_me = asyncio.run(pipeline.process_transcription("Tell me about Greenwich Vietnam", "en"))

    assert what_is["resolved_source"] == "local_db"
    assert tell_me["resolved_source"] == "local_db"
    assert "Generic Greenwich overview" in what_is["response_text"]
    assert "Generic Greenwich overview" in tell_me["response_text"]


def test_greenwich_overview_pin_does_not_bind_non_identity_questions(tmp_path: Path) -> None:
    pipeline = _build_pipeline(
        tmp_path,
        [
            (
                "en",
                "greenwich vietnam",
                "Overview answer should not be used.",
                "web-001",
                "Overview",
                "What is Greenwich Vietnam?",
            ),
        ],
    )

    prompts = [
        "Where is Greenwich Vietnam located?",
        "Does Greenwich Vietnam have dormitories?",
        "Tell me traffic near Greenwich Vietnam",
        "Is Greenwich Vietnam public or private?",
    ]

    for prompt in prompts:
        response = asyncio.run(pipeline.process_transcription(prompt, "en"))
        assert response["resolved_source"] != "local_db"
        assert "Overview answer" not in response["response_text"]


def test_corrected_university_substitution_comparison_goes_to_llm_not_overview(tmp_path: Path) -> None:
    pipeline = _build_pipeline(
        tmp_path,
        [
            (
                "vi",
                "greenwich viet nam,greenwich viet nam la gi",
                "Overview answer should not be used.",
                "web-001",
                "Overview",
                "Greenwich Viet Nam la gi?",
            ),
        ],
    )
    pipeline.transcript_cheats = _greenwich_force_rules()

    prompts = [
        "Khác biệt giữa trường đại học RMIT Việt Nam và các trường đại học khác là gì?",
        "học đại học HUFLIT thì khác gì hơn so với học những đại học khác tại Việt Nam",
    ]

    for prompt in prompts:
        response = asyncio.run(pipeline.process_transcription(prompt, "vi"))
        assert response["resolved_source"] == "llm_direct"
        assert "greenwich" in response["response_text"].lower()
        assert "Overview answer" not in response["response_text"]
