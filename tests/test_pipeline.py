"""Integration-style tests for the pipeline using fake providers."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from voice_loop.db import KnowledgeBase, KnowledgeMatch
from voice_loop.pipeline import VoicePipeline
from voice_loop.transcript_cheats import TranscriptCheatRule


class FakeSTTProvider:
    """Return a fixed transcript for pipeline testing."""

    async def transcribe(self, audio_path: Path, language: str) -> str:
        """Ignore the input file and return a deterministic transcript."""

        return "what are your office hours"


class FakeLLMProvider:
    """Return a fixed synthesis for pipeline testing."""

    async def generate_answer(self, language: str, question: str):
        """Return a deterministic answer for pipeline testing."""

        return "Support is available from 9 AM to 5 PM, Monday through Friday."


class DirectAnswerLLMProvider:
    """Return a deterministic direct answer when snippets are missing."""

    async def generate_answer(self, language: str, question: str):
        """Return a direct answer independent of external retrieval."""

        return "You can usually reset your password from account settings."


class LongLLMProvider:
    """Return a long answer so spoken compaction can be tested."""

    async def generate_answer(self, language: str, question: str):
        """Return multiple sentences."""

        return (
            "Sentence one answers the question clearly. "
            "Sentence two adds useful detail. "
            "Sentence three should not be spoken in live mode."
        )


class FailingLLMProvider:
    """Raise an error to exercise pipeline hard fallback behavior."""

    async def generate_answer(self, language: str, question: str):
        """Fail answer generation deterministically."""

        raise RuntimeError("intentional llm failure")


class TimeoutLLMProvider:
    """Delay until timeout to exercise secondary LLM fallback behavior."""

    async def generate_answer(self, language: str, question: str):
        """Sleep long enough that short timeout settings cancel this provider."""

        await asyncio.sleep(0.05)
        return "this answer should not be used"


class EmptyLLMProvider:
    """Return an empty answer to exercise LLM empty-output diagnostics."""

    async def generate_answer(self, language: str, question: str):
        """Return no usable answer."""

        return ""


class SlowDirectLLMProvider:
    """Return an answer after a short delay for thinking-cue tests."""

    async def generate_answer(self, language: str, question: str):
        """Delay before returning a complete answer."""

        await asyncio.sleep(0.03)
        return "Here is the complete answer."


class FakeTTSProvider:
    """Write a tiny fake audio file for pipeline testing."""

    async def synthesize(self, text: str, language: str, output_path: Path) -> Path:
        """Create a small placeholder MP3 file and return its path."""

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fake-audio")
        return output_path


class SlowTTSProvider:
    """Delay long enough to trigger a tiny test timeout."""

    async def synthesize(self, text: str, language: str, output_path: Path) -> Path:
        await asyncio.sleep(0.05)
        output_path.write_bytes(b"slow-primary")
        return output_path


class FailingTempWritingTTSProvider:
    """Write a primary temp file, then fail so fallback must own final output."""

    async def synthesize(self, text: str, language: str, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"primary-temp")
        raise RuntimeError("primary failed")


class FallbackTTSProvider:
    """Write distinguishable fallback audio for TTS fallback tests."""

    async def synthesize(self, text: str, language: str, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"fallback-audio")
        return output_path


class EchoQuestionLLMProvider:
    """Return the received question so tests can assert final transcript routing."""

    async def generate_answer(self, language: str, question: str):
        """Echo question text back as the generated answer."""

        return question


def _build_empty_pipeline(tmp_path: Path, llm_provider) -> VoicePipeline:
    """Build a pipeline with no local DB rows."""

    db_path = tmp_path / "knowledge.sqlite3"
    output_dir = tmp_path / "out"
    knowledge_base = KnowledgeBase(db_path, retrieval_mode="hybrid", confidence_low=0.55)
    knowledge_base.ensure_schema()
    return VoicePipeline(
        knowledge_base=knowledge_base,
        stt_provider=FakeSTTProvider(),
        llm_provider=llm_provider,
        fallback_llm_provider=None,
        primary_tts_provider=FakeTTSProvider(),
        fallback_tts_provider=None,
        output_dir=output_dir,
        timeout_seconds=3.0,
        llm_timeout_seconds=0.2,
        llm_direct_min_query_tokens=1,
    )


def _build_pipeline_with_rows(tmp_path: Path, rows: list[tuple[str, str, str, str, str, str]]) -> VoicePipeline:
    """Build a pipeline backed by explicit curated rows."""

    import sqlite3

    db_path = tmp_path / "knowledge.sqlite3"
    output_dir = tmp_path / "out"
    knowledge_base = KnowledgeBase(db_path, retrieval_mode="hybrid", confidence_low=0.55)
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
        stt_provider=FakeSTTProvider(),
        llm_provider=DirectAnswerLLMProvider(),
        fallback_llm_provider=None,
        primary_tts_provider=FakeTTSProvider(),
        fallback_tts_provider=None,
        output_dir=output_dir,
        timeout_seconds=3.0,
        llm_direct_min_query_tokens=1,
    )


def test_pipeline_uses_local_db_and_renders_audio(tmp_path: Path) -> None:
    """A local DB match should produce a response and an audio file."""

    db_path = tmp_path / "knowledge.sqlite3"
    output_dir = tmp_path / "out"
    knowledge_base = KnowledgeBase(db_path)
    knowledge_base.ensure_schema()
    knowledge_base.seed_demo_rows()

    pipeline = VoicePipeline(
        knowledge_base=knowledge_base,
        stt_provider=FakeSTTProvider(),
        llm_provider=FakeLLMProvider(),
        fallback_llm_provider=None,
        primary_tts_provider=FakeTTSProvider(),
        fallback_tts_provider=None,
        output_dir=output_dir,
        timeout_seconds=3.0,
    )

    response = asyncio.run(pipeline.process_transcription("when can i contact support office", "en"))

    assert response["resolved_source"] in {"local_db", "llm_direct", "fallback"}
    assert Path(response["audio_output_path"]).exists()


def test_pipeline_shortens_verbose_local_db_answers(tmp_path: Path) -> None:
    """Local DB answers should be compacted before synthesis."""

    db_path = tmp_path / "knowledge.sqlite3"
    output_dir = tmp_path / "out"
    knowledge_base = KnowledgeBase(db_path)
    knowledge_base.ensure_schema()

    import sqlite3

    with sqlite3.connect(knowledge_base.db_path) as connection:
        connection.execute(
            """
            INSERT INTO knowledge_base(language, keywords, response)
            VALUES (?, ?, ?)
            """,
            (
                "en",
                "example long answer",
                "Sentence one. Sentence two. Sentence three. Sentence four.",
            ),
        )
        connection.commit()

    pipeline = VoicePipeline(
        knowledge_base=knowledge_base,
        stt_provider=FakeSTTProvider(),
        llm_provider=FakeLLMProvider(),
        fallback_llm_provider=None,
        primary_tts_provider=FakeTTSProvider(),
        fallback_tts_provider=None,
        output_dir=output_dir,
        timeout_seconds=3.0,
    )

    response = asyncio.run(pipeline.process_transcription("example long answer", "en"))

    assert response["resolved_source"] == "local_db"
    assert response["response_text"] == "Sentence one."


def test_pipeline_short_answer_prefers_sentence_with_requested_level(tmp_path: Path) -> None:
    """Spoken compaction should keep the sentence that directly answers level questions."""

    db_path = tmp_path / "knowledge.sqlite3"
    output_dir = tmp_path / "out"
    knowledge_base = KnowledgeBase(db_path)
    knowledge_base.ensure_schema()

    import sqlite3

    with sqlite3.connect(knowledge_base.db_path) as connection:
        connection.execute(
            """
            INSERT INTO knowledge_base(language, keywords, response, source_id, section, question)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "en",
                "required english proficiency level,ielts 6.0,admission english requirement",
                "The program is taught in English. Applicants must meet IELTS 6.0 or an equivalent Level 4/6 certificate.",
                "web-012",
                "Admission",
                "What is the required English proficiency level for admission?",
            ),
        )
        connection.commit()

    pipeline = VoicePipeline(
        knowledge_base=knowledge_base,
        stt_provider=FakeSTTProvider(),
        llm_provider=FakeLLMProvider(),
        fallback_llm_provider=None,
        primary_tts_provider=FakeTTSProvider(),
        fallback_tts_provider=None,
        output_dir=output_dir,
        timeout_seconds=3.0,
    )

    response = asyncio.run(
        pipeline.process_transcription("What is the required English proficiency level for admission?", "en")
    )

    assert response["resolved_source"] == "local_db"
    assert response["response_text"] == "Applicants need IELTS 6.0 or an equivalent Level 4/6 English certificate."


def test_pipeline_uses_spoken_override_for_majors_without_ellipsis(tmp_path: Path) -> None:
    """Majors answers should be complete and not end with spoken ellipses."""

    pipeline = _build_pipeline_with_rows(
        tmp_path,
        [
            (
                "en",
                "what majors does greenwich vietnam offer,majors greenwich vietnam",
                "Greenwich Vietnam currently offers the following majors: Information Technology, Graphic & Digital Design, Business Administration, Marketing Management (Specialized Major), Event Management (Specialized Major), Communication Management (Specialized Major), International Business (Specialized Major), Logistics and Supply Chain Management (Specialized Major), Multimedia Communication.",
                "web-004",
                "Programs",
                "What majors does Greenwich Vietnam offer?",
            ),
        ],
    )

    response = asyncio.run(pipeline.process_transcription("What major does it offer?", "en"))

    assert response["resolved_source"] == "local_db"
    assert "Multimedia Communication" in response["response_text"]
    assert "..." not in response["response_text"]


def test_pipeline_routes_graduation_ielts_to_graduation_answer(tmp_path: Path) -> None:
    """Combined graduation IELTS questions should return two separated facts."""

    pipeline = _build_pipeline_with_rows(
        tmp_path,
        [
            (
                "en",
                "graduation requirements ielts mandatory",
                "The graduation requirements are long. IELTS is not mandatory for admission.",
                "web-009",
                "Graduation",
                "What are the graduation requirements? Is IELTS mandatory?",
            ),
            (
                "en",
                "ielts mandatory required admission",
                "IELTS is not mandatory for admission.",
                "web-009-ielts",
                "Graduation",
                "Is IELTS mandatory?",
            ),
            (
                "en",
                "english proficiency level admission ielts 6.0",
                "Applicants must meet IELTS 6.0.",
                "web-012",
                "Admission",
                "What is the required English proficiency level for admission?",
            ),
        ],
    )

    response = asyncio.run(pipeline.process_transcription("What are the graduation requirements? Is IELTS mandatory?", "en"))

    assert response["resolved_source"] == "local_db"
    assert "Graduation requirements include" in response["response_text"]
    assert "IELTS is not mandatory at admission" in response["response_text"]


def test_pipeline_routes_graduation_only_to_graduation_answer(tmp_path: Path) -> None:
    """Graduation-only questions should not include the IELTS admission pathway."""

    pipeline = _build_pipeline_with_rows(
        tmp_path,
        [
            (
                "en",
                "graduation requirements ielts mandatory",
                "The graduation requirements are long. IELTS is not mandatory for admission.",
                "web-009",
                "Graduation",
                "What are the graduation requirements?",
            ),
            (
                "en",
                "ielts mandatory required admission",
                "IELTS is not mandatory for admission.",
                "web-009-ielts",
                "Graduation",
                "Is IELTS mandatory?",
            ),
        ],
    )

    response = asyncio.run(pipeline.process_transcription("What are the graduation requirements?", "en"))

    assert response["resolved_source"] == "local_db"
    assert response["response_text"] == (
        "Graduation requirements include specialized knowledge, professional skills, and graduation English proficiency."
    )


def test_pipeline_routes_ielts_mandatory_to_ielts_only_answer(tmp_path: Path) -> None:
    """IELTS-only questions should not use the graduation requirements answer."""

    pipeline = _build_pipeline_with_rows(
        tmp_path,
        [
            (
                "en",
                "graduation requirements",
                "Graduation requirements are long.",
                "web-009",
                "Graduation",
                "What are the graduation requirements?",
            ),
            (
                "en",
                "ielts mandatory required admission",
                "IELTS is not mandatory for admission.",
                "web-009-ielts",
                "Graduation",
                "Is IELTS mandatory?",
            ),
        ],
    )

    response = asyncio.run(pipeline.process_transcription("Is IELTS mandatory?", "en"))

    assert response["resolved_source"] == "local_db"
    assert response["response_text"] == (
        "IELTS is not mandatory at admission; students may submit IELTS if available or use the integrated English pathway."
    )


def test_pipeline_spoken_duration_override_keeps_decimal_complete(tmp_path: Path) -> None:
    """Duration answer should not be cut at the 1.5 year decimal."""

    pipeline = _build_pipeline_with_rows(
        tmp_path,
        [
            (
                "en",
                "duration bachelor master program",
                "The program duration at Greenwich Vietnam is 3 years for Bachelor's programs and 1.5 years for Master's programs.",
                "web-006",
                "Duration",
                "What is the duration of the program?",
            ),
        ],
    )

    response = asyncio.run(
        pipeline.process_transcription("What is the duration of the bachelor's program and the master's program?", "en")
    )

    assert response["resolved_source"] == "local_db"
    assert "1.5 years" in response["response_text"]
    assert not response["response_text"].endswith("1.")


def test_pipeline_spoken_campus_override_does_not_cut_abbreviation(tmp_path: Path) -> None:
    """Campus answer should list all cities without ending at TP."""

    pipeline = _build_pipeline_with_rows(
        tmp_path,
        [
            (
                "en",
                "campus campuses hanoi da nang ho chi minh can tho",
                "Greenwich Vietnam has 4 campuses in Hanoi, Da Nang, Ho Chi Minh City, and Can Tho.",
                "web-017",
                "Campus",
                "How many campuses does Greenwich Vietnam have?",
            ),
        ],
    )

    response = asyncio.run(pipeline.process_transcription("How many campuses does Greenwich Vietnam have?", "en"))

    assert response["resolved_source"] == "local_db"
    assert "Can Tho" in response["response_text"]
    assert not response["response_text"].endswith("TP.")


def test_pipeline_routes_admission_proficiency_to_web_012(tmp_path: Path) -> None:
    """Admission proficiency questions should still use web-012."""

    pipeline = _build_pipeline_with_rows(
        tmp_path,
        [
            (
                "en",
                "graduation requirements ielts mandatory",
                "The graduation requirements are long. IELTS is not mandatory for admission.",
                "web-009",
                "Graduation",
                "What are the graduation requirements? Is IELTS mandatory?",
            ),
            (
                "en",
                "english proficiency level admission ielts 6.0",
                "Applicants must meet IELTS 6.0.",
                "web-012",
                "Admission",
                "What is the required English proficiency level for admission?",
            ),
        ],
    )

    response = asyncio.run(pipeline.process_transcription("What is the required English proficiency level for admission?", "en"))

    assert response["resolved_source"] == "local_db"
    assert response["response_text"] == "Applicants need IELTS 6.0 or an equivalent Level 4/6 English certificate."


def test_pipeline_routes_semester_amount_to_tuition_amount(tmp_path: Path) -> None:
    """Amount questions per semester should use tuition amount instead of payment cadence."""

    pipeline = _build_pipeline_with_rows(
        tmp_path,
        [
            (
                "en",
                "tuition fee total program greenwich vietnam semester amount",
                "The total tuition fee is 450,000,000 VND. The major tuition fee is 50,000,000 VND per semester.",
                "web-014",
                "Tuition",
                "What is the total tuition fee for the entire program at Greenwich Vietnam?",
            ),
            (
                "en",
                "tuition fees paid monthly per semester academic year",
                "Tuition fees will be paid per semester. For detailed tuition fee information, please refer to: https://greenwich.edu.vn/hoc-phi/",
                "web-033",
                "Tuition",
                "Are tuition fees paid monthly, per semester, or per academic year?",
            ),
        ],
    )

    response = asyncio.run(pipeline.process_transcription("How much per semester?", "en"))

    assert response["resolved_source"] == "local_db"
    assert "50,000,000 VND per semester" in response["response_text"]


def test_pipeline_omits_urls_and_ellipsis_from_spoken_db_answer(tmp_path: Path) -> None:
    """Generic spoken DB compaction should not expose raw URLs or trailing ellipses."""

    pipeline = _build_pipeline_with_rows(
        tmp_path,
        [
            (
                "en",
                "application documents admission program",
                "To prepare your application, please register online here: https://tuyensinh.greenwich.edu.vn/ More details are available from staff.",
                "web-031",
                "Admission",
                "What documents are required for admission to the program?",
            ),
        ],
    )

    response = asyncio.run(pipeline.process_transcription("What documents are required for admission?", "en"))

    assert "http" not in response["response_text"]
    assert "..." not in response["response_text"]


def test_pipeline_uses_direct_llm_when_db_misses(tmp_path: Path) -> None:
    """A DB miss should route directly to LLM fallback."""

    db_path = tmp_path / "knowledge.sqlite3"
    output_dir = tmp_path / "out"
    knowledge_base = KnowledgeBase(db_path)
    knowledge_base.ensure_schema()

    pipeline = VoicePipeline(
        knowledge_base=knowledge_base,
        stt_provider=FakeSTTProvider(),
        llm_provider=DirectAnswerLLMProvider(),
        fallback_llm_provider=None,
        primary_tts_provider=FakeTTSProvider(),
        fallback_tts_provider=None,
        output_dir=output_dir,
        timeout_seconds=3.0,
        llm_direct_min_query_tokens=1,
    )

    response = asyncio.run(pipeline.process_transcription("how do i reset password", "en"))

    assert response["resolved_source"] == "llm_direct"
    assert "reset" in response["response_text"].lower()


def test_pipeline_compacts_direct_llm_answers(tmp_path: Path) -> None:
    """Direct LLM answers should stay short enough for voice output."""

    db_path = tmp_path / "knowledge.sqlite3"
    output_dir = tmp_path / "out"
    knowledge_base = KnowledgeBase(db_path)
    knowledge_base.ensure_schema()

    pipeline = VoicePipeline(
        knowledge_base=knowledge_base,
        stt_provider=FakeSTTProvider(),
        llm_provider=LongLLMProvider(),
        fallback_llm_provider=None,
        primary_tts_provider=FakeTTSProvider(),
        fallback_tts_provider=None,
        output_dir=output_dir,
        timeout_seconds=3.0,
        llm_direct_min_query_tokens=1,
    )

    response = asyncio.run(pipeline.process_transcription("explain quantum entanglement", "en"))

    assert response["resolved_source"] == "llm_direct"
    assert "Sentence two" in response["response_text"]
    assert "Sentence three" not in response["response_text"]


def test_pipeline_logs_direct_llm_compaction_decision(tmp_path: Path, caplog) -> None:
    """Direct LLM compaction diagnostics should show raw and compacted lengths."""

    db_path = tmp_path / "knowledge.sqlite3"
    output_dir = tmp_path / "out"
    knowledge_base = KnowledgeBase(db_path)
    knowledge_base.ensure_schema()
    pipeline = VoicePipeline(
        knowledge_base=knowledge_base,
        stt_provider=FakeSTTProvider(),
        llm_provider=LongLLMProvider(),
        fallback_llm_provider=None,
        primary_tts_provider=FakeTTSProvider(),
        fallback_tts_provider=None,
        output_dir=output_dir,
        timeout_seconds=3.0,
        llm_direct_min_query_tokens=1,
    )

    caplog.set_level(logging.INFO)
    response = asyncio.run(pipeline.process_transcription("explain quantum entanglement", "en"))

    assert response["resolved_source"] == "llm_direct"
    assert any("Direct LLM compaction diagnostics:" in record.getMessage() for record in caplog.records)
    assert any("decision=first_two_sentences" in record.getMessage() for record in caplog.records)


def test_pipeline_logs_incomplete_direct_llm_rejection(tmp_path: Path, caplog) -> None:
    """Compaction diagnostics should show when an incomplete fragment is rejected."""

    db_path = tmp_path / "knowledge.sqlite3"
    knowledge_base = KnowledgeBase(db_path)
    knowledge_base.ensure_schema()
    pipeline = VoicePipeline(
        knowledge_base=knowledge_base,
        stt_provider=FakeSTTProvider(),
        llm_provider=DirectAnswerLLMProvider(),
        fallback_llm_provider=None,
        primary_tts_provider=FakeTTSProvider(),
        fallback_tts_provider=None,
        output_dir=tmp_path / "out",
        timeout_seconds=3.0,
        llm_direct_min_query_tokens=1,
    )

    caplog.set_level(logging.INFO)
    response_text = pipeline._compact_direct_llm_response("Recent global events include ongoing conflicts in")

    assert response_text == ""
    assert any("decision=rejected_incomplete" in record.getMessage() for record in caplog.records)


def test_pipeline_logs_tts_diagnostics(tmp_path: Path, caplog, monkeypatch) -> None:
    """TTS diagnostics should expose input length and output byte size."""

    db_path = tmp_path / "knowledge.sqlite3"
    output_dir = tmp_path / "out"
    knowledge_base = KnowledgeBase(db_path)
    knowledge_base.ensure_schema()
    pipeline = VoicePipeline(
        knowledge_base=knowledge_base,
        stt_provider=FakeSTTProvider(),
        llm_provider=DirectAnswerLLMProvider(),
        fallback_llm_provider=None,
        primary_tts_provider=FakeTTSProvider(),
        fallback_tts_provider=None,
        output_dir=output_dir,
        timeout_seconds=3.0,
        llm_direct_min_query_tokens=1,
    )

    monkeypatch.setenv("VOICE_LOOP_DEBUG_AUDIO_IO", "true")
    caplog.set_level(logging.INFO)
    response = asyncio.run(pipeline.process_transcription("how do i reset password", "en"))

    assert Path(response["audio_output_path"]).stat().st_size > 0
    assert any("TTS synthesis diagnostics:" in record.getMessage() for record in caplog.records)
    assert any("TTS synthesis result:" in record.getMessage() and "bytes=" in record.getMessage() for record in caplog.records)


def test_pipeline_primary_tts_timeout_triggers_fallback(tmp_path: Path) -> None:
    """Slow primary TTS should fall back to the secondary provider."""

    db_path = tmp_path / "knowledge.sqlite3"
    output_dir = tmp_path / "out"
    knowledge_base = KnowledgeBase(db_path)
    knowledge_base.ensure_schema()
    pipeline = VoicePipeline(
        knowledge_base=knowledge_base,
        stt_provider=FakeSTTProvider(),
        llm_provider=DirectAnswerLLMProvider(),
        fallback_llm_provider=None,
        primary_tts_provider=SlowTTSProvider(),
        fallback_tts_provider=FallbackTTSProvider(),
        output_dir=output_dir,
        timeout_seconds=0.01,
        llm_direct_min_query_tokens=1,
    )

    response = asyncio.run(pipeline.process_transcription("how do i reset password", "en"))

    assert Path(response["audio_output_path"]).read_bytes() == b"fallback-audio"


def test_pipeline_fallback_tts_timeout_is_capped(tmp_path: Path, caplog) -> None:
    """Fallback TTS should use a bounded timeout when primary fails."""

    db_path = tmp_path / "knowledge.sqlite3"
    output_dir = tmp_path / "out"
    knowledge_base = KnowledgeBase(db_path)
    knowledge_base.ensure_schema()
    pipeline = VoicePipeline(
        knowledge_base=knowledge_base,
        stt_provider=FakeSTTProvider(),
        llm_provider=DirectAnswerLLMProvider(),
        fallback_llm_provider=None,
        primary_tts_provider=FailingTempWritingTTSProvider(),
        fallback_tts_provider=FallbackTTSProvider(),
        output_dir=output_dir,
        timeout_seconds=8.0,
        llm_direct_min_query_tokens=1,
    )

    caplog.set_level(logging.INFO)
    response = asyncio.run(pipeline.process_transcription("how do i reset password", "en"))

    assert Path(response["audio_output_path"]).read_bytes() == b"fallback-audio"
    assert any(
        "provider=fallback" in record.getMessage() and "timeout_seconds=5.0" in record.getMessage()
        for record in caplog.records
    )


def test_pipeline_primary_tts_temp_output_does_not_overwrite_fallback(tmp_path: Path) -> None:
    """Primary temp output should not replace fallback output after primary failure."""

    db_path = tmp_path / "knowledge.sqlite3"
    output_dir = tmp_path / "out"
    knowledge_base = KnowledgeBase(db_path)
    knowledge_base.ensure_schema()
    pipeline = VoicePipeline(
        knowledge_base=knowledge_base,
        stt_provider=FakeSTTProvider(),
        llm_provider=DirectAnswerLLMProvider(),
        fallback_llm_provider=None,
        primary_tts_provider=FailingTempWritingTTSProvider(),
        fallback_tts_provider=FallbackTTSProvider(),
        output_dir=output_dir,
        timeout_seconds=8.0,
        llm_direct_min_query_tokens=1,
    )

    response = asyncio.run(pipeline.process_transcription("how do i reset password", "en"))
    audio_path = Path(response["audio_output_path"])

    assert audio_path.read_bytes() == b"fallback-audio"
    assert not audio_path.with_name(f"{audio_path.stem}.primary{audio_path.suffix}").exists()


def test_pipeline_uses_adaptive_llm_timeout_budget(tmp_path: Path) -> None:
    """Direct LLM budget should be shorter by default and longer for current queries."""

    pipeline = _build_empty_pipeline(tmp_path, DirectAnswerLLMProvider())
    pipeline.llm_timeout_seconds = 15.0

    assert pipeline._llm_timeout_for_query("Tell me a short joke") == 8.0
    assert pipeline._llm_timeout_for_query("What is the latest weather in Ha Noi?") == 10.0


def test_pipeline_rejects_incomplete_llm_fragments(tmp_path: Path) -> None:
    """Incomplete direct LLM fragments should not be spoken."""

    class FragmentLLMProvider:
        async def generate_answer(self, language: str, question: str):
            return "Để trả lời câu"

    db_path = tmp_path / "knowledge.sqlite3"
    output_dir = tmp_path / "out"
    knowledge_base = KnowledgeBase(db_path)
    knowledge_base.ensure_schema()

    pipeline = VoicePipeline(
        knowledge_base=knowledge_base,
        stt_provider=FakeSTTProvider(),
        llm_provider=FragmentLLMProvider(),
        fallback_llm_provider=None,
        primary_tts_provider=FakeTTSProvider(),
        fallback_tts_provider=None,
        output_dir=output_dir,
        timeout_seconds=3.0,
        llm_direct_min_query_tokens=1,
    )

    response = asyncio.run(pipeline.process_transcription("Ai thắng trận bóng gần nhất?", "vi"))

    assert response["resolved_source"] == "fallback"
    assert "Để trả lời câu" not in response["response_text"]


def test_pipeline_rejects_compacted_incomplete_llm_endings(tmp_path: Path) -> None:
    """Compacted LLM answers should not speak dangling fragments."""

    class FragmentLLMProvider:
        async def generate_answer(self, language: str, question: str):
            return "Recent global events include ongoing conflicts in"

    db_path = tmp_path / "knowledge.sqlite3"
    output_dir = tmp_path / "out"
    knowledge_base = KnowledgeBase(db_path)
    knowledge_base.ensure_schema()

    pipeline = VoicePipeline(
        knowledge_base=knowledge_base,
        stt_provider=FakeSTTProvider(),
        llm_provider=FragmentLLMProvider(),
        fallback_llm_provider=None,
        primary_tts_provider=FakeTTSProvider(),
        fallback_tts_provider=None,
        output_dir=output_dir,
        timeout_seconds=3.0,
        llm_direct_min_query_tokens=1,
    )

    response = asyncio.run(pipeline.process_transcription("summarize the latest", "en"))

    assert response["resolved_source"] == "fallback"
    assert "ongoing conflicts in" not in response["response_text"]


def test_pipeline_rejects_announced_llm_fragment(tmp_path: Path) -> None:
    """Generated clauses ending at announced should not be spoken."""

    class FragmentLLMProvider:
        async def generate_answer(self, language: str, question: str):
            return "This week, NVIDIA and Microsoft announced"

    db_path = tmp_path / "knowledge.sqlite3"
    output_dir = tmp_path / "out"
    knowledge_base = KnowledgeBase(db_path)
    knowledge_base.ensure_schema()

    pipeline = VoicePipeline(
        knowledge_base=knowledge_base,
        stt_provider=FakeSTTProvider(),
        llm_provider=FragmentLLMProvider(),
        fallback_llm_provider=None,
        primary_tts_provider=FakeTTSProvider(),
        fallback_tts_provider=None,
        output_dir=output_dir,
        timeout_seconds=3.0,
        llm_direct_min_query_tokens=1,
    )

    response = asyncio.run(pipeline.process_transcription("summarize the latest AI news this week", "en"))

    assert response["resolved_source"] == "fallback"
    assert "announced" not in response["response_text"]


def test_pipeline_allows_complete_llm_answer_ending_in_year(tmp_path: Path) -> None:
    """Complete answers ending in a year should not be mistaken for cut decimal fragments."""

    class YearEndingLLMProvider:
        async def generate_answer(self, language: str, question: str):
            return "Manchester United won 3-0 on May 24, 2026."

    db_path = tmp_path / "knowledge.sqlite3"
    output_dir = tmp_path / "out"
    knowledge_base = KnowledgeBase(db_path)
    knowledge_base.ensure_schema()
    pipeline = VoicePipeline(
        knowledge_base=knowledge_base,
        stt_provider=FakeSTTProvider(),
        llm_provider=YearEndingLLMProvider(),
        fallback_llm_provider=None,
        primary_tts_provider=FakeTTSProvider(),
        fallback_tts_provider=None,
        output_dir=output_dir,
        timeout_seconds=3.0,
        llm_direct_min_query_tokens=1,
    )

    response = asyncio.run(pipeline.process_transcription("who won the latest match", "en"))

    assert response["resolved_source"] == "llm_direct"
    assert response["response_text"].endswith("2026.")


def test_pipeline_allows_complete_llm_answer_ending_in_score(tmp_path: Path) -> None:
    """Complete answers ending in a score should not be mistaken for numeric fragments."""

    class ScoreEndingLLMProvider:
        async def generate_answer(self, language: str, question: str):
            return "Manchester United won against Brighton 3-0."

    db_path = tmp_path / "knowledge.sqlite3"
    output_dir = tmp_path / "out"
    knowledge_base = KnowledgeBase(db_path)
    knowledge_base.ensure_schema()
    pipeline = VoicePipeline(
        knowledge_base=knowledge_base,
        stt_provider=FakeSTTProvider(),
        llm_provider=ScoreEndingLLMProvider(),
        fallback_llm_provider=None,
        primary_tts_provider=FakeTTSProvider(),
        fallback_tts_provider=None,
        output_dir=output_dir,
        timeout_seconds=3.0,
        llm_direct_min_query_tokens=1,
    )

    response = asyncio.run(pipeline.process_transcription("who won the latest match", "en"))

    assert response["resolved_source"] == "llm_direct"
    assert response["response_text"].endswith("3-0.")


def test_pipeline_rejects_short_vietnamese_llm_fragment(tmp_path: Path) -> None:
    """Short Vietnamese weather fragments should not be spoken as answers."""

    class FragmentLLMProvider:
        async def generate_answer(self, language: str, question: str):
            return "Thoi tiet Ha"

    db_path = tmp_path / "knowledge.sqlite3"
    output_dir = tmp_path / "out"
    knowledge_base = KnowledgeBase(db_path)
    knowledge_base.ensure_schema()

    pipeline = VoicePipeline(
        knowledge_base=knowledge_base,
        stt_provider=FakeSTTProvider(),
        llm_provider=FragmentLLMProvider(),
        fallback_llm_provider=None,
        primary_tts_provider=FakeTTSProvider(),
        fallback_tts_provider=None,
        output_dir=output_dir,
        timeout_seconds=3.0,
        llm_direct_min_query_tokens=1,
    )

    response = asyncio.run(pipeline.process_transcription("thoi tiet ha noi", "vi"))

    assert response["resolved_source"] == "fallback"
    assert "Thoi tiet Ha" not in response["response_text"]


def test_pipeline_skips_direct_llm_for_noisy_pose_price_query(tmp_path: Path) -> None:
    """Known misheard fragments should not trigger grounded LLM answers."""

    db_path = tmp_path / "knowledge.sqlite3"
    output_dir = tmp_path / "out"
    knowledge_base = KnowledgeBase(db_path)
    knowledge_base.ensure_schema()

    pipeline = VoicePipeline(
        knowledge_base=knowledge_base,
        stt_provider=FakeSTTProvider(),
        llm_provider=LongLLMProvider(),
        fallback_llm_provider=None,
        primary_tts_provider=FakeTTSProvider(),
        fallback_tts_provider=None,
        output_dir=output_dir,
        timeout_seconds=3.0,
        llm_direct_min_query_tokens=1,
    )

    response = asyncio.run(
        pipeline.process_transcription_with_metrics(
            "How much pose Simulator",
            "en",
            stt_elapsed_ms=None,
            language_confidence=0.70,
            language_reason="en_marker_majority",
        )
    )

    assert response["resolved_source"] == "fallback"
    assert "repeat" in response["response_text"].lower()


def test_pipeline_reprompts_long_noisy_weak_db_capture(tmp_path: Path) -> None:
    """Long noisy captures should not route DB when match evidence is weak."""

    pipeline = _build_pipeline_with_rows(
        tmp_path,
        [
            (
                "en",
                "how much per semester",
                "The semester tuition is 40 million VND.",
                "tuition-001",
                "Tuition",
                "How much per semester?",
            ),
        ],
    )

    response = asyncio.run(
        pipeline.process_transcription_with_metrics(
            "How much basimexers today I mean hoc phi maybe simulator",
            "en",
            stt_elapsed_ms=None,
            language_confidence=0.55,
            language_reason="marker_tie",
        )
    )

    assert response["resolved_source"] == "fallback"
    assert "40 million" not in response["response_text"]
    assert "repeat" in response["response_text"].lower()


def test_pipeline_preserves_long_clean_db_question_with_embedded_phrase(tmp_path: Path) -> None:
    """A long question with a clean exact phrase should still route locally."""

    pipeline = _build_pipeline_with_rows(
        tmp_path,
        [
            (
                "en",
                "how much per semester",
                "The semester tuition is 40 million VND.",
                "tuition-001",
                "Tuition",
                "How much per semester?",
            ),
        ],
    )

    response = asyncio.run(
        pipeline.process_transcription_with_metrics(
            "Could you tell me how much per semester at Greenwich Vietnam for a new student",
            "en",
            stt_elapsed_ms=None,
            language_confidence=0.55,
            language_reason="marker_tie",
        )
    )

    assert response["resolved_source"] == "local_db"
    assert "40 million" in response["response_text"]


def test_pipeline_preserves_long_clean_domain_questions(tmp_path: Path) -> None:
    """Long clean domain questions should not be blocked by noisy-capture guards."""

    pipeline = _build_pipeline_with_rows(
        tmp_path,
        [
            (
                "en",
                "what majors does greenwich vietnam offer",
                "Greenwich Vietnam offers IT, business, design, and communication majors.",
                "web-004",
                "Programs",
                "What majors does Greenwich Vietnam offer?",
            ),
            (
                "en",
                "how many campuses does greenwich vietnam have",
                "Greenwich Vietnam has 4 campuses.",
                "web-017",
                "Campus",
                "How many campuses does Greenwich Vietnam have?",
            ),
            (
                "en",
                "admission requirements greenwich vietnam",
                "Admission requirements include academic records and English proficiency.",
                "web-011",
                "Admission",
                "What are the admission requirements for Greenwich Vietnam?",
            ),
        ],
    )

    prompts = [
        ("Could you tell me what majors does Greenwich Vietnam offer for new students", "Information Technology"),
        ("Could you tell me how many campuses does Greenwich Vietnam have right now", "4 campuses"),
        ("Could you explain admission requirements Greenwich Vietnam for applicants", "academic records"),
    ]
    for prompt, expected_text in prompts:
        response = asyncio.run(
            pipeline.process_transcription_with_metrics(
                prompt,
                "en",
                stt_elapsed_ms=None,
                language_confidence=0.55,
                language_reason="marker_tie",
            )
        )
        assert response["resolved_source"] == "local_db"
        assert expected_text in response["response_text"]


def test_pipeline_long_uncertain_out_of_domain_turn_does_not_update_context_anchor(tmp_path: Path) -> None:
    """Long uncertain out-of-domain turns should not replace the current DB anchor."""

    pipeline = _build_pipeline_with_rows(
        tmp_path,
        [
            (
                "en",
                "greenwich vietnam",
                "Greenwich Vietnam is an international joint program.",
                "web-001",
                "Overview",
                "What is Greenwich Vietnam?",
            ),
        ],
    )

    first_turn = asyncio.run(pipeline.process_transcription("Tell me about Greenwich Vietnam", "en"))
    second_turn = asyncio.run(
        pipeline.process_transcription_with_metrics(
            "Manchester United latest score and maybe Greenwich Vietnam tuition random words",
            "en",
            stt_elapsed_ms=None,
            language_confidence=0.45,
            language_reason="marker_tie",
        )
    )

    assert first_turn["resolved_source"] == "local_db"
    assert second_turn["resolved_source"] != "local_db"
    assert pipeline._context_anchor is not None
    assert pipeline._context_anchor.source_id == "web-001"


def test_pipeline_clarifies_english_vietnam_misrecognition(tmp_path: Path) -> None:
    """The common English-for-Greenwich misrecognition should ask a choice."""

    pipeline = _build_empty_pipeline(tmp_path, DirectAnswerLLMProvider())

    response = asyncio.run(pipeline.process_transcription("What is English Vietnam", "en"))

    assert response["resolved_source"] == "fallback"
    assert response["response_text"] == "Did you mean Greenwich Vietnam, or English in Vietnam?"


def test_pipeline_prepares_multiline_text_for_tts() -> None:
    """TTS input should be flattened for generated poems."""

    text = VoicePipeline._prepare_text_for_tts("Line one,\nLine two.\nLine three.")

    assert "\n" not in text
    assert text == "Line one, Line two. Line three."


def test_pipeline_falls_back_when_llm_fails(tmp_path: Path) -> None:
    """When direct LLM fails, pipeline should return hard fallback."""

    db_path = tmp_path / "knowledge.sqlite3"
    output_dir = tmp_path / "out"
    knowledge_base = KnowledgeBase(db_path)
    knowledge_base.ensure_schema()

    pipeline = VoicePipeline(
        knowledge_base=knowledge_base,
        stt_provider=FakeSTTProvider(),
        llm_provider=FailingLLMProvider(),
        fallback_llm_provider=None,
        primary_tts_provider=FakeTTSProvider(),
        fallback_tts_provider=None,
        output_dir=output_dir,
        timeout_seconds=3.0,
        llm_direct_min_query_tokens=1,
    )

    response = asyncio.run(pipeline.process_transcription("how do i reset password", "en"))

    assert response["resolved_source"] == "fallback"


def test_pipeline_applies_transcript_cheat_when_context_matches(tmp_path: Path) -> None:
    """Context-gated cheat rules should rewrite known STT mishears."""

    db_path = tmp_path / "knowledge.sqlite3"
    output_dir = tmp_path / "out"
    knowledge_base = KnowledgeBase(db_path)
    knowledge_base.ensure_schema()

    pipeline = VoicePipeline(
        knowledge_base=knowledge_base,
        stt_provider=FakeSTTProvider(),
        llm_provider=EchoQuestionLLMProvider(),
        fallback_llm_provider=None,
        primary_tts_provider=FakeTTSProvider(),
        fallback_tts_provider=None,
        output_dir=output_dir,
        timeout_seconds=3.0,
        llm_direct_min_query_tokens=1,
        transcript_cheats=(
            TranscriptCheatRule(
                wrong_phrase="remix",
                corrected_phrase="greenwich",
                required_context_terms=("university", "vietnam"),
            ),
        ),
    )

    response = asyncio.run(pipeline.process_transcription("where is remix university", "en"))

    assert response["resolved_source"] == "llm_direct"
    assert "greenwich" in response["response_text"].lower()
    assert "remix" not in response["response_text"].lower()


def test_pipeline_does_not_apply_transcript_cheat_without_context(tmp_path: Path) -> None:
    """Context-gated cheat rules should not alter unrelated transcripts."""

    db_path = tmp_path / "knowledge.sqlite3"
    output_dir = tmp_path / "out"
    knowledge_base = KnowledgeBase(db_path)
    knowledge_base.ensure_schema()

    pipeline = VoicePipeline(
        knowledge_base=knowledge_base,
        stt_provider=FakeSTTProvider(),
        llm_provider=EchoQuestionLLMProvider(),
        fallback_llm_provider=None,
        primary_tts_provider=FakeTTSProvider(),
        fallback_tts_provider=None,
        output_dir=output_dir,
        timeout_seconds=3.0,
        llm_direct_min_query_tokens=1,
        transcript_cheats=(
            TranscriptCheatRule(
                wrong_phrase="remix",
                corrected_phrase="greenwich",
                required_context_terms=("university", "vietnam"),
            ),
        ),
    )

    response = asyncio.run(pipeline.process_transcription("play my remix playlist", "en"))

    assert response["resolved_source"] == "llm_direct"
    assert "remix" in response["response_text"].lower()
    assert "greenwich" not in response["response_text"].lower()


def test_pipeline_logs_llm_failure_type_and_reason(tmp_path: Path, caplog) -> None:
    """Direct LLM failure logs should include exception type and non-empty reason."""

    db_path = tmp_path / "knowledge.sqlite3"
    output_dir = tmp_path / "out"
    knowledge_base = KnowledgeBase(db_path)
    knowledge_base.ensure_schema()

    pipeline = VoicePipeline(
        knowledge_base=knowledge_base,
        stt_provider=FakeSTTProvider(),
        llm_provider=FailingLLMProvider(),
        fallback_llm_provider=None,
        primary_tts_provider=FakeTTSProvider(),
        fallback_tts_provider=None,
        output_dir=output_dir,
        timeout_seconds=3.0,
        llm_direct_min_query_tokens=1,
    )

    caplog.set_level(logging.WARNING)
    asyncio.run(pipeline.process_transcription("how do i reset password", "en"))

    warning_messages = [record.getMessage() for record in caplog.records if record.levelno >= logging.WARNING]
    assert any("type=RuntimeError" in message for message in warning_messages)
    assert any("reason=intentional llm failure" in message for message in warning_messages)


def test_pipeline_logs_stage_timings_for_request_turn(tmp_path: Path, caplog) -> None:
    """Stage telemetry log should include STT, DB, LLM, and TTS timing labels."""

    db_path = tmp_path / "knowledge.sqlite3"
    output_dir = tmp_path / "out"
    knowledge_base = KnowledgeBase(db_path)
    knowledge_base.ensure_schema()

    pipeline = VoicePipeline(
        knowledge_base=knowledge_base,
        stt_provider=FakeSTTProvider(),
        llm_provider=DirectAnswerLLMProvider(),
        fallback_llm_provider=None,
        primary_tts_provider=FakeTTSProvider(),
        fallback_tts_provider=None,
        output_dir=output_dir,
        timeout_seconds=3.0,
        llm_direct_min_query_tokens=1,
    )

    caplog.set_level(logging.INFO)
    asyncio.run(
        pipeline.process_transcription_with_metrics(
            "how do i reset password",
            "en",
            stt_elapsed_ms=42.0,
        )
    )

    info_messages = [record.getMessage() for record in caplog.records if record.levelno == logging.INFO]
    stage_logs = [message for message in info_messages if message.startswith("Stage timings (ms):")]
    assert stage_logs
    stage_log = stage_logs[-1]
    assert "stt=42" in stage_log
    assert "db=" in stage_log
    assert "llm=" in stage_log
    assert "tts=" in stage_log


def test_pipeline_logs_token_guard_skip_diagnostics(tmp_path: Path, caplog) -> None:
    """Fallback diagnostics should identify when LLM direct is skipped by token guard."""

    db_path = tmp_path / "knowledge.sqlite3"
    output_dir = tmp_path / "out"
    knowledge_base = KnowledgeBase(db_path)
    knowledge_base.ensure_schema()

    pipeline = VoicePipeline(
        knowledge_base=knowledge_base,
        stt_provider=FakeSTTProvider(),
        llm_provider=DirectAnswerLLMProvider(),
        fallback_llm_provider=None,
        primary_tts_provider=FakeTTSProvider(),
        fallback_tts_provider=None,
        output_dir=output_dir,
        timeout_seconds=3.0,
        llm_direct_min_query_tokens=3,
    )

    caplog.set_level(logging.INFO)
    response = asyncio.run(pipeline.process_transcription("hi there", "en"))

    assert response["resolved_source"] == "fallback"
    info_messages = [record.getMessage() for record in caplog.records if record.levelno == logging.INFO]
    assert any("llm_status=skipped_token_guard" in message for message in info_messages)
    assert any("tokens=2 min_tokens=3" in message for message in info_messages)


def test_pipeline_logs_llm_timeout_failure_diagnostics(tmp_path: Path, caplog) -> None:
    """Fallback diagnostics should identify LLM attempts that fail by timeout."""

    db_path = tmp_path / "knowledge.sqlite3"
    output_dir = tmp_path / "out"
    knowledge_base = KnowledgeBase(db_path)
    knowledge_base.ensure_schema()

    pipeline = VoicePipeline(
        knowledge_base=knowledge_base,
        stt_provider=FakeSTTProvider(),
        llm_provider=TimeoutLLMProvider(),
        fallback_llm_provider=None,
        primary_tts_provider=FakeTTSProvider(),
        fallback_tts_provider=None,
        output_dir=output_dir,
        timeout_seconds=3.0,
        llm_timeout_seconds=0.01,
        llm_direct_min_query_tokens=1,
    )

    caplog.set_level(logging.INFO)
    response = asyncio.run(pipeline.process_transcription("how do i reset password", "en"))

    assert response["resolved_source"] == "fallback"
    messages = [record.getMessage() for record in caplog.records]
    assert any("llm_status=failed" in message for message in messages)
    assert any("llm_failure_type=TimeoutError" in message for message in messages)


def test_pipeline_logs_empty_llm_output_diagnostics(tmp_path: Path, caplog) -> None:
    """Fallback diagnostics should identify attempted LLM calls with empty output."""

    db_path = tmp_path / "knowledge.sqlite3"
    output_dir = tmp_path / "out"
    knowledge_base = KnowledgeBase(db_path)
    knowledge_base.ensure_schema()

    pipeline = VoicePipeline(
        knowledge_base=knowledge_base,
        stt_provider=FakeSTTProvider(),
        llm_provider=EmptyLLMProvider(),
        fallback_llm_provider=None,
        primary_tts_provider=FakeTTSProvider(),
        fallback_tts_provider=None,
        output_dir=output_dir,
        timeout_seconds=3.0,
        llm_direct_min_query_tokens=1,
    )

    caplog.set_level(logging.INFO)
    response = asyncio.run(pipeline.process_transcription("how do i reset password", "en"))

    assert response["resolved_source"] == "fallback"
    assert any("llm_status=empty_response" in record.getMessage() for record in caplog.records)


def test_pipeline_plays_thinking_cue_for_slow_direct_llm(tmp_path: Path) -> None:
    """Slow direct LLM calls should trigger the supplied transition cue once."""

    pipeline = _build_empty_pipeline(tmp_path, SlowDirectLLMProvider())
    cue_calls = 0

    async def cue_callback() -> None:
        nonlocal cue_calls
        cue_calls += 1

    response = asyncio.run(
        pipeline.process_transcription_with_metrics(
            "Explain quantum entanglement.",
            "en",
            stt_elapsed_ms=100.0,
            thinking_cue_delay_seconds=0.001,
            thinking_cue_callback=cue_callback,
        )
    )

    assert cue_calls == 1
    assert response["resolved_source"] == "llm_direct"
    assert response["response_text"] == "Here is the complete answer."


def test_pipeline_failed_llm_on_weak_query_reprompts_instead_of_hard_fallback(tmp_path: Path) -> None:
    """Weak transcripts should get a repeat prompt after failed LLM direct."""

    pipeline = _build_empty_pipeline(tmp_path, FailingLLMProvider())

    response = asyncio.run(
        pipeline.process_transcription_with_metrics(
            "What is",
            "en",
            stt_elapsed_ms=100.0,
            language_confidence=0.95,
            language_reason="en_lead_marker",
        )
    )

    assert response["resolved_source"] == "fallback"
    assert response["response_text"] == "I may have misheard that. Please repeat the question."


def test_pipeline_routes_to_llm_when_db_confidence_is_below_threshold(tmp_path: Path) -> None:
    """Low-confidence local matches should route to direct LLM fallback."""

    db_path = tmp_path / "knowledge.sqlite3"
    output_dir = tmp_path / "out"
    knowledge_base = KnowledgeBase(
        db_path,
        retrieval_mode="lexical",
        confidence_high=0.95,
        confidence_low=0.95,
    )
    knowledge_base.ensure_schema()
    knowledge_base.seed_demo_rows()

    pipeline = VoicePipeline(
        knowledge_base=knowledge_base,
        stt_provider=FakeSTTProvider(),
        llm_provider=DirectAnswerLLMProvider(),
        fallback_llm_provider=None,
        primary_tts_provider=FakeTTSProvider(),
        fallback_tts_provider=None,
        output_dir=output_dir,
        timeout_seconds=3.0,
        llm_direct_min_query_tokens=1,
    )

    response = asyncio.run(pipeline.process_transcription("when can i contact support office", "en"))

    assert response["resolved_source"] == "llm_direct"


def test_pipeline_applies_transcript_cheat_with_viet_nam_context_variant(tmp_path: Path) -> None:
    """Context matching should treat `vietnam` and `viet nam` as equivalent variants."""

    db_path = tmp_path / "knowledge.sqlite3"
    output_dir = tmp_path / "out"
    knowledge_base = KnowledgeBase(db_path)
    knowledge_base.ensure_schema()

    pipeline = VoicePipeline(
        knowledge_base=knowledge_base,
        stt_provider=FakeSTTProvider(),
        llm_provider=EchoQuestionLLMProvider(),
        fallback_llm_provider=None,
        primary_tts_provider=FakeTTSProvider(),
        fallback_tts_provider=None,
        output_dir=output_dir,
        timeout_seconds=3.0,
        llm_direct_min_query_tokens=1,
        transcript_cheats=(
            TranscriptCheatRule(
                wrong_phrase="remix",
                corrected_phrase="greenwich",
                required_context_terms=("vietnam",),
            ),
        ),
    )

    response = asyncio.run(pipeline.process_transcription("Hay ke toi nghe ve remix Viet Nam", "vi"))

    assert response["resolved_source"] == "llm_direct"
    assert "greenwich" in response["response_text"].lower()


def test_pipeline_uses_secondary_llm_provider_after_primary_timeout(tmp_path: Path) -> None:
    """When primary direct LLM times out, pipeline should use secondary provider response."""

    db_path = tmp_path / "knowledge.sqlite3"
    output_dir = tmp_path / "out"
    knowledge_base = KnowledgeBase(db_path)
    knowledge_base.ensure_schema()

    pipeline = VoicePipeline(
        knowledge_base=knowledge_base,
        stt_provider=FakeSTTProvider(),
        llm_provider=TimeoutLLMProvider(),
        fallback_llm_provider=DirectAnswerLLMProvider(),
        primary_tts_provider=FakeTTSProvider(),
        fallback_tts_provider=None,
        output_dir=output_dir,
        timeout_seconds=3.0,
        llm_timeout_seconds=0.01,
        llm_direct_min_query_tokens=1,
    )

    response = asyncio.run(pipeline.process_transcription("how do i reset password", "en"))

    assert response["resolved_source"] == "llm_direct"
    assert "reset" in response["response_text"].lower()


def test_pipeline_context_link_resolves_short_followup_to_prior_topic(tmp_path: Path) -> None:
    """Short follow-up turns should reuse the prior local topic anchor when score improves."""

    db_path = tmp_path / "knowledge.sqlite3"
    output_dir = tmp_path / "out"
    knowledge_base = KnowledgeBase(
        db_path,
        retrieval_mode="hybrid",
        confidence_high=0.9,
        confidence_low=0.7,
    )
    knowledge_base.ensure_schema()

    import sqlite3

    with sqlite3.connect(knowledge_base.db_path) as connection:
        connection.executemany(
            """
            INSERT INTO knowledge_base(language, keywords, response, source_id, section, question)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    "en",
                    "greenwich vietnam",
                    "Greenwich Vietnam is an international joint program.",
                    "web-001",
                    "Overview",
                    "What is Greenwich Vietnam?",
                ),
                (
                    "en",
                    "tuition fee greenwich vietnam",
                    "The total tuition fee is 450,000,000 VND.",
                    "web-014",
                    "Tuition",
                    "What is the total tuition fee for the entire program at Greenwich Vietnam?",
                ),
            ],
        )
        connection.commit()

    pipeline = VoicePipeline(
        knowledge_base=knowledge_base,
        stt_provider=FakeSTTProvider(),
        llm_provider=DirectAnswerLLMProvider(),
        fallback_llm_provider=None,
        primary_tts_provider=FakeTTSProvider(),
        fallback_tts_provider=None,
        output_dir=output_dir,
        timeout_seconds=3.0,
        llm_direct_min_query_tokens=1,
        context_link_enabled=True,
        context_link_max_turn_gap=2,
        context_link_short_query_max_tokens=6,
        context_link_min_score_delta=0.08,
    )

    first_turn = asyncio.run(pipeline.process_transcription("Tell me about Greenwich Vietnam", "en"))
    second_turn = asyncio.run(pipeline.process_transcription("What is tuition fee cost?", "en"))

    assert first_turn["resolved_source"] == "local_db"
    assert second_turn["resolved_source"] == "local_db"
    assert "450,000,000" in second_turn["response_text"]


def test_pipeline_rejects_low_score_generic_local_db_match(tmp_path: Path) -> None:
    """Generic low-score questions should not route to a barely matching DB row."""

    db_path = tmp_path / "knowledge.sqlite3"
    knowledge_base = KnowledgeBase(db_path, confidence_low=0.55)
    knowledge_base.ensure_schema()
    pipeline = VoicePipeline(
        knowledge_base=knowledge_base,
        stt_provider=FakeSTTProvider(),
        llm_provider=EchoQuestionLLMProvider(),
        fallback_llm_provider=None,
        primary_tts_provider=FakeTTSProvider(),
        fallback_tts_provider=None,
        output_dir=tmp_path / "out",
        timeout_seconds=3.0,
        llm_direct_min_query_tokens=1,
    )
    db_match = KnowledgeMatch(
        response="Greenwich Vietnam is an international university alliance.",
        score=0.55,
        matched_keyword="greenwich vietnam",
        retrieval_mode="lexical",
        source_id="web-001",
        question="What is Greenwich Vietnam?",
    )

    assert not pipeline._should_accept_local_db_match(db_match, "What is your nickname?")


def test_pipeline_accepts_low_score_domain_local_db_match(tmp_path: Path) -> None:
    """Known domain evidence should still allow threshold-level DB matches."""

    db_path = tmp_path / "knowledge.sqlite3"
    knowledge_base = KnowledgeBase(db_path, confidence_low=0.55)
    knowledge_base.ensure_schema()
    pipeline = VoicePipeline(
        knowledge_base=knowledge_base,
        stt_provider=FakeSTTProvider(),
        llm_provider=EchoQuestionLLMProvider(),
        fallback_llm_provider=None,
        primary_tts_provider=FakeTTSProvider(),
        fallback_tts_provider=None,
        output_dir=tmp_path / "out",
        timeout_seconds=3.0,
        llm_direct_min_query_tokens=1,
    )
    db_match = KnowledgeMatch(
        response="The total tuition is 450,000,000 VND.",
        score=0.55,
        matched_keyword="tuition fee",
        retrieval_mode="lexical",
        source_id="web-014",
        question="What is the tuition fee?",
    )

    assert pipeline._should_accept_local_db_match(db_match, "How much is the tuition fee?")


def test_pipeline_rejects_context_link_for_low_confidence_no_marker_turn(tmp_path: Path) -> None:
    """Noisy low-confidence turns should not inherit the prior topic anchor."""

    db_path = tmp_path / "knowledge.sqlite3"
    output_dir = tmp_path / "out"
    knowledge_base = KnowledgeBase(
        db_path,
        retrieval_mode="hybrid",
        confidence_high=0.9,
        confidence_low=0.7,
    )
    knowledge_base.ensure_schema()

    import sqlite3

    with sqlite3.connect(knowledge_base.db_path) as connection:
        connection.executemany(
            """
            INSERT INTO knowledge_base(language, keywords, response, source_id, section, question)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    "en",
                    "greenwich vietnam",
                    "Greenwich Vietnam is an international joint program.",
                    "web-001",
                    "Overview",
                    "What is Greenwich Vietnam?",
                ),
                (
                    "en",
                    "routing number greenwich vietnam",
                    "This should not be reached by a noisy transcript.",
                    "web-999",
                    "Noise",
                    "Routing number Greenwich Vietnam?",
                ),
            ],
        )
        connection.commit()

    pipeline = VoicePipeline(
        knowledge_base=knowledge_base,
        stt_provider=FakeSTTProvider(),
        llm_provider=EchoQuestionLLMProvider(),
        fallback_llm_provider=None,
        primary_tts_provider=FakeTTSProvider(),
        fallback_tts_provider=None,
        output_dir=output_dir,
        timeout_seconds=3.0,
        llm_direct_min_query_tokens=1,
        context_link_enabled=True,
        context_link_max_turn_gap=2,
        context_link_short_query_max_tokens=6,
        context_link_min_score_delta=0.08,
    )

    first_turn = asyncio.run(pipeline.process_transcription("Tell me about Greenwich Vietnam", "en"))
    second_turn = asyncio.run(
        pipeline.process_transcription_with_metrics(
            "Routing number.",
            "en",
            stt_elapsed_ms=None,
            language_confidence=0.45,
            language_reason="no_marker_hit",
        )
    )

    assert first_turn["resolved_source"] == "local_db"
    assert second_turn["resolved_source"] == "fallback"
    assert "repeat" in second_turn["response_text"].lower()


def test_pipeline_blocks_context_link_for_out_of_domain_sports_turn(tmp_path: Path) -> None:
    """Out-of-domain named entities should not inherit a tuition context anchor."""

    pipeline = _build_pipeline_with_rows(
        tmp_path,
        [
            (
                "en",
                "tuition fee greenwich vietnam",
                "The total tuition fee is 450,000,000 VND.",
                "web-014",
                "Tuition",
                "What is the total tuition fee for the entire program at Greenwich Vietnam?",
            ),
        ],
    )

    first_turn = asyncio.run(pipeline.process_transcription("How much is tuition fee?", "en"))
    second_turn = asyncio.run(pipeline.process_transcription("This is about Manchester United", "en"))

    assert first_turn["resolved_source"] == "local_db"
    assert second_turn["resolved_source"] == "llm_direct"
    assert "tuition" not in second_turn["response_text"].lower()


def test_pipeline_llm_direct_turn_does_not_create_context_anchor(tmp_path: Path) -> None:
    """LLM direct answers should not become DB context anchors for later turns."""

    pipeline = _build_pipeline_with_rows(
        tmp_path,
        [
            (
                "en",
                "greenwich vietnam",
                "Greenwich Vietnam is an international joint program.",
                "web-001",
                "Overview",
                "What is Greenwich Vietnam?",
            ),
        ],
    )

    response = asyncio.run(pipeline.process_transcription("Who won the latest Manchester United match?", "en"))

    assert response["resolved_source"] == "llm_direct"
    assert pipeline._context_anchor is None


def test_pipeline_pins_canonical_faq_intents_over_generic_greenwich_match(tmp_path: Path) -> None:
    """Known FAQ intents should not fall back to the generic Greenwich overview row."""

    db_path = tmp_path / "knowledge.sqlite3"
    output_dir = tmp_path / "out"
    knowledge_base = KnowledgeBase(
        db_path,
        retrieval_mode="hybrid",
        confidence_high=0.9,
        confidence_low=0.55,
    )
    knowledge_base.ensure_schema()

    import sqlite3

    with sqlite3.connect(knowledge_base.db_path) as connection:
        connection.executemany(
            """
            INSERT INTO knowledge_base(language, keywords, response, source_id, section, question)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    "vi",
                    "greenwich viet nam,greenwich viet nam la gi",
                    "Generic answer should not win.",
                    "web-001",
                    "Overview",
                    "Greenwich Viet Nam la gi?",
                ),
                (
                    "vi",
                    "hoc phi toan khoa greenwich viet nam",
                    "Hoc phi toan khoa la 450.000.000 VND.",
                    "web-014",
                    "Tuition",
                    "Hoc phi toan khoa cua Greenwich Viet Nam la bao nhieu?",
                ),
                (
                    "vi",
                    "greenwich viet nam co bao nhieu co so",
                    "Greenwich Viet Nam co 4 co so.",
                    "web-017",
                    "Campus",
                    "Greenwich Viet Nam co bao nhieu co so?",
                ),
                (
                    "vi",
                    "sinh vien co co hoi di du hoc hoac trao doi khong",
                    "Sinh vien co co hoi di du hoc va trao doi.",
                    "web-020",
                    "Exchange",
                    "Sinh vien co co hoi di du hoc hoac trao doi khong?",
                ),
            ],
        )
        connection.commit()

    pipeline = VoicePipeline(
        knowledge_base=knowledge_base,
        stt_provider=FakeSTTProvider(),
        llm_provider=EchoQuestionLLMProvider(),
        fallback_llm_provider=None,
        primary_tts_provider=FakeTTSProvider(),
        fallback_tts_provider=None,
        output_dir=output_dir,
        timeout_seconds=3.0,
        llm_direct_min_query_tokens=1,
    )

    tuition = asyncio.run(pipeline.process_transcription("hoc phi toan khoa greenwich viet nam", "vi"))
    campus = asyncio.run(pipeline.process_transcription("greenwich viet nam co bao nhieu co so", "vi"))
    exchange = asyncio.run(pipeline.process_transcription("co hoi di du hoc hoac trao doi khong", "vi"))

    assert "450.000.000" in tuition["response_text"]
    assert "4 co so" in campus["response_text"] or "4 cơ sở" in campus["response_text"]
    assert "du hoc" in exchange["response_text"]


def test_pipeline_expands_clipped_campus_intent_before_context_link(tmp_path: Path) -> None:
    """Clipped VI campus questions should route to the campus row instead of a generic anchor."""

    db_path = tmp_path / "knowledge.sqlite3"
    output_dir = tmp_path / "out"
    knowledge_base = KnowledgeBase(
        db_path,
        retrieval_mode="hybrid",
        confidence_high=0.9,
        confidence_low=0.55,
    )
    knowledge_base.ensure_schema()

    import sqlite3

    with sqlite3.connect(knowledge_base.db_path) as connection:
        connection.executemany(
            """
            INSERT INTO knowledge_base(language, keywords, response, source_id, section, question)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    "vi",
                    "greenwich viet nam",
                    "Generic answer should not win.",
                    "web-001",
                    "Overview",
                    "Greenwich Việt Nam là gì?",
                ),
                (
                    "vi",
                    "greenwich viet nam co bao nhieu co so,co so greenwich viet nam",
                    "Greenwich Việt Nam có 4 cơ sở.",
                    "web-017",
                    "Campus",
                    "Greenwich Việt Nam có bao nhiêu cơ sở?",
                ),
            ],
        )
        connection.commit()

    pipeline = VoicePipeline(
        knowledge_base=knowledge_base,
        stt_provider=FakeSTTProvider(),
        llm_provider=EchoQuestionLLMProvider(),
        fallback_llm_provider=None,
        primary_tts_provider=FakeTTSProvider(),
        fallback_tts_provider=None,
        output_dir=output_dir,
        timeout_seconds=3.0,
        llm_direct_min_query_tokens=1,
        context_link_enabled=True,
    )

    first_turn = asyncio.run(pipeline.process_transcription("Greenwich Việt Nam là gì?", "vi"))
    second_turn = asyncio.run(
        pipeline.process_transcription_with_metrics(
            "Việt Nam có bao nhiêu cơ sở?",
            "vi",
            stt_elapsed_ms=None,
            language_confidence=0.98,
            language_reason="vi_diacritic",
        )
    )

    assert first_turn["resolved_source"] == "local_db"
    assert second_turn["resolved_source"] == "local_db"
    assert "4 cơ sở" in second_turn["response_text"]
