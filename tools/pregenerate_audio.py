"""Tool to pre-generate spoken audio files for all knowledge base answers using the configured TTS provider."""

from __future__ import annotations

import asyncio
import logging
import re
import sqlite3
import unicodedata
from pathlib import Path

from voice_loop.config import AppConfig
from voice_loop.db import KnowledgeBase, KnowledgeMatch
from voice_loop.factory import build_tts_provider
from voice_loop.pipeline import VoicePipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger("pregenerate_audio")


def _clean_question_for_filename(question: str | None) -> str:
    """Normalize question text into a short, safe, lowercase ASCII string for filenames."""
    if not question:
        return ""
    # Normalize and strip diacritics
    normalized = unicodedata.normalize("NFKD", question).encode("ascii", "ignore").decode("utf-8")
    # Keep alphanumeric characters and spaces
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", "", normalized).lower().strip()
    # Take first 4 words
    words = cleaned.split()[:4]
    return "_".join(words)


async def main() -> None:
    # 1. Load application config
    import os
    lang = os.getenv("VOICE_LOOP_LANGUAGE", "en")
    config = AppConfig.from_env(language=lang)
    db_path = Path(config.db_path)
    if not db_path.is_file():
        LOGGER.error("Database file not found at %s. Please seed it first.", db_path)
        return

    # 2. Setup output folder
    # We store the Q&A pre-rendered audio in data/live_audio/qa_pre_rendered/
    qa_audio_dir = db_path.parent / "live_audio" / "qa_pre_rendered"
    qa_audio_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Pre-rendered audio files will be saved to: %s", qa_audio_dir)

    # 3. Build TTS provider
    LOGGER.info("Building TTS provider: %s", config.tts_provider)
    tts_provider = build_tts_provider(config)

    # 4. Connect to database and ensure schema
    kb = KnowledgeBase(db_path)
    kb.ensure_schema()

    # 5. Build minimal pipeline for spoken compaction overrides
    pipeline = VoicePipeline(
        knowledge_base=kb,
        stt_provider=None,
        llm_provider=None,
        fallback_llm_provider=None,
        primary_tts_provider=None,
        fallback_tts_provider=None,
        output_dir=qa_audio_dir,
    )

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT id, language, response, source_id, question, audio_path FROM knowledge_base"
        ).fetchall()
    except sqlite3.OperationalError as e:
        LOGGER.error("Error reading database: %s", e)
        conn.close()
        return

    LOGGER.info("Found %d rows in knowledge_base table.", len(rows))

    # Keep track of active relative paths so we can delete old orphaned audio files
    active_relative_paths = set()
    rows_to_process = []

    # First pass: calculate target naming and clean text
    for row in rows:
        row_id = row["id"]
        language = row["language"]
        response_text = row["response"]
        source_id = row["source_id"]
        question_text = row["question"]
        existing_audio = row["audio_path"]

        if not response_text or not response_text.strip():
            LOGGER.warning("Skipping empty response for row ID %d", row_id)
            continue

        # Build mock KnowledgeMatch to utilize pipeline spoken answer override and sentence selection logic
        db_match = KnowledgeMatch(
            response=response_text,
            score=1.0,
            matched_keyword="",
            retrieval_mode="lexical",
            source_id=source_id,
            section=None,
            question=question_text,
            exact_hit_count=1,
            fuzzy_hit_count=0,
            matched_keyword_token_count=1,
            query_token_count=1,
            keyword_coverage=1.0,
            query_coverage=1.0,
            score_margin=1.0,
            whole_phrase_match=True,
            audio_path=existing_audio,
        )

        # Compact the response exactly like the live assistant does
        compacted_text = pipeline._compact_local_db_response(
            response_text,
            query="",
            db_match=db_match,
            language=language,
        )

        # Strip markdown symbols so TTS doesn't stumble
        cleaned_text = VoicePipeline._prepare_text_for_tts(compacted_text)

        # Generate descriptive filename
        file_prefix = source_id if source_id else f"row_{row_id}"
        q_suffix = _clean_question_for_filename(question_text)
        if q_suffix:
            filename_base = f"{file_prefix}_{q_suffix}_{language}"
        else:
            filename_base = f"{file_prefix}_{language}"

        # Clean prefix from characters unsuitable for filenames
        safe_base = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in filename_base)
        file_name = f"{safe_base}.mp3"
        output_path = qa_audio_dir / file_name

        db_relative_path = f"data/live_audio/qa_pre_rendered/{file_name}"
        active_relative_paths.add(db_relative_path)

        rows_to_process.append({
            "row_id": row_id,
            "language": language,
            "cleaned_text": cleaned_text,
            "output_path": output_path,
            "db_relative_path": db_relative_path,
            "existing_audio": existing_audio,
            "file_prefix": file_prefix,
        })

    # Second pass: clean up orphaned audio files in the pre_rendered directory
    existing_files = list(qa_audio_dir.glob("*.mp3"))
    cleaned_orphans = 0
    for f in existing_files:
        rel_p = f"data/live_audio/qa_pre_rendered/{f.name}"
        if rel_p not in active_relative_paths:
            try:
                f.unlink()
                cleaned_orphans += 1
            except OSError:
                pass
    if cleaned_orphans:
        LOGGER.info("Cleaned up %d orphaned/outdated audio files.", cleaned_orphans)

    # Third pass: synthesize and update DB
    success_count = 0
    skipped_count = 0

    for item in rows_to_process:
        row_id = item["row_id"]
        language = item["language"]
        cleaned_text = item["cleaned_text"]
        output_path = item["output_path"]
        db_relative_path = item["db_relative_path"]
        existing_audio = item["existing_audio"]
        file_prefix = item["file_prefix"]

        # If audio file already exists and matches DB path, skip
        if existing_audio == db_relative_path and output_path.is_file():
            skipped_count += 1
            continue

        LOGGER.info(
            "Synthesizing audio for Row ID %d (%s, lang=%s): '%s...'",
            row_id,
            file_prefix,
            language,
            cleaned_text[:50],
        )

        try:
            await tts_provider.synthesize(
                text=cleaned_text,
                language=language,
                output_path=output_path,
            )
            conn.execute(
                "UPDATE knowledge_base SET audio_path = ? WHERE id = ?",
                (db_relative_path, row_id),
            )
            conn.commit()
            LOGGER.info("Saved and updated database path: %s", db_relative_path)
            success_count += 1
        except Exception as e:
            LOGGER.error("Failed to synthesize Row ID %d: %s", row_id, e)

    conn.close()
    LOGGER.info(
        "Audio pre-generation complete. Success: %d, Skipped: %d, Total: %d",
        success_count,
        skipped_count,
        len(rows_to_process),
    )


if __name__ == "__main__":
    asyncio.run(main())
