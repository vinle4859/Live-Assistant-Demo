"""Factories for assembling the default production pipeline."""

from __future__ import annotations

import logging

from .config import AppConfig
from .db import KnowledgeBase
from .pipeline import VoicePipeline
from .providers.demo import DemoSpeechToTextProvider, DemoTextToSpeechProvider
from .providers.edge_tts import EdgeTTSProvider
from .providers.google_stt import GoogleSpeechToTextProvider
from .providers.google_tts import GoogleTextToSpeechProvider
from .providers.llm import GeminiLLMProvider, HeuristicLLMProvider

LOGGER = logging.getLogger(__name__)


def build_default_pipeline(config: AppConfig) -> VoicePipeline:
    """Build the default Google-first pipeline from a runtime configuration."""

    knowledge_base = KnowledgeBase(
        config.db_path,
        retrieval_mode=config.qa_retrieval_mode,
        lexical_top_k=config.qa_lexical_top_k,
        vector_top_k=config.qa_vector_top_k,
        confidence_low=config.qa_confidence_low,
    )
    knowledge_base.ensure_schema()
    if config.qa_seed_auto_sync and not knowledge_base.has_curated_rows():
        inserted_rows = knowledge_base.seed_from_qa_json(config.qa_seed_json_path)
        if inserted_rows > 0:
            LOGGER.info(
                "Imported %d curated QA rows from %s.",
                inserted_rows,
                config.qa_seed_json_path,
            )
    if config.seed_demo_data and knowledge_base.is_empty():
        knowledge_base.seed_demo_rows()

    stt_provider = _build_stt_provider(config)
    primary_tts_provider = _build_tts_provider(config)
    fallback_tts_provider = EdgeTTSProvider()
    llm_provider, fallback_llm_provider = _build_llm_providers(config)

    return VoicePipeline(
        knowledge_base=knowledge_base,
        stt_provider=stt_provider,
        llm_provider=llm_provider,
        fallback_llm_provider=fallback_llm_provider,
        primary_tts_provider=primary_tts_provider,
        fallback_tts_provider=fallback_tts_provider,
        output_dir=config.output_dir,
        timeout_seconds=config.provider_timeout_seconds,
        llm_timeout_seconds=config.llm_timeout_seconds,
        llm_direct_min_query_tokens=config.llm_direct_min_query_tokens,
        context_link_enabled=config.context_link_enabled,
        context_link_max_turn_gap=config.context_link_max_turn_gap,
        context_link_short_query_max_tokens=config.context_link_short_query_max_tokens,
        context_link_min_score_delta=config.context_link_min_score_delta,
        transcript_cheats=config.transcript_cheats,
    )


def _build_llm_providers(config: AppConfig):
    """Choose primary/secondary LLM providers with deterministic low-data fallback behavior."""

    secondary_provider = HeuristicLLMProvider()
    try:
        if config.google_cloud_project:
            primary_provider = GeminiLLMProvider(
                model=config.gemini_model,
                project=config.google_cloud_project,
                location=config.google_cloud_location,
                enable_google_search=config.llm_enable_google_search,
                timeout_seconds=config.llm_timeout_seconds,
            )
            # In production wiring, skip heuristic secondary fallback to avoid question-echo responses
            # when cloud generation times out. Pipeline then uses the hard localized fallback phrase.
            return primary_provider, None
    except Exception:
        return secondary_provider, None
    return secondary_provider, None


def _build_stt_provider(config: AppConfig):
    """Choose the STT provider based on configuration."""

    if config.stt_provider == "demo":
        return DemoSpeechToTextProvider()
    return GoogleSpeechToTextProvider(
        timeout_seconds=config.provider_timeout_seconds,
        model=config.stt_model or None,
        hint_phrases=config.stt_hint_phrases,
        project=config.google_cloud_project,
        location=config.stt_location,
        language_mode=config.language_mode,
    )


def _build_tts_provider(config: AppConfig):
    """Choose the TTS provider based on configuration."""

    if config.tts_provider == "demo":
        return DemoTextToSpeechProvider()
    return GoogleTextToSpeechProvider(timeout_seconds=config.provider_timeout_seconds)
