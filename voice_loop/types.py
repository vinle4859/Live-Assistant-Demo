"""Shared type definitions used across the voice pipeline."""

from __future__ import annotations

from typing import Literal, TypedDict

LanguageCode = Literal["en", "vi"]
ResolutionSource = Literal["local_db", "llm_direct", "fallback"]


class PipelineResponse(TypedDict):
    """Structured result produced by a pipeline run."""

    detected_language: LanguageCode
    transcription: str
    resolved_source: ResolutionSource
    response_text: str
    audio_output_path: str
