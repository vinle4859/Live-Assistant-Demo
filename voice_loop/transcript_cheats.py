"""Transcript cheat-rule parsing and guarded correction helpers."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


@dataclass(frozen=True)
class TranscriptCheatRule:
    """A single transcript correction rule with optional context guards.

    Attributes:
        wrong_phrase: Phrase expected from STT mis-transcription.
        corrected_phrase: Phrase to use when the rule applies.
        required_context_terms: Optional context terms. At least one must appear
            in the transcript before this correction is applied.
    """

    wrong_phrase: str
    corrected_phrase: str
    required_context_terms: tuple[str, ...] = ()


def normalize_text(text: str) -> str:
    """Normalize text to lowercase ASCII-like tokens for robust matching."""

    lowered = text.lower().replace("đ", "d")
    ascii_like = unicodedata.normalize("NFD", lowered).encode("ascii", "ignore").decode("ascii")
    return " ".join(_WORD_RE.findall(ascii_like))


def apply_transcript_cheats(transcript: str, rules: tuple[TranscriptCheatRule, ...]) -> tuple[str, tuple[str, ...]]:
    """Apply transcript correction rules and return corrected text plus audit labels."""

    corrected = transcript
    applied_labels: list[str] = []

    for rule in rules:
        if not rule.wrong_phrase.strip() or not rule.corrected_phrase.strip():
            continue
        if rule.required_context_terms and not _has_required_context(corrected, rule.required_context_terms):
            continue

        corrected, did_replace = _replace_phrase_whole_words(
            corrected,
            wrong_phrase=rule.wrong_phrase,
            corrected_phrase=rule.corrected_phrase,
        )
        if did_replace:
            label = f"{rule.wrong_phrase}=>{rule.corrected_phrase}"
            applied_labels.append(label)

    return corrected, tuple(applied_labels)


def _has_required_context(transcript: str, required_terms: tuple[str, ...]) -> bool:
    """Return whether at least one required context term exists in transcript."""

    normalized_transcript = normalize_text(transcript)
    if not normalized_transcript:
        return False

    padded_transcript = f" {normalized_transcript} "
    compact_transcript = normalized_transcript.replace(" ", "")
    for term in required_terms:
        normalized_term = normalize_text(term)
        if not normalized_term:
            continue
        if f" {normalized_term} " in padded_transcript:
            return True
        # Also match compact token forms so `vietnam` matches `viet nam` and vice versa.
        compact_term = normalized_term.replace(" ", "")
        if compact_term and compact_term in compact_transcript:
            return True
    return False


def _replace_phrase_whole_words(transcript: str, wrong_phrase: str, corrected_phrase: str) -> tuple[str, bool]:
    """Replace full-word phrase matches in a case-insensitive manner."""

    wrong_tokens = [token for token in wrong_phrase.strip().split() if token]
    if not wrong_tokens:
        return transcript, False

    escaped_tokens = [re.escape(token) for token in wrong_tokens]
    pattern = r"(?<![A-Za-z0-9])" + r"\s+".join(escaped_tokens) + r"(?![A-Za-z0-9])"
    updated_transcript, replacements = re.subn(
        pattern,
        corrected_phrase,
        transcript,
        flags=re.IGNORECASE,
    )
    return updated_transcript, replacements > 0
