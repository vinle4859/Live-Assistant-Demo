"""Lightweight language detection for routing text to the correct TTS voice."""

from __future__ import annotations

_VN_CHARS = frozenset(
    "àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ"
    "ÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ"
)

_VN_PHRASES = (
    "xin chào",
    "tôi là",
    "cảm ơn",
    "kính chào",
    "quý khách",
    "chương trình",
    "sự kiện",
    "hãy nói",
    "thưa",
    "mời",
)


def detect_language(text: str) -> str:
    """Return 'vi' if text contains Vietnamese characters or key phrases, else 'en'.

    This is intentionally simple — a single diacritic is enough to route to the
    Vietnamese TTS voice.  The check is O(n) on the text length and allocation-free
    for the character scan.
    """
    if any(c in _VN_CHARS for c in text):
        return "vi"
    lowered = text.lower()
    for phrase in _VN_PHRASES:
        if phrase in lowered:
            return "vi"
    return "en"
