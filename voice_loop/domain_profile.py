"""Domain profile configuration for event-specific assistant behavior."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DomainProfile:
    """Scoped domain hints used before lower-level providers run."""

    name: str
    stt_hint_phrases: tuple[str, ...] = ()


GREENWICH_PROFILE = DomainProfile(
    name="greenwich",
    stt_hint_phrases=(
        "Greenwich Vietnam",
        "Greenwich Việt Nam",
        "University of Greenwich",
        "Đại học Greenwich",
    ),
)

EMPTY_PROFILE = DomainProfile(name="none")


def build_domain_profile(name: str) -> DomainProfile:
    """Return a known domain profile by name, defaulting to no domain hints."""

    normalized = name.strip().lower()
    if normalized == "greenwich":
        return GREENWICH_PROFILE
    return EMPTY_PROFILE
