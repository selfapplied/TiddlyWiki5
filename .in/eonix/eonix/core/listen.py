from __future__ import annotations


def should_emit(resonance_score: float, threshold: float) -> bool:
    return resonance_score >= threshold


def silence() -> str:
    return ""

