from __future__ import annotations


def reflect(text: str) -> str:
    return text[::-1]


def is_reflection_stable(text: str) -> bool:
    return text == reflect(reflect(text))

