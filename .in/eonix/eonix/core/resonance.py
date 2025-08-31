from __future__ import annotations

import zlib


def _deflate_len(text: str) -> int:
    data = text.encode("utf-8", errors="ignore")
    return len(zlib.compress(data))


def score_resonance(context: str, fragment: str, l1_lambda: float = 0.5) -> float:
    """Return resonance score: compression gain minus L1 penalty.

    score = (deflate_len(context) - deflate_len(context + fragment)) - l1_lambda * len(fragment)
    Higher is better. Negative or small values favor silence.
    """
    base = _deflate_len(context)
    with_frag = _deflate_len(context + fragment)
    gain = base - with_frag
    penalty = l1_lambda * len(fragment)
    return float(gain - penalty)

