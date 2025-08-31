from __future__ import annotations

from typing import Iterable, Optional

from eonix.core.listen import should_emit, silence
from eonix.core.resonance import score_resonance


def emit_best_fragment(
    context: str,
    candidates: Iterable[str],
    threshold: float = 1.0,
    l1_lambda: float = 0.5,
) -> str:
    best_fragment: Optional[str] = None
    best_score: float = float("-inf")
    for fragment in candidates:
        s = score_resonance(context, fragment, l1_lambda=l1_lambda)
        if s > best_score:
            best_score, best_fragment = s, fragment
    if best_fragment is None:
        return silence()
    return best_fragment if should_emit(best_score, threshold) else silence()

