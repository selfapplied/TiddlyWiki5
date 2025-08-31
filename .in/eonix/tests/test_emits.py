from __future__ import annotations

from eonix.core.emitter import emit_best_fragment
from eonix.core.resonance import score_resonance
from eonix.zipc.reflect import reflect


def test_silence_allowed_when_threshold_high():
    context = "two seeds meet"
    candidates = ["a", "b", " "]
    out = emit_best_fragment(context, candidates, threshold=10.0, l1_lambda=0.5)
    assert out == ""


def test_emit_is_short_when_threshold_low():
    context = "abc abc abc"
    candidates = ["a", "b", " "]
    out = emit_best_fragment(context, candidates, threshold=-10.0, l1_lambda=0.1)
    assert len(out) <= 1


def test_reflection_is_involution():
    s = "abba"
    assert reflect(reflect(s)) == s


def test_resonance_prefers_repetition_over_noise():
    context = "aaaaaa"
    score_a = score_resonance(context, "a", l1_lambda=0.1)
    score_z = score_resonance(context, "z", l1_lambda=0.1)
    assert score_a >= score_z

