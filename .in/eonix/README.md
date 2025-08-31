# Eonix (listener LM)

Emits minimal fragments by resonance; prefers silence.

- Resonance: deflate compression gain minus L1 length penalty
- Early-stop via threshold; silence wins when resonance is low
- Optional reflection and octwave typing hooks

## CLI

```bash
PYTHONPATH=. python -m eonix.cli "two seeds meet"
# or if installed
# eonix "two seeds meet"
```

Exit code is 0 when a fragment is emitted, 1 when silence is chosen.