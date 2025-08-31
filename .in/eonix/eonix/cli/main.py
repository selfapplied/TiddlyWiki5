from __future__ import annotations

import argparse
import sys
import string

from eonix.core.emitter import emit_best_fragment


def default_glyph_bank() -> list[str]:
    letters = list(string.ascii_lowercase + string.ascii_uppercase)
    punctuation = list(".,-:;!?()[]{}")
    space = [" "]
    return letters + punctuation + space


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Eonix listener: emits minimal fragments by resonance")
    parser.add_argument("prompt", nargs="?", default="", help="prompt/context text")
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--lambda", dest="l1_lambda", type=float, default=0.5)
    parser.add_argument("--hush", action="store_true", help="only print when resonance is very high")
    args = parser.parse_args(argv)

    candidates = default_glyph_bank()
    fragment = emit_best_fragment(args.prompt, candidates, threshold=args.threshold, l1_lambda=args.l1_lambda)

    if args.hush and not fragment:
        return 1

    sys.stdout.write(fragment)
    sys.stdout.flush()
    return 0 if fragment else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

