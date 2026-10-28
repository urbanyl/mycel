#!/usr/bin/env python3
"""Specification invariants that must hold for every shipped example.

Run in CI and locally:

    python tools/check_spec.py
"""

from __future__ import annotations

import pathlib
import sys

FORBIDDEN = ";{}"
ALLOWED_CONTROL = {9, 10, 13}


def main() -> int:
    root = pathlib.Path(__file__).resolve().parent.parent
    examples = sorted((root / "examples").glob("*.myc"))
    failures = 0

    if not examples:
        print("error: no examples found")
        return 1

    for path in examples:
        raw = path.read_bytes()
        hits = [
            (i, b)
            for i, b in enumerate(raw)
            if b > 126 and b not in ALLOWED_CONTROL or b < 32 and b not in ALLOWED_CONTROL
        ]
        if hits:
            print(f"FAIL {path.name}: non-ASCII bytes at offsets {[i for i, _ in hits[:5]]}")
            failures += 1
        else:
            print(f"ok   {path.name}: ASCII-only")

    for path in examples:
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), 1):
            code = line.split("#", 1)[0]
            for ch in FORBIDDEN:
                if ch in code:
                    print(f"FAIL {path.name}:{lineno}: forbidden character {ch!r}")
                    failures += 1

    if failures:
        print(f"\n{failures} spec invariant violation(s)")
        return 1
    print("\nall spec invariants hold")
    return 0


if __name__ == "__main__":
    sys.exit(main())
