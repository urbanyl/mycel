"""Command line interface for the MYCEL reference front end."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from . import __version__
from .checker import check_source
from .lexer import LexError, tokenize
from .parser import ParseError, parse_source
from .units import UnitError, parse_unit


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def cmd_check(args: argparse.Namespace) -> int:
    failures = 0
    for path in args.paths:
        try:
            source = _read(path)
        except OSError as exc:
            print(f"error: cannot read {path}: {exc}", file=sys.stderr)
            failures += 1
            continue
        diags = check_source(source, strict_angles=args.strict_angles)
        errors = [d for d in diags if d.severity == "error"]
        warnings = [d for d in diags if d.severity == "warning"]
        if args.quiet and not errors:
            continue
        print(f"\n== {path} ==")
        for d in diags:
            print(d.render())
        if not diags:
            print("  ok - every obligation discharged")
        else:
            print(f"\n  {len(errors)} error(s), {len(warnings)} warning(s)")
        failures += len(errors)
    return 1 if failures else 0


def cmd_tokens(args: argparse.Namespace) -> int:
    try:
        for tok in tokenize(_read(args.path)):
            print(f"{tok.line:5d}:{tok.col:<4d} {tok.kind:8s} {tok.value!r}")
    except LexError as exc:
        print(f"error[{exc.code}]: {exc.message} at {exc.line}:{exc.col}", file=sys.stderr)
        return 1
    return 0


def cmd_ast(args: argparse.Namespace) -> int:
    try:
        entries = parse_source(_read(args.path))
    except (LexError, ParseError) as exc:
        code = getattr(exc, "code", "MYC0000")
        print(f"error[{code}]: {exc}", file=sys.stderr)
        return 1
    for e in entries:
        conn = f" {e.connective}" if e.connective else ""
        typ = f"  :  {e.type.raw}" if e.type else ""
        params = f"  ({', '.join(e.params)})" if e.params else ""
        print(f"{e.line:5d}  {e.role:9s} {e.name or '':<20s}{conn}{params}{typ}")
        if e.expr:
            print(f"       expr   {e.expr_text()}")
        if e.seed is not None:
            print(f"       seed   {e.seed_text()}")
    return 0


def cmd_units(args: argparse.Namespace) -> int:
    try:
        unit = parse_unit(" ".join(args.expr))
    except UnitError as exc:
        print(f"error[MYC0401]: {exc}", file=sys.stderr)
        return 1
    print(f"input      : {' '.join(args.expr)}")
    print(f"normalised : {unit}")
    print(f"exponents  : {tuple(str(x) for x in unit.exps)}")
    if unit.tags:
        print(f"tags       : {', '.join(unit.tags)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mycel",
        description="MYCEL reference front end - lexer, ledger parser, unit and axis checker.",
    )
    p.add_argument("--version", action="version", version=f"mycel {__version__}")
    sub = p.add_subparsers(dest="command")

    chk = sub.add_parser("check", help="verify one or more .myc files")
    chk.add_argument("paths", nargs="+")
    chk.add_argument("--strict-angles", action="store_true")
    chk.add_argument("-q", "--quiet", action="store_true")
    chk.set_defaults(func=cmd_check)

    tok = sub.add_parser("tokens", help="dump the token stream")
    tok.add_argument("path")
    tok.set_defaults(func=cmd_tokens)

    ast = sub.add_parser("ast", help="dump parsed ledger entries")
    ast.add_argument("path")
    ast.set_defaults(func=cmd_ast)

    uni = sub.add_parser("units", help="normalise an SI unit expression")
    uni.add_argument("expr", nargs="+")
    uni.set_defaults(func=cmd_units)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
