"""Static checker for MYCEL ledgers.

This is the reference implementation of the obligations that do not need
the full MLIR backend: character set, entry structure, type-annotation
well-formedness, axis uniqueness, unit parsing, seed/genesis rules, and
cotangent-target resolution.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from .lexer import LexError, Token, tokenize
from .parser import CONNECTIVES, Entry, ParseError, parse_source


@dataclass
class Diagnostic:
    severity: str
    code: str
    message: str
    line: int
    col: int = 1
    hint: Optional[str] = None

    def render(self) -> str:
        head = f"{self.severity}[{self.code}]: {self.message}"
        loc = f"  --> line {self.line}, col {self.col}"
        tail = f"  = note: {self.hint}" if self.hint else ""
        return f"{head}\n{loc}{tail}"


class _Ctx:
    def __init__(self, strict_angles: bool) -> None:
        self.strict_angles = strict_angles
        self.diags: List[Diagnostic] = []
        self.fields: set = set()
        self.assemblies: set = set()
        self.bound: set = set()
        self.params: set = set()

    def error(self, code: str, msg: str, line: int, col: int = 1, hint: Optional[str] = None) -> None:
        self.diags.append(Diagnostic("error", code, msg, line, col, hint))

    def warn(self, code: str, msg: str, line: int, col: int = 1, hint: Optional[str] = None) -> None:
        self.diags.append(Diagnostic("warning", code, msg, line, col, hint))


def _known_target(root: str, ctx: _Ctx) -> bool:
    if root in ctx.params or root in ctx.fields or root in ctx.assemblies:
        return True
    return False


def _scan_grad_targets(tokens: Sequence[Token], ctx: _Ctx, line: int) -> None:
    for idx, tok in enumerate(tokens):
        if tok.kind == "IDENT" and tok.value == "grad":
            j = idx + 1
            if j >= len(tokens) or tokens[j].value != "(":
                ctx.error("MYC0502", "grad(...) requires a parameter and a loss", tok.line, tok.col)
                continue
            j += 1
            depth = 0
            arg: List[Token] = []
            while j < len(tokens):
                t = tokens[j]
                if t.value == "(":
                    depth += 1
                elif t.value == ")":
                    if depth == 0:
                        break
                    depth -= 1
                elif t.value == "," and depth == 0:
                    break
                arg.append(t)
                j += 1
            if not arg:
                ctx.error("MYC0502", "grad(...) has an empty gradient target", tok.line, tok.col)
                continue
            root = arg[0].value
            if not _known_target(root, ctx):
                ctx.warn(
                    "MYC0502",
                    f"gradient target {root!r} is not declared in this ledger",
                    arg[0].line,
                    arg[0].col,
                    "declare it with '@ <name> := ...' or check for a typo",
                )


def check_entries(entries: Iterable[Entry], strict_angles: bool = False) -> List[Diagnostic]:
    ctx = _Ctx(strict_angles)

    entries = list(entries)

    # Pass 1: collect names so that forward references resolve.
    for e in entries:
        if e.role == "field" and e.name:
            ctx.fields.add(e.name)
            ctx.fields.add(e.name.split(".")[0])
        elif e.role == "assembly" and e.name:
            ctx.assemblies.add(e.name)
            ctx.params.update(e.params)
        elif e.role == "directive" and e.name:
            ctx.fields.add(e.name)

    bound_by_bind: set = set()

    def leaf(name: str) -> str:
        return name.split(".")[-1]

    for e in entries:
        if e.role == "field":
            if e.name is None:
                ctx.error("MYC0203", "field has no name", e.line)
                continue
            if e.connective == ":=":
                bound_by_bind.add(leaf(e.name))
            if e.connective in ("<<", "<<<"):
                has_seed = e.seed is not None and len(e.seed) > 0
                genesis = leaf(e.name) in bound_by_bind
                if not has_seed and not genesis:
                    ctx.error(
                        "MYC0601",
                        f"'{e.connective}' field {e.name!r} has no seed and no ':=' genesis",
                        e.line,
                        hint="write 'seed <expr>' or bind the same name with ':=' first",
                    )
            if e.type is None and e.connective == "=>":
                ctx.warn(
                    "MYC0401",
                    f"differentiable field {e.name!r} has no type annotation",
                    e.line,
                    hint="a '=>' edge joins the cotangent graph; its dimensions must be verifiable",
                )
            _scan_grad_targets(e.expr, ctx, e.line)

        elif e.role == "assembly":
            if not e.name:
                ctx.error("MYC0203", "assembly has no name", e.line)
            if e.type is not None:
                ctx.error("MYC0203", "assemblies cannot carry a type annotation", e.line)
            _scan_grad_targets(e.expr, ctx, e.line)

        elif e.role == "assert":
            if not e.expr:
                ctx.error("MYC0203", "empty assert", e.line)

        elif e.role == "directive":
            pass

        elif e.role == "export":
            if not e.name:
                ctx.error("MYC0203", "export has no symbol", e.line)

        if e.type is not None and ctx.strict_angles and e.type.unit.dimensionless:
            tags = e.type.unit.tags
            if tags and e.seed is not None and e.seed_text().strip() == "0":
                ctx.warn(
                    "MYC0404",
                    f"{e.name!r} is seeded with a plain 0 but typed as {tags[0]}",
                    e.line,
                    hint="write 0*rad, or relax with --no-strict-angles",
                )

    return ctx.diags


def check_source(source: str, strict_angles: bool = False) -> List[Diagnostic]:
    try:
        entries = parse_source(source)
    except LexError as exc:
        return [Diagnostic("error", exc.code, exc.message, exc.line, exc.col)]
    except ParseError as exc:
        return [Diagnostic("error", exc.code, exc.message, exc.line, exc.col)]
    return check_entries(entries, strict_angles=strict_angles)


def check_file(path: str | Path, strict_angles: bool = False) -> List[Diagnostic]:
    text = Path(path).read_text(encoding="utf-8")
    return check_source(text, strict_angles=strict_angles)
