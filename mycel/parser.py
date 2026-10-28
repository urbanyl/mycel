"""Ledger parser.

A MYCEL source file is a set of entries, one per line, in no meaningful
order. The parser turns that set into structured ``Entry`` records. It is
deliberately tolerant about expressions (they are kept as tokens) and
strict about structure (roles, connectives, brackets, type annotations).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from .axes import Axes, AxisError
from .lexer import Token, tokenize
from .units import Unit, UnitError, parse_unit

ROLE_SIGILS = {"@": "field", "%": "assembly", "$": "directive", ">": "export"}
CONNECTIVES = {":=", "~>", "=>", "<<", "<<<", "!>"}
MANIFOLDS = {"real", "S2", "S3", "SO3", "SE3", "Tan", "Cot"}
OPEN = {"(": ")", "[": "]"}
CLOSE = {")": "(", "]": "["}


class ParseError(ValueError):
    def __init__(self, code: str, message: str, line: int, col: int) -> None:
        super().__init__(f"{code}: {message} (line {line}, col {col})")
        self.code = code
        self.message = message
        self.line = line
        self.col = col


@dataclass(frozen=True)
class TypeRef:
    manifold: str
    axes: Axes
    unit: Unit
    raw: str


@dataclass
class Entry:
    role: str
    name: Optional[str]
    connective: Optional[str]
    expr: Tuple[Token, ...]
    seed: Optional[Tuple[Token, ...]]
    type: Optional[TypeRef]
    line: int
    raw: str
    params: Tuple[str, ...] = field(default_factory=tuple)

    def expr_text(self) -> str:
        return _render(self.expr)

    def seed_text(self) -> str:
        return _render(self.seed or ())


def _render(tokens: Sequence[Token]) -> str:
    """Best-effort readable reconstruction of a token run."""
    out: List[str] = []
    prev: Optional[str] = None
    tight_after = {"[", "(", "."}
    tight_before = {"]", ")", ",", "."}
    for tok in tokens:
        v = tok.value
        if out and prev is not None:
            if prev in tight_after or v in tight_before:
                out.append(v)
            elif prev in {".*", "**", "~*"}:
                out.append(v)
            else:
                out.append(" " + v)
        else:
            out.append(v)
        prev = v
    return "".join(out)


def _depth_delta(val: str) -> int:
    if val in OPEN:
        return 1
    if val in CLOSE:
        return -1
    return 0


def _split_top(tokens: Sequence[Token]) -> Tuple[List[Token], Optional[List[Token]], Optional[List[Token]]]:
    """Split an entry body into (expr, seed, type-annotation)."""
    depth = 0
    expr: List[Token] = []
    seed: Optional[List[Token]] = None
    typ: Optional[List[Token]] = None
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.kind == "OP" and tok.value in OPEN:
            depth += 1
        elif tok.kind == "OP" and tok.value in CLOSE:
            depth -= 1
            if depth < 0:
                raise ParseError("MYC0203", "unbalanced closing bracket", tok.line, tok.col)
        if depth == 0:
            if tok.kind == "IDENT" and tok.value == "seed" and seed is None and typ is None:
                seed = []
                i += 1
                continue
            if tok.kind == "OP" and tok.value == ":" and typ is None:
                typ = []
                i += 1
                continue
        if typ is not None:
            typ.append(tok)
        elif seed is not None:
            seed.append(tok)
        else:
            expr.append(tok)
        i += 1
    if depth != 0:
        last = tokens[-1] if tokens else None
        raise ParseError(
            "MYC0203",
            "unbalanced opening bracket",
            last.line if last else 1,
            last.col if last else 1,
        )
    return expr, seed, typ


def _parse_type(tokens: Sequence[Token], raw: str) -> TypeRef:
    if not tokens:
        raise ParseError("MYC0203", "empty type annotation", 1, 1)
    head = tokens[0]
    if head.kind != "IDENT" or head.value not in MANIFOLDS:
        raise ParseError(
            "MYC0403",
            f"unknown manifold {head.value!r}; expected one of "
            + ", ".join(sorted(MANIFOLDS)),
            head.line,
            head.col,
        )
    manifold = head.value
    rest = list(tokens[1:])
    axes = Axes(())
    if rest and rest[0].kind == "OP" and rest[0].value == "[":
        depth = 0
        end = None
        for idx, tok in enumerate(rest):
            if tok.value == "[":
                depth += 1
            elif tok.value == "]":
                depth -= 1
                if depth == 0:
                    end = idx
                    break
        if end is None:
            raise ParseError("MYC0203", "unterminated axis list", head.line, head.col)
        inner = rest[1:end]
        try:
            axes = _parse_axes(inner)
        except AxisError as exc:
            raise ParseError("MYC0301", str(exc), head.line, head.col) from exc
        rest = rest[end + 1 :]
    unit_txt = "".join(tok.value for tok in rest).strip()
    try:
        unit = parse_unit(unit_txt) if unit_txt else parse_unit("1")
    except UnitError as exc:
        raise ParseError("MYC0401", str(exc), head.line, head.col) from exc
    return TypeRef(manifold=manifold, axes=axes, unit=unit, raw=raw.strip())


def _parse_axes(tokens: Sequence[Token]) -> Axes:
    chunks: List[List[Token]] = []
    current: List[Token] = []
    depth = 0
    for tok in tokens:
        if tok.kind == "OP" and tok.value in OPEN:
            depth += 1
        elif tok.kind == "OP" and tok.value in CLOSE:
            depth -= 1
        if tok.kind == "OP" and tok.value == "," and depth == 0:
            chunks.append(current)
            current = []
            continue
        current.append(tok)
    if current:
        chunks.append(current)

    from .axes import Axis

    items = []
    seen = set()
    for chunk in chunks:
        if not chunk:
            continue
        name_tok = chunk[0]
        if name_tok.kind != "IDENT":
            raise AxisError(f"axis name must be an identifier, got {name_tok.value!r}")
        if name_tok.value in seen:
            raise AxisError(f"duplicate axis name {name_tok.value!r}")
        seen.add(name_tok.value)
        extent = "".join(t.value for t in chunk[1:])
        items.append(Axis(name_tok.value, extent or None))
    return Axes(tuple(items))


def _parse_name(tokens: Sequence[Token], i: int) -> Tuple[str, int]:
    if i >= len(tokens) or tokens[i].kind != "IDENT":
        tok = tokens[i] if i < len(tokens) else None
        raise ParseError(
            "MYC0203",
            "expected a name after the role sigil",
            tok.line if tok else 1,
            tok.col if tok else 1,
        )
    parts = [tokens[i].value]
    i += 1
    while (
        i + 1 < len(tokens)
        and tokens[i].kind == "OP"
        and tokens[i].value == "."
        and tokens[i + 1].kind == "IDENT"
    ):
        parts.append(tokens[i + 1].value)
        i += 2
    return ".".join(parts), i


def _params(tokens: Sequence[Token]) -> Tuple[str, ...]:
    """Names of the top-level comma-separated parameters inside (...) ."""
    depth = 0
    inside = False
    names: List[str] = []
    chunk_first: Optional[str] = None
    for tok in tokens:
        if tok.kind == "OP" and tok.value == "(" and not inside:
            inside = True
            depth = 1
            continue
        if not inside:
            continue
        if tok.kind == "OP" and tok.value in OPEN:
            depth += 1
        elif tok.kind == "OP" and tok.value in CLOSE:
            depth -= 1
            if depth == 0:
                inside = False
                if chunk_first:
                    names.append(chunk_first)
                    chunk_first = None
        elif tok.kind == "OP" and tok.value == "," and depth == 1:
            if chunk_first:
                names.append(chunk_first)
            chunk_first = None
        elif tok.kind == "IDENT" and chunk_first is None and depth == 1:
            chunk_first = tok.value
    if chunk_first:
        names.append(chunk_first)
    return tuple(names)


def parse_source(source: str) -> List[Entry]:
    tokens = [t for t in tokenize(source) if t.kind != "EOF"]
    lines = source.replace("\r\n", "\n").split("\n")

    groups: List[List[Token]] = []
    current: List[Token] = []
    for tok in tokens:
        if tok.kind == "NEWLINE":
            if current:
                groups.append(current)
            current = []
            continue
        current.append(tok)
    if current:
        groups.append(current)

    entries: List[Entry] = []
    for group in groups:
        head = group[0]
        role: Optional[str] = None
        if head.kind == "OP" and head.value in ROLE_SIGILS:
            role = ROLE_SIGILS[head.value]
            i = 1
        elif head.kind == "IDENT" and head.value == "assert":
            role = "assert"
            i = 1
        else:
            raise ParseError(
                "MYC0201",
                f"unknown entry role {head.value!r}; expected one of @ % $ assert >",
                head.line,
                head.col,
            )

        start_line = head.line
        end_line = group[-1].line
        raw = " ".join(l.strip() for l in lines[start_line - 1 : end_line]).strip()

        name: Optional[str] = None
        connective: Optional[str] = None
        body: List[Token] = []
        params: Tuple[str, ...] = ()

        if role == "directive":
            if i < len(group) and group[i].kind == "IDENT":
                name = group[i].value
                i += 1
            body = group[i:]
        elif role == "assert":
            body = group[i:]
            if not body:
                raise ParseError("MYC0203", "empty assert", head.line, head.col)
        elif role == "export":
            name, i = _parse_name(group, i)
            body = group[i:]
            params = _params(body)
        else:
            name, i = _parse_name(group, i)
            if i < len(group) and group[i].kind == "OP" and group[i].value in CONNECTIVES:
                connective = group[i].value
                i += 1
            body = group[i:]
            if role == "assembly":
                params = _params(body)

        expr, seed, type_tokens = _split_top(body)
        type_ref = _parse_type(type_tokens, raw) if type_tokens is not None else None

        entries.append(
            Entry(
                role=role,
                name=name,
                connective=connective,
                expr=tuple(expr),
                seed=tuple(seed) if seed is not None else None,
                type=type_ref,
                line=start_line,
                raw=raw,
                params=params,
            )
        )
    return entries
