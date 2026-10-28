"""SI unit algebra over Q^7.

Units are exponent vectors on the seven SI base dimensions. The exponents
are Fractions, because sqrt(m^2/s^2) appears in every normalisation and
every RMS, and integer exponents cannot express it without lying.

All derived names are resolved eagerly, so ``N*m`` and ``kg*m^2/s^2`` are
the *same* value, not merely convertible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, Iterable, Tuple

DIMENSIONS: Tuple[str, ...] = ("kg", "m", "s", "A", "K", "mol", "cd")

_ZERO = (Fraction(0),) * 7


def _v(*xs: int) -> Tuple[Fraction, ...]:
    return tuple(Fraction(x) for x in xs)


class UnitError(ValueError):
    """Raised when a unit expression cannot be parsed or is ambiguous."""


@dataclass(frozen=True)
class Unit:
    """An SI exponent vector, with optional dimensionless tags.

    Tags (``rad``, ``sr``) carry no dimension but distinguish angles from
    pure numbers under ``--strict-angles``.
    """

    exps: Tuple[Fraction, ...]
    tags: Tuple[str, ...] = ()

    # -- construction ---------------------------------------------------
    @staticmethod
    def of(**kwargs: int) -> "Unit":
        unknown = set(kwargs) - set(DIMENSIONS)
        if unknown:
            raise UnitError("unknown dimension(s): " + ", ".join(sorted(unknown)))
        return Unit(tuple(Fraction(kwargs.get(d, 0)) for d in DIMENSIONS))

    # -- arithmetic -----------------------------------------------------
    def __mul__(self, other: "Unit") -> "Unit":
        return Unit(
            tuple(a + b for a, b in zip(self.exps, other.exps)),
            tuple(sorted(set(self.tags) | set(other.tags))),
        )

    def __truediv__(self, other: "Unit") -> "Unit":
        return Unit(
            tuple(a - b for a, b in zip(self.exps, other.exps)),
            tuple(sorted(set(self.tags) ^ set(other.tags))),
        )

    def __pow__(self, power: Fraction) -> "Unit":
        p = Fraction(power)
        tags = self.tags if p == 1 else ()
        return Unit(tuple(e * p for e in self.exps), tags)

    def sqrt(self) -> "Unit":
        return self ** Fraction(1, 2)

    # -- queries --------------------------------------------------------
    @property
    def dimensionless(self) -> bool:
        return all(e == 0 for e in self.exps)

    def same_dimension(self, other: "Unit") -> bool:
        return self.exps == other.exps

    # -- rendering ------------------------------------------------------
    def __str__(self) -> str:
        if self.dimensionless:
            return "/".join(self.tags) if self.tags else "1"
        num = []
        den = []
        for name, e in zip(DIMENSIONS, self.exps):
            if e == 0:
                continue
            target = num if e > 0 else den
            mag = abs(e)
            if mag == 1:
                target.append(name)
            elif mag.denominator == 1:
                target.append(f"{name}^{mag.numerator}")
            else:
                target.append(f"{name}^{mag.numerator}/{mag.denominator}")
        out = "*".join(num) if num else "1"
        if den:
            out += "/" + "*".join(den)
        return out

    def __repr__(self) -> str:
        return f"Unit({self})"


DIMENSIONLESS = Unit(_ZERO)

_BASE: Dict[str, Unit] = {name: Unit.of(**{name: 1}) for name in DIMENSIONS}


def _d(**kw: int) -> Unit:
    return Unit.of(**kw)


# fmt: off
DERIVED: Dict[str, Unit] = {
    "N":   _d(kg=1, m=1, s=-2),
    "J":   _d(kg=1, m=2, s=-2),
    "W":   _d(kg=1, m=2, s=-3),
    "Pa":  _d(kg=1, m=-1, s=-2),
    "Hz":  _d(s=-1),
    "C":   _d(s=1, A=1),
    "V":   _d(kg=1, m=2, s=-3, A=-1),
    "Ohm": _d(kg=1, m=2, s=-3, A=-2),
    "T":   _d(kg=1, s=-2, A=-1),
    "F":   _d(kg=-1, m=-2, s=4, A=2),
    "H":   _d(kg=1, m=2, s=-2, A=-2),
    "S":   _d(kg=-1, m=-2, s=3, A=2),
    "Gy":  _d(m=2, s=-2),
    "Sv":  _d(m=2, s=-2),
    "bar": _d(kg=1, m=-1, s=-2),
}
# fmt: on

SYMBOLS: Dict[str, Unit] = {}
SYMBOLS.update(_BASE)
SYMBOLS.update(DERIVED)
SYMBOLS["1"] = DIMENSIONLESS
SYMBOLS["rad"] = Unit(_ZERO, ("rad",))
SYMBOLS["sr"] = Unit(_ZERO, ("sr",))

_TOKEN = re.compile(r"\s*(\*\*|[/^*]|[0-9]+|[A-Za-z_][A-Za-z0-9_]*|\(|\)|-)")
_EXP = re.compile(r"^(-?[0-9]+)(?:/([0-9]+))?$")


def _lex(text: str) -> Iterable[str]:
    pos = 0
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if not m:
            if text[pos:].strip() == "":
                return
            raise UnitError(f"unexpected character {text[pos]!r} in unit {text!r}")
        pos = m.end()
        yield m.group(1)


class _Parser:
    def __init__(self, tokens: Iterable[str], text: str) -> None:
        self.tokens = list(tokens)
        self.i = 0
        self.text = text

    def peek(self) -> str:
        return self.tokens[self.i] if self.i < len(self.tokens) else ""

    def take(self) -> str:
        tok = self.peek()
        self.i += 1
        return tok

    def parse(self) -> Unit:
        value = self.term()
        while self.peek() in ("*", "/"):
            op = self.take()
            rhs = self.term()
            value = value * rhs if op == "*" else value / rhs
        if self.i != len(self.tokens):
            raise UnitError(f"trailing tokens in unit {self.text!r}")
        return value

    def term(self) -> Unit:
        tok = self.take()
        if tok == "":
            raise UnitError(f"empty unit expression {self.text!r}")
        if tok == "(":
            inner = self.parse()
            if self.take() != ")":
                raise UnitError(f"unbalanced parentheses in unit {self.text!r}")
            base = inner
        elif tok in SYMBOLS:
            base = SYMBOLS[tok]
        elif tok.replace(".", "", 1).isdigit():
            if float(tok) != 1.0:
                raise UnitError(f"numeric literal {tok!r} is not a unit")
            base = DIMENSIONLESS
        else:
            raise UnitError(f"unknown unit symbol {tok!r} in {self.text!r}")

        if self.peek() == "^":
            self.take()
            exp_tok = self.take()
            m = _EXP.match(exp_tok)
            if not m:
                raise UnitError(f"bad exponent {exp_tok!r} in {self.text!r}")
            num = int(m.group(1))
            den = int(m.group(2)) if m.group(2) else 1
            if den == 0:
                raise UnitError("zero denominator in exponent")
            base = base ** Fraction(num, den)
        elif self.peek() == "**":
            raise UnitError("use '^' for exponents, not '**'")
        return base


def parse_unit(text: str) -> Unit:
    """Parse an SI unit expression such as ``kg*m^2/s^2`` or ``N*m/rad``."""
    text = text.strip()
    if not text:
        return DIMENSIONLESS
    return _Parser(_lex(text), text).parse()
