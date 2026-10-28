"""Tokenizer for the MYCEL ledger surface syntax.

The alphabet is exactly the 94 printable ASCII code points. Anything else
in code is MYC0101; ';', '{' and '}' are MYC0102 regardless of context.

Newline terminates an entry. A trailing backslash continues an entry onto
the next physical line, so the lexer emits no NEWLINE for it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

FORBIDDEN = {";", "{", "}"}

MULTI = (
    "<<<",
    ">>",
    "<<",
    "~>",
    "=>",
    "!>",
    ":=",
    "~~",
    "||",
    "..",
    ".*",
    "**",
    "~*",
    "++",
    ">=",
    "<=",
    "==",
    "!=",
)

SINGLE = set("@%$><=+-/^?|*:,.()[]'!~")

KEYWORDS = {"assert", "seed", "real", "S2", "S3", "SO3", "SE3", "Tan", "Cot"}


@dataclass(frozen=True)
class Token:
    kind: str
    value: str
    line: int
    col: int

    def __str__(self) -> str:
        return f"{self.kind}({self.value!r})"


class LexError(ValueError):
    def __init__(self, code: str, message: str, line: int, col: int) -> None:
        super().__init__(f"{code}: {message} (line {line}, col {col})")
        self.code = code
        self.message = message
        self.line = line
        self.col = col


def _is_ident_start(ch: str) -> bool:
    return ch.isalpha() or ch == "_"


def _is_ident_part(ch: str) -> bool:
    return ch.isalnum() or ch == "_"


def tokenize(source: str) -> List[Token]:
    """Tokenize MYCEL source. Raises LexError on the first violation."""
    tokens: List[Token] = []
    i = 0
    line = 1
    col = 1
    n = len(source)

    def err(code: str, message: str, l: int, c: int) -> "LexError":
        return LexError(code, message, l, c)

    while i < n:
        ch = source[i]

        # -- line endings -------------------------------------------------
        if ch == "\r":
            i += 1
            if i < n and source[i] == "\n":
                i += 1
            tokens.append(Token("NEWLINE", "\n", line, col))
            line += 1
            col = 1
            continue
        if ch == "\n":
            i += 1
            tokens.append(Token("NEWLINE", "\n", line, col))
            line += 1
            col = 1
            continue

        # -- whitespace ---------------------------------------------------
        if ch in " \t":
            i += 1
            col += 1
            continue

        # -- continuation -------------------------------------------------
        if ch == "\\":
            j = i + 1
            while j < n and source[j] in " \t":
                j += 1
            if j < n and source[j] == "\r":
                j += 1
            if j < n and source[j] == "\n":
                i = j + 1
                line += 1
                col = 1
                continue
            raise err("MYC0102", "backslash is only legal at end of line", line, col)

        # -- string literals ----------------------------------------------
        if ch == '"':
            start = i
            start_col = col
            i += 1
            col += 1
            while i < n and source[i] not in '"\r\n':
                i += 1
                col += 1
            if i >= n or source[i] != '"':
                raise err("MYC0104", "unterminated string literal", line, start_col)
            i += 1
            col += 1
            tokens.append(Token("STRING", source[start + 1 : i - 1], line, start_col))
            continue

        # -- comments -----------------------------------------------------
        if ch == "#":
            while i < n and source[i] not in "\r\n":
                i += 1
                col += 1
            continue

        # -- alphabet check -----------------------------------------------
        if ord(ch) > 126 or ord(ch) < 32:
            raise err(
                "MYC0101",
                f"non-ASCII byte U+{ord(ch):04X} in code; only the 94 printable "
                "ASCII code points are permitted outside comments",
                line,
                col,
            )
        if ch in FORBIDDEN:
            raise err("MYC0102", f"forbidden character {ch!r}", line, col)

        # -- numbers ------------------------------------------------------
        if ch.isdigit():
            start = i
            start_col = col
            while i < n and source[i].isdigit():
                i += 1
                col += 1
            if i < n and source[i] == "." and i + 1 < n and source[i + 1].isdigit():
                i += 1
                col += 1
                while i < n and source[i].isdigit():
                    i += 1
                    col += 1
            if i < n and source[i] in "eE":
                j = i + 1
                if j < n and source[j] in "+-":
                    j += 1
                if j < n and source[j].isdigit():
                    i = j
                    while i < n and source[i].isdigit():
                        i += 1
                    col = start_col + (i - start)
            tokens.append(Token("NUMBER", source[start:i], line, start_col))
            continue

        # -- identifiers --------------------------------------------------
        if _is_ident_start(ch):
            start = i
            start_col = col
            while i < n and _is_ident_part(source[i]):
                i += 1
                col += 1
            if i < n and source[i] == "'":
                i += 1
                col += 1
            word = source[start:i]
            tokens.append(Token("IDENT", word, line, start_col))
            continue

        # -- operators ----------------------------------------------------
        matched = None
        for op in MULTI:
            if source.startswith(op, i):
                matched = op
                break
        if matched:
            tokens.append(Token("OP", matched, line, col))
            i += len(matched)
            col += len(matched)
            continue

        if ch in SINGLE:
            tokens.append(Token("OP", ch, line, col))
            i += 1
            col += 1
            continue

        raise err("MYC0101", f"unexpected character {ch!r}", line, col)

    tokens.append(Token("EOF", "", line, col))
    return tokens
