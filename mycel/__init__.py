"""MYCEL - the mycelial reactive field language.

Reference front end: lexer, ledger parser, SI unit algebra, axis algebra,
static checker and command line interface.

The compiler backend described in docs/03-compiler-pipeline.md is not
implemented here. This package exists so the specification can be tested
rather than merely believed.
"""

from .axes import Axis, Axes, AxisError
from .checker import Diagnostic, check_source, check_file
from .lexer import LexError, Token, tokenize
from .parser import Entry, ParseError, TypeRef, parse_source
from .units import DIMENSIONS, Unit, UnitError, parse_unit

__version__ = "0.9.1"
__all__ = [
    "Axis",
    "Axes",
    "AxisError",
    "Diagnostic",
    "DIMENSIONS",
    "Entry",
    "LexError",
    "ParseError",
    "Token",
    "TypeRef",
    "Unit",
    "UnitError",
    "check_file",
    "check_source",
    "parse_source",
    "parse_unit",
    "tokenize",
    "__version__",
]
