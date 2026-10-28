import pytest

from mycel.parser import ParseError, parse_source
from mycel.units import parse_unit


SRC = """
@ h << ( -h + tanh( W_hh .*[pre] h ) ) / tau   seed 0   : real[post n, pre n] 1
"""


def test_field_entry_with_seed_and_type():
    (entry,) = parse_source(SRC)
    assert entry.role == "field"
    assert entry.name == "h"
    assert entry.connective == "<<"
    assert entry.seed is not None
    assert entry.seed_text().strip() == "0"
    assert entry.type is not None
    assert entry.type.manifold == "real"
    assert entry.type.axes.names() == ("post", "pre")
    assert entry.type.unit == parse_unit("1")


def test_dotted_name():
    (entry,) = parse_source("@ tissue.W_hh << f : real[post 6] 1")
    assert entry.name == "tissue.W_hh"


def test_assembly_params():
    src = "% Biped( mass 38kg , height 1.34m )\n  @ trunk := Rigid(mass)\n"
    first, _ = parse_source(src)
    assert first.role == "assembly"
    assert first.name == "Biped"
    assert first.params == ("mass", "height")


def test_assert_has_no_name():
    (entry,) = parse_source("assert pair( tau , dth ) : real[] W")
    assert entry.role == "assert"
    assert entry.name is None
    assert entry.type is not None
    assert entry.type.unit == parse_unit("W")


def test_unbalanced_bracket_is_reported():
    with pytest.raises(ParseError) as exc:
        parse_source("@ h := ( a + b\n")
    assert exc.value.code == "MYC0203"


def test_unknown_manifold():
    with pytest.raises(ParseError) as exc:
        parse_source("@ a := 1 : tensor[] 1")
    assert exc.value.code == "MYC0403"


def test_bad_unit():
    with pytest.raises(ParseError) as exc:
        parse_source("@ a := 1 : real[] furlong")
    assert exc.value.code == "MYC0401"


def test_unknown_role():
    with pytest.raises(ParseError) as exc:
        parse_source("hello there\n")
    assert exc.value.code == "MYC0201"


def test_directive():
    (entry,) = parse_source("$ world dt=1ms solver=rk4")
    assert entry.role == "directive"
    assert entry.name == "world"


def test_export():
    (entry,) = parse_source("> ports(h, nu, y)")
    assert entry.role == "export"
    assert entry.name == "ports"
    assert entry.params == ("h", "nu", "y")
