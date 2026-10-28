from fractions import Fraction

import pytest

from mycel.units import Unit, UnitError, parse_unit


def test_newton_metre_is_the_same_type_as_kg_m2_s2():
    assert parse_unit("N*m") == parse_unit("kg*m^2/s^2")
    assert str(parse_unit("N*m")) == "kg*m^2/s^2"


def test_watt_is_joule_per_second():
    assert parse_unit("W") == parse_unit("J/s")


def test_square_root_closes_the_algebra():
    assert parse_unit("m^2/s^2").sqrt() == parse_unit("m/s")


def test_rational_exponents_are_exact():
    u = parse_unit("m")
    assert u ** Fraction(1, 3) != u
    assert (u ** Fraction(1, 3)) ** 3 == u


def test_dimensionless_one():
    one = parse_unit("1")
    assert one.dimensionless
    assert parse_unit("rad").dimensionless


def test_angle_tags_survive():
    assert parse_unit("rad").tags == ("rad",)


def test_product_and_division():
    assert parse_unit("N*s") == parse_unit("kg*m/s")
    assert parse_unit("N/m") == parse_unit("kg/s^2")


def test_unknown_symbol_is_rejected():
    with pytest.raises(UnitError):
        parse_unit("furlong/s")


def test_double_star_is_rejected_with_a_hint():
    with pytest.raises(UnitError) as exc:
        parse_unit("m**2")
    assert "^" in str(exc.value)


def test_numeric_literal_must_be_one():
    with pytest.raises(UnitError):
        parse_unit("2")


def test_of_helper():
    assert Unit.of(kg=1, m=2, s=-2) == parse_unit("J")
