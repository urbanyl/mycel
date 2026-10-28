import pytest

from mycel.axes import Axes, AxisError


def A(spec: str) -> Axes:
    return Axes.parse(spec)


def test_parse_axes():
    axes = A("batch B, time T, head 12, proj 64")
    assert axes.names() == ("batch", "time", "head", "proj")
    assert axes.get("head").extent == "12"


def test_elementwise_requires_identical_axes():
    A("batch B, time T").elementwise(A("batch B, time T"))
    with pytest.raises(AxisError):
        A("batch B, time T").elementwise(A("batch B, head H, time T"))


def test_contraction_removes_the_named_axis():
    q = A("batch B, time T, head H, proj 64")
    k = A("batch B, key T, head H, proj 64")
    assert q.contract(k, "proj").names() == ("batch", "time", "head", "key")


def test_contraction_rejects_extent_conflict():
    with pytest.raises(AxisError):
        A("proj 64").contract(A("proj 32"), "proj")


def test_outer_product_requires_disjoint_axes():
    assert A("batch B").outer(A("proj 64")).names() == ("batch", "proj")
    with pytest.raises(AxisError):
        A("batch B").outer(A("batch B"))


def test_duplicate_axis_name_is_rejected():
    with pytest.raises(AxisError):
        A("time T, time T")


def test_missing_axis_in_contraction():
    with pytest.raises(AxisError):
        A("batch B").contract(A("proj 64"), "proj")
