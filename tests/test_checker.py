from mycel.checker import check_source


def codes(diags):
    return [d.code for d in diags]


def test_clean_ledger_has_no_errors():
    src = """
$ world dt=1ms
@ a := 1.0                : real[] 1
@ h << -h / 0.1  seed 0   : real[] 1
assert h : real[] 1
"""
    diags = check_source(src)
    assert [d for d in diags if d.severity == "error"] == []


def test_recurrence_without_seed_or_genesis_is_an_error():
    src = "@ h << -h / 0.1 : real[] 1\n"
    diags = check_source(src)
    assert "MYC0601" in codes(diags)


def test_genesis_can_be_the_bind_line():
    src = """
@ h := 0.0     : real[] 1
@ h << -h/0.1  : real[] 1
"""
    diags = check_source(src)
    assert "MYC0601" not in codes(diags)


def test_genesis_by_leaf_name_through_a_path():
    src = """
% Tissue( n 256 )
  @ W := ?N(0, 1)          : real[a n, b n] 1
@ Tissue.W << -0.1 * grad(Tissue.W, L)
"""
    diags = check_source(src)
    assert "MYC0601" not in codes(diags)


def test_unknown_grad_target_is_a_warning():
    src = "@ W << -0.1 * grad(nowhere, L)  seed 0 : real[] 1\n"
    diags = check_source(src)
    assert "MYC0502" in codes(diags)
    assert all(d.severity == "warning" for d in diags if d.code == "MYC0502")


def test_forbidden_character():
    diags = check_source("@ a := 1;\n")
    assert codes(diags) == ["MYC0102"]


def test_non_ascii():
    diags = check_source("@ \u00b5 := 1\n")
    assert codes(diags) == ["MYC0101"]


def test_strict_angles_warns_on_plain_zero_seed():
    src = "@ phi << 2.0  seed 0 : real[] rad\n"
    diags = check_source(src, strict_angles=True)
    assert "MYC0404" in codes(diags)


def test_missing_type_on_a_flow_warns():
    src = "@ y => W .*[in] x\n"
    diags = check_source(src)
    assert "MYC0401" in codes(diags)
