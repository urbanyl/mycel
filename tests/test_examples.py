from pathlib import Path

import pytest

from mycel.checker import check_file
from mycel.parser import parse_source

EXAMPLES = sorted(Path(__file__).resolve().parent.parent.joinpath("examples").glob("*.myc"))


def test_examples_exist():
    assert len(EXAMPLES) >= 4


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_example_has_no_errors(path):
    diags = check_file(path)
    errors = [d for d in diags if d.severity == "error"]
    assert errors == [], "\n".join(d.render() for d in errors)


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_example_is_ascii_only(path):
    raw = path.read_bytes()
    bad = [(i, b) for i, b in enumerate(raw) if b > 126 and b not in (0x0D, 0x0A, 0x09)]
    assert bad == []


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_example_parses_and_has_fields(path):
    entries = parse_source(path.read_text(encoding="utf-8"))
    assert any(e.role == "field" for e in entries)
    assert any(e.role == "assert" for e in entries)
