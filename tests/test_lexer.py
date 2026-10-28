import pytest

from mycel.lexer import LexError, tokenize


def values(src: str):
    return [t.value for t in tokenize(src) if t.kind not in ("NEWLINE", "EOF")]


def test_connectives_are_single_tokens():
    assert values("@ h << f seed 0") == ["@", "h", "<<", "f", "seed", "0"]


def test_longest_match_wins():
    assert values("a <<< b") == ["a", "<<<", "b"]
    assert values("a << b") == ["a", "<<", "b"]
    assert values("a .* [p] b") == ["a", ".*", "[", "p", "]", "b"]


def test_comment_is_dropped():
    assert values("a := 1  # this is a comment\n") == ["a", ":=", "1"]


def test_continuation_suppresses_the_newline():
    toks = tokenize("a := 1 \\\n  + 2\n")
    kinds = [t.kind for t in toks]
    assert kinds.count("NEWLINE") == 1


def test_forbidden_characters():
    for ch in ";{}":
        with pytest.raises(LexError) as exc:
            tokenize(f"a := 1 {ch}\n")
        assert exc.value.code == "MYC0102"


def test_non_ascii_is_rejected():
    with pytest.raises(LexError) as exc:
        tokenize("@ h << f  # ok\n@ x := \u00b5\n")
    assert exc.value.code == "MYC0101"


def test_non_ascii_inside_a_comment_is_fine():
    tokenize("# mycelium, myc\u00e9lium, \u03bc\n@ a := 1\n")


def test_numbers_with_units_and_exponents():
    assert values("dt 0.5ms") == ["dt", "0.5", "ms"]
    assert values("eta 3e-4") == ["eta", "3e-4"]
    assert values("n^-0.5") == ["n", "^", "-", "0.5"]


def test_positions():
    toks = tokenize("@ h << f\n")
    assert toks[0].line == 1 and toks[0].col == 1
    assert toks[1].col == 3
