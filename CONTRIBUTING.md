# Contributing to MYCEL

MYCEL is a specification-first project. Most of what exists is prose, grammar, and a
reference front end. That means the highest-value contributions are **arguments**, not
patches.

## The three things that help most

### 1. Adversarial examples

Write a `.myc` file that *should* compile and does not, or one that should *not* and does.
Attach it to an issue with what you expected and what happened.

The most useful adversarial examples break one of the five axioms
([docs/00-philosophy.md](docs/00-philosophy.md)):

- a program that needs a sequential dependency the ledger cannot express
- a field whose dimensions are correct but whose checker rejects it
- a `grad` that must cross a `~>` edge
- a case where `.*[a]` is ambiguous

### 2. Unit and axis algebra edge cases

Rational exponents have surprising corners. `sqrt(N*m)` is not obviously `N^0.5 * m^0.5` in
every derived-unit table. If you find a pair of units that should unify and do not, that is
a bug worth a lot.

### 3. Solver and integrator heuristics

`myc-flow-stiffness` is specified but unimplemented, and it is the pass most likely to be
wrong in an interesting way. Proposals that come with a counterexample ODE are far more
valuable than proposals that come with an opinion.

## Ground rules for the surface syntax

The surface is **ASCII-only**. This is not negotiable:

- Only the 94 printable ASCII code points are legal in code.
- Comments may be UTF-8, but the examples in `examples/` must be byte-for-byte ASCII.
- No `;`, `{`, `}` anywhere in code.
- No new keyword may shadow an existing operator token.

Adding an operator requires:

1. An entry in `spec/operators.tsv` with precedence and associativity.
2. A production in `spec/grammar.ebnf`.
3. A row in the table in `docs/01-language-spec.md`.
4. A test in `tests/test_lexer.py` proving the token does not break longest-match.

## Working on the front end

```bash
git clone https://github.com/urbanyl/mycel.git
cd mycel
pip install -e ".[dev]"
pytest
```

Style: standard library only, no dependencies, type hints on public functions. The package
targets Python 3.9+, so no `match` statements and no `X | Y` in annotations at runtime
(use `from __future__ import annotations`).

Before opening a PR:

```bash
pytest
mycel check examples/01-biped-walker.myc
mycel check examples/02-transformer-block.myc
mycel check examples/03-thermostat.myc
mycel check examples/04-hello-field.myc
```

All four must report `ok - every obligation discharged`.

## Working on the specification

Prose changes are reviewed for **falsifiability**. Every claim should be one that a
counterexample could disprove. "MYCEL is expressive" is not a claim. "Every field is a pure
function of its inputs within a tick, with the sole exceptions of `!>` and hardware `~>`" is.

If you add a construct, add:

- what it desugars to in the RCT
- what it does under `grad`
- what it lowers to on each of the five backend families
- what it does to WCET

## Commit messages

```
area: short imperative summary

Optional body explaining why, not what.
```

Areas: `spec`, `frontend`, `docs`, `examples`, `assets`, `ci`.

## Code of conduct

Be technical, be kind, and assume the other person has read the spec more carefully than you
have. See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## License

By contributing you agree that your contribution is licensed under the MIT License.
