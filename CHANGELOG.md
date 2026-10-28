# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions follow the spec version, with a patch number for the reference front end.

## [0.9.1] - 2026-10-28

### Added

- Reference front end package `mycel-lang`:
  - `mycel/lexer.py` - tokenizer for the 94-code-point ASCII surface
  - `mycel/parser.py` - ledger parser with roles, connectives, seeds and type refs
  - `mycel/units.py` - SI unit algebra over `Q^7` with rational exponents
  - `mycel/axes.py` - named axis algebra (elementwise / contract / outer)
  - `mycel/checker.py` - static obligations and diagnostics
  - `mycel/cli.py` - `check`, `tokens`, `ast`, `units` subcommands
- 59 tests covering units, axes, lexing, parsing, checking and the examples
- Four `.myc` programs: biped walker, transformer block, thermostat, hello-field
- SVG assets: banner, logo, compilation pipeline, closed-loop field topology
- Documentation set (`docs/00` .. `docs/08`)
- Normative `spec/grammar.ebnf` and `spec/operators.tsv`
- GitHub Actions CI (Linux, macOS, Windows x Python 3.9-3.13) and a wheel build job

### Changed

- Surface syntax moved to **ASCII-only**. Every Unicode operator from the 0.9 draft has an
  ASCII equivalent: `⟲ -> <<`, `⊗_a -> .*[a]`, `⊣ -> !>`, `↝ -> =>`, `⇶ -> <<<`,
  `∇ -> grad(...)`, `∫ -> int[a..b]`, `⋄ -> ?`, `⌁ -> seed`, `⤳ -> |`.
- Contraction now always names its axis at the call site (`.*[a]`), so a weight matrix with
  two axes of equal extent can never be silently mis-contracted.

## [0.9.0] - 2026-10-27

### Added

- Initial specification: axioms A1-A5, CRFT execution model, SGTS type system,
  MLIR dialect set, hardware backends, zero-allocation memory model.
- Biped and transformer worked examples.

[0.9.1]: https://github.com/urbanyl/mycel/compare/v0.9.0...v0.9.1
[0.9.0]: https://github.com/urbanyl/mycel/releases/tag/v0.9.0
