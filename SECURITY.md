# Security Policy

## Supported versions

| Version | Supported |
|---|---|
| 0.9.x | yes |
| < 0.9 | no |

## Scope

MYCEL is currently a specification plus a reference **front end**. There is no code
generator, no runtime, and no network service, so the attack surface is deliberately small.

Relevant issues:

- Parser or lexer crashes on hostile input (denial of service in tooling)
- Path traversal in the CLI when reading `.myc` files
- Any behaviour that causes a `.myc` file to execute code during `check` (it must not)
- Supply-chain issues in the GitHub Actions workflows

Out of scope for now:

- Findings that require the (unimplemented) compiler backend
- Theoretical issues in the memory model, unless you can show the specification is
  internally inconsistent

## Reporting

Open a private security advisory via
[GitHub Security Advisories](https://github.com/urbanyl/mycel/security/advisories/new).

Please include a minimal `.myc` reproducer and the exact command you ran.

## Design commitments

If and when the backend exists, these are the invariants we intend to hold:

- `mycel check` never executes user code. It parses and verifies only.
- Generated artifacts contain no dynamic allocation on the hot path, so allocator-based
  exploitation has no surface there.
- Every artifact ships certificates (geometry, aliasing, timing) so a deployed binary can be
  audited without the build machine.
