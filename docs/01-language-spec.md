# 01 — Language Specification

**Version:** `0.9a` · **Status:** design draft · **Surface:** ASCII-only (94 printable code points)

---

## 1. Identity

| | |
|---|---|
| **Name** | MYCEL |
| **Expansion** | *Mycelial Reactive Field Language* |
| **Etymology** | *mycelium*, the hyphal lattice of a fungus — a decentralised network that routes by demand and rewires itself; also *my cell* |
| **File extension** | `.myc` |
| **Paradigm** | Continuous Reactive Field Topologies (CRFT), differentiable dataflow, neuro-evolutionary |
| **Evaluation** | Non-sequential. Monotone frontier advancement per clock domain. |
| **Typing** | Static, structural, dimension-polymorphic; all checks erased before codegen |
| **Memory** | Static arena planning; zero steady-state allocations |

A program denotes a section `Φ ∈ Γ(E)` of a bundle `E → (Time × Space)` satisfying a system `𝒮` of differential-algebraic equations, together with a variation law `𝒱` on the discrete topology of its own operators. Compilation is the triple *(discretize, partition, schedule)*.

---

## 2. Execution model

A compiled artifact is a **Resolved Field Topology (RFT)**: a hypergraph whose vertices are fields and whose hyperedges are operators, each annotated with a rate class, an effect row, and a device placement.

### 2.1 Phase hydration

1. **Elaboration** — declarations unify into a hypergraph. Free names inside `%` assemblies resolve at instantiation (assemblies are *open* hypergraphs).
2. **Stratification** — every field inherits a rate class from its connective:

   | Rate class | Connectives | Typical period |
   |---|---|---|
   | physics | `!>`, `~>` on hardware ports | 0.5–2 ms |
   | neural | `~>`, `<<` | 0.5–2 ms |
   | gradient | `=>`, `grad(...)` | 5–20 ms |
   | evolution | `<<<` | 0.1–5 s |
   | io | `~>` on dataset ports | event-driven |

3. **Frontier advancement** — each clock domain advances independently on a monotone timestamp lattice. Where a domain consumes another's output, an explicit hydration barrier `||` inserts a versioned handoff. There is no global lockstep and no global clock.
4. **Hybrid push/pull** — hardware events push forward from sensor ports; actuator demands pull backward. Dead subgraphs therefore never execute, dynamically as well as statically.
5. **Fixed-point closure** — algebraic cycles (contact impulses depending on penetration depending on impulses) are not errors. They lower to `myc.flow.solve`, with the relaxation chosen from the cycle's contraction properties.
6. **Determinism** — all `?` draws are counter-based: `rng(hash(tick, field_id, lane))`. Results are bit-identical across devices and thread counts.

### 2.2 On neuromorphic silicon

There is no clock. `<<` becomes a self-timed neuron population; `~>` becomes an AER route; `||` degrades into an asynchronous handshake. The same source compiles to a lockstep SIMT kernel or a free-running event fabric.

---

## 3. Lexical structure

### 3.1 Character set

The source alphabet is exactly the 94 printable ASCII code points:

```
A-Z a-z 0-9  _ . : = < > ~ | * + - / ^ ? @ % $ ! ( ) [ ] , ' # \   space
```

Nothing else is permitted in code. `--ascii-only` (default) rejects any non-ASCII byte outside comments; `--allow-unicode-comments` relaxes this for `#` lines only.

### 3.2 Tokens, longest match first

```
<<<   >>    <<    ~>    =>    !>    :=    ~~    ||    ..
.*    **    ~*    ++    >=    <=    ==    !=
@  %  $  >  |  *  +  -  /  ^  ?  =  <  :  ,  .  [  ]  (  )  '  !
```

Identifiers are `[A-Za-z_][A-Za-z0-9_]*`. An identifier may contain a trailing prime (`'`) to denote a derived or reference value: `thref`, `theta'`.

### 3.3 Lines

- One entry per line. Newline is the only terminator; `;` is a lexical error.
- A trailing `\` continues the entry onto the next physical line.
- `#` starts a comment that runs to end of line.
- Whitespace and indentation are **never** significant. Indentation inside `%` assemblies is cosmetic.

---

## 4. Grammar

```
ledger      := entry*
entry       := role  ident  [ connective expr ]  [ seed ]  [ ':' type ]  [ guard ]
role        := '@' | '%' | '$' | 'assert' | '>' | '#'
connective  := ':=' | '~>' | '=>' | '<<' | '<<<' | '!>'
seed        := 'seed' expr
guard       := '|' expr

type        := manifold '[' axes ']' unit
manifold    := 'real' | 'S2' | 'S3' | 'SE3' | 'SO3' | 'Tan' | 'Cot'
axes        := axis (',' axis)*
axis        := ident expr
unit        := unit-expr | '1'

expr        := term (binop term)*
binop       := '.*' '[' ident ']' | '**' | '~*' | '*' | '/' | '+' | '-'
             | '++' | '|' | '>=' | '<=' | '==' | '!='
term        := number unit-suffix?
             | ident
             | ident '(' arglist? ')'
             | ident '[' ident ']' '(' arglist? ')'
             | '(' expr ')'
             | '?' ident '(' arglist? ')'
             | 'grad' '(' expr ',' expr ')'
             | 'int' '[' expr '..' expr ']' expr
             | 'argmax' '[' expr ']' '(' expr ')'
             | '-' term
```

The normative EBNF is in [`spec/grammar.ebnf`](../spec/grammar.ebnf).

---

## 5. Roles

| Sigil | Meaning | Notes |
|---|---|---|
| `@` | declare a field | the only way to introduce a value |
| `%` | declare an assembly | an open hypergraph with named ports |
| `$` | world / target directive | `dt`, `gravity`, `solver`, backend hints |
| `assert` | compile-time obligation | discharged by SGTS, then erased |
| `>` | export a port | only legal inside `%` |
| `#` | comment | to end of line |

An assembly has no `return`. It has ports:

```myc
% LiquidTissue(n 256, inn 44, out 12)
  @ W_hh := ?N(0, n^-0.5)  : real[post n, pre n] 1
  @ h    << ( -h + tanh( W_hh .*[pre] h + W_xh .*[in] x + b ) ) / tau  seed 0
  > ports(h, nu, y)
```

`x`, `W_xh`, `b`, `tau`, `nu`, `y` are free names. They resolve against the instantiating ledger when the assembly is applied: `@ tissue := LiquidTissue(256, 44, 12)`. This is *lexical open-hypergraph instantiation*, not dynamic scope: the resolution is fully static and every resolved edge appears in the RCT.

---

## 6. Connectives

| Token | Name | Semantics |
|---|---|---|
| `:=` | bind | Timeless definition, evaluated once. Also the implicit `seed` of any later `<<` or `<<<` on the same name. |
| `~>` | synapse | Event-driven stream. Carries discrete events or spikes. **Not differentiable** — an implicit cotangent cut. |
| `=>` | flow | Differentiable stream. Allocates a cotangent slot and joins the reverse-mode graph. |
| `<<` | recur | Continuous-time state. `h << f` asserts `ḣ = f`, with the seed as initial condition. |
| `<<<` | evolve | Structural evolution on a slower timescale. Non-differentiable, variation-driven. |
| `!>` | actuate | Terminal write into a physical or effector manifold; imposes a boundary condition. |

A name may be bound with `:=` and then lifted into time with `<<`. The static binding supplies the genesis:

```myc
@ W_hh := ?N(0, n^-0.5)          # genesis
@ W_hh << -eta * grad(W_hh, L)   # ā = -eta * gradient; seed is the line above
```

---

## 7. Expression operators

| Token | Name | Rule |
|---|---|---|
| `*` | elementwise | Axes must be **identical**. There is no broadcasting. |
| `.*[a]` | contraction | Sums the named axis `a`. All other axes must align. |
| `**` | outer product | All axes must be **disjoint**; result carries the union. |
| `~*` | manifold product | Geometric / convolutional product under metric `g`; generalises convolution, attention pooling, gauge transport. |
| `+` `-` `/` `^` | pointwise | Identical axes; scalars broadcast only when one side is axis-free. |
| `++` | fuse | Topological union of two hypergraphs sharing ports. Used for residual composition of *structures*, not values. |
| `cat[a](...)` | concatenate | Introduces or extends axis `a`. Never implicit. |
| `>>[a]` | reindex | Layout / label change along `a`. Free at runtime; explicit in IR. |
| `?` | sample | Counter-based draw: `?N(0, 1)`, `?U(-0.05, 0.05)`. |
| `grad(x, L)` | gradient | Reverse-mode query. Emits the transposed subgraph reaching `x` from `L`. |
| `int[a..b]` | measure integral | Replaces every accumulation loop. |
| `argmax[a](...)` | rank select | Replaces branching on a rank. |
| `x \| p` | guard | `x` is defined only where predicate `p` holds; elsewhere it contributes nothing (additive identity). |
| `cut(x)` | cotangent cut | Explicit, auditable stop-gradient. |
| `surro(x, d)` | surrogate | Attaches pseudo-derivative `d` to the non-differentiable event `x`. |
| `f ~~ f_adj` | adjoint | Declares an analytic adjoint; autodiff skips tracing it. |
| `a \|\| b` | hydrate | Versioned barrier between clock domains. |
| `pair(a, b)` | pairing | The only well-typed product of `Tan` and `Cot`; yields an energy/power. |

### 7.1 Why `.*` and not `@`

Contraction is written `.*[axis]` so that the summed axis is always *named at the call site*. Silent Einstein summation over every shared axis is convenient until a weight matrix has two axes of the same extent — at which point the compiler must guess, and guessing is how dimension bugs are born.

### 7.2 Why `**` is not `.*`

`A ** B` requires disjoint axes and returns the union. `A * B` requires identical axes and returns them unchanged. `A .*[a] B` removes `a`. Three operators, three unambiguous shapes, zero broadcasting.

---

## 8. Types

```
real[batch B, time T, head H, proj 64] 1
Tan[joint 12] rad/s
Cot[joint 12] N*m
real[coord 3] m/s^2
S3 1
```

Units are `Q^7` exponent vectors over `(kg, m, s, A, K, mol, cd)`, written with `*`, `/`, `^` and a table of derived names (`N`, `J`, `W`, `Pa`, `Hz`, `V`, `Ohm`, `T`, `F`, `H`, `C`, `S`, `Gy`, `Sv`). `N*m ≡ kg*m^2/s^2`. `rad` and `sr` are dimensionless with a *tag* so that `rad` and `1` unify but warn under `--strict-angles`.

`Tan` and `Cot` never unify. They compose only through `pair`, which yields `real[]`.

Full treatment: [02 — Type System](02-type-system.md).

---

## 9. Determinism and purity

- Every field is a pure function of its inputs *within a tick*.
- The only impurity is `!>`, which mutates the external world, and `~>` on hardware ports, which observes it.
- Effect rows make this explicit: a field that reads a hardware port has effect `+io`; one that writes has `+physics`.
- `assert` may inspect effect rows: `assert pure(grad(tissue.W_hh, L))`.

---

## 10. Error taxonomy

| Code | Meaning |
|---|---|
| `MYC0101` | non-ASCII byte in code |
| `MYC0102` | forbidden character (`;`, `{`, `}`) |
| `MYC0201` | unknown connective |
| `MYC0301` | axis mismatch — no implicit broadcasting |
| `MYC0302` | dimension contradiction (integer solver) |
| `MYC0303` | dimension contradiction (nonlinear; requires `--smt`) |
| `MYC0401` | unit mismatch |
| `MYC0402` | `Tan`/`Cot` unification attempt |
| `MYC0501` | gradient path is cut by `cut(...)` |
| `MYC0502` | `grad` target unreachable from loss |
| `MYC0601` | `<<` has no seed and no `:=` genesis |
| `MYC0701` | `wcet(tick) > dt_phys` |
| `MYC0702` | event queue bound not provable |

---

Next: [02 — Type System](02-type-system.md) · [03 — Compiler Pipeline](03-compiler-pipeline.md)
