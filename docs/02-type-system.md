# 02 — The Spatial Geometry Type System (SGTS)

Every MYCEL value has a type:

```
⧼ 𝕄^{a₁ n₁, …, a_k n_k} │ 𝔲 ⧽          written in ASCII as:   𝕄[a₁ n₁, …, a_k n_k] 𝔲
```

where `𝕄` is a manifold, `aᵢ` are **named axes**, `nᵢ` are extents (integer variables, constants, or bounded symbols), and `𝔲 ∈ ℚ⁷` is an SI exponent vector.

Three orthogonal lattices are checked at compile time. All three are erased before codegen — the runtime representation of a value carries no type, no axis names, and no units.

---

## 1. Axis algebra

Positional dimensions are the single largest source of silent bugs in numerical code. `(B, T, H, D)` and `(B, H, T, D)` have the same rank, and a positional type system cannot tell them apart. MYCEL names axes.

| Operation | Requires | Result |
|---|---|---|
| `A * B` | `axes(A) ≡ axes(B)` | same axes |
| `A + B`, `A - B` | `axes(A) ≡ axes(B)` | same axes |
| `A .*[a] B` | `a ∈ axes(A) ∩ axes(B)`, all **other** axes align | `axes(A) ∪ axes(B) − {a}` |
| `A ** B` | `axes(A) ∩ axes(B) = ∅` | `axes(A) ∪ axes(B)` |
| `cat[a](A, B)` | all axes except `a` align | `axes(A ∪ B)` with `a` extended |
| `A >>[a] foo` | `a ∈ axes(A)` | free reindex / layout change |
| `argmax[a](A)` | `a ∈ axes(A)` | `axes(A) − {a}` |
| `int[τ] A` | `τ` names a time-like axis | `axes(A) − {time}` |

**Rule: no implicit broadcasting.** If you want a scalar to participate, it must be axis-free (`real[]`). If you want two axes to align, spell it out with `>>[a]`. This forbids the entire class of "silently broadcast the wrong way" failures.

### 1.1 Extent solving

Extents are integer terms. Unification reduces to a Presburger formula in the theory of linear integer arithmetic:

```
T = H * 64            → linear, solved by the Presburger solver
B = ceil(N / 32)      → linear with div/mod, still Presburger
d = sqrt(vocab)       → nonlinear, deferred to Z3 under --smt
```

The default pipeline runs the linear fragment. The `--smt` flag enables Z3 for nonlinear relations; unsolved obligations become *bounded deployment asserts* rather than errors, so a model can still ship with a symbolic batch dimension.

---

## 2. Unit algebra

Units are exponent vectors in `ℚ⁷` over `(kg, m, s, A, K, mol, cd)`.

```
N  = kg·m·s⁻²              → (1, 1, -2, 0, 0, 0, 0)
J  = N·m = kg·m²·s⁻²       → (1, 2, -2, 0, 0, 0, 0)
W  = J/s = kg·m²·s⁻³       → (1, 2, -3, 0, 0, 0, 0)
Pa = N/m²                  → (1, -1, -2, 0, 0, 0, 0)
T  = kg·s⁻²·A⁻¹            → (1, 0, -2, -1, 0, 0, 0)
```

`rad` and `sr` are dimensionless but **tagged**; under `--strict-angles`, `real[] rad` does not unify with `real[] 1` without an explicit `rad_to_1(x)`. This catches the classic `sin(x)` where `x` was in degrees.

### 2.1 Why rational exponents

Because `sqrt(m^2/s^2) = m/s` appears in every normalization, every standard deviation, and every RMS. Integer exponents force you to either lie or write casts. `ℚ⁷` closes under `*`, `/`, and `^r` for rational `r`.

### 2.2 Transparent derived units

The parser resolves derived names eagerly, so `N*m` and `kg*m^2/s^2` are the *same type*, not merely convertible. Diagnostics print whichever spelling is shortest.

---

## 3. Tangent / cotangent discipline

This is the part of SGTS with no analogue in mainstream numerical languages, and it is the one that pays for itself fastest in robotics.

A configuration manifold `Q` (for a biped, `12` revolute joints) has:

- a **tangent bundle** `TQ`: velocities, accelerations, forces-along-motion
- a **cotangent bundle** `T*Q`: momenta, torques, gradients

They are *not* the same space. They are canonically paired by the metric, and confusing them is the source of sign errors, transposed Jacobians, and "why does my PD controller inject energy" bugs.

MYCEL writes them as `Tan[...]` and `Cot[...]`, and refuses to unify them.

```myc
@ th    ~> agent.enc.angle   : Tan[joint 12] rad
@ dth   ~> agent.enc.rate    : Tan[joint 12] rad/s
@ tau   :=  ...              : Cot[joint 12] N*m

@ tau   !> agent.joints                       # legal: actuator consumes a covector
assert pair(tau, dth) : real[] W               # the ONLY legal combination
```

`pair(tau, dth)` is the natural pairing `⟨τ, θ̇⟩`; it produces mechanical power in watts. Anything else is a type error:

```
MYC0402: cannot unify Cot[joint 12] with Tan[joint 12]
  hint: did you mean pair(tau, dth)?  it yields real[] W
```

### 3.1 Pullback and pushforward

Jacobians are typed by which bundle they move between:

```
J  : Tan[Q] → Tan[X]          (forward kinematics Jacobian)
J* : Cot[X] → Cot[Q]          (its transpose, mapping end-effector forces to joint torques)
```

`>>[a]` on a `Tan` value with a Jacobian performs pushforward; on a `Cot` value it performs pullback. Writing the wrong one is a compile error, not a mysterious slowdown.

### 3.2 Manifold-aware primitives

| Manifold | Primitive | Guarantee |
|---|---|---|
| `S2` / `S3` | `normalize`, `slerp`, `log`, `exp` | stays on the manifold |
| `SO3` / `SE3` | `compose`, `inverse`, `adjoint` | group axioms asserted |
| `Simplex` (from `S{n-1}` with positive parts) | `softmax[a]` | `sum[a] = 1`, `>= 0` |
| `Tan`/`Cot` | `pair`, `pullback`, `pushforward` | no bundle mixing |

A `softmax` over a simplex-typed value is compiled knowing the constraint `Σ = 1`, which lets the compiler fuse the max-shift, the exponential, and the normalisation into a single pass — and lets `assert sum[key](Pp) == 1` be discharged without running the program.

---

## 4. Effects

Each field carries an effect row: `{physics, spike, stochastic, evolve, io}` plus purity.

```myc
@ h << ...              # pure, +stochastic (it draws ?N at genesis)
@ tau !> agent.joints   # +physics
@ x ~> dataset.tokens   # +io
```

Effects are used for:

- **Scheduling** — `+physics` fields join the physics domain; `+io` fields are demand-driven.
- **Differentiation** — `grad` may only traverse pure and `+spike` (with `surro`) edges. A `+physics` edge is a hard boundary, and attempting to differentiate across it is `MYC0501`.
- **Assertions** — `assert pure(grad(tissue.W_hh, L))` proves the learning update has no hidden I/O.

---

## 5. Erasure

After verification, a value's type is reduced to a **layout descriptor**:

```
{L, H, T, D}   // axis order, as chosen by the layout optimiser
{contiguous | strided | blocked(k) | sharded(d)}
{element type, alignment}
```

The `myc-mem-plan` pass uses these descriptors to choose blocking and to prove that no two buffers alias. Nothing about units or manifold structure survives into the binary.

**Cost of the type system at runtime: zero.**

---

## 6. Worked example

```myc
@ Kp    := 120.0                 : real[] N*m/rad
@ Kd    := 6.5                   : real[] N*m*s/rad
@ thref := ...                   : Tan[joint 12] rad
@ dth   := ...                   : Tan[joint 12] rad/s

@ tauPD := Kp * (thref - th) + Kd * (0 - dth)
```

Verification of `Kp * (thref - th)`:

```
thref - th : Tan[joint 12] rad
Kp         : real[] N*m/rad
product    : Tan[joint 12] N*m          ← axis: Tan
```

Verification of `Kd * (-dth)`:

```
-dth       : Tan[joint 12] rad/s
Kd         : real[] N*m*s/rad
product    : Tan[joint 12] N*m
```

Sum: `Tan[joint 12] N*m`. But `tau` must be `Cot[joint 12] N*m` — the assertion at the actuation site fails:

```
MYC0402: Cot[joint 12] N*m ≠ Tan[joint 12] N*m
```

The fix is to route through the metric explicitly (`Cot` from a `Tan` requires a metric, and the compiler will not supply one silently):

```myc
@ tauPD := metric_Q **[] ( Kp * (thref - th) + Kd * (0 - dth) )  : Cot[joint 12] N*m
```

This is the kind of error that costs a robotics team three days of debugging on hardware. Here it costs one compile.

---

Next: [03 — Compiler Pipeline](03-compiler-pipeline.md)
