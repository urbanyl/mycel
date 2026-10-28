# 07 — FAQ

### Is this a real language?

It is a **design specification** with a working reference *front end* (lexer, parser, unit/axis checker, CLI) in this repository. The compiler backend described in [03](03-compiler-pipeline.md) is specified in detail but not yet implemented. The repository is explicit about which parts exist and which are designed.

### Why ASCII only?

Unicode glyphs make a specification beautiful and a toolchain hostile. `⟲` cannot be typed on a phone, has no stable case-folding, breaks `grep`, defeats most editors' column math, and requires a font you cannot guarantee on a colleague's machine. The paradigm does not need exotic glyphs — it needs unambiguous operators. `<<`, `~>`, `=>`, `!>` carry the same meaning with none of the tax.

Comments may still be UTF-8.

### Why no `if`?

Because `if` is a statement about *control*, and MYCEL programs have no control. What you actually want in a dataflow program is one of three things:

- **masking** → `x | predicate`
- **selection** → `argmax[a](...)` or `softmax`-weighted mixing
- **modulation** → `a * gate`

All three are branch-free, therefore vectorisable and differentiable. A "branch" that only selects data is not a branch.

### Why no `for`?

Because every loop over time is an ODE (`<<`), every loop over a collection is an axis, and every loop over a window is `int[τ]`. If you find yourself wanting a counter, you are probably describing a field with a phase — model the phase instead:

```myc
@ phi << 2 * pi * f_cpg   seed 0
@ clk := cat[phase](sin(phi), cos(phi))
```

That is a gait generator, and it composes, differentiates, and lowers to a neuromorphic oscillator.

### How do you write a counter, then?

```myc
@ n << 1.0                seed 0        : real[] 1
@ k := floor(n)                         : real[] 1
```

`n` is a field whose derivative is `1`; `k` is its integer part. The compiler is free to lower this to a hardware counter, but the semantics are those of a field.

### How do you express a state machine?

As a phase field plus guarded routing. A finite automaton over a continuous phase:

```myc
@ phase  << omega                       seed 0
@ stance := 1 | sin(phase) > 0
@ swing  := 1 - stance
```

The compiler sees two complementary fields with no divergence. On a GPU that is a `select`; on Loihi it is two populations that inhibit each other.

### Does `<<` mean "loop forever"?

No. It means "this name denotes the solution curve of this ODE over the artifact's lifetime." If you want a bounded integration, say so with a window: `int[0s..2s]`. If you want a simulation that terminates, you do not want `<<`.

### What happens when the ODE is stiff?

`myc-flow-stiffness` estimates the spectral radius of `∂f/∂h` and `myc-flow-discretize` picks an integrator per field — RK4, exponential, Rosenbrock, or event-driven LIF. You can override with a solver tag:

```myc
@ h << ... seed 0 : real[post n] 1
$ solver tissue.h = rosenbrock
```

Choosing an explicit method on a stiff field is a warning, not an error, because sometimes you want to see it explode.

### Can I call into C or Python?

From the *host* side, yes: an `+io` field may be backed by a foreign function. From inside the dataflow, no — a foreign call has unknown effects and unknown cotangents, which would silently poison `grad` and `wcet`. `cut(...)` makes the boundary explicit if you truly need it.

### Does `grad` allocate a tape?

No. There is no tape object. `grad(x, L)` triggers *compilation* of the transposed subgraph, and the cotangent buffers are placed by `myc-mem-plan` like any other buffer. A gradient step is a kernel, not a recording.

### How do I debug?

Four tools, in increasing order of effort:

1. `mycel check file.myc` — static obligations, unit/axis diagnostics.
2. `mycel trace file.myc --field tissue.h --window 2s` — dump a field's trajectory.
3. `assert` obligations as executable tests: `assert max[t](norm(h)) < 10`.
4. The artifact certificates — geometry proofs, aliasing proofs, WCET proof — for post-hoc analysis.

### Why is torque a cotangent?

Because torque is defined by its pairing with angular velocity: power `= ⟨τ, ω⟩`. That pairing is a covector acting on a vector. Treating torque as a vector works until you change coordinates, at which point you need the transpose of the Jacobian, and if your language does not know the difference you will apply the wrong one. `Cot`/`Tan` makes that impossible.

### What is `~*` for, really?

For operations where the geometry matters: convolution on a translation-invariant metric, gauge-covariant transport in a neural field, or attention interpreted as a kernel on a manifold. If your metric is flat and your group is translation, `~*` lowers to standard convolution and you should just write that.

### Is this just a DSL for neural nets?

No, but neural nets are a strict subset. The example biped uses a liquid neural tissue, a PD controller, a contact solver, a CPG, an optimiser, and a genetic algorithm — all in the same topology, all compiled together. That is the point: the boundary between "controller", "model", and "learning algorithm" is an implementation detail, not a language boundary.

### Why "MYCEL"?

*Mycelium* is the hyphal lattice of a fungus: a network with no centre that routes toward demand, remembers past scarcity in its geometry, and rewires itself continuously while remaining one organism. That is a fair description of a compiled artifact.

### Is there a roadmap?

Yes — [08 — Roadmap](08-roadmap.md).

---

Next: [08 — Roadmap](08-roadmap.md)
