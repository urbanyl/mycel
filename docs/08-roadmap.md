# 08 — Roadmap

MYCEL is a specification-first project. The order below is chosen so that every milestone produces something usable, not so that every layer is finished before the next begins.

---

## ✅ v0.9a — current

- [x] Axioms and semantics
- [x] ASCII surface syntax (94 code points)
- [x] Operator algebra: `.*`, `**`, `~*`, `++`, `|`, `!>`, `~>`, `=>`, `<<`, `<<<`
- [x] SGTS: named axes, `Q^7` units, `Tan`/`Cot` discipline
- [x] MLIR dialect design (`myc.topos`, `myc.geom`, `myc.flow`, `myc.diff`, `myc.neuro`, `myc.evo`, `myc.mem`, `myc.sched`)
- [x] Zero-allocation memory model
- [x] Reference front end: lexer, parser, unit algebra, axis checker, CLI
- [x] Worked examples: biped walker, transformer block, thermostat

---

## v0.10 — executable core

- [ ] `mycel run` interpreter for scalar and small-tensor ledgers
- [ ] Full axis algebra checker with Presburger extents
- [ ] `int[a..b]` and `<<` in the interpreter (RK4 + event-driven LIF)
- [ ] `grad` in the interpreter (reverse-mode over the RCT)
- [ ] Golden tests from `examples/`

**Exit criterion:** `examples/03-thermostat.myc` and `examples/04-hello-field.myc` run and produce stable trajectories.

---

## v0.11 — enough MLIR to be useful

- [ ] Out-of-tree MLIR project skeleton with all eight dialects registered
- [ ] `myc-geom-verify` with `mlir-presburger` integration
- [ ] `myc.flow` → `linalg` on tensors for the transformer example
- [ ] `linalg` → `llvm` CPU backend
- [ ] `mycel build examples/02-transformer-block.myc --target=cpu`

**Exit criterion:** the transformer block trains on a tiny corpus, on CPU, with no PyTorch in the build graph.

---

## v0.12 — accelerators and autodiff

- [ ] `myc-diff-lower` with checkpointing knapsack
- [ ] `gpu` → `nvvm` and `rocdl` backends
- [ ] `gpu` → `metal-msl`
- [ ] `myc-mem-plan` with aliasing certificates
- [ ] `mycel build --target=cuda|metal`

**Exit criterion:** the same `.myc` file produces a CUDA kernel and a Metal shader, and both agree bit-for-bit on a recorded seed.

---

## v0.13 — embodiment

- [ ] `myc.flow.solve` for algebraic cycles (contact LCP)
- [ ] `myc-wcet` static timing analysis
- [ ] MuJoCo-shaped bridge: export `!>`/`~>` ports to a simulator
- [ ] `examples/01-biped-walker.myc` runs a closed loop in simulation

**Exit criterion:** the biped walks forward for 60 seconds without falling, trained online in the loop.

---

## v0.14 — neuro-evolution

- [ ] `myc.evo` population lowering to a `variant` axis
- [ ] Shadow-world rollouts through the differentiable fork
- [ ] `<<<` host-side structural mutation with an archive
- [ ] Speciation with a compatibility distance

**Exit criterion:** evolution improves fitness over a static-topology baseline on the biped.

---

## v0.15 — neuromorphic

- [ ] `myc-neuro-eventify` (LTC → compartments, density pruning)
- [ ] Lava backend for Loihi 2 (via `myc.neuro` → NxSDK)
- [ ] MetaTF backend for Akida
- [ ] Latency-envelope assertions replacing WCET on clockless targets

**Exit criterion:** the biped's tissue runs on a Loihi 2 partition at a measured energy per inference, with no global clock.

---

## v1.0 — what "done" means

- [ ] Front end frozen at `0.10` semantics
- [ ] All five backend families shipping
- [ ] Certificates (geometry, aliasing, timing) reproducible from source alone
- [ ] A conformance suite that any implementation can run
- [ ] At least one third-party implementation of the front end

---

## Non-goals

Written down so they are not quietly smuggled in later:

- **Not a general-purpose language.** No filesystem API, no sockets, no string processing beyond identifiers.
- **Not a Python replacement.** If you want a scripting language with tensor sugar, use JAX.
- **Not a hardware description language.** `!>` writes to an actuator; it does not synthesise logic.
- **Not a theorem prover.** SGTS discharges dimension and unit obligations. It will not prove your controller is stable.
- **Not zero-cost on every axis.** The static memory model costs compile time; a 40-minute build for a real-time artifact is expected, not a bug.

---

## How to help

The highest-leverage contributions right now, in order:

1. **Adversarial examples.** Write a `.myc` file that *should* compile and see if the checker rejects it. File the difference.
2. **Unit-algebra edge cases.** Rational exponents have surprising corners (`sqrt(N*m)` is not `N^0.5 * m^0.5` in every derived-unit table).
3. **Solver selection heuristics.** `myc-flow-stiffness` is the pass most likely to be wrong in an interesting way.
4. **Docs.** Every claim in this repository should be falsifiable. If one is not, say so.

See [CONTRIBUTING.md](../CONTRIBUTING.md).
