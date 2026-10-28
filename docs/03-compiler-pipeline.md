# 03 — Compiler Pipeline

```
.myc ledger
   │  parse            LL(1), newline-delimited
   ▼
   AST
   │  elaborate        instantiate open hypergraphs, desugar cat/../>> into axis ops
   ▼
   RCT                Reactive Core Topology — typed, effect-annotated hypergraph
   │  convert
   ▼
   myc.topos  ──►  myc.geom  ──►  myc.flow  ──►  myc.diff
                                     │
                                     ├──► myc.neuro   (spiking / liquid primitives)
                                     └──► myc.evo     (population axes)
   │
   ▼
   linalg on tensors
   │  fuse, tile, vectorize
   ▼
   one-shot-bufferize     static shapes only
   │
   ▼
   myc.mem-plan           liveness ─► interval colouring ─► arenas
   │
   ├──► gpu ──► nvvm | rocdl | spirv | metal-msl
   ├──► tosa ──► vendor NPU toolchain
   ├──► myc.neuro ──► lava (Loihi 2) | metatf (Akida)
   └──► llvm / emitc ──► CPU
```

---

## 1. Front end

### 1.1 Lexer

The token set is fixed and small enough that longest-match is a single switch. `<<<`, `<<`, `~>`, `=>`, `!>`, `:=`, `~~`, `||`, `..`, `.*`, `**`, `~*`, `++` are the only multi-character tokens that can collide with a prefix of another. A reference implementation lives in [`mycel/lexer.py`](../mycel/lexer.py).

### 1.2 Parser

The ledger grammar is LL(1): the role sigil determines the production, and every entry is line-terminated. There is no expression ambiguity to resolve with precedence climbing beyond the usual arithmetic ladder.

### 1.3 Elaborator

The elaborator does three things that matter semantically:

1. **Instantiates open hypergraphs.** `LiquidTissue(256, 44, 12)` resolves free names `x`, `W_xh`, `b`, `tau`, `nu`, `y` against the instantiating ledger. Resolution is static; every resolved edge is materialised in the RCT.
2. **Desugars sugar into explicit axis operations.** `cat[in](a, b, c)` becomes three `geom.concat` ops plus a `geom.assert_dim`. `A .*[a] B` becomes `geom.contract {axis = a}`.
3. **Propagates effect rows.** A field that transitively reads a `+io` port inherits `+io`. Purity is *not* inferred — it is computed and then checked against every `grad`.

The RCT is the last representation that carries names. Everything downstream is SSA.

---

## 2. Dialects

All dialects are defined in ODS (`*.td`) and registered as an out-of-tree MLIR project.

### `myc.topos` — the topology layer

```mlir
%h = myc.topos.field "tissue.h" {rate = #myc.rate<neural>} : !myc.tensor<...>
%f = myc.topos.fuse %a, %b : (!myc.tensor<...>, !myc.tensor<...>) -> !myc.tensor<...>
%p = myc.topos.project %x along "head" : ...
%c = myc.topos.instantiate "LiquidTissue"(n = 256 : i64)
%b = myc.topos.barrier %a, %b {domain_a = 1, domain_b = 2} : ...
```

### `myc.geom` — axes, units, manifolds

```mlir
%0 = myc.geom.contract %A, %B {axis = "pre"} : ...
%1 = myc.geom.reindex %A {perm = ["time","head"]} : ...
myc.geom.assert_unit %t : !myc.unit<(kg:1, m:2, s:-2)>
myc.geom.assert_dim %t : !myc.axis<"time", T>
```

The dialect also defines the type syntax:

```mlir
!myc.tensor<f32, [batch B, time T, head 12, proj 64], unit<(kg:0, m:0, s:0, A:0, K:0, mol:0, cd:0)>>
!myc.cotensor<f32, [joint 12], unit<(kg:1, m:2, s:-2)>>
!myc.event<i8, axon 16>
```

`!myc.tensor` and `!myc.cotensor` do not unify. There is no implicit cast; `myc.geom.metric_raise` / `metric_lower` are the only worms.

### `myc.flow` — continuous time

```mlir
%h = myc.flow.recur seed(%h0) solver(#myc.solver<rk4 dt=1ms>) {
^bb0(%h : !myc.tensor<...>, %x : !myc.tensor<...>):
  %d = ... : !myc.tensor<...>
  myc.flow.yield %d
}
%i = myc.flow.integrate %f over #myc.window<t-2s, t> : ...
%s = myc.flow.solve %cycle {method = "semi_implicit"} : ...
```

### `myc.diff` — cotangents

```mlir
%g = myc.diff.grad %L wrt %W : (!myc.tensor<f32, [], unit<>>, !myc.tensor<f32, [n,n]>) -> !myc.cotensor<...>
myc.diff.checkpoint %layer {policy = "sqrt"}
%x = myc.diff.cut %y : ...
%z = myc.diff.surrogate %spike, %dsigma : ...
```

### `myc.neuro` — spiking and liquid primitives

```mlir
%v = myc.neuro.lif %i {tau_m = 20ms, v_th = 1.0, v_reset = 0.0} : ...
%h = myc.neuro.ltc %x, %w {tau = 20ms} : ...
myc.neuro.aer_route %spikes to "core[3].axon[17]"
myc.neuro.stdp %pre, %post {a_plus = 0.01, a_minus = 0.012, tau = 20ms}
```

### `myc.evo` — populations and variation

```mlir
%P = myc.evo.population %G {axis = "variant", n = 64}
%Q = myc.evo.mutate %P {sigma = 0.02, mask = ...}
%R = myc.evo.speciate %Q {delta = 3.0}
%S = myc.evo.select %R by %fitness {mode = "elitist", k = 4}
```

### `myc.mem` — the static memory plan

```mlir
%a = myc.mem.arena "tick" {bytes = 4194304, align = 64}
%r = myc.mem.ring %h {depth = 5}          // RK4 history + 1
myc.mem.persist %W {path = "weights.bin"}
myc.mem.alias %x, %y                       // proven WAR-free by the planner
```

---

## 3. Pass pipeline

### 3.1 `myc-geom-verify`

The load-bearing pass. It discharges every `assert` and every implicit dimension obligation.

- **Axis obligations** are emitted as Presburger formulas and solved in linear integer arithmetic. `T = H*64` is a one-variable linear equation; `B = ceil(N/32)` is a div/mod constraint; both are decidable.
- **Unit obligations** normalise to `ℚ⁷` vectors and compare exactly.
- **Bundle obligations** check that `Tan`/`Cot` never unify and that every `pair` consumes one of each.
- **Nonlinear extents** (`d = sqrt(vocab)`) fall back to Z3 under `--smt`. Unresolved symbolic extents become bounded deployment asserts with a diagnostic that names the deployment flag which supplies them.

Diagnostics are formatted with a caret under the offending token and a suggested rewrite, because a type error in a tensor language is useless without one:

```
error[MYC0301]: axis mismatch
  --> walker.myc:41:33
   |
41 | @ tauNN := cat[head](y, clk) .*[head] W_m
   |                                 ^^^^^^ axis 'head' has extent 2 on the left, 12 on the right
   = note: W_m expects joint-space input; did you mean .*[phase]?
```

### 3.2 `myc-flow-normalize` and `myc-flow-stiffness`

Every `<<` is a field equation. The pass rewrites each into DAE normal form `ḣ = f(h, x, t)` and estimates the spectral radius of `∂f/∂h`.

This matters more than it sounds. A liquid time-constant network with state-dependent `τ(x)` spanning 1 ms to 200 ms is **stiff**: an explicit integrator would need a step 200× smaller than the natural tick to remain stable. The pass splits the Jacobian into stiff and non-stiff blocks and assigns different treatment to each.

### 3.3 `myc-flow-discretize`

Chooses, per field:

| Condition | Integrator |
|---|---|
| `ρ(J)·dt ≪ 1` | explicit RK4 |
| linear part dominant, `J` near-constant | exponential integrator (`expm`-preconditioned) |
| stiff, `J` available | semi-implicit / Rosenbrock |
| state-dependent threshold events | event-driven LIF (exact between events) |
| algebraic cycle | `flow.solve` with fixed-point or LCP |

The choice is recorded in the artifact metadata so a numerical failure is reproducible.

### 3.4 `myc-diff-lower`

Reverse-mode over the `=>` subgraph.

1. Build the cotangent graph by transposing each op.
2. Prune to the targets actually named by `grad(...)` calls. Unqueried subgraphs never receive cotangent buffers.
3. Insert `checkpoint` boundaries where the recompute/peak-memory knapsack says so.
4. Emit `diff.grad` ops in a schedule that matches the forward domain's rate class.

Because `~>` is a hard cut, a `grad` that reaches a `~>` edge stops there. That is not a limitation — it is the honest statement that contact impulses are not differentiable. Where you want a pseudo-gradient, you say so:

```myc
@ nu := surro(h, dsigmoid(h - vth))
```

### 3.5 `myc-neuro-eventify`

Lowers continuous liquid fields into event-driven equivalents:

- `myc.neuro.ltc` → compartment group with per-synapse time constants
- dense `W_hh` → sparse connectivity, pruned to a fan-in budget with a sparsity target
- `<<` recurrence → event-driven update, valid only between spike times
- `surro` → retained for training-dominated profiles, dropped for inference profiles

The pass can be skipped with `--no-eventify` when the target is a dense accelerator.

### 3.6 `myc-sched-partition`

Hypergraph min-cut with a latency-and-energy cost model:

```
minimise   Σ_e  latency(e) · traffic(e)
subject to Σ_{v ∈ dev} energy(v) ≤ E_dev
           wcet(tick) ≤ dt_phys
```

Every cut edge becomes a transport primitive: DMA ring, PCIe stream, or shared-memory double buffer. Edges whose endpoints share a rate class are preferred for co-location; edges that cross rate classes are candidates for hydration barriers.

### 3.7 `myc-mem-plan`

See [05 — Memory Model](05-memory-model.md). Produces a static `{arena, offset, bytes}` triple for every buffer, and a proof certificate that no two live ranges overlap illegally.

### 3.8 `myc-wcet`

Computes a static worst-case per tick from the schedule and the per-op cost model, then compares against `dt_phys`. Failure is a compile error, not a runtime surprise:

```
error[MYC0701]: worst-case tick exceeds the physics period
  wcet(tick) = 2.41 ms   dt_phys = 2.00 ms
  = note: 3 fields overflow their rate class; consider --allow-eventify or splitting tissue.h
```

---

## 4. Bufferisation and lowering

`one-shot-bufferize` runs **after** all shapes are static. There is no dynamic `memref` in a shipped artifact. Then:

- `linalg-fuse` merges producer/consumer elementwise chains.
- `linalg-tile` + `vectorize` produce target-appropriate kernels; tile sizes come from the layout descriptor chosen by `myc-mem-plan`.
- `gpu` / `nvvm` / `rocdl` / `spirv` / `metal` handle accelerators.
- `tosa` handles NPUs.
- `myc.neuro` lowers to Lava (Loihi 2) or MetaTF (Akida).
- `llvm` + `emitc` handle CPU, with `emitc` used when a target needs predictable C-level codegen for certification.

---

## 5. Artifact format

A `.mycbin` is a zip-like container:

```
manifest.json         dialects, versions, solver choices, rate classes
memory.map            arena layout, static addresses, alignment
weights.bin           mmap-able, aligned, optional compression
kernels/              per-device binaries
certificates/         geometry proofs, wcet proof, aliasing proof
```

The certificates are shipped because a real-time system must be auditable after the fact: they let you answer "why was this tick 2.4 ms" without a profiler and without the build machine.

---

Next: [04 — Hardware Backends](04-hardware-backends.md)
