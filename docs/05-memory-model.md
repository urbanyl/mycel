# 05 — Zero-Allocation Memory Model

## The claim

> **After a one-tick warmup, a MYCEL artifact calls the allocator exactly zero times.**

This is not an aspiration or a best-effort arena. It is a consequence of four static facts, each enforced by a pass, and each shipped as a certificate in the artifact.

| Fact | Enforced by |
|---|---|
| Every shape is resolved or bounded | `myc-geom-verify` |
| Every live range is known | `myc-mem-plan` |
| Every `<<` history is a fixed-depth ring | `myc-flow-discretize` |
| Every event queue has a proven bound | `myc-wcet` |

---

## 1. Static shapes

`myc-geom-verify` erases every symbolic extent, either by solving for it or by bounding it at deployment time. Anything that survives is a constant in the generated code.

The two escape hatches are explicit:

```myc
$ world dt=1ms
$ deploy batch B=8
```

`B` may stay symbolic through compilation but must be bound before the artifact ships. `myc-wcet` runs against the bound value; you cannot ship an artifact whose tick time is unknown.

---

## 2. Interval colouring

`myc-mem-plan` consumes the SSA liveness of every buffer and produces an interval graph. Colour that graph and each colour becomes a slot in an arena.

```
buffer        live range          colour
──────────────────────────────────────────
%h             [0, 14]              0
%k1            [2, 6]               1
%k2            [4, 8]               1     ← k1 and k2 never overlap
%k3            [6, 10]              1
%k4            [8, 12]              1
%x             [0, 40]              2
%attn_scores   [12, 24]             3
```

Two buffers share bytes only when their live ranges are disjoint **and** no write-after-read hazard exists. Aliasing is proven by the same analysis that proves liveness; the certificate records the proof obligation for each alias.

### 2.1 Allocation is a pointer bump

```llvm
%arena = alloca [4194304 x i8], align 64
; named offsets, no dynamic size, no malloc
```

An arena reset per tick is `store i8* %base, %cursor` — one instruction, O(1), deterministic, zero fragmentation.

---

## 3. Per-phase arenas

Not one arena but several, each reset by its own domain:

| Arena | Owner | Reset |
|---|---|---|
| `tick` | physics domain | every control tick |
| `grad` | gradient domain | every gradient step |
| `evo` | evolution domain | every evolutionary generation |
| `persist` | weights | never (mmap) |
| `event` | neuromorphic domain | per event batch |

Separation means a slow gradient step never holds a lease on scratch the fast control loop needs. It also means a fault in one domain cannot corrupt another's memory: the reset is the isolation.

---

## 4. Gradient memory

Reverse-mode liveness is mirror-symmetric. `myc-diff-lower` lays cotangent slots out **backwards** in the `grad` arena and frees them as the sweep retreats:

```
forward:   A ─► B ─► C ─► D ─► L
liveness:  A: [0,16]  B: [4,16]  C: [8,16]  D: [12,16]
reverse:   dA ◄── dB ◄── dC ◄── dD
slots:     dL and dD alias the tail of the C region, etc.
```

### 4.1 Checkpointing as a knapsack

For deep stacks of `=>` fields, holding every intermediate is impossible. `myc-diff-checkpoint` solves:

```
minimise   peak_memory
subject to Σ recompute_cost(c) ≤ budget
```

over partition points, then marks the selected boundaries with `myc.diff.checkpoint`. This is how a 70-billion-parameter `<<` field fits a fixed device budget *by construction* rather than by luck.

---

## 5. State history

An integrator needs history. RK4 needs the four stage derivatives; a multistep method needs the last `k` states. Either way the depth is known at compile time:

```
ring_depth = ceil(horizon / dt) + 1
```

for windowed integrals, and `k + 1` for multistep. The ring is a fixed `mem.ring` allocation; the write index is a single integer modulo a power-of-two depth. There is no growth, no resize, and no unbounded history.

---

## 6. Persistent weights

```myc
@ tissue.W_hh := ?N(0, n^-0.5) : real[post n, pre n] 1
```

`mem.persist` maps the weight bundle into the process. On unified-memory systems, host and accelerator alias the same physical pages — the "zero-copy" claim is literal.

Optimiser updates are **in place** with a fused kernel:

```
tau = -eta * grad(W, L)
W += tau                       // one pass, no temporary weight copy
```

There is no `.data` swap, no shadow parameter set, no double-buffered weights. The consequence: an interrupted tick leaves weights in a well-defined state, and a memory-mapped weight file is always consistent.

---

## 7. Event queues

`~>` queues are lock-free single-producer/single-consumer rings. Capacities are proven, not guessed:

```
capacity(aer_route) = ceil(max_fanout × max_rate × envelope)
```

`max_fanout` comes from the pruned connectivity, `max_rate` from the neuron model's refractory period, and `envelope` from the latency assertion. If the product cannot be bounded, compilation fails with `MYC0702` rather than dropping spikes at runtime.

This is a hard requirement for neuromorphic deployment: a dropped spike is a silent behavioural change, and silent behavioural changes in a robot are unacceptable.

---

## 8. What "zero allocation" excludes

Honesty requires the boundary:

- **Model loading** allocates. Weights are mapped once at startup.
- **Dynamic graph surgery** — `<<<` with a genuinely unbounded genome — allocates on the evolution domain only, and never while the physics domain is live.
- **Debug instrumentation** may allocate.
- **Host-side dataset loading** (`+io` fields) allocates, but on the io domain and never inside a control tick.

Everything on the hot path is static. Everything outside it is quarantined to its own domain and arena.

---

## 9. Why this matters

Three concrete consequences, in order of importance:

1. **Worst-case execution time is a compile-time theorem.** `assert wcet(tick) <= dt_phys` is discharged by static analysis. A control loop that meets its deadline in simulation meets it on hardware, because there is no allocator latency tail.
2. **Debuggability.** Static addresses mean a core dump is a complete description of machine state. You can inspect field values post-mortem without symbolising heap graphs.
3. **Determinism.** No address-dependent behaviour, no hash-seed-dependent iteration, no allocator-dependent layout. Two runs produce identical bytes.

---

Next: [06 — Standard Fields](06-standard-fields.md)
