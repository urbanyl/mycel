# 04 — Hardware Backends

MYCEL targets five families: NVIDIA/AMD GPUs, Apple Silicon, NPUs, neuromorphic chips, and CPUs. The same `<<` lowers to radically different execution models, which is the point.

---

## 1. GPU — CUDA and ROCm

**Path:** `linalg` → `vector` → `gpu` → `nvvm` / `rocdl`

The `myc.sched-partition` pass co-locates fields that share a rate class so their whole tick becomes one fused kernel. Because the topology is explicit, the compiler knows that `W_hh .*[pre] h` feeds `tanh` feeds `/tau` feeds the Euler/RK4 update, and can fuse all of it without a materialisation.

### Residency

A `<<` field's state is *persistent across ticks*. The scheduler keeps it in shared memory (or in registers for small `n`) so a tick costs no global round-trip:

```
smem[  h  |  k1  |  k2  |  k3  |  k4  ]      // RK4 scratch, 5 × n × 4 B
```

For the example tissue (`n = 256`), the RK4 scratch is 5 KB — trivial. For a 4096-wide transformer layer, the pass instead tiles the sequence axis and keeps `T`-slices resident.

### Precision

A precision-propagation pass respects unit obligations. A result typed `Cot[joint 12] N*m` may not be demoted below the tolerance implied by any `assert` mentioning it. `--precision=bfloat16` is a *request*; the pass reports exactly which fields stayed at `f32` and why.

### Determinism

`?` draws are counter-based, so multi-block reductions produce bit-identical results regardless of block count. `atomicAdd` order-dependence is eliminated by a deterministic tree reduction at the block level.

---

## 2. Apple Silicon

**Path:** `gpu` → MSL, plus ANE for eligible subgraphs

Two properties make Apple an unusually good MYCEL target:

1. **Unified memory.** `myc.mem.persist` weights are genuinely zero-copy: the CPU's encoder, the GPU's kernels, and the ANE all address the same physical pages. The "weights are memory-mapped" story is literal, not aspirational.
2. **Sustained low-power ticks.** A 1 kHz control loop on an M-series chip costs a few hundred milliwatts, which makes always-on embodied agents viable on battery.

`myc.geom`'s layout descriptors map onto MSL's `simdgroup_matrix` and threadgroup memory without an intermediate tiling pass.

---

## 3. NPU — TOSA and vendor toolchains

**Path:** `myc.*` → `tosa` → vendor compiler (Qualcomm, MediaTek, etc.)

Eligible subgraphs are those the pass can prove are affine-plus-activation: contractions, elementwise chains, `softmax`, `norm_lay`, `gelu`. Manifold products (`~*`) lower to convolution when the metric is translation-invariant, and are rejected otherwise.

The pass emits a *fallback list*: any op it cannot place on the NPU returns to the GPU/CPU partition, and the split is chosen by the same min-cut as any other device boundary.

---

## 4. Neuromorphic — Loihi 2 and Akida

This is where MYCEL's model stops being a convenience and starts being a necessity.

**Path:** `myc.neuro` → Lava (Loihi 2) or MetaTF (Akida)

| MYCEL | Loihi 2 | Akida |
|---|---|---|
| `<<` liquid field | compartment group with per-synapse `τ` | event layer with on-chip learning |
| `~>` synapse | AER route, `core[3].axon[17]` | event router |
| `surro(...)` | training profile only | on-chip plasticity rule |
| `\|\|` hydration | async handshake between cores | mesh packet |
| `<<<` evolution | host-side, then re-flash | host-side, then re-flash |

### No clock

There is no global timebase. Each compartment integrates its own membrane potential and emits when it crosses threshold. `myc.sched-domains` therefore does not produce a schedule — it produces **rate envelopes** and a set of handshake points.

The `wcet` obligation becomes a **latency envelope** obligation:

```myc
assert envelope(spike_path) <= 2ms @ p99
```

### Sparsity

`myc-neuro-eventify` prunes dense connectivity to a fan-in budget with a sparsity target, and reports the energy model:

```
core[0..3]: 12,288 synapses, 3.1% active per tick, est. 41 µJ
```

A dense GPU implementation of the same tissue would be ~200× more expensive per inference. That is the entire commercial argument for the backend.

### What does not port

Anything whose semantics need a global barrier every tick. `int[a..b]` over long windows and batched `argmax` across the whole population stay on the host. The compiler tells you which, and why.

---

## 5. CPU — LLVM and emitc

**Path:** `llvm` or `emitc`

Used for: real-time control on a microcontroller, certification-friendly builds, and the reference interpreter of last resort.

`emitc` is offered because some certification regimes (DO-178C, ISO 26262) require inspectable C with no unscheduled allocation. MYCEL's static memory plan makes that emit path unusually clean: there is no allocator to inspect because there is no allocation.

---

## 6. Heterogeneous partitioning

A single artifact may span all five. The partitioner is a hypergraph min-cut:

```
minimise    Σ_e  latency(e) · traffic(e)
subject to  energy(v) ≤ E_dev
            wcet(tick) ≤ dt_phys
```

Edge costs are learned per-device from a cost model shipped with the toolchain, and can be overridden with a JSON profile.

Typical split for the biped example:

| Device | Fields |
|---|---|
| MCU (CPU) | contact solver, safety clamps, `!>` at 2 kHz |
| GPU | liquid tissue, gradient updates at 1 kHz / 100 Hz |
| NPU | vision encoder (if present) |
| Neuromorphic | the tissue, if `--neuro` and inference-only |
| Host | `<<<` evolution at 0.25 Hz |

The `||` barriers appear exactly at the rate-class boundaries, and the artifact metadata names them so a timing analysis can start from a known list.

---

## 7. Cost model

Every backend ships a cost model consumed by `myc-sched-partition`:

```json
{
  "device": "loihi2",
  "energy_pj": { "synapse_event": 23, "neuron_update": 41, "spike_route": 12 },
  "latency_us": { "handshake": 3.5, "packet": 0.9, "fanin_1024": 7.2 }
}
```

Numbers are estimates, refreshed per toolchain release, and explicitly marked as such. The compiler prints which decisions depended on them, so you can re-run with `--cost-model=custom.json` and see what changes.

---

Next: [05 — Memory Model](05-memory-model.md)
