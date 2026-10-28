# 00 — Philosophy

> *No neuron is in charge. No instruction pointer decides. And yet the lattice routes what is needed, where it is needed, when it is needed — and rewires itself while doing so.*

## The problem with programs that "step"

Every mainstream language inherits a model of computation that descends from the von Neumann machine: a program counter walks a sequence of instructions, and state changes only when the counter moves. That model is excellent for transforming files and answering requests. It is a poor fit for anything that is *always running*, that *is coupled to a physical body*, and whose behaviour is *the joint solution of many simultaneous constraints*.

Robotics, spiking neural networks, continuous control, and differentiable simulations all live in the same regime:

- **They never terminate.** There is no "end of the program" — there is only the next tick.
- **They are simultaneous.** Forty sensors, twelve actuators, and a quarter-million synapses all describe one state at one instant.
- **They are recursive through the world.** Actuation changes the body, the body changes the sensors, the sensors change the controller. The loop is not a programming construct; it is physics.
- **They are learned.** The parameters, and eventually the *topology* of the computation, are themselves state variables with their own dynamics.

Writing such systems in an imperative language means encoding a continuous field as a hand-timed loop, a joint solution as a sequence of mutations, and a physical feedback loop as a `while (true)` with sleeps. Each of those translations is a place where the semantics and the code drift apart.

MYCEL refuses the translation.

---

## Five axioms

### A1 — A program is a world, not a recipe

A `.myc` file is a **ledger**: a set of declarations about fields, each terminated by a newline, in no particular order. Reordering the lines changes nothing. Deleting a line removes a field and every edge attached to it, and nothing else. There is no "start", no "end", no entry point.

```myc
@ h << ( -h + tanh( W_hh .*[pre] h + W_xh .*[in] x + b ) ) / tau   seed 0
```

This does not *compute* `h`. It *asserts* that `h` is the solution curve of that differential equation. What computes it — an RK4 integrator on a GPU, a semi-implicit solver on a CPU, a population of leaky compartments on a neuromorphic die — is a lowering decision, not a semantic one.

### A2 — Time is a dimension, not a control construct

There is no `for` and no `while`. There are three ways to express repetition, and each is a *property of the data*, not of control flow:

| Repetition over | Construct |
|---|---|
| continuous time | `<<` — the solution operator of an ODE |
| a measure (window, batch, population) | `int[a..b]`, or an extra **named axis** |
| structural variation over slow time | `<<<` — a Markov variation kernel on the operator graph |

A population of 64 candidate topologies is not a process pool. It is an axis named `variant` with extent 64, and evaluating all of them is one tensor operation.

### A3 — Branching is routing

There is no `if`. A signal that should exist only under a condition is *guarded*:

```myc
@ fall := 1 | norm(lam) < eps          # defined only where the predicate holds
@ S    := ( Q .*[proj] K ) / sqrt(64) | time >= key    # causal mask, not a branch
```

A signal that should choose among alternatives is *selected* by rank:

```myc
@ G <<< argmax[LP](P) ++ archive_best
```

A signal that should be modulated is *gated* multiplicatively:

```myc
@ tau := clamp(tauNN + tauPD, tau_max) * balance(q, lam)
```

Guarded routing, rank selection, and gating are all **data-dependent but control-free**: they have no divergent control flow, which means they vectorize, they differentiate, and they compile to the same kernel regardless of which lane is active.

### A4 — Every differentiable edge is auto-adjoint

There is no `backward()`. There is no tape object, no `requires_grad`, no optimizer. An edge declared with `=>` carries a **primal slot and a cotangent slot by construction**. Asking for `grad(θ, L)` is a *query over the cotangent bundle*: the compiler walks every `=>` edge that reaches `θ` from `L`, emits the transposed program, and allocates cotangent buffers for exactly those edges.

If no `grad(...)` mentions a subgraph, that subgraph's cotangent buffers are never allocated. Gradient memory is not a fixed tax; it is a consequence of what you asked to differentiate.

### A5 — Nothing is allocated at runtime

There is no garbage collector and no manual `malloc`. After a one-tick warmup, the compiled artifact calls the allocator **zero times**. This follows from four static facts:

1. Every shape is resolved or bounded at compile time.
2. Every live range is known, so buffers can be coloured into fixed arenas.
3. Every `<<` history is a ring of statically known depth.
4. Every event queue has a proven maximum in-flight occupancy.

A real-time controller must be able to state its worst-case tick time. MYCEL makes that a compiler obligation (`assert wcet(tick) <= dt_phys`) rather than a profiling exercise.

---

## What this buys you

| Property | Consequence |
|---|---|
| **Order-free surface** | Merge conflicts are rare; diffs are additive; tooling is trivial. |
| **Named axes** | Transposes are layout, not semantics. Silent broadcasting is impossible. |
| **`Q^7` unit algebra** | `N*m` and `kg*m^2/s^2` unify; `N` and `N*m` do not. Dimension bugs die at compile time. |
| **`Tan` / `Cot` discipline** | Torque is a covector, velocity a vector. The only well-typed product is mechanical power. |
| **Structural autodiff** | No tape, no library, no framework — the language *is* the AD system. |
| **Multi-rate by construction** | Physics at 2 kHz, gradients at 100 Hz, evolution at 0.25 Hz — one artifact, four clock domains. |
| **Backend-neutral ODEs** | The same `<<` lowers to a fused CUDA kernel, a Metal shader, or a Loihi compartment group. |

---

## What this costs you

Intellectual honesty requires the ledger:

- **Debugging is different.** You cannot step through a field. You inspect trajectories, cotangents, and invariant violations. The `assert` surface is therefore much richer than in a procedural language, and it is load-bearing.
- **Some algorithms are awkward.** Anything that genuinely needs a mutable heap with pointer-chasing graphs resists the topology model. MYCEL is not a general-purpose language and does not pretend to be.
- **The compiler is the product.** A beautiful surface with a weak backend is a toy. Most of the engineering is in `myc.geom`, `myc.flow`, and `myc.mem-plan`.

---

## Naming

*Mycelium* is the subterranean hyphal lattice of a fungus: a network with no centre that routes nutrients toward demand, remembers past famines in its own branching geometry, and rewires itself continuously while remaining a single organism. That is a fairly precise description of a compiled MYCEL artifact.

It also reads as *my cell*, which is either a happy accident or the entire point.

Next: [01 — Language Specification](01-language-spec.md)
