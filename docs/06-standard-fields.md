# 06 — Standard Fields and Combinators

The standard library is deliberately small. Anything that can be written as a field topology is *not* a builtin; builtins exist only where the compiler needs to know the mathematics to verify, fuse, or lower it.

Notation: `T` is a tensor type, `a` an axis name, `τ` a time window.

---

## 1. Algebraic combinators

| Form | Type rule | Notes |
|---|---|---|
| `A * B` | `axes(A) ≡ axes(B)` | elementwise; no broadcasting |
| `A .*[a] B` | `a` shared, other axes align | contraction |
| `A ** B` | `axes(A) ∩ axes(B) = ∅` | outer product |
| `A ~* B` | metric `g` supplied | manifold/geometric product |
| `A + B`, `A - B` | `axes(A) ≡ axes(B)` | elementwise |
| `A / B`, `A ^ r` | scalars broadcast only if axis-free | |
| `cat[a](A, …)` | other axes align | introduces or extends `a` |
| `A >>[a] f` | `a ∈ axes(A)` | free reindex |
| `pair(u, v)` | `u : Tan`, `v : Cot` (or vice versa) | yields `real[]` energy |
| `cut(A)` | any `=>` value | severs the cotangent edge |

---

## 2. Reductions

| Form | Result |
|---|---|
| `argmax[a](A)` | `axes(A) − {a}`, selects the rank-1 index |
| `max[a](A)`, `min[a](A)` | `axes(A) − {a}` |
| `sum[a](A)`, `mean[a](A)` | `axes(A) − {a}` |
| `var[a](A)`, `std[a](A)` | `axes(A) − {a}` |
| `norm[a](A)` | `axes(A) − {a}`, Euclidean |
| `int[τ](A)` | integrates over a time window |
| `argmax[a](A)` after `softmax` | yields a *distribution*, not a branch |

---

## 3. Manifold primitives

### `S2` / `S3`

| Form | Guarantee |
|---|---|
| `normalize(x)` | unit norm, asserted |
| `slerp(a, b, t)` | geodesic on the sphere |
| `log(x)`, `exp(v)` | tangent-space round-trip |
| `quat_mul(a, b)` | double-cover aware |

### `SO3` / `SE3`

| Form | Guarantee |
|---|---|
| `compose(A, B)` | group law |
| `inverse(A)` | `compose(A, inverse(A)) == identity`, discharged statically |
| `adjoint(A)` | maps `Tan` twists to `Tan` twists |
| `tolog(A)` | `SO3 → real[coord 3] rad` |

### `Tan` / `Cot`

| Form | Guarantee |
|---|---|
| `pair(u, v)` | only legal bundle crossing |
| `metric_raise(v)` | `Cot → Tan` with explicit metric |
| `metric_lower(u)` | `Tan → Cot` with explicit metric |
| `pullback(J, f)` | `Cot[X] → Cot[Q]` |
| `pushforward(J, v)` | `Tan[Q] → Tan[X]` |

### Simplex

| Form | Guarantee |
|---|---|
| `softmax[a](x)` | `sum[a] == 1`, all entries `>= 0` |
| `logsoftmax[a](x)` | log-domain, numerically stable |
| `entropy[a](p)` | `real[]` nats |

### Layernorm and friends

| Form | Notes |
|---|---|
| `norm_lay[a](x)` | two-pass fused; `a` must be the normalised axis |
| `norm_rms[a](x)` | one-pass |
| `gelu(x)`, `silu(x)` | `~~` adjoint supplied by the library |
| `erf(x)` | used by `gelu` |

---

## 4. Stochastic combinators

| Form | Notes |
|---|---|
| `?N(mu, sigma)` | counter-based normal |
| `?U(a, b)` | uniform |
| `?Categorical(p, a)` | draw over axis `a` |
| `?Dropout(p)` | multiplicative gate, mask recorded for the adjoint |

All draws are keyed by `hash(tick, field_id, lane)` and are therefore reproducible across devices, thread counts, and schedules. A recorded mask makes dropout's adjoint exact rather than statistical.

---

## 5. Neural primitives

| Form | Model |
|---|---|
| `lif(u, tau_m, vth)` | leaky integrate-and-fire, event output |
| `ltc(x, W, tau)` | liquid time-constant |
| `syn(w, pre, post)` | synaptic current |
| `stdp(pre, post, a_plus, a_minus)` | spike-timing plasticity |
| `surro(s, d)` | attach pseudo-derivative `d` to event `s` |

`surro` is mandatory for differentiating through spikes. There is no implicit straight-through estimator; you must name the derivative you are pretending is true, because the choice of surrogate *is* the modelling decision.

---

## 6. Evolutionary combinators

| Form | Notes |
|---|---|
| `genome(field, …)` | packages structural genes |
| `mutate(g, sigma)` | weight perturbation |
| `sprout(g, criterion)` | adds a synapse where `criterion` is high |
| `prune(g, threshold)` | removes where `‖w‖ < threshold` |
| `crossover(g1, g2)` | compatible-gene exchange |
| `speciate(g, delta)` | compatibility distance clustering |
| `clone(g, axis n)` | replicates onto a new population axis |

Population operations never spawn processes. `clone(G, variant 64)` adds an axis; every downstream op becomes a batched tensor operation; `argmax[LP](P)` selects.

---

## 7. Physics and body

| Form | Notes |
|---|---|
| `Rigid(mass, inertia, …)` | body with `Tan`/`Cot`-aware state |
| `Revolute(parent, axis, range)` | 1-DoF joint |
| `Chain(links…)` | serial chain, composed Jacobians |
| `Contact(port, mu, radius)` | Coulomb contact; algebraic cycle by construction |
| `mirror_x` | reflection; flips chirality, preserves units |

`Contact` is the canonical algebraic cycle: normal force depends on penetration depth, penetration depth depends on the integrator's predicted position, which depends on normal force. The compiler turns this into `myc.flow.solve` with an LCP or semi-implicit relaxation — it does not ask you to hand-write a substep loop.

---

## 8. Composing a topology

There is no `import`. A field is available if it is in scope of the ledger; an assembly is instantiated by applying it:

```myc
@ tissue  := LiquidTissue(256, 44, 12)
@ cortex  := LiquidTissue(512, 12, 12)
@ x2      := cat[in](tissue.h, cortex.h)     # both tissues see the same inputs
```

Because assemblies are open hypergraphs, `cortex` resolves `x` against the *same* ledger as `tissue`. That is how composite systems are built without a module system and without argument plumbing.

---

Next: [07 — FAQ](07-faq.md)
