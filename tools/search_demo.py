"""Fast search of the phase pathway, for the live viewer."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mycel.walker.controller import PARAM_SLICES, Walker, initial_theta  # noqa: E402

SRC = (ROOT / "examples" / "05-walker-learn.myc").read_text(encoding="utf-8")
OUT = ROOT / "runs" / "demo"
OUT.mkdir(parents=True, exist_ok=True)

offsets, cursor = {}, 0
for name, shape in PARAM_SLICES:
    size = int(np.prod(shape))
    offsets[name] = (cursor, size)
    cursor += size
G0, GS = offsets["G_ph"]
B0, BS = offsets["b_out"]


def score(dist, fell, duration):
    alive = np.where(np.isinf(fell), duration, fell)
    return dist - 2.0 * (duration - alive)


rng = np.random.default_rng(5)
duration = 3.0
best_phase = np.zeros(GS)
best_bias = np.zeros(BS)
best = -1e9
sigma = 1.0

for r in range(6):
    V = 160
    phases = best_phase + sigma * rng.standard_normal((V, GS))
    biases = best_bias + 0.3 * sigma * rng.standard_normal((V, BS))
    batch = np.tile(initial_theta(rng), (V, 1))
    batch[:, G0 : G0 + GS] = phases
    batch[:, B0 : B0 + BS] = biases
    ep = Walker(SRC, variants=V, seed=11).run(batch, duration=duration)
    sc = score(ep.distance, ep.fallen_at, duration)
    i = int(np.argmax(sc))
    if sc[i] > best:
        best = float(sc[i])
        best_phase, best_bias = phases[i].copy(), biases[i].copy()
    print(f"round {r}: best {ep.distance[i]:6.2f} m  sigma {sigma:.3f}")
    sigma *= 0.75

theta = np.tile(initial_theta(np.random.default_rng(3)), (1, 1))[0]
theta[G0 : G0 + GS] = best_phase
theta[B0 : B0 + BS] = best_bias
np.savez_compressed(OUT / "theta.npz", theta=theta)

check = Walker(SRC, variants=1, seed=11).run(theta.reshape(1, -1), duration=8.0)
print(f"8 s check: {check.distance[0]:.2f} m, fell at {check.fallen_at[0]:.2f}")
print("G_ph =")
print(np.round(best_phase.reshape(8, 2), 2))
