"""Fast demo: search the 24 phase-pathway parameters, then render the walk.

This is a smoke-sized version of `mycel train` used to produce a visual
quickly. The full search optimises all 1152 parameters of the tissue.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mycel.walker.controller import N_PARAMS, PARAM_SLICES, Walker, initial_theta  # noqa: E402
from mycel.walker.render import contact_sheet, learning_curve, render_gif  # noqa: E402

SRC = (ROOT / "examples" / "05-walker-learn.myc").read_text(encoding="utf-8")
OUT = ROOT / "runs" / "demo"
OUT.mkdir(parents=True, exist_ok=True)

offsets = {}
cursor = 0
for name, shape in PARAM_SLICES:
    size = int(np.prod(shape))
    offsets[name] = (cursor, size)
    cursor += size

G0, GS = offsets["G_ph"]
B0, BS = offsets["b_out"]


def make_theta(rng: np.random.Generator, phase: np.ndarray, bias: np.ndarray) -> np.ndarray:
    theta = initial_theta(rng)
    theta[G0 : G0 + GS] = phase
    theta[B0 : B0 + BS] = bias
    return theta


def score(dist: np.ndarray, fell: np.ndarray, duration: float) -> np.ndarray:
    alive = np.where(np.isinf(fell), duration, fell)
    return dist - 2.0 * (duration - alive)


rng = np.random.default_rng(5)
duration = 4.0

best_phase = np.zeros(GS)
best_bias = np.zeros(BS)
best = -1e9

sigma = 0.9
for round_index in range(7):
    V = 192
    phases = best_phase + sigma * rng.standard_normal((V, GS))
    biases = best_bias + 0.3 * sigma * rng.standard_normal((V, BS))
    base = initial_theta(rng)
    batch = np.tile(base, (V, 1))
    batch[:, G0 : G0 + GS] = phases
    batch[:, B0 : B0 + BS] = biases

    walker = Walker(SRC, variants=V, seed=11)
    episode = walker.run(batch, duration=duration)
    sc = score(episode.distance, episode.fallen_at, duration)
    idx = int(np.argmax(sc))
    if sc[idx] > best:
        best = float(sc[idx])
        best_phase = phases[idx].copy()
        best_bias = biases[idx].copy()
    print(
        f"round {round_index}  best {episode.distance[idx]:6.2f} m  "
        f"fell {episode.fallen_at[idx]:5.2f}  sigma {sigma:.2f}"
    )
    sigma *= 0.72

print("phase weights:")
print(np.round(best_phase.reshape(8, 2), 2))

theta = make_theta(np.random.default_rng(3), best_phase, best_bias).reshape(1, -1)
walker = Walker(SRC, variants=1, seed=11)
episode = walker.run(theta, duration=8.0, record_every=4)
print(f"final demo: {episode.distance[0]:.2f} m, fell at {episode.fallen_at[0]:.2f}")

trace = episode.trace
np.savez_compressed(OUT / "trace.npz", pos=trace["pos"], cmd=trace["cmd"], dt=trace["dt"])
np.savez_compressed(OUT / "theta.npz", theta=theta[0])

render_gif(
    trace,
    OUT / "walk.gif",
    title="MYCEL / biped, neural controller",
    subtitle="liquid tissue 24 cells - phase-search over 24 parameters",
)
contact_sheet(
    {"pos": trace["pos"][:1], "dt": trace["dt"]},
    trace,
    OUT / "contact_sheet.png",
    labels=("start", "walking"),
)

history = {"generation": list(range(7)), "best_distance": [0.0] * 7,
           "mean_distance": [0.0] * 7, "best_fitness": [0.0] * 7, "fell_rate": [0.0] * 7}
print("wrote", OUT)
