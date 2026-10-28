"""Glue between the MYCEL ledger and the physical body.

The ledger decides *what to compute*; this module only feeds it sensor
tensors and applies the actuator sink. Every gait decision is taken by
the interpreted program -- nothing about walking is hard-coded here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from ..runtime.program import Program
from ..runtime.tensor import T
from .physics import Biped

N = 24
N_IN = 14
N_OUT = 8
PARAM_SLICES = (
    ("W", (N, N)),
    ("U", (N, N_IN)),
    ("b", (N,)),
    ("W_out", (N_OUT, N)),
    ("G_ph", (N_OUT, 4)),
    ("b_out", (N_OUT,)),
)
N_PARAMS = sum(int(np.prod(shape)) for _, shape in PARAM_SLICES)

SCALES = {"W": 0.18, "U": 0.18, "b": 0.0, "W_out": 0.25, "G_ph": 0.45, "b_out": 0.0}

AX = ("variant",)


GAIT = np.zeros((8, 4))
GAIT[0] = [0.0, -0.90, 0.0, 0.0]    # foot L forward offset
GAIT[1] = [-0.60, 0.0, 2.20, 0.0]   # foot L lift
GAIT[2] = [0.0, 0.90, 0.0, 0.0]     # foot R forward offset
GAIT[3] = [-0.60, 0.0, 0.0, 2.20]   # foot R lift
GAIT[5] = [0.30, -0.20, 0.0, 0.0]   # left push-off
GAIT[6] = [-0.30, 0.20, 0.0, 0.0]   # right push-off


def initial_theta(rng: np.random.Generator) -> np.ndarray:
    """A genome whose phase pathway already alternates the two legs.

    Everything else is small random noise: the search starts from an
    anti-phase rhythm, not from a walking policy.
    """
    blocks = []
    for name, shape in PARAM_SLICES:
        if name == "G_ph":
            block = GAIT.reshape(-1) + 0.10 * rng.standard_normal(int(np.prod(shape)))
        elif name == "W_out":
            block = 0.08 * rng.standard_normal(int(np.prod(shape)))
        else:
            block = SCALES[name] * rng.standard_normal(int(np.prod(shape)))
        blocks.append(block)
    return np.concatenate(blocks)


def unpack(theta: np.ndarray) -> Dict[str, T]:
    """Map a flat parameter vector (V, P) onto the ledger's field names."""
    V = theta.shape[0]
    out: Dict[str, T] = {}
    offset = 0
    for name, shape in PARAM_SLICES:
        size = int(np.prod(shape))
        block = theta[:, offset : offset + size].reshape((V,) + shape)
        offset += size
        if name == "W":
            out[name] = T(("variant", "post", "pre"), block)
        elif name == "U":
            out[name] = T(("variant", "post", "in"), block)
        elif name == "b":
            out[name] = T(("variant", "post"), block)
        elif name == "W_out":
            out[name] = T(("variant", "out", "post"), block)
        elif name == "G_ph":
            out[name] = T(("variant", "out", "phase"), block)
        else:
            out[name] = T(("variant", "out"), block)
    return out


def pack(theta: np.ndarray) -> np.ndarray:
    return theta


@dataclass
class Episode:
    distance: np.ndarray
    fitness: np.ndarray
    fallen_at: np.ndarray
    upright: np.ndarray
    reward: np.ndarray
    trace: Optional[Dict[str, np.ndarray]] = None


class Walker:
    """One ledger, one body, batched over ``variants``."""

    def __init__(self, source: str, variants: int, dt: float = 1 / 240, seed: int = 0) -> None:
        self.variants = variants
        self.dt = dt
        self.program = Program.from_source(source, seed=seed)
        self.body = Biped(variants=variants, dt=dt, seed=seed)
        self.t = 0.0

    # ------------------------------------------------------------------
    def bind(self, theta: np.ndarray) -> None:
        for name, value in unpack(theta).items():
            self.program.set(name, value)

    def _scalar(self, arr: np.ndarray) -> T:
        return T(AX, np.asarray(arr, dtype=np.float64).reshape(self.variants))

    def _feed(self) -> None:
        body = self.body
        p = self.program
        p.set_input("pitch", self._scalar(body.torso_pitch()))
        p.set_input("dpitch", self._scalar(body.torso_pitch_rate()))
        ang = body.joint_angles()
        rate = body.joint_rates()
        for i, name in enumerate(("hipL", "kneeL", "hipR", "kneeR")):
            p.set_input(name, self._scalar(ang[:, i]))
            p.set_input("d" + name, self._scalar(rate[:, i]))
        contact = body.contact.astype(np.float64)
        p.set_input("contactL", self._scalar(contact[:, 0]))
        p.set_input("contactR", self._scalar(contact[:, 1]))
        p.set_input("vx", self._scalar(body.vel[:, 1, 0]))

    # ------------------------------------------------------------------
    def run(
        self,
        theta: np.ndarray,
        duration: float = 4.0,
        record_every: int = 0,
        stop_on_fall: bool = True,
    ) -> Episode:
        self.program.reset()
        self.body.reset()
        self.bind(theta)
        self.t = 0.0

        V = self.variants
        x0 = self.body.pos[:, 1, 0].copy()
        fallen_at = np.full(V, np.inf)
        upright_acc = np.zeros(V)
        steps = int(duration / self.dt)
        frames: List[np.ndarray] = []
        targets: List[np.ndarray] = []

        for i in range(steps):
            self._feed()
            outputs = self.program.step(self.dt)
            cmd = outputs["cmd"].data
            self.body.set_command(cmd)
            self.body.step()
            self.t += self.dt

            newly = np.isinf(fallen_at) & self.body.fallen
            fallen_at[newly] = self.t
            upright_acc += np.cos(self.body.torso_pitch()) * (self.t < fallen_at)
            if stop_on_fall and np.all(~np.isinf(fallen_at)):
                break

            if record_every and i % record_every == 0:
                frames.append(self.body.pos.copy())
                targets.append(cmd.copy())

        distance = self.body.pos[:, 1, 0] - x0
        reward = np.array(self.program.get("rew").data, dtype=np.float64).reshape(V)
        alive = np.where(np.isinf(fallen_at), duration, fallen_at)
        fitness = distance - 1.2 * (duration - alive) + 0.10 * upright_acc * self.dt
        trace = None
        if frames:
            trace = {
                "pos": np.stack(frames, axis=0),
                "cmd": np.stack(targets, axis=0),
                "dt": np.array(record_every * self.dt),
            }
        return Episode(
            distance=distance,
            fitness=fitness,
            fallen_at=fallen_at,
            upright=upright_acc * self.dt,
            reward=reward,
            trace=trace,
        )
