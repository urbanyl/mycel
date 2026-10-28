"""A vectorised planar biped.

Six particles form the skeleton::

        0  shoulder
        |
        1  hip
       / \\
      2    4        knee
      |    |
      3    5        foot

Bones are XPBD distance constraints. The controller does not command
torques: it commands *where to put each foot*, which is what a locomotion
policy actually decides. A foot in swing is pulled toward
``hip + (dx, lift)``; a foot in stance is pinned to the world at the point
where it landed. The body is then a free inverted pendulum over that pin,
so it falls forward under gravity and has to catch itself on the next
foot. Bad foot placement means falling; good placement means walking.

Everything is batched: ``pos`` has shape (V, 6, 2) and a whole population
is simulated in one numpy pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np

SHOULDER, HIP, KNEE_L, FOOT_L, KNEE_R, FOOT_R = range(6)
N_POINTS = 6
FEET = (FOOT_L, FOOT_R)

BONES = (
    (SHOULDER, HIP),
    (HIP, KNEE_L),
    (KNEE_L, FOOT_L),
    (HIP, KNEE_R),
    (KNEE_R, FOOT_R),
    (HIP, FOOT_L),
    (HIP, FOOT_R),
)
BONE_LENGTHS = np.array([0.50, 0.42, 0.40, 0.42, 0.40, 0.760, 0.760])
KNEE_STANCE = 0.90
KNEE_SWING = 0.02
LEG_REST_MIN = 0.640
LEG_REST_SPAN = 0.170
MASSES = np.array([6.0, 6.0, 1.4, 1.0, 1.4, 1.0])
INV_MASS = 1.0 / MASSES

GRAVITY = -9.81
MIN_PHASE = 0.075
LIFT_HIGH = 0.055
LIFT_LOW = 0.015


def _angles(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angle of (b - a) measured from straight down, positive forward (+x)."""
    d = b - a
    return np.arctan2(d[..., 0], -d[..., 1])


def _knee(hip: np.ndarray, foot: np.ndarray, l1: float = 0.42, l2: float = 0.40) -> np.ndarray:
    """Knee position for a two-link leg, bending backward (+x)."""
    d = foot - hip
    length = float(np.linalg.norm(d))
    length = min(max(length, 1e-6), l1 + l2 - 1e-4)
    u = d / length
    a = (l1 * l1 - l2 * l2 + length * length) / (2.0 * length)
    h = np.sqrt(max(l1 * l1 - a * a, 0.0))
    perp = np.array([u[1], -u[0]])
    if perp[0] < 0:
        perp = -perp
    return hip + a * u + h * perp


def _rotate(pos: np.ndarray, pivot: int, members, phi: np.ndarray) -> None:
    c = np.cos(phi)[:, None]
    s = np.sin(phi)[:, None]
    d = pos[:, members, :] - pos[:, pivot : pivot + 1, :]
    x = d[..., 0] * c - d[..., 1] * s
    y = d[..., 0] * s + d[..., 1] * c
    pos[:, members, 0] = pos[:, pivot, 0][:, None] + x
    pos[:, members, 1] = pos[:, pivot, 1][:, None] + y


@dataclass
class Biped:
    variants: int = 1
    dt: float = 1.0 / 240.0
    substeps: int = 3
    iterations: int = 5
    seed: int = 0
    foot_speed: float = 2.6          # m/s, swing foot tracking speed
    damping: float = 0.9995
    torso_gain: float = 0.25

    pos: np.ndarray = field(init=False)
    prev: np.ndarray = field(init=False)
    vel: np.ndarray = field(init=False)
    pin: np.ndarray = field(init=False)
    swing: np.ndarray = field(init=False)
    cmd: np.ndarray = field(init=False)
    fallen: np.ndarray = field(init=False)
    contact: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.reset()

    # ------------------------------------------------------------------
    def reset(self) -> None:
        V = self.variants
        pos = np.zeros((V, N_POINTS, 2))
        hip = np.array([0.0, 0.700])
        pos[:, HIP] = hip
        pos[:, SHOULDER] = hip + (0.0, 0.50)
        for knee_idx, foot_idx, fx in ((KNEE_L, FOOT_L, -0.165), (KNEE_R, FOOT_R, 0.165)):
            foot = np.array([fx, 0.0])
            pos[:, knee_idx] = _knee(hip, foot)
            pos[:, foot_idx] = foot
        self.pos = pos
        self.prev = pos.copy()
        self.vel = np.zeros_like(pos)
        self.pin = pos[:, FEET, :].copy()          # (V, 2, 2)
        self.swing = np.zeros((V, 2), dtype=bool)
        self.phase_timer = np.zeros((V, 2))
        self.cmd = np.zeros((V, 8))
        self.fallen = np.zeros(V, dtype=bool)
        self.contact = np.ones((V, 2), dtype=bool)
        self.travel = np.zeros(V)
        self.time = 0.0

    # ------------------------------------------------------------------
    def set_command(self, values: np.ndarray) -> None:
        """``values`` (V, 8): dxL, liftL, dxR, liftR, lean, pushL, pushR, spare."""
        v = np.clip(np.asarray(values, dtype=np.float64), -1.0, 1.0)
        cmd = np.empty_like(v)
        cmd[:, 0] = 0.30 * v[:, 0]
        cmd[:, 1] = 0.12 * np.maximum(v[:, 1] + 0.12, 0.0) / 1.12
        cmd[:, 2] = 0.30 * v[:, 2]
        cmd[:, 3] = 0.12 * np.maximum(v[:, 3] + 0.12, 0.0) / 1.12
        cmd[:, 4] = 0.28 * v[:, 4]
        cmd[:, 5] = v[:, 5]
        cmd[:, 6] = v[:, 6]
        cmd[:, 7] = v[:, 7]
        self.cmd = cmd

    # ------------------------------------------------------------------
    def step(self) -> None:
        h = self.dt / self.substeps
        lift = self.cmd[:, [1, 3]]
        was_swing = self.swing
        want = np.where(self.swing, lift > LIFT_LOW, lift > LIFT_HIGH)
        # a leg cannot switch phase faster than MIN_PHASE: this forbids
        # chatter-stepping, which would otherwise drag the body along
        locked = want != self.swing
        locked &= self.phase_timer < MIN_PHASE
        self.swing = np.where(locked, self.swing, want)
        self.phase_timer = np.where(self.swing != was_swing, 0.0, self.phase_timer + self.dt)
        # a biped always keeps one support foot
        both = self.swing[:, 0] & self.swing[:, 1]
        if np.any(both):
            loser = np.where(lift[:, 0] < lift[:, 1], 0, 1)
            self.swing[both, loser[both]] = False
        landing = (~self.swing) & was_swing
        for j, foot in enumerate(FEET):
            idx = landing[:, j]
            if np.any(idx):
                self.pin[idx, j, :] = self.pos[idx, foot, :]

        for _ in range(self.substeps):
            self.vel[:, :, 1] += GRAVITY * h
            # ankle push-off: a stance leg can thrust along the ground
            push = self.cmd[:, 5:7]
            thrust = np.where(~self.swing, push, 0.0).sum(axis=1)
            self.vel[:, HIP, 0] += 6.0 * thrust * h
            self.prev = self.pos.copy()
            self.pos = self.pos + self.vel * h
            for _ in range(self.iterations):
                self._solve_bones()
                self._solve_feet(h)
                self._solve_knee_guard()
                self._solve_torso()
                self._solve_ground()
            self._ground_friction()
            self.vel = (self.pos - self.prev) / h
            np.clip(self.vel, -8.0, 8.0, out=self.vel)
            self.vel *= self.damping

        self.time += self.dt
        self.contact = np.stack(
            [self.pos[:, FOOT_L, 1] < 0.02, self.pos[:, FOOT_R, 1] < 0.02], axis=1
        )
        self._detect_fall()

    # ------------------------------------------------------------------
    def _solve_bones(self) -> None:
        pos = self.pos
        for i, ((a, b), rest) in enumerate(zip(BONES, BONE_LENGTHS)):
            if i >= 5:
                stiff = np.where(self.swing[:, i - 5], KNEE_SWING, KNEE_STANCE)
            else:
                stiff = 1.0
            d = pos[:, b] - pos[:, a]
            dist = np.linalg.norm(d, axis=1) + 1e-9
            err = (dist - rest) * stiff
            wsum = INV_MASS[a] + INV_MASS[b]
            corr = (d / dist[:, None]) * (err / wsum)[:, None]
            pos[:, a] += INV_MASS[a] * corr
            pos[:, b] -= INV_MASS[b] * corr

    def _solve_feet(self, h: float) -> None:
        pos = self.pos
        limit = self.foot_speed * h
        hip_x = pos[:, HIP, 0]
        for j, foot in enumerate(FEET):
            swinging = self.swing[:, j]
            target_x = np.where(swinging, hip_x + self.cmd[:, 2 * j], self.pin[:, j, 0])
            target_y = np.where(swinging, self.cmd[:, 2 * j + 1], 0.0)
            dx = np.clip(target_x - pos[:, foot, 0], -limit, limit)
            dy = np.clip(target_y - pos[:, foot, 1], -limit, limit)
            pos[:, foot, 0] = np.where(swinging, pos[:, foot, 0] + dx, self.pin[:, j, 0])
            pos[:, foot, 1] = np.where(swinging, pos[:, foot, 1] + dy, 0.0)

    def _solve_knee_guard(self) -> None:
        pos = self.pos
        for knee, hip, foot in ((KNEE_L, HIP, FOOT_L), (KNEE_R, HIP, FOOT_R)):
            thigh = _angles(pos[:, hip], pos[:, knee])
            shank = _angles(pos[:, knee], pos[:, foot])
            bend = thigh - shank
            bad = bend < 0.0
            if np.any(bad):
                phi = np.where(bad, -0.5 * bend, 0.0)
                _rotate(pos, knee, [foot], phi)

    def _solve_torso(self) -> None:
        up = self.pos[:, SHOULDER] - self.pos[:, HIP]
        theta = np.arctan2(up[:, 0], up[:, 1])
        err = theta - self.cmd[:, 4]
        _rotate(self.pos, HIP, [SHOULDER], self.torso_gain * np.clip(err, -0.12, 0.12))

    def _solve_ground(self) -> None:
        np.maximum(self.pos[:, :, 1], 0.0, out=self.pos[:, :, 1])

    def _ground_friction(self) -> None:
        touching = self.pos[:, :, 1] <= 1e-4
        tang = self.pos[:, :, 0] - self.prev[:, :, 0]
        keep = np.array([0.0, 0.0, 0.15, 0.0, 0.15, 0.0])
        self.pos[:, :, 0] = np.where(
            touching, self.prev[:, :, 0] + tang * keep[None, :], self.pos[:, :, 0]
        )

    def _detect_fall(self) -> None:
        if self.time < 0.40:
            return
        height = self.pos[:, HIP, 1]
        pitch = np.abs(self.torso_pitch())
        self.fallen |= (height < 0.50) | (pitch > 1.2)

    # ------------------------------------------------------------------
    def torso_pitch(self) -> np.ndarray:
        up = self.pos[:, SHOULDER] - self.pos[:, HIP]
        return np.arctan2(up[:, 0], up[:, 1])

    def torso_pitch_rate(self) -> np.ndarray:
        up = (self.pos[:, SHOULDER] - self.pos[:, HIP]) - (
            self.prev[:, SHOULDER] - self.prev[:, HIP]
        )
        return np.arctan2(up[:, 0], up[:, 1]) / self.dt

    def joint_angles(self) -> np.ndarray:
        tl = _angles(self.pos[:, HIP], self.pos[:, KNEE_L])
        tr = _angles(self.pos[:, HIP], self.pos[:, KNEE_R])
        sl = _angles(self.pos[:, KNEE_L], self.pos[:, FOOT_L])
        sr = _angles(self.pos[:, KNEE_R], self.pos[:, FOOT_R])
        return np.stack([tl, sl - tl, tr, sr - tr], axis=1)

    def joint_rates(self) -> np.ndarray:
        a = self.joint_angles()
        prev = getattr(self, "_prev_angles", a)
        self._prev_angles = a.copy()
        return (a - prev) / self.dt

    def render_state(self) -> Dict[str, np.ndarray]:
        return {
            "pos": self.pos.copy(),
            "contact": self.contact.copy(),
            "cmd": self.cmd.copy(),
        }
