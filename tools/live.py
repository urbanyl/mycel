"""Live window: watch the MYCEL biped walk in real time."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation  # noqa: E402

from mycel.walker.controller import Walker  # noqa: E402

BG = "#080C11"
FG = "#C9DEE8"
MUTED = "#5E7A8A"
CYAN = "#4FD6E9"
MINT = "#6FE9D4"
VIOLET = "#B78CFF"
AMBER = "#FFB86B"

SHOULDER, HIP, KNEE_L, FOOT_L, KNEE_R, FOOT_R = range(6)
STEPS_PER_FRAME = 4
DT = 1.0 / 240.0
RESET_AT = 24.0

SRC = (ROOT / "examples" / "05-walker-learn.myc").read_text(encoding="utf-8")
THETA = np.load(ROOT / "runs" / "demo" / "theta.npz")["theta"].reshape(1, -1)

walker = Walker(SRC, variants=1, seed=11)
walker.program.reset()
walker.body.reset()
walker.bind(THETA)

fig, ax = plt.subplots(figsize=(11.5, 5.6))
fig.patch.set_facecolor(BG)
try:
    fig.canvas.manager.set_window_title("MYCEL - live biped")
except Exception:
    pass

history = []


def style():
    ax.set_facecolor(BG)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#17303C")


def draw_ground(x0, x1):
    ax.plot([x0, x1], [0, 0], color="#2C5A6B", lw=2.4, zorder=1)
    ax.fill_between([x0, x1], -0.6, 0, color="#0B141C", zorder=0)
    start = int(np.floor(x0))
    end = int(np.ceil(x1))
    for m in range(start, end + 1):
        major = m % 5 == 0
        h = 0.13 if major else 0.06
        ax.plot([m, m], [0, -h], color="#2C5A6B", lw=1.4 if major else 1.0, zorder=1)
        if major:
            ax.text(m, -0.20, f"{m} m", color=MUTED, fontsize=8,
                    family="monospace", ha="center", va="top")


def draw_body(pos, alpha=1.0, lw=6.0):
    segs = [
        (pos[SHOULDER], pos[HIP], VIOLET, lw + 2),
        (pos[HIP], pos[KNEE_L], CYAN, lw),
        (pos[KNEE_L], pos[FOOT_L], CYAN, lw),
        (pos[HIP], pos[KNEE_R], CYAN, lw),
        (pos[KNEE_R], pos[FOOT_R], CYAN, lw),
    ]
    for a, b, color, width in segs:
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=width,
                solid_capstyle="round", alpha=alpha, zorder=4)
    for idx, color, size in ((HIP, MINT, 34), (KNEE_L, MINT, 24), (KNEE_R, MINT, 24),
                             (FOOT_L, AMBER, 28), (FOOT_R, AMBER, 28)):
        ax.scatter([pos[idx][0]], [pos[idx][1]], s=size, color=color,
                   zorder=5, alpha=alpha)
    head = pos[SHOULDER] + np.array([0.0, 0.115])
    ax.scatter([head[0]], [head[1]], s=120, color=VIOLET, zorder=5, alpha=alpha)
    ax.scatter([head[0]], [head[1]], s=44, color=BG, zorder=6, alpha=alpha)


def frame(_):
    for _ in range(STEPS_PER_FRAME):
        walker._feed()
        out = walker.program.step(DT)
        walker.body.set_command(out["cmd"].data)
        walker.body.step()
        walker.t += DT
        if walker.body.pos[0, HIP, 0] > RESET_AT:
            walker.program.reset()
            walker.body.reset()
            walker.bind(THETA)
            history.clear()

    pos = walker.body.pos[0].copy()
    x = float(pos[HIP, 0])
    history.append(pos)
    if len(history) > 16:
        history.pop(0)

    ax.clear()
    style()
    ax.set_xlim(x - 2.4, x + 3.2)
    ax.set_ylim(-0.55, 1.95)
    draw_ground(x - 2.6, x + 3.4)

    for k, old in enumerate(history[:-1]):
        draw_body(old, alpha=0.04 + 0.22 * k / max(len(history) - 1, 1), lw=3.0)
    draw_body(pos)

    ax.text(0.015, 0.97, "MYCEL  /  liquid tissue 24 cells  /  live",
            transform=ax.transAxes, color=FG, fontsize=13, family="monospace",
            va="top")
    ax.text(0.015, 0.91, "tau*h' = -h + tanh(W.h + U.x + b)   ->   foot targets",
            transform=ax.transAxes, color=MUTED, fontsize=9, family="monospace",
            va="top")

    speed = float(walker.body.vel[0, HIP, 0])
    ax.text(0.985, 0.97, f"{x:6.2f} m", transform=ax.transAxes, color=MINT,
            fontsize=15, family="monospace", ha="right", va="top")
    ax.text(0.985, 0.895, f"{speed:5.2f} m/s", transform=ax.transAxes, color=CYAN,
            fontsize=11, family="monospace", ha="right", va="top")
    ax.text(0.985, 0.835, f"t = {walker.t:6.2f} s", transform=ax.transAxes,
            color=MUTED, fontsize=9, family="monospace", ha="right", va="top")

    return ()


if __name__ == "__main__":
    try:
        anim = FuncAnimation(fig, frame, interval=16, blit=False, cache_frame_data=False)
        fig.tight_layout(pad=0.4)
        plt.show()
    except Exception:
        (ROOT / "runs" / "demo" / "live.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise
