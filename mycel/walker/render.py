"""Rendering for the walking demo: GIF, contact sheet, learning curve."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

BG = "#080C11"
FG = "#C9DEE8"
MUTED = "#5E7A8A"
CYAN = "#4FD6E9"
MINT = "#6FE9D4"
VIOLET = "#B78CFF"
AMBER = "#FFB86B"

SHOULDER, HIP, KNEE_L, FOOT_L, KNEE_R, FOOT_R = range(6)


def _bones(pos: np.ndarray):
    return [
        (pos[SHOULDER], pos[HIP]),
        (pos[HIP], pos[KNEE_L]),
        (pos[KNEE_L], pos[FOOT_L]),
        (pos[HIP], pos[KNEE_R]),
        (pos[KNEE_R], pos[FOOT_R]),
    ]


def _draw_body(ax, pos: np.ndarray, alpha: float = 1.0) -> None:
    for (a, b) in _bones(pos):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=CYAN, lw=5.0, solid_capstyle="round", alpha=alpha, zorder=3)
    torso = [pos[SHOULDER], pos[HIP]]
    ax.plot([torso[0][0], torso[1][0]], [torso[0][1], torso[1][1]], color=VIOLET, lw=7.0,
            solid_capstyle="round", alpha=alpha, zorder=4)
    for idx, color, size in ((HIP, MINT, 26), (KNEE_L, MINT, 20), (KNEE_R, MINT, 20),
                             (FOOT_L, AMBER, 22), (FOOT_R, AMBER, 22)):
        ax.scatter([pos[idx][0]], [pos[idx][1]], s=size, color=color, zorder=5, alpha=alpha)
    head = pos[SHOULDER] + np.array([0.0, 0.11])
    ax.scatter([head[0]], [head[1]], s=90, color=VIOLET, zorder=5, alpha=alpha)
    ax.scatter([head[0]], [head[1]], s=34, color=BG, zorder=6, alpha=alpha)


def _style(ax) -> None:
    ax.set_facecolor(BG)
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-0.15, 1.9)
    ax.set_aspect("equal")
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#17303C")


def _ground(ax, x0: float, x1: float) -> None:
    ax.plot([x0, x1], [0, 0], color="#2C5A6B", lw=2.0, zorder=1)
    ax.fill_between([x0, x1], -0.35, 0, color="#0C1720", zorder=0)


def render_gif(
    trace: Dict[str, np.ndarray],
    path: str | Path,
    title: str = "",
    subtitle: str = "",
    fps: int = 30,
    stride: int = 2,
    trail: int = 26,
) -> Path:
    pos = np.asarray(trace["pos"])
    if pos.ndim == 4:
        pos = pos[:, 0]
    frames = range(0, len(pos), stride)
    dt = float(trace["dt"]) * stride

    fig, ax = plt.subplots(figsize=(9.6, 5.2), dpi=90)
    fig.patch.set_facecolor(BG)
    _style(ax)

    def update(i: int):
        ax.clear()
        _style(ax)
        p = pos[i]
        x = p[HIP, 0]
        ax.set_xlim(x - 2.1, x + 2.4)
        _ground(ax, x - 2.2, x + 2.6)

        for j in range(max(0, i - trail), i, 2):
            _draw_body(ax, pos[j], alpha=0.06 + 0.30 * (j - (i - trail)) / max(trail, 1))

        _draw_body(ax, p)

        ax.text(0.02, 0.96, title, transform=ax.transAxes, color=FG, fontsize=12,
                family="monospace", va="top")
        ax.text(0.02, 0.90, subtitle, transform=ax.transAxes, color=MUTED, fontsize=9,
                family="monospace", va="top")
        ax.text(0.98, 0.96, f"x = {p[HIP,0]:5.2f} m", transform=ax.transAxes, color=MINT,
                fontsize=11, family="monospace", ha="right", va="top")
        ax.text(0.98, 0.90, f"t = {i*dt:5.2f} s", transform=ax.transAxes, color=MUTED,
                fontsize=9, family="monospace", ha="right", va="top")

    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)
    path = Path(path)
    anim.save(path, writer=PillowWriter(fps=fps), dpi=90)
    plt.close(fig)
    return path


def contact_sheet(
    bad: Dict[str, np.ndarray],
    good: Dict[str, np.ndarray],
    path: str | Path,
    labels: Sequence[str] = ("before training", "after training"),
) -> Path:
    fig, axes = plt.subplots(2, 4, figsize=(14.0, 5.6), dpi=100)
    fig.patch.set_facecolor(BG)
    for row, (trace, label) in enumerate(zip((bad, good), labels)):
        pos = np.asarray(trace["pos"])
        if pos.ndim == 4:
            pos = pos[:, 0]
        picks = np.linspace(0, len(pos) - 1, 4).astype(int)
        for col, i in enumerate(picks):
            ax = axes[row][col]
            _style(ax)
            p = pos[i]
            x = p[HIP, 0]
            ax.set_xlim(x - 1.2, x + 1.2)
            _ground(ax, x - 1.3, x + 1.3)
            _draw_body(ax, p)
            if col == 0:
                ax.text(0.03, 0.95, label, transform=ax.transAxes, color=FG, fontsize=10,
                        family="monospace", va="top")
            ax.text(0.03, 0.06, f"t={i*float(trace['dt']):.2f}s", transform=ax.transAxes,
                    color=MUTED, fontsize=8, family="monospace")
    fig.tight_layout(pad=0.6)
    path = Path(path)
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return path


def learning_curve(history: Dict[str, List[float]], path: str | Path) -> Path:
    g = np.asarray(history["generation"])
    best = np.asarray(history["best_distance"])
    mean = np.asarray(history["mean_distance"])
    fell = np.asarray(history["fell_rate"])

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9.0, 5.4), dpi=110, sharex=True,
                                  gridspec_kw={"height_ratios": [2.2, 1.0]})
    fig.patch.set_facecolor(BG)
    for a in (ax, ax2):
        a.set_facecolor(BG)
        a.tick_params(colors=MUTED, labelsize=9)
        for spine in a.spines.values():
            spine.set_color("#17303C")
        a.grid(color="#12232C", lw=0.8)

    ax.plot(g, best, color=MINT, lw=2.2, label="best distance in population")
    ax.plot(g, mean, color=CYAN, lw=1.4, alpha=0.8, label="population mean")
    ax.set_ylabel("metres travelled in 4 s", color=FG, family="monospace", fontsize=10)
    ax.legend(facecolor=BG, edgecolor="#17303C", labelcolor=FG, fontsize=9, loc="upper left")
    ax.set_title("MYCEL / evolving a walking controller", color=FG, family="monospace",
                 fontsize=12, loc="left")

    ax2.plot(g, 100 * fell, color=VIOLET, lw=1.8)
    ax2.set_ylabel("% fallen", color=FG, family="monospace", fontsize=10)
    ax2.set_xlabel("generation", color=FG, family="monospace", fontsize=10)

    fig.tight_layout()
    path = Path(path)
    fig.savefig(path, facecolor=BG)
    plt.close(fig)
    return path
