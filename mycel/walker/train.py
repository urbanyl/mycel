"""Evolutionary training loop for the walking ledger.

The ledger declares the genome, the variation operator and the fitness.
This scheduler executes ``<<<``: a mirrored-sampling evolution strategy
with rank shaping and Adam on the mean. A population is an axis, so the
whole generation is one batched integration.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from ..parser import Entry, parse_source
from .controller import N_PARAMS, Walker


def _directives(entries: List[Entry]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for e in entries:
        if e.role == "directive":
            words = e.raw.split()
            for word in words[2:]:
                if "=" in word:
                    k, v = word.split("=", 1)
                    out[k] = v
    return out


def _rank_shaping(fitness: np.ndarray) -> np.ndarray:
    order = np.argsort(fitness)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(fitness), dtype=np.float64)
    return ranks / (len(fitness) - 1.0) - 0.5


@dataclass
class History:
    generation: List[int] = field(default_factory=list)
    best_distance: List[float] = field(default_factory=list)
    mean_distance: List[float] = field(default_factory=list)
    best_fitness: List[float] = field(default_factory=list)
    fell_rate: List[float] = field(default_factory=list)

    def as_dict(self) -> Dict[str, List[float]]:
        return {
            "generation": self.generation,
            "best_distance": self.best_distance,
            "mean_distance": self.mean_distance,
            "best_fitness": self.best_fitness,
            "fell_rate": self.fell_rate,
        }


def train(
    source: str,
    out_dir: str | Path,
    generations: Optional[int] = None,
    population: Optional[int] = None,
    sigma: Optional[float] = None,
    lr: Optional[float] = None,
    duration: float = 4.0,
    seed: int = 7,
    log: Optional[Callable[[str], None]] = print,
) -> Dict:
    entries = parse_source(source)
    cfg = _directives(entries)
    gens = generations or int(cfg.get("generations", 60))
    pop = population or int(cfg.get("variant", cfg.get("V", 64)))
    if pop % 2:
        pop += 1
    sig = sigma if sigma is not None else float(cfg.get("sigma", 0.09))
    rate = lr if lr is not None else float(cfg.get("rate", 0.06))

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    theta = rng.normal(0.0, 0.05, N_PARAMS)

    walker = Walker(source, variants=pop, seed=seed)
    history = History()

    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    pairs = pop // 2

    best_theta = theta.copy()
    best_fit = -np.inf
    t0 = time.time()

    for gen in range(gens + 1):
        noise = rng.standard_normal((pairs, N_PARAMS))
        batch = np.concatenate([theta + sig * noise, theta - sig * noise], axis=0)
        episode = walker.run(batch, duration=duration)

        fit = np.where(np.isfinite(episode.fitness), episode.fitness, -1e6)
        f_plus, f_minus = fit[:pairs], fit[pairs:]
        shaped = _rank_shaping(fit)
        graded = shaped[:pairs] - shaped[pairs:]

        grad = (noise.T @ graded) / (pairs * sig)

        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad * grad)
        mhat = m / (1 - beta1 ** (gen + 1))
        vhat = v / (1 - beta2 ** (gen + 1))
        theta = theta + rate * mhat / (np.sqrt(vhat) + eps)

        idx = int(np.argmax(fit))
        if fit[idx] > best_fit:
            best_fit = float(fit[idx])
            best_theta = batch[idx].copy()

        dist = episode.distance
        fell = np.mean(np.isfinite(episode.fallen_at))
        history.generation.append(gen)
        history.best_distance.append(float(dist.max()))
        history.mean_distance.append(float(dist.mean()))
        history.best_fitness.append(float(fit.max()))
        history.fell_rate.append(float(fell))

        if log and (gen % 5 == 0 or gen == gens):
            elapsed = time.time() - t0
            log(
                f"gen {gen:3d}  best {dist.max():6.2f} m  mean {dist.mean():6.2f} m  "
                f"fell {100*fell:5.1f}%  sigma {sig:.3f}  {elapsed:5.1f}s"
            )

    # ---- final evaluation and recording ---------------------------------
    demo = Walker(source, variants=1, seed=seed)
    final_theta = best_theta.reshape(1, -1)
    recorded = demo.run(final_theta, duration=6.0, record_every=4)
    baseline = Walker(source, variants=1, seed=seed).run(
        rng.normal(0.0, 0.05, (1, N_PARAMS)).reshape(1, -1),
        duration=6.0,
        record_every=4,
    )

    result = {
        "generations": gens,
        "population": pop,
        "params": int(N_PARAMS),
        "sigma": sig,
        "learning_rate": rate,
        "duration": duration,
        "best_fitness": float(recorded.fitness[0]),
        "distance": float(recorded.distance[0]),
        "fallen_at": None if not np.isfinite(recorded.fallen_at[0]) else float(recorded.fallen_at[0]),
        "baseline_distance": float(baseline.distance[0]),
        "history": history.as_dict(),
        "theta": best_theta.tolist(),
    }

    np.savez_compressed(
        out / "policy.npz",
        theta=best_theta,
        baseline=baseline.trace["pos"] if baseline.trace else np.zeros(0),
    )
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    if recorded.trace is not None:
        np.savez_compressed(
            out / "trace.npz",
            pos=recorded.trace["pos"],
            cmd=recorded.trace["cmd"],
            dt=recorded.trace["dt"],
        )
    if baseline.trace is not None:
        np.savez_compressed(
            out / "trace_baseline.npz",
            pos=baseline.trace["pos"],
            cmd=baseline.trace["cmd"],
            dt=baseline.trace["dt"],
        )

    return result
