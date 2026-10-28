"""Executable ledger.

Turns a parsed ``.myc`` file into a tick function:

    program = Program.from_source(source)
    program.reset()
    program.set_input("pitch", T((), 0.01))
    outputs = program.step(dt=1/240)
    outputs["body.joints"]  ->  T(("j",), [...])

Supported connectives: ``:=`` (bind), ``~>`` (input port), ``<<``
(continuous state, RK4), ``!>`` (actuator sink). ``assert`` entries are
discharged by the static checker and ignored here. ``<<<`` is executed by
the evolutionary scheduler in ``mycel.walker.train``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from ..lexer import Token
from ..parser import Entry, parse_source
from .expr import compile_expr
from .tensor import T

Fn = Callable[[Dict[str, T]], T]


class ProgramError(ValueError):
    pass


class WindowState:
    """Sliding time integral for ``int[a..b] expr``."""

    def __init__(self) -> None:
        self.t = 0.0
        self.cum: Optional[T] = None
        self.samples: List[tuple] = []
        self.pending: Optional[T] = None

    def integral(self, lo: float, hi: float, value: T) -> T:
        self.pending = value
        zero = T(value.axes, np.zeros(value.data.shape))
        if self.cum is None:
            return zero
        upper = min(hi, self.t)
        if upper <= lo:
            return zero
        return T(self.cum.axes, self.cum.data - self._cum_at(lo))

    def _cum_at(self, when: float) -> np.ndarray:
        if not self.samples or when <= self.samples[0][0]:
            return np.zeros_like(self.cum.data)
        best = self.samples[0][1]
        for t, snap in self.samples:
            if t <= when:
                best = snap
            else:
                break
        return best

    def advance(self, dt: float) -> None:
        self.t += dt
        if self.pending is None:
            return
        step = self.pending
        if self.cum is None:
            self.cum = T(step.axes, np.zeros(step.data.shape))
        self.cum = T(self.cum.axes, self.cum.data + step.data * dt)
        if not self.samples or self.t - self.samples[-1][0] >= 1.0 / 512.0:
            self.samples.append((self.t, self.cum.data.copy()))
            if len(self.samples) > 4096:
                self.samples = self.samples[::2]
        self.pending = None

    def reset(self) -> None:
        self.t = 0.0
        self.cum = None
        self.samples = []
        self.pending = None


@dataclass
class Field:
    name: str
    kind: str  # "bind" | "input" | "state" | "sink"
    connective: str
    sink_target: str = ""
    frozen: bool = False


def _has_window(tokens: Sequence[Token]) -> bool:
    return any(t.kind == "IDENT" and t.value == "int" for t in tokens)


SCHEDULER_SYMBOLS = {"genome", "rollout", "clone", "vary", "archive_best", "mutate"}


def _is_scheduler(tokens: Sequence[Token]) -> bool:
    """True for expressions owned by the ``<<<`` evolutionary scheduler."""
    for t in tokens:
        if t.kind == "OP" and t.value == "<<<":
            return True
        if t.kind == "IDENT" and t.value in SCHEDULER_SYMBOLS:
            return True
    return False


def _is_genesis(tokens: Sequence[Token]) -> bool:
    """True if the expression draws randomness: evaluated once, at genesis."""
    return any(
        (t.kind == "OP" and t.value == "?")
        or (t.kind == "IDENT" and t.value in ("randn", "zeros"))
        for t in tokens
    )


class Program:
    def __init__(self, entries: Sequence[Entry], seed: int = 0) -> None:
        self.entries = list(entries)
        self.rng = np.random.default_rng(seed)
        self.fields: List[Field] = []
        self.compiled: Dict[str, Fn] = {}
        self.seeds: Dict[str, Fn] = {}
        self.windows: Dict[str, WindowState] = {}
        self.env: Dict[str, T] = {}
        self.outputs: Dict[str, T] = {}
        self.sinks: Dict[str, str] = {}
        self.states: Dict[str, T] = {}
        self.time = 0.0
        self._build()

    # -- construction ---------------------------------------------------
    @classmethod
    def from_source(cls, source: str, seed: int = 0) -> "Program":
        return cls(parse_source(source), seed=seed)

    def _build(self) -> None:
        for e in self.entries:
            if e.role != "field" or e.name is None:
                continue
            if e.connective == "~>":
                self.fields.append(Field(e.name, "input", "~>"))
                continue
            if e.connective == "<<<":
                continue
            if e.connective == "!>":
                self.fields.append(Field(e.name, "sink", "!>", e.expr_text()))
                self.sinks[e.name] = e.expr_text()
                continue
            if e.connective in (":=", "=>"):
                if _is_scheduler(e.expr) or (e.seed is not None and _is_scheduler(e.seed)):
                    continue
                self.fields.append(Field(e.name, "bind", ":=", frozen=_is_genesis(e.expr)))
                self.compiled[e.name] = compile_expr(e.expr)
                if _has_window(e.expr):
                    self.windows[e.name] = WindowState()
                continue
            if e.connective == "<<":
                self.fields.append(Field(e.name, "state", "<<"))
                self.compiled[e.name] = compile_expr(e.expr)
                if e.seed is not None:
                    self.seeds[e.name] = compile_expr(e.seed)
                if _has_window(e.expr):
                    self.windows[e.name] = WindowState()
                continue
            raise ProgramError(
                f"field {e.name!r}: connective {e.connective!r} is not executable "
                "(supported: := ~> << !>)"
            )

    @staticmethod
    def _sink_key(e: Entry) -> str:
        return f"!{e.name}"

    # -- execution ------------------------------------------------------
    def reset(self) -> None:
        self.env = {"__rng__": self.rng}  # type: ignore[dict-item]
        self.states = {}
        self.outputs = {}
        self.time = 0.0
        for w in self.windows.values():
            w.reset()
        for f in self.fields:
            if f.kind == "bind" and f.frozen:
                self.env[f.name] = self._eval(f, self.env)

    def _eval(self, f: Field, env: Dict[str, T]) -> T:
        env["__rng__"] = self.rng  # type: ignore[assignment]
        if f.name in self.windows:
            env["__window__"] = self.windows[f.name]  # type: ignore[assignment]
        return self.compiled[f.name](env)

    def set_input(self, name: str, value: T) -> None:
        self.env[name] = value

    def set(self, name: str, value: T) -> None:
        self.env[name] = value

    def get(self, name: str) -> T:
        return self.env[name]

    def sink(self, name: str) -> T:
        return self.outputs[name]

    def step(self, dt: float) -> Dict[str, T]:
        env = self.env
        for f in self.fields:
            if f.kind == "input":
                continue
            if f.kind == "bind":
                if f.frozen:
                    continue
                env[f.name] = self._eval(f, env)
                continue
            if f.kind == "state":
                self._integrate(f, dt, env)
                continue
            if f.kind == "sink":
                if f.name not in env:
                    raise ProgramError(
                        f"actuator {f.name!r} has no source field; bind it with ':=' first"
                    )
                self.outputs[f.name] = env[f.name]
        for w in self.windows.values():
            w.advance(dt)
        self.time += dt
        return self.outputs

    def _integrate(self, f: Field, dt: float, env: Dict[str, T]) -> None:
        fn = self.compiled[f.name]

        def deriv(h: T) -> T:
            local = dict(env)
            local[f.name] = h
            local["__rng__"] = self.rng  # type: ignore[assignment]
            return fn(local)

        h = self.states.get(f.name)
        if h is None:
            seed_val = self.seeds[f.name](dict(env, __rng__=self.rng)) if f.name in self.seeds else T.scalar(0.0)
            h = seed_val + 0.0 * deriv(seed_val)
            self.states[f.name] = h

        half = dt * 0.5
        k1 = deriv(h)
        k2 = deriv(h + half * k1)
        k3 = deriv(h + half * k2)
        k4 = deriv(h + dt * k3)
        h = h + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        self.states[f.name] = h
        env[f.name] = h

    # -- introspection --------------------------------------------------
    def describe(self) -> str:
        glyph = {"bind": ":=", "input": "~>", "state": "<<", "sink": "!>"}
        return "\n".join(f"  {glyph[f.kind]:2s} {f.name}" for f in self.fields)
