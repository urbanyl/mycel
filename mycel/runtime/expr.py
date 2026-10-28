"""Expression compiler for the MYCEL runtime.

Compiles the token run of a ledger entry into a Python closure ``f(env)``
where ``env`` maps field names to named-axis tensors.

Supported surface: numbers with SI suffixes, dotted names, ``+ - * / ^``,
``.*[axis]`` contraction, ``**`` outer product, ``>>[a,b]`` reindex,
function calls, ``?Dist(...)`` sampling and ``int[a..b] expr`` windows.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Sequence

import numpy as np

from ..lexer import Token
from .tensor import T, AxisMismatch, cat

Env = Dict[str, T]
Fn = Callable[[Env], T]

UNIT_SCALE = {
    "s": 1.0,
    "ms": 1e-3,
    "us": 1e-6,
    "min": 60.0,
    "m": 1.0,
    "mm": 1e-3,
    "cm": 1e-2,
    "kg": 1.0,
    "g": 1e-3,
    "N": 1.0,
    "rad": 1.0,
    "deg": np.pi / 180.0,
    "Hz": 1.0,
}

FUNCTIONS = {
    "sin",
    "cos",
    "tanh",
    "sigmoid",
    "exp",
    "log",
    "sqrt",
    "abs",
    "norm",
    "clamp",
    "mean",
    "sum",
    "argmax",
    "pair",
    "dot",
    "relu",
}


class ExprError(ValueError):
    pass


class _P:
    def __init__(self, tokens: Sequence[Token]) -> None:
        self.t = list(tokens)
        self.i = 0

    # -- helpers --------------------------------------------------------
    def peek(self, k: int = 0) -> str:
        j = self.i + k
        return self.t[j].value if j < len(self.t) else ""

    def kind(self, k: int = 0) -> str:
        j = self.i + k
        return self.t[j].kind if j < len(self.t) else "EOF"

    def take(self) -> Token:
        tok = self.t[self.i]
        self.i += 1
        return tok

    def expect(self, value: str) -> Token:
        if self.peek() != value:
            raise ExprError(f"expected {value!r}, got {self.peek()!r}")
        return self.take()

    def at_end(self) -> bool:
        return self.i >= len(self.t)

    # -- grammar --------------------------------------------------------
    def parse(self) -> Fn:
        fn = self.add()
        if not self.at_end():
            raise ExprError(f"trailing token {self.peek()!r}")
        return fn

    def add(self) -> Fn:
        fn = self.mul()
        while self.peek() in ("+", "-"):
            op = self.take().value
            rhs = self.mul()
            fn = _bin(op, fn, rhs)
        return fn

    def mul(self) -> Fn:
        fn = self.unary()
        while True:
            v = self.peek()
            if v in ("*", "/"):
                self.take()
                fn = _bin(v, fn, self.unary())
            elif v == ".*":
                self.take()
                axis = self.axis_name()
                fn = _contract(fn, axis, self.unary())
            elif v == "**":
                self.take()
                lhs = fn
                rhs = self.unary()
                fn = lambda env, l=lhs, r=rhs: l(env).outer(r(env))
            else:
                return fn

    def unary(self) -> Fn:
        if self.peek() == "-":
            self.take()
            inner = self.unary()
            return lambda env: -inner(env)
        return self.power()

    def power(self) -> Fn:
        base = self.postfix()
        if self.peek() == "^":
            self.take()
            exponent = self.unary()
            return lambda env: base(env) ** exponent(env)
        return base

    def postfix(self) -> Fn:
        fn = self.primary()
        while self.peek() == ">>":
            self.take()
            self.expect("[")
            first = self.take().value
            if self.peek() == ":":
                self.take()
                target = self.take().value
                self.expect("]")
                fn = (lambda f, a=first, b=target: (lambda env: f(env).rename(a, b)))(fn)
                continue
            axes = [first]
            while self.peek() == ",":
                self.take()
                axes.append(self.take().value)
            self.expect("]")
            fn = (lambda f, ax: (lambda env: f(env).reindex(ax)))(fn, tuple(axes))
        return fn

    def axis_name(self) -> str:
        self.expect("[")
        name = self.take().value
        self.expect("]")
        return name

    def primary(self) -> Fn:
        tok = self.t[self.i]
        if tok.kind == "NUMBER":
            self.take()
            value = float(tok.value)
            if self.kind() == "IDENT" and self.peek() in UNIT_SCALE:
                value *= UNIT_SCALE[self.take().value]
            return lambda env, v=value: T.scalar(v)

        if tok.value == "(":
            self.take()
            inner = self.add()
            self.expect(")")
            return inner

        if tok.value == "?":
            self.take()
            dist = self.take().value
            self.expect("(")
            args = self.args()
            self.expect(")")
            return _sample(dist, args)

        if tok.kind == "IDENT":
            name = tok.value
            if name == "int" and self.peek(1) == "[":
                self.take()
                self.expect("[")
                lo = self.add()
                self.expect("..")
                hi = self.add()
                self.expect("]")
                body = self.add()
                return _window(lo, hi, body)

            self.take()
            while self.peek() == "." and self.kind(1) == "IDENT":
                self.take()
                name += "." + self.take().value

            if name in ("zeros", "randn") and self.peek() == "[":
                self.take()
                specs = []
                while True:
                    ax = self.take().value
                    size = int(float(self.take().value))
                    specs.append((ax, size))
                    if self.peek() == ",":
                        self.take()
                        continue
                    break
                self.expect("]")
                args = []
                if self.peek() == "(":
                    self.take()
                    args = self.args()
                    self.expect(")")
                return _shaped(name, specs, args)

            axis = None
            if self.peek() == "[":
                self.take()
                axis = self.take().value
                self.expect("]")

            if self.peek() == "(":
                self.take()
                args = self.args()
                self.expect(")")
                if axis is not None and name not in ("cat", "argmax"):
                    raise ExprError(f"{name!r} does not take an axis argument")
                return _call(name, axis, args)

            if axis is not None:
                raise ExprError(f"unexpected axis argument on {name!r}")
            return lambda env, n=name: _lookup(env, n)

        raise ExprError(f"unexpected token {tok.value!r}")

    def args(self) -> List[Fn]:
        out: List[Fn] = []
        if self.peek() == ")":
            return out
        while True:
            out.append(self.add())
            if self.peek() == ",":
                self.take()
                continue
            return out


# ---------------------------------------------------------------------------
# node builders
# ---------------------------------------------------------------------------


def _bin(op: str, a: Fn, b: Fn) -> Fn:
    if op == "+":
        return lambda env: a(env) + b(env)
    if op == "-":
        return lambda env: a(env) - b(env)
    if op == "*":
        return lambda env: a(env) * b(env)
    return lambda env: a(env) / b(env)


def _contract(a: Fn, axis: str, b: Fn) -> Fn:
    return lambda env: a(env).contract(b(env), axis)


def _lookup(env: Env, name: str) -> T:
    try:
        return env[name]
    except KeyError:
        raise ExprError(f"undefined field {name!r}") from None


def _window(lo: Fn, hi: Fn, body: Fn) -> Fn:
    def evaluate(env: Env) -> T:
        state = env.get("__window__")
        if state is None:
            raise ExprError("int[...] used outside a windowed program")
        return state.integral(lo(env).data.item(), hi(env).data.item(), body(env))

    return evaluate


def _sample(dist: str, args: List[Fn]) -> Fn:
    def evaluate(env: Env) -> T:
        rng = env.get("__rng__")
        if rng is None:
            rng = np.random.default_rng(0)
        vals = [a(env) for a in args]
        if dist == "N":
            mu, sigma = (v.data for v in vals)
            return T(vals[0].axes, rng.normal(mu, sigma))
        if dist == "U":
            lo, hi = (v.data for v in vals)
            return T(vals[0].axes, rng.uniform(lo, hi))
        raise ExprError(f"unknown distribution ?{dist}")

    return evaluate


def _call(name: str, axis, args: List[Fn]) -> Fn:
    if name == "cat":
        if axis is None:
            raise ExprError("cat requires an axis: cat[in](...)")
        return lambda env: cat(axis, [a(env) for a in args])

    if name == "argmax":
        if axis is None:
            raise ExprError("argmax requires an axis: argmax[axis](...)")
        return lambda env: _argmax(args[0](env), axis)

    if name == "norm":
        return lambda env: _norm(args[0](env))

    if name == "clamp":
        def clamp(env):
            x = args[0](env)
            m = args[1](env)
            return _minimum(_maximum(x, -m), m)

        return clamp

    if name == "pair":
        def pair(env):
            a, b = args[0](env), args[1](env)
            order = tuple(dict.fromkeys(a.axes + b.axes))
            return T((), float(np.sum(a.align(order) * b.align(order))))

        return pair

    if name == "dot":
        def dot(env):
            a, b = args[0](env), args[1](env)
            shared = [ax for ax in a.axes if ax in b.axes]
            if not shared:
                raise ExprError("dot needs at least one shared axis")
            out = a
            for ax in shared:
                out = out.contract(b, ax)
                b = b.sum(ax)
            return out

        return dot

    if name not in FUNCTIONS:
        raise ExprError(f"unknown function {name!r}")

    unary = {
        "sin": np.sin,
        "cos": np.cos,
        "tanh": np.tanh,
        "exp": np.exp,
        "log": np.log,
        "sqrt": np.sqrt,
        "abs": np.abs,
        "relu": lambda x: np.maximum(x, 0.0),
        "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-x)),
    }[name]
    return lambda env: T(args[0](env).axes, unary(args[0](env).data))


def _shaped(name: str, specs, args: List[Fn]) -> Fn:
    axes = tuple(a for a, _ in specs)
    shape = tuple(s for _, s in specs)

    if name == "zeros":
        return lambda env: T(axes, np.zeros(shape))

    def evaluate(env: Env) -> T:
        rng = env.get("__rng__")
        if rng is None:
            raise ExprError("randn used outside a program with a generator")
        mu = args[0](env).data if args else 0.0
        sd = args[1](env).data if len(args) > 1 else 1.0
        return T(axes, rng.normal(mu, sd, size=shape))

    return evaluate


def _argmax(t: T, axis: str) -> T:
    idx = t.axes.index(axis)
    return T(t.axes[:idx] + t.axes[idx + 1 :], np.argmax(t.data, axis=idx).astype(np.float64))


def _norm(t: T) -> T:
    return T((), float(np.sqrt(np.sum(t.data * t.data))))


def _maximum(a: T, b: T) -> T:
    order = tuple(dict.fromkeys(a.axes + b.axes))
    return T(order, np.maximum(a.align(order), b.align(order)))


def _minimum(a: T, b: T) -> T:
    order = tuple(dict.fromkeys(a.axes + b.axes))
    return T(order, np.minimum(a.align(order), b.align(order)))


def compile_expr(tokens: Sequence[Token]) -> Fn:
    if not tokens:
        raise ExprError("empty expression")
    return _P(tokens).parse()
