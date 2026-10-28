"""Named-axis tensors for the MYCEL runtime.

A tensor is a numpy array plus an ordered tuple of axis *names*. Every
operation in the specification is defined in terms of those names, so the
runtime never relies on positional broadcasting:

    A * B        requires identical axes
    A .*[a] B    contracts the named axis a
    A ** B       requires disjoint axes
    cat[a](...)  concatenates on a

The implementation lowers contraction to ``numpy.einsum`` using a letter
per axis name.
"""

from __future__ import annotations

import string
from typing import Iterable, Sequence, Tuple

import numpy as np

LABELS = string.ascii_lowercase


class AxisMismatch(ValueError):
    pass


def _label_map(axes: Sequence[str]) -> dict:
    if len(axes) > len(LABELS):
        raise AxisMismatch(f"too many axes ({len(axes)}) for einsum lowering")
    return {ax: LABELS[i] for i, ax in enumerate(axes)}


class T:
    """A tensor with named axes."""

    __slots__ = ("axes", "data")

    def __init__(self, axes: Sequence[str], data) -> None:
        self.axes = tuple(axes)
        data = np.asarray(data, dtype=np.float64)
        if data.ndim != len(self.axes):
            data = data.reshape(self.axes_shape_hint(data))
        self.data = data

    def axes_shape_hint(self, data) -> Tuple[int, ...]:
        return tuple(data.shape[: len(self.axes)]) if len(self.axes) else ()

    # -- helpers --------------------------------------------------------
    @staticmethod
    def scalar(value: float) -> "T":
        return T((), float(value))

    def size(self, axis: str) -> int:
        return int(self.data.shape[self.axes.index(axis)])

    def shape(self) -> dict:
        return {ax: int(n) for ax, n in zip(self.axes, self.data.shape)}

    def __repr__(self) -> str:
        return f"T({self.axes}, {self.data.shape})"

    # -- broadcasting ---------------------------------------------------
    def _reordered(self, order: Sequence[str]) -> np.ndarray:
        perm = [self.axes.index(ax) for ax in order if ax in self.axes]
        arr = self.data.transpose(perm) if perm else self.data
        if len(perm) < len(order):
            shape = []
            i = 0
            for ax in order:
                if ax in self.axes:
                    shape.append(arr.shape[i])
                    i += 1
                else:
                    shape.append(1)
            arr = arr.reshape(shape)
        return arr

    @staticmethod
    def _union(a: "T", b: "T") -> Tuple[str, ...]:
        out = list(a.axes)
        for ax in b.axes:
            if ax not in out:
                out.append(ax)
        return tuple(out)

    def align(self, order: Sequence[str]) -> np.ndarray:
        return self._reordered(order)

    # -- elementwise ----------------------------------------------------
    def _ew(self, other, fn, opname: str) -> "T":
        other = _as_t(other)
        order = self._union(self, other)
        a = self._reordered(order)
        b = other._reordered(order)
        return T(order, fn(a, b))

    def __add__(self, other) -> "T":
        return self._ew(other, np.add, "+")

    def __sub__(self, other) -> "T":
        return self._ew(other, np.subtract, "-")

    def __mul__(self, other) -> "T":
        return self._ew(other, np.multiply, "*")

    def __truediv__(self, other) -> "T":
        return self._ew(other, np.divide, "/")

    def __pow__(self, other) -> "T":
        return self._ew(other, np.power, "^")

    def __neg__(self) -> "T":
        return T(self.axes, -self.data)

    def __rsub__(self, other) -> "T":
        return _as_t(other)._ew(self, np.subtract, "-")

    def __radd__(self, other) -> "T":
        return self._ew(other, np.add, "+")

    def __rmul__(self, other) -> "T":
        return self._ew(other, np.multiply, "*")

    def __rtruediv__(self, other) -> "T":
        return _as_t(other)._ew(self, np.divide, "/")

    # -- structural -----------------------------------------------------
    def contract(self, other, axis: str) -> "T":
        other = _as_t(other)
        if axis not in self.axes or axis not in other.axes:
            raise AxisMismatch(
                f"contract: axis {axis!r} must be present in both operands "
                f"({self.axes} vs {other.axes})"
            )
        out_axes = []
        for ax in self.axes + other.axes:
            if ax != axis and ax not in out_axes:
                out_axes.append(ax)
        labels = _label_map(sorted(set(self.axes + other.axes), key=(self.axes + other.axes).index))
        sub_a = "".join(labels[ax] for ax in self.axes)
        sub_b = "".join(labels[ax] for ax in other.axes)
        sub_o = "".join(labels[ax] for ax in out_axes)
        data = np.einsum(f"{sub_a},{sub_b}->{sub_o}", self.data, other.data)
        return T(out_axes, data)

    def outer(self, other) -> "T":
        other = _as_t(other)
        overlap = set(self.axes) & set(other.axes)
        if overlap:
            raise AxisMismatch(
                "outer product '**' requires disjoint axes, shared: "
                + ", ".join(sorted(overlap))
            )
        labels = _label_map(self.axes + other.axes)
        sub_a = "".join(labels[ax] for ax in self.axes)
        sub_b = "".join(labels[ax] for ax in other.axes)
        data = np.einsum(f"{sub_a},{sub_b}->{sub_a}{sub_b}", self.data, other.data)
        return T(self.axes + other.axes, data)

    def reindex(self, order: Sequence[str]) -> "T":
        order = tuple(order)
        if set(order) == set(self.axes):
            return T(order, self.data.transpose([self.axes.index(a) for a in order]))
        if len(order) == 1 and len(self.axes) == 1:
            # explicit relabel of a single axis; size is carried over
            return T(order, self.data)
        raise AxisMismatch(f"reindex to {order} from {self.axes}")

    def rename(self, old: str, new: str) -> "T":
        if old not in self.axes:
            raise AxisMismatch(f"rename: no axis {old!r} in {self.axes}")
        if new in self.axes and new != old:
            raise AxisMismatch(f"rename: axis {new!r} already present in {self.axes}")
        return T(tuple(new if ax == old else ax for ax in self.axes), self.data)

    def sum(self, axis: str) -> "T":
        idx = self.axes.index(axis)
        return T(self.axes[:idx] + self.axes[idx + 1 :], self.data.sum(axis=idx))

    def mean(self, axis: str) -> "T":
        idx = self.axes.index(axis)
        return T(self.axes[:idx] + self.axes[idx + 1 :], self.data.mean(axis=idx))


def _as_t(x) -> T:
    if isinstance(x, T):
        return x
    if np.isscalar(x):
        return T.scalar(float(x))
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 0:
        return T.scalar(float(arr))
    return T(tuple(f"i{i}" for i in range(arr.ndim)), arr)


def cat(axis: str, parts: Iterable) -> T:
    """Concatenate on ``axis``.

    Operands that do not already carry ``axis`` are broadcast over the
    common axes and then given a size-1 ``axis`` slot, so
    ``cat[in](pitch, sin(ang), ...)`` builds a feature vector from any
    mixture of scalars and vectors that share their other axes.
    """
    parts = [_as_t(p) for p in parts]
    order = []
    for p in parts:
        for ax in p.axes:
            if ax != axis and ax not in order:
                order.append(ax)
    sizes = {}
    for p in parts:
        for ax in p.axes:
            if ax != axis:
                sizes[ax] = max(sizes.get(ax, 1), p.size(ax))
    target = tuple(sizes[ax] for ax in order)

    arrays = []
    for p in parts:
        extra = [ax for ax in p.axes if ax not in order and ax != axis]
        if extra:
            raise AxisMismatch(f"cat[{axis}]: operand {p.axes} has extra axes {extra}")
        if axis in p.axes:
            arr = p.reindex([ax for ax in order if ax in p.axes] + [axis]).data
            arr = np.broadcast_to(arr, target + (p.size(axis),))
        else:
            arr = np.expand_dims(p.align([ax for ax in order if ax in p.axes]), -1)
            arr = np.broadcast_to(arr, target + (1,))
        arrays.append(arr)
    return T(tuple(order) + (axis,), np.concatenate(arrays, axis=-1))
