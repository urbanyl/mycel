"""Named axis algebra.

Positional dimensions cannot distinguish (B, T, H, D) from (B, H, T, D),
and that ambiguity is the single largest source of silent numerical bugs.
MYCEL names its axes and refuses implicit broadcasting.

The rules implemented here are the ones from docs/02-type-system.md:

    A * B        requires axes(A) == axes(B)
    A .*[a] B    requires a shared, all other axes aligned
    A ** B       requires axes(A) and axes(B) disjoint
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple


class AxisError(ValueError):
    """Raised on any axis-algebra violation."""


@dataclass(frozen=True)
class Axis:
    """A named axis with a symbolic or literal extent."""

    name: str
    extent: Optional[str] = None

    def __str__(self) -> str:
        return f"{self.name} {self.extent}" if self.extent else self.name


@dataclass(frozen=True)
class Axes:
    items: Tuple[Axis, ...] = ()

    # -- construction ---------------------------------------------------
    @staticmethod
    def parse(spec: str) -> "Axes":
        """Parse ``batch B, time T, head 12`` into named axes."""
        items = []
        for chunk in spec.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            parts = chunk.split()
            if len(parts) == 1:
                items.append(Axis(parts[0]))
            elif len(parts) == 2:
                items.append(Axis(parts[0], parts[1]))
            else:
                raise AxisError(f"malformed axis declaration {chunk!r}")
        names = [a.name for a in items]
        if len(set(names)) != len(names):
            dupes = sorted({n for n in names if names.count(n) > 1})
            raise AxisError("duplicate axis name(s): " + ", ".join(dupes))
        return Axes(tuple(items))

    @staticmethod
    def of(*names: str) -> "Axes":
        return Axes(tuple(Axis(n) for n in names))

    # -- queries --------------------------------------------------------
    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterable[Axis]:
        return iter(self.items)

    def __contains__(self, name: str) -> bool:
        return any(a.name == name for a in self.items)

    def names(self) -> Tuple[str, ...]:
        return tuple(a.name for a in self.items)

    def get(self, name: str) -> Axis:
        for a in self.items:
            if a.name == name:
                return a
        raise AxisError(f"no axis named {name!r} in {self}")

    def __str__(self) -> str:
        return ", ".join(str(a) for a in self.items)

    # -- algebra --------------------------------------------------------
    def elementwise(self, other: "Axes", op: str = "*") -> "Axes":
        """Elementwise ops require *identical* axis sets. No broadcasting."""
        if self.names() != other.names():
            raise AxisError(
                f"axis mismatch in '{op}': {self} vs {other}"
            )
        return self

    def contract(self, other: "Axes", axis: str) -> "Axes":
        """Contract over a named axis; every other axis must align."""
        if axis not in self or axis not in other:
            raise AxisError(
                f"cannot contract over {axis!r}: present in {self} but not {other}"
            )
        a0, b0 = self.get(axis), other.get(axis)
        if a0.extent and b0.extent and a0.extent != b0.extent:
            raise AxisError(
                f"contracted axis {axis!r} has extent {a0.extent} on the left "
                f"and {b0.extent} on the right"
            )

        left = [a for a in self.items if a.name != axis]
        right = [a for a in other.items if a.name != axis]

        # Axes present on both sides must agree; axes present on one side
        # are carried through unchanged. Contracting never invents axes.
        for a in left:
            match = next((b for b in right if b.name == a.name), None)
            if match is not None and a.extent and match.extent and a.extent != match.extent:
                raise AxisError(
                    f"axis {a.name!r} has extent {a.extent} on the left "
                    f"and {match.extent} on the right"
                )

        merged = list(left)
        existing = {a.name for a in left}
        for b in right:
            if b.name not in existing:
                merged.append(b)
                existing.add(b.name)
        return Axes(tuple(merged))

    def outer(self, other: "Axes") -> "Axes":
        """Outer product requires disjoint axis sets."""
        overlap = set(self.names()) & set(other.names())
        if overlap:
            raise AxisError(
                "outer product '**' requires disjoint axes, shared: "
                + ", ".join(sorted(overlap))
            )
        return Axes(self.items + other.items)
