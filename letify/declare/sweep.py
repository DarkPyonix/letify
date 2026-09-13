"""Sweep, a declared search space.

Passing a Sweep where a scalar argument is expected declares that the argument
varies. That replaces an explicit map call: the caller describes the space and
letify decides how to walk it.

``grid`` takes the Cartesian product of its axes, which is what hyperparameter
search wants. ``zip`` pairs them position by position, which is what a prepared
list of configurations wants. Two spaces can be joined with ``|``.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import product


@dataclass(frozen=True, slots=True)
class Sweep:
    """A finite set of keyword argument combinations."""

    points: tuple[dict[str, object], ...]

    def __len__(self) -> int:
        return len(self.points)

    def __iter__(self) -> Iterator[dict[str, object]]:
        return iter(self.points)

    def __or__(self, other: Sweep) -> Sweep:
        """Union of two spaces, keeping declaration order and dropping duplicates."""
        seen: list[dict[str, object]] = []
        for point in (*self.points, *other.points):
            if point not in seen:
                seen.append(point)
        return Sweep(tuple(seen))

    def with_fixed(self, **fixed: object) -> Sweep:
        """Add arguments that stay the same across every point."""
        return Sweep(tuple({**fixed, **point} for point in self.points))

    def __repr__(self) -> str:
        axes = sorted({key for point in self.points for key in point})
        return f"<Sweep {len(self.points)} points over {axes}>"


def grid(**axes: Sequence[object]) -> Sweep:
    """Cartesian product of the given axes.

    ``grid(lr=[1e-4, 3e-4], bs=[16, 32])`` is four points. A scalar value is
    treated as a single element axis, so it stays fixed across the space.
    """
    if not axes:
        return Sweep(())
    names = list(axes)
    values = [_as_sequence(axes[name]) for name in names]
    points = tuple(dict(zip(names, combo, strict=True)) for combo in product(*values))
    return Sweep(points)


def zip_(**axes: Sequence[object]) -> Sweep:
    """Pair the axes position by position.

    ``zip(lr=[1e-4, 3e-4], bs=[16, 32])`` is two points. Every axis with more
    than one value must have the same length.
    """
    if not axes:
        return Sweep(())
    names = list(axes)
    values = [_as_sequence(axes[name]) for name in names]
    lengths = {len(v) for v in values if len(v) != 1}
    if len(lengths) > 1:
        detail = ", ".join(f"{n}={len(v)}" for n, v in zip(names, values, strict=True))
        raise ValueError(f"zip axes must have equal length, got {detail}")
    size = lengths.pop() if lengths else 1
    points = tuple(
        {name: (v[i] if len(v) > 1 else v[0]) for name, v in zip(names, values, strict=True)}
        for i in range(size)
    )
    return Sweep(points)


def _as_sequence(value: object) -> tuple[object, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return (value,)
    return tuple(value)
