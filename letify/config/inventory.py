"""What accelerators a provider account has, and how many of each.

This is the only thing that bounds how much letify runs at once. Three facts force that,
and no single number on the launcher can express any of them.

A Colab account's available accelerators depend on its tier and its compute unit balance,
so the kinds are per account and they change without letify being told.

A shared department machine holds several cards in one box, and which indices are free
moves with whoever else is logged in. So an entry names the indices it may use and letify
takes only those that are actually free when a session starts.

A run can take more than one card. On a four card machine, a run taking two is two
concurrent sessions rather than four, which a number bounding sessions cannot say.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..errors import ConfigError


def read_indices(value: Any) -> tuple[int, ...]:
    """Read declared device indices, as ``"0-3"``, ``"0-1,6-7"`` or ``[0, 1, 6]``.

    A range because that is how a shared box is described by whoever hands it out: cards
    zero through three are yours. Returned sorted and without duplicates, because the order
    a user wrote them in is not a preference letify can honour once cards are busy.
    """
    if isinstance(value, (list, tuple)):
        found = []
        for item in value:
            if isinstance(item, bool) or not isinstance(item, int):
                raise ConfigError(f"device indices must be whole numbers, not {item!r}")
            found.append(item)
        return _checked(found, value)

    if not isinstance(value, str) or not value.strip():
        raise ConfigError(
            f"device indices must be a range such as '0-3' or a list such as [0, 1], not {value!r}"
        )

    found = []
    for part in value.split(","):
        piece = part.strip()
        if not piece:
            raise ConfigError(f"device indices {value!r} has an empty entry")
        if "-" in piece:
            low, _, high = piece.partition("-")
            if not low.strip().isdigit() or not high.strip().isdigit():
                raise ConfigError(f"device indices {value!r} is not a range of numbers")
            first, last = int(low), int(high)
            if last < first:
                raise ConfigError(f"device indices {value!r} counts backwards")
            found.extend(range(first, last + 1))
        elif piece.isdigit():
            found.append(int(piece))
        else:
            raise ConfigError(f"device indices {value!r} is not a number or a range")
    return _checked(found, value)


def _checked(found: list[int], original: Any) -> tuple[int, ...]:
    if not found:
        raise ConfigError(f"device indices {original!r} names none")
    if any(index < 0 for index in found):
        raise ConfigError(f"device indices {original!r} has a negative index")
    return tuple(sorted(set(found)))


@dataclass(frozen=True, slots=True)
class Devices:
    """How many of one accelerator an account has, and which indices if it chooses them."""

    accelerator: str
    count: int = 1
    indices: tuple[int, ...] = ()

    @property
    def chooses_indices(self) -> bool:
        """Whether letify picks the physical device, rather than the provider assigning it."""
        return bool(self.indices)

    @classmethod
    def read(cls, accelerator: str, body: Any) -> Devices:
        """Read one entry of a ``devices`` table.

        ``indices`` alone gives the count. ``count`` alone is a provider that assigns the
        device itself. Neither is one of that accelerator. Both are accepted only when they
        agree, because two statements of one fact leave no way to tell which was meant.
        """
        if body is None or body is True:
            return cls(accelerator)
        if isinstance(body, int) and not isinstance(body, bool):
            return cls(accelerator, count=_positive(accelerator, body))
        if not isinstance(body, dict):
            raise ConfigError(
                f"devices.{accelerator} must be a table such as "
                f'{{ indices = "0-3" }} or {{ count = 2 }}, not {body!r}'
            )

        unknown = set(body) - {"count", "indices"}
        if unknown:
            named = ", ".join(sorted(unknown))
            raise ConfigError(
                f"devices.{accelerator} has no field {named}. It takes 'count' and 'indices'."
            )

        indices = read_indices(body["indices"]) if "indices" in body else ()
        declared = body.get("count")
        if declared is None:
            return cls(accelerator, count=len(indices) or 1, indices=indices)

        count = _positive(accelerator, declared)
        if indices and count != len(indices):
            raise ConfigError(
                f"devices.{accelerator} declares count {count} and {len(indices)} indices. "
                f"Drop the count, because the indices already say how many there are."
            )
        return cls(accelerator, count=count, indices=indices)

    def to_dict(self) -> dict[str, Any]:
        return {"count": self.count, "indices": list(self.indices)}


def _positive(accelerator: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"devices.{accelerator} count must be a whole number, not {value!r}")
    if value < 1:
        raise ConfigError(f"devices.{accelerator} count must be at least one, not {value}")
    return value


def read_table(options: dict[str, Any]) -> dict[str, Devices]:
    """Read a provider entry's inventory.

    ``devices`` is the full form. ``gpus`` is the older list, which means one of each with
    no indices chosen, and stays because an entry that only needs to name its accelerators
    should not have to write a table.
    """
    table = options.get("devices")
    if isinstance(table, dict) and table:
        return {str(name): Devices.read(str(name), body) for name, body in table.items()}

    declared = options.get("gpus")
    if isinstance(declared, (list, tuple)) and declared:
        return {str(name): Devices(str(name)) for name in declared}
    return {}


__all__ = ["Devices", "read_indices", "read_table"]
