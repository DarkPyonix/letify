"""Step capture: the operator trace, detecting a repeated sequence and matching repetitions.

This module owns what spec "Step capture" describes on the client: the trace key of every
eager operator, deciding when the last operators repeat, recording a step, and comparing
each dispatched operator with the step's next one while a repetition runs. It does not own
queueing, templates or sending, which are ``client``, or how an operator's metadata is
computed, which is ``tensor``.
"""

from __future__ import annotations

from typing import Any

#: The fewest operators a step holds.
MIN_STEP = 8

#: The most operators a step holds, and how far back a trace key's distances reach.
MAX_STEP = 4096

#: A trace key packs the distance back and the output position into one integer.
_SHIFT = 20
_POSITION = (1 << _SHIFT) - 1


class Step:
    """A registered sequence of operators.

    ``ops`` holds ``(template, shape id, wiring, news)`` per operator, where ``wiring`` has
    one entry per tensor argument, the offset of its handle among the handles a repetition
    creates or -1 for an external one, and ``news`` has one flag per tensor output, True
    where the output is a new handle.
    """

    __slots__ = ("mutates", "news_before", "ops", "sid")

    def __init__(self, sid: int, ops: tuple):
        self.sid = sid
        self.ops = ops
        self.mutates = False
        before = [0]
        for op in ops:
            before.append(before[-1] + sum(op[3]))
        #: How many handles the operators before each position create.
        self.news_before = tuple(before)


class Tracer:
    """The trace of eager operators and the state of the running repetition."""

    def __init__(self) -> None:
        self.steps: dict[tuple, Step] = {}
        self.reset()
        #: The step being replayed, or None while dispatch is eager.
        self.active: Step | None = None
        self.size = 0
        self.pos = 0
        self.start = 0
        self.first = 0
        self.externals: list[int] = []
        self.scalars: list[Any] = []
        self.blobs: list[Any] = []

    # -- trace ----------------------------------------------------------------------

    def reset(self) -> None:
        """Forget the trace, so detection starts again from the next operator."""
        self.keys: list[tuple] = []
        self.records: list[tuple] = []
        self.last_seen: dict[tuple, int] = {}
        self.created: dict[int, int] = {}
        self.count = 0
        self.offset = 0

    def record(self, tid: int, shape_id: int, handles: tuple, outs: list) -> tuple | None:
        """Trace one eager operator. Returns ``(step, new)`` when a step was just detected."""
        n = self.count
        self.count = n + 1
        # The wiring is compared per window in ``_window``, because a distance back to an
        # input created before the window, such as a parameter, grows every repetition.
        key = (tid, shape_id)
        self.keys.append(key)
        self.records.append((tid, shape_id, handles, outs))
        if len(self.keys) > 3 * MAX_STEP:
            self._trim()

        previous = self.last_seen.get(key)
        self.last_seen[key] = n
        if previous is None:
            return None
        period = n - previous
        if period < MIN_STEP or period > MAX_STEP:
            return None
        end = n - self.offset + 1
        if end - 2 * period < 0:
            return None
        keys = self.keys
        if keys[end - period : end] != keys[end - 2 * period : end - period]:
            return None
        current = self._window(self.records[end - period : end])
        if current is None or current != self._window(
            self.records[end - 2 * period : end - period]
        ):
            return None
        return self._register(current)

    def _trim(self) -> None:
        drop = len(self.keys) - 2 * MAX_STEP
        del self.keys[:drop]
        del self.records[:drop]
        self.offset += drop
        self.last_seen = {}

    @staticmethod
    def _window(records: list[tuple]) -> tuple | None:
        """A window's operators wired against the handles the window itself creates.

        None when those handles are not one consecutive range, which replay needs.
        """
        made = [handle for record in records for handle in record[3] if handle is not None]
        first = made[0] if made else 0
        if made != list(range(first, first + len(made))):
            return None
        ops = []
        for tid, shape_id, handles, outs in records:
            wiring = tuple(h - first if made and h >= first else -1 for h in handles)
            news = tuple(handle is not None for handle in outs)
            ops.append((tid, shape_id, wiring, news))
        return tuple(ops)

    def _register(self, key: tuple) -> tuple:
        self.reset()
        step = self.steps.get(key)
        if step is not None:
            return step, False
        step = self.steps[key] = Step(len(self.steps) + 1, key)
        return step, True

    # -- replay ---------------------------------------------------------------------

    def begin(self, step: Step, first: int) -> None:
        """Expect the step's first operator, with the repetition's handles from ``first``."""
        self.active = step
        self.size = len(step.ops)
        self.pos = 0
        self.start = 0
        self.first = first
        self.externals = []
        self.scalars = []
        self.blobs = []

    def match(self, tid: int, shape_id: int, handles: tuple, scalars: list, blobs: list) -> bool:
        """Compare an operator with the step's next one, and bind it when it matches."""
        step = self.active
        assert step is not None
        expected = step.ops[self.pos]
        if expected[0] != tid or expected[1] != shape_id:
            return False
        first = self.first
        wiring = expected[2]
        externals = []
        for index, handle in enumerate(handles):
            if handle >= first:
                if wiring[index] != handle - first:
                    return False
            elif wiring[index] != -1:
                return False
            else:
                externals.append(handle)
        if externals:
            self.externals.extend(externals)
        if scalars:
            self.scalars.extend(scalars)
        if blobs:
            self.blobs.extend(blobs)
        self.pos += 1
        return True

    @property
    def unfinished(self) -> bool:
        """Whether a repetition has matched operators, sent or not, and is not complete."""
        return self.active is not None and self.pos > 0

    def take(self) -> tuple | None:
        """The matched operators not yet queued, as a step entry, or None when there are none."""
        step = self.active
        if step is None or self.pos == self.start:
            return None
        entry = (
            step.sid,
            self.first,
            self.start,
            self.pos,
            tuple(self.externals),
            tuple(self.scalars),
            tuple(self.blobs),
        )
        self.start = self.pos
        self.externals = []
        self.scalars = []
        self.blobs = []
        return entry


__all__ = ["MAX_STEP", "MIN_STEP", "Step", "Tracer"]
