"""Pipeline and LinkCache: racing strategies, choosing one, and remembering it.

Owns the choice rules of the spec's "Choosing a link" and "Link cache" sections, and the
one line it prints for each of those decisions. It does not own how a strategy connects
or what a link carries.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config.secrets import account_directory, write_secret
from ..errors import ProviderUnavailable
from . import nat
from .announce import Say, say
from .probe import Probe, ProbeResult

#: Wait after the first connect for a lower ranked strategy.
GRACE_SECONDS = 2.0
#: A strategy below this share of the fastest in either direction is rejected.
REJECT_RATIO = 0.25
#: A cached strategy is kept while its probe reaches this share of the cached throughput.
CACHE_RATIO = 0.5
#: Seconds a cached choice is trusted before the pipeline races again, so a strategy that
#: has become faster, or a cached strategy that has quietly degraded, is not stuck forever.
CACHE_TTL_SECONDS = 3600.0

MIB = 1024 * 1024
#: The default floor: a link under 10 MiB/s in either direction, or over 300 ms round trip,
#: is a failure rather than a connection. See docs/INTENT.md, "A slow link is a failure,
#: not a fallback". Accounts can override the shared policy; providers cannot choose
#: a lower default.
DEFAULT_MIN_MIB_PER_S = 10.0
DEFAULT_MAX_RTT_MS = 300.0

#: Share of the floor the collapsed direction stays under for a probe with a healthy
#: round trip to be read as a punched flow whose bulk packets are dropped.
POLICED_SHARE = 0.05
#: How many times the collapsed direction the other one reaches in that reading.
POLICED_RATIO = 10.0


class _BelowFloor(ProviderUnavailable):
    """Every connected strategy measured below the floor.

    Spec "Choosing a link": this does not end the race, because a strategy still
    connecting may clear the floor. The race raises it only once nothing is left to wait
    for, and it reaches the caller as the ``ProviderUnavailable`` it is.
    """


@dataclass(frozen=True)
class LinkFloor:
    """The slowest link an account accepts. A probe below this fails the connection."""

    min_bps: float
    max_rtt_ms: float

    @classmethod
    def default(cls) -> LinkFloor:
        return cls(min_bps=DEFAULT_MIN_MIB_PER_S * MIB, max_rtt_ms=DEFAULT_MAX_RTT_MS)

    def violations(self, result: ProbeResult) -> list[str]:
        """Each direction, or the round trip, that the probe failed to clear."""
        found = _below(
            [
                ("up", result.upload_bps, self.min_bps),
                ("down", result.download_bps, self.min_bps),
            ],
            1.0,
            "the floor",
        )
        if result.rtt_ms > self.max_rtt_ms:
            found.append(
                f"round trip {result.rtt_ms:.1f} ms above the floor {self.max_rtt_ms:.0f} ms"
            )
        if found:
            if self.policed(result):
                found.append(
                    "the round trip is healthy and one direction is tens of times the "
                    "other, which is what a network that lets a hole punched flow "
                    "establish and then drops its bulk packets looks like, not a slow "
                    "path. Both strategies that punch run into it while an ordinary "
                    "outbound connection from this machine stays fast; reverse_ssh is the "
                    "strategy that does not punch"
                )
            found.append("account overrides: min_mib_per_s and max_rtt_ms")
        return found

    def policed(self, result: ProbeResult) -> bool:
        """Whether the probe looks like a punched flow whose bulk packets are dropped.

        The measured signature is a healthy round trip with one direction tens of times
        the other: the flow establishes, answers every round trip, and then carries almost
        nothing one way. A link that is merely slow is slow for the round trip too and is
        slow both ways, so it keeps the plain rejection. ``POLICED_SHARE`` of the floor
        bounds the collapsed direction and ``POLICED_RATIO`` the gap between them.
        """
        if result.rtt_ms > self.max_rtt_ms:
            return False
        slower = min(result.upload_bps, result.download_bps)
        faster = max(result.upload_bps, result.download_bps)
        if slower >= POLICED_SHARE * self.min_bps:
            return False
        return faster >= POLICED_RATIO * slower


@dataclass(frozen=True)
class Fingerprint:
    """The network this machine is on: its public IP address and default route interface."""

    public_ip: str | None
    interface: str | None

    def matches(self, other: Fingerprint) -> bool:
        return self.public_ip is not None and self == other


def network_fingerprint(
    stun: tuple[str, int] = nat.DEFAULT_STUN, *, timeout: float = 5.0
) -> Fingerprint:
    try:
        public_ip: str | None = nat.stun_mapping(0, stun, timeout=timeout)[0]
    except (OSError, ValueError, EOFError):
        public_ip = None
    return Fingerprint(public_ip, nat.default_route_interface())


#: The cache file's format. A file carrying another value, or none, is ignored and
#: remeasured rather than read with the wrong field meanings.
CACHE_VERSION = 2


@dataclass(frozen=True)
class CachedLink:
    strategy: str
    probe: ProbeResult
    fingerprint: Fingerprint
    #: When the choice was written, ``time.time()``. Used to expire the cache.
    cached_at: float


class LinkCache:
    """``~/.letify/accounts/<alias>/link.json``."""

    def __init__(self, alias: str):
        self.alias = alias

    @property
    def path(self) -> Path:
        return account_directory(self.alias) / "link.json"

    def load(self) -> CachedLink | None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if int(data["version"]) != CACHE_VERSION:
                return None
            return CachedLink(
                strategy=str(data["strategy"]),
                probe=ProbeResult.from_dict(data["probe"]),
                fingerprint=Fingerprint(**data["fingerprint"]),
                cached_at=float(data["cached_at"]),
            )
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def save(
        self,
        strategy: str,
        probe: ProbeResult,
        fingerprint: Fingerprint,
        *,
        cached_at: float | None = None,
    ) -> None:
        body = {
            "version": CACHE_VERSION,
            "strategy": strategy,
            "probe": probe.to_dict(),
            "fingerprint": {"public_ip": fingerprint.public_ip, "interface": fingerprint.interface},
            "cached_at": cached_at if cached_at is not None else time.time(),
        }
        write_secret(self.alias, "link.json", json.dumps(body, indent=2))


def _mib(bps: float) -> str:
    return f"{bps / MIB:.1f} MiB/s"


def _numbers(result: ProbeResult) -> str:
    return (
        f"round trip {result.rtt_ms:.1f} ms, "
        f"up {_mib(result.upload_bps)}, down {_mib(result.download_bps)}"
    )


def _below(pairs: Sequence[tuple[str, float, float]], ratio: float, against: str) -> list[str]:
    """Each direction whose value is below ``ratio`` of its reference, as a readable phrase."""
    share = f"{ratio:.0%}"
    return [
        f"{direction} {_mib(value)} below {share} of {against} {_mib(reference)}"
        for direction, value, reference in pairs
        if value < ratio * reference
    ]


def _measured_on(link: Any, measured: ProbeResult | None) -> Any:
    """Record on the link what the probe measured, and return the link.

    Every path out of ``connect`` goes through this, because the placement reads
    ``rtt_ms`` to decide how many connections to use and a link with none keeps a single
    stream. Spec "Several connections at once".
    """
    if measured is not None:
        link.rtt_ms = measured.rtt_ms
        link.upload_bps = measured.upload_bps
    return link


def _close(link: Any) -> None:
    try:
        link.close()
    except Exception:
        pass


class Pipeline:
    """Race the applicable strategies and return the chosen Link, printing each decision."""

    def __init__(
        self,
        strategies: Sequence[Any],
        *,
        target: Any,
        alias: str,
        probe: Any = None,
        cache: LinkCache | None = None,
        fingerprint: Callable[[], Fingerprint] = network_fingerprint,
        grace: float = GRACE_SECONDS,
        timeout: float = 60.0,
        say: Say = say,
        previous: str | None = None,
        floor: LinkFloor | None = None,
        cache_ttl: float = CACHE_TTL_SECONDS,
    ):
        self.strategies = list(strategies)
        self.target = target
        self.alias = alias
        self.probe = probe or Probe()
        self.cache = cache
        self.fingerprint = fingerprint
        self.grace = grace
        self.timeout = timeout
        self.say = say
        #: The strategy of a closed link to this account, when this connects to it again.
        self.previous = previous
        #: The slowest probe this account accepts. A probe below it fails the connection.
        self.floor = floor or LinkFloor.default()
        self.cache_ttl = cache_ttl

    def _label(self, strategy: Any) -> str:
        """The strategy's name in a log line, with forward SSH's address and port."""
        label = getattr(strategy, "label", None)
        return label(self.target) if callable(label) else strategy.name

    def _say(self, message: str) -> None:
        self.say(f"{self.alias}: {message}")

    def connect(self) -> Any:
        link = self._connect()
        if self.previous is not None:
            self._say(f"link re-established over {link.strategy}, was {self.previous}")
        return link

    def _connect(self) -> Any:
        reasons: list[str] = []
        skipped: list[str] = []
        applicable = []
        for strategy in self.strategies:
            unmet = strategy.needs(self.target)
            if unmet:
                reasons.append(f"{strategy.name}: skipped, {unmet}")
                skipped.append(f"{strategy.name} ({unmet})")
            else:
                applicable.append(strategy)
        skipped_note = f"; skipped {', '.join(skipped)}" if skipped else ""
        if not applicable:
            self.say(f"connecting to {self.alias}: no strategy applies{skipped_note}")
            raise ProviderUnavailable("shell", self._explain(reasons))
        if len(applicable) == 1:
            strategy = applicable[0]
            self.say(
                f"connecting to {self.alias}: using {self._label(strategy)} alone, "
                f"without a race or a cache{skipped_note}"
            )
            link = strategy.assume(self.target)
            if not getattr(strategy, "probed", True):
                # The provider fallback cannot carry the probe, so there is nothing to
                # hold it to the floor with; its own slowness is the accepted cost of
                # having no better path, not a hidden one.
                return link
            try:
                measured = self._measure(link)
            except Exception:
                # Nothing to compare against, so a failed probe does not reject the only
                # strategy there is; the floor still applies when the probe itself works.
                return link
            if measured is not None:
                floored = self.floor.violations(measured)
                if floored:
                    self._say(f"rejected {link.strategy}: below the floor: {', '.join(floored)}")
                    _close(link)
                    raise ProviderUnavailable(
                        "shell",
                        self._explain(
                            [*reasons, f"{link.strategy}: below the floor: {', '.join(floored)}"]
                        ),
                    )
            return _measured_on(link, measured)

        network = self.fingerprint() if self.cache is not None else None
        if self.cache is not None and network is not None:
            kept = self._from_cache(applicable, network, reasons)
            if kept is not None:
                return kept

        raced = [self._label(s) for s in applicable if getattr(s, "probed", True)]
        held = [s.name for s in applicable if not getattr(s, "probed", True)]
        held_note = f"; {', '.join(held)} held back" if held else ""
        self.say(
            f"connecting to {self.alias}: trying {', '.join(raced) or 'nothing probed'}"
            f"{skipped_note}{held_note}"
        )
        link, measured = self._race(applicable, reasons)
        # Kept on the link so status can say how far away the machine is, and so the
        # placement knows whether splitting a blob across connections is worth it.
        _measured_on(link, measured)
        if self.cache is not None and network is not None and measured is not None:
            self.cache.save(link.strategy, measured, network)
            self._say(f"cache rewritten: {link.strategy}")
        return link

    # -- the cache -----------------------------------------------------------

    def _from_cache(self, applicable: list[Any], network: Fingerprint, reasons: list[str]) -> Any:
        entry = self.cache.load() if self.cache else None
        if entry is None:
            return None
        name = entry.strategy
        if not network.matches(entry.fingerprint):
            self._say(f"cached {name} rejected: network fingerprint changed")
            return None
        age = time.time() - entry.cached_at
        if age > self.cache_ttl:
            self._say(
                f"cached {name} rejected: stale, cached {age:.0f} s ago "
                f"(over {self.cache_ttl:.0f} s), racing again so a faster strategy is retried"
            )
            return None
        strategy = next((s for s in applicable if s.name == name), None)
        if strategy is None:
            self._say(f"cached {name} rejected: it is not applicable")
            return None
        cached = entry.probe
        self._say(
            f"trying cached {name} alone "
            f"(cached up {_mib(cached.upload_bps)}, down {_mib(cached.download_bps)})"
        )
        began = time.monotonic()
        try:
            link = strategy.attempt(self.target)
        except Exception as exc:
            self._say(f"cached {name} rejected: failed to connect: {exc}")
            return None
        self._say(f"{name} connected in {time.monotonic() - began:.1f} s")
        try:
            measured = self._measure(link)
        except Exception as exc:
            self._say(f"cached {name} rejected: probe failed: {exc}")
            _close(link)
            return None
        if measured is None:
            self._say(f"cached {name} rejected: it cannot carry the probe")
            _close(link)
            return None
        floored = self.floor.violations(measured)
        if floored:
            self._say(f"cached {name} rejected: below the floor: {', '.join(floored)}")
            reasons.append(f"{name}: below the floor: {', '.join(floored)}")
            _close(link)
            return None
        short = _below(
            [
                ("up", measured.upload_bps, cached.upload_bps),
                ("down", measured.download_bps, cached.download_bps),
            ],
            CACHE_RATIO,
            "cached",
        )
        if short:
            self._say(f"cached {name} rejected: {', '.join(short)}")
            _close(link)
            return None
        self._say(f"cached {name} accepted")
        return _measured_on(link, measured)

    # -- the race ------------------------------------------------------------

    def _race(self, applicable: list[Any], reasons: list[str]) -> tuple[Any, ProbeResult | None]:
        condition = threading.Condition()
        outcomes: dict[int, Any] = {}
        state: dict[str, Any] = {"decided": False, "first": None, "first_name": None}
        began = time.monotonic()
        #: One event per attempt, set once the choice is made so running attempts stop.
        cancels = [threading.Event() for _ in applicable]

        def run(index: int, strategy: Any) -> None:
            try:
                outcome: Any = strategy.attempt(self.target, cancel=cancels[index])
            except nat.Cancelled:
                self._say(
                    f"{strategy.name} cancelled after {time.monotonic() - began:.1f} s: "
                    f"another strategy was chosen"
                )
                return
            except Exception as exc:
                outcome = exc
            elapsed = time.monotonic() - began
            with condition:
                late = state["decided"]
                if not late:
                    outcomes[index] = outcome
                    if (
                        not isinstance(outcome, Exception)
                        and getattr(strategy, "probed", True)
                        and state["first"] is None
                    ):
                        state["first"] = time.monotonic()
                        state["first_name"] = strategy.name
                    condition.notify_all()
            if isinstance(outcome, Exception):
                self._say(f"{strategy.name} failed after {elapsed:.1f} s: {outcome}")
            elif late:
                self._say(f"{strategy.name} connected in {elapsed:.1f} s after the choice, closed")
                _close(outcome)
            else:
                self._say(f"{strategy.name} connected in {elapsed:.1f} s")

        for index, strategy in enumerate(applicable):
            threading.Thread(target=run, args=(index, strategy), daemon=True).start()

        probed = {i for i, s in enumerate(applicable) if getattr(s, "probed", True)}
        deadline = time.monotonic() + self.timeout
        held = {s.name for s in applicable if not getattr(s, "probed", True)}
        #: Strategies already offered to _choose, so a later round considers only new ones.
        seen: set[int] = set()
        rejected: _BelowFloor | None = None

        # Spec "Choosing a link": the race is not over until a link is accepted. A round
        # that ends with every connected strategy below the floor drops the grace period
        # and waits out the rest of the timeout for the strategies still connecting.
        while True:
            with condition:
                while len(outcomes) < len(applicable):
                    fresh = [
                        index
                        for index, outcome in outcomes.items()
                        if index not in seen and not isinstance(outcome, Exception)
                    ]
                    if state["first"] is None and probed <= outcomes.keys() and fresh:
                        # Every probed strategy failed, so an unprobed one is all that is left.
                        break
                    now = time.monotonic()
                    limit = deadline if state["first"] is None else state["first"] + self.grace
                    if now >= limit:
                        break
                    condition.wait(limit - now)
                settled = dict(outcomes)

            connected = []
            for index, strategy in enumerate(applicable):
                if index in seen or index not in settled:
                    continue
                seen.add(index)
                outcome = settled[index]
                if isinstance(outcome, Exception):
                    reasons.append(f"{strategy.name}: {outcome}")
                else:
                    connected.append(outcome)

            if connected:
                connected.sort(key=lambda link: link.rank)
                try:
                    chosen, measured = self._choose(connected, reasons, held)
                except _BelowFloor as below:
                    rejected = below
                else:
                    with condition:
                        state["decided"] = True
                    for index, event in enumerate(cancels):
                        if index not in seen:
                            event.set()
                    self._report_unsettled(applicable, seen, reasons)
                    first = state["first_name"]
                    if first is not None and first != chosen.strategy:
                        self._say(f"switching from {first} to {chosen.strategy}")
                    return chosen, measured

            if len(seen) == len(applicable) or time.monotonic() >= deadline:
                break
            # Nothing acceptable yet, so the strategies still connecting get the rest of
            # the timeout rather than the grace period that a rejected link started.
            with condition:
                state["first"] = None
                state["first_name"] = None

        with condition:
            state["decided"] = True
        for index, event in enumerate(cancels):
            if index not in seen:
                event.set()
        self._report_unsettled(applicable, seen, reasons)
        if rejected is not None:
            raise rejected
        self._say("no strategy connected")
        raise ProviderUnavailable("shell", self._explain(reasons))

    def _report_unsettled(self, applicable: list[Any], seen: set[int], reasons: list[str]) -> None:
        """Name every strategy that had not connected when the choice was made."""
        for index, strategy in enumerate(applicable):
            if index not in seen:
                reasons.append(f"{strategy.name}: did not connect in time")
                self._say(f"{strategy.name} timed out: not connected when the choice was made")

    def _measure(self, link: Any) -> ProbeResult | None:
        """Probe a link and print the numbers. None when the link cannot carry the probe."""
        try:
            stream = link.probe_stream()
            if stream is None:
                return None
            measured = self.probe.measure(stream)
        except Exception as exc:
            self._say(f"{link.strategy} probe failed: {exc}")
            raise
        self._say(f"{link.strategy} probe: {_numbers(measured)}")
        return measured

    def _fall_back(self, link: Any) -> None:
        self._say(f"chose {link.strategy}: no probed strategy connected")
        self._say("falling back to the provider's own path")

    def _choose(
        self, connected: list[Any], reasons: list[str], held: set[str]
    ) -> tuple[Any, ProbeResult | None]:
        """Pick among connected links. ``held`` names the strategies that cannot be probed."""
        floored_out = [reason for reason in reasons if ": below the floor:" in reason]
        if len(connected) == 1:
            link = connected[0]
            try:
                measured = self._measure(link)
            except Exception:
                measured = None
            if measured is None and floored_out:
                _close(link)
                raise _BelowFloor("shell", self._explain(reasons))
            if link.strategy in held:
                self._fall_back(link)
            else:
                if measured is not None:
                    floored = self.floor.violations(measured)
                    if floored:
                        below = f"{link.strategy}: below the floor: {', '.join(floored)}"
                        self._say(f"rejected {below}")
                        _close(link)
                        reasons.append(below)
                        raise _BelowFloor("shell", self._explain(reasons))
                self._say(f"chose {link.strategy}: only one connected")
            return link, measured

        probed: list[tuple[Any, ProbeResult]] = []
        unprobed = []
        for link in connected:
            try:
                measured = self._measure(link)
            except Exception:
                _close(link)
                continue
            if measured is None:
                unprobed.append(link)
            else:
                probed.append((link, measured))

        above_floor: list[tuple[Any, ProbeResult]] = []
        for link, measured in probed:
            floored = self.floor.violations(measured)
            if floored:
                self._say(f"rejected {link.strategy}: below the floor: {', '.join(floored)}")
                below = f"{link.strategy}: below the floor: {', '.join(floored)}"
                floored_out.append(below)
                reasons.append(below)
                _close(link)
            else:
                above_floor.append((link, measured))
        probed = above_floor

        if probed:
            fastest_up = max(result.upload_bps for _, result in probed)
            fastest_down = max(result.download_bps for _, result in probed)
            kept = []
            for link, result in probed:
                short = _below(
                    [
                        ("up", result.upload_bps, fastest_up),
                        ("down", result.download_bps, fastest_down),
                    ],
                    REJECT_RATIO,
                    "the fastest",
                )
                if short:
                    self._say(f"rejected {link.strategy}: {', '.join(short)}")
                else:
                    kept.append((link, result))
            chosen, measured = kept[0]
            if len(probed) == 1:
                self._say(f"chose {chosen.strategy}: only one connected and probed")
            else:
                self._say(f"chose {chosen.strategy}: lowest rank within 25% of the fastest")
        elif floored_out:
            for link in unprobed:
                _close(link)
            self._say("every probe was below the floor")
            raise _BelowFloor(
                "shell", self._explain([*reasons, "every probe was below the floor"])
            )
        elif unprobed:
            chosen, measured = unprobed[0], None
            self._fall_back(chosen)
        else:
            self._say("every probe failed")
            raise ProviderUnavailable("shell", self._explain([*reasons, "every probe failed"]))
        for link in connected:
            if link is not chosen:
                _close(link)
        return chosen, measured

    def _explain(self, reasons: list[str]) -> str:
        return f"no connection strategy reached {self.alias}: " + "; ".join(reasons)


__all__ = [
    "CachedLink",
    "Fingerprint",
    "LinkCache",
    "LinkFloor",
    "Pipeline",
    "network_fingerprint",
]
