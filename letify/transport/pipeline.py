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

MIB = 1024 * 1024


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


@dataclass(frozen=True)
class CachedLink:
    strategy: str
    probe: ProbeResult
    fingerprint: Fingerprint


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
            return CachedLink(
                strategy=str(data["strategy"]),
                probe=ProbeResult.from_dict(data["probe"]),
                fingerprint=Fingerprint(**data["fingerprint"]),
            )
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def save(self, strategy: str, probe: ProbeResult, fingerprint: Fingerprint) -> None:
        body = {
            "strategy": strategy,
            "probe": probe.to_dict(),
            "fingerprint": {"public_ip": fingerprint.public_ip, "interface": fingerprint.interface},
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
            self.say(
                f"connecting to {self.alias}: using {self._label(applicable[0])} alone, "
                f"without a race{skipped_note}"
            )
            return applicable[0].assume(self.target)

        network = self.fingerprint() if self.cache is not None else None
        if self.cache is not None and network is not None:
            kept = self._from_cache(applicable, network)
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
        if self.cache is not None and network is not None and measured is not None:
            self.cache.save(link.strategy, measured, network)
            self._say(f"cache rewritten: {link.strategy}")
        return link

    # -- the cache -----------------------------------------------------------

    def _from_cache(self, applicable: list[Any], network: Fingerprint) -> Any:
        entry = self.cache.load() if self.cache else None
        if entry is None:
            return None
        name = entry.strategy
        if not network.matches(entry.fingerprint):
            self._say(f"cached {name} rejected: network fingerprint changed")
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
        return link

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
        with condition:
            while len(outcomes) < len(applicable):
                connected = [i for i, o in outcomes.items() if not isinstance(o, Exception)]
                if state["first"] is None and probed <= outcomes.keys() and connected:
                    # Every probed strategy failed, so an unprobed one is all that is left.
                    break
                now = time.monotonic()
                limit = deadline if state["first"] is None else state["first"] + self.grace
                if now >= limit:
                    break
                condition.wait(limit - now)
            state["decided"] = True
            settled = dict(outcomes)
        for index, event in enumerate(cancels):
            if index not in settled:
                event.set()

        connected = []
        for index, strategy in enumerate(applicable):
            outcome = settled.get(index)
            if outcome is None:
                reasons.append(f"{strategy.name}: did not connect in time")
                self._say(f"{strategy.name} timed out: not connected when the choice was made")
            elif isinstance(outcome, Exception):
                reasons.append(f"{strategy.name}: {outcome}")
            else:
                connected.append(outcome)
        if not connected:
            self._say("no strategy connected")
            raise ProviderUnavailable("shell", self._explain(reasons))
        connected.sort(key=lambda link: link.rank)
        held = {s.name for s in applicable if not getattr(s, "probed", True)}
        chosen, measured = self._choose(connected, reasons, held)
        first = state["first_name"]
        if first is not None and first != chosen.strategy:
            self._say(f"switching from {first} to {chosen.strategy}")
        return chosen, measured

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
        if len(connected) == 1:
            link = connected[0]
            try:
                measured = self._measure(link)
            except Exception:
                measured = None
            if link.strategy in held:
                self._fall_back(link)
            else:
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


__all__ = ["CachedLink", "Fingerprint", "LinkCache", "Pipeline", "network_fingerprint"]
