"""Pipeline and LinkCache: racing strategies, choosing one, and remembering it.

Owns the choice rules of the spec's "Choosing a link" and "Link cache" sections. It does
not own how a strategy connects or what a link carries.
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
from .probe import Probe, ProbeResult

#: Wait after the first connect for a lower ranked strategy.
GRACE_SECONDS = 2.0
#: A strategy below this share of the fastest in either direction is rejected.
REJECT_RATIO = 0.25
#: A cached strategy is kept while its probe reaches this share of the cached throughput.
CACHE_RATIO = 0.5


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


def _close(link: Any) -> None:
    try:
        link.close()
    except Exception:
        pass


class Pipeline:
    """Race the applicable strategies and return the chosen Link."""

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
    ):
        self.strategies = list(strategies)
        self.target = target
        self.alias = alias
        self.probe = probe or Probe()
        self.cache = cache
        self.fingerprint = fingerprint
        self.grace = grace
        self.timeout = timeout

    def connect(self) -> Any:
        reasons: list[str] = []
        applicable = []
        for strategy in self.strategies:
            unmet = strategy.needs(self.target)
            if unmet:
                reasons.append(f"{strategy.name}: skipped, {unmet}")
            else:
                applicable.append(strategy)
        if not applicable:
            raise ProviderUnavailable("shell", self._explain(reasons))
        if len(applicable) == 1:
            return applicable[0].assume(self.target)

        network = self.fingerprint() if self.cache is not None else None
        if self.cache is not None and network is not None:
            kept = self._from_cache(applicable, network)
            if kept is not None:
                return kept

        link, measured = self._race(applicable, reasons)
        if self.cache is not None and network is not None and measured is not None:
            self.cache.save(link.strategy, measured, network)
        return link

    # -- the cache -----------------------------------------------------------

    def _from_cache(self, applicable: list[Any], network: Fingerprint) -> Any:
        entry = self.cache.load() if self.cache else None
        if entry is None or not network.matches(entry.fingerprint):
            return None
        strategy = next((s for s in applicable if s.name == entry.strategy), None)
        if strategy is None:
            return None
        try:
            link = strategy.attempt(self.target)
        except Exception:
            return None
        try:
            stream = link.probe_stream()
            measured = self.probe.measure(stream) if stream is not None else None
        except Exception:
            measured = None
        if (
            measured is not None
            and measured.upload_bps >= CACHE_RATIO * entry.probe.upload_bps
            and measured.download_bps >= CACHE_RATIO * entry.probe.download_bps
        ):
            return link
        _close(link)
        return None

    # -- the race ------------------------------------------------------------

    def _race(self, applicable: list[Any], reasons: list[str]) -> tuple[Any, ProbeResult | None]:
        condition = threading.Condition()
        outcomes: dict[int, Any] = {}
        state = {"decided": False, "first": None}

        def run(index: int, strategy: Any) -> None:
            try:
                outcome: Any = strategy.attempt(self.target)
            except Exception as exc:
                outcome = exc
            with condition:
                late = state["decided"]
                if not late:
                    outcomes[index] = outcome
                    if not isinstance(outcome, Exception) and state["first"] is None:
                        state["first"] = time.monotonic()
                    condition.notify_all()
            if late and not isinstance(outcome, Exception):
                _close(outcome)

        for index, strategy in enumerate(applicable):
            threading.Thread(target=run, args=(index, strategy), daemon=True).start()

        deadline = time.monotonic() + self.timeout
        with condition:
            while len(outcomes) < len(applicable):
                now = time.monotonic()
                limit = deadline if state["first"] is None else state["first"] + self.grace
                if now >= limit:
                    break
                condition.wait(limit - now)
            state["decided"] = True
            settled = dict(outcomes)

        connected = []
        for index, strategy in enumerate(applicable):
            outcome = settled.get(index)
            if outcome is None:
                reasons.append(f"{strategy.name}: did not connect in time")
            elif isinstance(outcome, Exception):
                reasons.append(f"{strategy.name}: {outcome}")
            else:
                connected.append(outcome)
        if not connected:
            raise ProviderUnavailable("shell", self._explain(reasons))
        connected.sort(key=lambda link: link.rank)

        if len(connected) == 1:
            link = connected[0]
            try:
                stream = link.probe_stream()
                return link, self.probe.measure(stream) if stream is not None else None
            except Exception:
                return link, None

        probed: list[tuple[Any, ProbeResult]] = []
        unprobed = []
        for link in connected:
            try:
                stream = link.probe_stream()
                if stream is None:
                    unprobed.append(link)
                else:
                    probed.append((link, self.probe.measure(stream)))
            except Exception:
                _close(link)

        if probed:
            fastest_up = max(result.upload_bps for _, result in probed)
            fastest_down = max(result.download_bps for _, result in probed)
            kept = [
                (link, result)
                for link, result in probed
                if result.upload_bps >= REJECT_RATIO * fastest_up
                and result.download_bps >= REJECT_RATIO * fastest_down
            ]
            chosen, measured = kept[0]
        elif unprobed:
            chosen, measured = unprobed[0], None
        else:
            raise ProviderUnavailable("shell", self._explain([*reasons, "every probe failed"]))
        for link in connected:
            if link is not chosen:
                _close(link)
        return chosen, measured

    def _explain(self, reasons: list[str]) -> str:
        return f"no connection strategy reached {self.alias}: " + "; ".join(reasons)


__all__ = ["CachedLink", "Fingerprint", "LinkCache", "Pipeline", "network_fingerprint"]
