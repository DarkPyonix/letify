"""The persistent letify agent and node pairing over the signalling service.

Owns the agent's state directory (its node key and the paired list), token v2, the pairing
exchange on both sides (``pair`` for the user, ``AgentService`` for the agent), and the long
running loop that gives each paired node one ``SignalAgent`` conversation. It does not own the
mailbox transport or the envelope (``linksignal.py``), the shim, signing (the sidecar), the
command line (``cli.py``) or the account written by the login (``config/login.py``). Spec
"Agent and pairing". Standard library only.
"""

from __future__ import annotations

import base64
import binascii
import contextlib
import hashlib
import hmac
import json
import os
import re
import secrets
import tempfile
import threading
import time
import tomllib
import urllib.parse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import linkd
from .linksignal import (
    DEFAULT_URL,
    Incoming,
    Refused,
    SignalAgent,
    SignalClient,
    SignalError,
    SignalWatch,
    open_envelope,
    seal,
)

#: The clear prefix of a token v2. A Tailcat token has none, so it is refused by name.
PREFIX = "letify-agent2:"
#: Seconds a pairing secret lives, wrong proofs it survives, and refusals answered per node.
SECRET_TTL_S = 900
MAX_WRONG = 5
MAX_REFUSALS = 5
#: Longest label kept for a paired node, and the shortest prefix ``unpair`` accepts.
LABEL_CHARS = 64
MIN_PREFIX = 8
#: Conversations with a running shim at once, and the seconds a conversation with no shim lives.
MAX_CONVERSATIONS = 4
IDLE_S = 120.0
#: Seconds one read of the loop waits before it checks for changes, and the pause after the
#: 1st, 2nd, ... failure in a row.
POLL_S = 5.0
BACKOFF_S = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 60.0)
#: Sids remembered per node so a finished bootstrap is never started again.
SEEN_SIDS = 64
PAIR_DOMAIN = "letify-agent pair v1"
REFUSED_TEXT = "the pairing secret is wrong, used or expired"

_HEX64 = re.compile(r"[0-9a-f]{64}")


class AgentError(Exception):
    """A failure of the agent's own state or commands."""


class TokenError(AgentError, ValueError):
    """Text that is not a usable token v2."""


class PairingRefused(AgentError):
    """The agent answered the pairing with a refusal."""


# -- the state ----------------------------------------------------------------------------


def clean_label(text: object) -> str:
    """A label of printable characters, at most ``LABEL_CHARS`` of them."""
    if not isinstance(text, str):
        return ""
    return "".join(ch for ch in text if ch.isprintable())[:LABEL_CHARS]


class PairedStore:
    """``paired.json``: the node ids the agent serves, each with a label and a time."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def mtime(self) -> float:
        try:
            return self.path.stat().st_mtime_ns
        except OSError:
            return 0.0

    def nodes(self) -> dict[str, dict[str, Any]]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as error:
            raise AgentError(f"cannot read {self.path}: {error}") from error
        nodes = data.get("nodes") if isinstance(data, dict) else None
        if not isinstance(nodes, dict):
            raise AgentError(f"{self.path} has no 'nodes' object")
        return {k: v for k, v in nodes.items() if _HEX64.fullmatch(k) and isinstance(v, dict)}

    def _write(self, nodes: dict[str, dict[str, Any]]) -> None:
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(dir=self.path.parent, prefix=".paired-")
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump({"nodes": nodes}, handle, indent=2)
            os.chmod(temporary, 0o600)
            os.replace(temporary, self.path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(temporary)
            raise

    def add(self, node: str, label: str = "", *, at: int | None = None) -> None:
        with self._lock:
            nodes = self.nodes()
            nodes[node] = {
                "label": clean_label(label),
                "at": int(time.time() if at is None else at),
            }
            self._write(nodes)

    def remove(self, node: str) -> str:
        """Remove the node with this id or unique prefix, and return its full id."""
        text = node.strip().lower()
        if len(text) < MIN_PREFIX or not re.fullmatch(r"[0-9a-f]+", text):
            raise AgentError(f"give a node id or a prefix of at least {MIN_PREFIX} hex characters")
        with self._lock:
            nodes = self.nodes()
            found = [n for n in nodes if n.startswith(text)]
            if not found:
                raise AgentError(f"no paired node starts with {text}")
            if len(found) > 1:
                raise AgentError(f"{len(found)} paired nodes start with {text}; give more of it")
            del nodes[found[0]]
            self._write(nodes)
            return found[0]


class AgentHome:
    """``~/.letify/agent/``: the agent's node key and its paired list."""

    def __init__(self, path: Path | None = None) -> None:
        if path is None:
            from ..paths import letify_root

            path = letify_root() / "agent"
        self.path = Path(path)
        self.key_path = self.path / "link.key"
        self.paired = PairedStore(self.path / "paired.json")

    def ensure(self) -> None:
        self.path.mkdir(mode=0o700, parents=True, exist_ok=True)
        with contextlib.suppress(OSError):
            os.chmod(self.path, 0o700)


def open_daemon(
    home: AgentHome, *, binary: str | os.PathLike[str] | None = None
) -> linkd.LinkDaemon:
    """Start the agent's sidecar on its persistent key, which the sidecar creates as 0600."""
    home.ensure()
    return linkd.LinkDaemon(home.key_path, binary=binary)


def _check_url(url: str, error: type[AgentError]) -> str:
    parts = urllib.parse.urlsplit(url)
    if parts.scheme == "https" and parts.hostname:
        return url.rstrip("/")
    if parts.scheme == "http" and parts.hostname in ("127.0.0.1", "localhost", "::1"):
        return url.rstrip("/")
    raise error(
        f"the signalling URL {url!r} must be https, or http for a loopback host, not "
        f"{parts.scheme or 'a bare name'}"
    )


def signal_url(home: AgentHome, flag: str | None = None) -> str:
    """The service URL: the flag, ``LETIFY_SIGNAL_URL``, ``signal`` in config.toml, the default."""
    if flag:
        return _check_url(flag, AgentError)
    env = os.environ.get("LETIFY_SIGNAL_URL")
    if env:
        return _check_url(env, AgentError)
    with contextlib.suppress(OSError, ValueError):
        value = tomllib.loads((home.path / "config.toml").read_text(encoding="utf-8")).get("signal")
        if isinstance(value, str) and value:
            return _check_url(value, AgentError)
    return DEFAULT_URL


# -- the token ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Token:
    agent: str
    secret: str
    signal: str
    version: str
    exp: int


def encode_token(*, agent: str, secret: str, signal: str, version: str, exp: int) -> str:
    raw = json.dumps(
        {
            "v": 2,
            "agent": agent,
            "secret": secret,
            "signal": signal,
            "version": version,
            "exp": exp,
        },
        separators=(",", ":"),
    ).encode()
    return PREFIX + base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _looks_like_tailcat(text: str) -> bool:
    try:
        fields = json.loads(base64.urlsafe_b64decode(text + "=" * (-len(text) % 4)))
    except (binascii.Error, ValueError):
        return False
    return isinstance(fields, dict) and "tailcat" in fields


def parse_token(text: str) -> Token:
    """Read a token v2, or raise TokenError saying what is wrong with the text."""
    text = text.strip()
    if not text.startswith(PREFIX):
        if _looks_like_tailcat(text):
            raise TokenError(
                "this is a Tailcat token from 'letify client shell connect'; this login needs "
                f"the token that 'letify agent serve' prints, which starts with {PREFIX}"
            )
        raise TokenError(
            f"that is not a token: it should start with {PREFIX} and be printed by "
            "'letify agent serve'"
        )
    body = text[len(PREFIX) :]
    try:
        fields = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
    except (binascii.Error, ValueError):
        raise TokenError(
            "that is not a token: the part after the prefix is not decodable"
        ) from None
    if not isinstance(fields, dict) or fields.get("v") != 2:
        raise TokenError("that is not a token v2")
    agent, secret, signal = fields.get("agent"), fields.get("secret"), fields.get("signal")
    version, exp = fields.get("version"), fields.get("exp")
    if not (isinstance(agent, str) and _HEX64.fullmatch(agent)):
        raise TokenError("the token's agent is not a node id")
    if not (isinstance(secret, str) and _HEX64.fullmatch(secret)):
        raise TokenError("the token's secret is not 64 hex characters")
    if not (isinstance(signal, str) and isinstance(version, str) and type(exp) is int):
        raise TokenError("the token has no signal URL, version or expiry")
    return Token(agent, secret, _check_url(signal, TokenError), version, exp)


def proof(secret: str, user: str, agent: str) -> str:
    """The HMAC that shows the user side knows ``secret``, bound to both node ids."""
    message = f"{PAIR_DOMAIN}\n{user}\n{agent}".encode()
    return hmac.new(bytes.fromhex(secret), message, hashlib.sha256).hexdigest()


# -- the user side ------------------------------------------------------------------------


def pair(
    client: SignalClient,
    token: str | Token,
    *,
    label: str = "",
    timeout_s: float = 30.0,
) -> str:
    """Pair this node with the agent of ``token`` and return the agent's node id.

    Raises TokenError for an expired token (before anything is sent), PairingRefused when the
    agent refuses, and SignalError when the service or the agent does not answer.
    """
    fields = parse_token(token) if isinstance(token, str) else token
    if fields.exp <= time.time():
        raise TokenError("the token expired; run 'letify agent serve' again for a new one")
    user = client.node_id
    sid = secrets.token_hex(16)
    line = json.dumps(
        {
            "op": "pair",
            "proof": proof(fields.secret, user, fields.agent),
            "label": clean_label(label),
        },
        separators=(",", ":"),
    )
    client.send(fields.agent, seal(client.daemon, fields.agent, sid, 1, line))
    deadline = time.monotonic() + timeout_s
    after = 0
    while time.monotonic() < deadline:
        messages, _ = client.recv(after, wait_s=min(2.0, max(0.0, deadline - time.monotonic())))
        for message in messages:
            after = max(after, message.seq)
            try:
                opened = open_envelope(
                    client.daemon, message.body, peer=fields.agent, sid=sid, last_n=0
                )
                event = json.loads(opened.line)
            except (Refused, ValueError):
                continue
            if not isinstance(event, dict):
                continue
            if event.get("event") == "paired" and event.get("agent") == fields.agent:
                return fields.agent
            if event.get("event") == "pair_refused":
                raise PairingRefused(str(event.get("error") or REFUSED_TEXT))
    raise SignalError(f"the agent {fields.agent[:12]} did not answer in {timeout_s:.0f} s")


# -- the agent ----------------------------------------------------------------------------


@dataclass
class _Secret:
    value: str
    exp: float
    wrong: int = 0


@dataclass
class _Conversation:
    sid: str
    agent: SignalAgent
    touched: float
    started: bool = False
    ended_logged: bool = False


def _short(node: str) -> str:
    return node[:12]


class AgentService:
    """The long running agent: pairing, and one conversation per paired node.

    ``serve`` returns after ``stop``. ``log`` receives one line per event and never a secret.
    """

    def __init__(
        self,
        client: SignalClient,
        home: AgentHome,
        *,
        binary: str,
        python: str,
        bootstrap: str,
        relay: str,
        version: str | None = None,
        bind: str = "0.0.0.0:0",
        worker_command: str | None = None,
        relay_command: str | None = None,
        log: Callable[[str], None] = lambda line: None,
        max_conversations: int = MAX_CONVERSATIONS,
        secret_ttl_s: float = SECRET_TTL_S,
        poll_s: float = POLL_S,
        backoff_s: Sequence[float] = BACKOFF_S,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if version is None:
            import letify

            version = letify.__version__
        self._client = client
        self.home = home
        self._version = version
        self._agent_options: dict[str, Any] = {
            "binary": binary,
            "python": python,
            "bootstrap": bootstrap,
            "relay": relay,
            "version": version,
            "bind": bind,
            "worker_command": worker_command,
            "relay_command": relay_command,
        }
        self._log = log
        self._max = max_conversations
        self._ttl = secret_ttl_s
        self._poll_s = poll_s
        self._backoff = tuple(backoff_s) or (1.0,)
        self._clock = clock
        self._stop = threading.Event()
        self._watch: SignalWatch | None = None
        self._lock = threading.Lock()
        self._secrets: list[_Secret] = []
        self._conversations: dict[str, _Conversation] = {}
        self._seen_sids: dict[str, list[str]] = {}
        self._seen_pairs: dict[tuple[str, str, int], float] = {}
        self._refusals: dict[str, list[float]] = {}
        self._known: set[str] = set()
        self._stamp: float = -1.0
        #: Shims started, and the reasons messages were dropped (the last 50).
        self.started = 0
        self.dropped: list[str] = []

    @property
    def node_id(self) -> str:
        return self._client.node_id

    def _drop(self, reason: str) -> None:
        self.dropped.append(reason)
        del self.dropped[:-50]

    def new_token(self, signal: str) -> str:
        """A token v2 with a fresh secret, valid for ``secret_ttl_s`` and one pairing."""
        secret = secrets.token_bytes(32).hex()
        exp = int(self._clock() + self._ttl)
        with self._lock:
            self._secrets.append(_Secret(secret, exp))
        return encode_token(
            agent=self.node_id, secret=secret, signal=signal, version=self._version, exp=exp
        )

    def stop(self) -> None:
        """End ``serve`` and every conversation that has not handed over."""
        self._stop.set()
        if self._watch is not None:
            self._watch.close()

    # -- the loop ---------------------------------------------------------------------------

    def serve(self) -> None:
        self._known = set(self.home.paired.nodes())
        self._stamp = self.home.paired.mtime()
        failures = 0
        watch = self._client.watch()
        self._watch = watch
        while not self._stop.is_set():
            try:
                messages = watch.read(self._poll_s)
            except SignalError as error:
                if failures == 0:
                    self._log(f"{error}; retrying with a backoff")
                delay = self._backoff[min(failures, len(self._backoff) - 1)]
                failures += 1
                self._stop.wait(delay)
                continue
            if failures:
                self._log("the service answers again")
                failures = 0
            self._refresh()
            for message in messages:
                try:
                    self._dispatch(message)
                except Exception as error:  # one bad message must not end the agent
                    self._drop(f"{type(error).__name__}: {error}")
            self._reap()
        watch.close()
        for conversation in list(self._conversations.values()):
            conversation.agent.stop()

    def _refresh(self) -> None:
        stamp = self.home.paired.mtime()
        if stamp == self._stamp:
            return
        self._stamp = stamp
        try:
            nodes = set(self.home.paired.nodes())
        except AgentError as error:
            self._drop(str(error))
            return
        for gone in self._known - nodes:
            conversation = self._conversations.pop(gone, None)
            if conversation is not None:
                conversation.agent.kill()
            self._log(f"unpaired {_short(gone)}")
        self._known = nodes

    def _reap(self) -> None:
        now = time.monotonic()
        for sender, conversation in list(self._conversations.items()):
            agent = conversation.agent
            if agent.alive or (not conversation.started and now - conversation.touched < IDLE_S):
                continue
            if conversation.started and not conversation.ended_logged:
                self._log(f"conversation {_short(sender)} ended")
            del self._conversations[sender]

    # -- one message ------------------------------------------------------------------------

    def _dispatch(self, message: Incoming) -> None:
        try:
            opened = open_envelope(
                self._client.daemon, message.body, peer=message.sender, sid=None, last_n=0
            )
            line = json.loads(opened.line)
        except (Refused, ValueError) as error:
            self._drop(str(error))
            return
        sender = opened.sender
        if isinstance(line, dict) and line.get("op") == "pair":
            self._pairing(sender, opened.sid, opened.n, line)
            return
        if sender not in self.home.paired.nodes():
            self._drop(f"a message from {_short(sender)} which is not paired")
            return
        conversation = self._conversations.get(sender)
        if conversation is None or conversation.sid != opened.sid:
            conversation = self._begin(sender, opened.sid)
            if conversation is None:
                return
        conversation.touched = time.monotonic()
        before = conversation.agent.started
        conversation.agent.handle(message)
        if conversation.agent.started > before:
            self.started += 1
            conversation.started = True
            self._log(f"conversation {_short(sender)} started")

    def _tell(self, sender: str, sid: str, event: dict[str, Any]) -> None:
        body = seal(self._client.daemon, sender, sid, 1, json.dumps(event, separators=(",", ":")))
        try:
            self._client.send(sender, body)
        except SignalError as error:
            self._drop(str(error))

    def _begin(self, sender: str, sid: str) -> _Conversation | None:
        seen = self._seen_sids.setdefault(sender, [])
        if sid in seen:
            self._drop("a message of a bootstrap that has ended or was replaced")
            return None
        previous = self._conversations.pop(sender, None)
        if previous is not None:
            previous.agent.stop()
            if previous.started:
                self._log(f"conversation {_short(sender)} ended")
        running = sum(1 for c in self._conversations.values() if c.agent.alive)
        seen.append(sid)
        del seen[:-SEEN_SIDS]
        if running >= self._max:
            self._tell(
                sender,
                sid,
                {"event": "error", "error": f"the agent is busy: {running} conversations"},
            )
            return None
        options = {k: v for k, v in self._agent_options.items() if v is not None}
        conversation = _Conversation(
            sid, SignalAgent(self._client, paired=[sender], **options), time.monotonic()
        )
        self._conversations[sender] = conversation
        return conversation

    def _pairing(self, sender: str, sid: str, n: int, line: dict[str, Any]) -> None:
        now = time.monotonic()
        for key, when in list(self._seen_pairs.items()):
            if now - when > 120:
                del self._seen_pairs[key]
        key = (sender, sid, n)
        if key in self._seen_pairs:
            self._drop(f"a pairing envelope of {_short(sender)} that was replayed")
            return
        self._seen_pairs[key] = now
        claimed = line.get("proof")
        matched: _Secret | None = None
        with self._lock:
            live = [s for s in self._secrets if s.exp > self._clock()]
            self._secrets = live
            if isinstance(claimed, str) and _HEX64.fullmatch(claimed):
                for candidate in live:
                    expected = proof(candidate.value, sender, self.node_id)
                    if hmac.compare_digest(expected, claimed) and matched is None:
                        matched = candidate
            if matched is not None:
                self._secrets.remove(matched)
            else:
                for candidate in live:
                    candidate.wrong += 1
                self._secrets = [s for s in live if s.wrong < MAX_WRONG]
        if matched is not None:
            self.home.paired.add(sender, line.get("label", ""))
            self._known.add(sender)
            self._stamp = self.home.paired.mtime()
            self._log(f"paired {_short(sender)} ({clean_label(line.get('label', ''))})")
            self._tell(
                sender,
                sid,
                {"event": "paired", "agent": self.node_id, "version": self._version},
            )
            return
        recent = [t for t in self._refusals.get(sender, []) if now - t < SECRET_TTL_S]
        self._refusals[sender] = recent
        self._log(f"refused pairing from {_short(sender)}")
        if len(recent) < MAX_REFUSALS:
            recent.append(now)
            self._tell(sender, sid, {"event": "pair_refused", "error": REFUSED_TEXT})


__all__ = [
    "PREFIX",
    "AgentError",
    "AgentHome",
    "AgentService",
    "PairedStore",
    "PairingRefused",
    "Token",
    "TokenError",
    "encode_token",
    "open_daemon",
    "pair",
    "parse_token",
    "proof",
    "signal_url",
]
