"""Kaggle, one Kaggle account reached through the browser session cookie.

This module owns the cookie the account is declared with, the token chain that turns the
cookie into a live Jupyter proxy URL, the weekly accelerator quota read from the internal
API, and the channel to a worker kept alive in one kernel cell of the session letify starts.
It does not own the login, which is in ``letify.config.login``, or the kernel execution
itself, which is in ``kaggle_adapter.py``. It opens no tunnel or port forward of any kind,
and it sends no keep-alive request.

The cookie is the whole credential. Only the web session principal can mint the Jupyter
proxy token: the internal endpoints treat an API key as anonymous and answer empty. So there
is no API token, no ``kaggle.json`` and no session URL registered by hand. letify starts an
interactive session on a notebook it owns, reads the routed proxy URL through Firebase and
Firestore, and opens the channel over it.
"""

from __future__ import annotations

import base64
import binascii
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from .. import tools
from ..config import ProviderConfig
from ..config.secrets import account_directory, write_secret
from ..declare.instance import Host, Instance
from ..errors import (
    ConfigError,
    ProtocolError,
    ProviderUnavailable,
    RuntimeFailure,
    RuntimeLost,
    UnsupportedMode,
)
from ..protocol import wire
from ..runtime.channel import Connection, FramedChannel
from .base import Provider
from .usage import Usage

#: Seconds one REST request to Kaggle or to the proxy may take.
REST_TIMEOUT = 30

#: Seconds to wait for a freshly started session to publish its Jupyter proxy URL.
SESSION_START_TIMEOUT = 300.0

if TYPE_CHECKING:
    from ..runtime.channel import Channel
    from ..runtime.session import Runtime

#: Accelerators a Kaggle session can be started with, the memory of one card, and the name
#: the internal API knows the card by. Both are ones the web app offers, so a session can be
#: started on either. CPU is the empty compute, so it carries no accelerator name.
GPUS = {
    "P100": {"vram_gb": 16, "accelerator": "NVIDIA_TESLA_P100"},
    "T4": {"vram_gb": 16, "accelerator": "NVIDIA_TESLA_T4"},
}
TPUS = ("TPU_V3_8",)

#: The internal Kaggle service surface the web app uses, authenticated by the session cookie.
KAGGLE_INTERNAL = "https://www.kaggle.com/api/i/"
KERNELS_SERVICE = "kernels.KernelsService/"
USERS_SERVICE = "users.UsersService/"

#: The host that routes to a session's Jupyter server, and the language id for Python.
JUPYTER_PROXY_HOST = "https://kkb-production.jupyter-proxy.kaggle.net"
PYTHON_LANGUAGE_ID = 8

#: The Firebase exchange and the Firestore document that carries the proxy URL.
IDENTITY_TOOLKIT = "https://identitytoolkit.googleapis.com/v1/accounts:signInWithCustomToken"
FIRESTORE_BASE = "https://firestore.googleapis.com/v1/"
FIRESTORE_DOCUMENT = (
    "projects/kkb-production/databases/(default)/documents/sessions/{sid}/data/JupyterURL"
)

#: The title letify gives the notebook it owns, and the trivial body that makes a session
#: runnable. A freshly created empty notebook cancels its own session because it has nothing
#: to run, so the session is started with one committed cell.
NOTEBOOK_TITLE = "letify runtime"
NOTEBOOK_BODY = json.dumps(
    {
        "cells": [
            {
                "cell_type": "code",
                "source": "pass\n",
                "metadata": {},
                "outputs": [],
                "execution_count": None,
            }
        ],
        "metadata": {
            "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
)


def urlopen(request: urllib.request.Request, timeout: float = REST_TIMEOUT) -> Any:
    """Open one request. One seam, so a test can stand in for every live Kaggle service."""
    return urllib.request.urlopen(request, timeout=timeout)


def read_cookie(alias: str) -> str | None:
    """The browser session cookie registered for an account, or None."""
    path = account_directory(alias) / "cookie"
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8").strip() or None


def split_url(url: str) -> tuple[str, str | None]:
    """The server base, which is the URL without its query, and the ``token`` parameter.

    A routed proxy URL carries the token in its path, not a query, so the token is None and
    the base is the whole URL. A loopback test URL may still carry ``?token=``.
    """
    parts = urllib.parse.urlsplit(url)
    base = urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path.rstrip("/"), "", ""))
    token = dict(urllib.parse.parse_qsl(parts.query)).get("token")
    return base, token


#: The cookie that carries the session's expiry, an alg:none JWT with an ``exp`` claim.
CLIENT_TOKEN_COOKIE = "CLIENT-TOKEN"

#: Cookie names a usable Kaggle session must carry. The principal cookie, the CSRF token
#: and the JWT that dates the session; missing any one means the copy was partial.
REQUIRED_COOKIES = ("ka_sessionid", CLIENT_TOKEN_COOKIE, "XSRF-TOKEN")


def parse_cookie(cookie: str) -> dict[str, str]:
    """Split a ``name=value; name=value`` cookie header into a mapping.

    Only the first ``=`` separates a pair, so a base64 value ending in ``==`` survives.
    """
    jar: dict[str, str] = {}
    for part in cookie.strip().split(";"):
        part = part.strip()
        if "=" in part:
            name, value = part.split("=", 1)
            jar[name.strip()] = value.strip()
    return jar


def require_cookie_shape(cookie: str) -> dict[str, str]:
    """Return the parsed jar, or raise ``ValueError`` naming the cookies that are missing."""
    jar = parse_cookie(cookie)
    missing = [name for name in REQUIRED_COOKIES if not jar.get(name)]
    if missing:
        raise ValueError(
            "the Kaggle cookie is missing " + ", ".join(missing) + "; copy the whole cookie "
            "of a logged-in kaggle.com tab"
        )
    return jar


def _client_token_claims(cookie: str) -> dict[str, Any]:
    token = parse_cookie(cookie).get(CLIENT_TOKEN_COOKIE)
    if not token:
        raise ValueError("the Kaggle cookie has no CLIENT-TOKEN, so its expiry cannot be read")
    parts = token.split(".")
    if len(parts) < 2:
        raise ValueError("the Kaggle CLIENT-TOKEN is not a JWT")
    payload = parts[1] + "=" * (-len(parts[1]) % 4)
    try:
        claims = json.loads(base64.urlsafe_b64decode(payload))
    except (binascii.Error, ValueError):
        raise ValueError("the Kaggle CLIENT-TOKEN payload could not be decoded") from None
    if not isinstance(claims, dict):
        raise ValueError("the Kaggle CLIENT-TOKEN payload is not an object")
    return claims


def _parse_iso8601(text: str) -> datetime:
    """Parse an ISO 8601 instant, tolerating a trailing Z and over-long fractional seconds."""
    value = text.strip().replace("Z", "+00:00")
    match = re.match(r"^(.*\.\d{6})\d*([+-]\d{2}:\d{2})?$", value)
    if match:
        value = match.group(1) + (match.group(2) or "")
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).replace(microsecond=0)


def cookie_expiry(cookie: str) -> datetime:
    """When the session cookie expires, from the CLIENT-TOKEN ``exp`` claim (UTC)."""
    exp = _client_token_claims(cookie).get("exp")
    if not exp:
        raise ValueError("the Kaggle CLIENT-TOKEN has no exp claim")
    return _parse_iso8601(str(exp))


def cookie_days_left(cookie: str, now: datetime | None = None) -> float:
    """Days until the cookie expires; negative once it has."""
    moment = now or datetime.now(UTC)
    return (cookie_expiry(cookie) - moment).total_seconds() / 86400.0


def cookie_headers(cookie: str) -> dict[str, str]:
    """Headers that authenticate an internal Kaggle call as the cookie's session."""
    jar = require_cookie_shape(cookie)
    return {
        "Content-Type": "application/json",
        "cookie": cookie,
        "x-xsrf-token": jar["XSRF-TOKEN"],
        "x-kaggle-build-version": jar.get("build-hash", "1"),
    }


def verify_cookie(cookie: str) -> str:
    """Prove the cookie is a live login by reading the account, and return its display name.

    ``users.UsersService/GetCurrentUser`` answers with the account only for a real web
    session; an anonymous or stale cookie comes back empty. Raises ``ValueError`` when the
    call fails or the cookie is not accepted, so the caller can refuse the login.
    """
    request = urllib.request.Request(
        KAGGLE_INTERNAL + USERS_SERVICE + "GetCurrentUser",
        data=b"{}", method="POST", headers=cookie_headers(cookie),
    )
    try:
        with urlopen(request) as response:
            body = json.loads(response.read())
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise ValueError(f"the Kaggle cookie could not be checked: {type(exc).__name__}") from None
    name = body.get("displayName") or (body.get("user") or {}).get("displayName")
    if not name:
        raise ValueError(
            "the Kaggle cookie was refused; log in to kaggle.com and copy a fresh cookie"
        )
    return str(name)


def require_live_cookie(alias: str) -> str:
    """The cookie for a run, or a ``ConfigError`` telling the user to log in again.

    Spec "Kaggle account": a run whose cookie is missing or past its ``exp`` cannot mint the
    proxy token, so it is refused here rather than failing deeper. The message names the one
    fact letify has, the expiry, and never states the session ended for another reason.
    """
    cookie = read_cookie(alias)
    if cookie is None:
        raise ConfigError(
            f"{alias} has no Kaggle cookie. Log in to kaggle.com, copy the cookie of that "
            f"tab, and run: letify login kaggle {alias}"
        )
    try:
        require_cookie_shape(cookie)
        left = cookie_days_left(cookie)
    except ValueError as exc:
        raise ConfigError(f"{alias}: {exc}") from None
    if left <= 0:
        raise ConfigError(
            f"{alias}: the Kaggle cookie has expired. Log in to kaggle.com again and run: "
            f"letify login kaggle {alias}"
        )
    return cookie


class KaggleSessionEnded(RuntimeLost):
    """The Kaggle Jupyter Server session no longer answers."""


def _call(cookie: str, path: str, body: dict[str, Any]) -> dict[str, Any]:
    """One internal Kaggle call as the cookie's session, returning the decoded reply."""
    request = urllib.request.Request(
        KAGGLE_INTERNAL + path, data=json.dumps(body).encode(), method="POST",
        headers=cookie_headers(cookie),
    )
    try:
        with urlopen(request) as response:
            reply = json.loads(response.read())
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise RuntimeFailure(f"Kaggle {path} failed: {type(exc).__name__}") from None
    if not isinstance(reply, dict):
        raise RuntimeFailure(f"Kaggle {path} returned an unexpected reply")
    return reply


def _post(url: str, body: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
    request = urllib.request.Request(
        url, data=json.dumps(body).encode(), method="POST",
        headers={**headers, "Content-Type": "application/json"},
    )
    with urlopen(request) as response:
        return json.loads(response.read())


def notebook_id(alias: str, cookie: str) -> int:
    """The id of the notebook letify owns for this account, creating it once and reusing it.

    letify runs on a notebook it controls, not one the user picks, so it can commit the body
    a session needs to start. The id is kept in the account directory so the same notebook is
    reused rather than a new one created for every run.
    """
    path = account_directory(alias) / "notebook_id"
    if path.is_file():
        try:
            return int(path.read_text(encoding="utf-8").strip())
        except ValueError:
            pass
    reply = _call(
        cookie,
        KERNELS_SERVICE + "CreateKernelWithSettings",
        {
            "title": NOTEBOOK_TITLE,
            "kernelLanguageId": PYTHON_LANGUAGE_ID,
            "isPrivate": True,
            "sourceType": "EDITOR_TYPE_NOTEBOOK",
        },
    )
    kernel = reply.get("id")
    if kernel is None:
        raise RuntimeFailure(f"{alias}: Kaggle did not return a notebook id")
    write_secret(alias, "notebook_id", str(int(kernel)))
    return int(kernel)


def start_run(cookie: str, kernel_id: int, accelerator: str | None) -> int:
    """Start an interactive session on the notebook and return its run id.

    ``CommitAndRun`` is what the editor's Run does: it commits the notebook body and starts
    the session in one call. ``CreateKernelSession`` on an empty notebook wedges it, so this
    is the call that reliably starts a session letify controls. Empty compute is a CPU
    session; an accelerator name asks for that card.
    """
    session = _call(cookie, KERNELS_SERVICE + "GetOrCreateKernelSession", {"kernelId": kernel_id})
    sequence = (session.get("draft") or {}).get("sequence")
    compute: dict[str, Any] = {"internet": {"isEnabled": False}}
    if accelerator:
        compute["accelerator"] = accelerator
    reply = _call(
        cookie,
        KERNELS_SERVICE + "CommitAndRun",
        {
            "dataSources": [],
            "isLanguageTemplate": False,
            "newText": NOTEBOOK_BODY,
            "newTitle": NOTEBOOK_TITLE,
            "scriptId": kernel_id,
            "scriptLanguageName": "LANGUAGE_PYTHON",
            "editorType": "EDITOR_TYPE_NOTEBOOK",
            "sequence": sequence,
            "compute": compute,
            "versionName": "letify",
            "versionType": "INTERACTIVE",
            "isImport": True,
        },
    )
    run = reply.get("kernelRunId")
    if run is None:
        raise RuntimeFailure("Kaggle did not start a session run")
    return int(run)


def firebase_id_token(cookie: str) -> str:
    """Exchange the session's Firebase custom token for an id token that reads Firestore.

    The custom token is minted only for a live web principal, so an empty one means the
    cookie is not that. That is reported as a config error, because a fresh login is the fix.
    """
    config = _call(cookie, KERNELS_SERVICE + "GetFirebaseConfig", {})
    api_key = config.get("apiKey")
    auth = _call(cookie, KERNELS_SERVICE + "GetFirebaseAuthToken", {})
    custom = auth.get("authToken")
    if not api_key or not custom:
        raise ConfigError(
            "the Kaggle cookie is not a live login, so no Firebase token was minted. Log in "
            "to kaggle.com again and re-run letify login kaggle."
        )
    try:
        exchanged = _post(
            IDENTITY_TOOLKIT + "?key=" + urllib.parse.quote(api_key),
            {"token": custom, "returnSecureToken": True},
            {},
        )
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise RuntimeFailure(f"the Firebase exchange failed: {type(exc).__name__}") from None
    token = exchanged.get("idToken")
    if not token:
        raise RuntimeFailure("the Firebase exchange returned no id token")
    return str(token)


def webtier_session(cookie: str, id_token: str, run_id: int) -> str:
    """Register the Firestore auth for this run and return its web tier session id."""
    reply = _call(
        cookie,
        KERNELS_SERVICE + "UpdateUserKernelFirestoreAuth",
        {"firebaseIdToken": id_token, "kernelRunId": run_id},
    )
    sid = reply.get("sessionId")
    if not sid:
        raise RuntimeFailure("Kaggle did not register the Firestore session")
    return str(sid)


#: The proxy token in the Firestore document, either as a query parameter or in the path.
_PROXY_TOKEN = re.compile(r'token=([^"&\\]+)')
_PROXY_PATH = re.compile(r'/k/\d+/([^/"]+)/proxy')


def jupyter_token(id_token: str, webtier: str, deadline: float) -> str:
    """Poll Firestore for the session's JupyterURL and return the proxy token from it.

    A session takes a little while to publish the document after it starts, so this waits
    until the deadline rather than failing on the first empty read.
    """
    url = FIRESTORE_BASE + urllib.parse.quote(
        FIRESTORE_DOCUMENT.format(sid=webtier), safe="/()"
    )
    request = urllib.request.Request(url, headers={"Authorization": "Bearer " + id_token})
    while True:
        try:
            with urlopen(request) as response:
                document = response.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            if exc.code not in (403, 404):
                raise RuntimeFailure(f"reading the Jupyter URL failed: {exc.code}") from None
            document = ""
        except (urllib.error.URLError, OSError) as exc:
            raise RuntimeFailure(f"reading the Jupyter URL failed: {type(exc).__name__}") from None
        match = _PROXY_TOKEN.search(document) or _PROXY_PATH.search(document)
        if match:
            return match.group(1)
        if time.monotonic() >= deadline:
            raise KaggleSessionEnded(
                "the Kaggle session did not publish a Jupyter URL in time. It may have failed "
                "to start or run out of quota. Try again, or check the account's quota with "
                "letify usage."
            )
        time.sleep(3)


def live_session_url(alias: str, cookie: str, accelerator: str | None) -> tuple[int, str]:
    """Start a session and return its run id and routed Jupyter proxy URL.

    The whole token chain lives here: start the run, exchange the Firebase token, register
    the Firestore auth, read the proxy token and build the routed URL. The token rides in the
    URL path because the proxy rejects it as a header.
    """
    kernel = notebook_id(alias, cookie)
    run = start_run(cookie, kernel, accelerator)
    id_token = firebase_id_token(cookie)
    webtier = webtier_session(cookie, id_token, run)
    deadline = time.monotonic() + SESSION_START_TIMEOUT
    token = jupyter_token(id_token, webtier, deadline)
    return run, f"{JUPYTER_PROXY_HOST}/k/{run}/{token}/proxy"


def cancel_run(cookie: str, run_id: int) -> None:
    """End a session run, best effort, so its accelerator quota is released."""
    try:
        _call(cookie, KERNELS_SERVICE + "CancelKernelSession", {"kernelSessionId": run_id})
    except RuntimeFailure:
        pass


def adapter_command() -> list[str]:
    """The argument list that starts the Kaggle adapter through uv."""
    uv = tools.find_uv()
    if uv is None:
        raise ProviderUnavailable("kaggle", tools.missing_uv_message())
    return tools.script_command(tools.KAGGLE_KERNEL, uv, tools.KAGGLE_ADAPTER)


class Session:
    """The REST side of one Kaggle Jupyter Server session, through the standard library."""

    def __init__(self, alias: str, url: str):
        self.alias = alias
        self._url = url
        self.base, self.token = split_url(url)
        self.host = urllib.parse.urlsplit(url).netloc

    def _request(self, method: str, path: str, body: dict[str, Any] | None = None) -> bytes:
        query = f"?{urllib.parse.urlencode({'token': self.token})}" if self.token else ""
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(f"{self.base}{path}{query}", data=data, method=method)
        if self.token:
            request.add_header("Authorization", f"token {self.token}")
        if data is not None:
            request.add_header("Content-Type", "application/json")
        with urlopen(request) as response:
            return response.read()

    def alive(self) -> bool:
        """Whether ``/api/status`` answers 200."""
        try:
            self._request("GET", "/api/status")
        except (urllib.error.URLError, OSError, ValueError):
            return False
        return True

    def wait_alive(self, timeout: float | None = None) -> None:
        """Wait until ``/api/status`` answers, so the first kernel is not refused."""
        deadline = time.monotonic() + (SESSION_START_TIMEOUT if timeout is None else timeout)
        while not self.alive():
            if time.monotonic() >= deadline:
                raise self.ended()
            time.sleep(3)

    def ended(self) -> KaggleSessionEnded:
        """The runtime is lost, named without claiming which of the two causes it was.

        The proxy answers 404 for an ended session, for a URL that no longer routes, and for
        a session id that never existed, so a status read that is not 200 cannot tell them
        apart. Saying the session ended would state as fact something this read does not
        establish.
        """
        return KaggleSessionEnded(
            f"{self.alias}: the Kaggle Jupyter Server session did not answer. It may have "
            f"ended, since Kaggle ends a session after 20 minutes idle or at its 12 hour "
            f"limit, or the session may have failed to start. Run the function again to start "
            f"a new session."
        )

    def create_kernel(self) -> str:
        if not self.alive():
            raise self.ended()
        try:
            reply = json.loads(self._request("POST", "/api/kernels", {"name": "python3"}))
        except (urllib.error.URLError, OSError, ValueError) as exc:
            if not self.alive():
                raise self.ended() from None
            raise RuntimeFailure(
                f"{self.alias}: the Kaggle session refused a new kernel: {type(exc).__name__}"
            ) from None
        return str(reply["id"])

    def delete_kernel(self, kernel: str) -> None:
        """Best effort, because a session that already ended has no kernel to delete."""
        try:
            self._request("DELETE", f"/api/kernels/{urllib.parse.quote(kernel)}")
        except (urllib.error.URLError, OSError, ValueError):
            pass


class KaggleChannel(FramedChannel):
    """Carries the worker's frames over one kernel cell, through the adapter bridge.

    The frames are the ones that run over SSH. Only the plumbing differs: a kernel carries
    text, so the worker writes each frame as a base64 line, as the Modal sandbox already
    does. The bridge is an ordinary subprocess, so its pipes are what a write and a read
    reach, and the cell on the other side of it lives for the runtime.
    """

    text_frames = True

    #: Bytes handed to the bridge per write, matching the Modal channel.
    WRITE_LIMIT = 1 << 20

    def __init__(
        self, command: list[str], env: dict[str, str], *, name: str, session: Session
    ):
        self.command = command
        self.env = env
        self.name = name
        #: Asked whether the session is still there when the worker stops answering.
        self.session = session
        self._process: Any = None
        self._connection = None
        self._raw = bytearray()

    def start(self) -> None:
        import os
        import subprocess

        if self._connection is not None:
            return
        try:
            self._process = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
                env={**os.environ, **self.env},
            )
        except OSError as exc:
            raise RuntimeFailure(
                f"{self.name}: could not start the Kaggle bridge: {exc}",
                command=" ".join(self.command[:3]),
            ) from exc
        self._connection = Connection(
            self.name,
            self._write,
            wire.chunks_readinto(self._read_chunks),
            self._emit,
            death_detail=self._raw_text,
        )
        self._send_worker()
        self._await_ready()

    def _write(self, view: memoryview) -> int:
        piece = view[: self.WRITE_LIMIT]
        process = self._process
        assert process is not None and process.stdin is not None
        process.stdin.write(base64.b64encode(piece) + b"\n")
        process.stdin.flush()
        return piece.nbytes

    def _read_chunks(self) -> list[bytes]:
        """The next frame line, decoded. A line that is not base64 is output from before the
        worker started, such as the interpreter's own error, and is kept for the failure."""
        process = self._process
        assert process is not None and process.stdout is not None
        while True:
            line = process.stdout.readline()
            if not line:
                return []
            try:
                return [base64.b64decode(line.strip(), validate=True)]
            except (binascii.Error, ValueError):
                self._raw += line

    def request(self, payload: dict[str, Any], *, timeout: float | None = None):
        try:
            return super().request(payload, timeout=timeout)
        except ProtocolError as exc:
            raise self._verdict(exc) from exc

    def stream(self, payload: dict[str, Any], *, timeout: float | None = None):
        try:
            yield from super().stream(payload, timeout=timeout)
        except ProtocolError as exc:
            raise self._verdict(exc) from exc

    def _startup_failure(self, expired: bool, cause: Exception) -> Exception:
        """A worker that never said hello, answered by the same question as a later death."""
        return self._verdict(cause)

    def _verdict(self, cause: Exception) -> Exception:
        """Whether the session still answers, which is as much as one status read settles.

        Spec "Kaggle Jupyter Server session": the bridge exiting is the worker dying, and
        one read of the session's status is what decides how that is reported. A worker that
        died inside an answering session is this runtime's failure, and a retry would meet it
        again. A session that does not answer is a lost runtime, so a retry may start a new
        one. The read does not say why it stopped answering, so ``ended`` does not claim to.
        """
        if not self.session.alive():
            return self.session.ended()
        return RuntimeFailure(
            f"{self.name}: the Kaggle worker stopped while the session was still answering: "
            f"{cause}",
            stderr=self._raw_text(),
        )

    def _raw_text(self) -> str:
        process = self._process
        detail = bytes(self._raw[-2000:]).decode("utf-8", "replace")
        if process is not None and process.stderr is not None:
            try:
                process.stderr.flush()
            except (OSError, ValueError):
                pass
        return detail

    def _kill(self) -> None:
        if self._process is not None:
            self._process.kill()

    def close(self) -> None:
        import subprocess

        process = self._process
        self._connection = None
        if process is None:
            return
        self._process = None
        for stream in (process.stdin, process.stdout, process.stderr):
            try:
                if stream is not None:
                    stream.close()
            except OSError:
                pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


class Kaggle(Provider):
    """One Kaggle account."""

    kind = "kaggle"
    default_persistence = "ephemeral"
    has_fast_path = False
    needs_lease = False

    #: A session carries one worker for the runtime, so the object and blob tables survive.
    persistent_channel = True

    #: The kernel channel is the only link to a session, and it cannot carry a device
    #: stream: Kaggle offers no inbound port and letify dials no link out of the session.
    serves_host_local = False

    usage_unit = "GPU hours"
    usage_source = "the weekly accelerator quota the Kaggle cookie reads"

    default_workspace = "/kaggle/working/letify"

    def account_note(self) -> str | None:
        """How the account's cookie is doing, for `letify providers`."""
        cookie = read_cookie(self.alias)
        if cookie is None:
            return "no cookie; run letify login kaggle"
        try:
            left = cookie_days_left(cookie)
        except ValueError:
            return "cookie unreadable; log in again"
        if left <= 0:
            return "cookie EXPIRED; log in again"
        return f"cookie expires in {int(left)} days"

    def available(self) -> bool:
        return tools.find_uv() is not None

    def discover(self) -> Mapping[str, Instance]:
        """The fixed list of accelerators a Kaggle session offers. No call is made."""
        table: dict[str, Instance] = {"CPU": Instance(self, gpu=None)}
        table.update(
            {name: Instance(self, gpu=name, vram_gb=spec["vram_gb"]) for name, spec in GPUS.items()}
        )
        table.update({name: Instance(self, tpu=name) for name in TPUS})
        return table

    def store_backend(self) -> str:
        return "filesystem"

    def check_mode(self, instance: Instance) -> None:
        """Refuse ``host="local"``, which reaches here only through ``let.providers.any``."""
        if instance.placement is Host.local:
            raise UnsupportedMode(
                f"{self.alias} cannot serve host='local': the kernel channel is the only "
                f"link to a Kaggle session and it cannot carry a device stream. Use "
                f"host='remote'."
            )

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        #: The session, kernel, channel and run id each runtime runs its programs in.
        self._kernels: dict[str, tuple[Session, str, KaggleChannel, int]] = {}

    def open_channel(self, runtime: Runtime) -> Channel:
        """Start a session from the cookie and open one worker in one cell of it."""
        cookie = require_live_cookie(self.alias)
        instance = getattr(runtime, "instance", None)
        gpu = getattr(instance, "gpu", None)
        accelerator = GPUS[gpu]["accelerator"] if gpu in GPUS else None
        run, url = live_session_url(self.alias, cookie, accelerator)
        session = Session(self.alias, url)
        session.wait_alive()
        kernel = session.create_kernel()
        channel = KaggleChannel(
            adapter_command(),
            {"LETIFY_JUPYTER_URL": url, "LETIFY_KERNEL_ID": kernel},
            name=runtime.name,
            session=session,
        )
        self._kernels[runtime.name] = (session, kernel, channel, run)
        return channel

    def stop(self, runtime: Runtime) -> None:
        """Close the bridge, delete the kernel, then cancel the session run.

        In that order: closing the bridge's standard input ends the cell's read loop, so the
        worker exits on its own rather than being cut off mid frame. The run is letify's own,
        started for this runtime, so cancelling it releases the accelerator quota it holds.
        """
        held = self._kernels.pop(runtime.name, None)
        if held is None:
            return
        session, kernel, channel, run = held
        try:
            channel.close()
        except OSError:
            pass
        session.delete_kernel(kernel)
        cookie = read_cookie(self.alias)
        if cookie is not None:
            cancel_run(cookie, run)

    def report_usage(self) -> Usage:
        """The weekly accelerator quota, read from the cookie rather than any API key."""
        cookie = require_live_cookie(self.alias)
        stats = _call(cookie, KERNELS_SERVICE + "GetAcceleratorQuotaStatistics", {})
        gpu = stats.get("gpuQuota")
        if not isinstance(gpu, dict):
            raise RuntimeFailure(f"{self.alias}: Kaggle reported no GPU quota")
        total = _seconds_to_hours(gpu.get("totalTimeAllowed"))
        used = _seconds_to_hours(gpu.get("timeUsed"))
        remaining = None if total is None or used is None else max(total - used, 0.0)
        notes = []
        refresh = stats.get("quotaRefreshTime")
        if refresh:
            notes.append(f"resets {refresh}")
        tpu = stats.get("tpuQuota")
        if isinstance(tpu, dict):
            tpu_total = _seconds_to_hours(tpu.get("totalTimeAllowed"))
            tpu_used = _seconds_to_hours(tpu.get("timeUsed"))
            if tpu_total is not None and tpu_used is not None:
                notes.append(
                    f"TPU {tpu_used:g} h used, {max(tpu_total - tpu_used, 0.0):g} h left "
                    f"of {tpu_total:g}"
                )
        return Usage(
            alias=self.alias,
            kind=self.kind,
            unit=self.usage_unit,
            source=self.usage_source,
            remaining=remaining,
            limit=total,
            used=used,
            note="; ".join(notes) or None,
        )


def _seconds_to_hours(value: Any) -> float | None:
    """Read a duration such as ``216000s`` or ``216000`` as hours."""
    text = str(value or "").strip().removesuffix("s").strip()
    try:
        return float(text) / 3600.0
    except ValueError:
        return None


__all__ = [
    "GPUS",
    "TPUS",
    "Kaggle",
    "KaggleSessionEnded",
    "Session",
    "adapter_command",
    "cancel_run",
    "cookie_days_left",
    "cookie_expiry",
    "cookie_headers",
    "live_session_url",
    "read_cookie",
    "require_cookie_shape",
    "require_live_cookie",
    "split_url",
    "verify_cookie",
]
