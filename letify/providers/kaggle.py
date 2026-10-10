"""Kaggle, one Kaggle account reached through the browser session cookie and the API token.

This module owns the cookie the account is declared with, the token chain that turns the
cookie into a live Jupyter proxy URL, the ephemeral notebook created for each run, the
weekly accelerator quota, and the channel to a worker kept alive in one kernel cell of the
session letify starts. It does not own the login, which is in ``letify.config.login``, or
the kernel execution itself, which is in ``kaggle_adapter.py``. It opens no tunnel or port
forward of any kind, and it sends no keep-alive request.

Two credentials, both required, with a boundary kept on purpose: the cookie is for the
interactive session alone, starting it, minting its Jupyter proxy URL through Firebase and
Firestore, and ending it. Only the web session principal can do any of that, since the
internal endpoints that mint the proxy token treat an API key as anonymous and answer
empty. Everything else, deleting the notebook a run created and reading the weekly quota,
goes through the official Kaggle CLI instead, authenticated by the API token in
``KAGGLE_API_TOKEN``, and never falls back to the cookie: an account declared before the token
became required is refused with a ``ConfigError`` at the point that needs it, not served
from the cookie as a substitute.
"""

from __future__ import annotations

import base64
import binascii
import json
import math
import os
import re
import textwrap
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .. import tools
from ..config import ProviderConfig
from ..config.secrets import account_directory
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
from ..runtime.channel import Connection, FramedChannel, _drain, _Tail
from .base import Provider
from .usage import Usage

#: Seconds one REST request to Kaggle or to the proxy may take.
REST_TIMEOUT = 30

#: Seconds to wait for a freshly started session to publish its Jupyter proxy URL.
SESSION_START_TIMEOUT = 300.0

#: The least time left on the cookie's exp claim that a run may start with. A run that
#: starts closer to expiry than this risks the cookie dying mid session, with no way to
#: extend it and no way for letify to tell the difference from any other lost runtime.
MIN_COOKIE_HOURS = 1.0

if TYPE_CHECKING:
    from ..runtime.channel import Channel
    from ..runtime.session import Runtime

#: GPU session shapes, with memory per card, device count and the internal API name.
#: CPU is the empty compute, so it carries no accelerator name.
GPUS = {
    "T4": {"vram_gb": 16, "devices": 2, "accelerator": "NVIDIA_TESLA_T4", "model": "T4"},
}
TPUS = ("TPU_V3_8",)
#: Cards Kaggle no longer serves, and what it does instead. Offering one would let a
#: declaration say P100 and run on something else: spec "Kaggle", the accelerators.
RETIRED = {
    "P100": (
        "Kaggle retired the Tesla P100 on 2026-09-15 and switches a notebook that asks "
        "for it to two T4s, so the card and the card count would both differ from the "
        "declaration. Use T4, which is two cards of 16 GB"
    ),
}

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
#: What the kernel bridge carries, measured sending 64 MiB on a live account. The
#: floor compares a link against this rather than refusing one that beats it.
BRIDGE_MIB_PER_S = 2.0
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

    Checks only what the cookie itself states: its shape, and its ``exp`` claim. That claim
    is not proof the cookie still works, since Kaggle has been seen to invalidate a session
    server side well before ``exp``, with the claim left unchanged; a cookie this check
    passes can still be refused by Kaggle itself. ``Kaggle._require_live_cookie`` is what
    confirms liveness online, with ``verify_cookie``, for the two paths that are about to
    make a call the cookie must actually be live for. This function on its own is for a read
    that does not reach the network, such as ``account_note``.

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
    if left * 24 < MIN_COOKIE_HOURS:
        raise ConfigError(
            f"{alias}: the Kaggle cookie expires in under {MIN_COOKIE_HOURS:.0f} hour, too "
            f"little to start and run a session. Log in to kaggle.com again and run: "
            f"letify login kaggle {alias}"
        )
    return cookie


class KaggleSessionEnded(RuntimeLost):
    """The Kaggle Jupyter Server session no longer answers."""


def _reply_message(error: urllib.error.HTTPError) -> str:
    """The ``message`` of an error reply's JSON body, or empty. Never the request's cookie."""
    try:
        body = json.loads(error.read()[:4096])
    except (OSError, ValueError):
        return ""
    message = body.get("message") if isinstance(body, dict) else None
    return message.strip()[:200] if isinstance(message, str) else ""


class _Refused(RuntimeFailure):
    """A call Kaggle answered with an HTTP error, carrying the status the caller reads."""

    def __init__(self, message: str, *, status: int):
        self.status = status
        super().__init__(message)


def _call(cookie: str, path: str, body: dict[str, Any]) -> dict[str, Any]:
    """One internal Kaggle call as the cookie's session, returning the decoded reply."""
    request = urllib.request.Request(
        KAGGLE_INTERNAL + path, data=json.dumps(body).encode(), method="POST",
        headers=cookie_headers(cookie),
    )
    try:
        with urlopen(request) as response:
            reply = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        # The status and Kaggle's own message, so a refused start says what was refused.
        detail = _reply_message(exc)
        raise _Refused(
            f"Kaggle {path} failed: HTTP {exc.code}" + (f", {detail}" if detail else ""),
            status=exc.code,
        ) from None
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


def new_notebook(alias: str, cookie: str) -> tuple[int, str | None]:
    """Create one ephemeral notebook for this run and return its id and its URL slug.

    Spec "Kaggle session token chain": a notebook is created fresh for every run and
    deleted when the run ends, so none is kept across runs and there is nothing on disk to
    go stale. The returned slug is confirmed on the live account and is what the CLI's
    ``kernels delete`` addresses the notebook by; a run on an account with no API token
    simply never deletes it through the CLI.
    """
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
    slug = reply.get("currentUrlSlug")
    return int(kernel), (str(slug) if slug else None)


def start_run(
    cookie: str, kernel_id: int, accelerator: str | None, alias: str | None = None
) -> int:
    """Start an interactive session on the notebook and return its run id.

    ``CommitAndRun`` is what the editor's Run does: it commits the notebook body and starts
    the session in one call. ``CreateKernelSession`` on an empty notebook wedges it, so this
    is the call that reliably starts a session letify controls. Empty compute is a CPU
    session; an accelerator name asks for that card.
    """
    session = _call(cookie, KERNELS_SERVICE + "GetOrCreateKernelSession", {"kernelId": kernel_id})
    sequence = (session.get("draft") or {}).get("sequence")
    # Internet on: the environment step downloads from PyPI and astral.sh. A session
    # started without it has no network at all. Spec "Kaggle Jupyter Server session".
    compute: dict[str, Any] = {"internet": {"isEnabled": True}}
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
        # Spec "Kaggle", not making sessions faster than Kaggle allows: the refusal has no
        # message of its own, so letify names the cause it can see.
        detail = refusal_detail(alias) if alias else ""
        raise RuntimeFailure(f"Kaggle did not start a session run.{detail}")
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


def live_session_url(
    alias: str, cookie: str, accelerator: str | None
) -> tuple[int, str, int, str | None]:
    """Create this run's ephemeral notebook and return its run id, proxy URL, notebook id
    and slug.

    The whole token chain lives here: create the notebook, start the run, exchange the
    Firebase token, register the Firestore auth, read the proxy token and build the routed
    URL. The token rides in the URL path because the proxy rejects it as a header. A refused
    start is raised as is: the notebook was created a moment ago for this run alone, so a
    403 on it is the account's own answer, not a stale notebook's.
    """
    kernel, slug = new_notebook(alias, cookie)
    # Spec "Kaggle", finding the notebooks letify owns: recorded before anything can fail,
    # so a run that dies next still leaves a trail to its notebook.
    record_notebook(alias, slug)
    run = start_run(cookie, kernel, accelerator, alias)
    id_token = firebase_id_token(cookie)
    webtier = webtier_session(cookie, id_token, run)
    deadline = time.monotonic() + SESSION_START_TIMEOUT
    token = jupyter_token(id_token, webtier, deadline)
    return run, f"{JUPYTER_PROXY_HOST}/k/{run}/{token}/proxy", kernel, slug


def cancel_run(cookie: str, run_id: int) -> None:
    """End a session run, best effort, so its accelerator quota is released."""
    try:
        _call(cookie, KERNELS_SERVICE + "CancelKernelSession", {"kernelSessionId": run_id})
    except RuntimeFailure:
        pass


def read_api_token(alias: str) -> str | None:
    """Read the account's API token with file whitespace stripped, or None if unavailable."""
    try:
        token = (account_directory(alias) / "access_token").read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return None
    return token or None


def require_api_token(alias: str) -> str:
    """The account's API token, or a ``ConfigError`` telling the user to log in again.

    Spec "Kaggle account": deleting the notebook a run created and reading the weekly quota
    go through the official CLI only, with no cookie-based fallback, so an account without
    the access_token file cannot do either until it logs in again.
    """
    token = read_api_token(alias)
    if token is None:
        raise ConfigError(
            f"{alias} has no Kaggle API token. Log in again with: letify login kaggle "
            f"{alias} --username <owner> --key <KGAT_token>, exactly as shown on kaggle.com"
        )
    return token


def read_notebook_owner(alias: str) -> str | None:
    """Read the public notebook owner saved by login, or None if unavailable."""
    try:
        owner = (account_directory(alias) / "username").read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return None
    return owner or None


def notebook_record(alias: str) -> Path:
    """Where the refs of the notebooks letify created on this account are kept."""
    return account_directory(alias) / "notebooks"


#: Seconds to leave between sessions on one account, a guess rather than a measurement.
#: What is measured is one refusal a minute after the previous session. Spec "Kaggle", not
#: making sessions faster than Kaggle allows.
MIN_SESSION_INTERVAL_S = 120.0


def recorded_entries(alias: str) -> list[tuple[str, int | None, float | None]]:
    """Each ref letify created, the pid that made it and when, in order.

    The pid and the time are None for a line an older letify wrote, which recorded
    neither.
    """
    try:
        text = notebook_record(alias).read_text(encoding="utf-8")
    except OSError:
        return []
    seen: list[tuple[str, int | None, float | None]] = []
    known: set[str] = set()
    for line in text.splitlines():
        fields = line.strip().split("\t")
        ref = fields[0] if fields else ""
        if not ref or ref in known:
            continue
        known.add(ref)
        owner: int | None = None
        created: float | None = None
        if len(fields) > 1 and fields[1]:
            try:
                owner = int(fields[1])
            except ValueError:
                owner = None
        if len(fields) > 2 and fields[2]:
            try:
                created = float(fields[2])
            except ValueError:
                created = None
        seen.append((ref, owner, created))
    return seen


def newest_creation(alias: str) -> float | None:
    """When the newest recorded notebook on this account was created, or None."""
    times = [created for _ref, _owner, created in recorded_entries(alias) if created is not None]
    return max(times) if times else None


def wait_before_session(alias: str, interval: float) -> float:
    """Seconds still to wait before another session may be created on this account.

    Spec "Kaggle", not making sessions faster than Kaggle allows: Kaggle refuses a run
    made too soon after the last, and only waiting recovers it.
    """
    if interval <= 0:
        return 0.0
    newest = newest_creation(alias)
    if newest is None:
        return 0.0
    return max(0.0, interval - (time.time() - newest))


def refusal_detail(alias: str) -> str:
    """What to add to a refusal, naming how long ago the previous session was made.

    This needs no threshold to be useful: it says what happened, so a reader can tell this
    cause from any other. Empty when nothing is recorded to compare against.
    """
    newest = newest_creation(alias)
    if newest is None:
        return ""
    return (
        f" The previous session on this account was created {time.time() - newest:.0f} s ago, "
        f"and Kaggle refuses a run made too soon after the last."
    )


def recorded_notebooks(alias: str) -> list[str]:
    """The refs letify believes it created, in the order it created them."""
    return [ref for ref, _owner, _created in recorded_entries(alias)]


def _process_lives(pid: int) -> bool:
    """Whether a process with this id is still running on this machine."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Someone else's process, so it is running and not letify's to judge.
        return True
    except OSError:
        return True
    return True


def orphaned_notebooks(alias: str) -> list[str]:
    """The recorded refs the account still has whose creating process is gone.

    Spec "Kaggle", stopping what a dead process left behind: a notebook a live process is
    still using is never reported, so nothing deletes a running session's notebook.
    """
    entries = recorded_entries(alias)
    if not entries:
        return []
    present = notebooks_on_account(alias)
    if present is None:
        return []
    here = set(present)
    return [
        ref
        for ref, owner, _created in entries
        if ref in here and (owner is None or not _process_lives(owner))
    ]


def record_notebook(alias: str, slug: str | None) -> None:
    """Note that letify created this notebook, so a later process can still find it.

    Spec "Kaggle", finding the notebooks letify owns: Kaggle names the notebook itself, so
    the record is the only thing that tells letify's from the user's own. Never raises; a
    notebook letify cannot record is one it may fail to clean up, not a failed run.
    """
    username = read_notebook_owner(alias)
    if not slug or username is None:
        return
    try:
        path = notebook_record(alias)
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            # The pid as well, so a live process's notebook is never taken for an orphan.
            handle.write(f"{username}/{slug}\t{os.getpid()}\t{time.time():.3f}\n")
    except OSError:
        pass


def forget_notebook(alias: str, slug: str | None) -> None:
    """Drop a notebook from the record, once it is gone from the account."""
    username = read_notebook_owner(alias)
    if not slug or username is None:
        return
    forget_ref(alias, f"{username}/{slug}")


def forget_ref(alias: str, ref: str) -> None:
    """Drop one ref from the record, keeping the pid of every other line."""
    kept = [row for row in recorded_entries(alias) if row[0] != ref]
    try:
        notebook_record(alias).write_text(
            "".join(_record_line(*row) for row in kept), encoding="utf-8"
        )
    except OSError:
        pass


def _record_line(ref: str, owner: int | None, created: float | None) -> str:
    """One record line, keeping whichever fields the entry has."""
    fields = [ref]
    if owner is not None:
        fields.append(str(owner))
        if created is not None:
            fields.append(f"{created:.3f}")
    return "\t".join(fields) + "\n"


def notebooks_on_account(alias: str) -> list[str] | None:
    """The refs the account has, through the official CLI, or None when it cannot be asked.

    None and an empty list mean different things: nothing answered, against the account
    having no notebooks. Never raises.
    """
    if read_api_token(alias) is None:
        return None
    uv = tools.find_uv()
    if uv is None:
        return None
    import csv
    import io
    import subprocess

    try:
        result = subprocess.run(
            [*tools.kaggle_cli_command(uv), "kernels", "list", "--mine", "--csv"],
            env=tools.kaggle_cli_environment(alias),
            capture_output=True,
            text=True,
            timeout=REST_TIMEOUT,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    try:
        rows = list(csv.DictReader(io.StringIO(result.stdout or "")))
    except csv.Error:
        return None
    return [(row.get("ref") or "").strip() for row in rows if (row.get("ref") or "").strip()]


def list_notebooks_via_cli(alias: str) -> list[str]:
    """The recorded notebooks this account still has. Spec "Kaggle", finding them.

    A ref the record names and the account no longer has was deleted elsewhere, so it is
    dropped from the record rather than reported.
    """
    recorded = recorded_notebooks(alias)
    if not recorded:
        return []
    present = notebooks_on_account(alias)
    if present is None:
        return []
    here = set(present)
    live = [ref for ref in recorded if ref in here]
    for ref in recorded:
        if ref not in here:
            forget_ref(alias, ref)
    return live


def delete_notebook_via_cli(alias: str, slug: str | None) -> bool:
    """Delete the ephemeral notebook through the official CLI, the only way it is deleted.

    Spec "Kaggle session token chain": the cookie is for the interactive session alone, so
    deleting the notebook afterward never touches it, win or lose. Returns whether the
    command exited 0; never raises, because the accelerator quota is already released by
    cancelling the run, and a notebook this call fails to delete is not a new failure for
    the runtime that just ended.
    """
    if not slug:
        return False
    token = read_api_token(alias)
    if token is None:
        return False
    username = read_notebook_owner(alias)
    if username is None:
        return False
    uv = tools.find_uv()
    if uv is None:
        return False
    import subprocess

    try:
        result = subprocess.run(
            [*tools.kaggle_cli_command(uv), "kernels", "delete", "-y", f"{username}/{slug}"],
            env=tools.kaggle_cli_environment(alias),
            capture_output=True,
            timeout=REST_TIMEOUT,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def delete_notebook_best_effort(alias: str, kernel_id: int, slug: str | None) -> None:
    """Delete the run's notebook through the official CLI, best effort.

    Never raises. A notebook the CLI could not delete, for want of a token, of uv, of a
    slug, or because the command itself failed, is reported with a warning naming it by
    slug or id, never by its cookie, so the user can delete it by hand on kaggle.com.
    """
    if delete_notebook_via_cli(alias, slug):
        forget_notebook(alias, slug)
        return
    import sys

    name = f"slug {slug}" if slug else f"id {kernel_id}"
    print(
        f"letify: {alias}: could not delete the Kaggle notebook ({name}); delete it by hand "
        f"on kaggle.com",
        file=sys.stderr,
    )


def run_cli_quota(alias: str) -> str | None:
    """Run ``kaggle quota --format json`` and return its standard output, or None.

    Spec "Remaining usage, Kaggle": quota is read through the official CLI only, with no
    cookie fallback, so a missing token, a missing uv, or a non-zero exit all mean there is
    nothing to report rather than a reason to ask the cookie instead.
    """
    token = read_api_token(alias)
    if token is None:
        return None
    uv = tools.find_uv()
    if uv is None:
        return None
    import subprocess

    try:
        result = subprocess.run(
            [*tools.kaggle_cli_command(uv), "quota", "--format", "json"],
            env=tools.kaggle_cli_environment(alias),
            capture_output=True,
            text=True,
            timeout=REST_TIMEOUT,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout if result.returncode == 0 else None


def _parse_cli_quota(text: str) -> dict[str, dict[str, float]] | None:
    """Parse the measured CLI list schema, returning None for unreadable quota output.

    Hour values are supplied directly. The naive refresh timestamp has no established
    timezone, so it cannot establish Unix seconds for Usage.resets_at.
    """
    try:
        rows = json.loads(text)
        if not isinstance(rows, list):
            return None
        result: dict[str, dict[str, float]] = {}
        for row in rows:
            if not isinstance(row, dict):
                return None
            resource = row.get("resource")
            if resource not in ("GPU", "TPU"):
                return None
            name = resource.lower()
            if name in result:
                return None
            datetime.fromisoformat(row["refreshAt"])
            hours = {}
            for field in ("used", "remaining", "total"):
                value = row[field]
                if not isinstance(value, str) or not value.endswith("h"):
                    return None
                number = float(value[:-1])
                if not math.isfinite(number) or number < 0:
                    return None
                hours[field] = number
            result[name] = hours
        return result if "gpu" in result else None
    except (ValueError, KeyError, TypeError):
        return None


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
        #: The bridge's standard error, drained as it arrives so the pipe never fills and
        #: the tail is there for the report when the bridge exits.
        self._stderr = _Tail()
        self._stderr_reader: Any = None

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
        import threading

        self._stderr = _Tail()
        self._stderr_reader = threading.Thread(
            target=_drain, args=(self._process.stderr, self._stderr.add), daemon=True
        )
        self._stderr_reader.start()
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
        line = memoryview(base64.b64encode(piece) + b"\n")
        # The pipe is unbuffered, and a blocking write into a full pipe returns short when
        # a signal such as SIGCHLD arrives while it waits. The line goes in until every
        # byte of it is in the pipe. Spec "Kaggle Jupyter Server session".
        while line:
            line = line[process.stdin.write(line) :]
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
        """A worker that never said hello: the watchdog's own kill is named as such, and any
        other death is answered by the same question as a later one."""
        if expired:
            from ..runtime.channel import STARTUP_TIMEOUT

            # The bridge died because letify killed it, so the shim's traceback below is the
            # effect and not the cause. Spec "Kaggle Jupyter Server session", Failure.
            return RuntimeFailure(
                f"{self.name}: the Kaggle worker did not say hello within "
                f"{STARTUP_TIMEOUT:.0f} s, so letify killed the bridge. A slow machine or "
                f"proxy needs a longer wait",
                stderr=self._raw_text(),
            )
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
        """What the bridge wrote: standard output from before the worker started, and its
        standard error, complete once the bridge has exited."""
        import subprocess

        process = self._process
        if process is not None:
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
        if self._stderr_reader is not None:
            self._stderr_reader.join(1)
        raw = bytes(self._raw[-2000:]).decode("utf-8", "replace")
        return "\n".join(part for part in (raw.strip(), self._stderr.text().strip()) if part)

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


def _capture_stdout(source: str) -> str:
    """``source`` with its standard output captured into ``__letify_value__``.

    A ``from __future__`` import has to be the first statement of the module, so those
    lines are hoisted above the wrapper rather than indented under it. Without that the
    rendezvous program, which begins with one, fails to compile and the punch reports that
    the remote half gave no answer.
    """
    lines = source.splitlines(keepends=True)
    future = [line for line in lines if line.lstrip().startswith("from __future__")]
    rest = [line for line in lines if not line.lstrip().startswith("from __future__")]
    return (
        "".join(future)
        + "import contextlib as _c, io as _io\n"
        + "__letify_buffer__ = _io.StringIO()\n"
        + "with _c.redirect_stdout(__letify_buffer__):\n"
        + textwrap.indent("".join(rest), "    ")
        + "\n__letify_value__ = __letify_buffer__.getvalue()\n"
    )


def _key_path():
    """Where letify keeps the key pair a Kaggle rendezvous authorizes."""
    from pathlib import Path as _Path

    return _Path.home() / ".ssh" / "id_letify"


def _public_key() -> str | None:
    """letify's own public key, so the rendezvous can authorize it on the machine."""
    try:
        return _key_path().with_suffix(".pub").read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def _private_key() -> str | None:
    """The private half, which the SSH command over a punched link logs in with."""
    path = _key_path()
    return str(path) if path.exists() else None


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
    usage_source = "the weekly accelerator quota the official Kaggle CLI's quota command reads"

    # Spec "Kaggle runtimes": /kaggle/working is the notebook's output directory on its
    # own 19.5 GB device, while /kaggle is the overlay with about 1 TB free.
    default_workspace = "/kaggle/letify"

    def account_note(self) -> str | None:
        """How the account's cookie is doing, for `letify providers`, with no network call.

        This is the cookie's own ``exp`` claim, not a confirmed answer: Kaggle can refuse a
        cookie before ``exp`` passes, and this note cannot tell that apart from a cookie
        that still works. ``letify usage`` makes the one call that actually checks.
        """
        cookie = read_cookie(self.alias)
        if cookie is None:
            return "no cookie; run letify login kaggle"
        try:
            left = cookie_days_left(cookie)
        except ValueError:
            return "cookie unreadable; log in again"
        if left <= 0:
            return "cookie EXPIRED; log in again"
        return f"cookie's exp claim says {int(left)} days left (not checked live; see letify usage)"

    def available(self) -> bool:
        return tools.find_uv() is not None

    def discover(self) -> Mapping[str, Instance]:
        """The fixed list of accelerators a Kaggle session offers. No call is made."""
        table: dict[str, Instance] = {"CPU": Instance(self, gpu=None)}
        table.update(
            {
                name: Instance(self, gpu=name, vram_gb=spec["vram_gb"], devices=spec["devices"])
                for name, spec in GPUS.items()
            }
        )
        table.update({name: Instance(self, tpu=name) for name in TPUS})
        return table

    discovers_sessions = True

    def sessions(self) -> list[str]:
        """The notebooks letify owns on this account, which is where an orphan shows up."""
        return list_notebooks_via_cli(self.alias)

    def _leave_a_gap(self) -> None:
        """Wait out the interval since the last session, rather than be refused.

        Spec "Kaggle", not making sessions faster than Kaggle allows: a refusal costs an
        account that answers nothing for an unknown time, and waiting costs patience.
        """
        from ..transport.announce import printer

        waiting = wait_before_session(self.alias, self.min_session_interval_s)
        if waiting <= 0:
            return
        printer(bool(self.__dict__.get("announce", True)))(
            f"{self.alias}: waiting {waiting:.0f} s before another session, because Kaggle "
            f"refuses one made too soon after the last"
        )
        time.sleep(waiting)

    def orphans(self) -> list[str]:
        """Notebooks this account still has whose creating process is gone."""
        return orphaned_notebooks(self.alias)

    def stop_orphan(self, ref: str) -> bool:
        """Delete one orphaned notebook, which removes its session with it.

        The run id died with the process that made it, so `cancel_run` cannot reach the
        run; deleting the notebook is the way in. Spec "Kaggle", stopping what a dead
        process left behind.
        """
        slug = ref.partition("/")[2] or ref
        if not delete_notebook_via_cli(self.alias, slug):
            return False
        forget_ref(self.alias, ref)
        return True

    def expected_cards(self, instance: Any) -> tuple[str, int] | None:
        """What a Kaggle session must have, because Kaggle may answer with another card."""
        gpu = getattr(instance, "gpu", None)
        spec = GPUS.get(gpu) if gpu else None
        if spec is None:
            return None
        return str(spec["model"]), int(spec["devices"])

    def retired_reason(self, name: str) -> str | None:
        """Why Kaggle no longer serves a card it once did. Spec "Kaggle", the accelerators."""
        return RETIRED.get(name.upper())

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

    #: How long a cookie confirmed live with ``verify_cookie`` is trusted before it is
    #: checked online again. Starting several runtimes in quick succession then costs one
    #: round trip, not one per runtime.
    COOKIE_LIVENESS_TTL = 60.0

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        #: The session, kernel, channel, run id, notebook id and notebook slug each runtime
        #: runs its programs in. The notebook is this runtime's own, created in
        #: ``open_channel`` and deleted in ``stop``.
        self._kernels: dict[str, tuple[Session, str, KaggleChannel, int, int, str | None]] = {}
        #: The link chosen over the bridge, per runtime, closed with the session.
        self._links: dict[str, Any] = {}
        #: Monotonic time of the last confirmed-live cookie check, or None before the first.
        self._cookie_checked_at: float | None = None

    def _require_live_cookie(self) -> str:
        """The account's cookie, confirmed live with Kaggle, not only unexpired.

        Spec "Kaggle account": the ``exp`` claim ``require_live_cookie`` checks is not proof
        the session still works, so a cookie that passes it is also checked online with
        ``verify_cookie`` here, once per ``COOKIE_LIVENESS_TTL``. A cookie Kaggle refuses is
        reported as that, separately from an expired one, because the fix the user needs to
        hear is the same either way, log in again, but stating it ended for the wrong reason
        would be stating something this check does not establish.
        """
        cookie = require_live_cookie(self.alias)
        now = time.monotonic()
        checked = self._cookie_checked_at
        if checked is not None and now - checked < self.COOKIE_LIVENESS_TTL:
            return cookie
        try:
            verify_cookie(cookie)
        except ValueError as exc:
            if "could not be checked" in str(exc):
                # The check itself did not complete, a transport problem rather than an
                # answer from Kaggle, so this is not evidence the cookie is bad: the call
                # that was about to be made reports its own failure instead.
                raise RuntimeFailure(f"{self.alias}: {exc}") from None
            raise ConfigError(
                f"{self.alias}: the Kaggle cookie has not expired, but Kaggle no longer "
                f"accepts it. Log in to kaggle.com again and run: letify login kaggle "
                f"{self.alias}"
            ) from None
        self._cookie_checked_at = now
        return cookie

    # -- an SSH link over the kernel ----------------------------------------------

    def rendezvous_over(self, channel: Any) -> Any:
        """The bridge as a rendezvous: it runs one program on the machine and reports.

        Spec "Kaggle runtimes", An SSH link over the kernel. The kernel is how letify
        reaches the machine; it does not have to be how the session is carried.
        """
        from ..transport.rendezvous import CommandRendezvous, prepare_tailcat

        bridge = channel

        class KaggleRendezvous(CommandRendezvous):
            """``eval`` on the open bridge, which is a standard library program on the VM."""

            def extras(self) -> dict[str, Any]:
                # The VM has no key of letify's yet, and no SSH server running.
                extra: dict[str, Any] = {"start_sshd": True}
                public = _public_key()
                if public:
                    extra["authorized_key"] = public
                return extra

            def run_python(self, source: str, timeout: float | None) -> str:
                # The rendezvous program prints its answer, and ``eval`` returns what the
                # source left in __letify_value__, not what it printed: the print goes out
                # as a STDOUT frame and never reaches the caller. So the source runs with
                # its standard output captured into that name.
                captured = _capture_stdout(source)
                value, _ = bridge.request(
                    {"op": "eval", "source": captured}, timeout=timeout
                )
                return value if isinstance(value, str) else str(value or "")

        rendezvous = KaggleRendezvous()
        # A Kaggle image ships no tailcat, so the request has to carry an installed path.
        rendezvous._prepare_tailcat = prepare_tailcat(
            rendezvous.run_python, self.config.option("tailcat_binary")
        )
        return rendezvous

    def target_over(self, channel: Any, *, name: str | None = None) -> Any:
        """What the strategies need, with no address because there is none to dial."""
        from ..transport import nat
        from ..transport.strategies import Target

        user = self.config.option("user")
        return Target(
            alias=self.alias,
            # Every run is a new machine whose ssh-keygen -A makes new host keys, so one
            # alias for the account would fail every session after the first.
            host_key_alias=f"letify-{self.alias}-{name}" if name else None,
            rendezvous=self.rendezvous_over(channel),
            remote_python=self.remote_python,
            stun=nat.DEFAULT_STUN,
            workspace=self.workspace_root,
            # The rendezvous authorized this key for root on the VM, so the link has to
            # offer it and name that user. Without them the VM closes the banner exchange.
            key=_private_key(),
            user=user if isinstance(user, str) and user else "root",
        )

    def strategies(self) -> list[Any]:
        """The strategies a session with no address can use, in rank order."""
        from ..transport.strategies import ProviderFallback, QuicUDP, TailcatUDP, TCPPunch

        return [TCPPunch(), TailcatUDP(), QuicUDP(), ProviderFallback(rank=6)]

    def link_over(self, channel: Any, *, name: str | None = None) -> Any:
        """Race the strategies over the bridge and return the link that wins."""
        from ..transport.announce import printer
        from ..transport.pipeline import LinkCache, Pipeline, network_fingerprint

        target = self.target_over(channel, name=name)
        return Pipeline(
            self.strategies(),
            target=target,
            alias=self.alias,
            cache=LinkCache(self.alias),
            fingerprint=lambda: network_fingerprint(target.stun),
            say=printer(self.announce),
            floor=self.link_floor,
        ).connect()

    @property
    def link_floor(self) -> Any:
        """The slowest probe this account accepts, as a shell account's does."""
        from ..transport.pipeline import MIB, LinkFloor

        default = LinkFloor.default()
        min_mib = self.config.option("min_mib_per_s")
        max_rtt = self.config.option("max_rtt_ms")
        return LinkFloor(
            min_bps=float(min_mib) * MIB if isinstance(min_mib, (int, float)) else default.min_bps,
            max_rtt_ms=float(max_rtt) if isinstance(max_rtt, (int, float)) else default.max_rtt_ms,
            fallback_bps=self.fallback_mib_per_s * MIB,
        )

    @property
    def remote_python(self) -> str:
        value = self.config.option("remote_python")
        return str(value) if value else "python3"

    def channel_over(self, bridge: Any, *, name: str) -> Channel:
        """A worker over the chosen link, or the bridge when nothing was chosen.

        Spec "Kaggle runtimes", An SSH link over the kernel: a network that cannot punch
        keeps the bridge, which is what every Kaggle session used until now.
        """
        import shlex as _shlex

        from ..protocol.worker import BOOTSTRAP
        from ..runtime.channel import PersistentChannel

        try:
            link = self.link_over(bridge, name=name)
        except Exception as exc:
            # Never silent: a session that quietly changed channel is one nobody can
            # account for. Spec "A floor rejection never picks something slower".
            from ..transport.announce import printer

            printer(bool(self.__dict__.get("announce", True)))(
                f"{self.alias}: the kernel bridge is carrying {name} at about "
                f"{self.fallback_mib_per_s:.1f} MiB/s, because no link was chosen: {exc}"
            )
            return bridge
        self._links[name] = link
        return PersistentChannel(
            link.ssh_command(f"{self.remote_python} -u -c {_shlex.quote(BOOTSTRAP)}"),
            name=name,
        )

    @property
    def min_session_interval_s(self) -> float:
        """Seconds to leave between sessions on this account, from the account or the default."""
        value = self.config.option("min_session_interval_s")
        if isinstance(value, (int, float)):
            return max(0.0, float(value))
        return MIN_SESSION_INTERVAL_S

    @property
    def fallback_mib_per_s(self) -> float:
        """The kernel bridge, measured at 2.0 MiB/s sending 64 MiB on a live account.

        Spec "A floor rejection never picks something slower": without this a punched link
        at down 8.1 MiB/s was refused for being under the 10 MiB/s floor and 26 GB went
        over the 2 MiB/s bridge instead.
        """
        value = self.config.option("fallback_mib_per_s")
        return max(0.0, float(value)) if isinstance(value, (int, float)) else BRIDGE_MIB_PER_S

    @property
    def transfer_connection_cost_s(self) -> float:
        """A punch, which is what one more stream costs here. Measured at about 10 s."""
        value = self.config.option("transfer_connection_cost_s")
        return max(0.0, float(value)) if isinstance(value, (int, float)) else 10.0

    def link(self, runtime: Runtime | None = None) -> Any:
        """The link carrying this runtime, or None when the bridge is carrying it.

        The placement reads ``rtt_ms`` from it to decide how many streams to use.
        """
        name = getattr(runtime, "name", None)
        return self._links.get(name) if name else None

    def transfer_channel(self, runtime: Runtime | None) -> Channel:
        """One more stream for a placement, on a connection of its own.

        Spec "Several connections at once": the link is asked for its own SSH command,
        which lands on a local forwarding port of its own and so punches again. Reusing
        the session's command would multiplex onto the one connection instead, and
        multiplexed channels do not add up.
        """
        import shlex as _shlex

        from ..protocol.worker import BOOTSTRAP
        from ..runtime.channel import PersistentChannel

        name = getattr(runtime, "name", None)
        link = self._links.get(name) if name else None
        if link is None:
            raise UnsupportedMode(
                f"{self.alias} is carried by the kernel bridge for {name}, which is one "
                f"connection and offers no second stream. A punched session does."
            )
        return PersistentChannel(
            link.ssh_command(f"{self.remote_python} -u -c {_shlex.quote(BOOTSTRAP)}"),
            name=f"{self.alias}-transfer",
        )

    def open_channel(self, runtime: Runtime) -> Channel:
        """Start a session from the cookie and open one worker in one cell of it.

        The notebook is created fresh for this runtime alone, never reused from an earlier
        one: spec "Kaggle session token chain".
        """
        if read_notebook_owner(self.alias) is None:
            raise ConfigError(
                f"{self.alias} has no Kaggle notebook owner. Log in again with: "
                f"letify login kaggle {self.alias} --username <owner>"
            )
        cookie = self._require_live_cookie()
        self._leave_a_gap()
        instance = getattr(runtime, "instance", None)
        gpu = getattr(instance, "gpu", None)
        accelerator = GPUS[gpu]["accelerator"] if gpu in GPUS else getattr(instance, "tpu", None)
        run, url, notebook, slug = live_session_url(self.alias, cookie, accelerator)
        session = Session(self.alias, url)
        session.wait_alive()
        kernel = session.create_kernel()
        channel = KaggleChannel(
            adapter_command(),
            {"LETIFY_JUPYTER_URL": url, "LETIFY_KERNEL_ID": kernel},
            name=runtime.name,
            session=session,
        )
        self._kernels[runtime.name] = (session, kernel, channel, run, notebook, slug)
        # Spec "Kaggle runtimes", An SSH link over the kernel: the bridge is the
        # rendezvous, and the session rides a link when one is chosen over it.
        return self.channel_over(channel, name=runtime.name)

    def stop(self, runtime: Runtime) -> None:
        """Close the bridge, delete the kernel, cancel the session run, delete the notebook.

        In that order: closing the bridge's standard input ends the cell's read loop, so the
        worker exits on its own rather than being cut off mid frame. The run is letify's own,
        started for this runtime, so cancelling it releases the accelerator quota it holds;
        that is the cookie's last job, ending the session it started. The notebook created
        for this run is then deleted, through the official CLI alone, with no cookie
        involved at all. Spec "Kaggle session token chain".
        """
        held = self._kernels.pop(runtime.name, None)
        if held is None:
            return
        session, kernel, channel, run, notebook, slug = held
        link = self._links.pop(runtime.name, None)
        if link is not None:
            try:
                link.close()
            except Exception:
                pass
        try:
            channel.close()
        except OSError:
            pass
        session.delete_kernel(kernel)
        cookie = read_cookie(self.alias)
        if cookie is not None:
            cancel_run(cookie, run)
        delete_notebook_best_effort(self.alias, notebook, slug)

    def report_usage(self) -> Usage:
        """The weekly accelerator quota, read through the official CLI alone.

        Spec "Remaining usage, Kaggle": this never touches the cookie, which is reserved
        for the interactive session and its Jupyter proxy URL. ``require_api_token`` raises
        when the account has no readable access_token file.
        """
        require_api_token(self.alias)
        text = run_cli_quota(self.alias)
        parsed = _parse_cli_quota(text) if text is not None else None
        if parsed is None:
            return Usage(
                alias=self.alias,
                kind=self.kind,
                unit=self.usage_unit,
                source=self.usage_source,
                remaining=None,
                note="the Kaggle CLI's quota output could not be read",
            )
        gpu = parsed["gpu"]
        total, used = gpu["total"], gpu["used"]
        notes = []
        tpu = parsed.get("tpu")
        if tpu is not None:
            notes.append(
                f"TPU {tpu['used']:g} h used, {tpu['remaining']:g} h "
                f"left of {tpu['total']:g}"
            )
        return Usage(
            alias=self.alias,
            kind=self.kind,
            unit=self.usage_unit,
            source=self.usage_source,
            remaining=gpu["remaining"],
            limit=total,
            used=used,
            note="; ".join(notes) or None,
            resources=(
                ({
                    "name": "TPU",
                    "unit": "TPU hours",
                    "remaining": tpu["remaining"],
                    "used": tpu["used"],
                    "limit": tpu["total"],
                    "resets_at": None,
                },)
                if tpu is not None else ()
            ),
        )


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
    "delete_notebook_best_effort",
    "delete_notebook_via_cli",
    "live_session_url",
    "read_api_token",
    "read_cookie",
    "require_api_token",
    "require_cookie_shape",
    "require_live_cookie",
    "run_cli_quota",
    "split_url",
    "verify_cookie",
]
