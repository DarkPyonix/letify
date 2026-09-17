"""An access token borrowed from the user's existing Google login.

Owns finding a login on this machine and turning it into a short-lived OAuth access token,
with the standard library only. It does not own what the token is used for, and it never
writes a credential anywhere.

The lookup order is the one in the spec section "Google login for gcs": the
``GOOGLE_OAUTH_ACCESS_TOKEN`` variable, then Application Default Credentials refreshed over
HTTP, then ``gcloud auth print-access-token``.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from ...errors import ProviderUnavailable, RuntimeFailure

#: Where Google exchanges a refresh token when the credentials file names no other place.
DEFAULT_TOKEN_URI = "https://oauth2.googleapis.com/token"

#: A token is replaced this many seconds before Google says it expires.
EXPIRY_MARGIN_S = 60.0

#: How long a token from a source that states no lifetime is assumed to live.
ASSUMED_LIFETIME_S = 1800.0


def credentials_path() -> Path:
    """The Application Default Credentials file this machine would use."""
    named = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if named:
        return Path(named).expanduser()
    configured = os.environ.get("CLOUDSDK_CONFIG")
    if configured:
        base = Path(configured).expanduser()
    elif sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", str(Path.home()))) / "gcloud"
    else:
        base = Path.home() / ".config" / "gcloud"
    return base / "application_default_credentials.json"


class TokenSource:
    """Hands out an access token, reusing it until shortly before it expires."""

    def __init__(self) -> None:
        self._token: str | None = None
        self._expires = 0.0
        self._lock = threading.Lock()

    def token(self) -> str:
        with self._lock:
            if self._token is None or time.time() >= self._expires - EXPIRY_MARGIN_S:
                self._token, lifetime = self._fetch()
                self._expires = time.time() + lifetime
            return self._token

    def _fetch(self) -> tuple[str, float]:
        given = os.environ.get("GOOGLE_OAUTH_ACCESS_TOKEN")
        if given:
            # The caller manages this token's lifetime, so it is read again next time.
            return given.strip(), EXPIRY_MARGIN_S
        path = credentials_path()
        if path.is_file():
            return self._from_credentials(path)
        gcloud = shutil.which("gcloud")
        if gcloud:
            return self._from_gcloud(gcloud)
        raise ProviderUnavailable(
            "gcs",
            "no Google login was found. Run 'gcloud auth application-default login', "
            "or set GOOGLE_OAUTH_ACCESS_TOKEN",
        )

    def _from_credentials(self, path: Path) -> tuple[str, float]:
        try:
            info = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ProviderUnavailable("gcs", f"{path} could not be read: {exc}") from exc
        kind = info.get("type")
        if kind != "authorized_user":
            raise ProviderUnavailable(
                "gcs",
                f"{path} holds a {kind!r} credential. Exchanging a service account key "
                f"needs an RSA signature the standard library cannot make; run "
                f"'gcloud auth activate-service-account' so gcloud can mint the token",
            )
        form = urllib.parse.urlencode(
            {
                "grant_type": "refresh_token",
                "client_id": info.get("client_id", ""),
                "client_secret": info.get("client_secret", ""),
                "refresh_token": info.get("refresh_token", ""),
            }
        ).encode()
        uri = str(info.get("token_uri") or DEFAULT_TOKEN_URI)
        request = urllib.request.Request(uri, data=form, method="POST")
        request.add_header("Content-Type", "application/x-www-form-urlencoded")
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                reply = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read()[:500].decode(errors="replace")
            raise RuntimeFailure(
                f"refreshing the Google login in {path} returned {exc.code}: {detail}"
            ) from exc
        except (urllib.error.URLError, OSError) as exc:
            raise RuntimeFailure(f"refreshing the Google login in {path} failed: {exc}") from exc
        return str(reply["access_token"]), float(reply.get("expires_in", ASSUMED_LIFETIME_S))

    def _from_gcloud(self, gcloud: str) -> tuple[str, float]:
        result = subprocess.run(
            [gcloud, "auth", "print-access-token"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        token = (result.stdout or "").strip()
        if result.returncode != 0 or not token:
            raise ProviderUnavailable(
                "gcs",
                "'gcloud auth print-access-token' found no Google login: "
                f"{(result.stderr or '').strip()[:300]}. Run 'gcloud auth login' or "
                "'gcloud auth application-default login'",
            )
        return token, ASSUMED_LIFETIME_S


__all__ = ["DEFAULT_TOKEN_URI", "TokenSource", "credentials_path"]
