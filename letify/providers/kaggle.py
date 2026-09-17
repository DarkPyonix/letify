"""Kaggle, one Kaggle account reached through the official Kaggle CLI run by uv.

This module owns the account's accelerator list, its remaining weekly quota read from
``kaggle quota --format json`` and the channel to a registered Kaggle Jupyter Server
session. It does not own the login, which is in ``letify.config.login``, or the kernel
execution itself, which is in ``kaggle_adapter.py``. It opens no tunnel or port forward of
any kind, and it sends no keep-alive request.
"""

from __future__ import annotations

import base64
import binascii
import json
import os
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
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
from ..runtime.channel import Connection, FramedChannel
from .base import Provider
from .usage import Usage

#: Seconds one REST request to the session may take.
REST_TIMEOUT = 30

#: A program's timeout when the caller gives none, and the adapter's extra allowance on top.
DEFAULT_PROGRAM_TIMEOUT = 3600.0
ADAPTER_GRACE = 60.0

if TYPE_CHECKING:
    from ..runtime.channel import Channel
    from ..runtime.session import Runtime

#: Accelerators a Kaggle session can be started with, and the memory of one card.
GPUS = {"P100": {"vram_gb": 16}, "T4": {"vram_gb": 16}}
TPUS = ("TPU_V3_8",)

#: The read-only call that answers the weekly quota.
QUOTA = ("quota", "--format", "json")


def hours(value: Any) -> float | None:
    """Read a figure such as ``3.25h`` as hours."""
    text = str(value or "").strip().removesuffix("h").strip()
    try:
        return float(text)
    except ValueError:
        return None


def parse_quota(output: str) -> dict[str, dict[str, Any]]:
    """The quota rows keyed by resource, from output that may carry warnings before the JSON."""
    start = output.find("[")
    if start < 0:
        return {}
    try:
        rows = json.loads(output[start:])
    except ValueError:
        return {}
    if not isinstance(rows, list):
        return {}
    return {
        str(row.get("resource")).upper(): row
        for row in rows
        if isinstance(row, dict) and row.get("resource")
    }


class KaggleSessionEnded(RuntimeLost):
    """The registered Kaggle Jupyter Server session no longer answers."""


def session_url(alias: str) -> str | None:
    """The Colab Compatible URL registered with ``--connect``, or None."""
    path = account_directory(alias) / "jupyter_url"
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8").strip() or None


def split_url(url: str) -> tuple[str, str | None]:
    """The server base, which is the URL without its query, and the ``token`` parameter."""
    parts = urllib.parse.urlsplit(url)
    base = urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path.rstrip("/"), "", ""))
    token = dict(urllib.parse.parse_qsl(parts.query)).get("token")
    return base, token


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
        with urllib.request.urlopen(request, timeout=REST_TIMEOUT) as response:
            return response.read()

    def alive(self) -> bool:
        """Whether ``/api/status`` answers 200."""
        try:
            self._request("GET", "/api/status")
        except (urllib.error.URLError, OSError, ValueError):
            return False
        return True

    def ended(self) -> KaggleSessionEnded:
        """The runtime is lost, named without claiming which of the two causes it was.

        The proxy answers 404 for an ended session, for a registered URL that no longer
        routes, and for a session id that never existed, so a status read that is not 200
        cannot tell them apart. Saying the session ended would state as fact something
        this read does not establish.
        """
        return KaggleSessionEnded(
            f"{self.alias}: the Kaggle Jupyter Server session at {self.host} did not answer. "
            f"It may have ended, since Kaggle ends a session after 20 minutes idle or at its "
            f"12 hour limit, or the registered URL may no longer route to it. Start a new "
            f"session in the Kaggle editor with Run, Kaggle Jupyter Server, then run: "
            f"letify login kaggle {self.alias} --connect <new Colab Compatible URL>"
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
                f"{self.alias}: the Kaggle session at {self.host} refused a new kernel: "
                f"{type(exc).__name__}"
            ) from None
        return str(reply["id"])

    def delete_kernel(self, kernel: str) -> None:
        """Best effort, because a session that already ended has no kernel to delete."""
        try:
            self._request("DELETE", f"/api/kernels/{urllib.parse.quote(kernel)}")
        except (urllib.error.URLError, OSError, ValueError):
            pass

    def run(self, kernel: str, source: str, timeout: float | None) -> str:
        """Run one program through the adapter and return its standard output."""
        env = dict(os.environ)
        env["LETIFY_JUPYTER_URL"] = self._url
        env["LETIFY_KERNEL_ID"] = kernel
        env["LETIFY_ADAPTER_MODE"] = "program"
        env["LETIFY_TIMEOUT"] = str(timeout or DEFAULT_PROGRAM_TIMEOUT)
        process = subprocess.Popen(
            adapter_command(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        limit = (timeout or DEFAULT_PROGRAM_TIMEOUT) + ADAPTER_GRACE
        try:
            stdout, stderr = process.communicate(source, timeout=limit)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            code = None
        else:
            code = process.returncode
        if code == 0:
            return stdout
        if code == 3:
            raise RuntimeFailure(
                f"{self.alias}: a program raised on the Kaggle session", stderr=stderr
            )
        if not self.alive():
            raise self.ended()
        reason = "timed out" if code is None else f"exited {code}"
        raise RuntimeFailure(f"{self.alias}: the Kaggle adapter {reason}", stderr=stderr)


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
        self._process: subprocess.Popen[bytes] | None = None
        self._connection = None
        self._raw = bytearray()

    def start(self) -> None:
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

    @property
    def persistent_channel(self) -> bool:  # type: ignore[override]
        """A registered session keeps one worker, so the object and blob tables survive."""
        return session_url(self.alias) is not None

    #: No device stream can reach a Kaggle session without a tunnel, which Kaggle forbids.
    serves_host_local = False

    usage_unit = "GPU hours"
    usage_source = "kaggle quota, the weekly accelerator quota endpoint"

    default_workspace = "/kaggle/working/letify"

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
                f"{self.alias} cannot serve host='local': Kaggle forbids tunnels and port "
                f"forwarding, so no device stream reaches the session. Use host='remote'."
            )

    def __init__(self, config: ProviderConfig):
        super().__init__(config)
        #: The session and kernel id each runtime runs its programs in.
        self._kernels: dict[str, tuple[Session, str, KaggleChannel]] = {}

    def cli(self, *args: str, cwd: str | None = None) -> str:
        """Run one Kaggle CLI command as this account and return its output."""
        uv = tools.find_uv()
        if uv is None:
            raise ProviderUnavailable(self.kind, tools.missing_uv_message())
        shown = " ".join([tools.KAGGLE.executable, *args])
        result = subprocess.run(
            [*tools.command(tools.KAGGLE, uv), *args],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=cwd,
            env=tools.kaggle_environment(self.alias),
        )
        if result.returncode != 0:
            detail = self.redact((result.stderr or result.stdout or "").strip())
            raise RuntimeFailure(
                f"{self.alias}: `{shown}` exited {result.returncode}", command=shown, stderr=detail
            )
        return result.stdout

    def redact(self, text: str) -> str:
        for secret in self._secrets():
            text = text.replace(secret, "***")
        return text

    def open_channel(self, runtime: Runtime) -> Channel:
        """One worker in one cell of the registered session, behind the adapter bridge."""
        url = session_url(self.alias)
        if url is None:
            raise ConfigError(
                f"{self.alias} has no registered Kaggle Jupyter Server session. Kaggle "
                f"publishes no API that starts one, so start it in the Kaggle editor with "
                f"Run, Kaggle Jupyter Server, then register its Colab Compatible URL with "
                f"`letify login kaggle {self.alias} --connect '<URL>'`."
            )
        session = Session(self.alias, url)
        kernel = session.create_kernel()
        channel = KaggleChannel(
            adapter_command(),
            {"LETIFY_JUPYTER_URL": url, "LETIFY_KERNEL_ID": kernel},
            name=runtime.name,
            session=session,
        )
        self._kernels[runtime.name] = (session, kernel, channel)
        return channel

    def stop(self, runtime: Runtime) -> None:
        """Close the bridge, then delete the kernel letify created.

        In that order: closing the bridge's standard input ends the cell's read loop, so the
        worker exits on its own rather than being cut off mid frame. The session itself is
        the user's and keeps running.
        """
        held = self._kernels.pop(runtime.name, None)
        if held is None:
            return
        session, kernel, channel = held
        try:
            channel.close()
        except OSError:
            pass
        session.delete_kernel(kernel)

    def _secrets(self) -> list[str]:
        """Values in the account's credential files, to hide from error output."""
        directory = account_directory(self.alias)
        found: list[str] = []
        token = directory / "access_token"
        if token.is_file():
            found.append(token.read_text(encoding="utf-8").strip())
        legacy = directory / "kaggle.json"
        if legacy.is_file():
            try:
                found.append(str(json.loads(legacy.read_text(encoding="utf-8")).get("key", "")))
            except (ValueError, AttributeError):
                pass
        return [value for value in found if value]

    def report_usage(self) -> Usage:
        uv = tools.find_uv()
        if uv is None:
            raise ProviderUnavailable(self.kind, tools.missing_uv_message())
        command = [*tools.command(tools.KAGGLE, uv), *QUOTA]
        shown = " ".join([tools.KAGGLE.executable, *QUOTA])
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=120,
            env=tools.kaggle_environment(self.alias),
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()
            for secret in self._secrets():
                detail = detail.replace(secret, "***")
            raise RuntimeFailure(
                f"{self.alias}: `{shown}` exited {result.returncode}", command=shown, stderr=detail
            )
        rows = parse_quota(result.stdout)
        gpu = rows.get("GPU")
        if gpu is None:
            raise RuntimeFailure(f"{self.alias}: `{shown}` reported no GPU quota", command=shown)
        notes = []
        refresh = gpu.get("refreshAt") or gpu.get("refresh_at")
        if refresh:
            notes.append(f"resets {refresh}")
        tpu = rows.get("TPU")
        if tpu is not None:
            notes.append(
                f"TPU {hours(tpu.get('used')):g} h used, {hours(tpu.get('remaining')):g} h left "
                f"of {hours(tpu.get('total')):g}"
            )
        return Usage(
            alias=self.alias,
            kind=self.kind,
            unit=self.usage_unit,
            source=self.usage_source,
            remaining=hours(gpu.get("remaining")),
            limit=hours(gpu.get("total")),
            used=hours(gpu.get("used")),
            note="; ".join(notes) or None,
        )


__all__ = [
    "GPUS",
    "QUOTA",
    "TPUS",
    "Kaggle",
    "KaggleSessionEnded",
    "Session",
    "adapter_command",
    "hours",
    "parse_quota",
    "session_url",
    "split_url",
]
