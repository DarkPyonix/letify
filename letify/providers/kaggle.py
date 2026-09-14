"""Kaggle, one Kaggle account reached through the official Kaggle CLI run by uv.

This module owns the account's accelerator list, its remaining weekly quota read from
``kaggle quota --format json``, the channel to a registered Kaggle Jupyter Server session
and the batch channel that pushes one script kernel per call. It does not own the login,
which is in ``letify.config.login``, or the kernel execution itself, which is in
``kaggle_adapter.py``. It opens no tunnel or port forward of any kind, because the Kaggle
Acceptable Use Policy forbids circumvention tools, and it sends no keep-alive request.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from time import monotonic, sleep
from typing import TYPE_CHECKING, Any

from .. import tools
from ..config import ProviderConfig
from ..config.secrets import account_directory
from ..declare.instance import Host, Instance
from ..errors import (
    ProviderUnavailable,
    RuntimeFailure,
    RuntimeLost,
    UnsupportedMode,
)
from ..runtime.channel import OneShotChannel
from .base import Provider
from .colab_files import ContentsTransfer
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


class JupyterTransfer(ContentsTransfer):
    """The contents API of a plain Jupyter server, authenticated with its token."""

    def _auth_query(self) -> dict[str, str]:
        return {"token": self.token} if self.token else {}

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"token {self.token}"} if self.token else {}


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
        return KaggleSessionEnded(
            f"{self.alias}: the Kaggle Jupyter Server session at {self.host} has ended. Kaggle "
            f"ends a session after 20 minutes idle or at its 12 hour limit. Start a new session "
            f"in the Kaggle editor with Run, Kaggle Jupyter Server, then run: "
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

    def transfer(self) -> ContentsTransfer:
        return JupyterTransfer(self.base, self.token or "")

    def run(self, kernel: str, source: str, timeout: float | None) -> str:
        """Run one program through the adapter and return its standard output."""
        env = dict(os.environ)
        env["LETIFY_JUPYTER_URL"] = self._url
        env["LETIFY_KERNEL_ID"] = kernel
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


#: Seconds a batch kernel may run when the account sets no ``batch_timeout``.
BATCH_TIMEOUT = 1800

#: Seconds between status reads, and the allowance past the timeout before letify gives up.
STATUS_INTERVAL = 30
STATUS_GRACE = 300

#: The machine shape ``kaggle kernels push --accelerator`` takes for each accelerator.
MACHINE_SHAPES = {"T4": "NvidiaTeslaT4", "P100": "NvidiaTeslaP100", "TPU_V3_8": "Tpu1VmV38"}


class BatchChannel(OneShotChannel):
    """Runs a declared call as one pushed Kaggle script kernel, as spec "Kaggle batch mode" says.

    Programs that return no value are held and sent with the next program that returns one,
    so a call costs one push. Nothing is pushed again after a failure or a timeout.
    """

    def __init__(self, provider: Kaggle, runtime: Runtime):
        super().__init__(self._run_batch, name=runtime.name)
        self.provider = provider
        self.instance = runtime.instance
        self.held: list[str] = []
        self.slug = "letify-" + re.sub(r"[^a-z0-9-]", "-", runtime.name.lower())

    def request(self, payload: dict[str, Any], *, timeout: float | None = None) -> tuple[Any, str]:
        op = payload.get("op")
        if op == "exec":
            self.held.append(payload["source"])
            return None, ""
        if op == "lease":
            return None, ""
        if op in ("put_file", "get_file", "pack_dir"):
            raise UnsupportedMode(
                f"{self.provider.alias}: {op} is not available in Kaggle batch mode, because a "
                f"pushed script has no file API. Register a session with --connect for it."
            )
        return super().request(payload, timeout=timeout)

    def _run_batch(self, source: str, timeout: float | None) -> str:
        program = "\n".join([*self.held, source])
        self.held = []
        seconds = self.provider.batch_timeout
        if timeout:
            seconds = min(seconds, int(timeout))
        kernel = f"{self.provider.username()}/{self.slug}"
        with tempfile.TemporaryDirectory(prefix="letify-kaggle-") as folder:
            (Path(folder) / "script.py").write_text(program, encoding="utf-8")
            meta = {
                "id": kernel,
                "title": self.slug,
                "code_file": "script.py",
                "language": "python",
                "kernel_type": "script",
                "is_private": True,
                "enable_gpu": False,
                "enable_tpu": False,
                "enable_internet": True,
            }
            (Path(folder) / "kernel-metadata.json").write_text(json.dumps(meta), encoding="utf-8")
            args = ["kernels", "push", "-p", folder, "--timeout", str(seconds)]
            shape = MACHINE_SHAPES.get(self.instance.accelerator)
            if shape:
                args += ["--accelerator", shape]
            self.provider.cli(*args, cwd=folder)
            self._wait(kernel, seconds)
            self.provider.cli("kernels", "output", kernel, "-p", folder, "-o", "-q", cwd=folder)
            log = Path(folder) / f"{self.slug}.log"
            text = log.read_text(encoding="utf-8") if log.is_file() else ""
        return stdout_of(text)

    def _wait(self, kernel: str, seconds: int) -> None:
        deadline = monotonic() + seconds + STATUS_GRACE
        while True:
            output = self.provider.cli("kernels", "status", kernel)
            found = re.search(r'has status "([^"]*)"', output)
            state = (found.group(1) if found else "").lower()
            if "complete" in state:
                return
            if "error" in state or "cancel" in state:
                message = re.search(r'Failure message: "(.*)"', output)
                detail = message.group(1) if message else state
                raise RuntimeFailure(
                    f"{self.provider.alias}: the Kaggle kernel {kernel} ended with {state}: "
                    f"{self.provider.redact(detail)}"
                )
            if monotonic() > deadline:
                raise RuntimeFailure(
                    f"{self.provider.alias}: the Kaggle kernel {kernel} did not finish within "
                    f"{seconds + STATUS_GRACE} s. It was not pushed again."
                )
            sleep(STATUS_INTERVAL)


def stdout_of(log: str) -> str:
    """The ``stdout`` entries of a Kaggle kernel log, or the log itself when it is not JSON."""
    try:
        entries = json.loads(log)
    except ValueError:
        return log
    if not isinstance(entries, list):
        return log
    return "".join(
        str(entry.get("data") or "")
        for entry in entries
        if isinstance(entry, dict) and entry.get("stream_name") == "stdout"
    )


class Kaggle(Provider):
    """One Kaggle account."""

    kind = "kaggle"
    default_persistence = "ephemeral"
    has_fast_path = False
    persistent_channel = False
    needs_lease = False

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
        self._kernels: dict[str, tuple[Session, str]] = {}
        #: The Kaggle username batch kernels are pushed under, read once.
        self._username: str | None = None

    @property
    def prepares_env(self) -> bool:  # type: ignore[override]
        """A registered session builds the environment. Batch mode uses the Kaggle image."""
        return session_url(self.alias) is not None

    @property
    def batch_timeout(self) -> int:
        value = self.config.option("batch_timeout")
        return int(value) if isinstance(value, (int, float)) and value > 0 else BATCH_TIMEOUT

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

    def username(self) -> str:
        """``username`` from ``kaggle.json``, or the one ``kaggle config view`` prints."""
        if self._username:
            return self._username
        legacy = account_directory(self.alias) / "kaggle.json"
        if legacy.is_file():
            try:
                name = json.loads(legacy.read_text(encoding="utf-8")).get("username")
            except (ValueError, AttributeError):
                name = None
            if isinstance(name, str) and name:
                self._username = name
                return name
        found = re.search(r"^- username: (\S+)\s*$", self.cli("config", "view"), re.MULTILINE)
        if found is None or found.group(1) == "None":
            raise RuntimeFailure(f"{self.alias}: `kaggle config view` printed no username")
        self._username = found.group(1)
        return self._username

    def open_channel(self, runtime: Runtime) -> Channel:
        """A one-shot channel whose programs run in a new kernel of the registered session."""
        from ..runtime.channel import OneShotChannel
        from .colab_files import ColabFiles

        url = session_url(self.alias)
        if url is None:
            return BatchChannel(self, runtime)
        session = Session(self.alias, url)
        kernel = session.create_kernel()
        self._kernels[runtime.name] = (session, kernel)

        def run(source: str, timeout: float | None) -> str:
            return session.run(kernel, source, timeout)

        files = ColabFiles(
            self.alias,
            runtime.name,
            run,
            workspace=self.workspace_root,
            transfer=session.transfer,
        )
        return OneShotChannel(run, name=runtime.name, files=files)

    def stop(self, runtime: Runtime) -> None:
        """Delete the kernel letify created. The session itself is the user's and keeps running."""
        held = self._kernels.pop(runtime.name, None)
        if held is None:
            return
        session, kernel = held
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
    "BATCH_TIMEOUT",
    "GPUS",
    "MACHINE_SHAPES",
    "QUOTA",
    "TPUS",
    "BatchChannel",
    "JupyterTransfer",
    "Kaggle",
    "KaggleSessionEnded",
    "Session",
    "adapter_command",
    "hours",
    "parse_quota",
    "session_url",
    "split_url",
]
