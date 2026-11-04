"""Development upload of a local ``letify-link`` binary to a runtime.

Owns the platform check, the sha256 cache path, the chunked transfer and the verification of
Spec "Sidecar interface" item 7 (Development upload). Does not own the command channels: the
caller supplies ``run`` (one Python program on the runtime, returning what it printed) and,
for a machine with a byte stream such as ssh, ``push``.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path

from ..errors import ProviderUnavailable

#: Raw bytes carried by one command channel call. Base64 makes the program 4/3 of this.
UPLOAD_CHUNK = 4 << 20
#: Seconds one call may take before it fails.
CALL_TIMEOUT = 300.0
#: The environment variable that names the binary when the account does not.
ENV_UPLOAD = "LETIFY_LINK_UPLOAD"

_ELF_MACHINES = {62: "x86_64", 183: "aarch64"}

_PROBE = """
import json, os, platform, hashlib
root = os.path.expanduser({workspace!r})
directory = os.path.join(root, ".letify", "bin")
os.makedirs(directory, exist_ok=True)
path = os.path.join(directory, {name!r})
have = False
if os.path.isfile(path) and os.access(path, os.X_OK) and os.path.getsize(path) == {size}:
    have = hashlib.sha256(open(path, "rb").read()).hexdigest() == {sha!r}
print("LINKUP " + json.dumps({{"system": platform.system(), "machine": platform.machine(),
                              "path": path, "have": have}}))
"""

_CHUNK = """
import base64
path = {part!r}
data = base64.b64decode(DATA)
with open(path, "wb" if {offset} == 0 else "r+b") as handle:
    handle.seek({offset})
    handle.write(data)
print("LINKUP ok")
"""

_FINISH = """
import hashlib, json, os
part = {part!r}
final = {final!r}
size = os.path.getsize(part)
got = hashlib.sha256(open(part, "rb").read()).hexdigest()
if size != {size} or got != {sha!r}:
    os.remove(part)
    print("LINKUP " + json.dumps({{"ok": False, "size": size, "sha": got}}))
else:
    os.chmod(part, 0o700)
    os.replace(part, final)
    print("LINKUP " + json.dumps({{"ok": True}}))
"""


def configured(option: object) -> str | None:
    """The local path from the account option, else from the environment, else None."""
    if isinstance(option, str) and option:
        return option
    named = os.environ.get(ENV_UPLOAD)
    return named or None


def local_arch(path: Path) -> str:
    """The ELF machine of a local file, or raise when it is not an ELF executable."""
    with open(path, "rb") as handle:
        head = handle.read(20)
    if len(head) < 20 or head[:4] != b"\x7fELF":
        raise ProviderUnavailable("letify-link", f"letify-link upload: {path} is not an ELF file")
    machine = int.from_bytes(head[18:20], "little" if head[5] == 1 else "big")
    return _ELF_MACHINES.get(machine, f"elf-machine-{machine}")


def _reply(alias: str, output: str) -> dict:
    for line in output.splitlines():
        if line.startswith("LINKUP "):
            rest = line[len("LINKUP ") :]
            return {"ok": True} if rest == "ok" else json.loads(rest)
    raise ProviderUnavailable(
        "letify-link", f"letify-link upload: {alias} answered no result: {output.strip()[-400:]}"
    )


def _digest(path: Path) -> tuple[str, int]:
    sha = hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        while block := handle.read(1 << 20):
            sha.update(block)
            size += len(block)
    return sha.hexdigest(), size


def ensure(
    alias: str,
    local: str | os.PathLike[str],
    *,
    workspace: str,
    run: Callable[[str, float], str],
    push: Callable[[str, Path], None] | None = None,
    chunk: int | None = None,
    report: Callable[[str, int], None] | None = None,
) -> str:
    """Make sure the runtime holds ``local`` and return its path there.

    ``push(remote_path, local)`` streams the whole file when the machine has a byte stream;
    without it the file goes through ``run`` in chunks of ``chunk`` bytes as base64.
    ``report(how, size)`` is told ``"cached"`` when the runtime already held the file and
    ``"uploaded"`` when it was sent, with the size in bytes.
    """
    path = Path(local).expanduser()
    if not path.is_file():
        raise ProviderUnavailable("letify-link", f"letify-link upload: {path} does not exist")
    arch = local_arch(path)
    sha, size = _digest(path)
    name = f"letify-link-{sha[:16]}"
    probe = _reply(
        alias,
        run(_PROBE.format(workspace=workspace, name=name, size=size, sha=sha), CALL_TIMEOUT),
    )
    if probe.get("system") != "Linux" or probe.get("machine") != arch:
        raise ProviderUnavailable(
            "letify-link",
            f"letify-link upload: {path} is built for {arch} but {alias} is "
            f"{probe.get('system')} {probe.get('machine')}",
        )
    final = str(probe["path"])
    if probe.get("have"):
        if report is not None:
            report("cached", size)
        return final
    part = final + ".part"
    if push is not None:
        push(part, path)
    else:
        step = chunk or UPLOAD_CHUNK
        with open(path, "rb") as handle:
            offset = 0
            while True:
                block = handle.read(step)
                if not block and offset:
                    break
                source = f"DATA = {base64.b64encode(block).decode()!r}\n" + _CHUNK.format(
                    part=part, offset=offset
                )
                _reply(alias, run(source, CALL_TIMEOUT))
                offset += len(block)
                if not block:
                    break
    done = _reply(
        alias, run(_FINISH.format(part=part, final=final, size=size, sha=sha), CALL_TIMEOUT)
    )
    if not done.get("ok"):
        raise ProviderUnavailable(
            "letify-link",
            f"letify-link upload: checksum mismatch on {alias}: expected {sha}, "
            f"got {done.get('sha')}",
        )
    if report is not None:
        report("uploaded", size)
    return final
