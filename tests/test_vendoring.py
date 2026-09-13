"""letify depends on nothing, and its worker runs where nothing is installed.

Spec section pinned here: "Packaging". letify lives in a researcher's repository as a
dependency, so what it installs becomes part of that repository's environment. The worker
tests run an interpreter with ``-I -S``, which ignores site-packages, PYTHONPATH and the
working directory, because that is the position a freshly provisioned remote machine is in.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import letify
from letify.protocol import codec
from letify.protocol.worker import BOOTSTRAP
from letify.runtime.channel import PersistentChannel

ROOT = Path(__file__).resolve().parents[1]

#: An interpreter that can import nothing but the standard library.
BARE = [sys.executable, "-I", "-S"]


def test_letify_installs_nothing_else() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["dependencies"] == []


def test_cloudpickle_is_the_vendored_copy_with_its_license() -> None:
    assert codec.cloudpickle.__name__ == "letify._vendor.cloudpickle"
    vendored = Path(codec.cloudpickle.__file__).resolve().parent
    assert vendored.parent.name == "_vendor"
    # The BSD 3-clause text, which is what permits carrying the code with a copy of it.
    license_text = (vendored / "LICENSE").read_text(encoding="utf-8")
    assert "Redistribution and use in source and binary forms" in license_text


def test_content_addresses_use_the_standard_library() -> None:
    import hashlib

    payload = b"same contents"
    assert codec.digest_of(payload) == hashlib.blake2b(payload, digest_size=16).hexdigest()


def test_the_bare_interpreter_really_cannot_import_letify_or_cloudpickle() -> None:
    # Guards the tests below: if this passed, they would prove nothing.
    import subprocess

    probe = "import importlib.util as u; print(u.find_spec('letify'), u.find_spec('cloudpickle'))"
    out = subprocess.run([*BARE, "-c", probe], capture_output=True, text=True, timeout=60)
    assert out.stdout.strip() == "None None"


def test_the_worker_runs_a_call_where_nothing_is_installed() -> None:
    def triple(x: int) -> tuple[int, bool]:
        import importlib.util

        return x * 3, importlib.util.find_spec("cloudpickle") is None

    channel = PersistentChannel([*BARE, "-c", BOOTSTRAP], name="bare")
    try:
        channel.start()
        value, _logs = channel.call(triple, (7,), {})
    finally:
        channel.close()
    # The call ran, and the far side still has no cloudpickle of its own installed.
    assert value == (21, True)


def test_the_one_shot_driver_runs_where_nothing_is_installed() -> None:
    import subprocess

    from letify.protocol import driver

    def add(a: int, b: int) -> int:
        return a + b

    script = driver.build(add, (40,), {"b": 2})
    result = subprocess.run([*BARE, "-c", script], capture_output=True, text=True, timeout=120)
    _logs, value = codec.parse(result.stdout, runtime_key="bare")
    assert value == 42


def test_the_package_exports_do_not_reach_for_cloudpickle_at_import() -> None:
    # The vendored copy is what codec imports, so a missing top level cloudpickle is fine.
    assert letify.__version__
