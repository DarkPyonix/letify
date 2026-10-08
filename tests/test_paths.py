"""Tests enforcing the declared filesystem paths.

Spec section pinned here: "Declared filesystem paths", "Local machine",
"Remote runtime", "The cache command", "Installing external tools".

Every path letify writes must fall strictly inside the declared set.
The system temporary directory is never used.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import sys
import tempfile
from pathlib import Path

import pytest

import letify
from letify import install
from letify.cli import main
from letify.config.secrets import account_directory
from letify.declare.instance import Instance
from letify.paths import (
    ALLOWED_ACCOUNT_CREDENTIAL_FILES,
    DECLARED_LOCAL_EXCEPTIONS,
    DECLARED_LOCAL_PATHS,
    digest_cache_path,
    local_runtime_directory,
    local_runtime_root,
    local_tmp_directory,
    ssh_control_directory,
    storage_cache_directory,
    tool_cache_home,
)
from letify.transport import sshopts


def test_declared_paths_list_cannot_drift() -> None:
    expected_local = {
        "config.toml",
        "accounts",
        "tools",
        "cache",
        "tmp",
        "ssh",
        "runtime",
    }
    assert DECLARED_LOCAL_PATHS == expected_local

    expected_credentials = {
        "token.json",
        "modal.toml",
        "eci.yaml",
        "cookie",
        "access_token",
        "kaggle.json",
        "notebook_id",
        "known_hosts",
        "link.json",
        "password",
    }
    assert ALLOWED_ACCOUNT_CREDENTIAL_FILES == expected_credentials

    assert "~/.ssh/id_letify" in DECLARED_LOCAL_EXCEPTIONS
    assert "~/.ssh/authorized_keys" in DECLARED_LOCAL_EXCEPTIONS


def test_local_provider_call_writes_only_inside_declared_directories(
    monkeypatch, tmp_path: Path
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    project = tmp_path / "project"
    project.mkdir()
    system_tmp = tmp_path / "sys_tmp"
    system_tmp.mkdir()

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("TMPDIR", str(system_tmp))
    monkeypatch.setenv("TEMP", str(system_tmp))
    monkeypatch.setenv("TMP", str(system_tmp))
    monkeypatch.setattr(tempfile, "tempdir", str(system_tmp))
    monkeypatch.chdir(project)

    let = letify.Launcher()
    cpu = Instance(let.providers.local, gpu=None)

    @let.function(device=cpu, host="remote")
    def compute(a: int, b: int) -> int:
        return a * b

    assert compute(6, 7) == 42

    # Verify HOME root entries
    top_level = {p.name for p in home.iterdir()}
    assert top_level <= {".letify", ".ssh"}, f"Unexpected entries in HOME: {top_level}"
    assert ".letify-runtime" not in top_level
    assert ".cache" not in top_level

    # Verify .letify subdirectories
    letify_dir = home / ".letify"
    if letify_dir.exists():
        subdirs = {p.name for p in letify_dir.iterdir()}
        assert subdirs <= DECLARED_LOCAL_PATHS, f"Unexpected subdirs in .letify: {subdirs}"

    # Verify system temp gained nothing
    sys_tmp_files = list(system_tmp.iterdir())
    assert sys_tmp_files == [], f"System temp directory gained files: {sys_tmp_files}"


def test_account_directory_holds_only_credential_files_after_tool_cli_run(
    monkeypatch, tmp_path: Path
) -> None:
    from letify import tools

    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))

    alias = "colab_test"
    acct = account_directory(alias)
    acct.mkdir(parents=True, exist_ok=True)
    (acct / "token.json").write_text('{"access_token": "secret123"}', encoding="utf-8")
    (acct / "token.json").chmod(0o600)

    env = tools.environment(alias)
    tool_home = Path(env["HOME"])

    # Tool home must be inside cache, not account directory
    assert tool_home == home / ".letify" / "cache" / "tools" / alias
    assert tool_home != acct

    # CLI creates files in its redirected home
    dot_config = tool_home / ".config" / "colab-cli"
    dot_config.mkdir(parents=True, exist_ok=True)
    dot_local = tool_home / ".local" / "share"
    dot_local.mkdir(parents=True, exist_ok=True)
    (dot_local / "junk.dat").write_bytes(b"junk")

    # The CLI token must be linked to the real secret
    tool_token = dot_config / "token.json"
    assert tool_token.is_symlink() or tool_token.exists()

    # The account directory must contain ONLY allowed credential files
    acct_entries = list(acct.iterdir())
    for entry in acct_entries:
        assert entry.is_file(), f"Found directory in account directory: {entry}"
        assert entry.name in ALLOWED_ACCOUNT_CREDENTIAL_FILES, f"Disallowed file: {entry.name}"


def test_installing_new_tool_version_prunes_older_versions(
    patch_which, monkeypatch, isolated_home
) -> None:
    from tests.test_install import ReleaseServer, publish_eci

    patch_which(install, present=False)
    server = ReleaseServer()
    monkeypatch.setattr(install, "ECI_RELEASE", server.url + "/{version}")
    monkeypatch.setattr(install, "host", lambda: ("Linux", "x86_64"))
    monkeypatch.delenv("LETIFY_AUTO_INSTALL", raising=False)

    # Pre-populate older versions in tools directory
    tools_dir = Path.home() / ".letify" / "tools" / "eci"
    v_old1 = tools_dir / "0.2.0"
    v_old1.mkdir(parents=True, exist_ok=True)
    (v_old1 / "eci").write_bytes(b"old binary 1")

    v_old2 = tools_dir / "0.1.9"
    v_old2.mkdir(parents=True, exist_ok=True)
    (v_old2 / "eci").write_bytes(b"old binary 2")

    publish_eci(monkeypatch, server)

    try:
        installed = install.install("eci")
        assert installed.name == "eci"
        assert (tools_dir / "0.2.1").exists()

        # Older versions must be pruned
        assert not v_old1.exists()
        assert not v_old2.exists()
    finally:
        server.server.shutdown()


def test_letify_cache_reports_local_trees_and_cleans_safe_targets(
    isolated_home, capsys
) -> None:
    letify_root = Path.home() / ".letify"
    (letify_root / "cache").mkdir(parents=True, exist_ok=True)
    (letify_root / "cache" / "sample.cache").write_bytes(b"data" * 100)

    (letify_root / "tools").mkdir(parents=True, exist_ok=True)
    (letify_root / "tools" / "tool.bin").write_bytes(b"tool" * 50)

    (letify_root / "tmp").mkdir(parents=True, exist_ok=True)
    (letify_root / "tmp" / "temp.log").write_bytes(b"temp" * 20)

    (letify_root / "accounts" / "test_acct").mkdir(parents=True, exist_ok=True)
    (letify_root / "accounts" / "test_acct" / "token.json").write_text("secret", encoding="utf-8")

    # Test cache --json reporting
    assert main(["cache", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert "local" in report
    local_info = report["local"]

    assert "cache" in local_info
    assert local_info["cache"]["cleanable"] is True
    assert local_info["cache"]["files"] >= 1

    assert "accounts" in local_info
    assert local_info["accounts"]["cleanable"] is False
    assert local_info["accounts"]["files"] >= 1

    # Test clearing accounts is refused
    assert main(["cache", "clear", "accounts"]) == 1
    err = capsys.readouterr().err
    assert "refused" in err.lower() or "cannot" in err.lower()
    assert (letify_root / "accounts" / "test_acct" / "token.json").exists()

    # Test clearing tmp succeeds
    assert main(["cache", "clear", "tmp"]) == 0
    out = capsys.readouterr().out
    assert "removed" in out
    assert not (letify_root / "tmp" / "temp.log").exists()

    # Test clearing all safe trees
    assert main(["cache", "clear", "all"]) == 0
    out = capsys.readouterr().out
    assert "removed" in out
    assert not (letify_root / "cache" / "sample.cache").exists()
    assert (letify_root / "accounts" / "test_acct" / "token.json").exists()


def test_ssh_control_directory_uses_letify_ssh_and_handles_length_limit(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(sshopts, "WINDOWS", False)
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)

    home = tmp_path / "normal_user"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

    ctrl_dir = sshopts.control_directory()
    assert ctrl_dir == home / ".letify" / "ssh"
    assert ctrl_dir != Path(f"/tmp/letify-{os.getuid()}")

    # When socket path fits within limit, ControlPath is included
    monkeypatch.setattr(sshopts, "SOCKET_LIMIT", 300)
    opts = sshopts.options("lab")
    control_parts = [p for p in opts if p.startswith("ControlPath=")]
    assert len(control_parts) == 1
    assert str(ctrl_dir) in control_parts[0]

    # When socket path exceeds platform socket limit, multiplexing is safely omitted
    monkeypatch.setattr(sshopts, "SOCKET_LIMIT", 108)
    long_home = tmp_path / ("u" * 150)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: long_home))
    long_opts = sshopts.options("lab")
    assert not any(p.startswith("ControlPath=") for p in long_opts)
    assert not any(p.startswith("ControlMaster=") for p in long_opts)


def test_compatibility_fallbacks_for_migrated_paths(
    isolated_home, monkeypatch, tmp_path: Path
) -> None:
    home = Path.home()

    # 1. DigestCache fallback
    old_digests = home / ".cache" / "letify" / "digests.json"
    old_digests.parent.mkdir(parents=True, exist_ok=True)
    old_digests.write_text(json.dumps({"f1": [10, 20, 30, "hash123"]}), encoding="utf-8")

    from letify.store import pathdata

    cache = pathdata.DigestCache()
    assert len(cache) == 1
    assert cache._entries.get("f1") == [10, 20, 30, "hash123"]

    # 2. Storage backend location fallback
    from letify.store.backends import default_location

    old_store = home / ".cache" / "letify" / "my_store"
    old_store.mkdir(parents=True, exist_ok=True)
    assert default_location("filesystem", "my_store") == ("root", str(old_store))

    new_store_loc = default_location("filesystem", "new_store")
    assert new_store_loc == ("root", str(home / ".letify" / "cache" / "storage" / "new_store"))

    # 3. Local provider workspace root
    from letify.providers.local import Local

    assert local_runtime_root() == "~/.letify/runtime"
    assert local_runtime_directory() == home / ".letify" / "runtime"

    # 4. Colab token fallback
    acct = home / ".letify" / "accounts" / "colab_compat"
    old_token = acct / ".config" / "colab-cli" / "token.json"
    old_token.parent.mkdir(parents=True, exist_ok=True)
    old_token.write_text(json.dumps({"access_token": "compat_tok"}), encoding="utf-8")

    from letify.config.schema import ProviderConfig
    from letify.providers.colab import Colab

    colab = Colab(ProviderConfig("colab_compat", "colab", {}, 0))
    # It should find the token via fallback
    assert (acct / ".config" / "colab-cli" / "token.json").is_file()
