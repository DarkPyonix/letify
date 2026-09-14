"""Tests for scripts/version.py, pinning SPEC.md Packaging, Versioning and releases."""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location(
        "letify_version_script", ROOT / "scripts" / "version.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


version = _load()


def _tree(tmp_path: Path, *, extension: bool = True) -> Path:
    (tmp_path / "letify").mkdir()
    (tmp_path / "letify-core").mkdir()
    shutil.copy(ROOT / "pyproject.toml", tmp_path / "pyproject.toml")
    shutil.copy(ROOT / "letify" / "__init__.py", tmp_path / "letify" / "__init__.py")
    shutil.copy(ROOT / "letify-core" / "Cargo.toml", tmp_path / "letify-core" / "Cargo.toml")
    shutil.copy(ROOT / "letify-core" / "Cargo.lock", tmp_path / "letify-core" / "Cargo.lock")
    if extension:
        ext = tmp_path / "letify-ext"
        ext.mkdir()
        (ext / "package.json").write_text(
            json.dumps({"name": "letify-status", "version": "0.0.1"}, indent=2) + "\n"
        )
        lock = {"name": "letify-status", "version": "0.0.1", "packages": {"": {"version": "0.0.1"}}}
        (ext / "package-lock.json").write_text(json.dumps(lock, indent=2) + "\n")
    return tmp_path


def test_the_repository_versions_agree() -> None:
    assert version.check(ROOT) == []


def test_set_writes_one_version_into_every_file(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    version.set_version(root, "2.3.4")
    found = version.read_versions(root)
    assert set(found.values()) == {"2.3.4"}
    assert len(found) == 7
    assert version.check(root, tag="v2.3.4") == []


def test_a_file_that_disagrees_with_pyproject_is_reported(tmp_path: Path) -> None:
    root = _tree(tmp_path, extension=False)
    version.set_version(root, "1.0.0")
    init = root / "letify" / "__init__.py"
    init.write_text(init.read_text().replace('__version__ = "1.0.0"', '__version__ = "1.0.1"'))
    errors = version.check(root)
    assert len(errors) == 1 and "letify/__init__.py" in errors[0]


def test_a_tag_that_differs_from_the_version_is_reported(tmp_path: Path) -> None:
    root = _tree(tmp_path, extension=False)
    version.set_version(root, "1.0.0")
    errors = version.check(root, tag="v1.0.1")
    assert len(errors) == 1 and "v1.0.1" in errors[0]


def test_set_refuses_a_version_that_is_not_semver(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        version.set_version(_tree(tmp_path), "1.2")


def test_the_extension_is_optional(tmp_path: Path) -> None:
    root = _tree(tmp_path, extension=False)
    version.set_version(root, "1.0.0")
    assert not any(name.startswith("letify-ext") for name in version.read_versions(root))
