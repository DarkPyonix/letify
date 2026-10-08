"""Module shipping: make modules named by Env.ship() importable on the runtime by reference.

Spec "Module shipping":
- A module named in Env.ship() is placed on the runtime inside the workspace root under
  <workspace root>/modules, and that directory is added to sys.path and PYTHONPATH.
- Because the module is importable by name on both ends, cloudpickle sends references rather
  than by-value definitions.
- An import inside the function body succeeds.
- Class identity across arguments and runtime imports is preserved, so issubclass holds.
- Shipped files travel once through the content addressed blob store and second calls reuse
  the cache.
- Nothing is written outside the workspace root.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

import letify
from letify.declare.env import Env
from letify.runtime import bootstrap


@pytest.fixture
def project(tmp_path: Path, monkeypatch) -> Path:
    """A private home, workspace root, and project root for the Local provider worker."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

    workspace = tmp_path / "runtime-workspace"
    workspace.mkdir()
    monkeypatch.setattr(bootstrap, "DEFAULT_WORKSPACE_ROOT", str(workspace))

    root = tmp_path / "study"
    root.mkdir()
    (root / "pyproject.toml").write_text("[project]\nname = 'study'\n", encoding="utf-8")
    monkeypatch.chdir(root)
    return root


@contextmanager
def temporary_package(directory: Path, name: str, files: dict[str, str]) -> Generator[None, None, None]:
    """Create a temporary importable package on sys.path and clean it up completely."""
    pkg_dir = directory / name
    pkg_dir.mkdir(parents=True, exist_ok=True)
    for rel_path, content in files.items():
        file_path = pkg_dir / rel_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")

    sys.path.insert(0, str(directory))
    try:
        yield
    finally:
        if str(directory) in sys.path:
            sys.path.remove(str(directory))
        for key in list(sys.modules):
            if key == name or key.startswith(f"{name}."):
                sys.modules.pop(key, None)
        shutil.rmtree(pkg_dir, ignore_errors=True)


def test_shipped_module_can_be_imported_inside_function_body(
    let: letify.Launcher, cpu: letify.Instance, project: Path, tmp_path: Path
) -> None:
    # Bug A: An import inside the declared function body previously failed with
    # ModuleNotFoundError because by-value shipping never put the package on the runtime's sys.path.
    files = {
        "__init__.py": "def package_version():\n    return '1.0.0'\n",
        "utils.py": "class Progress:\n    def step(self):\n        return 42\n",
    }
    with temporary_package(tmp_path, "creator_camp", files):
        env = Env().ship("creator_camp")

        @let.function(device=cpu, host=letify.remote, env=env)
        def train_encoder() -> int:
            from creator_camp.utils import Progress

            return Progress().step()

        assert train_encoder() == 42


def test_class_passed_as_argument_preserves_identity_and_issubclass(
    let: letify.Launcher, cpu: letify.Instance, project: Path, tmp_path: Path
) -> None:
    # Bug B: A class passed as an argument and one imported on the runtime previously had
    # distinct identities because by-value deserialization reconstructed a new class object,
    # breaking issubclass.
    files = {
        "__init__.py": "",
        "models.py": (
            "class BaseModel:\n"
            "    pass\n\n"
            "class AudioEncoderModel(BaseModel):\n"
            "    pass\n"
        ),
    }
    with temporary_package(tmp_path, "creator_camp", files):
        from importlib import import_module

        models_mod = import_module("creator_camp.models")
        encoder_cls = models_mod.AudioEncoderModel

        env = Env().ship("creator_camp")

        @let.function(device=cpu, host=letify.remote, env=env)
        def verify_model(model_class: type) -> bool:
            from creator_camp.models import AudioEncoderModel

            return issubclass(model_class, AudioEncoderModel)

        assert verify_model(encoder_cls) is True


def test_shipped_module_files_travel_once_and_second_call_reuses_cache(
    let: letify.Launcher, cpu: letify.Instance, project: Path, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    # Shipped modules travel through the content addressed blob store and are cached on the runtime.
    # The first call places the files; a second call finds them in the cache and transfers 0 blobs.
    files = {
        "__init__.py": "def helper():\n    return 100\n",
        "data.json": '{"key": "value"}\n',
    }
    with temporary_package(tmp_path, "cache_pkg", files):
        env = Env().ship("cache_pkg")

        @let.function(device=cpu, host=letify.remote, env=env)
        def run_call() -> int:
            import cache_pkg

            return cache_pkg.helper()

        assert run_call() == 100
        out1 = capsys.readouterr().err

        assert run_call() == 100
        out2 = capsys.readouterr().err

        assert "already on the runtime" in out2 or "already on" in out2


def test_nothing_is_written_outside_workspace_root(
    let: letify.Launcher, cpu: letify.Instance, project: Path, tmp_path: Path
) -> None:
    # Every remote path letify writes must stay inside the declared workspace root.
    workspace = tmp_path / "runtime-workspace"
    files = {
        "__init__.py": "def value():\n    return 7\n",
    }
    with temporary_package(tmp_path, "clean_pkg", files):
        env = Env().ship("clean_pkg")

        @let.function(device=cpu, host=letify.remote, env=env)
        def compute() -> int:
            import clean_pkg

            return clean_pkg.value()

        assert compute() == 7

        modules_dir = workspace / "modules"
        assert modules_dir.is_dir()
        assert (modules_dir / "clean_pkg" / "__init__.py").is_file()

        home = tmp_path / "home"
        assert not (home / ".letify-runtime").exists()


def test_compiled_extension_in_shipped_module_is_refused(
    let: letify.Launcher, cpu: letify.Instance, project: Path, tmp_path: Path
) -> None:
    # Shipped modules containing platform-specific compiled extensions must fail with ConfigError.
    files = {
        "__init__.py": "",
        "native.so": "compiled binary",
    }
    with temporary_package(tmp_path, "native_pkg", files):
        env = Env().ship("native_pkg")

        @let.function(device=cpu, host=letify.remote, env=env)
        def call_native() -> None:
            pass

        with pytest.raises(letify.ConfigError, match=r"cannot ship 'native_pkg'.*compiled extension"):
            call_native()


def test_missing_module_in_ship_raises_config_error(
    let: letify.Launcher, cpu: letify.Instance, project: Path
) -> None:
    env = Env().ship("no_such_module_exists")

    @let.function(device=cpu, host=letify.remote, env=env)
    def call_missing() -> None:
        pass

    with pytest.raises(letify.ConfigError, match=r"cannot ship 'no_such_module_exists'"):
        call_missing()


def test_one_shot_driver_imports_shipped_module_and_preserves_subclass(tmp_path: Path) -> None:
    from importlib import import_module
    from letify.protocol import codec, driver

    files = {
        "__init__.py": "",
        "classes.py": "class Base:\n    pass\n\nclass Derived(Base):\n    pass\n",
    }
    with temporary_package(tmp_path, "oneshot_pkg", files):
        mod = import_module("oneshot_pkg.classes")

        def work(cls: type) -> bool:
            from oneshot_pkg.classes import Base

            return issubclass(cls, Base)

        source = driver.build(work, (mod.Derived,), {}, modules=("oneshot_pkg",))
        result = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True)
        assert result.returncode == 0
        logs, value = codec.parse(result.stdout, runtime_key="one-shot")
        assert value is True
