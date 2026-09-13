"""Generated provider types, so an editor completes let.providers.<alias>.<accelerator>.

Spec section pinned here: "Generated provider types". The stub is checked by parsing the text
letify writes, because what a type checker reads is that text.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest
from conftest import FakeCompleted

import letify
import letify_providers
from letify import stubs
from letify.cli import main
from letify.providers import shell as shell_module


def classes(text: str) -> dict[str, ast.ClassDef]:
    return {node.name: node for node in ast.parse(text).body if isinstance(node, ast.ClassDef)}


def fields(node: ast.ClassDef) -> dict[str, str]:
    return {
        item.target.id: ast.unparse(item.annotation)
        for item in node.body
        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)
    }


def bases(node: ast.ClassDef) -> list[str]:
    return [ast.unparse(base) for base in node.bases]


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A user project with a pyproject.toml and a .letify, as the working directory."""
    root = tmp_path / "project"
    root.mkdir()
    (root / "pyproject.toml").write_text('[project]\nname = "study"\n', encoding="utf-8")
    monkeypatch.chdir(root)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

    def declare(body: str) -> Path:
        (root / ".letify").write_text(body, encoding="utf-8")
        return root

    return declare


# -- Spec: Generated provider types, naming --------------------------------------


@pytest.mark.parametrize(
    ("alias", "name"),
    [("colab_pro", "ColabPro"), ("lab_a100", "LabA100"), ("local", "Local"), ("box", "Box")],
)
def test_an_alias_becomes_a_class_named_in_camel_case(alias: str, name: str) -> None:
    assert stubs.class_name(alias) == name


def test_a_name_that_is_a_keyword_gets_provider_appended() -> None:
    assert stubs.class_name("none") == "NoneProvider"
    assert stubs.class_name("true") == "TrueProvider"


def test_two_aliases_that_would_share_a_name_are_told_apart_in_order(project) -> None:
    project('[lab_a]\nkind = "local"\n[labA]\nkind = "local"\n')
    let = letify.Launcher()
    view = fields(classes(stubs.render(let))["ProvidersView"])
    assert view["lab_a"] == "LabA"
    assert view["labA"] == "LabA2"


# -- Spec: Generated provider types, content -------------------------------------


def test_each_alias_subclasses_its_kind_and_names_its_accelerators(project) -> None:
    project(
        '[colab_pro]\nkind = "colab"\n'
        '[lab_a100]\nkind = "shell"\naddress = "h"\ngpus = ["A100"]\n'
    )
    text = stubs.render(letify.Launcher())
    found = classes(text)

    assert bases(found["ColabPro"]) == ["letify.providers.Colab"]
    assert fields(found["ColabPro"])["G4"] == "Instance"
    assert bases(found["LabA100"]) == ["letify.providers.Shell"]
    assert fields(found["LabA100"]) == {"A100": "Instance", "__getattr__": "None"}

    view = fields(found["ProvidersView"])
    assert view["colab_pro"] == "ColabPro"
    assert view["lab_a100"] == "LabA100"
    assert view["local"] == "Local"
    assert bases(found["ProvidersView"]) == ["letify.launcher.Providers"]


def test_an_alias_named_after_its_kind_does_not_shadow_the_base_class(project) -> None:
    # The base is referenced through its module, so a class called Colab can subclass Colab.
    project('[colab]\nkind = "colab"\n')
    found = classes(stubs.render(letify.Launcher()))
    assert bases(found["Colab"]) == ["letify.providers.Colab"]


def test_only_the_accounts_this_project_can_use_are_named(project, tmp_path) -> None:
    (Path.home() / ".letify").write_text(
        '[lab]\nkind = "shell"\naddress = "h"\n[colab_pro]\nkind = "colab"\nglobal = true\n',
        encoding="utf-8",
    )
    project('[box]\nkind = "local"\n')
    view = fields(classes(stubs.render(letify.Launcher()))["ProvidersView"])
    assert set(view) >= {"box", "colab_pro", "local"}
    assert "lab" not in view


def test_writing_the_stub_never_connects_to_a_machine(project, patch_run) -> None:
    # A shell account with no declared accelerators would have to be asked over SSH, so its
    # class keeps attribute lookup typed as Instance instead.
    recorder = patch_run(shell_module, result=FakeCompleted(stdout="NVIDIA A100, 81920\n"))
    project('[lab]\nkind = "shell"\naddress = "gpu.example.edu"\n')
    found = classes(stubs.render(letify.Launcher()))
    assert recorder.calls == []
    lab = found["Lab"]
    assert "__getattr__" not in fields(lab)
    methods = {item.name for item in lab.body if isinstance(item, ast.FunctionDef)}
    assert "__getattr__" in methods


def test_an_account_whose_provider_cannot_be_built_is_the_plain_provider(project) -> None:
    project('[odd]\nkind = "vastai"\n')
    view = fields(classes(stubs.render(letify.Launcher()))["ProvidersView"])
    assert view["odd"] == "letify.providers.Provider"


def test_the_stub_is_valid_python(project) -> None:
    project('[colab_pro]\nkind = "colab"\n[modal]\nkind = "modal"\n')
    compile(stubs.render(letify.Launcher()), "letify_providers.pyi", "exec")


# -- Spec: Generated provider types, where it goes -------------------------------


def test_the_stub_goes_in_typings_at_the_project_root(project, tmp_path, monkeypatch) -> None:
    root = project('[box]\nkind = "local"\n')
    nested = root / "src" / "study"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    assert stubs.project_root() == root
    assert stubs.target() == root / "typings" / "letify_providers.pyi"


def test_pyproject_moves_the_typings_directory(project) -> None:
    root = project('[box]\nkind = "local"\n')
    (root / "pyproject.toml").write_text(
        '[project]\nname = "study"\n[tool.letify]\ntypings = "types/stubs"\n', encoding="utf-8"
    )
    assert stubs.target() == root / "types" / "stubs" / "letify_providers.pyi"


def test_pyproject_can_turn_generation_off(project) -> None:
    root = project('[box]\nkind = "local"\n')
    (root / "pyproject.toml").write_text(
        '[project]\nname = "study"\n[tool.letify]\ntypings = false\n', encoding="utf-8"
    )
    assert stubs.target() is None


def test_with_no_pyproject_the_working_directory_is_the_root(tmp_path, monkeypatch) -> None:
    bare = tmp_path / "bare"
    bare.mkdir()
    monkeypatch.chdir(bare)
    assert stubs.project_root() == bare


# -- Spec: Generated provider types, when it is written --------------------------


def test_loading_the_configuration_writes_the_stub(project, monkeypatch) -> None:
    monkeypatch.setenv("LETIFY_STUBS", "1")
    root = project('[box]\nkind = "local"\n')
    letify.Launcher()
    written = root / "typings" / "letify_providers.pyi"
    assert "class Box(" in written.read_text(encoding="utf-8")


def test_the_environment_variable_turns_generation_off(project, monkeypatch) -> None:
    monkeypatch.setenv("LETIFY_STUBS", "0")
    root = project('[box]\nkind = "local"\n')
    letify.Launcher()
    assert not (root / "typings").exists()


def test_an_unchanged_stub_is_not_rewritten(project, monkeypatch) -> None:
    monkeypatch.setenv("LETIFY_STUBS", "1")
    root = project('[box]\nkind = "local"\n')
    letify.Launcher()
    written = root / "typings" / "letify_providers.pyi"
    before = written.stat().st_mtime_ns
    os.utime(written, ns=(before - 10_000_000_000, before - 10_000_000_000))
    stamped = written.stat().st_mtime_ns
    letify.Launcher()
    assert written.stat().st_mtime_ns == stamped


def test_the_command_line_writes_the_stub_on_demand(project, capsys) -> None:
    root = project('[box]\nkind = "local"\n')
    assert main(["stubs"]) == 0
    written = root / "typings" / "letify_providers.pyi"
    assert written.is_file()
    assert str(written) in capsys.readouterr().out


# -- Spec: Generated provider types, with no generated file ----------------------


def test_the_shipped_fallback_is_the_plain_providers_view() -> None:
    # A project that never generated a stub keeps exactly today's types.
    assert letify_providers.ProvidersView is letify.launcher.Providers
