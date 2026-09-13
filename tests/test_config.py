"""Reading .letify, and resolving credentials without writing them down.

Every test here traces to the "Configuration" section of docs/SPEC.md, which settles
three things: the home file holds accounts and the project file refines them, an alias
must be a Python identifier and some names are reserved, and a credential field names
where the value lives rather than holding it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import letify
from letify.config import CONFIG_NAME, load
from letify.config.secrets import from_keyring, resolve_secret


@pytest.fixture
def home_file(monkeypatch, tmp_path: Path):
    """Write a ~/.letify and point Path.home at the directory holding it."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

    def write(body: str) -> Path:
        path = home / CONFIG_NAME
        path.write_text(body, encoding="utf-8")
        return path

    return write


# -- Spec: Configuration -------------------------------------------------------


def test_the_project_file_refines_what_the_home_file_declared(home_file, config_file) -> None:
    # This is what lets a repository be cloned and run under someone else's accounts:
    # the account details stay in the home file and the project file changes the rest.
    home_file('[lab]\nkind = "shell"\naddress = "home.example.edu"\nuser = "researcher"\n')
    project = config_file('[lab]\nkind = "shell"\naddress = "project.example.edu"\n')

    config = load(project)

    entry = config.providers["lab"]
    assert entry.option("address") == "project.example.edu"
    assert entry.option("user") == "researcher"
    assert len(config.sources) == 2


def test_the_project_file_may_change_the_kind_of_a_declared_alias(home_file, config_file) -> None:
    home_file('[lab]\nkind = "shell"\naddress = "gpu.example.edu"\n')
    config = load(config_file('[lab]\nkind = "tunnel"\n'))
    assert config.providers["lab"].kind == "tunnel"
    assert config.providers["lab"].option("address") == "gpu.example.edu"


def test_the_home_file_can_be_left_out_entirely(home_file, config_file) -> None:
    # home=False is how a test stays isolated from the developer's own accounts.
    home_file('[secret_account]\nkind = "shell"\naddress = "private"\n')
    config = load(config_file('[lab]\nkind = "shell"\naddress = "a"\n'), home=False)
    assert "secret_account" not in config.providers


def test_the_local_provider_needs_no_declaration(config_file) -> None:
    config = load(config_file(""), home=False)
    assert config.providers["local"].kind == "local"


def test_declaration_order_sets_the_priority_for_any(config_file) -> None:
    config = load(
        config_file(
            '[first]\nkind = "shell"\naddress = "a"\n[second]\nkind = "shell"\naddress = "b"\n'
        ),
        home=False,
    )
    # local is appended last, because it was never declared.
    assert config.order == ["first", "second", "local"]


def test_the_defaults_table_is_merged_rather_than_treated_as_a_provider(config_file) -> None:
    config = load(config_file('[defaults]\nname = "study"\nmax_runtimes = 5\n'), home=False)
    assert config.defaults == {"name": "study", "max_runtimes": 5}
    assert "defaults" not in config.providers


@pytest.mark.parametrize("alias", ["any", "devices", "active"])
def test_a_reserved_alias_is_refused_and_says_why(alias: str, config_file) -> None:
    # These three names are already reached by attribute on let.providers, so an alias
    # taking one would shadow it.
    path = config_file(f'[{alias}]\nkind = "shell"\naddress = "a"\n')
    with pytest.raises(letify.ConfigError, match="reserved"):
        load(path, home=False)


def test_an_alias_with_a_hyphen_suggests_the_underscore_spelling(config_file) -> None:
    path = config_file('[colab-a]\nkind = "colab"\n')
    with pytest.raises(letify.ConfigError, match="Try 'colab_a'"):
        load(path, home=False)


def test_an_alias_that_is_not_an_identifier_is_refused(config_file) -> None:
    path = config_file('["9lab"]\nkind = "shell"\naddress = "a"\n')
    with pytest.raises(letify.ConfigError, match="not a Python identifier"):
        load(path, home=False)


def test_a_provider_entry_without_a_kind_is_refused(config_file) -> None:
    # kind is what selects the provider class, so there is nothing to build without it.
    path = config_file('[lab]\naddress = "gpu.example.edu"\n')
    with pytest.raises(letify.ConfigError, match="has no 'kind' field"):
        load(path, home=False)


def test_malformed_toml_names_the_file_it_could_not_read(config_file) -> None:
    path = config_file("[lab\nkind = broken\n")
    with pytest.raises(letify.ConfigError, match="is not valid TOML"):
        load(path, home=False)


def test_a_top_level_value_that_is_not_a_table_is_not_a_provider(config_file) -> None:
    config = load(config_file('version = 2\n[lab]\nkind = "shell"\naddress = "a"\n'), home=False)
    assert set(config.providers) == {"lab", "local"}


def test_a_missing_configuration_file_is_not_an_error(tmp_path: Path) -> None:
    # A project with no .letify still has the local machine.
    config = load(tmp_path / "absent", home=False)
    assert config.order == ["local"]
    assert config.sources == []


# -- Spec: Configuration, credential fields ------------------------------------


def test_a_secret_is_read_from_the_environment_first(monkeypatch) -> None:
    monkeypatch.setenv("LETIFY_TEST_TOKEN", "from-the-environment")
    options = {
        "access_token_env": "LETIFY_TEST_TOKEN",
        "access_token": "from-the-file",
    }
    assert resolve_secret(options, "access_token") == "from-the-environment"


def test_a_secret_falls_back_to_the_keyring(fake_keyring, monkeypatch) -> None:
    # The keyring is the second form, tried when the environment variable is unset.
    monkeypatch.delenv("LETIFY_TEST_TOKEN", raising=False)
    keyring = fake_keyring({("letify", "researcher"): "from-the-keyring"})
    options = {
        "access_token_env": "LETIFY_TEST_TOKEN",
        "access_token_keyring": "letify/researcher",
        "access_token": "from-the-file",
    }
    assert resolve_secret(options, "access_token") == "from-the-keyring"
    assert keyring.asked == [("letify", "researcher")]


def test_a_literal_secret_is_accepted_last(monkeypatch) -> None:
    # Only appropriate in ~/.letify, which is outside any repository.
    monkeypatch.setenv("LETIFY_TEST_TOKEN", "")
    options = {"access_token_env": "LETIFY_TEST_TOKEN", "access_token": "from-the-file"}
    assert resolve_secret(options, "access_token") == "from-the-file"


def test_a_credential_that_is_nowhere_returns_the_default() -> None:
    assert resolve_secret({}, "access_token") is None
    assert resolve_secret({}, "access_token", "fallback") == "fallback"


def test_a_keyring_entry_names_a_service_and_a_user(fake_keyring) -> None:
    fake_keyring({("letify", "researcher"): "value"})
    # Without the user half there is nothing to look up, so this is not a lookup miss.
    assert from_keyring("letify") is None


def test_the_keyring_package_is_optional(no_module) -> None:
    # keyring is an extra, so a missing install has to leave the other two forms working
    # rather than raising.
    no_module("keyring")
    assert from_keyring("letify/researcher") is None


def test_a_provider_entry_resolves_its_own_credentials(launcher_from, monkeypatch) -> None:
    monkeypatch.setenv("LETIFY_TEST_TOKEN", "secret-value")
    let = launcher_from(
        '[lab]\nkind = "shell"\naddress = "h"\naccess_token_env = "LETIFY_TEST_TOKEN"\n'
    )
    assert let.config.providers["lab"].secret("access_token") == "secret-value"
