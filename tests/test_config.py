"""Reading .letify, and resolving credentials without writing them down.

Every test here traces to the "Configuration" section of docs/SPEC.md, which settles
three things: the home file holds accounts and the project file refines them, an alias
must be a Python identifier and some names are reserved, and a credential field names
where the value lives rather than holding it.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import pytest

import letify
from letify.config import CONFIG_DIRECTORY, CONFIG_FILE, inventory, load, writer
from letify.config.secrets import account_directory, resolve_secret


@pytest.fixture
def home_file(monkeypatch, tmp_path: Path):
    """Write ~/.letify/config.toml and point Path.home at the directory holding it."""
    home = tmp_path / "home"
    (home / CONFIG_DIRECTORY).mkdir(parents=True)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

    def write(body: str) -> Path:
        path = home / CONFIG_DIRECTORY / CONFIG_FILE
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


def test_a_provider_entry_without_a_kind_is_refused(home_file, config_file) -> None:
    # kind is what selects the provider class, so there is nothing to build without it. A
    # project table may leave it out to take it from the home entry, so there the refusal
    # names how to declare the account instead.
    project = config_file('[lab]\naddress = "gpu.example.edu"\n')
    with pytest.raises(letify.ConfigError, match="letify login"):
        load(project, home=False)

    home_file('[lab]\naddress = "gpu.example.edu"\n')
    with pytest.raises(letify.ConfigError, match="has no 'kind' field"):
        load(project)


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


def test_letify_state_is_a_directory_at_home_and_in_the_project(home_file, tmp_path) -> None:
    # One directory per side, so a provider's credentials and state have somewhere to live
    # next to the configuration instead of in a file of their own.
    assert (CONFIG_DIRECTORY, CONFIG_FILE) == (".letify", "config.toml")
    home_file('[lab]\nkind = "shell"\naddress = "gpu.example.edu"\n')
    project = tmp_path / "project" / ".letify"
    project.mkdir(parents=True)
    (project / "config.toml").write_text("[lab]\n", encoding="utf-8")
    config = load(project)
    assert config.sources == [
        Path.home() / ".letify" / "config.toml",
        project / "config.toml",
    ]
    assert config.providers["lab"].option("address") == "gpu.example.edu"


def test_a_configuration_may_be_named_by_its_directory_or_its_file(config_file) -> None:
    directory = config_file('[box]\nkind = "local"\n')
    assert "box" in load(directory, home=False).providers
    assert "box" in load(directory / "config.toml", home=False).providers


def test_a_credential_is_read_from_the_account_directory(home_file) -> None:
    # The file is named after the field, in the alias's own directory, so a token never
    # appears in either config.toml.
    account = account_directory("elice_a100")
    assert account == Path.home() / ".letify" / "accounts" / "elice_a100"
    account.mkdir(parents=True)
    (account / "access_token").write_text("from-the-file\n", encoding="utf-8")
    assert resolve_secret({}, "access_token", alias="elice_a100") == "from-the-file"


def test_an_environment_variable_wins_over_the_account_file(home_file, monkeypatch) -> None:
    account = account_directory("elice_a100")
    account.mkdir(parents=True)
    (account / "access_token").write_text("from-the-file", encoding="utf-8")
    monkeypatch.setenv("LETIFY_TEST_TOKEN", "from-the-environment")
    options = {"access_token_env": "LETIFY_TEST_TOKEN"}
    assert resolve_secret(options, "access_token", alias="elice_a100") == "from-the-environment"


def test_a_provider_entry_resolves_its_own_account_directory(home_file, config_file) -> None:
    # A provider asks for a field by name and never needs to know where credentials live.
    account = account_directory("lab")
    account.mkdir(parents=True)
    (account / "auth_key").write_text("tskey", encoding="utf-8")
    home_file('[lab]\nkind = "tunnel"\naddress = "h"\n')
    entry = load(config_file("[lab]\n")).providers["lab"]
    assert entry.secret("auth_key") == "tskey"


def test_a_secret_is_read_from_the_environment_first(monkeypatch) -> None:
    monkeypatch.setenv("LETIFY_TEST_TOKEN", "from-the-environment")
    options = {
        "access_token_env": "LETIFY_TEST_TOKEN",
        "access_token": "from-the-file",
    }
    assert resolve_secret(options, "access_token") == "from-the-environment"


def test_a_literal_secret_is_accepted_last(monkeypatch) -> None:
    # Only appropriate in ~/.letify, which is outside any repository.
    monkeypatch.setenv("LETIFY_TEST_TOKEN", "")
    options = {"access_token_env": "LETIFY_TEST_TOKEN", "access_token": "from-the-file"}
    assert resolve_secret(options, "access_token") == "from-the-file"


def test_a_credential_that_is_nowhere_returns_the_default() -> None:
    assert resolve_secret({}, "access_token") is None
    assert resolve_secret({}, "access_token", "fallback") == "fallback"


def test_a_provider_entry_resolves_its_own_credentials(launcher_from, monkeypatch) -> None:
    monkeypatch.setenv("LETIFY_TEST_TOKEN", "secret-value")
    let = launcher_from(
        '[lab]\nkind = "shell"\naddress = "h"\naccess_token_env = "LETIFY_TEST_TOKEN"\n'
    )
    assert let.config.providers["lab"].secret("access_token") == "secret-value"


# -- Spec: Logging in ----------------------------------------------------------

#: A file with comments, a defaults table and two accounts, which is what a user who has
#: edited this by hand actually has.
ORIGINAL_FILE = """# my machines
[defaults]
name = "nvfp4"

[lab]
kind = "shell"
address = "old.example.edu"

# the fast one
[other]
kind = "modal"
"""


def test_an_alias_block_is_replaced_without_disturbing_the_rest_of_the_file() -> None:
    # Comments and unrelated accounts survive, because a user edits this file by hand.
    text = ORIGINAL_FILE
    updated = writer.write_block(text, "lab", {"kind": "shell", "address": "new.example.edu"})
    assert "# my machines" in updated
    assert "# the fast one" in updated
    assert "[other]" in updated
    assert 'name = "nvfp4"' in updated
    assert "new.example.edu" in updated
    assert "old.example.edu" not in updated
    assert updated.count("[lab]") == 1


def test_a_new_alias_block_is_appended_and_the_file_stays_parseable() -> None:
    updated = writer.write_block("", "lab", {"kind": "shell", "port": 2222, "persistent": True})
    parsed = tomllib.loads(updated)
    assert parsed["lab"] == {"kind": "shell", "port": 2222, "persistent": True}


def test_removing_an_alias_leaves_the_other_accounts_alone() -> None:
    text = writer.write_block(
        writer.write_block("", "a", {"kind": "modal"}), "b", {"kind": "local"}
    )
    updated = writer.remove_block(text, "a")
    parsed = tomllib.loads(updated)
    assert "a" not in parsed
    assert parsed["b"] == {"kind": "local"}


def test_a_value_is_written_in_the_toml_type_it_came_in_as() -> None:
    # A port written as a string would be a different value to whoever reads it back.
    updated = writer.write_block(
        "", "x", {"kind": "shell", "port": 22, "persistent": False, "gpus": ["A100", "H100"]}
    )
    parsed = tomllib.loads(updated)
    assert parsed["x"]["port"] == 22
    assert parsed["x"]["persistent"] is False
    assert parsed["x"]["gpus"] == ["A100", "H100"]


# -- Spec: The two files ---------------------------------------------------------


def test_a_home_account_the_project_does_not_name_is_not_available(home_file, config_file) -> None:
    # The home file is what this machine has. The project chooses from it.
    home_file('[lab]\nkind = "shell"\naddress = "gpu.example.edu"\n')
    config = load(config_file('[other]\nkind = "local"\n'))
    assert "lab" not in config.providers
    assert config.order == ["other", "local"]


def test_naming_the_alias_is_enough_to_use_a_home_account(home_file, config_file) -> None:
    # An empty table brings every setting with it, kind included.
    home_file('[lab]\nkind = "shell"\naddress = "gpu.example.edu"\nuser = "researcher"\n')
    config = load(config_file("[lab]\n"))
    entry = config.providers["lab"]
    assert entry.kind == "shell"
    assert entry.option("address") == "gpu.example.edu"
    assert entry.option("user") == "researcher"


def test_a_global_home_account_is_available_without_being_named(home_file, config_file) -> None:
    home_file(
        '[colab_pro]\nkind = "colab"\nglobal = true\n'
        '[lab]\nkind = "shell"\naddress = "gpu.example.edu"\n'
    )
    config = load(config_file('[other]\nkind = "local"\n'))
    assert "colab_pro" in config.providers
    assert "lab" not in config.providers
    # The marker says where an account is visible, not how to connect to it.
    assert config.providers["colab_pro"].option("global") is None


def test_with_no_project_file_only_global_accounts_and_local_exist(
    home_file, tmp_path, monkeypatch
) -> None:
    home_file(
        '[colab_pro]\nkind = "colab"\nglobal = true\n'
        '[lab]\nkind = "shell"\naddress = "gpu.example.edu"\n'
    )
    empty = tmp_path / "no-project"
    empty.mkdir()
    monkeypatch.chdir(empty)
    config = load()
    assert sorted(config.providers) == ["colab_pro", "local"]


def test_a_project_cannot_make_an_account_global(home_file, config_file) -> None:
    # Only the home file decides what every repository on the machine can reach, so the
    # marker is not read from a project and does not reach the provider.
    home_file('[lab]\nkind = "shell"\naddress = "gpu.example.edu"\n')
    config = load(config_file("[lab]\nglobal = true\n"))
    assert config.providers["lab"].option("global") is None


def test_naming_an_account_this_machine_does_not_have_says_what_to_run(
    home_file, config_file
) -> None:
    # With no kind in the project and no home entry, there is nothing to take the kind from.
    home_file('[other]\nkind = "local"\n')
    with pytest.raises(letify.ConfigError, match="letify login"):
        load(config_file("[lab]\n"))


def test_a_project_table_with_a_kind_needs_no_home_entry(config_file) -> None:
    config = load(config_file('[box]\nkind = "local"\n'), home=False)
    assert config.providers["box"].kind == "local"


def test_an_option_with_no_value_is_left_out_of_the_block() -> None:
    # Writing key = "" would declare an empty key path rather than no key path.
    updated = writer.write_block("", "x", {"kind": "shell", "user": None, "port": 22})
    assert "user" not in updated
    assert tomllib.loads(updated)["x"] == {"kind": "shell", "port": 22}


def test_removing_an_alias_that_is_not_there_changes_nothing() -> None:
    text = writer.write_block("", "a", {"kind": "modal"})
    assert writer.remove_block(text, "b") == text
    assert writer.drop(Path("nonexistent-file"), "a") is False


def test_a_dropped_alias_is_reported_as_dropped_only_when_it_was_there(tmp_path) -> None:
    file = tmp_path / ".letify"
    writer.update(file, "a", {"kind": "modal"})
    assert writer.drop(file, "b") is False
    assert writer.drop(file, "a") is True
    assert "[a]" not in file.read_text(encoding="utf-8")


def test_the_home_file_is_owner_only_where_the_platform_has_the_concept(tmp_path) -> None:
    # Windows permissions are access control lists, and chmod there sets the read only
    # flag, which is not what this means. So nothing is claimed on that platform.
    file = tmp_path / ".letify"
    writer.update(file, "a", {"kind": "modal"}, private=True)
    assert file.is_file()
    if not sys.platform.startswith("win"):
        assert file.stat().st_mode & 0o777 == writer.HOME_FILE_MODE


def test_a_quote_inside_a_value_does_not_end_the_string() -> None:
    # A path or a comment field with a quote in it must still parse.
    updated = writer.write_block("", "x", {"kind": "shell", "user": 'od"d'})
    assert tomllib.loads(updated)["x"]["user"] == 'od"d'


# -- Spec: Inventory -----------------------------------------------------------


def test_an_index_range_is_read_as_the_indices_it_names() -> None:
    # A range because a shared box is described that way by whoever hands it out: cards
    # zero through three are yours.
    assert inventory.read_indices("0-3") == (0, 1, 2, 3)
    assert inventory.read_indices([0, 1, 6]) == (0, 1, 6)
    assert inventory.read_indices("2") == (2,)
    assert inventory.read_indices("0-1,6-7") == (0, 1, 6, 7)


def test_an_index_range_that_is_not_one_is_refused_with_the_text() -> None:
    for bad in ("3-0", "a-b", "", "1-", [1, "x"]):
        with pytest.raises(letify.ConfigError, match="indices"):
            inventory.read_indices(bad)


def test_a_declared_count_needs_no_indices() -> None:
    # Colab assigns the device itself, so there is nothing to index and only a count to
    # declare.
    entry = inventory.Devices.read("G4", {"count": 2})
    assert entry.count == 2
    assert entry.indices == ()
    assert entry.chooses_indices is False


def test_declared_indices_are_the_count() -> None:
    entry = inventory.Devices.read("A100", {"indices": "0-3"})
    assert entry.indices == (0, 1, 2, 3)
    assert entry.count == 4
    assert entry.chooses_indices is True


def test_an_accelerator_with_neither_is_one_of_it() -> None:
    assert inventory.Devices.read("A100", {}).count == 1
    assert inventory.Devices.read("A100", {}).indices == ()


def test_a_count_that_disagrees_with_the_indices_is_refused() -> None:
    # Two statements of one fact, and no way to tell which the user meant.
    with pytest.raises(letify.ConfigError, match="count"):
        inventory.Devices.read("A100", {"indices": "0-3", "count": 2})
    # Agreeing is accepted, because a user restating it is not a mistake.
    assert inventory.Devices.read("A100", {"indices": "0-3", "count": 4}).count == 4


def test_a_device_table_becomes_the_providers_inventory(launcher_from) -> None:
    let = launcher_from(
        '[lab]\nkind = "shell"\naddress = "gpu.example.edu"\n'
        '[lab.devices]\nA100 = { indices = "0-3" }\nH100 = { count = 1 }\n'
    )
    lab = let.provider("lab")
    assert sorted(lab.inventory) == ["A100", "H100"]
    assert lab.inventory["A100"].indices == (0, 1, 2, 3)
    assert lab.inventory["H100"].count == 1
    # The table is also the accelerator list, so nothing has to be said twice.
    assert sorted(lab.instances) == ["A100", "H100"]


def test_the_older_gpu_list_still_means_one_of_each(launcher_from) -> None:
    let = launcher_from('[lab]\nkind = "shell"\naddress = "h"\ngpus = ["A100", "H100"]\n')
    lab = let.provider("lab")
    assert lab.inventory["A100"].count == 1
    assert lab.inventory["A100"].chooses_indices is False


# -- Spec: Workspace root ------------------------------------------------------


def test_a_workspace_is_read_from_the_home_file(home_file, config_file) -> None:
    home_file('[lab]\nkind = "shell"\naddress = "gpu.example.edu"\nworkspace = "/workspace/me"\n')
    project = config_file("[lab]\n")
    assert load(project).providers["lab"].option("workspace") == "/workspace/me"


def test_a_project_file_that_sets_a_workspace_is_refused(home_file, config_file) -> None:
    # A repository cannot know the write rules of every machine its users reach.
    home_file('[lab]\nkind = "shell"\naddress = "gpu.example.edu"\n')
    project = config_file('[lab]\nworkspace = "/workspace/me"\n')
    with pytest.raises(letify.ConfigError, match=r"workspace.*home"):
        load(project)
