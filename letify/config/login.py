"""Declaring one account, and setting up whatever it needs to be reachable.

Two files are written, because they answer different questions. ``~/.letify`` gets the
account, since a connection detail belongs to the machine. The project's ``.letify`` gets the
alias as an empty table, and nothing else, since everything else is either a secret or a
detail of one person's machine. Naming the alias is what makes the account available in the
project, and it is what makes a repository self describing: a teammate who clones it can see
which accounts it needs and run this command for them.

No credential is written to either config.toml. A token goes to a file in
``~/.letify/accounts/<alias>/``, readable by its owner only. An SSH password is
not stored at all: letify opens sessions with ``BatchMode=yes``, because a session is
started by the pool in the background with nobody present to answer a prompt, so the
password is accepted once, used to install a key, and dropped.

A shell or tunnel login also asks the machine for its GPUs once, over the confirmed key,
and writes the cards the user chose to ``[<alias>.devices]`` in the home file. What a
session later reserves from that table is the inventory's business, not this module's.
"""

from __future__ import annotations

import getpass
import subprocess
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..errors import ConfigError, LetifyError
from ..providers.naming import gib_from_mib, normalize_gpu
from . import secrets, writer
from .inventory import read_indices
from .schema import RESERVED_ALIASES
from .secrets import CONFIG_DIRECTORY

#: The configuration file inside each ``.letify`` directory.
CONFIG_FILE = "config.toml"


#: Where a key generated for letify goes. Its own name, so it is never confused with a key
#: the user made for something else and never regenerated over one.
DEFAULT_KEY = "~/.ssh/id_letify"

#: How a shell account authenticates. A key is the default because it is the only method
#: that works unattended on every platform letify runs on.
AUTH_METHODS = ("key", "password")


class LoginError(ConfigError):
    """The account could not be declared, with nothing written."""


@dataclass
class Answers:
    """What the command was told, on the command line or at the prompt."""

    alias: str
    kind: str
    values: dict[str, Any] = field(default_factory=dict)
    token: str | None = None
    interactive: bool = True
    install_key: bool = True

    def get(self, name: str) -> Any:
        return self.values.get(name)


# -- reading from the terminal, kept behind functions so a test can refuse them --


def read_line(prompt: str) -> str:
    return input(prompt).strip()


def read_password(prompt: str) -> str:
    return getpass.getpass(prompt)


def ask(answers: Answers, name: str, prompt: str, *, required: bool = True) -> str | None:
    """Return a value, prompting for it only when it was not given and may be asked for."""
    value = answers.get(name)
    if isinstance(value, str) and value:
        return value
    if not answers.interactive:
        if required:
            raise LoginError(
                f"{answers.alias} needs {name!r}. Pass --{name.replace('_', '-')} or drop "
                f"--no-input so it can be asked for."
            )
        return None
    answer = read_line(prompt)
    if answer:
        return answer
    if required:
        raise LoginError(f"{answers.alias} needs {name!r}, and nothing was entered.")
    return None


# -- SSH key setup -------------------------------------------------------------


def expand(path: str) -> Path:
    return Path(path).expanduser()


def ensure_key(key_path: str) -> Path:
    """Return the private key, generating an ed25519 pair if it is not there.

    Never generated over an existing key, because that would lock the user out of every
    other machine that already trusts it. No passphrase, because a passphrase puts the
    prompt back into a path that has nobody to answer it.
    """
    private = expand(key_path)
    if private.exists():
        return private
    private.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "ssh-keygen",
            "-t",
            "ed25519",
            "-N",
            "",
            "-C",
            "letify",
            "-f",
            str(private),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise LoginError(f"ssh-keygen failed: {result.stderr.strip() or result.stdout.strip()}")
    writer.restrict(private)
    return private


def install_key(*, address: str, user: str | None, port: int, key_path: str) -> None:
    """Append the public key to the machine's authorized_keys, over one connection.

    The password is typed here and used by this one command. It is not written to a file,
    the account directory or the environment, and it is not passed on the command line, where it
    would be visible to anything that can list processes.
    """
    public = expand(key_path).with_suffix(".pub")
    if not public.is_file():
        raise LoginError(f"{public} is missing, so there is no public key to install.")
    material = public.read_text(encoding="utf-8").strip()
    target = f"{user}@{address}" if user else address

    # Appended only when it is not already there, so running login twice does not grow the
    # file. The key arrives on stdin rather than in the command, which keeps it out of the
    # remote process list as well.
    remote = (
        "mkdir -p ~/.ssh && chmod 700 ~/.ssh && "
        "touch ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys && "
        'key=$(cat) && grep -qxF "$key" ~/.ssh/authorized_keys || '
        'printf "%s\\n" "$key" >> ~/.ssh/authorized_keys'
    )
    result = subprocess.run(
        ["ssh", "-p", str(port), "-o", "StrictHostKeyChecking=accept-new", target, remote],
        input=material,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        raise LoginError(
            f"installing the key on {target} failed: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )


def batch_ssh(
    *, address: str, user: str | None, port: int, key_path: str, remote: str
) -> list[str]:
    """The SSH command a pooled session would run, with no way to prompt."""
    target = f"{user}@{address}" if user else address
    return [
        "ssh",
        "-p",
        str(port),
        "-i",
        str(expand(key_path)),
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        target,
        remote,
    ]


def confirm_key(*, address: str, user: str | None, port: int, key_path: str) -> None:
    """Prove the key works before the alias is declared.

    With BatchMode, so what is proven is exactly what a pooled session will do. Failing
    here beats failing at the first call, which costs GPU time to find out.
    """
    target = f"{user}@{address}" if user else address
    result = subprocess.run(
        batch_ssh(address=address, user=user, port=port, key_path=key_path, remote="echo letify"),
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise LoginError(
            f"the key does not let letify into {target} without a prompt: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )


# -- the workspace root --------------------------------------------------------


def valid_workspace(alias: str, value: str) -> str:
    """Return the root when it is an absolute path or starts with ``~``, and refuse it otherwise."""
    if not value.startswith(("/", "~")):
        raise LoginError(
            f"{alias}: workspace must be an absolute path or start with ~, not {value!r}"
        )
    return value


def choose_workspace(answers: Answers, address: str | None) -> tuple[str, str]:
    """The root a shell or tunnel account writes under, and the default it was offered.

    ``--workspace`` decides it. Otherwise a terminal is asked with the default in brackets,
    and a script or a blank answer takes the default.
    """
    from ..runtime.bootstrap import DEFAULT_WORKSPACE_ROOT

    default = DEFAULT_WORKSPACE_ROOT
    given = answers.get("workspace")
    if isinstance(given, str) and given:
        chosen = given
    elif answers.interactive:
        chosen = read_line(f"Workspace root on {address} [{default}]: ") or default
    else:
        chosen = default
    return valid_workspace(answers.alias, chosen), default


def check_workspace(
    *, address: str, user: str | None, port: int, key_path: str, workspace: str
) -> None:
    """Prove the root can be created and written as the account's own user, over BatchMode."""
    from ..runtime.bootstrap import workspace_check

    target = f"{user}@{address}" if user else address
    command = batch_ssh(
        address=address, user=user, port=port, key_path=key_path, remote=workspace_check(workspace)
    )
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as exc:
        raise LoginError(
            f"the workspace {workspace} could not be checked on {target}: {exc}"
        ) from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}"
        raise LoginError(
            f"the workspace {workspace} cannot be created and written on {target} without "
            f"root, so nothing was written: {detail}"
        )
    print(f"workspace {workspace}: writable")


def record_workspace(answers: Answers, options: dict[str, Any]) -> None:
    """Write ``--workspace`` to an account whose kind has no machine to check it on."""
    given = answers.get("workspace")
    if isinstance(given, str) and given:
        options["workspace"] = valid_workspace(answers.alias, given)


# -- GPUs found at login -------------------------------------------------------

#: The one query run at login. ``index`` is the physical position the inventory names.
DEVICE_QUERY = "nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader"

#: What a machine with no devices table falls back to, said whenever nothing is recorded.
FALLBACK_NOTE = (
    "No devices table was written, so letify asks the machine for its GPUs at first use."
)


@dataclass(frozen=True)
class FoundDevices:
    """The cards of one normalized accelerator name, with their physical indices."""

    name: str
    indices: tuple[int, ...]
    memory_gb: int | None = None

    def describe(self) -> str:
        """One line such as ``A100: 4 cards, indices 0-3 (80 GB each)``."""
        many = len(self.indices) > 1
        line = (
            f"{self.name}: {len(self.indices)} card{'s' if many else ''}, "
            f"{'indices' if many else 'index'} {compact_indices(self.indices)}"
        )
        if self.memory_gb:
            line += f" ({self.memory_gb} GB{' each' if many else ''})"
        return line


def compact_indices(indices: tuple[int, ...] | list[int]) -> str:
    """``"0-3"`` for a contiguous run, ``"0,1,6"`` otherwise, both forms the inventory reads."""
    ordered = sorted(indices)
    if len(ordered) > 1 and ordered[-1] - ordered[0] == len(ordered) - 1:
        return f"{ordered[0]}-{ordered[-1]}"
    return ",".join(str(index) for index in ordered)


def group_devices(output: str) -> list[FoundDevices]:
    """Group nvidia-smi rows by normalized name, in the order the names first appear."""
    indices: dict[str, list[int]] = {}
    memory: dict[str, int | None] = {}
    for line in output.splitlines():
        index, _, rest = line.partition(",")
        name, _, total = rest.partition(",")
        if not index.strip().isdigit() or not name.strip():
            continue
        label = normalize_gpu(name)
        indices.setdefault(label, []).append(int(index))
        size = gib_from_mib(total)
        if size is not None and (memory.get(label) or 0) < size:
            memory[label] = size
    return [
        FoundDevices(label, tuple(sorted(found)), memory.get(label))
        for label, found in indices.items()
    ]


def detect_devices(
    *, address: str, user: str | None, port: int, key_path: str
) -> tuple[list[FoundDevices], str | None]:
    """Ask the machine for its GPUs once. Returns the groups, or none and the reason.

    A failure here never fails the login, because the account works without a table.
    """
    target = f"{user}@{address}" if user else address
    command = batch_ssh(
        address=address, user=user, port=port, key_path=key_path, remote=DEVICE_QUERY
    )
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as exc:
        return [], f"nvidia-smi could not be run on {target}: {exc}"
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}"
        return [], f"nvidia-smi did not answer on {target}: {detail}"
    groups = group_devices(result.stdout)
    if not groups:
        return [], f"nvidia-smi reports no GPU on {target}"
    return groups, None


def _chosen(group: FoundDevices, spec: str) -> tuple[int, ...]:
    """Read a choice of indices and refuse any the machine does not have."""
    try:
        indices = read_indices(spec)
    except ConfigError as exc:
        raise LoginError(f"{group.name}: {exc}") from None
    missing = [index for index in indices if index not in group.indices]
    if missing:
        raise LoginError(
            f"{group.name} has no card at index {', '.join(str(i) for i in missing)}. "
            f"The machine has {compact_indices(group.indices)}."
        )
    return indices


def choose_devices(answers: Answers, groups: list[FoundDevices]) -> dict[str, dict[str, str]]:
    """Decide which indices of each accelerator letify may use, and return the table.

    ``--indices NAME=SPEC`` decides one name. Otherwise a terminal is asked, and a script
    or a blank answer takes every card found.
    """
    given: dict[str, str] = {}
    for item in answers.get("indices") or []:
        name, separator, spec = str(item).partition("=")
        if not separator or not name.strip() or not spec.strip():
            raise LoginError(f"--indices takes NAME=SPEC, such as A100=0-3, not {item!r}")
        given[name.strip()] = spec.strip()
    found = {group.name: group for group in groups}
    unknown = sorted(set(given) - set(found))
    if unknown:
        raise LoginError(
            f"--indices names {', '.join(unknown)}, which the machine does not have. "
            f"It has {', '.join(found)}."
        )

    for group in groups:
        print(group.describe())
    table: dict[str, dict[str, str]] = {}
    for group in groups:
        if group.name in given:
            indices = _chosen(group, given[group.name])
        elif answers.interactive:
            indices = _ask_indices(group)
        else:
            indices = group.indices
        table[group.name] = {"indices": compact_indices(indices)}
    return table


def _ask_indices(group: FoundDevices) -> tuple[int, ...]:
    prompt = f"Indices letify may use for {group.name} [{compact_indices(group.indices)}]: "
    while True:
        answer = read_line(prompt)
        if not answer:
            return group.indices
        try:
            return _chosen(group, answer)
        except LoginError as exc:
            print(exc)


def record_devices(
    answers: Answers, *, address: str, user: str | None, port: int, key_path: str
) -> dict[str, dict[str, str]] | None:
    """Detect the machine's GPUs and return the chosen table, or ``None`` with a note."""
    groups, reason = detect_devices(address=address, user=user, port=port, key_path=key_path)
    if not groups:
        print(f"{reason}. {FALLBACK_NOTE}")
        return None
    return choose_devices(answers, groups)


# -- one flow per kind ---------------------------------------------------------


def shell_account(answers: Answers) -> dict[str, Any]:
    """Set up an SSH reachable machine and return what to write."""
    auth = str(answers.get("auth") or "key")
    if auth not in AUTH_METHODS:
        raise LoginError(f"auth must be one of {', '.join(AUTH_METHODS)}, not {auth!r}")
    if auth == "password" and sys.platform.startswith("win"):
        raise LoginError(
            "auth='password' needs sshpass, which has no Windows build. Use the default "
            "key authentication, which letify sets up for you, and the password is asked "
            "for once and never stored."
        )

    address = ask(answers, "address", "Machine address: ")
    user = ask(answers, "user", "SSH user (blank for the local name): ", required=False)
    port = int(answers.get("port") or 22)
    key_path = str(answers.get("key") or DEFAULT_KEY)

    options: dict[str, Any] = {"kind": answers.kind, "address": address}
    if user:
        options["user"] = user
    if port != 22:
        options["port"] = port
    if answers.get("persistent") is not None:
        options["persistent"] = bool(answers.get("persistent"))

    if auth == "password":
        options["auth"] = "password"
        # No BatchMode connection exists to check the root with, so it is only recorded.
        workspace, default = choose_workspace(answers, address)
        if workspace != default:
            options["workspace"] = workspace
        password = answers.token or (
            read_password(f"Password for {address}: ") if answers.interactive else None
        )
        if not password:
            raise LoginError(f"{answers.alias} needs a password to store for sshpass.")
        store_secret(answers.alias, "password", password)
        return options

    options["key"] = key_path
    if answers.install_key:
        ensure_key(key_path)
        install_key(address=address, user=user, port=port, key_path=key_path)
    confirm_key(address=address, user=user, port=port, key_path=key_path)
    workspace, default = choose_workspace(answers, address)
    check_workspace(address=address, user=user, port=port, key_path=key_path, workspace=workspace)
    if workspace != default:
        options["workspace"] = workspace
    devices = record_devices(answers, address=address, user=user, port=port, key_path=key_path)
    if devices:
        # Written by log_in as its own [<alias>.devices] table, not as a field of the account.
        options["devices"] = devices
    return options


def elice_account(answers: Answers) -> dict[str, Any]:
    """Record the zone and machine, and keep the access token in the account directory."""
    zone = ask(answers, "zone_id", "Elice zone id: ")
    machine = ask(answers, "machine_id", "Elice machine id: ")
    token = answers.token or (
        read_password("Elice access token: ") if answers.interactive else None
    )
    if not token:
        raise LoginError(f"{answers.alias} needs an access token. Pass --token or drop --no-input.")
    store_secret(answers.alias, "access_token", token)
    options: dict[str, Any] = {
        "kind": answers.kind,
        "zone_id": zone,
        "machine_id": machine,
    }
    endpoint = answers.get("endpoint")
    if isinstance(endpoint, str) and endpoint:
        options["endpoint"] = endpoint
    record_workspace(answers, options)
    return options


def modal_account(answers: Answers) -> dict[str, Any]:
    """Sign in to Modal by running ``modal token new`` through uv.

    ``MODAL_CONFIG_PATH`` points at ``~/.letify/accounts/<alias>/modal.toml``, so the token
    Modal writes lands in the account directory and the Modal adapter reads it from there.
    The command prints a link and waits for the browser approval, so its output is not
    captured. A sign in that fails, or that leaves no token file, writes nothing and removes
    a token file the attempt created.
    """
    from .. import tools

    uv = tools.find_uv()
    if uv is None:
        raise LoginError(tools.missing_uv_message())
    options: dict[str, Any] = {"kind": answers.kind}
    record_workspace(answers, options)
    profile = ask(
        answers,
        "profile",
        "Modal profile, naming the Modal workspace (blank for the default): ",
        required=False,
    )
    arguments = ["token", "new"]
    if profile:
        options["profile"] = profile
        arguments += ["--profile", profile]
    env = tools.modal_environment(answers.alias)
    token = Path(env["MODAL_CONFIG_PATH"])
    existed = token.exists()
    result = subprocess.run([*tools.command(tools.MODAL, uv), *arguments], env=env)
    if result.returncode != 0 or not token.is_file():
        if not existed:
            token.unlink(missing_ok=True)
        if result.returncode != 0:
            raise LoginError(
                f"the Modal sign in exited {result.returncode}, so nothing was written"
            )
        raise LoginError(
            f"the Modal sign in finished without writing {token}, so nothing was written"
        )
    writer.restrict(token)
    return options


def colab_account(answers: Answers) -> dict[str, Any]:
    """Sign in to Colab by running the Colab CLI's own login through uv.

    The CLI keeps its token under its home directory, and letify runs it with the account
    directory as that home, so the token ends up in ``~/.letify/accounts/<alias>/``. The
    CLI refreshes the token itself on later calls. ``colab sessions`` is the command run,
    because it signs in when there is no token and changes nothing when there is one.
    """
    from .. import tools

    uv = tools.find_uv()
    if uv is None:
        raise LoginError(tools.missing_uv_message())
    options: dict[str, Any] = {"kind": answers.kind}
    account = ask(answers, "account", "Google account email: ", required=False)
    if account:
        options["account"] = account
    record_workspace(answers, options)
    result = subprocess.run(
        [*tools.command(tools.COLAB, uv), "sessions"],
        env=tools.environment(answers.alias),
    )
    if result.returncode != 0:
        raise LoginError(f"the Colab sign in exited {result.returncode}, so nothing was written")
    return options


FLOWS = {
    "shell": shell_account,
    "tunnel": shell_account,
    "elice": elice_account,
    "colab": colab_account,
    "modal": modal_account,
}


# -- credentials ---------------------------------------------------------------


def store_secret(alias: str, name: str, secret: str) -> None:
    """Keep a credential in ``~/.letify/accounts/<alias>/<name>``, readable by its owner only."""
    secrets.write_secret(alias, name, secret)


def forget_secret(alias: str) -> bool:
    """Delete everything this machine holds for an account. Returns whether anything was there."""
    return secrets.forget_account(alias)


# -- what the command line calls ----------------------------------------------


def home_path() -> Path:
    return Path.home() / CONFIG_DIRECTORY / CONFIG_FILE


def project_path(path: str | Path | None = None) -> Path:
    if path is None:
        return Path.cwd() / CONFIG_DIRECTORY / CONFIG_FILE
    given = Path(path)
    return given / CONFIG_FILE if given.is_dir() else given


def check_alias(alias: str) -> None:
    if alias in RESERVED_ALIASES:
        raise LoginError(
            f"{alias!r} is reserved, because let.providers.{alias} already means "
            f"something else. Pick another alias."
        )
    if not alias.isidentifier():
        hint = f" Try {alias.replace('-', '_')!r}." if "-" in alias else ""
        raise LoginError(
            f"alias {alias!r} is not a Python identifier, so let.providers.{alias} "
            f"cannot work.{hint}"
        )


def log_in(answers: Answers, *, project: str | Path | None = None) -> tuple[bool, Path, Path]:
    """Declare the account and reference it from the project.

    Returns whether the account was newly written, and the two files touched. An account
    already in the home file is not asked for again, which is the common case in a second
    repository: the account was set up once and this project just needs to name it.
    """
    check_alias(answers.alias)
    if answers.kind not in FLOWS:
        known = ", ".join(sorted(FLOWS))
        raise LoginError(f"there is no login for kind {answers.kind!r}. Known kinds: {known}.")

    home = home_path()
    existing = home.read_text(encoding="utf-8") if home.is_file() else ""
    fresh = not writer.has_block(existing, answers.alias)
    devices = None
    if fresh:
        options = FLOWS[answers.kind](answers)
        devices = options.pop("devices", None)
        writer.update(home, answers.alias, options, private=True)
    else:
        if answers.get("workspace"):
            change_workspace(answers, existing, home)
        if answers.get("detect_devices"):
            devices = detect_again(answers, existing, home)
    if devices:
        writer.update(home, devices_table(answers.alias), devices, private=True)

    # An empty table names the account and carries no connection detail, so it is safe in a
    # repository. A table that is already there is left alone, because the project may have
    # overridden settings in it.
    target = project_path(project)
    existing_project = target.read_text(encoding="utf-8") if target.is_file() else ""
    if not writer.has_block(existing_project, answers.alias):
        writer.update(target, answers.alias, {})
    if devices:
        refresh_stubs(project)
    return fresh, home, target


def devices_table(alias: str) -> str:
    return f"{alias}.devices"


def change_workspace(answers: Answers, text: str, home: Path) -> None:
    """Check a new root for an already declared account and write it once it passes.

    A shell or tunnel account with a key and address is checked over SSH. Any other account
    has no machine to reach at login, so the root is recorded as given.
    """
    entry = tomllib.loads(text).get(answers.alias, {})
    workspace = valid_workspace(answers.alias, str(answers.get("workspace")))
    kind = entry.get("kind", answers.kind)
    if kind in ("shell", "tunnel") and entry.get("auth") != "password" and entry.get("address"):
        check_workspace(
            address=str(entry["address"]),
            user=entry.get("user"),
            port=int(entry.get("port") or 22),
            key_path=str(entry.get("key") or DEFAULT_KEY),
            workspace=workspace,
        )
    # The devices table is its own [<alias>.devices] block, which this write leaves alone.
    body = {key: value for key, value in entry.items() if key != "devices"}
    body["workspace"] = workspace
    writer.update(home, answers.alias, body, private=True)


def detect_again(answers: Answers, text: str, home: Path) -> dict[str, dict[str, str]] | None:
    """Ask an already declared machine for its GPUs, and return the table once confirmed."""
    entry = tomllib.loads(text).get(answers.alias, {})
    kind = entry.get("kind", answers.kind)
    if kind not in ("shell", "tunnel"):
        raise LoginError(f"--detect-devices applies to shell and tunnel accounts, not {kind!r}")
    if entry.get("auth") == "password" or not entry.get("address"):
        raise LoginError(
            f"{answers.alias} has no key and address to ask the machine with, so its GPUs "
            f"cannot be detected. Declare them in [{devices_table(answers.alias)}] by hand."
        )
    devices = record_devices(
        answers,
        address=str(entry["address"]),
        user=entry.get("user"),
        port=int(entry.get("port") or 22),
        key_path=str(entry.get("key") or DEFAULT_KEY),
    )
    if devices and answers.interactive:
        reply = read_line(f"Replace the devices table of {answers.alias} in {home}? [y/N] ")
        if not reply.lower().startswith("y"):
            print("The devices table was left as it was.")
            return None
    return devices


def refresh_stubs(project: str | Path | None) -> None:
    """Regenerate the provider types, so an editor completes the names just recorded.

    Building a launcher is what regenerates them. A stub that cannot be written never fails
    a login that has already written the account.
    """
    from ..launcher import Launcher

    try:
        Launcher(project, announce=False)
    except (OSError, LetifyError):
        pass


def log_out(alias: str) -> tuple[bool, bool]:
    """Take the account off this machine, leaving the project reference alone.

    The repository still needs that account. What changed is that this machine no longer
    has it, so the reference stays and says what to run to get it back.
    """
    removed = writer.drop(home_path(), alias)
    # A devices table left behind would declare an account with no kind.
    writer.drop(home_path(), devices_table(alias))
    forgotten = forget_secret(alias)
    return removed, forgotten


__all__ = [
    "AUTH_METHODS",
    "DEFAULT_KEY",
    "DEVICE_QUERY",
    "Answers",
    "FoundDevices",
    "LoginError",
    "ask",
    "batch_ssh",
    "check_alias",
    "choose_devices",
    "compact_indices",
    "confirm_key",
    "detect_devices",
    "ensure_key",
    "forget_secret",
    "group_devices",
    "home_path",
    "install_key",
    "log_in",
    "log_out",
    "project_path",
    "read_line",
    "read_password",
    "record_devices",
    "refresh_stubs",
    "store_secret",
]
