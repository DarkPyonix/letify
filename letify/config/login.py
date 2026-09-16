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
import json
import subprocess
import sys
import tomllib
from collections.abc import Callable
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


def install_key(
    *, address: str, user: str | None, port: int, key_path: str, proxy: str | None = None
) -> None:
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
        [
            "ssh",
            "-p",
            str(port),
            "-o",
            "StrictHostKeyChecking=accept-new",
            *proxy_options(proxy),
            target,
            remote,
        ],
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
    *,
    address: str,
    user: str | None,
    port: int,
    key_path: str,
    remote: str,
    proxy: str | None = None,
) -> list[str]:
    """The SSH command a pooled session would run, with no way to prompt.

    ``proxy`` is a ProxyCommand, such as ``tailcat <address> <port>`` for a tunnel account.
    """
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
        *proxy_options(proxy),
        target,
        remote,
    ]


def proxy_options(proxy: str | None) -> list[str]:
    return ["-o", f"ProxyCommand={proxy}"] if proxy else []


def confirm_key(
    *, address: str, user: str | None, port: int, key_path: str, proxy: str | None = None
) -> None:
    """Prove the key works before the alias is declared.

    With BatchMode, so what is proven is exactly what a pooled session will do. Failing
    here beats failing at the first call, which costs GPU time to find out.
    """
    target = f"{user}@{address}" if user else address
    result = subprocess.run(
        batch_ssh(
            address=address,
            user=user,
            port=port,
            key_path=key_path,
            remote="echo letify",
            proxy=proxy,
        ),
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
    *,
    address: str,
    user: str | None,
    port: int,
    key_path: str,
    workspace: str,
    proxy: str | None = None,
) -> None:
    """Prove the root can be created and written as the account's own user, over BatchMode."""
    from ..runtime.bootstrap import workspace_check

    target = f"{user}@{address}" if user else address
    command = batch_ssh(
        address=address,
        user=user,
        port=port,
        key_path=key_path,
        remote=workspace_check(workspace),
        proxy=proxy,
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
    *, address: str, user: str | None, port: int, key_path: str, proxy: str | None = None
) -> tuple[list[FoundDevices], str | None]:
    """Ask the machine for its GPUs once. Returns the groups, or none and the reason.

    A failure here never fails the login, because the account works without a table.
    """
    target = f"{user}@{address}" if user else address
    command = batch_ssh(
        address=address, user=user, port=port, key_path=key_path, remote=DEVICE_QUERY, proxy=proxy
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
    answers: Answers,
    *,
    address: str,
    user: str | None,
    port: int,
    key_path: str,
    proxy: str | None = None,
) -> dict[str, dict[str, str]] | None:
    """Detect the machine's GPUs and return the chosen table, or ``None`` with a note."""
    groups, reason = detect_devices(
        address=address, user=user, port=port, key_path=key_path, proxy=proxy
    )
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


#: The last line of the Elice machine menu, which records no machine.
CREATE_MACHINE = "Create a new machine with letify"


def elice_account(answers: Answers) -> dict[str, Any]:
    """Check the access token with eci, choose the zone and any machine, and keep the token.

    Spec "Logging in". Nothing is written until eci has accepted the token and the zone,
    so a mistyped token leaves no account and no secret file behind. No machine has to
    exist: without one, letify creates its own on first use.
    """
    from ..providers import elice

    confirm = None
    if answers.interactive:

        def confirm() -> bool:
            return read_line(elice.install_question()).lower() in ("y", "yes")

    try:
        binary = elice.find_eci(confirm=confirm)
    except LetifyError as exc:
        raise LoginError(str(exc)) from None
    given = answers.get("endpoint")
    endpoint = given if isinstance(given, str) and given else elice.DEFAULT_ENDPOINT
    token = answers.token or (
        read_password("Elice access token: ") if answers.interactive else None
    )
    if not token:
        raise LoginError(f"{answers.alias} needs an access token. Pass --token or drop --no-input.")
    unzoned = elice.eci_environment(answers.alias, token, endpoint, None)
    try:
        zones = elice.items(elice.eci(binary, ["zone", "list"], unzoned))
    except LetifyError as exc:
        raise LoginError(f"Elice refused the access token, so nothing was written: {exc}") from None

    zone = choose(answers, "zone_id", "zone", lambda: zones)
    env = elice.eci_environment(answers.alias, token, endpoint, zone)
    try:
        elice.eci(binary, ["config", "verify"], env, parse=False)
    except LetifyError as exc:
        raise LoginError(f"eci config verify failed, so nothing was written: {exc}") from None

    machine = answers.get("machine_id")
    if not (isinstance(machine, str) and machine):
        machine = None
        try:
            listed = [m for m in elice.items(elice.eci(binary, ["compute", "vm", "list"], env))]
        except LetifyError as exc:
            raise LoginError(f"could not list Elice machines: {exc}") from None
        listed = [m for m in listed if m.get("id")]
        if not listed:
            print("Elice lists no machine; letify creates one on first use.")
        elif answers.interactive:
            machine = choose_machine(listed)

    options: dict[str, Any] = {"kind": answers.kind, "zone_id": zone}
    if machine:
        options["machine_id"] = machine
    price_type = answers.get("price_type")
    if price_type:
        options["price_type"] = price_type
    if endpoint != elice.DEFAULT_ENDPOINT:
        options["endpoint"] = endpoint
    short_name = answers.get("organization")
    if not (isinstance(short_name, str) and short_name):
        try:
            organization = elice.eci(binary, ["org", "info"], env)
        except LetifyError:
            organization = None
        if isinstance(organization, dict):
            short_name = organization.get("name_short") or organization.get("nameShort")
    if isinstance(short_name, str) and short_name:
        options["organization"] = short_name
    billing = ask(
        answers, "billing_endpoint", "Elice billing API base URL (blank to skip): ", required=False
    )
    if billing:
        options["billing_endpoint"] = billing
    key_path = str(answers.get("key") or DEFAULT_KEY)
    ensure_key(key_path)
    options["key"] = key_path
    record_workspace(answers, options)
    store_secret(answers.alias, "access_token", token)
    return options


def choose_machine(listed: list[dict[str, Any]]) -> str | None:
    """A listed machine's id picked by number, or None for the last choice, create one."""
    for number, item in enumerate(listed, start=1):
        print(f"{number}. {item.get('name') or item['id']} ({item['id']})")
    last = len(listed) + 1
    print(f"{last}. {CREATE_MACHINE}")
    prompt = f"Elice machine [1-{last}]: "
    while True:
        answer = read_line(prompt)
        if answer.isdigit() and 1 <= int(answer) <= last:
            return None if int(answer) == last else str(listed[int(answer) - 1]["id"])
        print(f"Enter a number from 1 to {last}.")


def choose(
    answers: Answers, name: str, noun: str, fetch: Callable[[], list[dict[str, Any]]]
) -> str:
    """A given id, or one picked by number from what the API lists."""
    value = answers.get(name)
    if isinstance(value, str) and value:
        return value
    if not answers.interactive:
        raise LoginError(
            f"{answers.alias} needs {name!r}. Pass --{name.replace('_', '-')} or drop "
            f"--no-input so it can be chosen."
        )
    found = [item for item in fetch() if item.get("id")]
    if not found:
        raise LoginError(f"Elice lists no {noun} for this token, so there is nothing to choose.")
    for number, item in enumerate(found, start=1):
        print(f"{number}. {item.get('name') or item['id']} ({item['id']})")
    prompt = f"Elice {noun} [1-{len(found)}]: "
    while True:
        answer = read_line(prompt)
        if not answer and len(found) == 1:
            return str(found[0]["id"])
        if answer.isdigit() and 1 <= int(answer) <= len(found):
            return str(found[int(answer) - 1]["id"])
        print(f"Enter a number from 1 to {len(found)}.")


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
    # The rendezvous installs this key's public half on each runtime, which is what lets
    # tcp_punch log in. An existing key is reused, never regenerated.
    key_path = str(answers.get("key") or DEFAULT_KEY)
    ensure_key(key_path)
    options["key"] = key_path
    return options


#: The prompt for a Kaggle API token, read with hidden input.
KAGGLE_TOKEN_PROMPT = "Kaggle API token, or the path to kaggle.json: "

#: The read-only Kaggle CLI call that proves a token works.
KAGGLE_CHECK = ("quota", "--format", "json")

#: Files a Kaggle login may write in the account directory.
KAGGLE_ACCESS_TOKEN = "access_token"
KAGGLE_JSON = "kaggle.json"
KAGGLE_SESSION_URL = "jupyter_url"


def read_kaggle_token(alias: str, given: str) -> tuple[str, str, list[str]]:
    """Classify a Kaggle credential. Returns the file name, its content and the secrets in it.

    A value starting with ``{`` is a ``kaggle.json`` body, a value naming an existing file is
    that file's ``kaggle.json`` body, and anything else is an access token.
    """
    text = given.strip()
    candidate = Path(text).expanduser()
    if not text.startswith("{") and len(text) < 4096 and candidate.is_file():
        try:
            text = candidate.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise LoginError(f"{alias}: {candidate} could not be read: {exc}") from None
    if not text.startswith("{"):
        if not text or any(character.isspace() for character in text):
            raise LoginError(f"{alias}: a Kaggle access token is one string with no spaces")
        return KAGGLE_ACCESS_TOKEN, text, [text]
    try:
        body = json.loads(text)
    except ValueError:
        raise LoginError(f"{alias}: the kaggle.json given is not valid JSON") from None
    if not isinstance(body, dict):
        raise LoginError(f"{alias}: kaggle.json must be an object with username and key")
    missing = [name for name in ("username", "key") if not body.get(name)]
    if missing:
        raise LoginError(f"{alias}: kaggle.json has no {' or '.join(missing)}")
    content = json.dumps({"username": str(body["username"]), "key": str(body["key"])})
    return KAGGLE_JSON, content, [str(body["key"])]


def valid_session_url(alias: str, url: str) -> str:
    """Return the Colab Compatible URL when it is an HTTP address, and refuse it otherwise."""
    value = url.strip()
    if not value.startswith(("https://", "http://")):
        raise LoginError(
            f"{alias}: --connect takes the Colab Compatible URL from Run, Kaggle Jupyter Server "
            f"in the Kaggle editor, which starts with https://"
        )
    return value


def redact(text: str, secrets_in_text: list[str]) -> str:
    """Replace every occurrence of each secret with ``***``."""
    for secret in secrets_in_text:
        if secret:
            text = text.replace(secret, "***")
    return text


def kaggle_account(answers: Answers) -> dict[str, Any]:
    """Keep the Kaggle API token in the account directory and prove it with a read-only call.

    The token file is written before the check, because the Kaggle CLI reads it from there.
    A check that fails removes what this attempt wrote, so nothing is left behind.
    """
    from .. import tools

    uv = tools.find_uv()
    if uv is None:
        raise LoginError(tools.missing_uv_message())
    connect = answers.get("connect")
    url = None
    if isinstance(connect, str) and connect:
        url = valid_session_url(answers.alias, connect)
    given = answers.token or (read_password(KAGGLE_TOKEN_PROMPT) if answers.interactive else None)
    if not given:
        raise LoginError(
            f"{answers.alias} needs a Kaggle API token from kaggle.com Settings, API. "
            f"Pass --token or drop --no-input."
        )
    name, content, hidden = read_kaggle_token(answers.alias, given)
    options: dict[str, Any] = {"kind": answers.kind}
    record_workspace(answers, options)

    directory = secrets.account_directory(answers.alias)
    written = [name] if not (directory / name).exists() else []
    store_secret(answers.alias, name, content)
    try:
        result = subprocess.run(
            [*tools.command(tools.KAGGLE, uv), *KAGGLE_CHECK],
            capture_output=True,
            text=True,
            timeout=120,
            env=tools.kaggle_environment(answers.alias),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        forget_files(answers.alias, written)
        raise LoginError(
            f"the Kaggle CLI could not be run, so nothing was written: {redact(str(exc), hidden)}"
        ) from None
    if result.returncode != 0:
        forget_files(answers.alias, written)
        detail = redact((result.stderr or result.stdout or "").strip()[-2000:], hidden)
        raise LoginError(
            f"the Kaggle token check exited {result.returncode}, so nothing was written: {detail}"
        )
    if url is not None:
        try:
            devices = kaggle_devices(answers.alias, url)
        except LetifyError:
            forget_files(answers.alias, written)
            raise
        store_secret(answers.alias, KAGGLE_SESSION_URL, url)
        if devices:
            # Written by log_in as its own [<alias>.devices] table.
            options["devices"] = devices
    return options


#: Run on the session at a --connect login. It prints nvidia-smi rows, or a line saying why not.
KAGGLE_DEVICE_SOURCE = (
    "import shutil, subprocess\n"
    "tool = shutil.which('nvidia-smi')\n"
    "if tool is None:\n"
    "    print('nvidia-smi is not on this Kaggle session')\n"
    "else:\n"
    "    found = subprocess.run([tool, '--query-gpu=index,name,memory.total',\n"
    "                            '--format=csv,noheader'], capture_output=True, text=True)\n"
    "    print(found.stdout if found.returncode == 0 else\n"
    "          'nvidia-smi exited %d on this Kaggle session' % found.returncode)\n"
)


def kaggle_devices(alias: str, url: str) -> dict[str, dict[str, int]] | None:
    """Read the session's GPUs once through a kernel letify creates and deletes.

    Kaggle assigns the cards, so the table holds a count per name. A session with no GPU is
    not an error: a note is printed and ``None`` returned.
    """
    from ..providers.kaggle import Session

    session = Session(alias, url)
    kernel = session.create_kernel()
    try:
        output = session.run(kernel, KAGGLE_DEVICE_SOURCE, 120)
    finally:
        session.delete_kernel(kernel)
    rows = "\n".join(line.replace(", Tesla ", ", ") for line in output.splitlines())
    groups = group_devices(rows)
    if not groups:
        reason = output.strip().splitlines()[-1] if output.strip() else "nvidia-smi listed no GPU"
        print(f"{reason}. No devices table was written.")
        return None
    for group in groups:
        print(group.describe())
    return {group.name: {"count": len(group.indices)} for group in groups}


def forget_files(alias: str, names: list[str]) -> None:
    """Remove files one login attempt created in the account directory."""
    directory = secrets.account_directory(alias)
    for name in names:
        (directory / name).unlink(missing_ok=True)


def register_session(answers: Answers, text: str) -> dict[str, dict[str, int]] | None:
    """Replace the session URL of an already declared Kaggle account, and read its GPUs."""
    entry = tomllib.loads(text).get(answers.alias, {})
    if entry.get("kind", answers.kind) != "kaggle":
        return None
    url = valid_session_url(answers.alias, str(answers.get("connect")))
    devices = kaggle_devices(answers.alias, url)
    store_secret(answers.alias, KAGGLE_SESSION_URL, url)
    print(f"{answers.alias}: the Kaggle Jupyter Server session URL was replaced")
    return devices


#: The prompt for the token when ``--connect`` is not given.
TOKEN_PROMPT = "Token printed by 'letify client shell connect': "


def tunnel_account(answers: Answers) -> dict[str, Any]:
    """Set up a machine behind NAT from the token ``letify client shell connect`` printed.

    Every SSH command runs with ``ProxyCommand=tailcat <address> <agent port>``, and the
    account is written with no address. A failure at any step names the step.
    """
    from ..install import InstallError, ensure
    from ..transport import setup

    def failed(step: str, reason: object) -> LoginError:
        return LoginError(f"tunnel login failed at {step}: {reason}")

    try:
        tailcat = ensure(
            "tailcat",
            instructions=setup.tailcat_install_instructions(),
        )
    except InstallError as exc:
        raise failed("tailcat", exc) from None

    token = answers.get("connect")
    if not (isinstance(token, str) and token):
        if not answers.interactive:
            raise failed(
                "token",
                f"{answers.alias} needs the token. Pass --connect <token> or drop --no-input.",
            )
        token = read_line(TOKEN_PROMPT)
    try:
        fields = setup.decode_token(token)
    except ValueError as exc:
        raise failed("token", exc) from None

    address = str(fields["tailcat"])
    agent_port = int(fields["tailcat_port"])
    proxy = f"{tailcat} {address} {agent_port}"
    user = fields.get("user") or answers.get("user")
    port = int(fields.get("port") or answers.get("port") or 22)
    key_path = str(answers.get("key") or DEFAULT_KEY)
    options: dict[str, Any] = {
        "kind": "tunnel",
        "tailcat": address,
        "tailcat_port": agent_port,
    }
    if user:
        options["user"] = user
    options["port"] = port
    # Forward SSH from outside, when the machine publishes its SSH server. Login itself
    # still runs over Tailcat, so these are only recorded.
    public_address = fields.get("address") or answers.get("address")
    if isinstance(public_address, str) and public_address:
        options["address"] = public_address
    public_port = fields.get("public_port") or answers.get("public_port")
    if public_port:
        options["public_port"] = int(public_port)
    options["key"] = key_path
    if answers.get("persistent") is not None:
        options["persistent"] = bool(answers.get("persistent"))
    ssh = {"address": address, "user": user, "port": port, "key_path": key_path, "proxy": proxy}

    step = "key install"
    try:
        if answers.install_key:
            ensure_key(key_path)
            install_key(**ssh)
        step = "key confirmation"
        confirm_key(**ssh)
        step = "workspace"
        workspace, default = choose_workspace(answers, address)
        check_workspace(**ssh, workspace=workspace)
        if workspace != default:
            options["workspace"] = workspace
        step = "devices"
        devices = record_devices(answers, **ssh)
    except LoginError as exc:
        raise failed(step, exc) from None
    if devices:
        options["devices"] = devices
    return options


FLOWS = {
    "shell": shell_account,
    "tunnel": tunnel_account,
    "elice": elice_account,
    "colab": colab_account,
    "modal": modal_account,
    "kaggle": kaggle_account,
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
        if answers.get("connect"):
            devices = register_session(answers, existing)
        add_colab_key(answers, existing, home)
        existing = home.read_text(encoding="utf-8")
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


def add_colab_key(answers: Answers, text: str, home: Path) -> None:
    """Give an already declared Colab account with no key the key the rendezvous needs."""
    entry = tomllib.loads(text).get(answers.alias, {})
    if entry.get("kind", answers.kind) != "colab" or entry.get("key"):
        return
    key_path = str(answers.get("key") or DEFAULT_KEY)
    ensure_key(key_path)
    body = {key: value for key, value in entry.items() if key != "devices"}
    body["key"] = key_path
    writer.update(home, answers.alias, body, private=True)


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
