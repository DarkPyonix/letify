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
"""

from __future__ import annotations

import getpass
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..errors import ConfigError
from . import secrets, writer
from .schema import RESERVED_ALIASES
from .secrets import CONFIG_DIRECTORY

#: The configuration file inside each ``.letify`` directory.
CONFIG_FILE = "config.toml"


#: Where a key generated for letify goes. Its own name, so it is never confused with a key
#: the user made for something else and never regenerated over one.
DEFAULT_KEY = "~/.ssh/id_letify"

#: Kinds whose credential belongs to the vendor's own tool, with the command that owns it.
VENDOR_COMMANDS = {
    "colab": ("colab", "colab auth login"),
    "modal": ("modal", "modal setup"),
}

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


def confirm_key(*, address: str, user: str | None, port: int, key_path: str) -> None:
    """Prove the key works before the alias is declared.

    With BatchMode, so what is proven is exactly what a pooled session will do. Failing
    here beats failing at the first call, which costs GPU time to find out.
    """
    target = f"{user}@{address}" if user else address
    result = subprocess.run(
        [
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
            "echo letify",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise LoginError(
            f"the key does not let letify into {target} without a prompt: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )


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
    return options


def vendor_account(answers: Answers) -> dict[str, Any]:
    """Record the account identity and leave the credential to the vendor's own tool.

    Wrapping another tool's login would mean owning a token letify has no way to refresh,
    and the vendor's command already works. So what is checked is only that the tool is
    installed, which is the failure a user would otherwise meet at the first call.
    """
    binary, command = VENDOR_COMMANDS[answers.kind]
    if shutil.which(binary) is None:
        raise LoginError(
            f"the {binary!r} command is not on PATH, so letify cannot reach {answers.kind}. "
            f"Install it with the letify[{answers.kind}] extra, then authenticate with "
            f"'{command}'."
        )
    options: dict[str, Any] = {"kind": answers.kind}
    if answers.kind == "colab":
        account = ask(answers, "account", "Google account email: ", required=False)
        if account:
            options["account"] = account
    else:
        workspace = ask(
            answers, "workspace", "Modal workspace (blank for the default): ", required=False
        )
        if workspace:
            options["workspace"] = workspace
    return options


FLOWS = {
    "shell": shell_account,
    "tunnel": shell_account,
    "elice": elice_account,
    "colab": vendor_account,
    "modal": vendor_account,
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
    if fresh:
        options = FLOWS[answers.kind](answers)
        writer.update(home, answers.alias, options, private=True)

    # An empty table names the account and carries no connection detail, so it is safe in a
    # repository. A table that is already there is left alone, because the project may have
    # overridden settings in it.
    target = project_path(project)
    existing_project = target.read_text(encoding="utf-8") if target.is_file() else ""
    if not writer.has_block(existing_project, answers.alias):
        writer.update(target, answers.alias, {})
    return fresh, home, target


def log_out(alias: str) -> tuple[bool, bool]:
    """Take the account off this machine, leaving the project reference alone.

    The repository still needs that account. What changed is that this machine no longer
    has it, so the reference stays and says what to run to get it back.
    """
    removed = writer.drop(home_path(), alias)
    forgotten = forget_secret(alias)
    return removed, forgotten


__all__ = [
    "AUTH_METHODS",
    "DEFAULT_KEY",
    "VENDOR_COMMANDS",
    "Answers",
    "LoginError",
    "ask",
    "check_alias",
    "confirm_key",
    "ensure_key",
    "forget_secret",
    "home_path",
    "install_key",
    "log_in",
    "log_out",
    "project_path",
    "read_line",
    "read_password",
    "store_secret",
]
