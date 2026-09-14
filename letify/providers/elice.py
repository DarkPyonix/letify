"""Elice, a virtual machine on Elice Cloud Infrastructure.

Owns finding, launching, starting and stopping one machine through ``eci``, the command
Elice publishes at github.com/elice-dev/eci-cli, and reading the account's credit over
Elice's HTTP API. It does not own the session on the machine, which ``Shell`` runs over
forward SSH to the machine's public IP once the machine is started.

``eci`` is a standalone binary, so it runs as a separate process and is never installed
by letify. Every command gets the account's endpoint, token, zone and a configuration file
in the account directory through its environment, never through its arguments.

An idle machine bills no compute, while its disk and public IP keep billing. letify stops
machines and never deletes one. A spot machine may be stopped or deleted by Elice at any
time, which letify reports as ``SpotPreempted``.
"""

from __future__ import annotations

import json as _json
import os
import platform
import re
import secrets
import shutil
import string
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..config.secrets import account_directory, write_secret
from ..declare.instance import PRICE_TYPES, Instance
from ..errors import (
    ConfigError,
    LetifyError,
    ProviderUnavailable,
    RuntimeFailure,
    SpotPreempted,
    UnsupportedMode,
)
from .naming import normalize_gpu
from .shell import Shell
from .usage import Usage

if TYPE_CHECKING:
    from ..runtime.session import Runtime
    from ..transport.rendezvous import Rendezvous

DEFAULT_ENDPOINT = "https://portal.elice.cloud/api"

#: HTTP paths from Elice's Terraform provider, elice-dev/terraform-provider-eci. Only the
#: billing reads use them; machines are driven through eci.
VM_PATH = "/user/resource/compute/virtual_machine"
ALLOCATION_PATH = "/user/resource/compute/virtual_machine_allocation"
INSTANCE_TYPE_PATH = "/user/infra/instance_type"
PRICING_PATH = "/user/pricing"
ZONE_PATH = "/user/infra/zone"
ORGANIZATION_PATH = "/user/organization"

#: The billing API path the portal reads the organization's remaining credit from.
BILLING_STATS_PATH = "/stats"

#: A person is waiting for the usage table, so a billing read gets less than a call does.
USAGE_HTTP_TIMEOUT = 15.0

#: How often a machine's status is read while waiting, and how long each wait lasts.
POLL_SECONDS = 10.0
IDLE_WAIT_SECONDS = 300.0
START_WAIT_SECONDS = 600.0

#: What ``spot_fallback`` accepts. Spec "Spot preemption".
SPOT_FALLBACKS = ("none", "ondemand")

#: The key installed on a machine letify launches, as for the other SSH accounts.
DEFAULT_KEY = "~/.ssh/id_letify"

#: The user a launched machine is reached as. The documentation calls the launch password
#: the root password.
DEFAULT_USER = "root"

INSTALL_UNIX = (
    "curl -fsSL https://raw.githubusercontent.com/elice-dev/eci-cli/main/scripts/install.sh | sh"
)
INSTALL_WINDOWS = 'powershell -c "irm https://eci.sh/install.ps1 | iex"'

#: Standard error that means the token was refused or lacks permission.
REFUSED_MARKERS = ("401", "403", "unauthorized", "permission")

#: Symbols a generated password draws from.
PASSWORD_SYMBOLS = "!@#%^*_=+"


# -- the HTTP API, for billing -------------------------------------------------


def request(
    method: str,
    path: str,
    *,
    token: str,
    endpoint: str = DEFAULT_ENDPOINT,
    params: Mapping[str, Any] | None = None,
    json: Any = None,
    headers: Mapping[str, str] | None = None,
    timeout: float = 60.0,
) -> Any:
    """One request with the standard library HTTP client, answering the decoded body."""
    url = f"{endpoint.rstrip('/')}{path}"
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    data = None if json is None else _json.dumps(json).encode()
    call = urllib.request.Request(url, data=data, method=method)
    call.add_header("Authorization", f"Bearer {token}")
    call.add_header("Accept", "application/json")
    for name, value in (headers or {}).items():
        call.add_header(name, value)
    if data is not None:
        call.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(call, timeout=timeout) as response:
            status, raw = response.status, response.read()
    except urllib.error.HTTPError as exc:
        status, raw = exc.code, exc.read()
    except (urllib.error.URLError, OSError) as exc:
        raise RuntimeFailure(f"Elice {method} {path} failed: {exc}") from exc

    text = raw.decode(errors="replace")
    # This API answers 200 for every success, so anything else is a failure.
    if status != 200:
        try:
            body = _json.loads(text)
            detail = body.get("message") or str(body)
        except Exception:
            detail = text[:500]
        raise RuntimeFailure(f"Elice {method} {path} returned {status}: {detail}")
    return _json.loads(text) if text else {}


def items(body: Any) -> list[dict[str, Any]]:
    """The records of a listing, sent either as a bare list or under ``items``."""
    if isinstance(body, list):
        return [item for item in body if isinstance(item, dict)]
    if isinstance(body, dict):
        return [item for item in body.get("items", []) if isinstance(item, dict)]
    return []


# -- the eci command -----------------------------------------------------------


def eci_install_message(system: str | None = None) -> str:
    """Where eci comes from, for this operating system."""
    system = system if system is not None else platform.system()
    command = INSTALL_WINDOWS if system == "Windows" else INSTALL_UNIX
    return (
        f"eci, Elice's command line, is not on PATH. Install it with:\n\n  {command}\n\n"
        "Then run this command again"
    )


def find_eci(binary: str = "eci", *, interactive: bool = False) -> str:
    """The eci executable, or ``ProviderUnavailable`` carrying the install command.

    The default name goes through the lookup of spec "Installing external tools", which
    offers the install on a terminal when ``interactive``. Another name is run as written.
    """
    from .. import install

    if binary != "eci":
        found = shutil.which(binary)
        if found is None:
            raise ProviderUnavailable("elice", eci_install_message())
        return found
    try:
        return install.ensure("eci", interactive=interactive, instructions=eci_install_message())
    except install.InstallError as exc:
        raise ProviderUnavailable("elice", str(exc)) from None


def eci_environment(alias: str, token: str, endpoint: str, zone_id: str | None) -> dict[str, str]:
    """The environment every eci command of one account runs in. Spec "Elice machines"."""
    # The account directory is not created here, so a login that fails leaves nothing.
    env = {k: v for k, v in os.environ.items() if not k.startswith("ECI_")}
    env["ECI_API_ENDPOINT"] = endpoint
    env["ECI_API_TOKEN"] = token
    env["ECI_CONFIG"] = str(account_directory(alias) / "eci.yaml")
    if zone_id:
        env["ECI_ZONE_ID"] = zone_id
    return env


def shown(args: list[str]) -> str:
    """An eci command as it may be printed, with the password replaced by ``***``."""
    words, hide = [], False
    for arg in args:
        words.append("***" if hide else arg)
        hide = arg == "--password"
    return "eci " + " ".join(words)


def eci(binary: str, args: list[str], env: Mapping[str, str], *, parse: bool = True) -> Any:
    """Run one eci command, answering its JSON output, or its text when ``parse`` is False."""
    command = [binary, *args, *(["--format", "json"] if parse else [])]
    printable = shown(args)
    try:
        result = subprocess.run(command, env=dict(env), capture_output=True, text=True, timeout=900)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeFailure(f"{printable} failed: {exc}") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        head = f"{printable} exited {result.returncode}"
        if any(marker in detail.lower() for marker in REFUSED_MARKERS):
            head = (
                "Elice refused the access token or it lacks permission. A token is issued in "
                f"the portal under User management, User access token. {head}"
            )
        raise RuntimeFailure(head, command=printable, stderr=detail)
    if not parse:
        return result.stdout
    text = result.stdout.strip()
    if not text:
        return {}
    try:
        return _json.loads(text)
    except ValueError:
        raise RuntimeFailure(f"{printable} printed no JSON: {text[:200]}") from None


def generate_password(length: int = 20) -> str:
    """A machine password Elice accepts. Spec "Elice machines".

    Upper case, lower case, a digit and a symbol, and no three characters running
    consecutively up or down, such as ``123`` or ``cba``.
    """
    pools = [string.ascii_uppercase, string.ascii_lowercase, string.digits, PASSWORD_SYMBOLS]
    alphabet = "".join(pools)
    shuffler = secrets.SystemRandom()
    while True:
        chars = [secrets.choice(pool) for pool in pools]
        chars += [secrets.choice(alphabet) for _ in range(length - len(pools))]
        shuffler.shuffle(chars)
        candidate = "".join(chars)
        if not _has_run(candidate):
            return candidate


def _has_run(text: str) -> bool:
    return any(
        ord(b) - ord(a) == ord(c) - ord(b) and abs(ord(b) - ord(a)) == 1
        for a, b, c in zip(text, text[1:], text[2:], strict=False)
    )


def public_address(record: Mapping[str, Any]) -> str | None:
    """The machine's first public IP, under any key containing ``public_ip``."""
    for key, value in record.items():
        if "public_ip" not in str(key):
            continue
        if isinstance(value, str) and value:
            return value
        if isinstance(value, Mapping) and isinstance(value.get("ip"), str):
            return value["ip"]
        if isinstance(value, list) and value:
            first = value[0]
            if isinstance(first, str) and first:
                return first
            if isinstance(first, Mapping) and isinstance(first.get("ip"), str):
                return first["ip"]
    return None


class Elice(Shell):
    """One Elice Cloud Infrastructure zone under one access token."""

    kind = "elice"
    extra = "shell"
    default_persistence = "persistent"
    has_fast_path = True
    offers_spot = True
    #: Its machine lives only as long as a session, so utilization is read inside one.
    reads_machine = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        #: How many live runtimes are on each machine, so one stop does not end another.
        self._holders: dict[str, int] = {}
        #: Machines letify itself stopped, which are therefore not preempted.
        self._stopped: set[str] = set()
        #: Machines Elice took back, which a stop leaves alone.
        self._preempted: set[str] = set()
        #: The price type each machine was started at.
        self._machine_price: dict[str, str] = {}
        #: Accelerator shapes that fell back to ondemand after a preemption.
        self._fallen_back: set[tuple[str, int]] = set()
        self._machine_address: str | None = None
        self._pending_machine: str | None = None

    # -- configuration -------------------------------------------------------

    @property
    def endpoint(self) -> str:
        return str(self.config.option("endpoint", DEFAULT_ENDPOINT)).rstrip("/")

    @property
    def zone_id(self) -> str:
        value = self.config.option("zone_id")
        if not isinstance(value, str):
            raise ProviderUnavailable(self.kind, f"{self.alias} has no 'zone_id' field")
        return value

    @property
    def organization_id(self) -> str | None:
        value = self.config.option("organization_id")
        return value if isinstance(value, str) else None

    @property
    def machine_id(self) -> str | None:
        """An existing machine the account names. None lets letify create its own."""
        value = self.config.option("machine_id")
        return value if isinstance(value, str) and value else None

    @property
    def price_type(self) -> str:
        value = self.config.option("price_type", "ondemand")
        if value not in PRICE_TYPES:
            raise ConfigError(
                f'{self.alias}: price_type must be "ondemand" or "spot", not {value!r}'
            )
        return str(value)

    @property
    def spot_fallback(self) -> str:
        value = self.config.option("spot_fallback", "none")
        if value not in SPOT_FALLBACKS:
            raise ConfigError(
                f'{self.alias}: spot_fallback must be "none" or "ondemand", not {value!r}'
            )
        return str(value)

    @property
    def eci_binary(self) -> str:
        return str(self.config.option("eci_binary", "eci"))

    @property
    def user(self) -> str | None:
        value = self.config.option("user", DEFAULT_USER)
        return value if isinstance(value, str) else DEFAULT_USER

    @property
    def key_path(self) -> str | None:
        value = self.config.option("key", DEFAULT_KEY)
        return value if isinstance(value, str) else DEFAULT_KEY

    @property
    def address(self) -> str | None:
        """The started machine's public IP, or the account's ``address``."""
        found = self.target_address()
        if found:
            return found
        raise ProviderUnavailable(
            self.kind, f"{self.alias} has no started machine yet, so there is no address"
        )

    def target_address(self) -> str | None:
        if self._machine_address:
            return self._machine_address
        return super().target_address()

    def _token(self) -> str:
        token = self.config.secret("access_token")
        if not token:
            raise ProviderUnavailable(
                self.kind,
                f"{self.alias} needs an access token. Run 'letify login elice {self.alias}', "
                f"which keeps it in ~/.letify/accounts/{self.alias}/, or set access_token_env",
            )
        return token

    def _say(self, message: str) -> None:
        from ..transport.announce import printer

        printer(self.announce)(message)

    # -- the HTTP API ----------------------------------------------------------

    def _call(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        json: Any = None,
        base: str | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float = 60.0,
    ) -> Any:
        """One request under this account's token, answering the decoded body.

        ``base`` replaces the compute API endpoint, for the billing API that lives apart.
        """
        return request(
            method,
            path,
            token=self._token(),
            endpoint=base or self.endpoint,
            params=params,
            json=json,
            headers=headers,
            timeout=timeout,
        )

    @staticmethod
    def _items(body: Any) -> list[dict[str, Any]]:
        return items(body)

    # -- eci -------------------------------------------------------------------

    def _eci(self, args: list[str], *, parse: bool = True) -> Any:
        binary = find_eci(self.eci_binary, interactive=True)
        env = eci_environment(self.alias, self._token(), self.endpoint, self.zone_id)
        account_directory(self.alias).mkdir(parents=True, exist_ok=True)
        return eci(binary, args, env, parse=parse)

    def get_machine(self, machine: str) -> dict[str, Any] | None:
        """``eci compute vm get``, or None when Elice says there is no such machine."""
        try:
            record = self._eci(["compute", "vm", "get", machine])
        except RuntimeFailure as exc:
            if "not found" in exc.stderr.lower():
                return None
            raise
        return record if isinstance(record, dict) else None

    def find_machine(self, machine: str) -> dict[str, Any] | None:
        """The listed machine whose name or id is exactly ``machine``."""
        for record in items(self._eci(["compute", "vm", "list"])):
            if machine in (record.get("name"), record.get("id")):
                return record
        return None

    def machine_name(self, price_type: str) -> str:
        """``letify-<alias>``, with ``-spot`` for a spot machine. Spec "Elice machines"."""
        slug = re.sub(r"[^a-z0-9-]", "-", self.alias.lower().replace("_", "-"))
        return f"letify-{slug}" + ("-spot" if price_type == "spot" else "")

    # -- instances -----------------------------------------------------------

    def discover(self) -> Mapping[str, Instance]:
        """The instance types eci lists as activated. A declared ``gpus`` list skips it."""
        declared = self.config.option("gpus")
        if isinstance(declared, list) and declared:
            return {str(name): Instance(self, gpu=str(name)) for name in declared}

        table: dict[str, Instance] = {}
        for row in self._instance_types():
            devices = row.get("devices") or []
            label = normalize_gpu(str(devices[0])) if devices else "CPU"
            if label in table:
                continue
            table[label] = Instance(
                self,
                gpu=None if label == "CPU" else label,
                cpus=row.get("cpu_vcore"),
                memory_gb=row.get("memory_gib"),
            )
        return table

    def _instance_types(self) -> list[dict[str, Any]]:
        return items(self._eci(["instance-type", "list", "--activated", "true"]))

    def instance_type_for(self, instance: Instance) -> dict[str, Any]:
        """The instance type a launch uses for this instance. Spec "Elice machines"."""
        rows = self._instance_types()
        declared = self.config.option("instance_type")
        if isinstance(declared, str) and declared:
            found = next((r for r in rows if declared in (r.get("name"), r.get("id"))), None)
            return found or {"name": declared, "devices": [instance.gpu] if instance.gpu else []}
        if instance.gpu is None:
            matches = sorted(
                (r for r in rows if not r.get("devices")),
                key=lambda r: r.get("cpu_vcore") or 0,
            )
        else:
            matches = [
                r
                for r in rows
                if len(r.get("devices") or []) == instance.devices
                and all(normalize_gpu(str(d)) == instance.gpu for d in r.get("devices") or [])
            ]
        if not matches:
            offered = ", ".join(str(r.get("name")) for r in rows) or "none"
            shape = f"{instance.devices} x {instance.gpu}" if instance.gpu else "CPU"
            raise ProviderUnavailable(
                self.kind, f"Elice offers no instance type with {shape}. Offered: {offered}"
            )
        return matches[0]

    def store_backend(self) -> str:
        """The machine's own disk, reached over the same SSH session as a shell machine."""
        backend = self.config.option("store")
        return str(backend) if isinstance(backend, str) else "shell"

    # -- price type ----------------------------------------------------------

    def price_type_of(self, instance: Instance) -> str | None:
        """The instance's price type, else the account's, after any ondemand fallback."""
        wanted = instance.price_type or self.price_type
        if wanted == "spot" and (instance.accelerator, instance.devices) in self._fallen_back:
            return "ondemand"
        return wanted

    def _say_price(self, machine: str, type_name: str, price_type: str) -> None:
        try:
            rows = items(self._eci(["pricing", "list", "--resource-kind", "vm_allocation"]))
        except LetifyError:
            rows = []
        row = next(
            (r for r in rows if r.get("name") == type_name and r.get("pricing_type") == price_type),
            None,
        )
        price = row.get("price_per_hour") if row else None
        if price in (None, ""):
            self._say(f"{machine}: no {price_type} price listed for {type_name}")
        else:
            self._say(f"{machine}: {type_name} {price_type} at {price} KRW/hour")

    def _check_quota(self, machine: str, row: Mapping[str, Any]) -> None:
        """Refuse an ondemand launch the organization has no quota for. Spec "Price type"."""
        type_name = str(row.get("name"))
        try:
            compute = self._eci(["org", "info"])["resource_quota"]["compute"]
            if not isinstance(compute, dict):
                raise TypeError(type(compute).__name__)
        except (LetifyError, KeyError, TypeError):
            self._say(f"{machine}: the ondemand quota could not be read, launching anyway")
            return
        limits = compute.get("instance_types")
        limit = None
        if isinstance(limits, dict):
            for key in (row.get("id"), type_name):
                if key in limits:
                    limit = limits[key]
        accelerator = bool(row.get("devices"))
        if limit == 0 or (accelerator and compute.get("devices") == 0):
            raise ProviderUnavailable(
                self.kind,
                f"the Elice ondemand quota for {type_name} is 0. Request a quota increase in "
                f'the portal, or set price_type = "spot", which does not count against the quota',
            )

    # -- usage ---------------------------------------------------------------

    def pricing(self) -> list[dict[str, Any]]:
        """The zone's price list, including any preemptible option."""
        return self._items(self._call("GET", PRICING_PATH))

    #: Elice bills in Korean won. The credit comes from the billing API the portal reads,
    #: and the hourly rate from the live allocations priced against the zone price list.
    usage_unit = "KRW"
    usage_source = "Elice billing /stats for the credit, allocations priced from the price list"

    def credit_remaining(self) -> float:
        """The organization's remaining credit in won, from ``<billing_endpoint>/stats``."""
        endpoint = self.config.option("billing_endpoint")
        if not isinstance(endpoint, str) or not endpoint:
            raise ProviderUnavailable(
                self.kind,
                f"{self.alias} has no 'billing_endpoint' field, the Elice billing API base URL "
                f"the portal reads the credit from",
            )
        organization = self.config.option("organization")
        headers = {"x-elice-org-name-short": organization} if isinstance(organization, str) else {}
        body = self._call(
            "GET",
            BILLING_STATS_PATH,
            base=endpoint.rstrip("/"),
            headers=headers,
            timeout=USAGE_HTTP_TIMEOUT,
        )
        if not isinstance(body, dict):
            raise RuntimeFailure(f"Elice billing stats answered {type(body).__name__}")
        amount = body.get("total_credit_remaining_amount", body.get("totalCreditRemainingAmount"))
        if isinstance(amount, (int, float)):
            return float(amount)
        if isinstance(amount, str) and amount.strip():
            # The portal sends "<amount> <currency>", with KRW as the currency.
            return float(amount.split()[0].replace(",", ""))
        raise RuntimeFailure("Elice billing stats answered without total_credit_remaining_amount")

    def report_usage(self) -> Usage:
        """The remaining credit, and what the allocations that exist cost per hour.

        An allocation bills by the second while it is powered on. Storage keeps billing
        with no allocation running and is not in the rate, because the API prices the
        machine, not the disk. Either figure that cannot be read leaves a note and the
        other figure still stands.
        """
        notes: list[str] = []
        remaining: float | None = None
        rate: float | None = None
        try:
            remaining = self.credit_remaining()
        except (LetifyError, ValueError) as exc:
            notes.append(str(exc))
        try:
            rates = {
                str(item.get("instance_type_id")): item
                for item in self.pricing()
                if item.get("instance_type_id")
            }
            rate = 0.0
            for allocation in self.allocations():
                priced = rates.get(str(allocation.get("instance_type_id")))
                if priced is None:
                    continue
                amount = priced.get("price_per_hour")
                if isinstance(amount, (int, float)):
                    rate += float(amount)
        except LetifyError as exc:
            rate = None
            notes.append(str(exc))
        return Usage(
            alias=self.alias,
            kind=self.kind,
            unit=self.usage_unit,
            source=self.usage_source,
            remaining=remaining,
            rate_per_hour=rate,
            as_of=time.time(),
            note="; ".join(notes) or None,
            price_type=self.price_type,
        )

    def allocations(self, machine_id: str | None = None) -> list[dict[str, Any]]:
        params = {"filter_machine_id": machine_id} if machine_id else None
        return self._items(self._call("GET", ALLOCATION_PATH, params=params))

    # -- the pipeline ----------------------------------------------------------

    def rendezvous(self, runtime: Runtime | None = None) -> Rendezvous | None:
        """Forward SSH to the started machine's public IP.

        eci runs no command on a machine, so once it is started the remote half runs over
        SSH to its address.
        """
        if not self.target_address():
            return None
        from ..transport.rendezvous import ShellCommandRendezvous

        return ShellCommandRendezvous(self.ssh_command, self.remote_python)

    # -- sessions ------------------------------------------------------------

    def create_session(self, instance: Instance, name: str) -> None:
        """Find, launch or start the machine, then learn its address. Spec "Elice machines"."""
        find_eci(self.eci_binary)
        price_type = self.price_type_of(instance) or "ondemand"
        declared = self.machine_id
        password: str | None = None
        if declared:
            record = self.find_machine(declared)
            if record is None:
                raise ProviderUnavailable(
                    self.kind,
                    f"{self.alias} names machine_id {declared!r}, which Elice does not list",
                )
            machine = str(record.get("name") or declared)
            have = record.get("pricing_type")
            explicit = instance.price_type or self.config.option("price_type")
            if explicit and isinstance(have, str) and have != price_type:
                raise ConfigError(
                    f"{self.alias}: machine {machine} is priced {have}, but {price_type} was "
                    f"requested. A machine keeps the pricing it was created with"
                )
        else:
            machine = self.machine_name(price_type)
            record = self.find_machine(machine)
            if record is None:
                password = self._launch(instance, machine, price_type)
                record = self.get_machine(machine) or {}
        record = self._ensure_started(machine, record)
        address = public_address(record)
        if not address:
            raise ProviderUnavailable(
                self.kind,
                f"Elice machine {machine} has no public IP, so letify cannot reach it over SSH",
            )
        if self._machine_address and self._machine_address != address:
            self.close_link()
        self._machine_address = address
        if password is not None:
            self.authorize_key(address, password)
        self._pending_machine = machine
        self._machine_price[machine] = price_type
        self._holders[machine] = self._holders.get(machine, 0) + 1
        self._stopped.discard(machine)
        self._preempted.discard(machine)

    def _launch(self, instance: Instance, machine: str, price_type: str) -> str:
        """``eci compute vm launch``, answering the password the machine was given."""
        row = self.instance_type_for(instance)
        type_name = str(row.get("name"))
        if price_type == "spot" and not row.get("devices"):
            raise UnsupportedMode(
                f"{self.alias}: Elice offers spot pricing for accelerator types only, and "
                f"{type_name} has no accelerator"
            )
        self._say_price(machine, type_name, price_type)
        if price_type == "ondemand":
            self._check_quota(machine, row)
        password = generate_password()
        # Written before the launch runs, so a launch that outlives this process still has it.
        write_secret(self.alias, "machine_password", password)
        args = [
            "compute",
            "vm",
            "launch",
            "--name",
            machine,
            "--instance-type",
            type_name,
            "--password",
            password,
            "--wait",
        ]
        if price_type == "spot":
            args += ["--price-type", "spot"]
        image = self.config.option("image")
        if isinstance(image, str) and image:
            args += ["--image", image]
        disk = self.config.option("disk_gib")
        if isinstance(disk, (int, str)) and str(disk).isdigit():
            args += ["--size-gib", str(disk)]
        self._eci(args, parse=False)
        return password

    def _ensure_started(self, machine: str, record: Mapping[str, Any]) -> dict[str, Any]:
        status = str(record.get("status") or "")
        if status == "started":
            return dict(record)
        if status != "idle":
            self._wait_for(machine, "idle", IDLE_WAIT_SECONDS)
        self._eci(["compute", "vm", "start", machine], parse=False)
        return self._wait_for(machine, "started", START_WAIT_SECONDS)

    def _wait_for(self, machine: str, wanted: str, limit: float) -> dict[str, Any]:
        deadline = time.monotonic() + limit
        while True:
            record = self.get_machine(machine)
            if record is None:
                raise RuntimeFailure(
                    f"Elice machine {machine} disappeared while waiting for {wanted}"
                )
            status = str(record.get("status") or "")
            if status == wanted:
                return record
            if time.monotonic() >= deadline:
                raise RuntimeFailure(
                    f"Elice machine {machine} is still {status} after {limit:.0f} s, not {wanted}"
                )
            time.sleep(POLL_SECONDS)

    def authorize_key(self, address: str, password: str) -> None:  # pragma: no cover
        """Append letify's public key on a machine letify launched, logging in with its password.

        Needs a live machine. The password is read from the account directory by an
        ``SSH_ASKPASS`` program, so it is never an argument of the SSH command.
        """
        if sys.platform == "win32":
            raise UnsupportedMode(
                f"{self.alias}: installing the key with the machine password needs SSH_ASKPASS, "
                f"which Windows OpenSSH does not read. Install the key by hand"
            )
        from ..config.login import ensure_key

        private = ensure_key(self.key_path or DEFAULT_KEY)
        material = Path(f"{private}.pub").read_text(encoding="utf-8").strip()
        directory = account_directory(self.alias)
        askpass = directory / "askpass"
        askpass.write_text(f'#!/bin/sh\ncat "{directory / "machine_password"}"\n', encoding="utf-8")
        askpass.chmod(0o700)
        env = dict(os.environ)
        env.update(SSH_ASKPASS=str(askpass), SSH_ASKPASS_REQUIRE="force")
        env.setdefault("DISPLAY", "letify")
        remote = (
            "mkdir -p ~/.ssh && chmod 700 ~/.ssh && "
            "touch ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys && "
            'key=$(cat) && grep -qxF "$key" ~/.ssh/authorized_keys || '
            'printf "%s\\n" "$key" >> ~/.ssh/authorized_keys'
        )
        command = [
            "ssh",
            "-p",
            str(self.direct_port),
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            f"HostKeyAlias=letify-{self.alias}",
            "-o",
            f"UserKnownHostsFile={directory / 'known_hosts'}",
            "-o",
            "PubkeyAuthentication=no",
            f"{self.user}@{address}",
            remote,
        ]
        last = ""
        # sshd may come up a little after eci reports the machine started.
        for _ in range(10):
            result = subprocess.run(
                command, input=material, env=env, capture_output=True, text=True, timeout=120
            )
            if result.returncode == 0:
                return
            last = (result.stderr or result.stdout).strip()
            time.sleep(POLL_SECONDS)
        raise RuntimeFailure(f"installing the key on Elice machine {address} failed", stderr=last)

    def start(self, instance: Instance, env: Any, *, name: str, volumes: Any = ()) -> Runtime:
        runtime = super().start(instance, env, name=name, volumes=volumes)
        runtime.external_id = self._pending_machine
        return runtime

    def stop(self, runtime: Runtime) -> None:
        """Stop the machine once no other runtime is on it. Nothing is ever deleted."""
        machine = runtime.external_id
        if not machine:
            return
        remaining = self._holders.get(machine, 0) - 1
        if remaining > 0:
            self._holders[machine] = remaining
            return
        self._holders.pop(machine, None)
        if machine in self._preempted:
            return
        self._stopped.add(machine)
        try:
            self._eci(["compute", "vm", "stop", machine], parse=False)
        except LetifyError as exc:
            reason = str(exc).splitlines()[0]
            self._say(f"could not stop {machine}: {reason}. Run 'eci compute vm stop {machine}'")

    def diagnose(self, runtime: Runtime, failure: Exception) -> Exception:
        """Turn a failure on a spot machine Elice took back into ``SpotPreempted``.

        Spec "Spot preemption". Any other failure, or one whose machine cannot be read,
        is returned unchanged.
        """
        machine = getattr(runtime, "external_id", None)
        if not machine or machine in self._stopped:
            return failure
        if self._machine_price.get(machine) != "spot":
            return failure
        try:
            record = self.get_machine(machine)
        except LetifyError:
            return failure
        if record is not None and record.get("status") == "started":
            return failure
        state = "deleted" if record is None else str(record.get("status") or "unknown")
        self._say(f"{machine} was preempted by Elice (state {state})")
        if record is not None:
            self._say(
                f"{machine} keeps its disk and public IP, which keep billing. "
                f"'eci compute vm delete {machine} --cascade' removes them"
            )
        instance = runtime.instance
        if self.spot_fallback == "ondemand":
            self._fallen_back.add((instance.accelerator, instance.devices))
            self._say(
                f"{machine} preempted, retrying on ondemand machine {self.machine_name('ondemand')}"
            )
        else:
            self._say(f"{machine} preempted, retrying on spot")
        self._preempted.add(machine)
        return SpotPreempted(
            f"{machine} was preempted by Elice (state {state})",
            machine=machine,
            state=state,
            at=time.time(),
        )


__all__ = [
    "ALLOCATION_PATH",
    "DEFAULT_ENDPOINT",
    "INSTANCE_TYPE_PATH",
    "ORGANIZATION_PATH",
    "PRICING_PATH",
    "VM_PATH",
    "ZONE_PATH",
    "Elice",
    "eci",
    "eci_environment",
    "eci_install_message",
    "find_eci",
    "generate_password",
    "items",
    "public_address",
    "request",
]
