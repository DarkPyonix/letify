"""Command line entry points.

Enough to answer what comes up before any code is written: which providers are declared,
what they offer, what is running right now, whether a machine answers, and whether
forwarding CUDA calls to it is worth doing.
"""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .config import login
from .errors import LetifyError
from .launcher import Launcher


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="letify", description="Declarations that become infrastructure."
    )
    parser.add_argument("--version", action="version", version=f"letify {__version__}")
    parser.add_argument("--config", help="path to a .letify file")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("providers", help="list declared providers and their storage")
    sub.add_parser("devices", help="list the accelerators each provider offers")
    sub.add_parser("status", help="show live runtimes and what they are costing")

    usage = sub.add_parser("usage", help="show what each account has left")
    usage.add_argument("alias", nargs="?", help="one provider instead of all of them")
    usage.add_argument("--json", action="store_true", help="print the records unformatted")

    utilization = sub.add_parser(
        "utilization", help="show how hard each instance's accelerator is working"
    )
    utilization.add_argument("alias", nargs="?", help="one provider instead of all of them")
    utilization.add_argument("--json", action="store_true", help="print the records unformatted")

    log_in = sub.add_parser("login", help="declare an account and reference it here")
    log_in.add_argument("kind", help="provider kind: shell, tunnel, colab, modal, elice")
    log_in.add_argument("alias", nargs="?", help="name to reach it by; defaults to the kind")
    log_in.add_argument("--address", help="machine address, for shell and tunnel")
    log_in.add_argument("--user", help="SSH user")
    log_in.add_argument("--port", type=int, help="SSH port")
    log_in.add_argument("--key", help="SSH private key path")
    log_in.add_argument(
        "--auth",
        choices=login.AUTH_METHODS,
        help="how to authenticate; key is the default and the only one that works unattended",
    )
    log_in.add_argument(
        "--persistent",
        action="store_true",
        default=None,
        help="the machine keeps its disk between sessions",
    )
    log_in.add_argument("--zone-id", dest="zone_id", help="Elice zone id")
    log_in.add_argument("--machine-id", dest="machine_id", help="Elice machine id")
    log_in.add_argument("--endpoint", help="API endpoint, where it is not the default")
    log_in.add_argument("--account", help="account email, for Colab")
    log_in.add_argument("--workspace", help="workspace name, for Modal")
    log_in.add_argument(
        "--token", help="credential to file in the OS keyring rather than in any file"
    )
    log_in.add_argument(
        "--no-input",
        dest="interactive",
        action="store_false",
        help="fail rather than prompt, for a script",
    )
    log_in.add_argument(
        "--skip-key-install",
        dest="install_key",
        action="store_false",
        help="the key is already on the machine, so only confirm it",
    )

    log_out = sub.add_parser("logout", help="remove an account from this machine")
    log_out.add_argument("alias", help="provider alias to forget")

    check = sub.add_parser("check", help="check that a provider answers")
    check.add_argument("alias", help="provider alias from the configuration")

    probe = sub.add_parser("probe", help="measure whether host='local' is worth using")
    probe.add_argument("host", nargs="?", help="host name to measure the round trip to")

    efficiency = sub.add_parser(
        "efficiency", help="expected fraction of a direct run, from measured terms"
    )
    efficiency.add_argument("step_seconds", type=float, help="GPU time per step")
    efficiency.add_argument("syncs", type=int, help="host synchronizations per step")
    efficiency.add_argument("round_trip_ms", type=float, help="network round trip")

    return parser


def _describe_usage(row: dict) -> str:
    """One line for a usage row, saying plainly when there is no number."""
    if row.get("unmetered"):
        return "unmetered"
    unit = row.get("unit") or ""
    parts = []
    if row.get("remaining") is not None:
        left = f"{row['remaining']:g} {unit} left"
        if row.get("limit"):
            left += f" of {row['limit']:g}"
        parts.append(left)
    if row.get("rate_per_hour") is not None:
        parts.append(f"{row['rate_per_hour']:g} {unit}/hour running now")
    return ", ".join(parts) or f"not reported ({row.get('source')})"


def _describe_device(device: dict) -> str:
    """One line for a device reading, leaving out what the card did not report."""
    load = (
        f"{device['utilization_percent']:.0f}% busy"
        if device.get("utilization_percent") is not None
        else "load unknown"
    )
    parts = [f"gpu{device['index']}", str(device["name"]), load]
    if device.get("memory_total_gb"):
        used = device.get("memory_used_gb") or 0.0
        parts.append(f"{used:.1f}/{device['memory_total_gb']:.1f} GiB")
    if device.get("temperature_c") is not None:
        parts.append(f"{device['temperature_c']:.0f}C")
    if device.get("power_w") is not None:
        parts.append(f"{device['power_w']:.0f}W")
    return " ".join(parts)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "efficiency":
        from .remoting import efficiency as compute

        share = compute(args.step_seconds, args.syncs, args.round_trip_ms)
        print(f"{share * 100:.1f}% of a direct run")
        return 0

    if args.command == "login":
        alias = args.alias or args.kind
        answers = login.Answers(
            alias=alias,
            kind=args.kind,
            values={
                "address": args.address,
                "user": args.user,
                "port": args.port,
                "key": args.key,
                "auth": args.auth,
                "persistent": args.persistent,
                "zone_id": args.zone_id,
                "machine_id": args.machine_id,
                "endpoint": args.endpoint,
                "account": args.account,
                "workspace": args.workspace,
            },
            token=args.token,
            interactive=args.interactive,
            install_key=args.install_key,
        )
        try:
            fresh, home, project = login.log_in(answers, project=args.config)
        except LetifyError as exc:
            print(exc, file=sys.stderr)
            return 1
        if fresh:
            print(f"{alias} declared in {home}")
        else:
            print(f"{alias} was already declared in {home}, so nothing was asked for")
        print(f"{alias} referenced in {project}, which is safe to commit")
        return 0

    if args.command == "logout":
        removed, forgotten = login.log_out(args.alias)
        if not removed:
            print(
                f"{args.alias} is not declared in {login.home_path()}",
                file=sys.stderr,
            )
            return 1
        detail = " and its keyring entry" if forgotten else ""
        print(f"{args.alias} removed from {login.home_path()}{detail}")
        print("The project reference is left alone, because this repository still needs it")
        return 0

    let = Launcher(args.config, announce=False)

    if args.command == "providers":
        for alias in let.config.order:
            try:
                provider = let.provider(alias)
            except Exception as exc:
                print(f"{alias:20} unavailable: {exc}")
                continue
            row = f"{alias:20} {provider.kind:10} {provider.persistence:11}"
            channel = "persistent" if provider.persistent_channel else "one-shot"
            print(f"{row} channel={channel}")
        return 0

    if args.command == "devices":
        print(json.dumps(let.providers.devices, indent=2, sort_keys=True))
        return 0

    if args.command == "status":
        print(json.dumps(let.status(), indent=2))
        return 0

    if args.command == "usage":
        rows = let.usage(args.alias)
        if args.json:
            print(json.dumps(rows, indent=2))
            return 0
        for row in rows:
            alias = str(row["alias"])
            if "unavailable" in row:
                print(f"{alias:20} unavailable: {row['unavailable']}")
                continue
            line = f"{alias:20} {row['kind']!s:10} {_describe_usage(row)}"
            print(line)
            if row.get("note"):
                print(f"{'':20} {row['note']}")
        return 0

    if args.command == "utilization":
        rows = let.utilization(args.alias)
        if args.json:
            print(json.dumps(rows, indent=2))
            return 0
        for row in rows:
            alias = str(row["alias"])
            if "unavailable" in row:
                print(f"{alias:20} unavailable: {row['unavailable']}")
                continue
            head = f"{alias}.{row['accelerator']}"
            devices = row.get("devices") or []
            if not devices:
                print(f"{head:28} {row.get('reason') or 'nothing reported'}")
                continue
            for device in devices:
                print(f"{head:28} {_describe_device(device)}")
        return 0

    if args.command == "check":
        provider = let.provider(args.alias)
        checker = getattr(provider, "check", None)
        if checker is None:
            print(f"{args.alias} has no check step", file=sys.stderr)
            return 1
        print(checker())
        return 0

    if args.command == "probe":
        from .remoting import probe as run_probe

        capability = run_probe(args.host)
        print(
            json.dumps(
                {
                    "platform": capability.platform,
                    "core": capability.core,
                    "agent": capability.agent,
                    "round_trip_ms": capability.round_trip_ms,
                    "usable": capability.usable,
                    "costly": capability.costly,
                    "reason": capability.explain(),
                },
                indent=2,
            )
        )
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
