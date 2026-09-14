"""Command line entry points.

Enough to answer what comes up before any code is written: which providers are declared,
what they offer, what is running right now, whether a machine answers, and whether
forwarding PyTorch operators to it is worth doing.
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
    parser.add_argument("--config", help="a project .letify directory, or its config.toml")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("providers", help="list declared providers and their storage")
    sub.add_parser("devices", help="list the accelerators each provider offers")
    sub.add_parser("status", help="show live runtimes and what they are costing")
    sub.add_parser("stubs", help="write the provider types an editor completes")

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
    log_in.add_argument(
        "--public-port",
        dest="public_port",
        type=int,
        help="port forward SSH dials at --address when it differs from --port, for tunnel",
    )
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
    log_in.add_argument(
        "--workspace",
        metavar="PATH",
        help="the one directory letify writes under on the machine; checked for shell and tunnel",
    )
    log_in.add_argument("--profile", help="Modal profile, naming the Modal workspace to sign in to")
    log_in.add_argument(
        "--token", help="credential to keep in ~/.letify/accounts/<alias>/, never in a config file"
    )
    log_in.add_argument(
        "--connect",
        metavar="TOKEN",
        help="the token 'letify client shell connect' printed, for tunnel",
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
    log_in.add_argument(
        "--indices",
        action="append",
        metavar="NAME=SPEC",
        help="GPU indices letify may use for one accelerator, such as A100=0-3; repeatable",
    )
    log_in.add_argument(
        "--detect-devices",
        dest="detect_devices",
        action="store_true",
        help="ask an already declared machine for its GPUs again and replace its devices table",
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

    client = sub.add_parser("client", help="run letify's side on a remote machine")
    client_sub = client.add_subparsers(dest="client_command", required=True)
    client_shell = client_sub.add_parser("shell", help="a plain machine with no provider API")
    shell_sub = client_shell.add_subparsers(dest="shell_command", required=True)
    connect = shell_sub.add_parser(
        "connect", help="start the remote agent behind tailcat serve and print its address"
    )
    connect.add_argument(
        "--ssh-port", dest="ssh_port", type=int, default=22, help="this machine's SSH server"
    )
    connect.add_argument(
        "--public-address",
        dest="public_address",
        help="the address this machine's SSH server is reachable at from outside",
    )
    connect.add_argument(
        "--public-port",
        dest="public_port",
        type=int,
        help="the port published for this machine's SSH server, such as Docker's -p host port",
    )
    connect.add_argument("--tailcat", default="tailcat", help="the tailcat command")
    connect.add_argument(
        "--name", help="the alias the printed login command names; defaults to the host name"
    )

    return parser


def _client_shell_connect(args: argparse.Namespace) -> int:
    """Check tailcat and the SSH server, run the remote agent, and print the login command."""
    from .transport import setup
    from .transport.agent import Agent

    if not setup.tailcat_on_path(args.tailcat):
        print(setup.tailcat_install_instructions(), file=sys.stderr)
        return 1
    if not setup.ssh_answers(args.ssh_port):
        print(setup.sshd_missing_message(args.ssh_port), file=sys.stderr)
        return 1

    agent = Agent(ssh=("127.0.0.1", args.ssh_port), tailcat=args.tailcat)
    port = agent.bind()
    try:
        address = agent.start_tailcat()
    except (OSError, RuntimeError) as exc:
        agent.close()
        print(exc, file=sys.stderr)
        return 1
    fields = {
        "tailcat": address,
        "tailcat_port": port,
        "user": setup.local_user(),
        "port": args.ssh_port,
    }
    if args.public_address:
        fields["address"] = args.public_address
    if args.public_port:
        fields["public_port"] = args.public_port
    token = setup.encode_token(fields)
    alias = args.name or setup.default_alias()
    print("On your own machine, run:")
    print()
    print(f"  letify login tunnel {alias} --connect {token}")
    print()
    print("Keep this agent running: every connection to this machine goes through it.")
    print("To keep it running after you log out, start it inside tmux or with nohup:")
    print("  tmux new -s letify 'letify client shell connect'")
    print("  nohup letify client shell connect > letify-agent.log 2>&1 &")
    print("A restart gets a new address, so run the login again with the new token it prints.")
    sys.stdout.flush()
    try:
        agent.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover - interactive
        pass
    finally:
        agent.close()
    return 0


def _describe_usage(row: dict) -> str:
    """One line for a usage row, formatted by its unit as spec "Remaining usage" says."""
    from .providers.usage import describe_row

    return describe_row(row)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "efficiency":
        from .remoting import efficiency as compute

        share = compute(args.step_seconds, args.syncs, args.round_trip_ms)
        print(f"{share * 100:.1f}% of a direct run")
        return 0

    if args.command == "client":
        # Runs on the remote machine, which has no accounts, so no Launcher is built.
        return _client_shell_connect(args)

    if args.command == "login":
        alias = args.alias or args.kind
        answers = login.Answers(
            alias=alias,
            kind=args.kind,
            values={
                "address": args.address,
                "user": args.user,
                "port": args.port,
                "public_port": args.public_port,
                "key": args.key,
                "auth": args.auth,
                "persistent": args.persistent,
                "zone_id": args.zone_id,
                "machine_id": args.machine_id,
                "endpoint": args.endpoint,
                "account": args.account,
                "workspace": args.workspace,
                "profile": args.profile,
                "connect": args.connect,
                "indices": args.indices,
                "detect_devices": args.detect_devices,
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
        detail = " and its account directory" if forgotten else ""
        print(f"{args.alias} removed from {login.home_path()}{detail}")
        print("The project reference is left alone, because this repository still needs it")
        return 0

    let = Launcher(args.config, announce=False)

    if args.command == "stubs":
        from . import stubs

        written = stubs.write(let)
        if written is None:
            print("generation is turned off by [tool.letify] typings = false", file=sys.stderr)
            return 1
        print(written)
        return 0

    if args.command == "providers":
        for alias in let.config.order:
            try:
                provider = let.provider(alias)
            except Exception as exc:
                print(f"{alias:20} unavailable: {exc}")
                continue
            print(f"{alias:20} {provider.kind:10} {provider.persistence}")
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
        from . import render

        sys.stdout.write(render.usage_blocks(rows, render.Style.for_stream(sys.stdout)))
        return 0

    if args.command == "utilization":
        rows = let.utilization(args.alias)
        if args.json:
            print(json.dumps(rows, indent=2))
            return 0
        from . import render

        sys.stdout.write(render.utilization_blocks(rows, render.Style.for_stream(sys.stdout)))
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
