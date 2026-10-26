"""Command line entry points.

Enough to answer what comes up before any code is written: which providers are declared,
what they offer, what is running right now, whether a machine answers, and whether
forwarding PyTorch operators to it is worth doing.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

from . import __version__, render
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

    providers_parser = sub.add_parser("providers", help="list declared providers and their storage")
    providers_parser.add_argument(
        "--json", action="store_true", help="print the records unformatted"
    )
    devices_parser = sub.add_parser("devices", help="list the accelerators each provider offers")
    devices_parser.add_argument("--json", action="store_true", help="print the records unformatted")
    status_parser = sub.add_parser("status", help="show live runtimes and what they are costing")
    status_parser.add_argument("--json", action="store_true", help="print the records unformatted")
    sessions_parser = sub.add_parser("sessions", help="list provider-owned sessions")
    sessions_parser.add_argument(
        "--json", action="store_true", help="print the records unformatted"
    )
    stop = sub.add_parser("stop", help="stop what a dead process left behind")
    stop.add_argument(
        "--orphans",
        action="store_true",
        help="delete the sessions whose creating process is gone",
    )
    stop.add_argument("--alias", help="only this account")
    stop.add_argument("-y", "--yes", action="store_true", help="do not ask")
    stop.add_argument("--json", action="store_true", help="print the records unformatted")

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
    log_in.add_argument("kind", help="provider kind: shell, tunnel, colab, modal, elice, kaggle")
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
    log_in.add_argument(
        "--key", help="SSH private key path; with --username, the Kaggle API token instead"
    )
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
    log_in.add_argument(
        "--machine-id",
        dest="machine_id",
        help="Elice machine name or id; without it letify creates one on first use",
    )
    log_in.add_argument(
        "--price-type",
        dest="price_type",
        choices=("ondemand", "spot"),
        help="Elice price type for machines letify creates; ondemand by default",
    )
    log_in.add_argument("--endpoint", help="API endpoint, where it is not the default")
    log_in.add_argument(
        "--organization", help="Elice organization short name, sent as x-elice-org-name-short"
    )
    log_in.add_argument(
        "--billing-endpoint", dest="billing_endpoint", help="Elice billing API base URL"
    )
    log_in.add_argument("--account", help="account email, for Colab")
    log_in.add_argument(
        "--username",
        help=(
            "Kaggle username, with --key the API token from Settings > API on kaggle.com; "
            "required, for everything the official CLI covers that the cookie does not"
        ),
    )
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
        "--cookie",
        metavar="COOKIE",
        help=(
            "the browser cookie of a logged-in kaggle.com tab, or a file holding it, for "
            "kaggle; required alongside --username/--key, which the official CLI needs"
        ),
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
    log_in.add_argument(
        "--replace",
        action="store_true",
        help="run the login again for an alias that already exists, to renew its credentials",
    )

    log_out = sub.add_parser("logout", help="remove an account from this machine")
    log_out.add_argument("alias", help="provider alias to forget")

    check = sub.add_parser("check", help="check that a provider answers")
    check.add_argument("alias", help="provider alias from the configuration")

    cache = sub.add_parser("cache", help="show the data caches, or clear a provider's")
    cache.add_argument(
        "action",
        nargs="?",
        choices=("clear", "clean"),
        help="clear a runtime cache or local cache",
    )
    cache.add_argument("alias", nargs="?", help="provider alias or local cache target to clear")
    cache.add_argument("--json", action="store_true", help="print the records unformatted")

    probe = sub.add_parser("probe", help="measure whether host='local' is worth using")
    probe.add_argument("host", nargs="?", help="host name to measure the round trip to")
    probe.add_argument("--json", action="store_true", help="print the record unformatted")

    efficiency = sub.add_parser(
        "efficiency", help="expected fraction of a direct run, from measured terms"
    )
    efficiency.add_argument("step_seconds", type=float, help="GPU time per step")
    efficiency.add_argument("syncs", type=int, help="host synchronizations per step")
    efficiency.add_argument("round_trip_ms", type=float, help="network round trip")
    efficiency.add_argument("--json", action="store_true", help="print the record unformatted")

    runs_parser = sub.add_parser("runs", help="list the calls running from this machine")
    runs_parser.add_argument("--json", action="store_true", help="print the records unformatted")
    interrupt_parser = sub.add_parser(
        "interrupt", help="raise KeyboardInterrupt in a running call's body"
    )
    interrupt_parser.add_argument("run", help="a run id, as letify runs lists it")
    interrupt_parser.add_argument(
        "--json", action="store_true", help="print the record unformatted"
    )
    exec_parser = sub.add_parser("exec", help="run Python source inside a running call's worker")
    exec_parser.add_argument("run", help="a run id, as letify runs lists it")
    exec_parser.add_argument("code", nargs="?", help="the source to run")
    exec_parser.add_argument("-f", "--file", help="read the source from this file instead")
    exec_parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        metavar="S",
        help="seconds to wait for the answer; the source keeps running after that",
    )
    exec_parser.add_argument("--json", action="store_true", help="print the record unformatted")

    setup_parser = sub.add_parser(
        "setup", help="install tailcat or eci from its publisher ahead of time"
    )
    setup_parser.add_argument("tool", choices=("tailcat", "eci"), help="the tool to install")
    setup_parser.add_argument(
        "--where", action="store_true", help="print where the tool is and which copy letify uses"
    )

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
    from .install import InstallError, ensure
    from .transport import setup
    from .transport.agent import Agent

    if args.tailcat == "tailcat":
        try:
            args.tailcat = ensure("tailcat", instructions=setup.tailcat_install_instructions())
        except InstallError as exc:
            _fail(str(exc))
            return 1
    elif not setup.tailcat_on_path(args.tailcat):
        _fail(setup.tailcat_install_instructions())
        return 1
    if not setup.ssh_answers(args.ssh_port):
        _fail(setup.sshd_missing_message(args.ssh_port))
        return 1

    agent = Agent(ssh=("127.0.0.1", args.ssh_port), tailcat=args.tailcat)
    port = agent.bind()
    try:
        address = agent.start_tailcat()
    except (OSError, RuntimeError) as exc:
        agent.close()
        _fail(str(exc))
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
    style = _out()
    print(style.bold("On your own machine, run:"))
    print()
    print(f"  letify login tunnel {alias} --connect {token}")
    print()
    print(style.dim("Keep this agent running: every connection to this machine goes through it."))
    print(style.dim("To keep it running after you log out, start it inside tmux or with nohup:"))
    print("  tmux new -s letify 'letify client shell connect'")
    print("  nohup letify client shell connect > letify-agent.log 2>&1 &")
    print(
        style.dim(
            "A restart gets a new address, so run the login again with the new token it prints."
        )
    )
    sys.stdout.flush()
    try:
        agent.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover - interactive
        pass
    finally:
        agent.close()
    return 0


def _setup_tool(args: argparse.Namespace) -> int:
    """``letify setup <tool>``. Spec "Installing external tools"."""
    from . import install

    tool = args.tool
    if args.where:
        rows = install.where(tool)
        width = max(len(label) for label, _ in rows)
        for label, value in rows:
            print(f"{label.ljust(width)}  {value}")
        return 0
    version = install.version_of(tool)
    found = install.find(tool)
    if found is None:
        try:
            install.install_and_link(tool)
        except install.InstallError as exc:
            _fail(str(exc))
            return 1
        found = install.find(tool)
    if found is None:  # pragma: no cover - brew put tailcat off PATH
        _fail(f"{tool} was installed but letify cannot find it")
        return 1
    _say("ok", f"{tool} {version} at {found}")
    return 0


def _running_runtime(let: Launcher, alias: str):
    """A session this process already holds on that provider, or None.

    Spec "The cache command": the listing asks only what is already running, because a
    session it started would bill for the listing.
    """
    for runtime in let.pool.live:
        if getattr(runtime.provider, "alias", None) == alias:
            return runtime
    return None


def _data_cache_reply(let: Launcher, runtime, clear: bool) -> dict:
    """One ``data_cache`` request to a session that is already up."""
    from .store import pathdata

    provider = runtime.provider
    root = (runtime.workspace or provider.workspace_root).rstrip("/")
    return runtime.request(
        {
            "op": "data_cache",
            "dir": f"{root}/data/blobs",
            "budget": pathdata.budget_bytes(provider),
            "clear": clear,
        },
        timeout=600,
    )


def _data_cache_request(let: Launcher, alias: str, clear: bool) -> dict:
    """Ask a session on the provider's first instance about its file blob cache.

    This starts one when none is running, so only a named provider reaches it. Spec "The
    cache command".
    """
    from .declare import Env
    from .store import pathdata

    provider = let.provider(alias)
    from . import remote

    # The worker answers the request, so no PyTorch is needed on this side.
    instance = next(iter(provider.instances.values()))._placed(remote)
    with let.invocation():
        runtime = let.pool.acquire(instance, Env())
        try:
            root = (runtime.workspace or provider.workspace_root).rstrip("/")
            payload = {
                "op": "data_cache",
                "dir": f"{root}/data/blobs",
                "budget": pathdata.budget_bytes(provider),
                "clear": clear,
            }
            return runtime.request(payload, timeout=600)
        finally:
            let.pool.release(runtime)


def _measure_tree(path: Path) -> tuple[int, int]:
    """Return (file_count, total_bytes) for all files under path."""
    if not path.exists():
        return 0, 0
    if path.is_file():
        return 1, path.stat().st_size
    files = 0
    total_bytes = 0
    for root, _, filenames in os.walk(path):
        for name in filenames:
            p = Path(root) / name
            try:
                stat_res = p.stat(follow_symlinks=False)
                files += 1
                total_bytes += stat_res.st_size
            except OSError:
                pass
    return files, total_bytes


def _cache(let: Launcher, args: argparse.Namespace) -> int:
    """Spec "The cache command"."""
    from .paths import (
        accounts_directory,
        cache_directory,
        local_runtime_directory,
        local_tmp_directory,
        ssh_control_directory,
        tools_directory,
    )
    from .store import pathdata

    mib = 1 << 20
    local_trees: dict[str, tuple[Path, bool]] = {
        "cache": (cache_directory(), True),
        "tools": (tools_directory(), True),
        "tmp": (local_tmp_directory(), True),
        "ssh": (ssh_control_directory(), True),
        "runtime": (local_runtime_directory(), True),
        "accounts": (accounts_directory(), False),
    }

    if args.action in ("clear", "clean"):
        target = args.alias
        if not target:
            _fail("name the provider or local tree to clear: letify cache clear <target>")
            return 1
        if target in ("accounts", "config", "config.toml"):
            _fail(f"refused: {target} holds credentials or configuration and cannot be cleaned")
            return 1

        local_targets = ("cache", "tools", "tmp", "ssh", "runtime", "all")
        if target in local_targets:
            to_clean = ["cache", "tools", "tmp", "runtime"] if target == "all" else [target]
            total_removed = 0
            total_removed_bytes = 0
            for item in to_clean:
                dir_path, cleanable = local_trees[item]
                if cleanable and dir_path.exists():
                    cnt, sz = _measure_tree(dir_path)
                    total_removed += cnt
                    total_removed_bytes += sz
                    for entry in list(dir_path.iterdir()):
                        if entry.is_dir():
                            shutil.rmtree(entry, ignore_errors=True)
                        else:
                            entry.unlink(missing_ok=True)
            if args.json:
                return _json(
                    {
                        "target": target,
                        "removed": total_removed,
                        "removed_bytes": total_removed_bytes,
                    }
                )
            print(
                f"{target}: removed {total_removed} files "
                f"{total_removed_bytes / mib:.1f} MiB"
            )
            return 0

        # Provider target
        try:
            result = _data_cache_request(let, target, True)
        except LetifyError as exc:
            _fail(str(exc))
            return 1
        if args.json:
            return _json({"alias": target, **result})
        print(
            f"{target}: removed {result['removed']} files "
            f"{result['removed_bytes'] / mib:.1f} MiB"
        )
        return 0

    digests = pathdata.DigestCache()
    pruned = digests.prune()
    digests.save()

    local_stats: dict[str, dict] = {}
    for name, (path, cleanable) in local_trees.items():
        count, size_bytes = _measure_tree(path)
        local_stats[name] = {
            "files": count,
            "bytes": size_bytes,
            "cleanable": cleanable,
        }

    records: list[dict] = []
    for alias in let.config.order:
        try:
            provider = let.provider(alias)
            if not provider.persistent:
                records.append({"alias": alias, "kept": False})
                continue
            # Spec "The cache command": a listing bills nothing, so it starts a session
            # only where starting one is free, and otherwise asks one that is already up.
            if provider.kind == "local":
                result = _data_cache_request(let, alias, False)
            else:
                runtime = _running_runtime(let, alias)
                if runtime is None:
                    records.append({"alias": alias, "kept": True, "running": False})
                    continue
                result = _data_cache_reply(let, runtime, False)
        except LetifyError as exc:
            records.append({"alias": alias, "unavailable": str(exc)})
            continue
        records.append(
            {
                "alias": alias,
                "kept": True,
                "running": True,
                "files": result["files"],
                "bytes": result["bytes"],
                "budget": result["budget"],
            }
        )
    record = {
        "digests": {"entries": len(digests), "pruned": pruned},
        "local": local_stats,
        "providers": records,
    }
    if args.json:
        return _json(record)

    print(f"digest cache: {len(digests)} entries, {pruned} pruned")

    local_rows = []
    for name, (_path, cleanable) in local_trees.items():
        st = local_stats[name]
        clean_str = "yes" if cleanable else "no (credentials)"
        local_rows.append([name, str(st["files"]), f"{st['bytes'] / mib:.1f} MiB", clean_str])
    sys.stdout.write(render.table(["LOCAL", "FILES", "SIZE", "CLEANABLE"], local_rows, _out()))

    rows = []
    for row in records:
        if "unavailable" in row:
            rows.append([row["alias"], f"unavailable: {row['unavailable']}", "", ""])
        elif not row["kept"]:
            rows.append([row["alias"], "not kept between sessions", "", ""])
        elif not row.get("running"):
            rows.append([row["alias"], "no session running", "", ""])
        else:
            rows.append(
                [
                    row["alias"],
                    str(row["files"]),
                    f"{row['bytes'] / mib:.1f} MiB",
                    f"{row['budget'] / (1 << 30):.1f} GiB",
                ]
            )
    sys.stdout.write(render.table(["PROVIDER", "BLOBS", "SIZE", "BUDGET"], rows, _out()))
    return 0


def _describe_usage(row: dict) -> str:
    """One line for a usage row, formatted by its unit as spec "Remaining usage" says."""
    from .providers.usage import describe_row

    return describe_row(row)


def _out() -> render.Style:
    return render.Style.for_stream(sys.stdout)


def _fail(message: str) -> None:
    """One failure on standard error, marked as spec "Command line" says."""
    style = render.Style.for_stream(sys.stderr)
    print(f"{render.mark('fail', style)} {message}", file=sys.stderr)


def _say(kind: str, message: str) -> None:
    """One marked line on standard output."""
    print(f"{render.mark(kind, _out())} {message}")


def _detail(row: dict) -> str:
    """The reason a stop failed, when there is one to add."""
    return f": {row['error']}" if row.get("error") else ""


def _confirmed(question: str) -> bool:
    """Ask before deleting. Anything but yes is no, and no terminal is no."""
    import sys as _sys

    if not _sys.stdin.isatty():
        print(f"{question} not a terminal, so nothing is assumed. Pass -y to delete")
        return False
    try:
        answer = input(f"{question} [y/N] ").strip().lower()
    except EOFError:
        return False
    return answer in ("y", "yes")


def _run_control(args: argparse.Namespace) -> int:
    """``letify runs``, ``interrupt`` and ``exec``: spec "Controlling a running call".

    Exit status 0 on success, 2 for a usage error, 3 when the run is not found and 1 for
    any other failure.
    """
    from .errors import RunNotFound, UnsupportedMode
    from .runtime import control

    def refuse(message: str, status: int, kind: str = "Error") -> int:
        if args.json:
            _json({"ok": False, "type": kind, "error": message})
        else:
            _fail(message)
        return status

    if args.command == "runs":
        if not control.available():
            return refuse("run control is not available on Windows", 1, "UnsupportedMode")
        rows = control.list_runs()
        if args.json:
            return _json(rows)
        if not rows:
            print("no calls are running")
            return 0
        now = time.time()
        table = [("RUN", "FUNCTION", "RUNTIME", "PROVIDER", "AGE", "PID")] + [
            (
                str(row["run"]),
                str(row.get("function", "")),
                str(row.get("runtime", "")),
                str(row.get("provider", "")),
                _age(now - float(row.get("started_at", now))),
                str(row.get("pid", "")),
            )
            for row in rows
        ]
        widths = [max(len(line[column]) for line in table) for column in range(6)]
        for line in table:
            cells = (cell.ljust(width) for cell, width in zip(line, widths, strict=True))
            print("  ".join(cells).rstrip())
        return 0

    command: dict[str, object] = {"cmd": args.command}
    wait = 90.0
    if args.command == "exec":
        if (args.code is None) == (args.file is None):
            return refuse("give the source as CODE or with -f FILE, not both and not neither", 2)
        if args.timeout <= 0:
            return refuse("--timeout must be positive", 2)
        try:
            source = args.code if args.code is not None else Path(args.file).read_text("utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            return refuse(f"cannot read {args.file}: {exc}", 1)
        command.update({"source": source, "timeout": args.timeout})
        wait = args.timeout + 30
    try:
        answer = control.ask(args.run, command, wait=wait)
    except RunNotFound as exc:
        return refuse(str(exc), 3, "RunNotFound")
    except UnsupportedMode as exc:
        return refuse(str(exc), 1, "UnsupportedMode")
    failed = not answer.get("ok")
    if args.command == "exec" and answer.get("error"):
        failed = True
    if args.json:
        _json(answer)
        return 1 if failed else 0
    if not answer.get("ok"):
        kind = answer.get("type") or "Error"
        reason = answer.get("error") or answer.get("reason") or "failed"
        _fail(f"{kind}: {reason}" if kind != "Error" else str(reason))
        return 1
    if args.command == "interrupt":
        if not answer.get("interrupted"):
            _fail(f"{args.run} was not interrupted: {answer.get('reason', 'unknown reason')}")
            return 1
        _say("ok", f"{args.run} interrupted")
        return 0
    sys.stdout.write(str(answer.get("stdout") or ""))
    sys.stderr.write(str(answer.get("stderr") or ""))
    if answer.get("error"):
        sys.stderr.write(str(answer["error"]))
        return 1
    if answer.get("has_value"):
        print(answer["value"])
    assigned = answer.get("assigned") or []
    if assigned and not answer.get("applied"):
        _fail(
            "this Python does not write assignments to f_locals back into the running "
            f"frame, so {', '.join(map(str, assigned))} was not applied"
        )
    return 0


def _age(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m"


def _json(value: object) -> int:
    print(json.dumps(value, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _dispatch(args)
    except LetifyError as exc:
        _fail(str(exc))
        return 1


def _dispatch(args: argparse.Namespace) -> int:
    if args.command == "efficiency":
        from .remoting import efficiency as compute

        share = compute(args.step_seconds, args.syncs, args.round_trip_ms)
        if args.json:
            return _json({"efficiency": share})
        print(f"{share * 100:.1f}% of a direct run")
        return 0

    if args.command == "setup":
        return _setup_tool(args)

    if args.command in ("runs", "interrupt", "exec"):
        # Talks to the process that owns the run, so no Launcher is built.
        return _run_control(args)

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
                "price_type": args.price_type,
                "endpoint": args.endpoint,
                "organization": args.organization,
                "billing_endpoint": args.billing_endpoint,
                "account": args.account,
                "username": args.username,
                "workspace": args.workspace,
                "profile": args.profile,
                "connect": args.connect,
                "cookie": args.cookie,
                "indices": args.indices,
                "detect_devices": args.detect_devices,
            },
            token=args.token,
            interactive=args.interactive,
            install_key=args.install_key,
            replace=args.replace,
        )
        try:
            fresh, home, project = login.log_in(answers, project=args.config)
        except LetifyError as exc:
            _fail(str(exc))
            return 1
        if fresh:
            what = "renewed" if args.replace else "declared"
            _say("ok", f"{alias} {what} in {home}")
        else:
            _say(
                "warn",
                f"{alias} was already declared in {home}, so nothing was asked for. "
                f"Pass --replace to renew its credentials",
            )
        _say("ok", f"{alias} referenced in {project}, which is safe to commit")
        return 0

    if args.command == "logout":
        removed, forgotten = login.log_out(args.alias)
        if not removed:
            _fail(f"{args.alias} is not declared in {login.home_path()}")
            return 1
        detail = " and its account directory" if forgotten else ""
        _say("ok", f"{args.alias} removed from {login.home_path()}{detail}")
        print(
            _out().dim(
                "The project reference is left alone, because this repository still needs it"
            )
        )
        return 0

    let = Launcher(args.config, announce=False)

    if args.command == "stubs":
        from . import stubs

        written = stubs.write(let)
        if written is None:
            _fail("generation is turned off by [tool.letify] typings = false")
            return 1
        _say("ok", str(written))
        return 0

    if args.command == "providers":
        records: list[dict[str, str]] = []
        for alias in let.config.order:
            try:
                provider = let.provider(alias)
            except Exception as exc:
                records.append({"alias": alias, "unavailable": str(exc)})
                continue
            record = {
                "alias": alias,
                "kind": provider.kind,
                "persistence": str(provider.persistence),
            }
            note = provider.account_note()
            if note:
                record["note"] = note
            records.append(record)
        if args.json:
            return _json(records)
        style = _out()
        def row(r: dict[str, str]) -> list[str]:
            if "unavailable" in r:
                why = f"{render.mark('fail', style)} unavailable: {r['unavailable']}"
                return [r["alias"], why, "", ""]
            return [r["alias"], r["kind"], r["persistence"], r.get("note", "")]

        rows = [row(r) for r in records]
        sys.stdout.write(render.table(["ALIAS", "KIND", "PERSISTENCE", "NOTES"], rows, style))
        return 0

    if args.command == "devices":
        table = let.providers.devices
        if args.json:
            print(json.dumps(table, indent=2, sort_keys=True))
            return 0
        rows = [[alias, ", ".join(str(name) for name in names)] for alias, names in table.items()]
        sys.stdout.write(render.table(["PROVIDER", "ACCELERATORS"], rows, _out()))
        return 0

    if args.command == "sessions":
        rows = let.sessions()
        if args.json:
            return _json(rows)
        for row in rows:
            names = ", ".join(row["sessions"])
            detail = row.get("unavailable") or row.get("reason") or names or "no active sessions"
            print(f"{row['alias']}  {detail}")
        return 0

    if args.command == "stop":
        if not args.orphans:
            # Spec "Kaggle", stopping what a dead process left behind: a bare stop reads as
            # ending everything, so it ends nothing, and says why rather than only that.
            print(
                "nothing was stopped. A session whose process is still running ends when "
                "that process ends, because taking it away leaves that process calling a "
                "runtime that is gone: send SIGTERM to the process that started it, which "
                "letify handles by closing its sessions. Pass --orphans to delete what a "
                "dead process left."
            )
            return 0
        rows = let.orphans(args.alias)
        found = [(row["alias"], ref) for row in rows for ref in row.get("orphans", ())]
        unavailable = [row for row in rows if row.get("unavailable")]
        if args.json and not found:
            return _json(rows)
        for row in unavailable:
            print(f"{row['alias']}  {row['unavailable']}")
        if not found:
            print("no orphaned sessions")
            return 0
        print(f"{len(found)} orphaned session(s) to delete:")
        for alias, ref in found:
            print(f"  {alias}  {ref}")
        if not args.yes and not _confirmed("Delete them?"):
            print("nothing was stopped")
            return 0
        done = let.stop_orphans(args.alias)
        if args.json:
            return _json(done)
        failed = 0
        for row in done:
            if row.get("unavailable"):
                continue
            mark = "stopped" if row.get("stopped") else f"failed{_detail(row)}"
            failed += 0 if row.get("stopped") else 1
            print(f"{row['alias']}  {row['ref']}  {mark}")
        return 1 if failed else 0

    if args.command == "status":
        status = let.status()
        if args.json:
            return _json(status)
        # A live runtime is what costs money, so only its account is asked what is left.
        asked: dict[str, dict] = {}
        for runtime in status.get("runtimes") or []:
            alias = str(runtime["provider"])
            if alias not in asked:
                rows = let.usage(alias)
                asked[alias] = rows[0] if rows else {}
            runtime["usage"] = asked[alias] or None
        sys.stdout.write(render.status_text(status, _out()))
        return 0

    if args.command == "usage":
        rows = let.usage(args.alias)
        if args.json:
            return _json(rows)
        sys.stdout.write(render.usage_blocks(rows, render.Style.for_stream(sys.stdout)))
        return 0

    if args.command == "utilization":
        rows = let.utilization(args.alias)
        if args.json:
            return _json(rows)
        sys.stdout.write(render.utilization_blocks(rows, render.Style.for_stream(sys.stdout)))
        return 0

    if args.command == "cache":
        return _cache(let, args)

    if args.command == "check":
        provider = let.provider(args.alias)
        checker = getattr(provider, "check", None)
        if checker is None:
            _fail(f"{args.alias} has no check step")
            return 1
        answer = checker()
        _say("ok", f"{args.alias} answers")
        for line in str(answer).splitlines():
            print(f"  {line}")
        return 0

    if args.command == "probe":
        from .remoting import probe as run_probe

        capability = run_probe(args.host)
        record = {
            "platform": capability.platform,
            "core": capability.core,
            "agent": capability.agent,
            "round_trip_ms": capability.round_trip_ms,
            "usable": capability.usable,
            "costly": capability.costly,
            "reason": capability.explain(),
        }
        if args.json:
            return _json(record)
        if capability.usable and capability.costly:
            _say("warn", "forwarding usable but costly")
        elif capability.usable:
            _say("ok", "forwarding usable")
        else:
            _say("fail", "forwarding not usable")
        rtt = capability.round_trip_ms
        pairs = [
            ("platform", str(capability.platform)),
            ("core", str(capability.core)),
            ("agent", str(capability.agent)),
            ("round trip", f"{rtt:.1f} ms" if rtt is not None else "not measured"),
        ]
        style = _out()
        sys.stdout.write(render.fields(pairs, style))
        print(style.dim(str(record["reason"])))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
