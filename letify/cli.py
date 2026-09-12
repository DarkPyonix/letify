"""The ``letify`` command.

Enough to answer the questions that come up before any code is written: which
providers are declared, which accelerators they offer, what is running right now,
and whether a machine is close enough for CUDA call forwarding to be worth using.
"""

from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .launcher import Launcher


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="letify", description="Declarations that become infrastructure."
    )
    parser.add_argument("--version", action="version", version=f"letify {__version__}")
    parser.add_argument("--config", help="path to a .letify file")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("providers", help="list declared providers and their persistence")
    sub.add_parser("gpus", help="list the accelerators each provider offers")
    sub.add_parser("status", help="show open scopes and live runtimes")

    check = sub.add_parser("check", help="check that a provider answers")
    check.add_argument("alias", help="provider alias from the configuration")

    probe = sub.add_parser("probe", help="measure whether CUDA call forwarding is worth using")
    probe.add_argument("host", nargs="?", help="host name to measure the round trip to")

    args = parser.parse_args(argv)
    let = Launcher(args.config)

    if args.command == "providers":
        for alias in let.config.order:
            try:
                provider = let.provider(alias)
                row = f"{alias:20} {provider.kind:10} {provider.persistence:11}"
                print(f"{row} cpu={provider.default_cpu_placement}")
            except Exception as exc:
                print(f"{alias:20} unavailable: {exc}")
        return 0

    if args.command == "gpus":
        print(json.dumps(let.providers.gpus, indent=2, sort_keys=True))
        return 0

    if args.command == "status":
        print(json.dumps(let.status(), indent=2))
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
        from .remoting import probe as probe_remoting

        capability = probe_remoting(args.host)
        print(
            json.dumps(
                {
                    "tun_device": capability.tun_device,
                    "driver_shim": capability.driver_shim,
                    "round_trip_ms": capability.round_trip_ms,
                    "usable": capability.usable,
                    "reason": capability.explain(),
                },
                indent=2,
            )
        )
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
