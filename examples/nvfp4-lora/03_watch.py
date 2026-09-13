"""Step 4. Watch what it is costing while it runs, from another terminal.

Three questions, one per command line call, and this script only shows them in one place:

    letify status         what sessions exist and for how long
    letify usage          what is left on each account
    letify utilization    how hard each declared card is working

The third one is the one that catches the expensive mistake. A rented GPU sitting at 3 percent
is a dataloader problem, and it bills the same as one at 99 percent.

    python 03_watch.py --provider local --repeat 5
"""

from __future__ import annotations

import argparse
import time

import letify


def main() -> int:
    argue = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    argue.add_argument("--provider", default=None, help="one alias, or every declared one")
    argue.add_argument("--repeat", type=int, default=1, help="readings to take")
    argue.add_argument("--every", type=float, default=5.0, help="seconds between readings")
    arguments = argue.parse_args()

    let = letify.Launcher(announce=False)

    for row in let.usage(arguments.provider):
        if "unavailable" in row:
            print(f"{row['alias']:20} unavailable: {row['unavailable']}")
        elif row.get("unmetered"):
            print(f"{row['alias']:20} unmetered")
        elif row.get("remaining") is not None:
            print(f"{row['alias']:20} {row['remaining']:g} {row['unit']} left")
        else:
            print(f"{row['alias']:20} balance not reported ({row['source']})")

    print()
    for reading in range(arguments.repeat):
        if reading:
            time.sleep(arguments.every)
        for row in let.utilization(arguments.provider):
            if "unavailable" in row:
                continue
            head = f"{row['alias']}.{row['accelerator']}"
            if not row["devices"]:
                print(f"{head:28} {row['reason']}")
                continue
            for device in row["devices"]:
                busy = device["utilization_percent"]
                used = device["memory_used_gb"] or 0.0
                total = device["memory_total_gb"] or 0.0
                print(
                    f"{head:28} gpu{device['index']} "
                    f"{'?' if busy is None else f'{busy:3.0f}'}% busy "
                    f"{used:5.1f}/{total:5.1f} GiB"
                )

    # A session that this process did not start is not visible here, because the pool lives in
    # the process that owns it. Ask the machine itself for that, which is what utilization does.
    print(f"\n{len(let.status()['runtimes'])} session(s) held by this process")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
