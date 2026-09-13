"""Step 1. Prove the whole path before spending anything on it.

One trivial call confirms every piece at once: the provider answers, the session starts, the
environment installed, this project's code arrived, the card is the one that was paid for,
and the result comes back. It costs seconds. Finding the same failure twenty minutes into a
sweep costs the sweep.

    python 00_smoke.py --provider local
    python 00_smoke.py --provider colab_a --device G4
"""

from __future__ import annotations

import json

import common
import recipe

import letify
from letify.remoting import efficiency


def main() -> int:
    arguments = common.parser(__doc__.splitlines()[0]).parse_args()
    let = letify.Launcher()
    instance = common.pick(let, arguments.provider, arguments.device)
    print(f"asking {common.describe(instance)} what it is")

    # The declaration is the only place the site appears. Everything below is ordinary
    # Python, and recipe.environment does not know it is running anywhere unusual.
    inspect = let.function(device=instance, host=arguments.host, env=common.ENV, timeout=600)(
        recipe.environment
    )
    found = inspect()
    print(json.dumps(found, indent=2, sort_keys=True))

    if not found.get("torch"):
        print("\ntorch is not installed in that environment, so add it to the lock file:")
        print("    uv add torch --index https://download.pytorch.org/whl/cu128")
        return 1
    if not found.get("cuda_available"):
        print("\nthe machine answered but has no usable CUDA device, so there is nothing to rent")
        return 1
    if not found.get("nvfp4"):
        print(
            f"\n{found['device']} is compute capability {found['capability']}, below the 10.0 "
            f"that NVFP4 tensor cores need. The example still runs; the quantized path falls "
            f"back, which costs speed rather than correctness."
        )

    # Measured on the machine that will do the training, because this number decides whether
    # host='local' is worth using and it depends on the training loop rather than the link.
    counted = let.function(device=instance, host=arguments.host, env=common.ENV, timeout=600)(
        recipe.count_syncs
    )()
    if counted.get("syncs") is not None:
        print(f"\nhost synchronizations per step: {counted['syncs']}")
        print("expected share of a direct run over a 150 ms link:")
        for step in (0.05, 0.5, 4.0):
            share = efficiency(step, max(counted["syncs"], 0.01), 150)
            print(f"    {step:>4} s per step: {share * 100:5.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
