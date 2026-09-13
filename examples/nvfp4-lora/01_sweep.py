"""Step 2. The scenario: a LoRA sweep on a rented card, with every result kept.

Six configurations, two sessions, one declaration. What the example is really showing is
what you do not have to write: no session management, no transfer code, no cleanup, and no
wrapper around the training function, which stays the plain function in recipe.py.

    python 01_sweep.py --provider local
    python 01_sweep.py --provider colab_pro --device G4

How wide it runs is not an argument here. It is whatever the provider entry says the
account has, so two declared cards means two sessions and four means four:

    [colab_pro.devices]
    G4 = { count = 2 }

Costs, at the prices this project was started to deal with. The RTX PRO 6000 is about
975 KRW per hour through Colab compute units against about 4,070 KRW per hour on Modal, so
a six point sweep of ten minutes each is roughly 975 KRW rather than 4,070 KRW. Session
start, environment install and the first data transfer are all billed as GPU time, which is
why the sweep reuses a warm session per card instead of starting one per point.
"""

from __future__ import annotations

import common
import recipe

import letify


def main() -> int:
    argue = common.parser(__doc__.splitlines()[0])
    argue.add_argument("--steps", type=int, default=60, help="training steps per point")
    argue.add_argument("--tag", default="sweep-1", help="checkpoint name to write under")
    arguments = argue.parse_args()

    let = letify.Launcher()
    instance = common.pick(let, arguments.provider, arguments.device)
    cache = let.provider(arguments.provider).volume(common.VOLUME_NAME)
    capacity = instance.provider.capacity(instance.accelerator)
    print(f"sweeping on {common.describe(instance)}, {capacity} card(s) declared")

    # The space is declared, not looped over. Passing it where a scalar belongs is what says
    # the argument varies, so the same declaration serves one point and thirty.
    space = letify.grid(lr=[1e-4, 3e-4, 1e-3], rank=[8, 32]).with_fixed(steps=arguments.steps)
    print(f"{len(space)} points")

    # The keep_alive block below keeps the session between points, so the second point pays
    # nothing for setup, and keeps it long enough to pull the best adapter out afterwards. How
    # many sessions exist at once is the provider's inventory, not a number here.
    train = let.function(
        device=instance,
        host=arguments.host,
        env=common.ENV,
        volumes=[cache],
        timeout=3600,
        retries=1,
    )(recipe.train)

    with let.keep_alive():
        results = train(space)

        print()
        print(recipe.summarize(results))
        recipe.write_report(results, "sweep-report.json")
        print("\nwrote sweep-report.json")

        # The session is still warm, and it is the one holding the adapters, which is what makes
        # a checkpoint something that can be pulled out after the calls are done.
        best = min(results, key=lambda row: row["final_loss"])
        # The declaration is what is named, not a session. Which session ran the calls is
        # letify's answer, and it is the only one holding the adapter.
        digest = cache.absorb(train, best["adapter"], f"{arguments.tag}-best")
        print(
            f"kept the best adapter (rank {best['rank']}, lr {best['lr']:g}, "
            f"loss {best['final_loss']:.4f}) as {arguments.tag}-best at {digest[:12]}"
        )

    # Nothing to release. Leaving the keep_alive block ended the session, and the lease would
    # have ended it even if this process had been killed.
    print(f"\n{let.status()['live']} session(s) live after the block")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
