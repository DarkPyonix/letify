"""Step 2. The scenario: a LoRA sweep on a rented card, with every result kept.

Six configurations, two sessions, one declaration. What the example is really showing is
what you do not have to write: no session management, no transfer code, no cleanup, and no
wrapper around the training function, which stays the plain function in recipe.py.

    python 01_sweep.py --provider local
    python 01_sweep.py --provider colab_a --device G4 --concurrency 2

Costs, at the prices this project was started to deal with. The RTX PRO 6000 is about
975 KRW per hour through Colab compute units against about 4,070 KRW per hour on Modal, so
a six point sweep of ten minutes each is roughly 975 KRW rather than 4,070 KRW. Session
start, environment install and the first data transfer are all billed as GPU time, which is
why the sweep reuses one warm session per slot instead of starting six.
"""

from __future__ import annotations

import common
import recipe

import letify


def main() -> int:
    argue = common.parser(__doc__.splitlines()[0])
    argue.add_argument("--concurrency", type=int, default=1, help="sessions to use at once")
    argue.add_argument("--steps", type=int, default=60, help="training steps per point")
    argue.add_argument("--tag", default="sweep-1", help="checkpoint name to write under")
    arguments = argue.parse_args()

    let = letify.Launcher()
    instance = common.pick(let, arguments.provider, arguments.device)
    cache = let.provider(arguments.provider).volume(common.VOLUME_NAME)
    print(f"sweeping on {common.describe(instance)}, {arguments.concurrency} session(s) at once")

    # The space is declared, not looped over. Passing it where a scalar belongs is what says
    # the argument varies, so the same declaration serves one point and thirty.
    space = letify.grid(lr=[1e-4, 3e-4, 1e-3], rank=[8, 32]).with_fixed(steps=arguments.steps)
    print(f"{len(space)} points")

    # lifetime='process' keeps the session between points, so the second point pays nothing
    # for setup. Without it each point would start and stop its own session, and on a
    # provider that bills by the second the setup would cost more than the training.
    train = let.function(
        device=instance,
        host=arguments.host,
        env=common.ENV,
        volumes=[cache],
        lifetime="process",
        concurrency=arguments.concurrency,
        timeout=3600,
        retries=1,
    )(recipe.train)

    results = train(space)

    print()
    print(recipe.summarize(results))
    recipe.write_report(results, "sweep-report.json")
    print("\nwrote sweep-report.json")

    # The session is still warm, and it is the one holding the adapters. Asking the pool for
    # the same instance and environment hands back that session rather than starting another,
    # which is the whole reason a checkpoint can be pulled out after the calls are done.
    best = min(results, key=lambda row: row["final_loss"])
    # train.device rather than instance, because the declaration folded the host
    # placement into it and the pool keys a session by that too. Asking with the bare
    # instance starts a second session, which never ran the training and holds none of
    # its files.
    session = let.runtime(train.device, env=common.ENV, volumes=[cache])
    digest = cache.absorb(session, best["adapter"], f"{arguments.tag}-best")
    print(
        f"kept the best adapter (rank {best['rank']}, lr {best['lr']:g}, "
        f"loss {best['final_loss']:.4f}) as {arguments.tag}-best at {digest[:12]}"
    )

    # Nothing to release. The session ends with this process, and the lease means it would
    # end itself even if this process were killed.
    live = let.status()["runtimes"]
    print(f"\n{len(live)} session(s) live; they end with this process")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
