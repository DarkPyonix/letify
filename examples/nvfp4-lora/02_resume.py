"""Step 3. Carry on from the checkpoint, which is what makes a cheap card usable.

A cheap GPU is a preemptible GPU. Colab reclaims a runtime, a spot instance goes away, a lab
machine reboots. The answer is not a session that survives; it is a checkpoint that does.
This script puts the newest one back inside a fresh session and continues from the step it
stopped at.

    python 02_resume.py --provider local --tag sweep-1
"""

from __future__ import annotations

import common
import recipe

import letify

#: Where the checkpoint is placed inside the runtime, so the training function finds it as an
#: ordinary path rather than knowing anything about the store.
RESUME_PATH = f"{recipe.WORKSPACE}/resume.pt"


def main() -> int:
    argue = common.parser(__doc__.splitlines()[0])
    argue.add_argument("--tag", default="sweep-1", help="checkpoint name to continue from")
    argue.add_argument("--steps", type=int, default=60, help="further steps to train")
    argue.add_argument("--lr", type=float, default=3e-4)
    argue.add_argument("--rank", type=int, default=32)
    arguments = argue.parse_args()

    let = letify.Launcher()
    instance = common.pick(let, arguments.provider, arguments.device)
    cache = let.provider(arguments.provider).volume(common.VOLUME_NAME)

    name = f"{arguments.tag}-best"
    if cache.latest_checkpoint(name) is None:
        print(f"nothing stored under {name!r}. Run 01_sweep.py first.")
        return 1

    # The session is started before the call rather than by it, because the checkpoint has to
    # be inside the runtime before the training function looks for it. The pool hands the same
    # warm session to the call below.
    train = let.function(
        device=instance,
        host=arguments.host,
        env=common.ENV,
        volumes=[cache],
        lifetime="process",
        timeout=3600,
    )(recipe.train)

    # The declaration is named, not a session, so the checkpoint lands in the session the
    # call below is handed. It has to be there before the training function looks for it.
    digest = cache.resume(train, name, RESUME_PATH)
    print(f"put {name} ({digest[:12]}) inside the session at {RESUME_PATH}")

    result = train(
        lr=arguments.lr,
        rank=arguments.rank,
        steps=arguments.steps,
        resume_from=RESUME_PATH,
    )
    print(
        f"continued from step {result['resumed_at']} to "
        f"{result['resumed_at'] + result['steps']}, loss {result['final_loss']:.4f}"
    )

    # The name moves to the new blob and the old one stays where it is. Blobs are immutable
    # and a name is a few dozen bytes, so two sessions writing at once cannot lose each
    # other's work: one name wins and both checkpoints remain.
    moved = cache.absorb(train, result["adapter"], name)
    print(f"{name} now points at {moved[:12]}, and {digest[:12]} is still there")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
