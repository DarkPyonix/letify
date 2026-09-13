"""What every script in this scenario shares: the launcher, the alias and the environment.

Kept in one file so each numbered script is only the step it demonstrates. This module is
not shipped; it runs here and decides where the other code runs.
"""

from __future__ import annotations

import argparse

import letify

#: The environment for every declaration in this scenario. The lock file resolves for every
#: platform uv supports, so this same declaration drives a Linux runtime from Windows.
#: ``ship`` sends the training code by value, because the machine installs what the lock
#: file names and has no copy of this project.
ENV = letify.Env().ship("recipe")

#: Where checkpoints and environment archives live. Named rather than declared per script,
#: so two runs against the same provider share one cache.
VOLUME_NAME = "nvfp4-lora"


def parser(description: str) -> argparse.ArgumentParser:
    """The arguments every script takes, so the provider is chosen at the command line."""
    argue = argparse.ArgumentParser(description=description)
    argue.add_argument(
        "--provider",
        default="local",
        help="alias from .letify; 'local' needs no account and is the place to start",
    )
    argue.add_argument(
        "--device",
        default=None,
        help="accelerator name on that provider, such as G4 or A100; defaults to the first one",
    )
    argue.add_argument(
        "--host",
        default="remote",
        choices=["remote", "local"],
        help="'remote' ships the function to the machine, 'local' keeps Python here and "
        "forwards CUDA calls",
    )
    return argue


def pick(let: letify.Launcher, alias: str, name: str | None) -> letify.Instance:
    """Resolve the instance to run on, saying what was available when the name is wrong.

    An accelerator list comes from the provider rather than from this file, because what an
    account can offer is the provider's answer and it changes.
    """
    provider = let.provider(alias)
    offered = provider.instances
    if name:
        if name not in offered:
            raise SystemExit(
                f"{alias} does not offer {name!r}. It offers: {', '.join(sorted(offered))}"
            )
        return offered[name]

    accelerators = [key for key, instance in offered.items() if instance.gpu or instance.tpu]
    if not accelerators:
        # The local provider always registers a CPU shape, which is enough to run the
        # scenario end to end without a card.
        return offered["CPU"]
    return offered[sorted(accelerators)[0]]


def describe(instance: letify.Instance) -> str:
    vram = f", {instance.vram_gb} GiB" if instance.vram_gb else ""
    return f"{instance.provider.alias}.{instance.accelerator}{vram}"
