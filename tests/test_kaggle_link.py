"""An SSH link over the Kaggle kernel bridge.

Spec "Kaggle runtimes", An SSH link over the kernel. The bridge is the rendezvous, and
the strategies of Choosing a link are raced over it.
"""

from __future__ import annotations

from conftest import provider_of

from letify.providers.kaggle import Kaggle


def kaggle(**options):
    return provider_of(Kaggle, "kaggle_a", **options)


def test_kaggle_races_the_strategies_that_need_no_address() -> None:
    # Spec: forward SSH is skipped because a Kaggle session has no address to dial.
    names = [s.name for s in kaggle().strategies()]
    assert "direct_ssh" not in names
    assert "tcp_punch" in names
    assert "tailcat" in names
    assert names[-1] == "fallback" or "fallback" not in names


def test_the_rendezvous_runs_python_through_the_bridge() -> None:
    # Spec: the bridge runs one standard library program on the machine, which is exactly
    # what a rendezvous is.
    from letify.transport.rendezvous import CommandRendezvous

    asked: list[str] = []

    class Channel:
        def request(self, message, **kwargs):
            asked.append(message["op"])
            assert message["op"] == "eval"
            return ("LETIFY-ANSWER {}", None)

    provider = kaggle()
    rendezvous = provider.rendezvous_over(Channel())
    assert isinstance(rendezvous, CommandRendezvous)
    assert rendezvous.run_python("print('x')", 30.0) == "LETIFY-ANSWER {}"
    assert asked == ["eval"]


def test_the_target_names_no_address_and_carries_the_rendezvous() -> None:
    provider = kaggle()

    class Channel:
        def request(self, message, **kwargs):
            return ("", None)

    target = provider.target_over(Channel())
    assert target.address is None
    assert target.direct_ssh is None
    assert target.rendezvous is not None
    assert target.alias == "kaggle_a"


def test_a_session_with_no_link_keeps_the_bridge(monkeypatch) -> None:
    # Spec: when no link is chosen the bridge is the channel, as every Kaggle session did
    # until now, so a network that cannot punch is no worse off than before.
    from letify.errors import ProviderUnavailable

    provider = kaggle()

    class Channel:
        closed = False

        def request(self, message, **kwargs):
            return ("", None)

        def close(self):
            self.closed = True

    bridge = Channel()
    def nothing(channel):
        raise ProviderUnavailable("kaggle", "nothing connected")

    monkeypatch.setattr(provider, "link_over", nothing)
    assert provider.channel_over(bridge, name="k-1") is bridge
    assert not bridge.closed


def test_a_chosen_link_becomes_the_channel(monkeypatch) -> None:
    # Spec: when a link is chosen the session's channel is a worker started over it.
    provider = kaggle()
    built: list[list[str]] = []

    class Channel:
        def request(self, message, **kwargs):
            return ("", None)

        def close(self):
            pass

    class Link:
        strategy = "tcp_punch"
        rtt_ms = 120.0

        def ssh_command(self, remote_command=None):
            return ["ssh", "-p", "40000", "root@127.0.0.1", remote_command or ""]

    class Fake:
        def __init__(self, command, name=None, **kwargs):
            built.append(command)
            self.name = name

    monkeypatch.setattr(provider, "link_over", lambda channel: Link())
    monkeypatch.setattr("letify.runtime.channel.PersistentChannel", Fake)
    channel = provider.channel_over(Channel(), name="k-1")
    assert isinstance(channel, Fake)
    assert built and built[0][0] == "ssh"
