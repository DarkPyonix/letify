# Roadmap to 3.0

> letify ships letify-link in steps as 2.x releases and becomes 3.0 when letify-link is the default and Tailcat is gone.

This file is planning. The design is in [SPEC.md, letify-link](SPEC.md#letify-link). All crates and the Python package keep one version.

## Plan

| Release | Content |
|---|---|
| 2.11 | letify-link ships as an opt-in feature. The switch is the account option `link = "letify-link"`. With it off, the connection path is that of 2.10.x |
| 2.12 | The signalling Worker is deployed, the NAT-to-NAT connect is measured live, and the tunnel moves onto letify-link |
| 2.13 | letify-link is the default connection layer. `link = "ssh"` is the explicit opt-out. Tailcat, QUIC and the ssh data path stay |
| 3.0 | Tailcat is removed, with `letify setup tailcat`, `letify client shell connect` and the Tunnel account's `tailcat` fields. An account that names one is refused by name with the migration to `letify agent serve`. Done on `feat/remove-tailcat`. QUIC over a punched UDP pair is removed, with the `letify-quic` crate and binary, its strategy (rank 4) and its Python module, and letify-link carries UDP paths. Done on `feat/remove-quic`. The ssh data path is removed in a later step |

## Milestones

| Milestone | Content | State |
|---|---|---|
| M1 | Multi-socket TCP punch with spare sockets | Done in 2.10.0 |
| M2 | Session layer and path manager, with the hysteresis of the path manager and the head-of-line blocking measurement of many streams over TCP subflows against QUIC streams | Built on `feat/letify-link-spec`: the encrypted multiplexed session, TCP and UDP subflows and the path manager are in the `letify-link` crate. The head-of-line measurement is open |
| M3 | NAT type detection, port spraying, policing detection, Tailcat replacement | Built on the branch: NAT classification, the punch, the signalling client and the Cloudflare Worker in `letify-signal/`, and the agent with pairing. Measured only against fakes until `feat/link-signal-live`, which connects a tunnel account to a paired agent through the signalling service alone and measures it live on Kaggle and Colab (all paths were TCP punches, 4 of 4 in every run). The tunnel over letify-link (B4) is done in 2.12: a Tunnel account logs in through `letify agent serve`, the login runs over the session, and the machine's sshd is reachable through a session stream ([#84](https://github.com/DarkPyonix/letify/pull/84), [#85](https://github.com/DarkPyonix/letify/pull/85)). A Tailcat account keeps working and is told it is removed in 3.0. Remaining: UDP punch to a cloud VM |
| M4 | The `letify-link` crate and sidecar wired into letify | Built on the branch as opt-in: the sidecar, the shell, Colab and Kaggle bootstrap, transfer streams, takeover and helper commands over the session. Ships in 2.11 behind `link = "letify-link"`. Default and Tailcat removal are 3.0 |

Each milestone is judged by its own experiment pull request. A spec change that the verdict rejects does not reach `develop`.
