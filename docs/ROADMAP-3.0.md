# Roadmap to 3.0

> letify ships letify-link in steps as 2.x releases and becomes 3.0 when letify-link is the default and Tailcat is gone.

This file is planning. The design is in [SPEC.md, letify-link](SPEC.md#letify-link). All crates and the Python package keep one version.

## Plan

| Release | Content |
|---|---|
| 2.11 | letify-link ships as an opt-in feature. The switch is the account option `link = "letify-link"`. With it off, the connection path is that of 2.10.x |
| 2.12 | The signalling Worker is deployed, the NAT-to-NAT connect is measured live, and the tunnel moves onto letify-link |
| 3.0 | letify-link is the default connection layer and Tailcat is removed, with `letify setup tailcat` and the Tunnel account's `tailcat` fields |

## Milestones

| Milestone | Content | State |
|---|---|---|
| M1 | Multi-socket TCP punch with spare sockets | Done in 2.10.0 |
| M2 | Session layer and path manager, with the hysteresis of the path manager and the head-of-line blocking measurement of many streams over TCP subflows against QUIC streams | Built on `feat/letify-link-spec`: the encrypted multiplexed session, TCP and UDP subflows and the path manager are in the `letify-link` crate. The head-of-line measurement is open |
| M3 | NAT type detection, port spraying, policing detection, Tailcat replacement | Built on the branch: NAT classification, the punch, the signalling client and the Cloudflare Worker in `letify-signal/`, and the agent with pairing. Measured only against fakes. The live NAT-to-NAT connect is open and is the work of 2.12. Tailcat is untouched |
| M4 | The `letify-link` crate and sidecar wired into letify | Built on the branch as opt-in: the sidecar, the shell, Colab and Kaggle bootstrap, transfer streams, takeover and helper commands over the session. Ships in 2.11 behind `link = "letify-link"`. Default and Tailcat removal are 3.0 |

Each milestone is judged by its own experiment pull request. A spec change that the verdict rejects does not reach `develop`.
