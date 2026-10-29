# Roadmap to 3.0

> letify ships letify-link in steps as 2.x releases and becomes 3.0 when letify-link replaces the Python transport and Tailcat is gone.

This file is planning. The design is in [SPEC.md, letify-link](SPEC.md#letify-link). All crates and the Python package keep one version.

| Milestone | Content | State |
|---|---|---|
| M1 | Multi-socket TCP punch with spare sockets | Done in 2.10.0 |
| M2 | Session layer and path manager | Open |
| M3 | NAT type detection, port spraying, policing detection, Tailcat replacement | Open |
| M4 | The `letify-link` crate and sidecar wired into letify | Open, ships as 3.0 |

Each milestone is judged by its own experiment pull request. A spec change that the verdict rejects does not reach `develop`.
