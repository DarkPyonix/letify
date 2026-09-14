# Network

> How each provider is reached, what that path costs, and why the tunnel choices are what they are. Measurements are dated, because these numbers move.

All latency figures below assume a client in South Korea. Substitute your own round trip; every conclusion here is a function of it.

## Why latency decides the design

One number governs everything: `efficiency = T / (T + k * RTT)`, where `T` is GPU time per step and `k` is the number of points in that step where the host has to read a value back from the device.

This is why letify ships whole loops. Inside a shipped loop, those reads are local to the remote process and `k * RTT` disappears. Outside one, every read is a network round trip.

It is also why a faster GPU makes call forwarding worse. `T` shrinks and `RTT` does not.

| Round trip | LoRA step, 0.5 s, k=3 | Decode step, 3 ms, k=1 |
|---|---|---|
| 20 ms | 89 percent | about 12 percent, 43 tokens per second |
| 150 ms | 53 percent | about 2 percent, 6.5 tokens per second |
| 450 ms | 27 percent | under 1 percent, 2 tokens per second |

Decode throughput under forwarding is bounded near `1000 / (k * RTT)` tokens per second. At a 150 ms round trip an L4 and an RTX PRO 6000 both land near 5 to 6 tokens per second, so the card stops mattering. A generation loop that needs real speed has to be shipped, not forwarded.

## Colab

> Reached by the official Colab CLI. No tunnel, no terms risk.

The CLI creates and destroys sessions with `colab new` and `colab stop`, runs commands with `colab exec`, and offers `colab ssh`, which opens a shell over a WebSocket and can act as an OpenSSH `ProxyCommand` bridge with `--proxy-mode`. A background daemon keeps the runtime from idling out without a browser tab open, and authentication uses Application Default Credentials, so the whole path automates without a prompt.

**Terms.** Colab's FAQ disallows remote control such as SSH shells and remote desktops on free runtimes. Those restrictions are lifted on a paid plan while the compute unit balance is positive, and an exhausted balance reverts the account to the free tier policy. Accelerators themselves require a Pro or Pro plus entitlement. Session length is capped at 12 hours on the free tier and 24 hours on Pro plus.

**Storage does not survive.** Changing the accelerator type produces a new virtual machine with an empty disk, which was checked by writing a marker file and looking for it after the switch. This is why `Colab` is ephemeral and why a volume matters so much there.

**No fast path.** The control path crosses a Google frontend, so the round trip from Korea is on the order of 150 ms to 200 ms. `Colab.has_fast_path` is therefore false and asking for `host="local"` raises.

**Open question.** Whether `ssh -L` port forwarding works over `colab ssh --proxy-mode` is not documented. Since the bridge hands off to a standard OpenSSH client, forwarding should follow, but it has not been verified. A verified answer would give letify a data channel independent of `colab exec`.

## Shell

> A direct SSH address is the best path and the first one tried.

Order of preference is a direct address, then a jump host with `-J`, then a tunnel. University machines frequently allow one of the first two, and each step avoids a whole class of failure.

Where the machine hands out a fresh port on each restart, the configuration can set `port_command` to a command that prints the current port, which is read at connection time.

This is also the only provider family where call forwarding is worth offering. Root access allows a kernel mode tunnel, the path is layer 3 so raw sockets work, and the round trip on a domestic or campus link is small enough for the efficiency formula to come out well.

## Tunnel

> For a machine behind NAT that cannot accept an inbound connection. Tailscale by default, frp when UDP is blocked.

### Why Tailscale is the default

It needs no server of your own. It authenticates from an auth key with no prompt, which is what makes it scriptable. It is a layer 3 tunnel, so any TCP port works without declaring it. And when UDP is blocked it relays over TCP 443 instead of failing.

Costs to know about. The free personal plan covers 6 users with unlimited user devices, but only 1,000 ephemeral resource minutes per month, which is about 16 hours. Bringing up a Colab node on every session runs into that. The seventh user converts the whole tailnet to paid and bills every seat.

### Why frp is the fallback

Tailscale's relay fallback is a performance trap rather than a failure. Throughput on a relayed path has been measured as low as 2.2 Mbit/s across continents, where the direct path expected 30 to 40 Mbit/s, and Tailscale states that its relay servers limit throughput for fairness. A relayed path also adds 5 ms to 30 ms.

frp runs over TLS on port 443, supports arbitrary TCP as its main purpose, and is the easiest of the candidates to self-host. It needs a relay server with a public address, typically a small virtual machine at 4 to 6 USD per month, and relayed traffic counts against that machine's bandwidth twice.

### Candidates considered and rejected

| Candidate | Why not |
|---|---|
| ngrok | 1 GB per month free, then 0.10 USD per GB. Uneconomical for dataset transfer. |
| Cloudflare Tunnel | No peer to peer path, requires the client side to install cloudflared too, and the documentation warns that persistent connections may close unexpectedly. |
| ZeroTier | Always userspace crypto, so 200 to 400 Mbit/s at full CPU. Free tier narrowed to 10 devices and one network. Its Python binding has been unmaintained since 2022 and ships no wheel for current Python. |
| Nebula | No TCP fallback at all, so a network that blocks UDP blocks it entirely. Needs a lighthouse with a public address. Managed Nebula is free to 100 hosts, which is why it stays on the list as an alternative. |
| Raw WireGuard | No NAT traversal of its own, so it needs a relay server and a small control plane written by hand. No TCP fallback. |
| Headscale | A self-hosted Tailscale control plane. Removes the ephemeral minute limit, at the cost of a public server, TLS, and running your own relay, which Tailscale's own documentation calls an advanced operation. Worth it only if the free plan limit is actually reached. |

### MTU

Hold it at 1280 to 1400. Every mesh VPN in this class shows the same failure mode above that: the tunnel comes up, interactive commands work, and bulk transfers stall silently because a path MTU discovery black hole swallows large segments. WireGuard defaults to 1420, ZeroTier to 2800, and Nebula to 1300 precisely because of this. 1280 is the IPv6 minimum and always works.

### Why no tunnel helps Colab

Every candidate needs `/dev/net/tun` for a full layer 3 tunnel, and a Colab runtime does not have it. Tailscale's own issue tracker carries a report of running there that fails with `is CONFIG_TUN enabled in your kernel?`, and both Tailscale and ZeroTier documentation list Colab-style containers among the environments without the device.

Without the device, each candidate falls back to a userspace mode that is a SOCKS5 proxy rather than a network interface. Arbitrary TCP still works through the proxy, but nothing is captured transparently, no raw sockets or ICMP exist, and the throughput cost is severe: one report measured 902 Kbit/s through Tailscale's SOCKS5 userspace mode where the direct link did 83.9 Mbit/s, and Nebula's own project measures its userspace stack at roughly a quarter of kernel throughput.

The official CLI makes all of this unnecessary, which is why `Colab` does not use a tunnel.

## Elice

> Allocated through the Elice Cloud Infrastructure REST API, then reached over SSH.

The API base is `https://portal.elice.cloud/api` with a bearer token. Paths are published in Elice's own Terraform provider, which is open source, so this is a documented interface rather than a reverse engineered one.

The model separates a declared machine from a running one:

```
POST   /user/resource/compute/virtual_machine             declare a machine
POST   /user/resource/compute/virtual_machine_allocation  power on
DELETE /user/resource/compute/virtual_machine_allocation/{id}  power off
```

An allocation is exactly a letify session, so a session's span, one call or one `keep_alive` block, maps onto Elice's. letify allocates and releases, and does not create the machine; declare that once in the console or with Terraform and put its id in the configuration.

Costs. Compute bills by the second while allocated. Block storage keeps billing while the machine is stopped, and it disappears when the machine is deleted, so a forgotten machine still costs money with no allocation running. letify keeps the blob store on the machine's own disk there, with the `filesystem` backend.

Two product lines exist and only one is automatable. Elice Cloud Infrastructure has the API, the Terraform provider and a CLI. Run Box, the container product, is driven from the web console, and its SSH access is a tunnel host with an allocated port rather than a public address. letify's `Elice` provider targets Elice Cloud Infrastructure; a Run Box machine can still be used by declaring it as a plain `Shell` with the tunnel address and port.

**Open question.** Whether the SSH port is stable across a restart is not documented. If it is not, use `port_command` in the configuration.

## Modal

> Not a network path at all. Modal exposes function calls into a container.

There is nothing to tunnel to and no device to forward calls at, so `Modal.has_fast_path` is false and `host="local"` raises. Its volume is mounted from outside the container and sits in the same data centre as the GPU, which is why a persistent provider needs no separate cache tier.

The local path is a pipe to the Modal adapter, a process letify starts with `uv run --no-project --with "modal>=1.0,<2"`. Modal's client inside it reaches Modal's API over HTTPS. Every worker request crosses that adapter twice, once as a `write` and once as a `read_until`, as spec "Modal adapter" describes. The round trip of that path has not been measured.

## Call protocol throughput <!-- id: call-protocol-throughput -->

> Binary frames move a large argument or result at the link's speed over direct SSH, and at 500 to 860 MiB/s through a local pipe. Measured on 2026-09-14.

Base64 line framing cost more than the link. On the department server, an Intel Xeon E5-2650 v4, decoding base64 took about 2.3 s of CPU per 64 MiB, against 1.1 s on the wire. The frames of spec [Frames](SPEC.md#frames) carry pickle protocol 5 out-of-band buffers with no encoding, fill a received `bytes` value in place, and use 1 MiB pipes.

Local pipe, client and `Local` worker on one Linux machine, Python 3.13, best of two:

| Measurement | Base64 lines | Binary frames |
|---|---|---|
| Empty call round trip, median | 0.74 ms | 0.93 ms |
| 64 MiB argument | 23 MiB/s | 488 MiB/s |
| 64 MiB result | 46 MiB/s | 453 MiB/s |
| 512 MiB argument | 27 MiB/s | 859 MiB/s |
| 512 MiB result | 46 MiB/s | 525 MiB/s |
| Worker peak memory, 512 MiB argument | 3699 MiB | 1691 MiB |
| Call writing 1 MiB to stderr | hangs | completes |

`dept_gpu`, a `Shell` account over direct SSH from a university network in Daejeon to a server with a Tesla P100, one reused session. The raw SSH figures pipe the same sizes through `cat` with no letify involved, measured the same hour:

| Measurement | Base64 lines | Binary frames | Raw SSH |
|---|---|---|---|
| Command round trip | | | 48 ms |
| Empty call round trip, median | 1.1 ms | 1.2 ms | |
| 64 MiB argument, first send | 14.8 MiB/s | 70.4 MiB/s | 82 to 90 MiB/s |
| 64 MiB argument, repeated | | from the blob table, under 1 ms | |
| 64 MiB result, `os.urandom` on the server included | 34.0 MiB/s | 75.4 MiB/s | 93 to 94 MiB/s |
| 256 MiB, steady | | | 106 MiB/s up, 90 to 94 MiB/s down |

A first send of a 64 MiB argument costs three round trips on top of the bytes: `have`, `put_blob` and the call. At 35 to 48 ms each that is most of the gap to raw SSH.

`lab_docker`, a Tunnel account reached over Tailcat, whose link measured 69 ms and about 2 MiB/s when this was written:

| Measurement | Base64 lines | Raw SSH through Tailcat |
|---|---|---|
| Session start, environment build included | 42.65 s | |
| Command round trip over a reused connection | | 436 ms |
| Empty call round trip, median | 70.0 ms | |
| 64 MiB argument | 1.3 MiB/s | 2.01 MiB/s |
| 64 MiB result | 1.9 MiB/s | 2.59 MiB/s |

Base64 lines reached 65% of raw SSH on the argument here, where the link rather than the CPU is the limit, and the 1.78 times inflation of base64 accounts for the rest. Binary frames on `lab_docker` were not measured: its one card was running someone else's training throughout, and letify does not start a session on a busy card.

## Connection pipeline measurements

> The evidence for the strategy order and the rules in the spec's Transport section. Measured on 2026-09-13.

Setup: the client is a Linux Docker container on a university network in Daejeon, Korea, or a Windows 11 desktop on home Wi-Fi in Korea. The Colab VMs were CPU sessions; their region changed per session (Taiwan, Iowa, South Carolina, California, Oregon). Transfers were verified by SHA-256 where a size is given.

### Colab limits outbound UDP

Raw UDP, numbered 1200 byte packets sent at a fixed rate for 5 s, NAT punched by hand through STUN:

| Path | 1 MiB/s sent | 5 MiB/s sent | 20 MiB/s sent |
|---|---|---|---|
| Colab to Daejeon server | 788 of 4369 arrived | 804 of 21845 | 804 of 87381 |
| Daejeon server to Colab | 4372 of 4372 | 21881 of 21881 | 31970 of 87600 |
| Colab session A to session B | 1071 of 4369 | 1002 of 21863 | 988 of 87527 |
| Colab session B to session A | 1071 of 4369 | 1004 of 21863 | 1078 of 87527 |

About 1000 packets arrive per 5 s whatever the send rate, and the same holds between two Colab sessions. The limit is on Colab's outbound UDP, about 200 packets per second or 0.23 MiB/s. Inbound UDP to Colab is not limited at 5 MiB/s. Tailcat over the same path gave 0.59 MiB/s up and 0.10 MiB/s down, and pacing the sender at 2 MiB/s did not help, which rules out buffer overflow as the cause.

### Tailcat between two ordinary NATs

Windows desktop on home Wi-Fi to the Daejeon server, both behind NAT, no Colab:

| Round trip median | Upload | Download at 1 / 5 / 20 MiB/s | Download unpaced | Path |
|---|---|---|---|---|
| 3.2 ms | 24.31 MiB/s | 1.00 / 5.00 / 19.92 MiB/s | 32.03 MiB/s | direct for the whole run |

### TCP hole punching with Colab

Both sides bound one TCP port, learned the mapping from `stun.nextcloud.com:443`, and connected to each other at an agreed time while listening on the same port. Both NATs preserved the local port number. STUN servers on ports 3478 and 19302 timed out from the Daejeon network.

| Result | Round trip median | Upload, 1 connection | Download, 1 connection |
|---|---|---|---|
| connected on the first attempt, both sides | 180.5 ms (VM in the United States) | 12.97 MiB/s | 14.01 MiB/s |

### Colab's own paths

| Path | Upload | Download |
|---|---|---|
| `colab exec` round trip, `print(1)` | 1.7 s to 3.7 s | |
| `colab upload` / `colab download`, 32 MiB | 3.86 MiB/s | 7.35 MiB/s |
| `colab upload` / `colab download`, 256 MiB | fails: the whole file is one base64 request | 9.60 MiB/s |
| contents API direct, 256 MiB, 1 part | 2.65 MiB/s | 14.77 MiB/s |
| contents API direct, 256 MiB, 4 parallel parts | 9.74 MiB/s | 46.98 MiB/s |
| contents API direct, 256 MiB, 8 parallel parts | 17.77 MiB/s | 64.13 MiB/s |

`colab console` is not a usable channel: it is a terminal inside tmux, and a single input line of 4000 characters or more arrives truncated or mixed with terminal control sequences. The Colab runtime proxy reaches only port 8080 on the VM; other port prefixes return 404.

### letify-core copy throughput

A copy to the device through letify-core, with the payload streamed from the caller's buffer into the agent's staging buffer, moves about 2000 MiB/s over loopback TCP. The same payload through an encoded frame, the path before streaming, moves about 520 MiB/s. Both carry one 256 MiB `CopyToDevice` request between a `BufWriter` and a `BufReader` with `TCP_NODELAY`, on an AMD EPYC 7763, and report the median of five runs.

| Path | Three invocations |
|---|---|
| Encoded frame, before the change | 545, 544, 507 MiB/s |
| Encoded frame, after the change | 515, 514, 519 MiB/s |
| Streamed into staging | 2043, 1961, 1873 MiB/s |

A copy to the host, with the payload streamed from the agent's staging buffer into the caller's buffer, moves about 3250 MiB/s over loopback TCP. The same payload through an encoded frame and a copy into the caller's buffer, the path before streaming, moves about 520 MiB/s. Both carry one 256 MiB `Payload` reply over the same socket pair and machine and report the median of five runs. The copy on the device is not included.

| Path | Three invocations |
|---|---|
| Encoded frame, before the change | 522, 514, 519 MiB/s |
| Encoded frame, after the change | 528, 527, 540 MiB/s |
| Streamed into the destination | 3234, 3261, 3244 MiB/s |

Loopback removes the network, so these numbers bound what the copy path itself costs. On a real link the link rate decides throughput whenever it is below them. To reproduce, run `cargo test --release -p letify-wire --test throughput -- --ignored --nocapture --test-threads=1` in `letify-core/`.

## Measuring your own numbers

Three checks settle most of what is provider specific.

```bash
# Round trip to the machine, which sets the efficiency formula.
letify probe gpu.lab.example.edu

# Whether a layer 3 tunnel is possible at all.
ls -l /dev/net/tun

# Whether UDP egress is allowed, which decides direct path versus relay.
nc -zvu stun.l.google.com 19302
```

For the synchronization count `k`, run one real training step with `torch.cuda.set_sync_debug_mode("warn")` and count the warnings. That number, together with the round trip, is enough to decide whether call forwarding is worth using for a given workload, and it can be measured on any CUDA GPU because it does not depend on where the GPU is.
