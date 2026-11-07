# 9️⃣ The letify agent

> Make a machine with no public address reachable, with no SSH server and no relay run by letify.

[← Releasing](08-releasing.md) · [Guides](README.md)

---

## What it is for

`letify agent` is for a machine that cannot accept an inbound connection, such as a lab server behind NAT. You run `letify agent serve` on that machine and pair your own machine with it once. After that, your machine reaches it through the signalling service, a small web service that passes signed messages between the two.

The agent needs no SSH server and no Tailcat. letify runs no relay for it.

The terms used below:

- **Agent**: the long-running `letify agent serve` process on the address-less machine.
- **Signalling service**: the web service both sides talk to. The default is `https://letify.darkpyonix.dev`.
- **Pairing**: the one-time step that makes the agent accept your machine.
- **Node id**: the 64 hex character name of a machine's key. Each machine has one.
- **Token**: the one-line string `letify agent serve` prints for pairing.

---

## Before you start

- letify is installed on the agent machine (`uv add letify`).
- The `letify-link` binary is present. It ships in the platform wheels. When it is missing, the command fails with `letify-link is not in this wheel`. The environment variable `LETIFY_LINK_BIN` can name another executable.
- `nvidia-smi` is on the agent's `PATH`. letify reads the cards through it. On Kaggle it is in `/opt/bin`, so start the agent with `PATH=/opt/bin:$PATH letify agent serve`.
- The agent machine can open an outbound `https://` connection to the signalling service.
- The signalling service URL is `https://letify.darkpyonix.dev` unless you set another. In order of priority:
  1. `letify agent serve --signal <URL>`
  2. the environment variable `LETIFY_SIGNAL_URL`
  3. `signal` in `~/.letify/agent/config.toml`
  4. the default

  The URL must be `https://`. A plain `http://` URL is accepted only for a loopback host.

Your own machine needs letify installed. The token carries the service URL, so you do not set it there.

---

## 1. Start the agent

On the agent machine:

```bash
letify agent serve --name lab_box
```

It prints a login command to run on your own machine:

```
On your own machine, within 15 minutes, run:

  letify login tunnel lab_box --connect letify-agent2:eyJ2Ijoy...
```

`--name` sets the alias in that command. Without it letify picks a default alias.

The token in that command:

| Property | Value |
|---|---|
| Format | `letify-agent2:` followed by URL-safe base64 |
| Uses | one |
| Lifetime | 15 minutes |
| After 5 wrong proofs | discarded |
| After the agent restarts | invalid, because the secret lives in the running agent's memory |

Each run of `letify agent serve` prints a new token. The list of paired machines survives restarts.

The agent keeps running and logs one line per event to standard error. Keep it alive in `tmux` or another session that survives logout:

```bash
tmux new -s letify 'letify agent serve'
```

---

## 2. Pair your own machine

On your own machine, within 15 minutes, run the printed command:

```bash
letify login tunnel lab_box --connect letify-agent2:eyJ2Ijoy...
```

letify creates this account's key `~/.letify/accounts/lab_box/link.key`, sends a proof to the agent through the signalling service and waits up to 30 s for the answer. The proof is an HMAC computed from the token's secret and both node ids, so the secret never leaves your machine, and a proof made for one agent is useless to another.

On success the account is written with `kind = "tunnel"`, the agent's node id and the service URL. The agent logs `paired <12 hex of node> (<label>)`, where the label is `<user>@<hostname>` of your machine.

The login then opens one session to the agent, checks the workspace root and records the GPUs over it, so no SSH key and no SSH server are involved. A token that does not start with `letify-agent2:` takes the older Tailcat path of `letify client shell connect`. That path still works until 3.0, prints a notice that Tailcat is removed then, and is covered in [Tunnel](02-providers.md).

To reach the machine's SSH server anyway, call `Tunnel.ssh_forward()` on the account. It returns a loopback port, and `ssh -p <port> <user>@127.0.0.1` reaches the sshd that `letify agent serve --ssh-target HOST:PORT` names (default `127.0.0.1:22`). letify itself never needs it.

---

## 3. Where the state lives

On the agent machine, in `~/.letify/agent/` (under `LETIFY_HOME` when that is set), created with mode 0700:

| File | Holds |
|---|---|
| `link.key` | The agent's node key, mode 0600. Created on first run. The agent refuses it when group or others can read it |
| `paired.json` | The paired machines: node id, label and time. Mode 0600 |
| `config.toml` | Optional. `signal = "<URL>"` |

---

## 4. See and remove paired machines

```bash
letify agent paired
```

One line per paired machine: node id, date and label. It prints `no paired nodes` when there are none.

```bash
letify agent unpair <node>
```

`<node>` is the full node id or a unique prefix of at least 8 characters. A running agent notices the change within 5 s, ends that machine's conversation, even one that is running, and ignores its later messages. It does not need a restart.

To pair the machine again, run `letify agent serve` for a new token.

---

## Limits

- At most 4 machines may have a conversation running at once. Change it with `--max-conversations`. A paired machine over the cap gets `the agent is busy: <n> conversations`.
- Each paired machine has at most one conversation. A new one from the same machine ends the previous one, unless that one has already handed over to a running worker, which is left to finish.
- A message is at most 4096 bytes, and the signalling service keeps an undelivered message for at most 60 s.
- A message from a machine that is not paired gets no answer and starts nothing.

---

## What goes over the signalling service

Only signed bootstrap lines go over it: node ids, candidate addresses, NAT types, the letify version and the pairing proof. Nothing of your workload does. The session that follows is a separate encrypted connection between the two machines.

The service is not trusted for content. It can read, delay or drop messages, so the receiver verifies every message and drops one that is unsigned, stale, replayed or addressed to another node. The lines are signed, not encrypted. They hold no secret, token or key.

The agent never runs a command that a message supplies. It uses the programs of its own installed letify.

While idle the agent waits on one WebSocket to the service, not on repeated polling. If the socket drops, it reconnects after 1 s, doubling after each failure in a row up to 60 s.

---

## Stopping the agent

SIGTERM or Ctrl-C stops it. It ends every conversation that has not handed over to a running worker and exits with status 0.

---

## Troubleshooting

| Message | Cause and fix |
|---|---|
| `letify-link is not in this wheel` | The sidecar binary is missing. Install a platform wheel, or set `LETIFY_LINK_BIN` |
| `the signalling URL '...' must be https, or http for a loopback host, not ...` | Fix `--signal`, `LETIFY_SIGNAL_URL` or `signal` in `config.toml` |
| `that is not a token: it should start with letify-agent2: ...` | You pasted something else. Use the token `letify agent serve` printed |
| `this is a Tailcat token from 'letify client shell connect'; this login needs ...` | Wrong token. Use the one `letify agent serve` prints |
| `the token expired; run 'letify agent serve' again for a new one` | The 15 minutes passed. Run it again |
| `the pairing secret is wrong, used or expired` | One text for three causes: a typo, a token already used, or one past 15 minutes. Get a new token. After 5 refusals to your node in 15 minutes the agent stops answering |
| `the agent <12 hex> did not answer in 30 s` | The agent is not running, was restarted (its tokens are void), or cannot reach the service |
| `tunnel login failed at pairing: ...` / `at token: ...` / `at sidecar: ...` | The step that failed, then the reason |
| `the signalling service did not answer: ...` | The service or the network is down. The agent retries by itself |
| `no paired node starts with <text>` | Check the prefix against `letify agent paired` |
| `<n> paired nodes start with <text>; give more of it` | The prefix is not unique. Give more characters |
| `give a node id or a prefix of at least 8 hex characters` | `unpair` needs 8 or more |
| `cannot read <path>: ...` | `paired.json` is unreadable. Check its owner and mode |

If the agent log shows `refused pairing from <12 hex>`, a pairing attempt failed. Nothing there holds a secret, proof or token.

---

[← Releasing](08-releasing.md) · [Guides](README.md)
