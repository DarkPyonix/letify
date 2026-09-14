# Specification

> The current design of letify. Decisions only. Derivations and measurements live in the experiment pull requests, which each section links to when one exists.
>
> This file is the source of truth. Code follows it, and every test traces to a section here. See "How this project is built" in [CLAUDE.md](../CLAUDE.md).

## Declaration surface

> A declaration places three things and names no mechanism.

The public surface is four names: `Launcher`, `Env`, the `function` decorator it carries, and `session_cache`, which a declared body uses to keep a value for the length of its session. Everything else is reached through a provider object.

```python
import letify

let = letify.Launcher()
env = letify.Env()

colab = let.providers.colab_a

@let.function(device=colab.G4, host="remote", env=env)
def train(lr, bs):
    ...

train(lr=1e-4, bs=32)
```

The decorator takes `device`, `host`, `env`, `timeout` and `retries`. Any other argument is refused by Python as unexpected. `timeout` has no default: a deadline letify invented would end a two hour training run at whatever hour it guessed, which is letify deciding how long the user's own work is allowed to take. It takes no transport, no mode and no width: the three placements below settle where the work runs, and how much can run at once is the provider's inventory rather than a number on the declaration.

### The two placements <!-- id: the-three-placements -->

> `device` says where the accelerator is and `host` says where the host code runs. Both are said in the declaration and nowhere else.

`device` carries the provider, the account, the accelerator and how many of it one session takes, because those are one decision. `colab.G4` is such a value, and so is `lab.A100 * 2` for a run that trains across two cards. Core count and memory are not arguments: they arrive with the shape the provider registered, and a provider that offers several sizes registers them as separate shapes.

`host` is the CUDA word for the CPU side, paired with the device the declaration already placed. `"local"`, the default, keeps Python and the libraries in this process and forwards only CUDA calls. `"remote"` ships the declared function to the machine that holds the device.

`host` takes `letify.local` or `letify.remote`, which are the two members of a string enum defined next to `Instance`. The enum class itself is not part of the public surface: two named values say everything a declaration needs, and a class at the top of the package was one more name to learn for the same two choices. Because the members are strings, `host="remote"` is the same value. An unrecognized value raises at declaration time with both options named.

An instance carries no placement of its own. `colab.G4` says which card, and only the declaration's `host` says where the host code runs, so there is one place to read to know where a function runs. How long a session lives is not a declaration argument either: it is the `keep_alive` block around the calls, described under Lifetime.

### Invocation

> Calling a declared function runs it. Blocking behaviour is declared at the `def` site.

A declared function is called like any other. There is no second verb such as `.remote()`: a decorator that wraps a `def` and then needs another call to run it has moved the declaration out of the declaration.

A plain `def` blocks and returns its value. An `async def` returns a plain coroutine, so the standard library accepts it wherever a coroutine is expected.

A call takes the arguments the `def` declares and returns what the `def` returns. No argument value changes what a call means.

`Function.local()` runs the body in the calling process. It exists for testing a body with no provider; the preferred way to run locally is a `Local` provider, which keeps the production code path.

### Concurrency <!-- id: fan-out -->

> Many configurations run at once by calling the declared function concurrently. The inventory bounds how many run.

Running several configurations is repeated calls, one per configuration, inside `with let.keep_alive():` so the calls reuse warm sessions. Concurrent calls come from the language: `asyncio.gather` over calls to an `async def` declaration, or threads over a plain `def` one. letify adds no map call and no argument type for it.

Concurrent calls run as wide as the provider has devices for, and no wider. Nothing on the declaration bounds it, because a bound there would be a second statement of the same fact: the inventory already says how many cards exist, and a call that cannot reserve one waits for a call that holds one to finish, as described under Pooling.

A declaration taking two cards halves the width on a four card machine, which is arithmetic rather than policy. That is also why a width knob could not work: with `device=lab.A100 * 2`, a number that says how many runtimes may exist says nothing about how many cards they need.

## Provider model

> A provider object is one account on one kind of infrastructure. It registers the instances it offers, owns storage, opens the channel, and starts and stops the session.

```
Provider (abstract)
├── Local                 persistent
├── Modal                 persistent, reached through the Modal adapter
└── Shell                 ephemeral by default, reached through the connection pipeline
    ├── Colab             session created by the Colab CLI, rendezvous over colab exec
    ├── Tunnel            a machine behind NAT, rendezvous through letify client shell connect
    └── Elice             machine allocated through the Elice Cloud API, rendezvous over SSH to it
```

`Shell` is named for the shared ability, which is running a command on a remote machine, rather than for SSH, which is only its default transport.

A provider is built from one entry in the configuration file and reached by attribute on `let.providers`. Three attribute names there are reserved: `any` for a request that does not name a provider, `devices` for the registered accelerators of every provider, and `active` for the providers that currently hold a runtime.

### Provider properties

> Four class attributes set every behaviour that differs between providers, and none of them is a user-facing switch.

`persistence` says whether storage outlives a runtime.

`has_fast_path` says whether the machine is close enough for CUDA call forwarding to pay off. It does not gate the mode: a declaration that asks for forwarding over a long link gets a warning carrying the arithmetic and then runs.

`persistent_channel` says whether a worker process can be kept alive behind a pipe.

`needs_lease` says whether a session can outlive this process and keep billing.

| Provider | Persistence | Fast path | Channel | Store backend |
|---|---|---|---|---|
| `Local` | persistent | yes | persistent | `filesystem` |
| `Modal` | persistent | no | persistent | `modal` |
| `Colab` | ephemeral | no | persistent, or one-shot by configuration | `gcs` |
| `Shell` | ephemeral, overridable | yes | persistent | `filesystem` |
| `Tunnel` | ephemeral, overridable | yes | persistent | `filesystem` |
| `Elice` | ephemeral | yes | persistent | `filesystem` |

`Shell` and its subclasses default to ephemeral because a machine's disk policy is not knowable in advance. Assuming ephemeral costs time, since letify rebuilds the environment each runtime and the work still succeeds; assuming persistent fails outright when the disk turns out to be wiped. A configuration entry overrides it with `persistent = true`.

### Remaining usage

> Every provider is asked what is left on its account, reads the answer from the service itself, and says why when it cannot.

`Provider.usage()` returns a `Usage` record with these fields:

| Field | Means |
|---|---|
| `alias`, `kind` | The account and its provider kind |
| `unit` | What the account is metered in: `compute units`, `KRW`, `USD` or `GPU hours` |
| `source` | Where the figure came from, or why there is none |
| `remaining` | How much is left, in `unit` |
| `used`, `limit` | How much of the current allowance is spent, and the allowance, when the service states them |
| `rate_per_hour` | What is running now costs per hour, in `unit` |
| `resets_at` | When the allowance renews, as Unix seconds, when it renews on a schedule |
| `unmetered` | True when there is no quota at all |
| `as_of` | When the figure was read, as Unix seconds |
| `note` | Why a figure is missing, in one line |
| `resources` | Further allowances on the same account, such as Kaggle's TPU hours beside its GPU hours. A list of records with `name`, `unit`, `remaining`, `used`, `limit` and `resets_at`, empty when there are none |

Every field except `alias`, `kind`, `unit` and `source` may be `None`. A provider that cannot be read returns a record with `remaining` set to `None` and the reason in `note`. It does not raise.

Each provider reads its own service. Every call is read-only: none creates a runtime, a sandbox or a session.

| Provider | Unit | Remaining comes from |
|---|---|---|
| `Colab` | compute units | `GET https://colab.research.google.com/tun/m/ccu-info?authuser=0`, the call the Colab web page makes. `remaining` is `currentBalance` and `rate_per_hour` is `consumptionRateHourly`. The OAuth token is the Colab CLI's `.config/colab-cli/token.json` in the account directory. An expired token is refreshed in memory at its `token_uri` and the file is not rewritten |
| `Elice` | KRW | `GET <billing_endpoint>/stats` with the account's bearer token and, when `organization` is set, the `x-elice-org-name-short` header. `remaining` is `total_credit_remaining_amount`, sent as a number or as `"<amount> <currency>"`. `rate_per_hour` prices the live allocations against the zone price list. `billing_endpoint` is the Elice billing API base URL. The public portal assets do not carry it, so it has no default: `letify login elice` records it from `--billing-endpoint` or the prompt, and without it `note` says so |
| `Modal` | USD | The Modal adapter op `billing_summary`, which calls `modal.Workspace.billing.summary()` for the current month. `used` is the month's metered cost and `limit` is the monthly credit, `monthly_credit` in the entry or 30 USD, the Starter plan credit. `remaining` is `limit - used`, not below 0. `resets_at` is 00:00 UTC on the first of next month |
| `Kaggle` | GPU hours | The weekly GPU quota the Kaggle provider reads. `remaining`, `used` and `limit` are hours, and `resets_at` is the weekly renewal |
| `Local` | hours | unmetered: this machine bills nobody |
| `Shell`, `Tunnel` | hours | unmetered: a machine reached by SSH has no quota and no account behind it |

A configured command replaces the provider's own reading:

```toml
[colab_a]
kind = "colab"
usage_command = "my-colab-units"   # prints the remaining amount
usage_unit = "compute units"
usage_limit = 100.0
```

The last number in the command's output is read as the remaining amount. The command runs only when usage is asked for, never during a call.

`usage_limit` without `usage_command` is the plan allowance for the provider's own reading. It fills `limit` only when the service did not state one, and `used` becomes `limit - remaining`, not below 0. Colab reports a balance and no allowance, so this is how a Colab row gets a percentage.

`Launcher.usage()` asks every provider at once, one thread each, and waits at most `usage_timeout` seconds per provider, 20 by default. A provider that has not answered by then, or that raised, gets a row with `remaining` set to `None` and the reason in `note`. The other rows are unaffected.

`letify usage` prints one block per declared provider, and `letify usage <alias>` one provider. A provider whose optional dependency or setting is missing is reported as unavailable rather than skipped, so the output always lists every alias. `letify usage --json` prints the records unformatted.

Amounts are formatted by unit:

| Unit | Printed as |
|---|---|
| `KRW` | `12,345 KRW`, whole won with thousands separators |
| `USD` | `$29.50` |
| `compute units` | `99.93 compute units`, two decimals |
| a unit ending in `hours`, such as `GPU hours` | `12.5 GPU hours`, one decimal |
| any other | the number as given, then the unit |

A block is a header line, `<alias>  <kind>`, followed by lines indented by 2 spaces:

```
colab_a  colab
  [████████████████░░░░░░░░░░░░░░░░░░░░░░░░] 40% used
  60.00 compute units left of 100.00
  1.96 compute units/hour running now, about 1 d 6 h at this rate

kaggle  kaggle
  [████████████████████████░░░░░░░░░░░░░░░░] 60% used
  12.0 GPU hours left of 30.0
  resets in 4 d 6 h (2026-09-18 12:00 UTC)
  TPU [████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░] 10% used
      18.0 TPU hours left of 20.0

lab  shell
  no quota, unmetered
```

| Line | Printed when | Content |
|---|---|---|
| Gauge | `limit` is known and above 0 | `[<filled><empty>] <p>% used`, where `p` is `used / limit` rounded to a whole percent, with `used` taken as `limit - remaining` when the service did not state it |
| Amount | `remaining` is known | `<remaining> left`, then ` of <limit>` when the limit is known. With no limit the line ends `, limit unknown` and no gauge or percentage is printed |
| Reset | `resets_at` is known | `resets in <relative> (<YYYY-MM-DD HH:MM UTC>)`, or `reset due (<date>)` once the time has passed |
| Rate | `rate_per_hour` is known | `<rate>/hour running now`, then `, about <relative> at this rate` when the rate is above 0 and `remaining` is known |
| No quota | `unmetered` is true | `no quota, unmetered`, and no other line |
| Not reported | neither `remaining` nor `rate_per_hour` is known | `not reported` |
| Note | `note` is set | the note |
| Unavailable | the provider could not be built | `unavailable: <reason>` |

A relative time uses the two largest units of days, hours and minutes, as `4 d 6 h`, `3 h 12 min` or `45 min`. Each record in `resources` prints its own gauge, amount and reset lines under the account's lines. Its gauge line starts with its `name` and its other lines are indented to the gauge's bracket.

The gauge is 16 to 40 cells wide. Its width is the terminal width from `shutil.get_terminal_size`, minus the 2 space indent, the 2 brackets and 10 columns for the percentage text, then clamped to that range. A further allowance's gauge is also shorter by its name and a space, with the same 16 cell minimum, so a narrow terminal still gets a 16 cell gauge. A filled cell is `█` and an empty cell is `░` when the standard output encoding is UTF-8. Otherwise they are `#` and `-`.

Colour is added only when standard output is a terminal and the `NO_COLOR` environment variable is unset or empty. The gauge and its percentage are green while more than 50% of the allowance remains, yellow from 20% to 50%, and red below 20%. The header's alias is bold. Without colour no escape sequence is written.

### Elice machines <!-- id: elice-machines -->

> letify finds, creates, starts and stops an Elice Cloud Infrastructure virtual machine through Elice's own `eci` command, so an account needs no machine made in the portal beforehand.

`eci` is the standalone binary Elice publishes at github.com/elice-dev/eci-cli for macOS arm64, Linux x86_64 and Windows x86_64. letify runs it as a separate process and never bundles or mirrors it. `eci` is found by the lookup [Installing external tools](#confirmed-tool-install) describes. When it is found nowhere, `letify login elice` asks `eci, Elice's command line, is not installed. letify can download eci <version> from Elice's GitHub release into ~/.letify/tools; it is Elice's software, not part of letify. Install it now? [y/N]: `. `y` or `yes` installs it as [Installing external tools](#confirmed-tool-install) describes and the login continues. Any other answer, and a login with `--no-input`, stops the login with the install command and `Install it with: letify setup eci`, and writes nothing. An Elice session start never asks and never installs; it raises `ProviderUnavailable` with the same text. `eci` is never installed without that answer or `letify setup eci`, because its release carries no license. When `eci_binary` names a program that is not on `PATH`, the error carries the install command for this system:

- macOS and Linux: `curl -fsSL https://raw.githubusercontent.com/elice-dev/eci-cli/main/scripts/install.sh | sh`
- Windows: `powershell -c "irm https://eci.sh/install.ps1 | iex"`

Every `eci` command runs with four environment variables and no `eci` configuration of the user's own:

| Variable | Value |
|---|---|
| `ECI_API_ENDPOINT` | the account's `endpoint`, default `https://portal.elice.cloud/api`. The public sector portal is `https://portal.gov.elice.cloud/api` |
| `ECI_API_TOKEN` | the account's access token |
| `ECI_ZONE_ID` | the account's `zone_id`, once one is chosen |
| `ECI_CONFIG` | `~/.letify/accounts/<alias>/eci.yaml` |

The token is never an argument and never printed. Reads pass `--format json`. A command that exits non zero raises `RuntimeFailure` naming the command, with any password replaced by `***`, and its standard error. When that standard error contains `401`, `403`, `unauthorized` or `permission`, the message starts with `Elice refused the access token or it lacks permission` and says a token is issued in the portal under User management, User access token.

**Which machine.** `machine_id` names an existing machine by name or UUID. Without it the machine is named `letify-<alias>` for an ondemand machine and `letify-<alias>-spot` for a spot one, with the alias lowercased and `_` replaced by `-`. A machine is found with `eci compute vm list --format json`, taking the row whose `name` or `id` equals it exactly. The name is how a later start finds the same machine, so nothing is written back to a configuration file.

**Start.** Starting a session runs these steps in order:

1. Find the machine.
2. A declared `machine_id` that is not listed raises `ProviderUnavailable` naming it.
3. A missing letify machine is launched. The instance type is the account's `instance_type` when set. Otherwise it comes from `eci instance-type list --format json`, keeping the rows whose `activated` is not `false`, because `eci` has no option to filter them: a GPU instance takes a row whose `devices` normalize to the instance's accelerator and number exactly the instance's device count, and `CPU` takes the row with no devices and the fewest `cpu_vcore`. No match raises `ProviderUnavailable` listing the names that were offered. The price type, price line and quota check follow Price type below. The password is generated: 20 characters with an upper case letter, a lower case letter, a digit and a symbol, and no three characters that run consecutively up or down such as `123` or `cba`. It is written to `~/.letify/accounts/<alias>/machine_password` with mode 0600 before the launch runs, so a launch that succeeds after letify is interrupted still has its password. The command is `eci compute vm launch --name <name> --instance-type <type> --password <password> --wait --no-spec`, where `--no-spec` keeps a launch spec the user saved as `default` from changing the machine letify asked for, with `--price-type spot` for a spot machine and `--image <image>` and `--size-gib <disk_gib>` only when the account sets `image` and `disk_gib`. Without them `eci` chooses Ubuntu 24.04 AI/GPU with 50 GiB for an accelerator type and Ubuntu 24.04 Standard with 20 GiB for a CPU type.
4. A machine with status `started` is used as it is. A machine with status `idle` is started with `eci compute vm start <name>`. Any other status is a transition, and `eci compute vm get <name> --format json` is read every 10 seconds until it is `idle`, for up to 300 seconds, before the start. After a start the status is read every 10 seconds until it is `started`, for up to 600 seconds. Either limit raises `RuntimeFailure` naming the last status.
5. The address is the machine's first public IP in `eci compute vm get <name> --format json`: a string under a key containing `public_ip`, or the `ip` of the first entry of a list under such a key. No public IP raises `ProviderUnavailable` saying the machine has none. A machine reports `started` before its SSH server accepts connections, so after a launch or a start letify prints `letify: <name>: waiting for SSH on <address>` and tries a TCP connection to the address on the account's SSH port every 5 seconds, for up to 300 seconds, before any SSH step. A machine that was already `started` is not waited for. The limit raises `RuntimeFailure` saying SSH on the address did not answer.
6. Right after a launch, letify's key is installed. The key is `key`, default `~/.ssh/id_letify`, generated as at login when missing. Its public half is appended to `~/.ssh/authorized_keys` of `user`, default `ubuntu`, the login user `eci` gives a launched machine, over one SSH connection that reads the password through `SSH_ASKPASS` from the password file. The password is therefore never an argument of letify's own SSH command. A machine letify did not launch must already accept the account's key.

The runtime's `external_id` is the machine name, and the session runs over forward SSH to the address.

**Stop.** Ending a session acts once no other runtime of this provider is on the machine, and follows the account's `persistent` setting, default `false` for Elice. On a persistent account letify runs `eci compute vm stop <name>`: an idle machine bills no compute, while its disk and public IP keep billing, and the next session starts the same machine with its disk. On an account that is not persistent a machine letify launched is deleted with `eci compute vm delete <name> --cascade -y`, which removes its disk, network interface and public IP, so nothing keeps billing, and the next session launches a new machine. A machine the account names with `machine_id` is never deleted; it is stopped. A stop or delete that fails prints `letify: could not stop <name>: <reason>. Run 'eci compute vm stop <name>'` or `letify: could not delete <name>: <reason>. Run 'eci compute vm delete <name> --cascade -y'` and does not raise, because the session is already ending. `letify logout` deletes nothing on Elice.

#### Price type <!-- id: elice-price-type -->

> An Elice machine runs ondemand or spot, chosen on the account and overridable on one instance.

`price_type = "ondemand"` or `"spot"` on the account, default `ondemand`. Any other value raises `ConfigError`. Elice's reserved pricing is not offered. `instance.priced("spot")` and `instance.priced("ondemand")` return a copy with that price type, the way `n * instance` returns a copy with `n` devices, and any other argument raises `ValueError`. The price type is part of the pool key. A spot machine may be stopped or deleted by Elice at any time.

A spot instance on a provider that has no spot pricing, which is every provider except `elice`, raises `UnsupportedMode` when its session starts. On Elice, spot on a CPU instance type raises `UnsupportedMode` before anything is created, because Elice offers spot for accelerator types only. A declared `machine_id` keeps the pricing it was created with, so a requested price type that differs from the machine's `pricing_type` raises `ConfigError` naming both.

Before a launch, letify reads `eci pricing list --resource-kind vm_allocation --format json` and prints `letify: <name>: <type> <price type> at <price> KRW/hour` from the row whose `name` is the instance type and whose `pricing_type` is the price type, or `letify: <name>: no <price type> price listed for <type>` when there is no such row.

An ondemand launch checks quota first, and a spot launch does not, because spot does not count against Elice's compute quota. letify reads `eci org info --format json`. A value of 0 for the instance type's id or name under `resource_quota.compute.instance_types`, or a `resource_quota.compute.devices` of 0 for an accelerator type, raises `ProviderUnavailable` saying the ondemand quota for that type is 0 and that the portal takes a quota request or `price_type = "spot"` avoids the quota. A quota that cannot be read prints `letify: <name>: the ondemand quota could not be read, launching anyway` and does not refuse.

`Launcher.status()` reports `price_type` on each runtime, `ondemand` or `spot` on Elice and `None` elsewhere, and `letify status` prints it. An Elice `Usage` record carries the account's `price_type`, which `letify usage` prints and `--json` includes.

#### Spot preemption <!-- id: elice-spot-preemption -->

> A spot machine that Elice stopped or deleted is an infrastructure failure, raised as `SpotPreempted` and retried under the ordinary retry rule.

When a call on a runtime fails with `RuntimeFailure` or `ProtocolError`, the call path asks the provider to diagnose it before retrying. On Elice, for a spot runtime, the provider reads `eci compute vm get <name> --format json`. A machine that is gone, or whose status is not `started` while letify did not stop it, turns the failure into `SpotPreempted`, a `RuntimeLost`, carrying `machine`, `state` (the last status, or `deleted`) and `at` in Unix seconds. A read that itself fails leaves the original failure standing.

A preemption prints `letify: <name> was preempted by Elice (state <state>)`. When the machine still exists it also prints `letify: <name> keeps its disk and public IP, which keep billing. 'eci compute vm delete <name> --cascade' removes them`. letify removes nothing.

Recovery follows [Failure and retry](#failure-and-retry). The call already running fails, the runtime is discarded, and the call is retried up to `retries` times; when no retry is left, `SpotPreempted` itself is raised. A retry starts a machine again by the start steps above, so an idle machine is started and a deleted one is launched. There is no fall back to local execution. `spot_fallback = "none"` or `"ondemand"` on the account, default `none`, decides the price type of that retry. With `none` the retry uses spot again and prints `letify: <name> preempted, retrying on spot`. With `ondemand`, every later start of that instance in this process uses the ondemand machine and prints `letify: <name> preempted, retrying on ondemand machine <ondemand name>`.

A preemption loses the worker's memory and every session cache handle. Files survive only on the machine's disk, which includes the workspace root, when Elice stopped the machine rather than deleted it, and in volumes, which live in the store.

### Inventory

> A provider entry declares which accelerators the account can get and how many of each. That inventory is the only thing that bounds how much runs at once.

Three facts force this, and no launcher-level number can express any of them.

A Colab account's available accelerators depend on its state: the tier, and whether the compute unit balance is positive. So the kinds are per account, and they change without letify being told.

A shared department machine holds several cards in one box, and which indices are free moves with whoever else is logged in. So an entry names the indices it may use, `indices = "0-3"`, and at the moment a session starts letify takes only those registered indices that are actually free on the machine. A card another person is already computing on is skipped, not fought over.

A run can take more than one card. `device=lab.A100 * 2` asks for two, and on a four card machine that is two concurrent sessions rather than four. A number bounding how many sessions may exist cannot say that, which is the plain reason such a number is not in the declaration.

| Field | Means | For |
|---|---|---|
| `count` | How many of this accelerator the account can hold at once | A provider that assigns the device itself, such as Colab or Modal |
| `indices` | Which device indices on the machine letify may use, as `"0-3"` or `[0, 1, 6]` | A machine letify shares with other people, where it sets the visible devices itself |

An entry with `indices` has a count: the number of indices. An entry with neither is one of that accelerator.

For a `shell` or `tunnel` account, `letify login` writes this table from what the machine reports, as described under Logging in. An entry with no table is still usable: its accelerators are discovered on first access, as described under Instances.

Which registered indices are free is read with `nvidia-smi` at reservation time, not cached, because the answer changes while a run is queued. A card is taken as busy when another user is computing on it: a compute process on the card counts when it is not a worker of this client and is not owned by the login user. The login user is the operating system user letify runs as on that machine, the account's SSH user for `shell`, `tunnel` and `elice` and the current user for `Local`. A card running only the login user's own processes stays free for letify, and letify leaves those processes alone. When the login user is `root`, every compute process owned by `root` that is not a worker of this client counts as busy, because many people run as `root` on a shared machine and ownership proves nothing there. A process whose owner cannot be read, because its id is not visible in the namespace the query runs in, counts as busy, because ownership cannot be proven. Nothing else on the machine is inspected, and letify never kills anything.

The reading is taken on the machine that owns the cards. `Local` runs the queries itself. A `shell`, `tunnel` or `elice` account runs them over the account's link, two SSH commands per reservation: `nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits` maps each uuid to its index, and one `sh -c` script lists the compute processes with `nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits`, reads the owner of each listed process id with `stat -c %U /proc/<pid>`, and prints the login user with `id -un`. The owners are read in the same command as the listing, so a process id cannot be reused between the two.

A compute process is excluded when it is a worker of a session this client process started on that provider. A runtime records its worker's process id from the worker's `stat` reply when its persistent channel starts, and gives it up at shutdown. The worker keeps that id when it moves to the project interpreter, because the move is an `execv`. A process started by any other client process, letify or not, counts as busy.

A remote query that cannot run, because SSH failed, `nvidia-smi` exited non zero, `stat` is missing or the login user could not be read, raises `RuntimeFailure` naming the busy check. Treating the cards as free would put a run on a card someone else is computing on. When no registered index is free, the `InsufficientDevices` message names each busy index with the user names that own its processes, `unknown` for a process whose owner could not be read. Command lines are never read or shown.

A reserved session sets `CUDA_VISIBLE_DEVICES` to its reserved physical indices and `CUDA_DEVICE_ORDER=PCI_BUS_ID` in its worker before any user code runs, and keeps both when the worker moves to the project interpreter. The training code sees its cards as 0 upward in `nvidia-smi` order and needs to know nothing about which physical indices it was given. A provider that assigns the device itself sets neither.

### Instances

> An `Instance` is one accelerator shape on one provider account.

`colab.G4` is an `Instance`. It holds the provider, the accelerator name, the host placement, how many devices one session takes, and the core count, memory and VRAM the provider reported. `n * instance` returns a copy taking `n` devices, and `instance.priced("spot")` a copy with that price type, as [Price type](#elice-price-type) describes. An instance has no method that changes where the host code runs, because that is the declaration's `host`.

The device count is part of the pool key, because a session holding two cards is not interchangeable with one holding one.

Because an instance carries its provider, `device=colab.G4` fixes provider, account and accelerator in one argument. `let.providers.any.G4` defers the provider choice to the first declared provider that registers a matching accelerator, in configuration order.

Instance discovery is lazy and cached. A provider that must connect to enumerate its accelerators does so on first access, never at import time, and a configuration entry may list `gpus` or a `devices` table to skip the connection. `refresh()` asks again. A `shell` or `tunnel` account gets its `devices` table at login, so the connection on first access is the fallback for an entry written without one: a login where `nvidia-smi` did not answer, a password account, or an entry written by hand.

`Local` reads its accelerator names once per process, because asking `nvidia-smi` takes seconds on a laptop whose discrete GPU is asleep and the answer does not change while the process runs.

Accelerator names are normalized so they can be attributes. `NVIDIA RTX PRO 6000 Blackwell` becomes `RTX_PRO_6000`. Colab calls the same card `G4`, which is what its CLI accepts, and accepts `RTX_PRO_6000` as an alias for it.

Every provider that can start a session without an accelerator registers it as `CPU`, as `Local`, `Colab` and `Modal` do, and `cpu` finds it too. `Colab` creates such a session with `colab new` and no `--gpu` or `--tpu`. `Modal` creates such a sandbox with `gpu` null.

### GPU utilization

> How hard each declared instance's accelerator is working right now, read from the machine that owns it.

`letify utilization` reports, for each physical device, its utilization percentage, memory used against memory total, temperature and power draw. `nvidia-smi --query-gpu` is the single source for those readings, because it is the only reading present on every machine letify reaches and it needs no framework loaded.

Where the reading comes from depends on whether the machine outlives a session.

| Provider | Scope | Read from |
|---|---|---|
| `Local`, `Shell`, `Tunnel` | `machine` | The machine itself, with no session: `nvidia-smi` here for `Local`, and over the provider's link for `Shell` and `Tunnel`. One row per provider |
| `Colab`, `Elice`, `Modal`, `Kaggle` | `session` | Inside the instance's live session, by shipping the same reader function through the ordinary call protocol. One row per instance |

A `machine` reading is read-only. It runs the three `nvidia-smi` queries the busy check runs, starts no process on a card and reserves nothing. Each device in it also carries who holds the card:

| `holder` | Means |
|---|---|
| `letify` | This process has reserved the index |
| `others` | Another user is computing on it, by the busy check rule under Inventory. `users` names them |
| `mine` | Only the login user's own processes, or this client's workers, are computing on it |
| `free` | No compute process is on it |
| `unknown` | The utilization was read but the owner query failed |

The first row of that table that applies wins. A `session` device has `holder` set to `None`.

A `session` instance with no live session reports no devices and says why, because starting a session to measure its load would cost money and change the answer. A machine without `nvidia-smi`, or one the link cannot reach, reports no devices with that as the reason. Neither is an error: every declared provider is listed either way. `--json` prints the rows with `alias`, `accelerator` (`None` for a `machine` row), `scope`, `devices` and `reason`.

The command prints one block per provider, with the same header, indent, gauge characters, colour and width rules as `letify usage`:

```
dept_gpu  shell
  gpu0  Tesla P100-PCIE-16GB  free
    load   [████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  20%
    memory [████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  10%  1.6/16.0 GiB
    41C  38W

colab_pro_plus  colab
  no live session, so nothing to measure
```

A card's header line is `gpu<index>  <name>  <holder text>`, where the holder text is `reserved by letify`, `busy: <users>`, `in use by your processes`, `free` or `holder unknown`, coloured cyan, red, yellow, green and not at all. The load line is left out when the card reports no utilization, the memory line when it reports no total, and the last line holds whichever of temperature and power the card reported. Each gauge is 16 to 40 cells: the terminal width minus 4 columns of indent, 7 of label, 2 brackets, 5 of percentage and 16 of memory text. Both gauges are coloured by the unused share with the thresholds `letify usage` uses. A `session` row's reason is printed as `<accelerator>: <reason>`, or once without the accelerator when every instance of the provider gives the same reason.

The reading is taken at the moment it is asked for and carries no history. A load that has to be watched over time belongs in the caller's own loop, not in a CLI that shells out to `nvidia-smi` per poll.

## Execution modes

> Two modes exist. `host` picks between them and nothing derives it.

**Function shipping** (`host="remote"`) serializes the declared function with cloudpickle and runs it inside the runtime. The whole loop executes there, so its host synchronizations never cross the network.

**PyTorch forwarding** (`host="local"`) keeps Python, the data and the libraries in the local process and forwards PyTorch operators to a worker that holds the GPU. It supports PyTorch only. The local process needs any PyTorch build, a CPU build included, and the operators run on real CUDA tensors on the runtime. Local data and the local environment stay in place, at the cost of one network round trip at every point where the host reads a value back from the device. [PyTorch forwarding](#pytorch-forwarding) describes the mechanism.

The declared function runs in the calling process, inside a session whose device worker is started on first use. It is not retried: a failure part way through has already run the function's side effects here once.

A provider refuses a mode only when it cannot serve it. `Modal` refuses `host="local"` because it exposes function calls into a container and there is no device to forward at. `host="local"` is refused with `UnsupportedMode` when PyTorch does not import in this process or is older than 2.1. A provider without a fast path warns with its expected round trip and then runs, because the choice belongs to whoever wrote the declaration.

### Efficiency model

> Time per step under forwarding is `T + n * d + k * RTT`, where `T` is GPU time per step, `n` is operators per step, `d` is the local dispatch cost per operator and `k` is host synchronizations per step. Efficiency against a direct run is `T / (T + n * d + k * RTT)`.

`n * d` is paid in the local process whether or not the link is fast. `k * RTT` is paid only at a synchronization, because every other operator is queued and sent without waiting. Batching makes the round trip count equal `k`, independent of `n`.

The targets, for the benchmark model in [NETWORK.md](NETWORK.md#pytorch-forwarding-on-dept_gpu):

| Quantity | Target |
|---|---|
| `d`, local dispatch per operator | at most 100 us |
| Round trips per step with no host read | 0 |
| Round trips per `loss.item()` | 1 |
| Operators per round trip, reading once per 50 steps | at least 1000 |
| Copies of 256 MiB each way | limited by the link, not by letify |

`letify.remoting.efficiency(step_seconds, syncs, round_trip_ms)` computes the `k * RTT` part.

The table below holds for workloads whose GPU step is long, where `n * d` is negligible against `T`. Numbers for an RTX PRO 6000 with NVFP4, a 0.5 s micro step, at a 150 ms round trip:

| Workload | Function shipping | Forwarding, default settings | Forwarding, tuned |
|---|---|---|---|
| LoRA fine-tuning | about 99 percent | 53 percent | about 96 percent |
| Decode, batch 1 | hundreds of tokens per second | 2 to 7 tokens per second | unchanged |
| Evaluation, teacher forcing | about 99 percent | about 99 percent | about 99 percent |

`k` is about three for a default Hugging Face training step: the trainer's NaN filter every step, the SDPA attention mask check every forward, and logging or the gradient scaler. Tuning means turning the NaN filter off, removing the mask check with fixed length packing, and moving logging to the gradient accumulation boundary, which leaves about one synchronization per optimizer step.

A faster GPU makes forwarding worse, because `T` shrinks while `RTT` does not. The same step on an L4 in bf16 takes 1.8 s and reaches about 80 percent where the RTX PRO 6000 reaches 53 percent.

Decoding fails at any useful latency. A decode step for 4-bit weights on an RTX PRO 6000 is 2 ms to 3 ms and synchronizes once or twice per token, so throughput is bounded near `1000 / (k * RTT)` tokens per second regardless of the card.

`letify efficiency` exposes the formula on the command line.

## Channels

> A channel is how letify talks to a runtime, and which kind a provider offers decides what letify can do there.

A **persistent channel** keeps one worker process alive behind a pipe pair. Messages are binary frames, so the worker process with its session cache, the blob table and anything written to disk all survive between calls.

A **one-shot channel** can only run a command and collect its output. Every call starts a fresh process, so nothing persists. It exists because some transports offer nothing more, and it refuses the operations that need persistence rather than pretending.

The worker source cannot be sent on standard input as a script, because `python -` reads to end of file before compiling anything and the pipe has to stay open for requests. A small bootstrap stub passed with `-c` reads one line holding a decimal byte count from `sys.stdin.buffer`, reads that many bytes of UTF-8 worker source, executes them, and leaves standard input where it was. From then on both pipes carry frames only.

### Frames <!-- id: frames -->

> Every message on a persistent channel is a sequence of binary frames with a 16 byte header. Nothing is base64 encoded and no pipe is opened in text mode.

A frame is a header followed by `length` payload bytes. The header is `struct.Struct("<2sBBIQ")`:

| Field | Size | Value |
|---|---|---|
| magic | 2 bytes | `b"LF"`. Any other value is a `ProtocolError` |
| type | 1 byte | the frame type below |
| flags | 1 byte | 0, reserved |
| stream | 4 bytes | the stream id, unsigned, little endian |
| length | 8 bytes | payload bytes that follow, unsigned, little endian |

| Type | Name | Direction | Payload |
|---|---|---|---|
| 1 | `HELLO` | worker to client | the interpreter's `<major>.<minor>` in ASCII |
| 2 | `REQUEST` | client to worker | a message head |
| 3 | `REPLY` | worker to client | a message head |
| 4 | `DATA` | both | the next bytes of the out-of-band buffers of the message open on `stream` |
| 5 | `STDOUT` | worker to client | raw bytes the worker wrote to file descriptor 1 |
| 6 | `STDERR` | worker to client | raw bytes the worker wrote to file descriptor 2 |
| 7 | `SHUTDOWN` | client to worker | empty. The worker exits |

Streams multiplex the one pipe pair. Stream 0 carries `HELLO`, `STDOUT`, `STDERR` and `SHUTDOWN`. Each request takes the next odd stream id from 1 upward, and its `REQUEST`, its `REPLY` and the `DATA` frames of both use that id. Stream 2 carries the PyTorch device executor's messages, as [Transport](#forwarding-transport) describes.

A message is one Python object. It is pickled with protocol 5 and a `buffer_callback`, so every `PickleBuffer` inside it, such as a `bytearray`, a `bytes` value of 1 MiB or more at the top level or inside a list, tuple or dict, or a NumPy array, becomes an out-of-band buffer instead of being copied into the pickle. The head frame's payload is `<I>` buffer count, `<Q>` length of each buffer, then the pickle. The buffers follow in order as `DATA` frames of at most 8 MiB each. The sender writes each frame with `os.write` on a `memoryview` of the buffer, so no joined copy is made. The receiver preallocates one buffer per out-of-band buffer and fills it with `readinto`, then unpickles with `buffers=`. The high bit of a buffer's 8 byte length marks a buffer that unpickles as a `bytes` value. For such a buffer the receiver allocates an uninitialized `bytes` object of that length with `PyBytes_FromStringAndSize(NULL, n)` through `ctypes` and reads into it, so the unpickled value is that object and no copy is made. Where `ctypes` is unavailable it reads into a `bytearray` and `bytes()` copies it once. Any other buffer is read into a `bytearray`. Peak memory for a `bytes` value is therefore one copy on each side.

A PyTorch tensor is pickled by letify's own reducer rather than by `Tensor.__reduce_ex__`, which copies the whole storage into the pickle. The reducer applies to an object of exact type `torch.Tensor` or `torch.nn.Parameter` on the CPU, with a strided layout, no autograd history of its own (a leaf) and at least one element. Such a tensor, made contiguous first when it is not, pickles as `torch.frombuffer` over its bytes, then `reshape` to its shape, then `requires_grad_` when it requires grad, and `Parameter` wraps it when it was one. Its bytes are a `PickleBuffer` when they are 64 KiB or more, so they travel as an out-of-band buffer written from the tensor's own memory, and a `bytearray` copied into the pickle otherwise. Every other tensor keeps PyTorch's own pickling. The reducer references PyTorch functions only, so a worker without letify unpickles it, and a message without tensors pays nothing for it. The same reducer is used by `wire.dumps`, by `codec.dumps_call_parts` on top of cloudpickle, and by the `put_blob` pickle of [argument addressing](#argument-addressing). A view is sent as the bytes it covers, not the whole storage it views, and two tensors that share one storage arrive as two separate tensors.

On Linux both ends set every pipe they frame over to 1 MiB with `fcntl(F_SETPIPE_SZ)`, capped at `/proc/sys/fs/pipe-max-size`, so an 8 MiB chunk crosses in 8 writes instead of 128. A descriptor that is not a pipe, or a system that refuses, keeps its size.

Frames of different streams interleave. A writer holds the write lock for one frame at a time, so a `stat` or `lease` request is sent between two 8 MiB chunks of a large upload, and a reply is not delayed behind another stream's data.

The worker announces itself with a `HELLO` frame naming the version of the interpreter it runs on. A worker asked to move to another interpreter replies, then replaces its process with `os.execv(<interpreter>, [<interpreter>, "-u", "-c", <bootstrap stub>])` on the same pipes, and the channel sends the worker source again and waits for the new `HELLO`. No other request is sent between the `reexec` request and that `HELLO`.

### Worker output <!-- id: worker-output -->

> The worker's standard output and standard error are streamed to the client as they are written, and written live to the client's own standard output and standard error.

Before it sends `HELLO`, the worker duplicates the pipe it was started on to a private descriptor for frames, and points file descriptors 1 and 2 at two new pipes with `os.dup2`. One thread per pipe reads up to 64 KiB at a time and sends each read as a `STDOUT` or `STDERR` frame. So `print`, `sys.stderr`, `logging`, warnings, tracebacks and output from C extensions all arrive the same way, and a `\r` progress bar arrives as the bytes it wrote.

The client writes each `STDOUT` payload to its own `sys.stdout` and each `STDERR` payload to its own `sys.stderr` as it arrives, while the call is still running. `Launcher(stream_logs=False)` turns that off. The client keeps at most the last 64 KiB of output per request, which is what `Channel.request` returns as its logs and what a `ProtocolError` quotes when the worker dies. No output is held until a call returns.

The client reads the process's own standard error, where SSH and the bootstrap stub report failures, on a thread that keeps the last 64 KiB. No pipe is left unread, so no amount of output can block the worker.

Nothing in this path runs per training step. A `print` inside a loop costs one pipe write in the worker, and the loop does not wait for the client.

### Waiting for a reply <!-- id: waiting-for-a-reply -->

> Requests on one persistent channel may overlap. There is no lock across a whole call.

Any number of threads may send requests on one channel. Whichever waiting thread holds the read lock reads the next frame and hands it to the request it belongs to, so a `stat` or a lease renewal sent while a call runs gets its reply while the call is still running. On the worker, `stat` and `lease` are answered by the thread that reads frames. Every other request is queued and run in order on the worker's main thread, so user code runs on the main thread.

A request that passes its timeout kills the worker process, which ends every read, and raises `RuntimeFailure`.

A body may fork, as `multiprocessing` and a `DataLoader` with `num_workers > 0` do. The thread that reads frames may hold the lock of `sys.stdin` at the fork, so every process forked from the worker replaces `sys.stdin` with `/dev/null` before anything else runs in it, and a child that closes `sys.stdin`, as `multiprocessing` does, never waits for that lock. On Linux each process forked from the worker asks for `SIGKILL` when the worker's main thread exits (`prctl(PR_SET_PDEATHSIG)`), so a worker killed by a timeout or closed with its session takes its forked children with it. A worker that closes its pipe fails every open request with `ProtocolError` quoting the last output.

The worker never installs anything into the interpreter it starts on. Everything it runs before it moves to the project interpreter uses only the standard library: the ready line, workspace preparation, the environment build and the move itself. The worker imports cloudpickle only when it loads a call, so a system Python that lacks cloudpickle and refuses `pip install`, as an externally managed Python under PEP 668 does, still starts the worker. blake3 and letify are likewise imported only after the move, and blake3 falls back to blake2b where it is absent.

### Modal adapter <!-- id: modal-adapter -->

> The letify process never imports `modal`. A small adapter runs in its own uv environment and letify talks to it in JSON lines over its standard input and output.

The adapter is `letify/providers/modal_adapter.py`. It imports only the standard library and `modal`, so it runs by file path and needs none of letify's own dependencies. letify starts it with:

```
uv run --no-project --python 3.12 --with "modal>=1.0,<2" python -P <path to modal_adapter.py>
```

`-P` keeps the adapter's directory off `sys.path`, so `import modal` finds the Modal package and not `letify/providers/modal.py` beside it.

The Modal version range is pinned in `tools.MODAL`. The adapter runs with `MODAL_CONFIG_PATH` set to `~/.letify/accounts/<alias>/modal.toml`, and with any inherited `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET` and `MODAL_PROFILE` removed. The profile Modal uses is the one the sign in activated in that file. So the account file alone decides which Modal account acts, and two accounts coexist on one machine.

The protocol is one JSON object per line. letify sends `{"id": <int>, "op": <name>, ...}` and waits for the line with the same `id`. The adapter answers `{"id": <int>, "ok": true, "value": <any>}`, or `{"id": <int>, "ok": false, "kind": <kind>, "error": <text>}`. `kind` is `unavailable` when `modal` cannot be imported, `not_found` when a volume path does not exist, and `failure` otherwise. Requests are answered one at a time, in order.

| Op | Fields | Value |
|---|---|---|
| `hello` | none | `{"modal": <installed Modal version>}` |
| `create` | `app`, `args`, `packages`, `gpu`, `timeout`, `idle_timeout`, `volumes`, `ports` | `{"sandbox": <id>}`. Runs `app` as an ephemeral app on first use, builds `debian_slim` with `packages` installed, and starts `args` in a sandbox. Each port in `ports` is exposed with Modal `encrypted_ports` |
| `tunnel` | `sandbox`, `port` | `{"host": <text>, "port": <int>, "tls": <bool>}`. The address that reaches `port` inside the sandbox, from `Sandbox.tunnels()`. `tls` is true when the connection has to be made with TLS, which an encrypted port needs |
| `write` | `sandbox`, `data` | `null`. `data` is base64 of at most 1 MiB. Writes the decoded bytes to the sandbox's standard input and drains it. Modal refuses a write that would buffer more than 2 MiB, so the channel splits a larger frame into `write` requests of 1 MiB, in order |
| `read_until` | `sandbox`, `prefixes` | `{"lines": [...], "eof": <bool>}`. The sandbox's stdout lines up to and including the first that starts with one of `prefixes`, or every line left when the stream ends |
| `terminate` | `sandbox` | `null`. Also ends a sandbox this adapter did not create, found by id |
| `billing_summary` | none | `{"metered_cost": <text>, "billed_cost": <text>, "credits": <text>, "start": <Unix seconds>, "end": <Unix seconds>}` for the current month, from `modal.Workspace.billing.summary()`. Amounts are decimal text in USD. `credits` is the `Credits` adjustment, negative when credit was applied |
| `volume_put` | `volume`, `version`, `path`, `data` | `null`. `data` is base64 |
| `volume_get` | `volume`, `version`, `path` | base64 of the file |
| `volume_list` | `volume`, `version`, `path` | the paths under `path`, recursively |
| `volume_delete` | `volume`, `version`, `path` | `null`. Removes the file or directory at `path`. A missing path is not an error |

Every volume op creates the volume when it is missing, as version `version`.

The persistent channel to a sandbox starts on that sandbox's standard input and output, carried by `write` and `read_until`. The sandbox runs the bootstrap stub `python3 -u -c BOOTSTRAP`, and the worker source goes out first as the byte count line and source described above. Modal returns a sandbox's standard output as text, so the source the channel sends sets `_LETIFY_TEXT_FRAMES = True` after the frame code, and that worker writes each frame as lines of base64 of the frame's bytes, each line encoding at most 36 KiB, so no line is longer than 48 KiB. Modal delivers a stdout line longer than 64 KiB as several lines, which are not base64 on their own, and ends the stream on a line of 768 KiB. Each line decodes on its own, because 36 KiB is a multiple of 3 bytes. The channel reads those lines with `read_until` and `prefixes` `[""]`, one line per request, and joins the decoded bytes back into frames. Frames sent to the sandbox are raw bytes, base64 encoded only inside the `write` request. A `read_until` that ends at end of stream without a reply raises `ProtocolError`.

A missing uv raises `ProviderUnavailable` naming uv. A reply of kind `unavailable` raises `ProviderUnavailable` for `modal`. An adapter process that exits, or prints a line that is not the reply it was waiting for, raises `RuntimeFailure` carrying the adapter's standard error, because that is an infrastructure failure. A reply of kind `failure` raises `RuntimeFailure` with the adapter's message. One adapter process serves one provider or one backend and exits when its standard input closes.

#### Ending a sandbox while a request is blocked <!-- id: modal-abort -->

> A sandbox is terminated through a second adapter process, so a request blocked in the first adapter cannot keep the sandbox running.

A `read_until` blocks until the sandbox prints a line, and the adapter answers one request at a time. So while a call runs, the adapter that carries it cannot take a `terminate`.

- `Adapter` records the id of every sandbox `create` returned, until a `terminate` for it succeeds.
- A request interrupted after its line was written and before its reply was read, by `KeyboardInterrupt` or any other exception, leaves the adapter out of step. Every later request on that adapter raises `RuntimeFailure` without writing anything.
- `Modal.stop` and the `SHUTDOWN` frame written by `SandboxChannel.close` wait at most 5 s for the adapter's lock. Other requests wait as long as they need.
- `Adapter.abort()` starts a second adapter process for the same account, sends it `terminate` for every recorded sandbox, and closes it. It then kills the first adapter's process group with `SIGKILL`. The first adapter is started in its own session, so that group holds only the adapter and its children. An aborted adapter is closed, and the provider starts a new one for its next request.
- `Modal.stop` calls `abort()` when the adapter is out of step or its lock is still held after 5 s. The call timeout watchdog calls it through `SandboxChannel._kill`, so a call past its `timeout` ends its sandbox too.
- The adapter's `terminate` for an id it did not create finds the sandbox with `modal.Sandbox.from_id` and terminates it. A sandbox that is already gone is not an error.

Modal also bounds a sandbox that nothing terminates. `create` passes `timeout`, the entry option `timeout`, 3600 s by default, as the sandbox's maximum lifetime. It passes `idle_timeout`, the entry option `idle_timeout`, 600 s by default, after which Modal terminates a sandbox that is idle.

### Modal data channel <!-- id: modal-data-channel -->

> Once the worker has said hello over the sandbox's standard input and output, the channel moves its frames to a TCP connection through a Modal encrypted port. Standard input and output stay the control path and the fallback.

`Modal.open_channel` creates the sandbox with `ports` `[DATA_PORT]`, where `modal.DATA_PORT` is 8765. The provider option `data_channel = false` creates it with no port and keeps every frame on standard input and output.

After each `HELLO` that arrives over standard input and output, the first and the one after every `reexec`, the channel opens the data channel:

1. It makes a token of 32 random bytes with `secrets.token_hex(32)` and sends the request `{"op": "listen", "port": DATA_PORT, "token": <token>, "wait": 60}` over standard input and output.
2. The worker's frame reader answers that request itself, before reading another frame. It binds `0.0.0.0:<port>` with `SO_REUSEADDR`, sends the reply, and accepts connections until one authenticates, `wait` seconds pass, or a byte arrives on standard input.
3. The channel asks the adapter for `tunnel` on `DATA_PORT` and connects to that host and port with a 30 s timeout, wrapping the socket in TLS with the tunnel host as server name when `tls` is true. It sets `TCP_NODELAY` and writes the line `LETIFY-DATA <token>\n`.
4. The worker reads that line within 10 s and compares the token with `hmac.compare_digest`. A connection that sends anything else is closed and the worker accepts again. On a match it closes the listener, sends `HELLO` on the connection, sends every later frame there as binary frames under the old sender's lock so no frame is split, and its frame reader reads frames from the connection instead of standard input.
5. The channel waits for that `HELLO` and then carries every request over the connection.

TLS comes from the Modal tunnel. The channel's side of TLS is an `ssl.SSLObject` over two `ssl.MemoryBIO` buffers, not an `ssl.SSLSocket`, because the thread reading frames and a thread writing a request run at the same time and OpenSSL does not allow two threads inside one TLS object. One lock guards the TLS object and its buffers. A second lock orders the encrypted bytes on the socket. Socket reads and writes happen outside the first lock, so a write blocked on a full socket never stops the reading thread from decrypting. Each write encrypts at most 1 MiB. The token is what stops another client of the public tunnel address from speaking the protocol, and it travels only over the adapter's authenticated control path.

#### Parallel data streams <!-- id: modal-data-streams -->

> The data channel is `N` TCP connections through the same tunnel, called lanes, and a write of 1 MiB or more is split across all of them, because one TCP stream over a path with a round trip near 190 ms carries about 12 MiB/s.

`N` is the provider option `data_streams`, an integer from 1 to 16, 4 when it is not set. Any other value raises `ConfigError` naming the account. `data_streams = 1` is the single connection described above, with no segment framing.

With `N` above 1:

1. The `listen` request carries `"streams": N`. Lane 0 authenticates with the line `LETIFY-DATA <token>\n` and lane `i` from 1 to `N - 1` with `LETIFY-DATA <token> <i>\n`. The channel opens the `N` connections at the same time. The worker accepts until every lane from 0 to `N - 1` has authenticated once, and closes a connection with a wrong token, an index out of range or an index already taken. Only then does it close the listener and send `HELLO`.
2. The frames of both directions become one byte stream carried as segments. A segment is a frame header, `wire.HEADER`, with type 8 `SEGMENT`, flags 0, stream set to the lane index and length set to the segment's bytes, followed by the 8 byte little endian offset of its first byte in the byte stream, then the bytes. `SEGMENT` appears only on a lane, never inside the byte stream.
3. A write shorter than 1 MiB (`wire.STRIPE_MIN`) is one segment on lane 0, sent by the writing thread. A longer write is cut into `N` pieces of `ceil(length / N)` bytes, the last one shorter or absent, and piece `i` goes on lane `i`. Lane 0's piece is sent by the writing thread and every other piece by that lane's own sending thread, and the write returns once every piece is sent. One lock holds a write from its offset to its last piece, so offsets follow write order. The 8 MiB `DATA` chunks of [Frames](#frames) are writes, so a large buffer crosses all lanes chunk by chunk.
4. Each end has one reading thread per lane. It reads segments and hands them to one reassembly buffer keyed by offset, and the frame reader takes bytes from it in offset order only. A lane whose next segment does not start at the next undelivered offset waits before reading its bytes while the buffer holds more than 64 MiB (`wire.STRIPE_HOLD`), so memory held out of order stays bounded. A segment whose magic or type is wrong, or whose offset repeats bytes already received, ends the stream.
5. The byte stream ends for its reader once every lane has reached end of stream or failed, and the bytes received in order before that have been delivered. The channel handles that as a closed connection. One lane ending leaves the others running, because a worker that replies to `reexec` and then replaces its process closes all lanes at once, and the reply may still be arriving on another lane when the first end of stream does. A malformed segment ends the stream at once. Shutting the socket of every lane is how closing and a request timeout end a blocked read.

While the channel waits for `HELLO` each read waits at most 30 s, and a lane reading thread treats a socket timeout as no data yet. After `HELLO` reads wait without a limit.

Socket buffers are left to the kernel. `SO_SNDBUF` and `SO_RCVBUF` are never set, because setting them turns off the kernel's buffer tuning and the value is capped at `net.core.wmem_max`, 212 KiB by default.

When any step fails, a `tunnel` failure, a connection that does not open, or no `HELLO` within 30 s, the channel prints one line on stderr, `letify: <runtime>: the data channel did not open (<reason>); frames stay on standard input and output`, and keeps using standard input and output. The worker returns to reading standard input when a byte arrives there or its `wait` ends without an authenticated connection, so a request the channel sends over standard input after a failure is answered without waiting for `wait`.

A `reexec` request goes over the data connection. The worker replies there and replaces its process, which closes the connection. The channel then sends the worker source over standard input, waits for `HELLO` there, and opens the data channel again.

Closing the channel sends `SHUTDOWN` over the connection that carries frames and closes the socket. A connection that ends while requests are open fails them with `ProtocolError`, as a closed pipe does. A request timeout closes the socket, which ends the blocked read.

The adapter never deploys an app. `create` starts `modal.App(app).run()` the first time it sees an app name and holds that context for the adapter's lifetime. When standard input closes, the adapter terminates its remaining sandboxes and then leaves every app context, which stops the ephemeral app. An adapter that dies stops sending Modal's client heartbeat, and Modal stops the ephemeral app for it. So no app named `app` stays on the account after letify stops.

## Call protocol

> A call is a serialized function plus arguments, and the outcome comes back on the same channel.

The local side pickles `(function, args, kwargs)` with cloudpickle, protocol 5 and a `buffer_callback`, and sends `{"op": "call", "payload": <pickle>, "buffers": [<out-of-band buffers>]}` as one message in [frames](#frames). The outcome comes back as the message `{"ok": True, "value": <value>}` or `{"ok": False, "error": <text>, "traceback": <text>}`, so a large returned value travels as out-of-band buffers too.

Base64 appears only where a transport carries text: the one-shot driver, the Colab contents API, and the JSON lines of the Modal adapter.

On a one-shot channel the call travels inside a driver script that prints its base64 encoded outcome between `__LETIFY_RESULT_BEGIN__` and `__LETIFY_RESULT_END__`, so it can be found in a stream that also carries the user's prints. Absence of the marker is not a protocol quirk: it means the remote process died, and letify reports that as `ProtocolError` naming the likely causes.

An `async def` body is awaited on the remote side, so it runs to completion there and can use `await` internally.

### Session cache <!-- id: handles -->

> `letify.session_cache(key, factory)` returns the value stored under `key` in the current session, building it with `factory()` on first use. A call always returns its value.

```python
@let.function(device=colab.G4, host=letify.remote)
def generate(prompt):
    model = letify.session_cache("model", load_model)
    return model(prompt)
```

Inside a declared function running in a runtime, `session_cache` looks `key` up in a store held by the runtime's worker process. On first use it calls the zero-argument `factory` and stores the result. The value lives until the session ends: the call ends it, or the enclosing `keep_alive` block ends it.

Each session has its own cache. Concurrent sessions each build their own value, so no result depends on which session the pool picks for a call.

The store lives in the `letify` module, which the worker imports by reference. A dict at module level in the user's script cannot do this job, because cloudpickle ships `__main__` globals by value with every call, so each call sees a fresh copy. letify is importable in the runtime because the user's environment, keyed by `uv.lock`, includes it. A worker that cannot import letify, whether the call refers to letify when it is loaded or the body imports letify while it runs, fails the call with `RemoteError` whose message says that letify is not installed in the runtime's environment and that `uv add letify` fixes it.

Concurrent first use of one key builds once: the other callers wait for that build and receive its value. A factory that raises stores nothing, so the next use tries again.

Outside a runtime, in the local process or in `Function.local()`, `session_cache` is a memo for the life of the process with the same semantics, so a body tested locally behaves the same.

On a one-shot channel every call is a fresh process. `session_cache` works within that call and keeps nothing for the next one.

A returned value always comes back to the caller. There is no declaration argument that returns a reference instead.

### Argument addressing

> Large arguments are named by the hash of their contents, so the same value travels once.

A top-level argument whose serialized size is 64 KiB or more is replaced by a `Blob` reference. The runtime is asked which digests it already holds, and only the rest is sent with `put_blob`. A later call carrying the same value sends the reference instead of the bytes.

A `bytes` argument is hashed as it is. Any other argument is pickled with protocol 5 and a `buffer_callback`, and the digest is taken over the pickle and then each out-of-band buffer in order, with no joined copy. The digest is blake3 with 16 byte output, or blake2b from the standard library where blake3 is missing.

The digest of an immutable argument is cached on the `Runtime` for the life of the session, so a repeated argument is not hashed again. Immutable means a `bytes` object, or an object that supports weak references and exposes a read-only buffer, such as a NumPy array with `writeable` set to `False`. A `bytes` entry holds a reference to its object so its id cannot be reused, and at most 16 such entries are kept, least recently used first out. Any other argument is pickled and hashed on every call, because it may have changed.

The worker keeps the unpickled value of an immutable blob, so a repeated argument is not unpickled again either. For any other blob it keeps the pickle and the buffers, and unpickles a fresh copy for each call, so a call that mutates its argument does not change what the next call receives.

### Argument blobs on a persistent disk <!-- id: persistent-argument-blobs -->

> On a persistent provider an argument blob is also written under the workspace root, so a later session on the same machine receives the digest instead of the bytes.

A session whose provider is persistent, prepares a workspace root and has a persistent channel sends `{"op": "blob_dir", "path": "<workspace root>/blobs", "limit": <bytes>}` once, after the interpreter check. From then on the worker writes every blob it receives with `put_blob` to `<workspace root>/blobs/<first two hex characters>/<digest>`, before it replies. A `bytes` blob is the file as it is. Any other blob goes to `<digest>.pickle`: the eight bytes `LTFYPKL1`, one byte that is 1 for an immutable value, a big-endian 32-bit part count, a big-endian 64-bit size per part, then the pickle and each buffer in order. A file is written to a name ending in `.partial.<pid>` and renamed, so a reader never sees half a blob.

`have` reports a digest as held when it is in the worker's memory or its file exists, and sets that file's modification time to now. A call that names a blob the worker holds only on disk loads it into memory first. An ephemeral provider sends no `blob_dir`, so its worker writes nothing.

After each write the worker lists the blob directory and removes files, oldest modification time first, until their total size is at most `limit`. `limit` is 32 GiB. The file just written is never removed by its own write.

### Failure and retry

> Infrastructure failure may be retried. User code failure never is. Neither falls back to a slower path.

`RuntimeFailure` and `ProtocolError` mean the session misbehaved, so the runtime is discarded and the call is retried on a fresh one up to `retries` times. Before that, `Provider.diagnose(runtime, failure)` may replace the failure with a more specific infrastructure failure, as Elice does with `SpotPreempted`; the default returns it unchanged. A `SpotPreempted` left after the last retry is raised as it is. `RemoteError` means the shipped function raised, and it propagates with the remote traceback attached.

A `RuntimeFailure` raised for a failed command carries `command` and `stderr`, and its message names the command and the last 40 lines of stderr. The `RuntimeLost` raised after the last retry keeps the last failure's message, `command` and `stderr`.

letify never falls back to local execution or to a slower mode when the declared one is unavailable. A silent downgrade turns a four times slowdown into a mystery.

## Sessions

> A runtime is one live session and the only object that costs money.

Everything above a runtime is declaration. Creating one is when a provider actually powers something on; shutting it down is when the charge stops.

A runtime boots in seven steps:

1. Open the channel. The worker starts on the bootstrap interpreter: the account's `python` option, `python3` by default.
2. Arm the lease.
3. Prepare the workspace root, as Workspace root describes: expand `~` on the runtime, create the directory, make it the worker's working directory, and point `TMPDIR` at `<workspace root>/tmp`.
4. Build the environment: restore the environment archive, or run `uv sync` in the project directory, as Environment describes.
5. Move the worker to `<project directory>/.venv/bin/python`.
6. Check the worker's interpreter version against the local one.
7. Attach volumes.

Step 3 is skipped on the local provider, whose worker keeps the working directory of the process that started it. Steps 4 to 6 are skipped on the local provider, which already runs in the project's environment. Steps 1 to 5 run on the bootstrap interpreter and use only the standard library, as Channels describes. Steps 4 and 5 are skipped when the account sets `python`, which means the user manages the interpreter on that machine. In their place the worker checks that the interpreter can import cloudpickle, and one that cannot raises `ConfigError` naming the interpreter and the missing module. Step 6 still runs then.

### Pooling

> Runtimes are pooled by instance and environment, so the second call through a declaration pays nothing for setup.

The pool key is the instance key joined with the environment key. Two declarations that agree on both share runtimes, which is why nothing has to be said for two functions on one device to reuse a session.

How many sessions may exist is the provider's inventory and nothing else. Starting one reserves the devices its instance asks for. A call that cannot reserve them waits only while a session in this process that holds that accelerator is serving a call, because that session gives its devices back when the call finishes. A session this process is still starting counts as serving the call that started it; more concurrent calls than the inventory has cards for rely on exactly this. In every other case the devices cannot be allocated, and the call raises `InsufficientDevices` at once, naming what holds them: an idle session that a `keep_alive` block is keeping, a card another process is computing on, or a request for more devices than the account declares. Waiting there would never end, because nothing letify is running would free a device. `InsufficientDevices` is not retried, since a retry asks for the same devices from the same inventory. There is no ceiling on the launcher: a number there would be a guess about hardware the provider entry already describes, and when the two disagreed the smaller would win silently.

A session is never a value the caller holds. Pooling, reuse and teardown are decided from the declaration and the `keep_alive` block around it, so there is no call that starts a session, none that returns one, and none that takes one. `Runtime` exists, and letify hands it to a provider and to a volume, but it does not appear in anything a user writes.

### Lifetime

> A session ends with the call that needed it. `with let.keep_alive():` keeps sessions for the length of a block. Nothing else decides.

1. **The call.** A session ends when the call that started it finishes. Calls that overlap in time share that span: a session released by one call while another is still running stays up for the next call that matches it, and ends when the last overlapping call finishes.
2. **A `keep_alive` block.** Inside `with let.keep_alive():` a session is not ended when its call finishes, so the next call in the block that matches its instance and environment reuses it and pays no session start, which is minutes on Colab. When the block exits, every idle session ends; one still serving a call ends when that call finishes. Blocks nest, and only the outermost exit ends anything.
3. **The lease.** The local process renews a deadline inside the session every 30 seconds, and the worker exits on its own if the deadline passes. The grace period is 300 seconds, so a brief network drop does not kill a training run.

Keeping is a block rather than a declaration argument because it describes a stretch of the caller's program, not a property of one function: the same function is kept in one script and not in another. A block also has a visible end, so no session outlives the code that asked for it.

Nothing is torn down by hand and nothing is torn down on a timer. There is no release call and no shutdown call on the public surface, and everything left goes at process exit.

The lease is the one exception, and it is not a timer on the work: it covers the moment a process is killed outright, which is the one moment nothing can be told to anybody. `SIGKILL`, the out of memory killer and a power cut all run no code at all, so a session that only ends when asked would never be asked.

What the lease actually does is exit the worker process, which releases the occupancy. Whether that stops the billing depends on what the provider charges for, and infrastructure cannot choose to switch itself off: something that owns it has to. So the guarantee is per provider and letify states it rather than implying one.

| Provider | What is billed | Killed local process |
|---|---|---|
| `Local` | nothing | the subprocess dies with its parent |
| `Shell`, `Tunnel` | nothing; the card is occupied | the worker exits, so the card frees |
| `Modal` | the sandbox | **guaranteed.** A deadline is set when the sandbox is created and Modal enforces it |
| `Colab` | the runtime | not guaranteed. Colab's own idle policy is what ends it |
| `Elice` | the started machine | **not guaranteed.** A started machine bills compute until something runs `eci compute vm stop` |

Where it is not guaranteed, the preferred answer is a deadline at creation time, because the platform outlives the caller. Modal takes one and letify sets it. `eci compute vm launch` documents no deadline.

Where the platform takes none, the intended bound is reconciliation: the next letify process asks the provider what is running under this project's name and ends what nothing is watching. That is not immediate, and it is **not implemented yet**, so today an Elice machine left started by a killed process bills until somebody stops it. It is listed under Known gaps.

There is no detached execution. A detached run whose remote side is preempted would lose its results, so the local process stays the owner and durability comes from checkpoints in the store.

### Status reporting

> What is running, counted rather than described, with no internal bookkeeping in it.

`Launcher.status()` answers three questions: how many sessions exist, how many are serving a call, and what each one is. `live` and `busy` are counts, and `devices` reports each provider's inventory against what is reserved, so a reader can see at a glance whether a call is waiting for a card. `runtimes` describes each session: its name, provider, accelerator, the device indices it holds, placement, whether it is busy and how long it has been idle.

Nothing internal is reported. The pool holds a guard so that a session released by one call is not ended while an overlapping call is still running, and whether that guard is currently open is a fact about the pool's implementation rather than about what is running. A field among counts that looks like a count and is actually a boolean is worse than no field, because it is read as a count.

Each runtime also reports `price_type`, as [Price type](#elice-price-type) describes, `uptime_seconds`, the `link` strategy its provider connected over and the `rtt_ms` that link measured, each `None` where there is none. `letify status` asks `usage()` only of providers with a live runtime, because a runtime is what costs money, and adds that record as `usage`.

`status()` describes this process only. A session started by a different process is not in it, since the pool lives in the process that owns it. What a machine itself is doing is a different question, answered by `letify utilization`.

## Storage

> A volume is a content addressed blob store on whichever backend a provider has. Each provider holds one volume per project, filled automatically from what a call uses, so a declaration never names one.

Attaching a volume to an ephemeral provider moves the environment archive and the model cache into storage that outlives the runtime. A twenty gigabyte cache takes about 27 minutes to pull from a lab server over a 100 Mbit/s link, three to five minutes from the Hugging Face hub, and 40 to 60 seconds from a bucket inside the same infrastructure as the runtime. That difference is billed as GPU time.

### Content addressed layout

> Blobs are immutable and named by their hash. Mutable names live in a separate, tiny namespace of refs.

```
blobs/<first two hex characters>/<digest>
refs/<name>
```

Immutability buys two things. Concurrent writers cannot conflict, because different contents get different names, where a two way synchronization loses one writer's changes to the other. And nothing is verified twice, because holding a digest is proof of holding the contents.

Refs carry the mutable part, in the way Git keeps branch names apart from objects. A ref is a few dozen bytes, so a last writer wins race on one is harmless and both blobs survive it. letify reserves `env/<env key>-<platform>` for environment archives and `path/<path key>` for project data; nothing else is written there.

### Blob granularity

> The unit of a blob is a decision. Large files stand alone; trees of small files are packed.

A model shard is already large, so one file is one blob. An environment is tens of thousands of small files, so the whole tree is packed into one archive keyed by the environment key and the runtime's platform. That is where the speedup is: tens of thousands of round trips become one.

A content addressed store does not by itself reduce the bytes of a first transfer. What it improves is the metadata exchange, which becomes a single manifest read instead of one request per file, and every repeat transfer, which is skipped by name. Neither it nor a synchronization tool sends deltas within a changed file.

Extraction checks every member's path against the destination before unpacking, so an archive cannot write outside it. An environment archive is extracted with Python's `tar` filter rather than the `data` filter, because a `.venv` links its interpreter by absolute path and the `data` filter refuses such a link.

### Materializing into a runtime

> The runtime pulls a volume straight from the backend, using a short-lived credential borrowed from the local machine. Bytes never pass through the local process on the way in.

A volume holds a copy of the local project's environment and data, so its purpose is to make a new runtime start as if it were the local machine. That copy is only useful if it arrives fast, and the fast path is the backend's own network: a Colab runtime reading a Cloud Storage bucket is a transfer inside Google's infrastructure. Routing the bytes through the local process would cap every session start at the user's uplink.

The local machine writes to the backend directly as well. It does not relay through a runtime, because that adds a hop that is still bounded by the same uplink and spends billed runtime time.

No credential is stored on the remote side. For each materialization the local process derives a short-lived access token from its own login, scoped to reading the volume's prefix where the backend supports scoping, and sends it over the channel. The worker keeps it in memory only, never on disk or in the environment of user code, and drops it when the pull finishes. A backend that has no network path from the runtime, such as `filesystem` on a machine the local process can reach but the runtime cannot, falls back to writing through the channel.

A backend offers a pull by answering with a URL and the request headers that read one blob. The local process sends them in one `pull` request naming the destination path and whether to unpack it in the volume directory. The worker streams the response to the path, unpacks it when asked, and discards the request, headers included, before it reads the next one. `gcs` offers a pull. `filesystem` and `modal` do not, so they write through the channel. A one-shot channel has no worker to pull, so it writes through the channel as well.

The `gcs` read token is downscoped with a Credential Access Boundary: the local token is exchanged at `https://sts.googleapis.com/v1/token` for one that holds `roles/storage.objectViewer` on the volume's bucket only, with an availability condition restricting it to object names under the volume's prefix. A volume option `sts_endpoint` points the exchange elsewhere. A failed exchange raises `RuntimeFailure` rather than sending the unscoped token.

The environment archive is automatic. The first session that runs `uv sync` for an environment packs its project directory, `.venv` included, into its first volume under `env/<env key>-<platform>`. `<platform>` is the runtime's `sys.platform` and `platform.machine()` joined by a hyphen, for example `linux-x86_64`, and the env key includes the interpreter's major.minor version. Every later session on the provider with the same key and platform pulls that archive and unpacks it at `<workspace root>/project` instead of syncing.

A volume's files on a runtime live in its volume directory, `<workspace root>/volumes/<volume name>`. A blob materialized without a named destination is written to `<volume directory>/blobs/<first two hex characters>/<digest>`. The volume option `mount` names another directory for one volume; nothing else sets it. A restored `.venv/bin/python` that does not start is treated as no archive, and the session syncs. Nothing in the public surface names this step.

### Volumes on a persistent runtime <!-- id: persistent-volumes -->

> On a persistent provider the volume directory under the workspace root is the runtime's copy of the volume. A file is sent only when that copy does not already hold its digest, and no environment archive is packed or restored.

A persistent runtime keeps `<workspace root>/volumes/<volume name>` between sessions, so a later session already holds what an earlier one received. The volume directory holds a manifest, `.letify-manifest.json`, mapping each materialized destination path to the digest, size in bytes and modification time in nanoseconds it had when it was written.

Before a blob is written to a destination that is not unpacked, the local process asks the runtime for that manifest entry. When the entry names the same digest and the file on disk still has the recorded size and modification time, nothing is sent. Otherwise the blob is written as Materializing into a runtime describes, and the entry is recorded. A file changed on the runtime by anything other than letify fails the size or time comparison and is sent again. The manifest is replaced atomically, so a session that reads it while another writes sees one version or the other, and at worst sends a file twice.

A persistent provider builds its environment with `uv sync` in `<workspace root>/project/<env key>` every session, and never packs the project directory into a volume or unpacks an archive from one. A sync over an existing `.venv` checks it and installs nothing, so it is faster than any archive transfer, and the `.venv` is already on the disk the archive would be unpacked to.

An ephemeral provider keeps the behaviour of Materializing into a runtime: every file is sent each session, and the environment archive is packed and restored.

Nothing hands a session to the caller. There is no call that returns one, no argument that takes one, and no way to hold the wrong one, because which session serves a call is the pool's answer to work out from the declaration.

### Project data <!-- id: project-data -->

> A call's data is found in the call itself. Every `pathlib.Path` the function reaches travels with it: the local contents go up to the provider's volume, the runtime sees the same path already filled, and what the call changes under it comes back when the call ends.

The declared function is sent with cloudpickle, which serializes its closure variables, the globals it references, its default arguments and the call's arguments. letify's pickler intercepts every `pathlib.Path` among them through `reducer_override`. Nothing is declared: the function's own references are the declaration, which is what makes the decorator behave as a closure over the data it uses.

A path is project data when it exists on the local machine at call time, as a file or a directory. For each one:

1. The local contents are packed into content addressed blobs, large files as their own blobs and trees of small files as one archive, and only the digests the volume is missing are uploaded, directly to the backend.
2. The ref `path/<path key>` records the manifest, where the path key is the digest of the path resolved against the project root.
3. In the pickled call, the path is replaced by the path the runtime materializes it at, under the volume directory, so the function body uses it unchanged.
4. Before the call runs, the runtime pulls the manifest's blobs straight from the backend, as Materializing into a runtime describes.

A path that does not exist locally is an output location. It is created empty in the runtime and replaced the same way.

When the call returns, the runtime compares each replaced path with the manifest it started from. Files that are new or changed are stored as blobs in the volume, and the local process writes them back to the local path. So a checkpoint written under a referenced `Path` is on the local disk when the call returns, and the next session receives it as input with no separate resume step.

Only `pathlib.Path` objects are detected. A string that happens to name a local file is left alone, because no rule can tell a path from any other string, and uploading on a guess would send data the user did not mean to send.

Detection costs one `stat` per path per call, and packing is skipped when the manifest ref already matches the local tree's modification times and sizes.


### Backends

| Backend | Used by | Note |
|---|---|---|
| `filesystem` | `Local`, `Shell`, `Elice` | A directory. The local machine can be the origin others pull from. On Elice it sits on the machine's own disk. |
| `gcs` | `Colab` | A Colab runtime is a Compute Engine virtual machine, so this is an internal transfer. Use a multi-region bucket, because runtime placement is not selectable. The client is the standard library HTTP client against the Cloud Storage JSON API. |
| `modal` | `Modal` | A Modal volume, mounted beside the container. Reached through the Modal adapter, never through a `modal` import in the letify process. |

The `modal` backend uses Modal volume version 2, and sends `version` 2 on every volume op. A version 1 volume accepts a file above Modal's 4 MiB large-file limit and then fails to read it back. `ModalBackend.delete` removes a blob, and a blob already absent is not an error.

The `modal` backend acts as one Modal account. A volume option `account` names its alias, and a volume on a `Modal` provider defaults it to that provider's alias. A volume on another provider that names the `modal` backend without `account` raises `ConfigError`, because there is no account to act as.

Every backend answers "which of these digests are missing" with one listing rather than one request per digest, because object level requests are billed and add latency.

### Google login for gcs <!-- id: gcs-login -->

> The `gcs` backend borrows the user's existing Google login. It stores no credential and adds no dependency.

The client calls the Cloud Storage JSON API at `https://storage.googleapis.com`: a listing with `prefix` and `pageToken` for `missing`, a media upload for `put` and `write_ref`, and a media download for `get` and `read_ref`. A volume option `endpoint` points it elsewhere. A request that answers 404 means the object is absent; any other status that is not 2xx raises `RuntimeFailure` naming the status and the object.

The access token is looked up in this order, the first that answers wins:

1. The `GOOGLE_OAUTH_ACCESS_TOKEN` environment variable, used as it is.
2. Application Default Credentials: the file named by `GOOGLE_APPLICATION_CREDENTIALS`, or `application_default_credentials.json` in the gcloud configuration directory (`~/.config/gcloud/` on Linux and macOS, `%APPDATA%\gcloud\` on Windows). A file of type `authorized_user`, which `gcloud auth application-default login` writes, is exchanged for an access token by posting its refresh token to its `token_uri`, `https://oauth2.googleapis.com/token` by default, with the standard library HTTP client.
3. `gcloud auth print-access-token`, when `gcloud` is on `PATH`.

A token is reused until 60 s before it expires. A service account key file is refused with its reason, because exchanging one needs an RSA signature the standard library cannot make; `gcloud auth activate-service-account` followed by rule 3 covers that case. When no rule answers, the backend raises `ProviderUnavailable` for `gcs` saying that no Google login was found and naming `gcloud auth application-default login` and `GOOGLE_OAUTH_ACCESS_TOKEN`.

## Environment

> An `Env` is a declaration keyed by the project's uv files, not a built image. Every remote runtime builds it with `uv sync` into a project `.venv` and runs the worker with that `.venv`'s Python.

`Env()` reads `uv.lock`. `pip_install`, `run`, `vars` and `ship` refine it and return a new value. The project directory is the directory holding the lock file. The key is a hash of the contents of `uv.lock`, `pyproject.toml` and `.python-version` in that directory, `Env.python`, and the refinements, so two declarations that agree share a pooled runtime and a cached archive.

A uv lock file resolves for every platform uv supports, so one lock file drives a Linux runtime from a Windows or macOS client. A `pip freeze` list does not, because it carries platform specific pins.

### Building the environment on a runtime <!-- id: remote-uv-sync -->

> The runtime receives `pyproject.toml`, `uv.lock` and `.python-version`, runs `uv sync --frozen --no-install-project` in a project directory, and starts the worker with that directory's `.venv/bin/python`. A default `Env()` does the same; no path installs into the system Python or installs nothing.

This applies to every provider except `local`: `shell`, `tunnel`, `colab`, `elice` and `modal`. A Modal sandbox image carries only what starts the bootstrap worker, so letify reaches the sandbox through the sync like every other package.

1. The local side reads `pyproject.toml` and `uv.lock` from the project directory, and `.python-version` when it exists. A missing `pyproject.toml` or `uv.lock` raises `ConfigError` naming the file and `uv lock`, before anything runs on the runtime.
2. The worker writes them into the runtime's project directory, `<workspace root>/project/<env key>`. The workspace root is the provider's `workspace_root`, the one place every path letify writes on the runtime derives from, as Workspace root describes.
3. The worker runs `uv sync --frozen --no-install-project --python <major>.<minor>` in that directory, with the version `Env` records, as Interpreter version describes. uv creates `.venv` there, and downloads the interpreter when the runtime has none that matches.
4. Packages from `Env.pip_install` are installed with `uv pip install --python <project directory>/.venv/bin/python <packages>`. Commands from `Env.run` run through the shell with `<project directory>/.venv/bin` first on `PATH` and `VIRTUAL_ENV` set to the `.venv`. Variables from `Env.vars` are set in the worker's environment before the sync, so they reach every later step and user code.
5. The worker moves to `<project directory>/.venv/bin/python`, as Channels describes. On a one-shot channel every later program runs as `<project directory>/.venv/bin/python -c <program>`, a child of the provider's command, with its standard output passed through.

`--frozen` installs exactly what `uv.lock` says and never re-resolves, so the runtime holds the library versions the local `.venv` holds. `--no-install-project` is used because only the three metadata files travel: the project's own source is not on the runtime, so building the project there would fail. The project's own modules reach the runtime with the call instead, as Module shipping describes. letify itself is installed by the sync, because the project depends on it. A path dependency is installed from its locked path on the runtime, so one that exists only on the local disk fails the sync.

A failed sync raises `EnvironmentFailure` saying `uv sync failed on <runtime>`, with the command and the last lines of uv's standard error. `EnvironmentFailure` is a `RuntimeFailure`, so the call is retried on a fresh runtime.

### Environment on the sandbox disk <!-- id: modal-env-disk -->

> On `modal` the project directory, its `.venv` and uv's cache live on the sandbox's own disk, not on the workspace volume, because importing a large package from a Modal volume reads thousands of small files over the network.

A provider's `env_root` names where the project directory lives instead of the workspace root. It is None for every kind except `modal`, whose `env_root` is `/root/.letify-env`. When `env_root` is set:

1. The project directory is `<env_root>/project/<env key>`.
2. The sync sets no `UV_CACHE_DIR`, so uv uses its default cache under `~/.cache/uv` on the same disk and hard links from it.
3. No environment archive is packed or restored, as on any persistent provider.

The sandbox disk is discarded with the sandbox, so every Modal session syncs from the package index. Everything else under the workspace root, volumes, argument blobs and temporary files, stays on the volume.

### Interpreter version <!-- id: interpreter-version -->

> The runtime's `.venv` always runs the same Python major.minor as the local process, and `Env` guarantees it.

`Env.python` records the major.minor of the interpreter in the process that declares the `Env`, `sys.version_info[:2]`, and the runtime always runs `uv sync` with `--python <Env.python>`, whether or not the project has a `.python-version`. Without this a wide `requires-python` such as `>=3.11` lets uv pick a different minor version remotely, and cloudpickle's bytecode for a `__main__` function fails on it.

Two declarations cannot diverge from the local process. Before any session starts, a `.python-version` whose major.minor differs from the local interpreter, or an `Env.python` that differs from it, raises `InterpreterMismatch` naming both versions. A `.python-version` naming a patch release, such as `3.12.3`, is compared by its major.minor.

### uv on the runtime <!-- id: uv-on-runtime -->

> The runtime uses the uv it has, and installs uv under the home directory when it has none.

The worker looks for `uv` on `PATH`, then at `~/.local/bin/uv`. When neither exists it downloads `https://astral.sh/uv/install.sh` over HTTPS with the bootstrap interpreter's `urllib`, sending `User-Agent: letify/<version>` because astral.sh answers Python's default `Python-urllib` agent with 403, and runs it with `sh`, with `UV_INSTALL_DIR=~/.local/bin` and `UV_NO_MODIFY_PATH=1`. That is the same as `curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$HOME/.local/bin" UV_NO_MODIFY_PATH=1 sh`, without needing `curl`. It needs no root, because it writes only under the home directory. A failed download or a non-zero exit raises `EnvironmentFailure` saying `uv could not be installed on <runtime>` with the reason.

### uv cache <!-- id: uv-cache -->

> On a persistent provider the runtime's uv cache is `<workspace root>/uv-cache`, on the same filesystem as every project `.venv`, so a new env key is built from hard links. An ephemeral provider keeps uv's default cache.

uv installs a package into a `.venv` by hard linking it from its cache, and falls back to a full copy when the cache is on another filesystem. A container's home directory is often an overlay while the workspace root is a mounted disk, so the default cache under `~/.cache/uv` makes every new env key copy the whole environment.

The sync step sets `UV_CACHE_DIR=<workspace root>/uv-cache` for `uv sync` and `uv pip install` when the provider's `persistence` is `persistent` and it sets no `env_root`. On a runtime with a persistent workspace root the cache then outlives a container rebuild along with the projects built from it. An `Env.vars` entry naming `UV_CACHE_DIR` wins over this rule.

An ephemeral provider sets nothing. Its disk is discarded with the runtime, so a cache there is filled once per runtime wherever it lives, and moving it only matters when the home directory and the project are on different filesystems.

letify never deletes from the cache. The first sync on a runtime fills `<workspace root>/uv-cache` once, and a cache uv already had elsewhere is left in place. `uv cache prune` run with the same `UV_CACHE_DIR` removes entries no lock file needs any more; a file still hard linked from a `.venv` keeps its disk blocks until that `.venv` is removed as well.

### Interpreter check <!-- id: interpreter-check -->

> A worker whose Python major.minor differs from the local process fails the session start with both versions named, before any call is sent.

The worker reports `sys.version_info[:2]` in its ready line. After the environment step, the local side compares it with its own `sys.version_info[:2]`. A difference raises `InterpreterMismatch`, naming both versions and saying that cloudpickle's bytecode cannot run across them. It is not retried, because a fresh runtime builds the same interpreter. The check applies to every provider that builds its environment, and to an account that sets `python`. It does not apply to the local provider.

### Module shipping

> Modules in the lock file are installed remotely by name. Modules that are not travel with the call.

A package the lock file names is installed in the runtime and referenced by name. A package it does not name, such as the project's own code or an editable install, has to be sent by value, because the remote side either lacks it or holds an older copy. `Env.ship()` overrides the inference.

## Transport

> A `Shell` reaches its machine through a connection pipeline: several strategies are tried at once, the fastest acceptable one wins, and the winner is cached per account.

`Modal` and `Local` are not part of this. Modal is reached through the Modal adapter, described in [Modal adapter](#modal-adapter), with frames on a TCP connection through a Modal encrypted port as [Modal data channel](#modal-data-channel) describes, and Local starts its worker as a child process.

The measurements behind the order and the rules below are in [NETWORK.md](NETWORK.md#connection-pipeline-measurements).

### Connection strategies <!-- id: connection-strategies -->

> Four default strategies, ranked. A lower rank wins whenever it is acceptable.

| Rank | Strategy | Needs |
|---|---|---|
| 1 | Forward SSH to the machine's address | an address the user's machine can reach |
| 2 | TCP hole punching | a rendezvous, and NATs on both sides that allow a simultaneous open |
| 3 | UDP hole punching with Tailcat, then SSH over it | a rendezvous, and UDP in both directions |
| 4 | Provider fallback | the provider's own path, such as `colab exec` and the Colab file API |

Forward SSH is first because it needs no remote agent and costs one connection attempt, and when it works it is a kernel TCP connection. TCP hole punching comes before UDP because a punched TCP connection is also kernel TCP and is not subject to UDP rate limits. Tailcat comes next because its NAT traversal succeeds more often, but it runs in user space and a network that limits UDP limits it too. The provider fallback is last because it is the slowest.

A strategy whose needs are not met is skipped, not attempted. A `Shell` with no rendezvous has only rank 1. An account with no `address`, such as a Tunnel account that names only `tailcat`, has no rank 1: forward SSH is skipped with the reason `no address`, and no SSH command is built for a guessed or empty address. `letify check` on such an account runs over the strategies that remain.

An account reaches SSH on two ports when the machine publishes its SSH server under another port, such as a Docker container started with `-p 30501:8022`. `port` is the SSH server's port inside the machine, and the remote agent splices TCP punch and Tailcat connections to `127.0.0.1:<port>`. `public_port` is the port forward SSH dials at `address`. Forward SSH uses `public_port` when the account sets it, and otherwise `port`, read through `port_command` when that is set. TCP punching and Tailcat always use `port`.

Each stage is its own object behind a small interface, so a strategy can be added, removed or reordered by changing the subclass's list:

| Object | Owns |
|---|---|
| `Rendezvous` | sending a command to the remote side and exchanging addresses with it |
| `Strategy` | one way to connect: checking its needs, attempting, and returning a `Link` |
| `Link` | an established connection, used by both the worker channel and bulk transfer |
| `Probe` | a short measurement of a `Link` |
| `Pipeline` | running strategies, choosing one, and consulting the cache |
| `LinkCache` | remembering the winning strategy |

A `Shell` subclass differs from its parent only in its `Rendezvous` and its strategy list.

### Choosing a link <!-- id: choosing-a-link -->

> All applicable strategies start together. Among those that connect and pass the probe, the lowest rank is chosen, unless it is far slower than the fastest.

Strategies are raced rather than tried in turn, so a strategy that times out does not delay the others. Once the first strategy that can carry the probe connects, the pipeline waits a grace period of 2 s for lower ranked strategies before choosing.

A strategy that cannot carry the probe, such as the provider fallback, does not start the grace period when it connects. It is held back until every strategy that can carry the probe has failed, or until the race timeout of 60 s passes with none of them connected. A rendezvous that takes seconds, such as `colab exec` before a TCP punch, therefore runs inside its strategy's own attempt time, not inside the grace period.

Every connected strategy is probed: 30 round trips, then 2 s of transfer in each direction. A strategy whose throughput in either direction is below 25% of the fastest connected strategy in that direction is rejected. The lowest ranked strategy that remains is chosen. When only one strategy connects, it is chosen without comparison.

Strategies that lose are closed, including one that connects after the choice is made. Once the choice is made, every attempt still running is cancelled: the pipeline hands each attempt a cancel event and sets it, so a TCP punch stops waiting for its agreed start time and stops dialing within 0.1 s, and its attempt ends with a cancellation that is printed as cancelled rather than failed. An attempt that cannot observe the event, such as a provider call already in progress, runs to its end and is closed if it connects.

When only one strategy is applicable there is nothing to choose between, so it is used directly: it is not raced, not probed and not cached, and a failure surfaces at its first use. When a race of two or more ends with one connected strategy, that strategy is still probed so the cache has throughput to compare against, but a failed probe does not reject it.

A `Link` that cannot carry the probe, such as the provider fallback, is left out of the throughput comparison. It is chosen only when no probed strategy remains, and it is never written to the cache, so the next connection races again. A probe that fails on a strategy in a race of two or more rejects that strategy. When no strategy connects, the error names every strategy with the reason it failed or was skipped.

Every connection decision is printed, one line per event, on stderr with the `letify: ` prefix the session start line uses, so the user's stdout stays clean. The lines are on by default. `Launcher(announce=False)` silences them together with the session start line. The wording is not fixed, but each line names the account and carries these facts:

| Event | The line carries |
|---|---|
| Race start | the strategies attempted, each skipped strategy with its reason, and that the fallback is held back when it is applicable. Forward SSH is named with the address and port it dials, as `direct_ssh (<address>:<port>)` |
| Lone strategy | the one strategy used without a race, forward SSH with its address and port as above |
| Strategy outcome | connected with the seconds since the race started, failed with the exception text, timed out, or connected after the choice and closed |
| Probe | round trip in ms, upload and download in MiB/s, or the probe error |
| Choice | the chosen strategy and why: the lowest rank within 25% of the fastest, the only one connected, or the fallback because no probed strategy connected. Each strategy rejected by the 25% rule is named with its upload and download against the fastest |
| Switch | a strategy that connected first replaced by the chosen one, such as `tcp_punch` replacing `tailcat`, a link to the same account connected again after it was closed, with the old and new strategy, and a fall back to the provider's own path |

### Link cache <!-- id: link-cache -->

> The winning strategy is remembered per account and per network, so the next connection starts with it alone.

The cache lives in `~/.letify/accounts/<alias>/link.json`. It records the strategy, the probe results, and a network fingerprint: the local machine's public IP address and the name of its default route interface.

On the next connection the cached strategy is attempted alone. When it connects and its probe is at least 50% of the cached throughput in both directions, it is used. Otherwise, or when the fingerprint differs, the full race runs and the cache is rewritten. A fingerprint whose public IP address could not be learned matches nothing, so the race runs.

The cache's decisions are printed as the race's are, in [Choosing a link](#choosing-a-link): the cached strategy tried alone with its cached throughput, then accepted, or rejected with the reason (below 50% of the cached throughput with both numbers, failed to connect, probe failed, or the network fingerprint changed), and every rewrite of the cache with the strategy written.

The public IP address is learned from the same STUN servers the punch uses. The default route interface is read from `/proc/net/route` on Linux, `route -n get default` on macOS and `Get-NetRoute` on Windows.

### Rendezvous <!-- id: rendezvous -->

> Hole punching needs a way to start a program on the remote side and swap addresses. For Colab and Elice the provider layer fills that role; a plain machine behind NAT runs `letify client shell connect`.

| Provider | Rendezvous |
|---|---|
| `Colab` | the provider layer: `colab new` creates the runtime and `colab exec` runs the remote half |
| `Elice` | the provider layer: `eci` launches or starts the machine, and the remote half runs over forward SSH to its public IP |
| `Tunnel`, and a plain `Shell` behind NAT | the remote agent started by `letify client shell connect`, reached over Tailcat |

Colab and Elice never need `letify client shell connect`. Their create and open step is what puts letify's remote half on the machine.

`letify client shell connect` is run once on a plain machine by its user. The agent needs letify installed on that machine. Before starting anything it checks two things, and each failure prints what to do and exits 1:

1. `tailcat` is found by the lookup of [Installing external tools](#confirmed-tool-install). Otherwise it is installed automatically as that section describes, and when automatic install is turned off it prints the install command for the detected operating system and CPU architecture, for the Tailcat release pinned in `letify.transport.setup.TAILCAT_VERSION`: on Linux amd64, arm64 and armv7, `mkdir -p ~/.local/bin && curl -L <release tar.gz> | tar xz -C ~/.local/bin tailcat` with a note that `~/.local/bin` must be on `PATH`; on macOS, `brew install tailcat`; on Windows amd64 and arm64, the release zip and where to put `tailcat.exe`. Any other platform gets the releases page. The instructions end with `letify setup tailcat`.
2. An SSH server answers on `--ssh-port`, default 22: a TCP connection to `127.0.0.1` on that port must send a line starting with `SSH-` within 3 s. Otherwise it prints how to install and start one, for a Debian or Ubuntu container `apt-get install -y openssh-server`, `mkdir -p /run/sshd` and `/usr/sbin/sshd`.

It then starts the remote agent on a port the operating system chooses, starts `tailcat serve <agent port>` in front of it, and prints exactly one command for the user's own machine, `letify login tunnel <alias> --connect <token>`. The alias is `--name`, or this machine's host name with every character that is not a letter, digit or underscore replaced by `_`. The token is the URL-safe base64 encoding, without `=` padding, of the compact JSON object `{"tailcat": <address>, "tailcat_port": <agent port>, "user": <this machine's user name>, "port": <SSH port>}`. `--public-address` and `--public-port` add `"address"` and `"public_port"` to that object, for a machine whose SSH server is also reachable directly from outside under a published port. After the command it prints that the agent must keep running, how to keep it running with `tmux` or `nohup`, and that a restart prints a new address, so the login is run again with the new token.

A connection to the agent is told apart by its first bytes: `SSH-` is spliced to the machine's SSH server, and `LETIFY-RDV ` is followed by one JSON request line and answered with one JSON line. For such an account the pipeline connects over Tailcat first, runs `tailcat <address> <agent port>` to exchange the TCP punch mapping and start time over that link, and then races as specified: the Tailcat link is the rank 3 candidate, and when TCP punching passes the probe it takes over.

Both sides learn their public mapping from STUN servers reached over TCP on port 443, because networks that restrict outbound ports usually still allow 443. A punch starts at a time both sides agree on through the rendezvous. Each side connects from its bound port to the other's mapping and listens on the same port, so whichever direction's SYN arrives first completes the connection.

Every socket in a punch is bound with `SO_REUSEADDR`, and with `SO_REUSEPORT` where the platform has it, so the STUN connection, the listener and the connecting socket share one port. Both sides may complete a connection in each direction. The user's side takes the first connection that completes and writes a hello carrying a 16 byte token the two sides agreed on through the rendezvous. The remote side keeps the connection on which that hello arrives and closes the others.

The remote side's punch window is 15 s, or the request's `window` in seconds when it sets one. A punch that does not connect within it, or fails with a refused or unreachable connection, is expected, because the Tailcat link is already carrying the session. The agent then writes one line to stderr, `letify agent: TCP punch with <host>:<port> did not connect within <window> s; the Tailcat link is used instead`, or for a refused or unreachable peer `letify agent: TCP punch with <host>:<port> failed (<reason>); the Tailcat link is used instead`, and no traceback. Any other exception in the punch or the splice keeps its traceback.

Every SSH command letify builds sets `HostKeyAlias=letify-<alias>`, `StrictHostKeyChecking=accept-new` and `UserKnownHostsFile=~/.letify/accounts/<alias>/known_hosts`. Building the command creates that account directory with mode 0700 when it is missing, because `ssh` creates only `~/.ssh` and otherwise cannot record the key: it prints `Failed to add the host to the list of known hosts` on every connection and never pins the key. A host key that differs from the recorded one still fails the connection.

A punched connection first answers the probe, then on request is spliced to the remote machine's SSH server, and SSH runs over a local forwarding port. A second SSH connection over the same link punches again.

A local port that letify binds for forwarding is chosen by the operating system, never fixed, because Windows reserves port ranges that vary by machine. The remote end of a reverse forward is chosen the same way, with `ssh -R 0:`.

The remote half of a punch, a Tailcat listener or a reverse forward is one standard library Python module sent as source, so a Colab or Elice machine needs Python and nothing else. Through the provider layer it runs as a detached process that prints its answer and outlives the command.

Tailcat is run as `tailcat serve <port>` on the remote side, which prints `Server listening with new address: <address>`, and as `tailcat <address> <port>` in an SSH `ProxyCommand` on the user's side. It is applicable only when `tailcat` is on the user's PATH.

### Reverse SSH <!-- id: reverse-ssh -->

> An opt-in strategy for a remote machine that can reach the user's machine over SSH. It is not raced by default.

When an account sets `reverse_ssh`, the remote side opens an SSH connection to the user's machine and forwards a port back with `ssh -R`. It then takes rank 4 and the provider fallback moves to rank 5.

```toml
[lab_behind_nat]
kind = "shell"
reverse_ssh = { address = "home.example.com", port = 2222, user = "me" }
```

It is opt-in because it needs the user's machine to accept inbound SSH and needs a key on the remote side that can log in to it.

The key is per session. letify generates a new ed25519 key pair for each session and appends the public key to the user's `~/.ssh/authorized_keys` with `restrict,port-forwarding,command="/bin/false"` and the comment `letify-session <alias>`. The private key reaches the remote side through the rendezvous. Closing the link removes the line. Before a new session's key is added, every line carrying the `letify-session` marker for that alias is removed, so a crashed session never leaves a key behind.

### Colab <!-- id: colab-transport -->

> Colab is a `Shell` whose rendezvous is `colab exec`. It has no forward SSH, and its fallback is `colab exec` with the Colab file API.

The Colab CLI runs as `uv tool run --from google-colab-cli colab`, with `jupyter-kernel-client<1` pinned, because release 0.6.0 of the CLI calls an API that jupyter-kernel-client 1.0 removed. `colab new` and `colab stop` manage the session. `Colab.sessions()` reads `colab sessions` and returns the first word of each listing line. The CLI prints a session as `[<name>] <id> | Hardware: <hardware> | Variant: <variant>`, so a first word in square brackets gives the name inside them. A line starting with `[colab]` is a message from the CLI, such as `[colab] No active sessions found on server.`, and names no session, so an account with no session returns an empty list.

Colab limits outbound UDP to roughly 200 packets per second, so rank 3 is expected to lose the probe there. It stays in the list because the ratio rule removes it without a special case.

A `channel = "exec"` entry skips the pipeline and uses the fallback directly. A Colab VM has no SSH server on the port letify splices to, so the rendezvous request asks the remote half to start one first. The remote half connects to `127.0.0.1:<ssh_port>`, and when no line starting with `SSH-` arrives within 3 s it installs `openssh-server` if `/usr/sbin/sshd` is missing, creates `/run/sshd`, and starts `/usr/sbin/sshd -p <ssh_port> -o ListenAddress=127.0.0.1`. The port is explicit because a Colab image ships `sshd` configured for `127.0.0.1:2222`, so starting it with its own configuration leaves port 22 closed. An SSH server that already answers on the port is used as it is. The request carries the account's public key, `key` with `.pub` appended, which the remote side adds to `authorized_keys`. An account with no `key`, or whose `.pub` file is missing, has nothing the VM could authorize, so its Colab rendezvous is unavailable with the reason `no key`: `tcp_punch` and `tailcat` are skipped with that reason and the fallback carries the session. SSH over a punched or Tailcat link logs in as `root` unless `user` says otherwise.

The fallback sends calls with `colab exec` and bulk data through the Jupyter contents API that the Colab runtime proxy exposes: uploads are split into parts sent in parallel, each part in chunked `PUT` requests, and downloads read `/files/<path>` in parallel parts. The contents API root is `/` on the VM, not `/content`.

The proxy URL and token are the session's `url` and `token` in `.config/colab-cli/sessions.json` under the account directory, where the Colab CLI records them when `colab new` creates the session. A session that file does not name raises `RuntimeFailure` naming the file. Every request carries the token as the `colab-runtime-proxy-token` query parameter and the `X-Colab-Runtime-Proxy-Token` header, with `authuser=0`, as the CLI does.

An upload of a file to `<path>` sends parts of 32 MiB, eight at a time, each to `<path>.letify-part-<n>`. Eight is the most parallel parts [measured](NETWORK.md#colabs-own-paths), and it was the fastest in both directions. A part is sent as base64 chunks of 8 MiB: chunk `1` creates the file, later chunks append, and the last chunk of a part of more than one chunk is numbered `-1`. One `colab exec` program then joins the parts into `<path>` in order, removes them, and unpacks the archive when the request asks for it. A download reads the file's size from the contents model with `content=0`, then reads `/files/<path>` with `Range` requests of 32 MiB, eight at a time. `put_file`, `get_file` and `pack_dir` on the fallback channel use this path; `pack_dir` packs into a temporary file under `<workspace root>/tmp` with `colab exec` first. Any other status than 2xx raises `RuntimeFailure` naming the path and the status.

## Configuration

> Accounts live in the home file, project defaults live in the repository file, secrets live in neither.

letify keeps its state in two `.letify` directories. `~/.letify/` belongs to the machine and is never in a repository. `<project>/.letify/` belongs to the repository and is committed.

| Path | Holds |
|---|---|
| `~/.letify/config.toml` | Every account this machine has: kind and connection details, never a secret |
| `~/.letify/accounts/<alias>/` | That account's credentials and provider state, one directory per alias, owner only |
| `<project>/.letify/config.toml` | Project defaults, and the aliases of the accounts the project uses |

The project directory is `.letify/` in the working directory. `Launcher(config=...)` and `--config` name another `.letify` directory, or a `config.toml` directly.

### The two files <!-- id: two-files -->

> The home file is the set of accounts this machine has. The project file chooses from it and may override it. An account the project does not name is not available in that project.

An account in `~/.letify` is available in a project in exactly three cases:

1. **The project file names it.** A table with the same alias, even an empty one, is enough: `[colab_pro]` on its own makes the home account `colab_pro` available with all of its settings.
2. **The home entry is global.** `global = true` in the home entry makes the account available in every project, including one with no `.letify` at all. It is how an account meant for everything, such as a personal Colab, avoids being named in every repository.
3. **It is `local`.** The local machine is always available and never needs a declaration.

Every other home account does not exist as far as that project is concerned: `let.providers.<alias>` raises `UnknownProvider`, it is absent from `let.providers.aliases`, and `let.providers.any` never resolves to it.

When the project file names an account, its fields override the home entry's field by field. Fields the project does not set come from the home entry, including `kind`. A project table that sets `kind` itself is a complete declaration and needs no home entry, which is how an account with no connection details, such as a second `local`, is declared in the repository. A project table with no `kind` whose alias the home file does not have is a configuration error naming `letify login`, because there is nothing to take the kind from.

`global` is read from the home file only. A project cannot make an account global, because a repository deciding what every other repository on the machine can reach is the wrong direction.

An alias must be a Python identifier, because providers are reached by attribute access. `any`, `devices` and `active` are reserved. Declaration order sets the priority for `let.providers.any`.

A credential never appears in either `config.toml`. A field such as `access_token` is resolved in this order: the environment variable named by `access_token_env`, then the file `access_token` in `~/.letify/accounts/<alias>/`, then a literal value, which is accepted only so a home entry can carry a non secret default. A credential file is created with owner only permissions where the platform has them. Provider tools that keep their own login, the Colab CLI and the Modal SDK, keep it in the same account directory rather than in their usual location, which is what lets two accounts of one provider exist on one machine.

The OS keyring is not used. Reading it needs a package in the user's environment, and the provider tools that matter already store their tokens as files, so one mechanism covers every provider.

```toml
[colab_a]
kind = "colab"
account = "someone@example.com"

# Which accelerators this account can get, and how many at once. Colab assigns the device
# itself, so there are no indices to name.
[colab_a.devices]
G4 = { count = 2 }
T4 = { count = 2 }

[lab_a100]
kind = "shell"
address = "gpu.lab.example.edu"
user = "researcher"
key = "~/.ssh/id_ed25519"
persistent = true
workspace = "/workspace/researcher/letify"   # this server allows writes under /workspace only

# Eight cards in the box, four of them ours. letify takes only those of these four that are
# actually free when a session starts, so a card a colleague is computing on is skipped.
[lab_a100.devices]
A100 = { indices = "0-3" }

[elice_a100]
kind = "elice"
zone_id = "00000000-0000-0000-0000-000000000000"
price_type = "spot"            # ondemand by default; the machine letify-elice-a100-spot is created on first use
access_token_env = "ELICE_ACCESS_TOKEN"
```

The older `gpus = ["A100", "H100"]` list still works and means one of each, with letify choosing no indices. `devices` is what an entry uses once a count or an index range matters.

### Workspace root <!-- id: workspace-root -->

> Everything letify writes on a remote machine lives under one directory per account, the workspace root. An account sets it with `workspace`; otherwise its kind decides.

A machine sets rules on where an account may write. A department GPU server may allow writes only under `/workspace`, a Modal sandbox loses its disk when it ends while a Modal volume persists, and a Colab VM is discarded after the session. So the root is a property of one account on one machine.

`workspace = "<path>"` is set on an account entry in `~/.letify/config.toml`. The path starts with `/` or `~`, and `~` is expanded on the runtime, not on the local machine. Any other value raises `ConfigError` naming the account. A `workspace` field in a project `.letify/config.toml` raises `ConfigError` saying that it belongs in the home file, because a repository cannot know the rules of each machine its users reach.

When `workspace` is not set, the root is:

| Kind | Workspace root |
|---|---|
| `shell`, `tunnel`, `elice` | `~/.letify-runtime` |
| `colab` | `/content/letify` |
| `modal` | `/letify`, where a Modal volume named `<app>-workspace` is mounted in every sandbox, so the root persists across sandboxes |
| `local` | not used. A volume materialized on `local` without `mount` lands under `~/.letify-runtime` on this machine |

A `modal` account with `workspace` set mounts the same volume at that path instead, when the path starts with `/`.

Everything letify writes on the runtime is under the root:

| Path | Holds |
|---|---|
| `<workspace root>/project/<env key>` | the project files `uv sync` reads, and the `.venv` it builds, except on `modal`, as Environment on the sandbox disk describes |
| `<workspace root>/project/.<digest>.tar.gz` | an environment archive while it is unpacked, removed once the `.venv` starts |
| `<workspace root>/uv-cache` | uv's cache on a persistent provider, as uv cache describes |
| `<workspace root>/volumes/<volume name>` | a volume's materialized blobs and project data |
| `<workspace root>/blobs` | argument blobs on a persistent provider, as Argument blobs on a persistent disk describes |
| `<workspace root>/tmp` | temporary files, including the archive `pack_dir` builds on a one-shot channel; `TMPDIR` points here |

The worker's working directory is the root, so a relative path in user code resolves under it. The uv installer is the one exception to the root: uv goes to `~/.local/bin`, as uv on the runtime describes, because it is shared by every account on that home directory.

### Generated provider types

> Loading the configuration writes a type stub that names this project's accounts and their accelerators, so an editor completes `let.providers.colab_pro.G4` and flags a misspelled alias.

Aliases live in `.letify/config.toml`, not in code, so a type checker cannot know them. `Launcher()` therefore writes `letify_providers.pyi` describing the accounts this project can use under the two file rule above. `Launcher.providers` is typed as `letify_providers.ProvidersView`, and letify ships a `letify_providers` module whose `ProvidersView` is the plain `Providers`, so a project with no generated file keeps exactly today's types.

Each alias becomes a class named after it in CamelCase, `colab_pro` as `ColabPro` and `lab_a100` as `LabA100`, subclassing the provider class of its kind. A name that is a Python keyword gets `Provider` appended, and a name two aliases would share gets a number appended in declaration order. The class declares one `Instance` attribute per accelerator the account offers, and `ProvidersView` declares one attribute per alias. Where the accelerators are known, the class declares no fallback attribute lookup, so a misspelled accelerator is a type error; where they are not, attribute access stays typed as `Instance`.

Accelerators are taken from what can be known without a network call: the entry's `devices` table or `gpus` list, or the provider's fixed list for Colab and Modal, or this machine's own cards for `local`. Writing the stub never connects to a machine or an API. An alias whose provider cannot be built is typed as the plain `Provider`.

The file goes in a `typings` directory at the project root, which is Pyright's and Pylance's default stub path. The project root is the nearest directory upward from the working directory that holds a `pyproject.toml`, or the working directory when there is none. `[tool.letify] typings = "<path>"` in that `pyproject.toml` moves it, relative to the root, and the type checker's stub path has to point at the same place. `typings = false` turns generation off, and so does the environment variable `LETIFY_STUBS=0`.

The file is rewritten only when its content would change, so loading the configuration does not touch it on every run. `letify stubs` writes it on demand. It reflects one machine's `~/.letify/`, so it belongs in the project's `.gitignore`.

### Logging in

> One command writes the account to the home file and a reference to it in the project file, so a repository names the accounts it needs without holding any of them.

`letify login <kind> [alias]` declares one account. It writes two entries in two files, because the two files answer different questions.

`~/.letify/config.toml` gets the account: the address, the user, the key path, the zone, whatever that kind of provider needs to connect. It belongs to the machine and is never in a repository, so it is where a connection detail may live. A credential the login collects goes to `~/.letify/accounts/<alias>/`, created with owner only permissions where the platform has them.

Every kind except `local` takes `--workspace PATH`, which writes `workspace` to the account. `letify login shell` and `letify login tunnel` also ask for it, after the BatchMode key check and before devices are detected, with the kind's default in brackets: `Workspace root on <address> [~/.letify-runtime]: `. A blank answer, or `--no-input` without `--workspace`, takes the default and writes no field. The chosen path is then checked over the same BatchMode SSH options: `mkdir -p` on the path, then a write and removal of the probe file `<path>/.letify-probe`, as the account's own user, so a path that needs root fails. A failure writes nothing to either file and raises `LoginError` naming the path, the machine and the error the command printed. A password account has no BatchMode connection, so its workspace is recorded without the check. `colab`, `modal` and `elice` record `--workspace` without a remote check, because there is no machine to reach at login.

An account that is already in the home file is checked again only when `--workspace` is given. The check then runs with the connection details the home file holds, and the field is written when it passes.

`letify check <alias>` on a `shell`, `tunnel`, `colab` or `elice` account runs the same probe on the account's workspace root in the command that confirms the machine answers, and prints `workspace <path>: writable` or `workspace <path>: not writable: <error>`.

The project's `.letify/config.toml` gets the alias as an empty table, `[colab_pro]`. Nothing else, because everything else is either a secret or a detail of one person's machine, and naming the alias is what makes the account available in the project. That table is what makes the repository self describing: a teammate who clones it can run `letify login` for the aliases it names and nothing else has to be explained. A named alias the home file does not declare is a configuration error naming the command that fixes it.

An account that is already in the home file is not asked for again. `letify login lab` in a second repository writes only the reference, which is the common case: the account was set up once and every project since then just needs to name it.

`letify logout <alias>` removes the account from `~/.letify/config.toml` and deletes `~/.letify/accounts/<alias>/` with everything in it. It leaves the project reference alone, because the repository still needs that account; what changed is only that this machine no longer has it.

`letify login colab <alias>` signs in to Colab itself. It runs `colab sessions` through `uv tool run --python 3.13 --from google-colab-cli colab`, with `HOME` set to `~/.letify/accounts/<alias>/`. The Colab CLI keeps its token at a fixed path under its home directory, so the token lands in the account directory and the CLI refreshes it on later calls. Every later Colab command runs with the same `HOME`, which is what lets two Colab accounts live on one machine. uv's cache, Python installs and tools stay pinned to the real home, so a changed `HOME` downloads nothing again. A sign in that exits non zero writes nothing.

After the sign in succeeds, the Colab login records `key`, the SSH private key whose public half the Colab rendezvous installs on each runtime at connect time, as described under [Colab](#colab-transport). The key is `--key` when given, and `~/.ssh/id_letify` otherwise, the same key a `shell` login generates. The key lives under `~/.ssh` rather than in the account directory because it is not a Colab credential: it authorizes only the throwaway VM, and `letify logout` must not delete a key other accounts use. A missing key is generated as an ed25519 pair with no passphrase. An existing key is used as it is and never regenerated. A Colab account already in the home file with no `key` is the one exception to not being asked for again: `letify login colab <alias>` ensures the key the same way and adds `key` to the entry, leaving its other fields alone, without signing in again.

`letify login modal <alias>` signs in to Modal itself. It first asks for an optional Modal profile, which names the Modal workspace to sign in to. It then runs `modal token new` through `uv tool run --python 3.12 --with "modal>=1.0,<2" --from modal modal`, with `MODAL_CONFIG_PATH` set to `~/.letify/accounts/<alias>/modal.toml` and, when a profile was given, `--profile <profile>`. The profile is written as `profile`, because `workspace` is the workspace root. Modal's command prints a link and waits for the browser approval, so `modal` never has to be on `PATH` or in the project's environment. The token lands in the account directory, and the adapter reads it from there. A sign in that exits non zero, or exits zero without writing `modal.toml`, writes nothing to either `config.toml` and removes a `modal.toml` the attempt created.

`letify login elice <alias>` checks the access token with `eci` before it writes anything, and needs no machine to exist. Every `eci` command runs as [Elice machines](#elice-machines) describes. The steps run in this order, and a failure at any step writes nothing: no `config.toml` entry and no file in the account directory.

1. `eci` must be found as Elice machines describes. Otherwise it is installed automatically, and when automatic install is turned off `LoginError` carries the install command.
2. The token comes from `--token`, or else from a hidden prompt, `Elice access token: `. `--no-input` without `--token` refuses.
3. `eci zone list --format json`. `endpoint` is `--endpoint` or `https://portal.elice.cloud/api`. A non zero exit fails the login with `LoginError`, whose message starts with `Elice refused the access token`.
4. The zone is `--zone-id`. Without it, a terminal is shown the listed zones, one numbered line each, `1. <name> (<id>)`, and asked `Elice zone [1-<n>]: `. A blank answer takes the only zone when there is exactly one. An answer that is not a listed number is refused and asked again. `--no-input` without `--zone-id` refuses, and so does an empty list.
5. `eci config verify` with that zone. A non zero exit fails the login with `LoginError` naming its output.
6. The machine is `--machine-id` when given. Otherwise `eci compute vm list --format json` is read. With no machines listed, nothing is asked and no machine is recorded, and the login prints `Elice lists no machine; letify creates one on first use.` With machines listed, a terminal is shown them as `1. <name> (<id>)` followed by `<n+1>. Create a new machine with letify`, and asked `Elice machine [1-<n+1>]: `. A listed machine is recorded as `machine_id`; the last choice records none. `--no-input` without `--machine-id` records none.
7. `price_type` is `--price-type` and is written only when given.
8. `organization` is `--organization`, or else the `name_short` of `eci org info --format json`. It is written only when one of them gives a value.
9. `billing_endpoint` is `--billing-endpoint`, or else a terminal is asked `Elice billing API base URL (blank to skip): `. It is written only when given.
10. `key` is `--key` or `~/.ssh/id_letify`, generated as for Colab when missing.
11. The token is written to `~/.letify/accounts/<alias>/access_token` with mode 0600.

The account is written with `kind = "elice"`, `zone_id`, `key`, `machine_id` and `price_type` when chosen, and `endpoint` only when it is not the default.

`letify login tunnel <alias> --connect <token>` declares a machine behind NAT from the command `letify client shell connect` printed on it. Without `--connect`, a terminal is asked `Token printed by 'letify client shell connect': `, and `--no-input` refuses. The steps run in this order, and a failure at any step writes nothing to either file and raises `LoginError` whose message starts with `tunnel login failed at <step>: `:

1. `tailcat`: `tailcat` must be found by the lookup of Installing external tools, and every SSH command runs it by the path found. Otherwise it is installed automatically, and when automatic install is turned off the message is the same install instructions `letify client shell connect` prints, for this machine's operating system and architecture.
2. `token`: the token is decoded. A token that is not the base64 JSON described under Rendezvous, or that lacks `tailcat` or `tailcat_port`, is refused.
3. `key install`: the key is generated and installed as SSH authentication describes, over SSH with `-o ProxyCommand=tailcat <address> <agent port>`, logging in as the token's `user` on the token's `port`. `--skip-key-install` and `--key` work as for `shell`.
4. `key confirmation`: the key is confirmed with `BatchMode=yes` over the same `ProxyCommand`.
5. `workspace`: the workspace root is chosen and checked as for `shell`, over the same `ProxyCommand`.
6. `devices`: the GPUs are recorded as Recording devices at login describes, over the same `ProxyCommand`.

The account is written with `kind = "tunnel"`, `tailcat`, `tailcat_port`, `user`, `port` and `key` from the token and the options. It has no `address` unless the token or `--address` gives one, and `public_port` is written when the token or `--public-port` gives it. A value in the token wins over the option. Every login step still runs over Tailcat.

Credentials never enter either `config.toml`. A token goes to a file in the account directory. An SSH password is never stored at all, which the next section explains.

### Recording devices at login <!-- id: login-records-devices -->

> `letify login shell` and `letify login tunnel` ask the machine for its GPUs once, right after the key is confirmed, and write the ones the user chose to `[<alias>.devices]` in `~/.letify/config.toml`.

Detection happens at login and not at first use, for two reasons. The generated provider types read the `devices` table without a network call, so an editor completes the accelerator names only once the table exists. And on a shared machine the table is where the user says which cards are theirs, so asking at login is what makes the first session use those cards and no others.

1. After the key is confirmed with `BatchMode=yes`, the same SSH connection options run `nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader` once. A password account has no BatchMode check to follow, so it is not detected.
2. Each name is normalized with the accelerator name normalizer, and the cards are grouped by that name with their physical indices kept.
3. With a terminal, each group is shown as one line, `A100: 4 cards, indices 0-3 (80 GB each)`, and the user is asked `Indices letify may use for A100 [0-3]: `. A blank answer takes every card found. An answer takes the forms the inventory takes, `0-3` or `0,1,6`. An answer naming an index that was not found, or that cannot be read, is refused with the reason and asked again.
4. With `--no-input`, `--indices NAME=SPEC` chooses the indices for one name and may be repeated. A name without the option gets every card found. An option naming an accelerator that was not found, or an index that was not found, fails the login with nothing written.
5. The result is written as `[<alias>.devices]` with one `NAME = { indices = "..." }` line per name, through the same textual writer as the account, so every other table and comment in the file stays as it was. A contiguous choice is written as a range, `"0-3"`, and any other choice as a list, `"0,1,6"`.
6. When `nvidia-smi` is missing, exits non zero, or lists no GPU, the login still succeeds. Nothing is recorded, a note says why, and the accelerators are discovered on first access.
7. An account already in the home file is not detected again. `--detect-devices` asks the machine again with the connection details the home file holds, shows what it found, and replaces the table after the user confirms with `y`. With `--no-input` the flag itself is the confirmation. When nothing is found, the existing table is left alone.
8. After a table is written, the provider types are regenerated as `Launcher()` does, so completion picks up the names at once.

`letify logout <alias>` removes `[<alias>.devices]` together with `[<alias>]`, because a devices table without its account would declare an account with no kind.

### SSH authentication

> Key authentication, because the call path is non-interactive. A password is accepted once, to install the key, and then discarded.

letify opens sessions with `ssh -o BatchMode=yes`. That is not a preference: a session is started by the pool, in the background, possibly long after the call that needed it, so there is nobody present to answer a password prompt. A transport that requires interaction cannot carry a pooled session.

So `letify login shell` sets up key authentication and treats the password as a one-time input:

1. If the configured key does not exist, an ed25519 key is generated at `~/.ssh/id_letify` with no passphrase, because a passphrase would put the prompt back.
2. The public key is appended to the machine's `~/.ssh/authorized_keys`, over one interactive SSH connection that asks for the password in the terminal.
3. The password is used by that one command and then dropped. It is not written to a file or to the environment.
4. The connection is confirmed with `BatchMode=yes`, which proves the key works before the alias is declared rather than at the first call.
5. The machine's GPUs are detected over that confirmed connection, as described under Recording devices at login.

Every SSH command letify builds also carries `-o ControlMaster=auto`, `-o ControlPersist=60` and `-o ControlPath=<control directory>/<tag>-%C`, so the worker channel, the busy card check and every later command to the same machine share one authenticated connection, instead of paying about 0.24 s for a new one each. On Windows, where OpenSSH does not implement connection multiplexing, the three options are left out and each command opens its own connection.

Control sockets live in a short per-user directory, not under the home directory, because a Unix domain socket path is limited to 108 bytes on Linux and 104 bytes on macOS, including the terminating byte:

- The control directory is `$XDG_RUNTIME_DIR/letify` when `XDG_RUNTIME_DIR` is set, and `/tmp/letify-<uid>` otherwise. It is created with mode 0700.
- Before use, the directory is checked with `lstat`: it must be a directory, not a symbolic link, owned by the current user, and have no group or other permission bits. Any other state raises `ConfigError` and no SSH command is built.
- `<tag>` is the first 8 hex digits of the SHA-256 of the account alias, so accounts that reach the same machine keep separate connections. `%C` expands to 40 hex digits.
- The length checked is the path with `%C` expanded, plus the 17 bytes OpenSSH appends to name the temporary socket while it binds. If that reaches the platform limit, the three options are left out for that command and it opens its own connection. Every command also carries `-o Ciphers=^aes128-gcm@openssh.com,chacha20-poly1305@openssh.com`, which puts those two ciphers first in the client's default list, so a server that offers neither still connects. Compression stays off.

One other approach is not the default. `sshpass` feeds a stored password to each connection, which needs the password kept somewhere and exposes it in the process arguments of every call. `sshpass` is available as `auth = "password"` for a machine whose administrator forbids key authentication, reading the password from `~/.letify/accounts/<alias>/password`, and it refuses on Windows, where the tool does not exist.

### What each kind asks for

> Where a vendor owns the credential format, letify runs the vendor's own sign in and points it at the account directory.

| Kind | Written to the home file | Credential |
|---|---|---|
| `shell` | address, user, port, key path, `workspace` when it is not the default, and the `devices` table the machine reported | an SSH key, installed by `login`; no password stored |
| `tunnel` | `tailcat`, `tailcat_port`, user and port from the token `letify client shell connect` printed, key path, `workspace` when it is not the default, and the `devices` table; `address` and `public_port` only when the token or the options give them | an SSH key, installed by `login` over Tailcat; no password stored |
| `elice` | endpoint when not the default, zone, key path, and `machine_id`, `price_type` and `workspace` when given | access token in `~/.letify/accounts/<alias>/access_token`; the generated machine password in `machine_password` once letify launches a machine |
| `colab` | account email, `workspace` when given | the Colab CLI's token, written by its own sign in under `~/.letify/accounts/<alias>/` |
| `modal` | `profile` and `workspace`, each when given | Modal's token, written by `modal token new` to `~/.letify/accounts/<alias>/modal.toml` |
| `local` | nothing | none; this machine needs no declaration |

For `colab` and `modal`, letify runs the vendor's sign in through uv and does not parse or refresh the token. The vendor's client reads and refreshes it from the account directory.

### Interpreter override <!-- id: python-option -->

> `python` on an account names the interpreter the worker runs with. Setting it means the user manages that interpreter, so letify does not build the environment there.

Without `python`, a `shell`, `tunnel`, `colab` or `elice` account starts its bootstrap worker with `python3` and then runs the worker from the project `.venv`, as Building the environment on a runtime describes. With `python = "/path/to/python"`, the worker is started with that interpreter and stays on it: no project files are sent, no uv runs and no environment archive is read or written. That interpreter has to provide cloudpickle: letify installs nothing into it, and a session start on one that lacks it raises `ConfigError` naming the interpreter and `cloudpickle`. `ConfigError` is not retried, because a fresh runtime has the same interpreter. The interpreter check still applies. On `local`, `python` names the interpreter of the worker subprocess, which defaults to the interpreter running letify.

## PyTorch forwarding

> `host="local"` runs the user's PyTorch code here and executes its operators on the runtime's GPU, through PyTorch's own `__torch_dispatch__` extension point.

Six modules, all in `letify/remoting/device/`:

| Module | Holds |
|---|---|
| `tensor.py` | `RemoteTensor`, the local stand-in for a tensor on the runtime, and operator dispatch |
| `cuda.py` | The mapping of `"cuda"` onto the runtime's device and the `torch.cuda` functions letify provides |
| `client.py` | The operator queue, handles, synchronization and error reporting |
| `trace.py` | Step capture: the trace, detection, and matching a repetition against a step |
| `frames.py` | The `Transport` interface, its stream implementation and its channel implementation |
| `executor.py` | The worker that runs operators and steps on real tensors keyed by handle |

`frames.py` and `executor.py` import the standard library, `wire.py` and PyTorch only, because their source is sent to the runtime ahead of the executor start, where letify may be absent or older.

### Dispatch mechanism <!-- id: dispatch-mechanism -->

> A `RemoteTensor` is a wrapper subclass made with `torch.Tensor._make_wrapper_subclass` on the `meta` device, holding a handle to a tensor on the runtime and a local meta tensor.

The meta tensor carries shape, dtype, strides and storage offset, so every operator's output metadata is computed locally by running the same ATen operator on the meta tensors. Shape inference therefore never waits for the runtime.

PrivateUse1 is not used. On torch 2.5.1 a wrapper tensor on a device renamed through `torch.utils.rename_privateuse1_backend` aborts the process in autograd, because no device guard can be registered from Python. On torch 2.14.0 `torch.utils.backend_registration._setup_privateuseone_for_python_backend` registers one, and backward still fails an internal assert in the autograd engine's device queues. A wrapper subclass reporting `cuda` aborts the same way on a CPU build. The meta wrapper runs forward and backward on both versions. The probe is in the pull request.

The `meta` device is an implementation detail. The `TorchFunctionMode` of [Mapping cuda](#mapping-cuda) answers `device` as `cuda:0`, `is_cuda` as `True` and `get_device()` as `0` for a `RemoteTensor`, which is what code written for CUDA reads. `RemoteTensor` disables its own `__torch_function__`, so a tensor method enters Python once, in the mode, instead of twice.

Autograd runs locally. Backward operators and optimizer steps reach `__torch_dispatch__` like forward ones, so they are queued and executed on the runtime the same way. A tensor that requires grad on the runtime never exists: the runtime holds values only.

Inferred metadata is cached per operator. Every `RemoteTensor` carries its signature, the shape, strides, storage offset and dtype, and builds its meta tensor only when an inference needs one. One pass over an operator's arguments reads three things:

| Part | Holds |
|---|---|
| structure | the overload, the shape, strides and dtype of every tensor argument, the type of every `int` and `float` argument, and the value of every other argument |
| offsets | the storage offset of every tensor argument, in order |
| scalars | the values of the `int` and `float` arguments, in order |
| handles | the handles of the `RemoteTensor` arguments, in order |

A batch slice taken at a new position each step changes only its offset, so it keeps its structure. The metadata key is the structure plus the offsets and the scalars, except that a float argument of a `_foreach_` operator is left out, because an optimizer passes per step values such as bias corrections there and they never change output metadata. The offsets are left out too for an operator none of whose schema returns carries alias information, because such an operator returns new tensors whose metadata does not depend on where its inputs start in their storage. A batch at a new position therefore reuses the metadata of `addmm`, `mm` or `mse_loss` inferred at an earlier position, and only its view operators, such as `slice` and `t`, are inferred again. A hit builds the outputs with `_make_wrapper_subclass` from the recorded signatures and returns an input where the first inference returned that input, so a training step that repeats its operators runs each meta kernel once. An operator whose arguments include a value that cannot be a key, such as a generator, is inferred every time.

While forwarding is active, `RemoteTensor` is added to every `_foreach_supported_types` list PyTorch keeps, which in 2.5 is one in `torch.optim.optimizer` and one in `torch.utils._foreach_utils`, so an optimizer that picks its foreach path for CUDA tensors picks it here too and a step issues one operator per tensor list instead of one per parameter. A PyTorch without such a list keeps the per parameter path.

`aten.detach` and `aten.alias` are recognized before the arguments are read, and produce a new `RemoteTensor` sharing the same handle, with no operator sent. An in-place operator, or one writing to `out=`, returns the input it wrote to. Every other operator output gets a new handle.

An in-place operator that changes the view metadata of the input it returns, such as `as_strided_`, which the composite `adaptive_avg_pool2d` calls to give a `channels_last` result `channels_last` strides, is mirrored on that input's wrapper: its signature, and the wrapper's own shape, strides and storage offset, take the meta result's values before the operator returns. Only that wrapper changes, as in PyTorch, and other `RemoteTensor`s sharing its handle keep their metadata. The metadata cache records the new signature, so a cache hit and a replayed step apply the same change. Memory formats need nothing else: strides are part of every signature, so a `channels_last` tensor keeps its strides through inference and the runtime computes on tensors with the same strides.

A plain CPU tensor passed to an operator travels with it as a buffer and is a CPU tensor on the runtime, so a zero-dimensional CPU scalar mixes with device tensors as it does in PyTorch. A CPU tensor larger than 4 KiB flushes the queue immediately after its operator, so a later write to it in this process cannot change what the runtime received.

When the meta operator raises, the operator is sent at once and executed on the runtime, and the reply carries its output metadata. That covers data-dependent shapes such as `nonzero` and `masked_select`, and reports a genuine error with the runtime's own message.

### Kernel selection <!-- id: forwarding-kernel-selection -->

> `batch_norm` and `scaled_dot_product_attention` on a `RemoteTensor` run the kernel the runtime's own CUDA dispatch chooses, not the one a meta tensor chooses.

PyTorch picks the kernel for these two functions from the device when it dispatches. A meta tensor gets `native_batch_norm` and math attention. A CUDA tensor gets cuDNN batch norm, and flash, memory-efficient or cuDNN attention depending on shape, dtype and card. Those kernels give different values and use different memory.

When the executor's hello reports a CUDA device, the `TorchFunctionMode` of [Mapping cuda](#mapping-cuda) asks the executor which backend applies, with a `letify.kernel` request, once per distinct signature, and caches the answer in the client. A signature is the function and, for every tensor argument, its shape, strides, dtype and whether it is None, plus `training` and `eps` for batch norm and `dropout_p`, `is_causal`, `scale` and `enable_gqa` for attention. The executor answers by calling `torch._C._select_batch_norm_backend` or `torch._fused_sdp_choice` on empty tensors of that signature on its own device. A PyTorch without the selector answers `Native` or `MATH`.

Attention with `enable_gqa`, and flash attention for a head dimension that is not a multiple of 8, keep the ordinary path, because PyTorch reshapes or pads those before its fused kernel.

The mode then calls the chosen ATen operator directly: `aten.cudnn_batch_norm` for `Cudnn`, and `aten._scaled_dot_product_flash_attention`, `aten._scaled_dot_product_efficient_attention` or `aten._scaled_dot_product_cudnn_attention` for the attention backends. Each of them has a meta kernel that infers its outputs and an autograd formula that records its backward, so it is dispatched and forwarded like any other operator. The function returns the operator's first output, the normalized or attended tensor, and `cudnn_batch_norm` updates the running statistics in place as `batch_norm` does.

The ordinary path applies when the executor's device is CPU, when the runtime answers `Native` or math attention, or when an argument is not a `RemoteTensor`.

### Mapping cuda <!-- id: mapping-cuda -->

> Code written with `"cuda"`, `.cuda()` and `torch.cuda.is_available()` runs unchanged under `host="local"`.

While the declared function runs, a `TorchFunctionMode` rewrites a CUDA device in the `device` keyword of any torch function, and the positional device of `Tensor.to` and `Tensor.cuda`, to the runtime's device. A `cuda:N` with `N` other than 0 raises `UnsupportedMode`, because one session forwards to one device. A factory call, such as `torch.randn(..., device="cuda")`, runs on the runtime and its values are generated there.

`Module.cuda()` and `Module.to("cuda")` work, because they call `Tensor.cuda` and `Tensor.to` for each parameter. Assigning `tensor.data` between two `RemoteTensor`s moves the handle with the metadata.

These `torch.cuda` functions are replaced while the function runs, and restored afterwards:

| Function | Behaviour |
|---|---|
| `is_available()`, `is_initialized()` | `True` |
| `init()` | Nothing |
| `device_count()` | `1` |
| `current_device()` | `0` |
| `set_device(d)`, `device(d)` | Accepted for device 0, `UnsupportedMode` otherwise |
| `get_device_name(d=None)` | The runtime's device name |
| `get_device_properties(d=None)` | An object with the runtime's device `name`, `major`, `minor` and `total_memory` in bytes, read from the executor's hello |
| `get_device_capability(d=None)` | `(major, minor)` of the runtime's device, from the same hello, `(0, 0)` for a CPU executor |
| `is_current_stream_capturing()` | `False`, because a CUDA graph cannot be captured under `host="local"` |
| `synchronize(d=None)` | Flushes the queue and waits for the runtime, which surfaces a pending error |
| `manual_seed(s)`, `manual_seed_all(s)` | Seeds the runtime's generator for its device |
| `memory_allocated()`, `max_memory_allocated()`, `memory_reserved()` | The runtime's value, one round trip |
| `empty_cache()` | Queued and executed on the runtime |

`Stream`, `Event`, `current_stream`, `stream`, `CUDAGraph`, `graph`, `get_rng_state` and `set_rng_state` raise `UnsupportedMode` naming the function, because a stream, an event, a graph or a generator state lives in the runtime's process and has no local counterpart here. Every other `torch.cuda` attribute is PyTorch's own and behaves as it does on a machine without CUDA.

No replaced function initializes CUDA in this process. A CUDA build of PyTorch on a machine with no NVIDIA driver raises `CUDA driver version is insufficient` from any call that does, and a training loop makes such calls without naming them: `Adam.step()` and `AdamW.step()` call `is_current_stream_capturing()`, and `torch.cuda.is_bf16_supported()`, which `autocast` reads for `bfloat16`, calls `get_device_properties()`.

A process forked while forwarding is active, such as a `DataLoader` worker, starts with the mapping undone: `torch.cuda` holds PyTorch's own functions, the device rewrite is off and `current_client()` is None, so the worker's `torch.manual_seed` and its CPU tensors stay in that process. The client refuses to send from a process other than the one that connected it, raising `RuntimeLost` naming the fork, because the channel it would write to belongs to the parent.

### Pinned memory <!-- id: forwarding-pinned-memory -->

> Under `host="local"`, pinning host memory is a copy into ordinary memory that reports itself as pinned, so `DataLoader(pin_memory=True)` runs unchanged and never initializes CUDA in this process.

Pinned memory exists so the CUDA driver can copy to the device asynchronously. No driver runs in this process, and an upload is already asynchronous, as [Transfers](#forwarding-transfers) describes, so pinning has nothing to speed up. While forwarding is active:

| Function | Behaviour |
|---|---|
| `Tensor.pin_memory(device=None)` | A copy of the tensor in ordinary CPU memory, with the same shape, strides, dtype and values. A tensor that already reports itself pinned is returned as it is |
| `Tensor.is_pinned(device=None)` | `True` for a tensor `pin_memory` returned, and PyTorch's own answer for any other tensor, which is `False` without a driver |
| `torch.accelerator.is_available()` | `True` |
| `torch.accelerator.current_device_index()` | `0` |
| `torch.accelerator.set_device_index(i)`, `torch.accelerator.set_device_idx(i)` | Accepted for device 0, `UnsupportedMode` otherwise |

The two `Tensor` methods are replaced on the class and the `torch.accelerator` functions on the module, not in the `TorchFunctionMode`, because the `DataLoader` pins in a thread of its own, where the mode is not active. They are restored when forwarding ends, and undone in a forked process as the rest of the mapping is. A pinned tensor uploaded with `non_blocking=True` takes the same queued, asynchronous path as any other upload.

### Compilation <!-- id: forwarding-compile -->

> Under `host="local"`, `torch.compile` returns the function or module it is given, unchanged, and warns once that it runs eagerly, so a compiled training loop runs and computes what eager code computes.

Inductor, the default backend, cannot compile here: before tracing it creates a CUDA tensor in this process to set up a device context, which needs a driver this process does not have. Dynamo with any backend also traces into `RemoteTensor` dispatch, which is letify's own Python, and recompiles it for every operator. A replayed step already sends the whole step as one entry, as [Step capture](#forwarding-step-capture) describes, so compiling on the client has nothing left to batch.

While forwarding is active, `torch.compile(model, ...)` with any arguments returns `model` itself, and `torch.compile(...)` used as a decorator factory returns a decorator that returns the function itself. `Module.compile(...)` does nothing. The first such call in a process emits a `UserWarning` naming `host="local"` and eager execution. `torch.compile` and `Module.compile` are restored when forwarding ends and undone in a forked process, as the rest of the mapping is.

`torch.cuda.get_rng_state` and `torch.cuda.set_rng_state` stay refused, so code that saves and restores the generator state names the gap instead of silently restoring a state that is not the runtime's.

### Autocast <!-- id: forwarding-autocast -->

> Inside `torch.autocast("cuda")`, an operator on a `RemoteTensor` gets the argument casts CUDA autocast gives it, so a mixed precision loop computes in the same dtypes as on the runtime's own GPU.

A `RemoteTensor` lives on the `meta` device, so PyTorch's own CUDA autocast, a dispatch key on CUDA tensors, never sees it. The `TorchFunctionMode` of [Mapping cuda](#mapping-cuda) applies the casts instead, above autograd as autocast does, so each cast is recorded as a differentiable `to(dtype)` and gradients reach the float32 parameters in float32.

While `torch.is_autocast_enabled("cuda")` is true, a torch function named in one of three lists casts its floating point `RemoteTensor` arguments, top level or one list level down, other than `float64` ones, and then runs with autocast's casts applied. The lists follow PyTorch's CUDA autocast policy and are matched by the function's name:

| Policy | Cast | Functions |
|---|---|---|
| lower precision | to `torch.get_autocast_dtype("cuda")` | `conv1d`, `conv2d`, `conv3d`, `conv_transpose1d`, `conv_transpose2d`, `conv_transpose3d`, `conv_tbc`, `prelu`, `addmm`, `addmv`, `addr`, `matmul`, `__matmul__`, `__rmatmul__`, `einsum`, `mm`, `mv`, `linear`, `bmm`, `baddbmm`, `addbmm`, `chain_matmul`, `multi_dot`, `scaled_dot_product_attention`, `lstm_cell`, `gru_cell`, `rnn_tanh_cell`, `rnn_relu_cell` |
| float32 | to `float32` | `acos`, `asin`, `cosh`, `erfinv`, `exp`, `expm1`, `log`, `log10`, `log2`, `log1p`, `reciprocal`, `rsqrt`, `sinh`, `tan`, `pow`, `__pow__`, `softplus`, `layer_norm`, `group_norm`, `norm`, `cosine_similarity`, `poisson_nll_loss`, `cosine_embedding_loss`, `nll_loss`, `hinge_embedding_loss`, `kl_div`, `l1_loss`, `smooth_l1_loss`, `huber_loss`, `mse_loss`, `margin_ranking_loss`, `multilabel_margin_loss`, `soft_margin_loss`, `triplet_margin_loss`, `multi_margin_loss`, `binary_cross_entropy_with_logits`, `dist`, `pdist`, `cdist`, `renorm`, `logsumexp`, `softmax`, `log_softmax`, `sum`, `prod`, `cumsum`, `cumprod` |
| widest | to the widest floating dtype among those arguments | `addcdiv`, `addcmul`, `atan2`, `bilinear`, `cross`, `dot`, `vdot`, `grid_sample`, `index_put`, `scatter_add`, `tensordot`, `cat`, `stack` |

`cross_entropy` with class index targets and no label smoothing runs as CUDA's `cross_entropy_loss` does: `log_softmax` in its input's dtype, then `nll_loss` with that result cast to float32. With probability targets or label smoothing, its input is cast to float32.

Every other function runs with its arguments as they are. A cast is not cached: a parameter used twice in one region is cast twice, which gives the same values as autocast's weight cache. `GradScaler` is PyTorch's own and is not covered.

### The device worker <!-- id: device-worker -->

> One executor per session, running in a thread of the session's call worker, executing ATen operators on tensors keyed by integer handle.

The call worker already runs the interpreter the session built: the project `.venv` from [Building the environment on a runtime](#remote-uv-sync) on a remote machine, the account's `python` where one is named, and this interpreter on `Local`. It already sees only the session's cards, because `CUDA_VISIBLE_DEVICES` is set in its environment before any code imports a CUDA library. The client starts the executor with a `device` request naming `cuda` where the instance has a GPU and `cpu` otherwise, so the same executor is exercised on a machine without one. The request's source is `wire.py`, `frames.py` and `executor.py`, executed once per worker process. From then on the executor's messages travel on the call worker's channel as [Transport](#forwarding-transport) describes, so a session keeps one connection to the account.

`connect` starts an executor as a process of its own instead, with the command `python -u -c <stub>`, where the stub reads a length-prefixed source from standard input and executes it, as [Channels](#channels) describes for the call worker. Standard output then carries frames only: that process moves Python's `sys.stdout` onto standard error before executing anything. The test suite uses it to reach the executor without a session.

The worker's first message names its device, the device name, and its PyTorch version. The client refuses a worker whose PyTorch major.minor differs from its own, because ATen operator schemas change between minor versions.

An operator is named by its overload, such as `aten.addmm.default`, and resolved on the runtime through `torch.ops` once per name.

### Operator templates <!-- id: forwarding-templates -->

> Each distinct operator structure is described to the worker once, and every later call of it travels as a template number with its handles, scalars and blobs.

The client numbers structures, as [Dispatch mechanism](#dispatch-mechanism) defines them, from 1 in the order they are first dispatched. A structure's first use puts its definition in the batch ahead of the entry that uses it: the overload name and the argument layout, where each tensor argument is a handle position, each `int` or `float` a scalar position, each CPU tensor a blob position, and every other value is kept as it is. An eager entry is `(template, handles, scalars, blobs, outputs, want)`, where `outputs` holds a new handle for each tensor output and None for an output that is one of the inputs.

The worker turns each definition into one generated Python function that builds the positional and keyword arguments from the three lists, so an entry is executed with one call to build its arguments and one to run the operator. Requests such as a fetch, a seed or a memory query keep their `letify.` names.

### Batching and synchronization <!-- id: forwarding-batching -->

> Operators are queued locally and sent without waiting. Only a read of a value waits for the runtime.

The queue is sent when it holds 256 entries, when its oldest entry has waited 2 ms, or at a synchronization. A send never waits for a reply.

The dispatching thread appends an entry without taking a lock. When the queue is due, the dispatching thread pickles the batch and hands the bytes to a sender thread, which writes them. A full pipe or SSH buffer therefore blocks the sender thread, never the step, and the dispatching thread holds the GIL only for the pickling. A synchronization is the exception: its batch is written by the waiting thread itself when no earlier batch is still queued for or being written by the sender thread, because that thread waits for the reply anyway and handing the batch over costs a thread wake per read. A background thread sends only a queue that nothing has been added to for 50 ms, so operators do not wait behind idle time between steps, and it never sends the unfinished part of a captured step, which [Step capture](#forwarding-step-capture) leaves to the dispatching thread.

A synchronization is one round trip. These synchronize: `Tensor.item()`, `tolist()`, `cpu()` and `to("cpu")` without `non_blocking=True`, `bool()`, `int()` and `float()` of a tensor, which includes control flow on a tensor value, `repr()` and `str()` of a tensor, copying a device tensor into a CPU tensor, an operator whose meta inference raised, the `torch.cuda` queries in [Mapping cuda](#mapping-cuda), `torch.cuda.synchronize()`, and the end of the declared function.

A read of one tensor's value sends only what that value depends on when that can be decided from the queue alone. When no repetition of a captured step is unfinished, if the tensor was created by a queued eager entry and no entry after it writes to an argument, the entries up to and including that one are sent with the read, and the rest stay queued in order. If the tensor was created before the queue and no queued entry writes to an argument, the read is sent alone. An operator writes to an argument when its schema marks one as written, which covers in-place operators, `out=` and the running statistics of batch norm. Otherwise the whole queue, the unfinished part of a captured step included, is sent with the read. `torch.cuda.synchronize()` and the end of the declared function always send everything.

The client counts operators, queued entries, batches, round trips, released handles, metadata cache hits, templates, captured steps, replayed operators and fallbacks, and `Client.queued` is the number of entries not yet sent, so ops per round trip and synchronizations per step are read from the session rather than estimated. An operator is counted when it is dispatched, not when its batch is sent, and a replayed operator counts as an operator too. `letify.remoting.device.current_client()` returns the client of the innermost active forwarding, or None outside one, so code inside a `host="local"` function reads `current_client().stats`.

### Reads without waiting <!-- id: forwarding-async-reads -->

> A copy to the host with `non_blocking=True`, and `await letify.fetch(tensor)`, queue the read and return at once. The value is waited for only where it is used.

`Tensor.to("cpu", non_blocking=True)` and `host.copy_(device_tensor, non_blocking=True)` of the same shape return a CPU tensor at once, with the shape and dtype the eager copy has, and queue a fetch entry. The fetch reads the device tensor at its place in the operator order, so a later in-place operator does not change it. Nothing is sent and nothing waits.

Until the fetch's reply is applied, the CPU tensor is unfilled. A torch function called while forwarding is active with an unfilled tensor among its arguments, or one list level down, first waits for that tensor: the queue up to and including its fetch is sent, and replies are read until that fetch's reply is applied. That covers `item()`, `tolist()`, `numpy()`, `repr()`, indexing and arithmetic. Reading metadata, which is `shape`, `dtype`, `device`, `ndim`, `is_cuda`, `requires_grad`, `size()`, `dim()`, `numel()`, `stride()`, `element_size()` and `len()`, does not wait. Access that is not a torch function, such as the buffer of an array `numpy()` returned before the reply, does not wait.

A batch that holds fetch entries asks for one reply, which carries every fetch in the batch in queue order. Replies are read in the order their batches were sent, by the thread that needs one, so a synchronization applies every earlier fetch reply before its own. `torch.cuda.synchronize()` and the end of the declared function therefore fill every unfilled tensor. A synchronization's batch is written by the waiting thread only when no earlier reply is still unread, as well as no batch queued for the sender thread. When 1024 replies are unread, the next non-blocking read first reads the oldest.

`letify.fetch(tensor)` is the form for `async def` code. It queues the same fetch when called and returns an awaitable resolving to the CPU tensor, equal to `tensor.cpu()`. Awaiting it waits in a thread from `asyncio.to_thread`, so the event loop keeps running, and `asyncio.gather` over several fetches waits for all of them. Given a tensor that is not on the runtime, it resolves to `tensor.detach().cpu()` with no round trip. An `async def` declared with `host="local"` runs its coroutine to completion with `asyncio.run` in the thread its call runs in.

A fetch skipped because an earlier operator failed raises `RemoteError` where its tensor is used or its awaitable is awaited. A failure carried by a reply that no synchronization waits for is kept, and the next synchronization raises it.

### Step capture <!-- id: forwarding-step-capture -->

> A sequence of operators that repeats is registered with the worker once as a step, and each later repetition is queued as one entry naming the step and carrying only its handles, scalars and blobs.

A step is what a training loop repeats: forward, backward and the optimizer update. Backward and foreach optimizer operators reach `__torch_dispatch__` like forward ones, so they are traced and replayed the same way, and nothing about autograd or the optimizer is special cased. The local side of a replayed operator is unchanged: its outputs are built from the metadata cache, so autograd sees the same tensors it sees eagerly.

**Trace.** Every eagerly dispatched operator appends a trace key, its template number and its output layout, and a record of its input handles and new output handles. Scalars and blobs are not in the trace key, so the bias corrections an optimizer passes and the bounds of a batch slice vary without breaking a repetition. Requests are not traced and do not break the trace. An operator run at once because its meta inference raised ends the trace, and tracing starts again after it.

**Detection.** When an operator's trace key occurred earlier at a distance `P` of at least 8 and at most 4096 operators, and the last `P` trace keys equal the `P` before them, both windows are wired: each tensor argument becomes the offset of its handle among the handles its window creates, or `external` when the window did not create it. The handles a window creates must be one consecutive range. When the two wirings are equal, the last `P` operators are a step. A step is therefore registered after it has run eagerly twice in a row. A step records, for each operator, its template, the signatures of its outputs, and for each tensor argument either the offset of its handle among the handles the repetition creates or `external`. Its definition goes into the next batch. A step equal to one already registered keeps that number.

**Replay.** After a step is registered, each dispatched operator is compared with the step's next one. It matches when its template is the same, its output signatures from the metadata cache are the same, and each tensor argument was created at the expected offset in this repetition, or before this repetition began where the step expects `external`. A matching operator builds its outputs as eager dispatch does, taking handles from the same counter, so a repetition's new handles are one consecutive range, and it adds its external handles, scalars and blobs to the repetition. Nothing is queued per operator. When the last operator matches, one entry `(step, first handle, start, stop, externals, scalars, blobs, keep)` is queued, with `keep` filled in when the entry is taken into a batch as **Early release** describes, and the next repetition begins with the next operator.

**Position readers.** The first time an operator matches at a step position, the client generates a Python function for that position from its arguments, unless an argument is a plain CPU or meta tensor. The function checks, without building the structure, that the overload is the same object, and that the arguments have the same count, container types and lengths, keyword names, tensor shapes, strides and dtypes, `int` and `float` types, and other values. It returns the tensor arguments, the scalars and the metadata key's offsets and scalars. A later operator at that position that passes the checks, and whose first tensor belongs to the same client, takes the position's recorded metadata when its key is the recorded one and the metadata cache entry for its key otherwise. It is then matched as **Replay** describes. A failed check or a cache miss sends the operator down the full path, which matches or falls back as above, so the entries queued and the values are the same with and without the reader.

**Fallback.** An operator that does not match ends the repetition. The operators matched so far are queued as an entry with `stop` at the mismatch, and the mismatching operator is dispatched eagerly and starts a new trace. A shape that changes mid-run, a different operator, an argument from a different producer, and an inference that raises are all mismatches. The runtime has executed nothing of the repetition before its entry arrives, so a fallback runs every operator exactly once, in dispatch order, and values are the same as eager.

**Partial sends.** A synchronization during a repetition, and a matched operator carrying a CPU tensor larger than 4 KiB, queue the repetition so far as an entry and send the queue; the repetition then continues with `start` at the next operator and the same first handle. A handle created in a repetition and released before its entry is queued stays in the release list until the entry is queued.

**Early release.** A tensor a repetition creates is released on the runtime right after the last operator of the step that reads it, when nothing else can read it, rather than with the release list after the whole step. When a step is registered, the client and the worker both compute each created handle's release position from the wiring: the last position whose operator takes that offset as an argument, or the position that created it when no later operator of the step reads it. A step entry carries `keep`, the offsets whose release position lies in `start` to `stop - 1` and that must not be released early. The client computes `keep` when it takes the entry into a batch: an offset is kept when a `RemoteTensor` still names its handle, which is when its handle has not been released locally, or when an entry after it in that batch, an entry left in the queue, or the unfinished repetition reads the handle. A saved tensor autograd holds for backward is named by a `RemoteTensor`, so it is released only after the backward operator that reads it. A tensor the loop keeps across steps, such as the loss or a gradient, is kept, and released by the release list as before.

**Worker.** The worker executes operators `start` to `stop - 1` of the step in order, reading an argument's handle as `first handle + offset` or from `externals`, and stores each new output under the next handle counted from `first handle`. After each operator it drops the handles whose release position is that operator, other than those in `keep`, from its table. Each operator is the same call an eager entry makes, so a failure is recorded and reported as [Failure semantics](#forwarding-failure) describes, naming the operator. The worker does not use CUDA graphs, because the handles bound to a step, such as the batch, change between repetitions and a graph needs fixed input memory.

### Handles <!-- id: forwarding-handles -->

> The client assigns handles, so creating a tensor needs no reply, and a dropped tensor's handle is released with the next batch.

A handle is an integer from a per-session counter. `RemoteTensor`s that share a handle, through `detach` or an in-place result, share one reference object, and when the last of them is collected its handle is appended to a release list. The list travels in the next batch, and a batch is sent early when it reaches 4096 handles. The worker applies each release of a batch right after the last entry of that batch that reads the handle, as an argument of an eager entry, an external of a step entry or the tensor of a fetch, and before the batch's first entry when no entry of the batch reads it. An operator queued before its input was collected can travel in the same batch as that input's release, and still finds the input, while memory a collected tensor held is returned before the batch's later operators allocate.

A release never reaches the worker ahead of an operator that uses the handle. A batch takes the release list before it takes entries from the queue, so every entry dispatched before a handle was collected is in that batch or an earlier one. A batch carries no releases when it leaves entries in the queue, as a read that sends only its dependencies does, or while a repetition has matched operators not yet queued, whose externals are not in the queue. Those releases stay in the list for a later batch. This holds whichever thread sends the batch: the dispatching thread, the idle sender, or a collection that runs mid-step.

### Transfers <!-- id: forwarding-transfers -->

> A copy to the device and a copy to the host travel as out-of-band binary buffers, with no base64 and no copy beyond the one the kernel makes.

A copy to the device returns before its bytes are written. `Tensor.cuda()`, `Tensor.to("cuda")` and a factory call with a CPU tensor argument queue their entry and hand it to the sender thread, and only a read of a value that depends on the entry waits for the bytes. The runtime receives the CPU tensor's values as they were at the call, not at the write, so a CPU tensor larger than 4 KiB is sent as a private copy of its bytes made at the call, blocking or not. The client does not ask whether the memory is pinned, because that query can initialize CUDA in this process, and the copy costs memory speed against a link about 100 times slower.

A contiguous CPU tensor, or its private copy, is sent as a view of its memory, taken through `ctypes` from its data pointer, so no NumPy is needed. A non-contiguous one is made contiguous first. On the runtime the buffer is received into a `bytearray` and wrapped with `torch.frombuffer`. A copy to the host is made contiguous on the runtime, copied to CPU memory, sent as a view of that memory, received into a `bytearray` and wrapped with `torch.frombuffer`.

### Transport <!-- id: forwarding-transport -->

> The client and the worker talk through a `Transport` with three methods, so the stream underneath can be replaced without touching either.

```python
class Transport(Protocol):
    def send(self, head: bytes, buffers: Sequence[memoryview]) -> None: ...
    def recv(self) -> tuple[bytes, list[bytearray]]: ...
    def close(self) -> None: ...
```

`head` is a pickled message whose tensors are replaced by buffer indices. `buffers` are written in order without being joined to the head.

Both implementations put the frames of [Frames](#frames) on the wire: a message is a head frame, `REQUEST` from client to executor and `REPLY` back, whose payload is the buffer count, the buffer lengths and `head`, followed by the buffers as `DATA` frames written from the caller's memory and read into preallocated buffers. Device messages use stream 2, which no call uses because calls take odd stream ids.

`ChannelTransport` implements it on a session's persistent channel, beside the calls. The channel's reader hands every stream 2 reply to the transport, and the call worker's frame reader hands every stream 2 request to the executor thread. A thread waiting for a device reply polls the channel's read descriptor without blocking for up to 2 ms before it blocks in a read, because a dispatching thread that sleeps through the round trip resumes on a core that has left its fast state, which measured 1.1 ms of extra CPU per step on lab_docker with a read every step. `StreamTransport` implements it over a readable and a writable file descriptor, for an executor started by `connect`.

### Failure semantics <!-- id: forwarding-failure -->

> An operator that raises on the runtime is reported at the next synchronization, with its overload name and the runtime's traceback, and the session stays usable.

The worker records the first failure and skips the operators after it until a synchronization asks. The reply carries the failure, the client raises `RemoteError` whose message names the operator and whose `remote_traceback` is the runtime's, and the worker clears the failure. An operator that reads a handle whose producer failed or was skipped raises at the following synchronization, naming the handle. Tensors whose operators succeeded keep their values.

A lost worker process or link raises `RuntimeLost`, and the session is discarded.

### Version guards <!-- id: forwarding-versions -->

> PyTorch 2.1 or newer is required locally and on the runtime.

`_make_wrapper_subclass`, `TorchFunctionMode` and `torch.utils._pytree` are each present in 2.1. An older PyTorch raises `UnsupportedMode` naming the version found and the version needed.

### The driver stand-in is retired <!-- id: letify-core -->

> `host="local"` does not load letify-core. Standing in for the CUDA driver was replaced by operator forwarding.

Measured on 2026-09-14 against a Tesla P100 server with torch 2.5.1 cu121: libcudart 12.1 resolves 425 driver symbols by name and the stand-in `libcuda.so.1` exports 20, so CUDA initialization fails with `cudaErrorInsufficientDriver`. The private `cuGetExportTable` blocks adding symbols one by one, and PyTorch kernels arrive through fatbinary registration that `cuModuleLoadData` does not see. Operator forwarding depends on PyTorch's public extension points instead of the driver's private ones. The Rust crates in `letify-core/` and `letify.remoting.probe` remain in the tree and the wheels, and nothing on the `host="local"` path calls them.

## Command line

> Every command prints for a person by default and for a program with `--json`, through one renderer, `letify/render.py`.

### Output conventions

A style is chosen per stream, so standard output and standard error decide separately:

| Setting | Rule |
|---|---|
| Colour | Only when the stream is a terminal and `NO_COLOR` is unset or empty |
| Characters | Block characters and symbols when the stream encoding is UTF-8, ASCII otherwise |
| Width | `shutil.get_terminal_size`, 80 columns when it cannot be read |

| Mark | UTF-8 | ASCII | Colour | Means |
|---|---|---|---|---|
| success | `✓` | `+` | green | The command did what was asked |
| failure | `✗` | `x` | red | It did not, and the line says why |
| warning | `!` | `!` | yellow | It ran, with something to notice |

A heading, an alias at the top of a block and a table's column names are bold. Secondary text, such as a note, a path hint or a reset date, is dim. A table left-aligns each column to its longest cell with two spaces between columns and names its columns in capitals.

A `LetifyError` that reaches the command line prints `✗ <message>` on standard error and exits with 1. An argument error is argparse's own and exits with 2.

`--json` prints the records with no styling on `providers`, `devices`, `status`, `usage`, `utilization`, `probe` and `efficiency`.

Connection decision lines stay on standard error as `letify: <message>`, with the content set by Transport. When standard error is a terminal with colour, the `letify:` prefix is dim and nothing else changes.

### Commands

| Command | Prints |
|---|---|
| `providers` | A table `ALIAS  KIND  PERSISTENCE`. A provider that cannot be built has `✗ unavailable: <reason>` in place of kind and persistence. `--json` is a list of `{alias, kind, persistence}` or `{alias, unavailable}` |
| `devices` | A table `PROVIDER  ACCELERATORS`, the accelerators joined by `, `. `--json` is the mapping from alias to accelerator names |
| `status` | The header `<name>  <live> live, <busy> busy`, one block per live runtime, then a table `PROVIDER  ACCELERATOR  RESERVED  INDICES` with reserved as `<reserved>/<count>`. With no runtime the blocks are replaced by `no live session in this process`. `--json` is `Launcher.status()` |
| `usage`, `utilization` | As Remaining usage and GPU utilization describe |
| `probe` | A mark and `forwarding usable`, `forwarding usable but costly` or `forwarding not usable`, then aligned `platform`, `core`, `agent` and `round trip` fields and the reason, dim. `--json` is the capability record |
| `efficiency` | `<p>% of a direct run`. `--json` is `{"efficiency": <fraction>}` |
| `check` | `✓ <alias> answers`, then the machine's output indented by 2 spaces |
| `login` | `✓ <alias> declared in <home>`, or `! <alias> was already declared in <home>, so nothing was asked for`, then `✓ <alias> referenced in <project>, which is safe to commit` |
| `logout` | `✓ <alias> removed from <home>`, then the note about the project reference, dim |
| `stubs` | `✓ <path written>` |
| `setup <tool>` | `✓ <tool> <version> at <path>`, or with `--where` the `cache`, `link`, `PATH` and `uses` lines, as Installing external tools describes |
| `client shell connect` | `On your own machine, run:` bold, the login command plain so it can be copied, and the notes dim |

A runtime block in `status` is:

```
run-1  lab.P100  busy
  cards 0, 1  host remote  link forward-ssh, 42.0 ms
  up 1 h 2 min  idle 3 min
  about 2.07 compute units so far at 2.00 compute units/hour
  [████████████████████░░░░░░░░░░░░░░░░░░░░] 50% used
  50.00 compute units left of 100.00
```

The header ends `busy` while a call runs and `idle` otherwise. `cards` is left out where the provider assigns the device, `link` where there is none, and the round trip where it was not measured. The cost line needs a usage record with a rate, and is uptime times the rate, so it is an estimate and says `about`. The gauge and amount lines are the usage block's own lines for that record.

### Installing external tools <!-- id: confirmed-tool-install -->

> letify installs a missing `tailcat` or `eci` automatically the first time it needs one, and says so on standard error. Each is fetched from its publisher's GitHub release, Tailscale or Elice, verified against pinned SHA-256 digests, cached per version under `~/.letify/tools` and linked into the project's virtual environment. Neither binary is part of letify or covered by its license, and letify never writes to a shared `PATH` directory.

| Tool | Pinned version | Release | Assets |
|---|---|---|---|
| `tailcat` | `letify.transport.setup.TAILCAT_VERSION` | `https://github.com/tailscale/tailcat/releases/download/v<version>/` | `tailcat_<version>_linux_{amd64,arm64,armv7}.tar.gz`, `tailcat_<version>_windows_{amd64,arm64}.zip` |
| `eci` | `letify.install.ECI_VERSION` | `https://github.com/elice-dev/eci-cli/releases/download/<version>/` | `eci-darwin-arm64-<version>.tar.gz`, `eci-linux-x86_64-<version>.tar.gz`, `eci-windows-x86_64-<version>.zip` |

**Lookup.** When letify needs a tool it takes the first of:

1. The project environment: `<venv>/bin/<tool>`, or `<venv>\Scripts\<tool>.exe` on Windows. The environment is `sys.prefix` when letify runs inside a virtual environment, and otherwise `.venv` in the working directory when it holds `pyvenv.cfg`. A link letify made for another version is skipped here and replaced in step 2.
2. The cache for the pinned version, `~/.letify/tools/<tool>/<version>/<tool>` (`.exe` on Windows). A hit is linked into the project environment as below.
3. `PATH`. A user's own install is used as it is and never replaced.
4. The confirmed install.

An account's `tailcat_binary` or `eci_binary`, when set, skips the lookup and is run as written.

**Automatic install.** `tailcat` is installed when a command that needs it finds none: `letify client shell connect` and `letify login tunnel`. It happens with or without a terminal and asks nothing. `eci` is not installed automatically: `letify login elice` asks first, as [Elice machines](#elice-machines) describes, and otherwise only `letify setup eci` installs it. Two lines go to standard error, as connection decision lines do:

- before the download, `letify: installing <tool> <version> from <asset URL> into <version directory>`
- after it, `letify: <tool> <version> verified sha256 <digest>, linked at <path>`, where the path is the project link, or the cache path when nothing is linked

**Download progress.** The asset download reports progress on standard error, because a release archive can take minutes over a slow link. The download reads 64 KiB at a time. On a terminal one line is redrawn in place with a carriage return, at most every 0.1 s and once at the end: `letify: downloading <tool> <version> [<gauge>] <percent>% <done>/<total> MiB <rate> MiB/s`, where the gauge is the command line gauge 20 cells wide. When the server sends no `Content-Length` the gauge and percent are left out and the line shows `<done> MiB <rate> MiB/s`. The finished line ends with a newline. When standard error is not a terminal, nothing is redrawn: one line without a gauge is written at 25, 50, 75 and 100 percent, or a single line at the end when the size is unknown, so a log file gets at most four lines. `checksums.txt` is small and shows no progress.

Automatic install is on by default. `auto_install = false` at the top level of `~/.letify/config.toml` turns it off, and so does the environment variable `LETIFY_AUTO_INSTALL` set to `0`, `false` or `no`. The environment variable, when set to any value, decides over the file. Turned off, a missing tool fails with the install instructions followed by `Install it with: letify setup <tool>`. The Tailcat connection strategy never installs, because it runs inside a race; it is skipped with `tailcat is not on PATH; run 'letify setup tailcat'`.

**Installing.** On macOS with `brew` on `PATH`, `tailcat` is installed with `brew install tailcat` and not cached. Otherwise:

1. The asset for this operating system and architecture and the release's `checksums.txt` are downloaded with `urllib`. A platform with no asset fails with the releases page.
2. The asset's SHA-256 must equal the digest pinned in `letify.install` for that asset, and the line naming the asset in `checksums.txt` must carry the same digest. A mismatch fails naming both digests and writes nothing.
3. The archive is read member by member. A member whose path is absolute or contains `..`, or that is a device, a hard link, or a symbolic link pointing outside the archive, refuses the whole archive. For `tailcat` only the member `tailcat` (`tailcat.exe`) at the root or one directory down is kept. For `eci` the whole bundle is kept with its top directory stripped, because the `eci` binary loads the libraries next to it.
4. Extraction goes to a temporary directory next to the version directory, which is renamed onto `~/.letify/tools/<tool>/<version>/` after the binary is found in it. The binary is mode 0555 and every other file loses its write bits.

Elice's `install.sh` is not run, because it writes to `/usr/local` or `~/.local`, may call `sudo`, and appends to the user's shell profile. letify downloads the asset that script downloads and checks it against the same `checksums.txt`.

**Linking.** With a project environment, the cached tool is linked to `<venv>/bin/<tool>` and recorded in the marker file `<venv>/bin/.letify-<tool>`, which holds the version.

- `tailcat` is a hard link to the cached binary. When the hard link fails, for another file system or no hard link support, the binary is copied and `! linked by copy: <reason>` is printed once.
- `eci` is a launcher that runs the cached binary with the same arguments: a `#!/bin/sh` script with `exec`, or a `.cmd` file on Windows, because a hard link separated from the bundle cannot find its libraries.
- An existing link whose marker names another version is replaced.
- A file at that path that has no marker and is not the same file as the cache is not letify's. It is left untouched, `! <path> exists and was not created by letify; using <cache path>` is printed, and the cache path is used.

With no project environment nothing is linked and the cache path is used.

**`letify setup <tool>`**, for `tailcat` or `eci`, runs the lookup, installs into the cache when nothing is found, whatever the automatic install setting, links into the project environment, prints the two `letify:` lines when it installed, and prints `✓ <tool> <version> at <path>`. `--where` installs nothing and prints four aligned lines: `cache` with the cache path and `present` or `missing`, `link` with the link path and `letify <version>`, `not letify's`, `missing` or `no project environment`, `PATH` with the path found or `none`, and `uses` with the path the lookup chose or `nothing`.

### Machine-readable output <!-- id: machine-readable-output -->

> `letify usage --json`, `letify utilization --json` and `letify status --json` print JSON whose field names and types are a contract: a field may be added, but none is renamed, removed or retyped.

Each command prints one JSON document on standard output and exits 0. An error that stops the command prints a message on standard error and exits non zero, with nothing on standard output. The formatting of the human output does not change the JSON.

`letify usage --json` prints a list with one object per declared alias, in configuration order. A usage object carries every field of the `Usage` record under "Remaining usage", all keys always present:

| Key | Type |
|---|---|
| `alias`, `kind`, `unit`, `source` | string |
| `remaining`, `limit`, `used`, `rate_per_hour` | number or null |
| `resets_at`, `as_of` | number of Unix seconds, or null |
| `unmetered` | boolean |
| `note` | string or null |
| `resources` | list of `{"name", "unit"}` strings with `remaining`, `used`, `limit` and `resets_at` as number or null, empty when the account has no further allowance |

An alias whose provider cannot be built prints `{"alias": <string>, "unavailable": <string>}` instead.

`letify utilization --json` prints a list of rows as "GPU utilization" describes: one per `machine` provider and one per `session` instance with an accelerator. A row has `alias`, `kind` and `scope` (string), `accelerator` (string, or null on a `machine` row), `devices` (list) and `reason` (string or null, why `devices` is empty). A device object has `index` (integer), `name` (string), `utilization_percent`, `memory_used_gb`, `memory_total_gb`, `memory_percent`, `temperature_c`, `power_w` (number or null each), `holder` (one of `letify`, `others`, `mine`, `free`, `unknown`, or null on a `session` row), `users` (list of strings, the owners when `holder` is `others`) and `reserved` (boolean). An alias whose provider cannot be built prints `{"alias": <string>, "unavailable": <string>}`.

`letify status --json` prints the `Launcher.status()` object described under "Status reporting": `name` (string), `live` and `busy` (integer), `devices` (object keyed by alias, then by accelerator, each `{"count": integer, "reserved": integer, "indices": [integer]}`), `runtimes` (list of `{"name", "provider", "accelerator", "placement"}` strings, `devices` list, `busy` and `persistent_channel` booleans, `idle_seconds` number), `declared` and `config_sources` (lists of strings). `letify status` without the flag prints the same document.

### Editor extension <!-- id: editor-extension -->

> `letify-ext/` is a VS Code extension that shows the remaining quota and GPU load in the status bar, read only through the JSON above.

The extension runs `uv run letify <command> --json` in the first workspace folder. The command is the setting `letify.command`. It reads usage every `letify.usageIntervalSeconds`, 60 by default, and utilization and status every `letify.utilizationIntervalSeconds`, 10 by default while its view is visible and at the usage interval otherwise. It makes no network call of its own and writes no credential anywhere.

The quota status bar item shows the account with the lowest remaining share, `remaining / limit`, as `<alias> <percent>% left`, with `(<time to reset>)` when `resets_at` is known. An account with no limit is ranked after every account with one. The item turns to the warning color when the share left is below `letify.warningPercent`, 20 by default, and to the error color below `letify.errorPercent`, 5 by default. The GPU item shows `GPU <free>/<total> free <mean>%` when any device reports a `holder`, where a device is free when its `holder` is `free`. When no device reports one it shows `GPU <busy>/<total> busy <mean>%`, where a device is busy at `letify.busyPercent`, 10 by default, or above. The mean is over devices that report utilization.

Hovering either item shows its cards. Clicking one opens the letify view with the tabs Quota, GPU and Runtimes. A quota card shows a gauge of the share used, the reset time, the share of the period elapsed where the period length is known (7 days for `GPU hours`, the calendar month for `USD`), the projection `used / elapsed share` capped at 999 percent, and the hourly rate. Each record in `resources` adds a row under them: its `name`, the amount left of its limit, a gauge of the share used when the limit is known, and its reset time. A GPU card shows, per device, utilization and memory gauges, temperature, power and the holder label:

| `holder` | Label |
|---|---|
| `letify` | `letify reserved` |
| `others` | `other users: <users>` |
| `mine` | `yours` |
| `free` | `free` |
| `unknown` | `unknown` |
| null | `letify reserved` when `status` lists the index as reserved, nothing otherwise | Daily history is the mean GPU utilization and the quota spent per UTC day, kept by the extension in its own storage for 30 days from its own samples.

## Packaging

> One install, `uv add letify`, with no extras. It installs cloudpickle and blake3 and nothing else.

letify is a dependency inside a research repository, so it adds as little as possible to that repository's environment.

Provider tools run out of process and never in the user's `.venv`. Colab runs through `uv tool run --from google-colab-cli colab`. Modal's client runs in the Modal adapter, in a uv environment started with `uv run --no-project --with "modal>=1.0,<2"`, and its sign in runs through `uv tool run --from modal modal`. Elice uses the standard library HTTP client. The `gcs` blob store uses a standard library client too.

uv must be installed. letify finds it from the `UV` environment variable, then from `PATH`. If neither has it, letify raises an error that says uv is required.

The Python code is pure and links no Python extension. letify-core binaries are built in CI and placed in `letify/remoting/lib/`, so each release has one wheel per desktop platform, tagged `py3-none-<platform>`:

| Platform | Wheel platform tag |
|---|---|
| Linux x86_64 | `manylinux_2_28_x86_64` |
| Linux aarch64 | `manylinux_2_28_aarch64` |
| Windows x86_64 | `win_amd64` |
| Windows arm64 | `win_arm64` |
| macOS arm64 | `macosx_11_0_arm64` |
| macOS x86_64 | `macosx_10_12_x86_64` |

Linux wheels are built inside the `manylinux_2_28` containers, so the binaries need glibc 2.28 or newer. That covers RHEL 8, Debian 10 and Ubuntu 18.10 onward. The sdist carries no binaries, and an install from it has no letify-core.

### Versioning and releases <!-- id: versioning -->

> One version for everything, read from `pyproject.toml`, and a release is the push of the tag `v<version>`.

`[project].version` in `pyproject.toml` is the source of truth. These carry the same value:

| File | Field |
|---|---|
| `letify/__init__.py` | `__version__` |
| `letify-core/Cargo.toml` | `[workspace.package].version`, and the `letify-wire`, `letify-driver` and `letify-agent` entries in `Cargo.lock` |
| `letify-ext/package.json` | `version`, and `version` and `packages[""].version` in `package-lock.json`, when `letify-ext/` exists |
| Git tag | `v<version>` |

`python scripts/version.py set <version>` writes all of them and `python scripts/version.py check [--tag v<version>]` exits 1 naming each file that disagrees. The `ci` workflow runs the check on every push and pull request, and the `publish` workflow runs it with the pushed tag before building anything.

Pushing a tag `v*` runs `.github/workflows/publish.yml`. It builds the sdist, the six platform wheels above and, when `letify-ext/` exists, the extension's `letify-ext-<version>.vsix` after its unit tests pass. It then creates the GitHub Release for the tag with every one of those files attached, and uploads the sdist and wheels to PyPI. The extension is not published to the VS Code Marketplace.

## Known gaps

> Implemented and unimplemented, stated plainly so nobody builds on a promise.

- **PyTorch forwarding still pays one round trip per value read.** On dept_gpu the benchmark step takes a median 1.31 ms under `host="local"` against 1.72 ms directly when the loss is read once per 50 steps, and 4.94 ms against 1.78 ms when it is read every step. The measurement is in [NETWORK.md](NETWORK.md#pytorch-forwarding-on-dept_gpu).
- **PyTorch forwarding covers one device per session and no CUDA streams, events, graphs or generator state.** Custom CUDA extensions and Triton kernels compiled in this process cannot run, because nothing here compiles for the runtime's GPU. `torch.compile` is untested.
- **`Modal` and `Elice` are not exercised against the live services.** Their code follows each service's published interface, but neither has been run end to end. The `eci` commands and flags come from Elice's CLI documentation. The JSON field names letify reads from `eci` (`devices`, `cpu_vcore`, `pricing_type`, `price_per_hour`, `status`, the public IP, `resource_quota`) come from the models in Elice's Terraform provider, elice-dev/terraform-provider-eci, and are not checked against `eci` output. The field names above and the `ubuntu` login user were checked against `eci` 0.2.1 output on a live organization on 2026-09-15.
- **`eci compute vm launch` takes the machine password as an argument.** It is visible to other local users that can list processes while the launch runs. letify generates a password per account and never prints it. The Modal adapter's calls were checked against the signatures of Modal 1.5.5, and `letify login modal` has not been run against Modal's sign in.
- **The connection pipeline is not exercised against live networks.** `Rendezvous`, `Strategy`, `Link`, `Probe`, `Pipeline`, `LinkCache` and the remote agent are implemented and tested over loopback sockets and faked commands. Installing and starting `sshd` on a Colab VM over `colab exec` is not yet checked against a live runtime.
- **letify runs no command on an Elice machine through `eci`.** Elice's remote half runs over forward SSH to the machine's public IP, and the punch and Tailcat strategies need that SSH to succeed first.
- **Orphan reconciliation is not implemented.** A session whose controlling machine was killed outright is released by the lease on the providers where the process is the cost. Where the platform bills for the machine and takes no deadline, nothing ends it: an Elice machine bills compute until it is stopped. The intended answer is that the next letify process asks the provider what is running under this project's name and ends what nothing is watching, with a command to do it on demand. Neither exists yet.
- **Persistence detection is not implemented.** Deciding a machine's disk policy by writing a marker file and looking for it in a later runtime is a decision recorded here, not yet code.
