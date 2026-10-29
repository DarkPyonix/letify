//! The `letify-link` executable. Today it runs one mode, `relay`; the sidecar control protocol
//! is not part of it yet. It owns argument parsing and the node key file, not the relay.

use std::process::ExitCode;

const USAGE: &str = "usage: letify-link relay --listen ADDR --key PATH [--allow NODEID ...] [--rate-limit BYTES_PER_SECOND]";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("relay") => relay_mode(&args[1..]),
        _ => {
            eprintln!("{USAGE}");
            ExitCode::from(2)
        }
    }
}

#[cfg(not(feature = "relay"))]
fn relay_mode(_args: &[String]) -> ExitCode {
    eprintln!("this build of letify-link has no relay: rebuild with the Cargo feature `relay`");
    ExitCode::from(2)
}

#[cfg(feature = "relay")]
fn relay_mode(args: &[String]) -> ExitCode {
    match run_relay(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(why) => {
            eprintln!("letify-link relay: {why}");
            ExitCode::from(2)
        }
    }
}

#[cfg(feature = "relay")]
fn run_relay(args: &[String]) -> Result<(), String> {
    use letify_link::relay::{Relay, RelayConfig};
    use letify_link::{NodeId, PeerPolicy};

    let (mut listen, mut key, mut allow, mut rate) = (None, None, Vec::new(), None);
    let mut it = args.iter();
    while let Some(flag) = it.next() {
        let mut value = || it.next().cloned().ok_or(format!("{flag} needs a value"));
        match flag.as_str() {
            "--listen" => listen = Some(value()?),
            "--key" => key = Some(value()?),
            "--allow" => {
                let id: NodeId = value()?.parse().map_err(|e| format!("--allow: {e}"))?;
                allow.push(id);
            }
            "--rate-limit" => {
                rate = Some(
                    value()?
                        .parse::<u64>()
                        .map_err(|e| format!("--rate-limit: {e}"))?,
                )
            }
            other => return Err(format!("unknown argument {other}\n{USAGE}")),
        }
    }
    let listen = listen.ok_or(format!("--listen is required\n{USAGE}"))?;
    let key = key.ok_or(format!("--key is required\n{USAGE}"))?;
    let identity = load_or_create_key(&key)?;
    let policy = if allow.is_empty() {
        PeerPolicy::from_fn(|_| true)
    } else {
        PeerPolicy::allow(allow)
    };
    let cfg = RelayConfig {
        rate_limit: rate,
        ..RelayConfig::default()
    };
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| e.to_string())?;
    runtime.block_on(async move {
        let relay = Relay::bind(&listen, identity, policy, cfg)
            .await
            .map_err(|e| format!("cannot listen on {listen}: {e}"))?;
        println!("{}", relay.node_id());
        println!("listening on {}", relay.local_addr());
        relay.run().await;
        Ok(())
    })
}

/// Read the node secret from `path` (64 hex characters), or create the file with mode 0600.
#[cfg(feature = "relay")]
fn load_or_create_key(path: &str) -> Result<letify_link::Identity, String> {
    use letify_link::Identity;
    match std::fs::read_to_string(path) {
        Ok(text) => {
            let text = text.trim();
            if text.len() != 64 || !text.is_ascii() {
                return Err(format!("{path} does not hold 64 hex characters"));
            }
            let mut secret = [0u8; 32];
            for (i, b) in secret.iter_mut().enumerate() {
                *b = u8::from_str_radix(&text[2 * i..2 * i + 2], 16)
                    .map_err(|_| format!("{path} does not hold 64 hex characters"))?;
            }
            Ok(Identity::from_secret_bytes(secret))
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            let identity = Identity::generate();
            let hex: String = identity
                .secret_bytes()
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect();
            write_private(path, &hex).map_err(|e| format!("cannot write {path}: {e}"))?;
            Ok(identity)
        }
        Err(e) => Err(format!("cannot read {path}: {e}")),
    }
}

#[cfg(feature = "relay")]
fn write_private(path: &str, text: &str) -> std::io::Result<()> {
    use std::io::Write;
    let mut opts = std::fs::OpenOptions::new();
    opts.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    opts.open(path)?.write_all(text.as_bytes())
}
