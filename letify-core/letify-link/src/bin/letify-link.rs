//! The `letify-link` executable. It runs two modes: `serve`, the sidecar that the Python package
//! controls over a local socket, and `relay`. It owns argument parsing, not what either does.

use std::process::ExitCode;

const USAGE: &str = "usage: letify-link serve --control PATH|tcp --key PATH [--stream-dir DIR]\n       letify-link relay --listen ADDR --key PATH [--allow NODEID ...] [--rate-limit BYTES_PER_SECOND]";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("serve") => serve_mode(&args[1..]),
        Some("relay") => relay_mode(&args[1..]),
        _ => {
            eprintln!("{USAGE}");
            ExitCode::from(2)
        }
    }
}

fn serve_mode(args: &[String]) -> ExitCode {
    match run_serve(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(why) => {
            eprintln!("letify-link serve: {why}");
            ExitCode::from(2)
        }
    }
}

fn run_serve(args: &[String]) -> Result<(), String> {
    use letify_link::sidecar::{ServeOptions, load_or_create_key, serve};

    let (mut control, mut key, mut stream_dir) = (None, None, None);
    let mut it = args.iter();
    while let Some(flag) = it.next() {
        let mut value = || it.next().cloned().ok_or(format!("{flag} needs a value"));
        match flag.as_str() {
            "--control" => control = Some(value()?),
            "--key" => key = Some(value()?),
            "--stream-dir" => stream_dir = Some(value()?.into()),
            other => return Err(format!("unknown argument {other}\n{USAGE}")),
        }
    }
    let control = control.ok_or(format!("--control is required\n{USAGE}"))?;
    let key = key.ok_or(format!("--key is required\n{USAGE}"))?;
    let identity = load_or_create_key(&key)?;
    #[cfg(feature = "relay")]
    let mut builder = tokio::runtime::Builder::new_multi_thread();
    #[cfg(not(feature = "relay"))]
    let mut builder = tokio::runtime::Builder::new_current_thread();
    let runtime = builder.enable_all().build().map_err(|e| e.to_string())?;
    let opts = ServeOptions {
        control,
        stream_dir,
    };
    let result = runtime.block_on(serve(opts, identity));
    // Do not wait for tasks that are still copying: the control channel is gone.
    runtime.shutdown_background();
    result.map_err(|e| e.to_string())
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
    let identity = letify_link::sidecar::load_or_create_key(&key)?;
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
