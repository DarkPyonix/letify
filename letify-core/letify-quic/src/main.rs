//! Carries one letify session over QUIC.
//!
//! letify already knows how to learn both sides' public UDP endpoints through a
//! rendezvous, so this binary does not discover anything. It is given the local port to
//! bind, the peer's endpoint, and a token, and it puts a QUIC connection between them.
//!
//! Why QUIC at all, when Tailcat already carries a session over UDP: a network that
//! shapes UDP it cannot classify often lets QUIC through, because QUIC is what the web
//! runs on. The handshake is a real QUIC v1 handshake with a TLS certificate, so a device
//! that classifies by payload sees HTTP/3's transport rather than unknown datagrams.
//!
//! Two modes, matching how letify drives Tailcat:
//!
//!     letify-quic serve --bind <port> --peer <ip:port> --token <hex> --forward <port>
//!     letify-quic connect --bind <port> --peer <ip:port> --token <hex>
//!
//! `serve` accepts the connection and splices its first stream to `127.0.0.1:<forward>`.
//! `connect` opens the connection and pipes that stream on its own standard input and
//! output, so it can be an SSH `ProxyCommand`.

use std::net::{SocketAddr, UdpSocket};
use std::sync::Arc;

use quinn::crypto::rustls::{QuicClientConfig, QuicServerConfig};
use quinn::{ClientConfig, Endpoint, EndpointConfig, ServerConfig, TokioRuntime};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// The protocol name, so a peer speaking something else is refused by the handshake.
const ALPN: &[u8] = b"letify/1";
/// Seconds to wait for the peer, matching the pipeline's own connect timeout.
const TIMEOUT: u64 = 30;

struct Options {
    mode: String,
    bind: u16,
    peer: SocketAddr,
    token: Vec<u8>,
    forward: u16,
}

fn parse() -> Result<Options, String> {
    let mut args = std::env::args().skip(1);
    let mode = args.next().ok_or("expected 'serve' or 'connect'")?;
    let mut bind = 0u16;
    let mut peer = String::new();
    let mut token = String::new();
    let mut forward = 22u16;
    while let Some(flag) = args.next() {
        let value = args.next().ok_or_else(|| format!("{flag} needs a value"))?;
        match flag.as_str() {
            "--bind" => bind = value.parse().map_err(|_| "--bind must be a port")?,
            "--peer" => peer = value,
            "--token" => token = value,
            "--forward" => forward = value.parse().map_err(|_| "--forward must be a port")?,
            other => return Err(format!("unknown flag {other}")),
        }
    }
    if mode != "serve" && mode != "connect" {
        return Err(format!("unknown mode {mode}"));
    }
    let peer: SocketAddr = peer.parse().map_err(|_| "--peer must be <ip>:<port>")?;
    let token = decode(&token).ok_or("--token must be hexadecimal")?;
    if token.is_empty() {
        return Err("--token is required".into());
    }
    Ok(Options { mode, bind, peer, token, forward })
}

fn decode(text: &str) -> Option<Vec<u8>> {
    if text.len() % 2 != 0 {
        return None;
    }
    (0..text.len())
        .step_by(2)
        .map(|index| u8::from_str_radix(&text[index..index + 2], 16).ok())
        .collect()
}

/// A socket bound so the port letify punched can be reused, as the Python side does.
fn bound(port: u16) -> std::io::Result<UdpSocket> {
    let socket = socket2::Socket::new(
        socket2::Domain::IPV4,
        socket2::Type::DGRAM,
        Some(socket2::Protocol::UDP),
    )?;
    socket.set_reuse_address(true)?;
    #[cfg(unix)]
    socket.set_reuse_port(true)?;
    let address: SocketAddr = format!("0.0.0.0:{port}").parse().unwrap();
    socket.bind(&address.into())?;
    Ok(socket.into())
}

/// A self signed certificate, because the token authenticates the peer, not a name.
fn certificate() -> Result<(Vec<rustls::pki_types::CertificateDer<'static>>,
                            rustls::pki_types::PrivateKeyDer<'static>), String> {
    let generated = rcgen::generate_simple_self_signed(vec!["letify".into()])
        .map_err(|error| error.to_string())?;
    let chain = vec![generated.cert.der().clone()];
    let key = rustls::pki_types::PrivateKeyDer::try_from(
        generated.key_pair.serialize_der(),
    )
    .map_err(|error| error.to_string())?;
    Ok((chain, key))
}

/// Accepts any certificate: the token decides who the peer is, as it does for a punch.
#[derive(Debug)]
struct AnyPeer;

impl rustls::client::danger::ServerCertVerifier for AnyPeer {
    fn verify_server_cert(
        &self,
        _end_entity: &rustls::pki_types::CertificateDer<'_>,
        _intermediates: &[rustls::pki_types::CertificateDer<'_>],
        _server_name: &rustls::pki_types::ServerName<'_>,
        _ocsp: &[u8],
        _now: rustls::pki_types::UnixTime,
    ) -> Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        Ok(rustls::client::danger::ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        _message: &[u8],
        _cert: &rustls::pki_types::CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn verify_tls13_signature(
        &self,
        _message: &[u8],
        _cert: &rustls::pki_types::CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        rustls::crypto::ring::default_provider()
            .signature_verification_algorithms
            .supported_schemes()
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> std::process::ExitCode {
    let options = match parse() {
        Ok(options) => options,
        Err(message) => {
            eprintln!("letify-quic: {message}");
            return std::process::ExitCode::from(2);
        }
    };
    match run(options).await {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("letify-quic: {message}");
            std::process::ExitCode::FAILURE
        }
    }
}

async fn run(options: Options) -> Result<(), String> {
    let _ = rustls::crypto::ring::default_provider().install_default();
    let socket = bound(options.bind).map_err(|error| format!("bind {}: {error}", options.bind))?;
    if options.mode == "serve" {
        serve(socket, options).await
    } else {
        connect(socket, options).await
    }
}

async fn serve(socket: UdpSocket, options: Options) -> Result<(), String> {
    let (chain, key) = certificate()?;
    let mut crypto = rustls::ServerConfig::builder_with_provider(
        rustls::crypto::ring::default_provider().into(),
    )
    .with_protocol_versions(&[&rustls::version::TLS13])
    .map_err(|error| error.to_string())?
    .with_no_client_auth()
    .with_single_cert(chain, key)
    .map_err(|error| error.to_string())?;
    crypto.alpn_protocols = vec![ALPN.to_vec()];
    let config = ServerConfig::with_crypto(Arc::new(
        QuicServerConfig::try_from(crypto).map_err(|error| error.to_string())?,
    ));
    let endpoint = Endpoint::new(
        EndpointConfig::default(),
        Some(config),
        socket,
        Arc::new(TokioRuntime),
    )
    .map_err(|error| error.to_string())?;
    // The peer has to be sent to as well, so the punch holds while QUIC handshakes.
    let incoming = tokio::time::timeout(
        std::time::Duration::from_secs(TIMEOUT),
        endpoint.accept(),
    )
    .await
    .map_err(|_| "no QUIC connection arrived".to_string())?
    .ok_or("the endpoint closed")?;
    let connection = incoming.await.map_err(|error| error.to_string())?;
    let (mut send, mut recv) = connection
        .accept_bi()
        .await
        .map_err(|error| error.to_string())?;
    let mut seen = vec![0u8; options.token.len()];
    recv.read_exact(&mut seen)
        .await
        .map_err(|error| error.to_string())?;
    if seen != options.token {
        return Err("the peer presented the wrong token".into());
    }
    send.write_all(&options.token)
        .await
        .map_err(|error| error.to_string())?;
    let stream = tokio::net::TcpStream::connect(("127.0.0.1", options.forward))
        .await
        .map_err(|error| format!("connect 127.0.0.1:{}: {error}", options.forward))?;
    let (mut inbound, mut outbound) = stream.into_split();
    let up = tokio::io::copy(&mut recv, &mut outbound);
    let down = async {
        let mut buffer = vec![0u8; 1 << 16];
        loop {
            let read = inbound.read(&mut buffer).await?;
            if read == 0 {
                break;
            }
            send.write_all(&buffer[..read])
                .await
                .map_err(std::io::Error::other)?;
        }
        Ok::<(), std::io::Error>(())
    };
    let _ = tokio::join!(up, down);
    Ok(())
}

async fn connect(socket: UdpSocket, options: Options) -> Result<(), String> {
    let mut crypto = rustls::ClientConfig::builder_with_provider(
        rustls::crypto::ring::default_provider().into(),
    )
    .with_protocol_versions(&[&rustls::version::TLS13])
    .map_err(|error| error.to_string())?
    .dangerous()
    .with_custom_certificate_verifier(Arc::new(AnyPeer))
    .with_no_client_auth();
    crypto.alpn_protocols = vec![ALPN.to_vec()];
    let mut endpoint = Endpoint::new(
        EndpointConfig::default(),
        None,
        socket,
        Arc::new(TokioRuntime),
    )
    .map_err(|error| error.to_string())?;
    endpoint.set_default_client_config(ClientConfig::new(Arc::new(
        QuicClientConfig::try_from(crypto).map_err(|error| error.to_string())?,
    )));
    let connecting = endpoint
        .connect(options.peer, "letify")
        .map_err(|error| error.to_string())?;
    let connection = tokio::time::timeout(
        std::time::Duration::from_secs(TIMEOUT),
        connecting,
    )
    .await
    .map_err(|_| "the QUIC handshake did not complete".to_string())?
    .map_err(|error| error.to_string())?;
    let (mut send, mut recv) = connection
        .open_bi()
        .await
        .map_err(|error| error.to_string())?;
    send.write_all(&options.token)
        .await
        .map_err(|error| error.to_string())?;
    let mut seen = vec![0u8; options.token.len()];
    recv.read_exact(&mut seen)
        .await
        .map_err(|error| error.to_string())?;
    if seen != options.token {
        return Err("the peer presented the wrong token".into());
    }
    let mut input = tokio::io::stdin();
    let mut output = tokio::io::stdout();
    let up = async {
        let mut buffer = vec![0u8; 1 << 16];
        loop {
            let read = input.read(&mut buffer).await?;
            if read == 0 {
                break;
            }
            send.write_all(&buffer[..read])
                .await
                .map_err(std::io::Error::other)?;
        }
        Ok::<(), std::io::Error>(())
    };
    let down = async {
        let mut buffer = vec![0u8; 1 << 16];
        loop {
            let read = match recv.read(&mut buffer).await {
                Ok(Some(read)) => read,
                Ok(None) => break,
                Err(error) => return Err(std::io::Error::other(error)),
            };
            output.write_all(&buffer[..read]).await?;
            output.flush().await?;
        }
        Ok::<(), std::io::Error>(())
    };
    let _ = tokio::join!(up, down);
    Ok(())
}
