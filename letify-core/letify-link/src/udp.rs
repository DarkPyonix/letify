//! UDP paths: a reliable, ordered byte pipe over one punched UDP address pair, which the
//! session uses as a subflow. It owns the QUIC connection (quinn) and its tuning: BBR, the
//! windows, the socket buffers, the throwaway certificates. It owns no identity, no secrecy
//! and no framing: the Noise session above authenticates and encrypts everything it writes,
//! so the certificate check is skipped on purpose.
//!
//! One QUIC connection carries one bidirectional stream, and that stream is the transport.
//! A [`UdpListener`] serves many paths on one socket, a [`UdpPath::accept`] serves one.

use std::io;
use std::net::{SocketAddr, UdpSocket};
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Duration;

use quinn::congestion::BbrConfig;
use quinn::crypto::rustls::{QuicClientConfig, QuicServerConfig};
use quinn::{
    ClientConfig, Connection, Endpoint, EndpointConfig, IdleTimeout, RecvStream, SendStream,
    ServerConfig, TokioRuntime, TransportConfig, VarInt,
};
use socket2::SockRef;
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};
use tokio::sync::mpsc;
use tokio::task::AbortHandle;

/// The protocol name, so a peer that speaks something else is refused by the handshake.
const ALPN: &[u8] = b"letify-link/udp/1";
/// The first bytes on the stream. They make the accepting side see the stream at once,
/// because QUIC announces a stream only with its first data.
const STREAM_MAGIC: &[u8; 4] = b"LLUP";

/// Settings of a UDP path. The defaults carry 200 ms x 50 MiB/s, a bandwidth delay product
/// of 10 MiB, with room to spare.
#[derive(Clone, Debug)]
pub struct UdpConfig {
    /// Bytes the receiver accepts on the stream and on the connection before the reader
    /// consumes them. Default 16 MiB.
    pub receive_window: u32,
    /// Bytes the sender buffers, sent and not yet acknowledged included. Default 16 MiB.
    pub send_window: u64,
    /// Requested size of the socket send and receive buffers. The kernel may grant less.
    /// Default 8 MiB.
    pub socket_buffer: usize,
    /// A path with no packet for this long is dead. Default 30 s.
    pub idle_timeout: Duration,
    /// A QUIC keep-alive every this long keeps an idle path up. Default 10 s.
    pub keep_alive: Duration,
    /// Time limit for the QUIC handshake and the stream. Default 10 s.
    pub connect_timeout: Duration,
}

impl Default for UdpConfig {
    fn default() -> Self {
        UdpConfig {
            receive_window: 16 << 20,
            send_window: 16 << 20,
            socket_buffer: 8 << 20,
            idle_timeout: Duration::from_secs(30),
            keep_alive: Duration::from_secs(10),
            connect_timeout: Duration::from_secs(10),
        }
    }
}

fn other(e: impl std::fmt::Display) -> io::Error {
    io::Error::other(e.to_string())
}

fn transport(cfg: &UdpConfig) -> io::Result<Arc<TransportConfig>> {
    let mut t = TransportConfig::default();
    t.congestion_controller_factory(Arc::new(BbrConfig::default()));
    t.stream_receive_window(VarInt::from_u32(cfg.receive_window));
    t.receive_window(VarInt::from_u32(cfg.receive_window));
    t.send_window(cfg.send_window);
    t.max_concurrent_bidi_streams(1u8.into());
    t.max_concurrent_uni_streams(0u8.into());
    let idle = IdleTimeout::try_from(cfg.idle_timeout).map_err(other)?;
    t.max_idle_timeout(Some(idle));
    t.keep_alive_interval(Some(cfg.keep_alive));
    Ok(Arc::new(t))
}

fn provider() -> Arc<rustls::crypto::CryptoProvider> {
    Arc::new(rustls::crypto::ring::default_provider())
}

fn server_config(cfg: &UdpConfig) -> io::Result<ServerConfig> {
    let generated = rcgen::generate_simple_self_signed(vec!["letify".into()]).map_err(other)?;
    let chain = vec![generated.cert.der().clone()];
    let key = rustls::pki_types::PrivateKeyDer::try_from(generated.key_pair.serialize_der())
        .map_err(other)?;
    let mut crypto = rustls::ServerConfig::builder_with_provider(provider())
        .with_protocol_versions(&[&rustls::version::TLS13])
        .map_err(other)?
        .with_no_client_auth()
        .with_single_cert(chain, key)
        .map_err(other)?;
    crypto.alpn_protocols = vec![ALPN.to_vec()];
    let mut server =
        ServerConfig::with_crypto(Arc::new(QuicServerConfig::try_from(crypto).map_err(other)?));
    server.transport_config(transport(cfg)?);
    Ok(server)
}

fn client_config(cfg: &UdpConfig) -> io::Result<ClientConfig> {
    let mut crypto = rustls::ClientConfig::builder_with_provider(provider())
        .with_protocol_versions(&[&rustls::version::TLS13])
        .map_err(other)?
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(AnyPeer))
        .with_no_client_auth();
    crypto.alpn_protocols = vec![ALPN.to_vec()];
    let mut client =
        ClientConfig::new(Arc::new(QuicClientConfig::try_from(crypto).map_err(other)?));
    client.transport_config(transport(cfg)?);
    Ok(client)
}

/// Accepts any certificate. The Noise session above decides who the peer is.
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
        provider()
            .signature_verification_algorithms
            .supported_schemes()
    }
}

/// Ask for larger socket buffers. The kernel clamps the request, so a failure is not an error.
fn size_buffers(socket: &UdpSocket, bytes: usize) {
    let sock = SockRef::from(socket);
    let _ = sock.set_recv_buffer_size(bytes);
    let _ = sock.set_send_buffer_size(bytes);
}

fn endpoint(
    socket: UdpSocket,
    server: Option<ServerConfig>,
    cfg: &UdpConfig,
) -> io::Result<Endpoint> {
    size_buffers(&socket, cfg.socket_buffer);
    Endpoint::new(
        EndpointConfig::default(),
        server,
        socket,
        Arc::new(TokioRuntime),
    )
}

/// One UDP path: a QUIC connection and its one stream, usable as a session subflow.
pub struct UdpPath {
    send: SendStream,
    recv: RecvStream,
    conn: Connection,
    /// Keeps the endpoint, and with it the socket, alive as long as the path.
    _endpoint: Endpoint,
}

impl UdpPath {
    /// Connect over `socket` to `peer` with the default settings.
    pub async fn connect(socket: UdpSocket, peer: SocketAddr) -> io::Result<UdpPath> {
        Self::connect_with(socket, peer, &UdpConfig::default()).await
    }

    /// Connect over `socket` to `peer`. The socket is usually the one a punch opened.
    pub async fn connect_with(
        socket: UdpSocket,
        peer: SocketAddr,
        cfg: &UdpConfig,
    ) -> io::Result<UdpPath> {
        let endpoint = endpoint(socket, None, cfg)?;
        let connecting = endpoint
            .connect_with(client_config(cfg)?, peer, "letify")
            .map_err(other)?;
        let timeout = || io::Error::new(io::ErrorKind::TimedOut, "the UDP path did not open");
        let conn = tokio::time::timeout(cfg.connect_timeout, connecting)
            .await
            .map_err(|_| timeout())?
            .map_err(other)?;
        let (mut send, recv) = tokio::time::timeout(cfg.connect_timeout, conn.open_bi())
            .await
            .map_err(|_| timeout())?
            .map_err(other)?;
        send.write_all(STREAM_MAGIC).await.map_err(other)?;
        Ok(UdpPath {
            send,
            recv,
            conn,
            _endpoint: endpoint,
        })
    }

    /// Accept one path from `peer` over `socket` with the default settings.
    pub async fn accept(socket: UdpSocket, peer: SocketAddr) -> io::Result<UdpPath> {
        Self::accept_with(socket, peer, &UdpConfig::default()).await
    }

    /// Accept one path from `peer` over `socket`. A QUIC connection from any other address is
    /// refused.
    pub async fn accept_with(
        socket: UdpSocket,
        peer: SocketAddr,
        cfg: &UdpConfig,
    ) -> io::Result<UdpPath> {
        let mut listener = UdpListener::new(socket, Some(peer), cfg)?;
        listener.accept().await
    }

    /// The local address of the socket.
    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        self._endpoint.local_addr()
    }

    /// The address of the peer.
    pub fn peer_addr(&self) -> SocketAddr {
        self.conn.remote_address()
    }
}

impl Drop for UdpPath {
    /// Finish the stream so queued bytes still arrive, then close the connection once the peer
    /// has closed it or two seconds passed.
    fn drop(&mut self) {
        let _ = self.send.finish();
        let conn = self.conn.clone();
        if let Ok(rt) = tokio::runtime::Handle::try_current() {
            rt.spawn(async move {
                let _ = tokio::time::timeout(Duration::from_secs(2), conn.closed()).await;
                conn.close(VarInt::from_u32(0), b"");
            });
        } else {
            conn.close(VarInt::from_u32(0), b"");
        }
    }
}

impl AsyncRead for UdpPath {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        Pin::new(&mut self.recv).poll_read(cx, buf)
    }
}

impl AsyncWrite for UdpPath {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        Pin::new(&mut self.send).poll_write(cx, buf).map_err(other)
    }

    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.send).poll_flush(cx)
    }

    fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.send).poll_shutdown(cx)
    }
}

/// Serves UDP paths on one socket. A task accepts QUIC connections and finishes each one's
/// stream in its own task, so a peer that connects and never opens its stream holds up no one.
pub struct UdpListener {
    paths: mpsc::Receiver<UdpPath>,
    local: SocketAddr,
    task: AbortHandle,
}

impl Drop for UdpListener {
    fn drop(&mut self) {
        self.task.abort();
    }
}

impl UdpListener {
    /// Serve over `socket`. With `from` set, a QUIC connection from any other address is
    /// refused; with `None` any address is served.
    pub fn new(socket: UdpSocket, from: Option<SocketAddr>, cfg: &UdpConfig) -> io::Result<Self> {
        let local = socket.local_addr()?;
        let endpoint = endpoint(socket, Some(server_config(cfg)?), cfg)?;
        let (tx, paths) = mpsc::channel(16);
        let limit = cfg.connect_timeout;
        let task = tokio::spawn(async move {
            while let Some(incoming) = endpoint.accept().await {
                if from.is_some_and(|f| f != incoming.remote_address()) {
                    incoming.refuse();
                    continue;
                }
                let (tx, endpoint) = (tx.clone(), endpoint.clone());
                tokio::spawn(async move {
                    let opened = tokio::time::timeout(limit, async {
                        let conn = incoming.await.map_err(other)?;
                        let (send, mut recv) = conn.accept_bi().await.map_err(other)?;
                        let mut magic = [0u8; 4];
                        recv.read_exact(&mut magic).await.map_err(other)?;
                        if &magic != STREAM_MAGIC {
                            return Err(other("the stream does not start with the UDP magic"));
                        }
                        Ok::<_, io::Error>((conn, send, recv))
                    })
                    .await;
                    if let Ok(Ok((conn, send, recv))) = opened {
                        let _ = tx
                            .send(UdpPath {
                                send,
                                recv,
                                conn,
                                _endpoint: endpoint,
                            })
                            .await;
                    }
                });
            }
        });
        Ok(UdpListener {
            paths,
            local,
            task: task.abort_handle(),
        })
    }

    /// The local address of the socket.
    pub fn local_addr(&self) -> SocketAddr {
        self.local
    }

    /// The next path whose stream is open.
    pub async fn accept(&mut self) -> io::Result<UdpPath> {
        self.paths
            .recv()
            .await
            .ok_or_else(|| io::Error::new(io::ErrorKind::BrokenPipe, "the listener closed"))
    }
}
