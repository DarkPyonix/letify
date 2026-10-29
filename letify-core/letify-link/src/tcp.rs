//! TCP subflows created by the library: the socket options a letify-link path needs. It owns
//! nothing else; the session treats a TCP stream like any other transport.

use std::io;
use std::net::SocketAddr;
use std::time::Duration;

use socket2::{SockRef, TcpKeepalive};
use tokio::net::TcpStream;

/// Idle time before the kernel sends its first TCP keepalive probe.
pub const KEEPIDLE: Duration = Duration::from_secs(20);
/// Time between TCP keepalive probes.
pub const KEEPINTVL: Duration = Duration::from_secs(10);

/// Set `SO_KEEPALIVE` with `TCP_KEEPIDLE` 20 s and `TCP_KEEPINTVL` 10 s, and `TCP_NODELAY`,
/// because a session writes whole records and wants none of them held back.
pub fn tune(stream: &TcpStream) -> io::Result<()> {
    stream.set_nodelay(true)?;
    let keepalive = TcpKeepalive::new()
        .with_time(KEEPIDLE)
        .with_interval(KEEPINTVL);
    SockRef::from(stream).set_tcp_keepalive(&keepalive)
}

/// Connect to `addr` and tune the socket.
pub async fn connect(addr: SocketAddr) -> io::Result<TcpStream> {
    let stream = TcpStream::connect(addr).await?;
    tune(&stream)?;
    Ok(stream)
}
