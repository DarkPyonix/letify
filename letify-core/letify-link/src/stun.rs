//! A minimal STUN client (RFC 5389 Binding over UDP) and the NAT classification built on it.
//! It owns the Binding request and response, the retransmission of a lost request, and the
//! comparison of mappings from several servers. It does not own the choice of local sockets,
//! the exchange of the result (that is the rendezvous) or the punch that uses it.

use std::io;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::time::Duration;

use tokio::net::UdpSocket;
use tokio::time::timeout;

use crate::observe::{AddressObserver, ObserveFuture};

/// The magic cookie of RFC 5389, which starts every message after the type and the length.
const MAGIC_COOKIE: u32 = 0x2112_A442;
const BINDING_REQUEST: u16 = 0x0001;
const BINDING_SUCCESS: u16 = 0x0101;
const XOR_MAPPED_ADDRESS: u16 = 0x0020;
const MAPPED_ADDRESS: u16 = 0x0001;

/// STUN servers asked by default, as `host:port`. The first is the server letify's TCP punch
/// already uses (it also answers over UDP); the other two are public servers of Google and
/// Cloudflare. Classification needs at least [`MIN_SERVERS`] answers.
pub const DEFAULT_SERVERS: [&str; 3] = [
    "stun.nextcloud.com:443",
    "stun.l.google.com:19302",
    "stun.cloudflare.com:3478",
];

/// The fewest servers that must answer for a classification.
pub const MIN_SERVERS: usize = 3;

/// How the NAT maps a local UDP port to a public one, as seen from several servers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NatType {
    /// No classification was made.
    Unknown,
    /// The same public port for every destination: a peer can reach it after one punch.
    EndpointIndependent,
    /// A different public port per destination. `step` is the typical difference between the
    /// ports of consecutive destinations; 0 means the ports follow no pattern.
    Symmetric {
        /// Typical port difference between consecutive mappings, or 0 when unpredictable.
        step: i16,
    },
}

/// Build a Binding request with the 12 byte transaction id `txid`.
pub fn binding_request(txid: &[u8; 12]) -> [u8; 20] {
    let mut out = [0u8; 20];
    out[0..2].copy_from_slice(&BINDING_REQUEST.to_be_bytes());
    out[4..8].copy_from_slice(&MAGIC_COOKIE.to_be_bytes());
    out[8..20].copy_from_slice(txid);
    out
}

/// Read the mapped address from a Binding success response for `txid`. XOR-MAPPED-ADDRESS is
/// preferred over the legacy MAPPED-ADDRESS.
pub fn parse_binding_response(data: &[u8], txid: &[u8; 12]) -> io::Result<SocketAddr> {
    let bad = |why: &str| io::Error::new(io::ErrorKind::InvalidData, format!("STUN: {why}"));
    if data.len() < 20 {
        return Err(bad("reply shorter than the header"));
    }
    let kind = u16::from_be_bytes([data[0], data[1]]);
    let length = u16::from_be_bytes([data[2], data[3]]) as usize;
    if data[4..8] != MAGIC_COOKIE.to_be_bytes() || data[8..20] != txid[..] {
        return Err(bad("reply is for another transaction"));
    }
    if kind != BINDING_SUCCESS {
        return Err(bad("reply is not a Binding success"));
    }
    let mut body = data[20..].get(..length).ok_or_else(|| bad("truncated"))?;
    let mut plain = None;
    while body.len() >= 4 {
        let attr = u16::from_be_bytes([body[0], body[1]]);
        let size = u16::from_be_bytes([body[2], body[3]]) as usize;
        let value = body
            .get(4..4 + size)
            .ok_or_else(|| bad("truncated attribute"))?;
        if attr == XOR_MAPPED_ADDRESS || attr == MAPPED_ADDRESS {
            let xor = attr == XOR_MAPPED_ADDRESS;
            if let Some(addr) = decode_address(value, xor, txid) {
                if xor {
                    return Ok(addr);
                }
                plain = Some(addr);
            }
        }
        body = body.get(4 + size + (4 - size % 4) % 4..).unwrap_or(&[]);
    }
    plain.ok_or_else(|| bad("reply carries no mapped address"))
}

fn decode_address(value: &[u8], xor: bool, txid: &[u8; 12]) -> Option<SocketAddr> {
    let cookie = MAGIC_COOKIE.to_be_bytes();
    let mut port = u16::from_be_bytes([*value.get(2)?, *value.get(3)?]);
    if xor {
        port ^= (MAGIC_COOKIE >> 16) as u16;
    }
    let ip = match *value.get(1)? {
        1 => {
            let mut b: [u8; 4] = value.get(4..8)?.try_into().ok()?;
            if xor {
                b.iter_mut().zip(cookie).for_each(|(x, c)| *x ^= c);
            }
            IpAddr::V4(Ipv4Addr::from(b))
        }
        2 => {
            let mut b: [u8; 16] = value.get(4..20)?.try_into().ok()?;
            if xor {
                let mask: Vec<u8> = cookie.iter().chain(txid.iter()).copied().collect();
                b.iter_mut().zip(mask).for_each(|(x, m)| *x ^= m);
            }
            IpAddr::V6(Ipv6Addr::from(b))
        }
        _ => return None,
    };
    Some(SocketAddr::new(ip, port))
}

/// Number of times a Binding request is sent before the server counts as silent.
const TRIES: u32 = 3;

/// Ask `server` for the public address of `socket`, sending the request again on a lost
/// reply. `wait` is the total time, split over the tries.
pub async fn binding(
    socket: &UdpSocket,
    server: SocketAddr,
    wait: Duration,
) -> io::Result<SocketAddr> {
    let mut txid = [0u8; 12];
    getrandom::getrandom(&mut txid).map_err(io::Error::other)?;
    let request = binding_request(&txid);
    let mut buf = [0u8; 1500];
    for _ in 0..TRIES {
        socket.send_to(&request, server).await?;
        let round = async {
            loop {
                let (n, from) = socket.recv_from(&mut buf).await?;
                if from != server {
                    continue;
                }
                if let Ok(addr) = parse_binding_response(&buf[..n], &txid) {
                    return Ok::<_, io::Error>(addr);
                }
            }
        };
        if let Ok(found) = timeout(wait / TRIES, round).await {
            return found;
        }
    }
    Err(io::Error::new(
        io::ErrorKind::TimedOut,
        format!("no STUN reply from {server}"),
    ))
}

/// Resolve `host:port` names to IPv4 socket addresses, skipping names that do not resolve.
pub async fn resolve(names: &[&str]) -> Vec<SocketAddr> {
    let mut out = Vec::new();
    for name in names {
        if let Ok(mut found) = tokio::net::lookup_host(name).await
            && let Some(addr) = found.find(|a| a.is_ipv4())
        {
            out.push(addr);
        }
    }
    out
}

/// The mapping a server reported for one socket.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mapping {
    /// The STUN server that was asked.
    pub server: SocketAddr,
    /// The public address it saw.
    pub mapped: SocketAddr,
}

/// The result of asking several servers from one socket.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Classification {
    /// The kind of mapping.
    pub nat: NatType,
    /// What each answering server reported, in the order asked.
    pub mappings: Vec<Mapping>,
}

impl Classification {
    /// The last mapped address: the one the next punch from this socket is predicted from.
    pub fn last_mapped(&self) -> Option<SocketAddr> {
        self.mappings.last().map(|m| m.mapped)
    }
}

/// Classify the NAT from the mapped ports reported by consecutive destinations. The same port
/// everywhere is endpoint independent. Otherwise the mapping is symmetric and the step is the
/// median of the differences between consecutive ports (0 when that median is 0 or does not
/// fit an `i16`, which means no usable pattern).
pub fn classify_ports(ports: &[u16]) -> NatType {
    if ports.len() < 2 {
        return NatType::Unknown;
    }
    if ports.iter().all(|p| *p == ports[0]) {
        return NatType::EndpointIndependent;
    }
    let mut diffs: Vec<i32> = ports
        .windows(2)
        .map(|w| w[1] as i32 - w[0] as i32)
        .collect();
    diffs.sort_unstable();
    let step = i16::try_from(diffs[diffs.len() / 2]).unwrap_or(0);
    NatType::Symmetric { step }
}

/// An STUN server as an [`AddressObserver`].
#[derive(Clone, Copy, Debug)]
pub struct StunObserver {
    server: SocketAddr,
    wait: Duration,
}

impl StunObserver {
    /// An observer for `server`, waiting `wait` in total for its reply.
    pub fn new(server: SocketAddr, wait: Duration) -> Self {
        StunObserver { server, wait }
    }
}

impl AddressObserver for StunObserver {
    fn observe<'a>(&'a self, socket: &'a UdpSocket) -> ObserveFuture<'a> {
        Box::pin(async move {
            let mapped = binding(socket, self.server, self.wait).await?;
            Ok(Mapping {
                server: self.server,
                mapped,
            })
        })
    }
}

/// Ask each of `observers` in turn from `socket` and classify the mapping. They are asked one
/// after another, because concurrent requests would take consecutive ports in an order nobody
/// controls and hide the step. It fails unless at least [`MIN_SERVERS`] answer.
pub async fn classify_with(
    socket: &UdpSocket,
    observers: &[&dyn AddressObserver],
) -> io::Result<Classification> {
    let mut mappings = Vec::new();
    for observer in observers {
        if let Ok(mapping) = observer.observe(socket).await {
            mappings.push(mapping);
        }
    }
    if mappings.len() < MIN_SERVERS {
        return Err(io::Error::new(
            io::ErrorKind::TimedOut,
            format!(
                "{} of {} address observers answered, {MIN_SERVERS} are needed to classify the NAT",
                mappings.len(),
                observers.len()
            ),
        ));
    }
    let ports: Vec<u16> = mappings.iter().map(|m| m.mapped.port()).collect();
    Ok(Classification {
        nat: classify_ports(&ports),
        mappings,
    })
}

/// [`classify_with`] over STUN servers, each given `wait` to answer.
pub async fn classify(
    socket: &UdpSocket,
    servers: &[SocketAddr],
    wait: Duration,
) -> io::Result<Classification> {
    let observers: Vec<StunObserver> = servers
        .iter()
        .map(|s| StunObserver::new(*s, wait))
        .collect();
    let refs: Vec<&dyn AddressObserver> = observers
        .iter()
        .map(|o| o as &dyn AddressObserver)
        .collect();
    classify_with(socket, &refs).await
}

#[cfg(test)]
pub(crate) mod fake {
    //! An in-process STUN server on loopback that reports a rewritten port, to stand in for
    //! the NAT in front of the client.

    use super::*;
    use std::sync::{Arc, Mutex};

    /// How the fake NAT assigns public ports.
    #[derive(Clone, Copy)]
    pub enum Nat {
        /// Report the client's own port for every server.
        EndpointIndependent,
        /// Report `base + step * n` for the n-th server contacted by the client.
        Symmetric { base: u16, step: u16 },
    }

    /// One shared NAT table: servers created from it agree on the numbering.
    #[derive(Clone)]
    pub struct FakeNat {
        nat: Nat,
        seen: Arc<Mutex<Vec<SocketAddr>>>,
    }

    impl FakeNat {
        pub fn new(nat: Nat) -> Self {
            FakeNat {
                nat,
                seen: Arc::default(),
            }
        }

        /// Start one server and return its address.
        pub async fn server(&self) -> SocketAddr {
            let socket = UdpSocket::bind("127.0.0.1:0").await.unwrap();
            let addr = socket.local_addr().unwrap();
            let me = self.clone();
            tokio::spawn(async move {
                let mut buf = [0u8; 1500];
                while let Ok((n, from)) = socket.recv_from(&mut buf).await {
                    if n < 20 {
                        continue;
                    }
                    let txid: [u8; 12] = buf[8..20].try_into().unwrap();
                    let port = me.port_for(addr, from);
                    let _ = socket.send_to(&response(&txid, from, port), from).await;
                }
            });
            addr
        }

        fn port_for(&self, server: SocketAddr, client: SocketAddr) -> u16 {
            match self.nat {
                Nat::EndpointIndependent => client.port(),
                Nat::Symmetric { base, step } => {
                    let mut seen = self.seen.lock().unwrap();
                    let index = seen.iter().position(|s| *s == server).unwrap_or_else(|| {
                        seen.push(server);
                        seen.len() - 1
                    });
                    base + step * index as u16
                }
            }
        }
    }

    fn response(txid: &[u8; 12], client: SocketAddr, port: u16) -> Vec<u8> {
        let IpAddr::V4(ip) = client.ip() else {
            unreachable!()
        };
        let cookie = MAGIC_COOKIE.to_be_bytes();
        let mut attr = vec![0, 1];
        attr.extend((port ^ (MAGIC_COOKIE >> 16) as u16).to_be_bytes());
        attr.extend(ip.octets().iter().zip(cookie).map(|(a, c)| a ^ c));
        let mut out = Vec::new();
        out.extend(BINDING_SUCCESS.to_be_bytes());
        out.extend((attr.len() as u16 + 4).to_be_bytes());
        out.extend(cookie);
        out.extend(txid);
        out.extend(XOR_MAPPED_ADDRESS.to_be_bytes());
        out.extend((attr.len() as u16).to_be_bytes());
        out.extend(attr);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::fake::{FakeNat, Nat};
    use super::*;

    async fn servers(nat: &FakeNat, count: usize) -> Vec<SocketAddr> {
        let mut out = Vec::new();
        for _ in 0..count {
            out.push(nat.server().await);
        }
        out
    }

    #[tokio::test]
    async fn an_endpoint_independent_nat_reports_one_port_for_every_server() {
        let nat = FakeNat::new(Nat::EndpointIndependent);
        let servers = servers(&nat, 3).await;
        let socket = UdpSocket::bind("127.0.0.1:0").await.unwrap();
        let found = classify(&socket, &servers, Duration::from_secs(2))
            .await
            .unwrap();
        assert_eq!(found.nat, NatType::EndpointIndependent);
        let port = socket.local_addr().unwrap().port();
        assert!(found.mappings.iter().all(|m| m.mapped.port() == port));
    }

    #[tokio::test]
    async fn a_symmetric_nat_reports_its_port_step() {
        let nat = FakeNat::new(Nat::Symmetric {
            base: 40000,
            step: 3,
        });
        let servers = servers(&nat, 4).await;
        let socket = UdpSocket::bind("127.0.0.1:0").await.unwrap();
        let found = classify(&socket, &servers, Duration::from_secs(2))
            .await
            .unwrap();
        assert_eq!(found.nat, NatType::Symmetric { step: 3 });
        assert_eq!(found.last_mapped().unwrap().port(), 40009);
    }

    #[tokio::test]
    async fn classification_needs_three_answering_servers() {
        let nat = FakeNat::new(Nat::EndpointIndependent);
        let mut list = servers(&nat, 2).await;
        let silent = UdpSocket::bind("127.0.0.1:0").await.unwrap();
        list.push(silent.local_addr().unwrap());
        let socket = UdpSocket::bind("127.0.0.1:0").await.unwrap();
        let err = classify(&socket, &list, Duration::from_millis(300))
            .await
            .unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::TimedOut);
    }

    /// An observer that answers from a script, with no socket traffic at all.
    struct Scripted(Option<SocketAddr>, SocketAddr);

    impl AddressObserver for Scripted {
        fn observe<'a>(&'a self, _socket: &'a UdpSocket) -> ObserveFuture<'a> {
            Box::pin(async move {
                self.0
                    .map(|mapped| Mapping {
                        server: self.1,
                        mapped,
                    })
                    .ok_or_else(|| io::Error::from(io::ErrorKind::TimedOut))
            })
        }
    }

    fn scripted(port: Option<u16>, n: u8) -> Scripted {
        Scripted(
            port.map(|p| SocketAddr::from(([203, 0, 113, 5], p))),
            SocketAddr::from(([198, 51, 100, n], 3478)),
        )
    }

    #[tokio::test]
    async fn classification_runs_over_in_process_observers() {
        let socket = UdpSocket::bind("127.0.0.1:0").await.unwrap();
        let list = [
            scripted(Some(5000), 1),
            scripted(Some(5004), 2),
            scripted(None, 3),
            scripted(Some(5008), 4),
        ];
        let refs: Vec<&dyn AddressObserver> =
            list.iter().map(|o| o as &dyn AddressObserver).collect();
        let found = classify_with(&socket, &refs).await.unwrap();
        assert_eq!(found.nat, NatType::Symmetric { step: 4 });
        assert_eq!(found.mappings.len(), 3);
        let err = classify_with(&socket, &refs[..2]).await.unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::TimedOut);
    }

    #[tokio::test]
    async fn stun_and_qad_observers_classify_together() {
        let nat = FakeNat::new(Nat::EndpointIndependent);
        let stun = StunObserver::new(nat.server().await, Duration::from_secs(2));
        let id = crate::Identity::generate();
        let qad = crate::observe::QadResponder::bind("127.0.0.1:0", id.clone())
            .await
            .unwrap();
        let qad_addr = qad.local_addr();
        tokio::spawn(qad.run());
        let q1 = crate::observe::QadObserver::new(qad_addr, Some(id.node_id()));
        let q2 = crate::observe::QadObserver::new(qad_addr, None);
        let socket = UdpSocket::bind("127.0.0.1:0").await.unwrap();
        let found = classify_with(&socket, &[&stun, &q1, &q2]).await.unwrap();
        assert_eq!(found.nat, NatType::EndpointIndependent);
    }

    #[test]
    fn the_step_is_the_median_so_one_stray_mapping_does_not_hide_it() {
        assert_eq!(
            classify_ports(&[1000, 1002, 1004, 1500, 1502]),
            NatType::Symmetric { step: 2 }
        );
        assert_eq!(
            classify_ports(&[1000, 900, 800]),
            NatType::Symmetric { step: -100 }
        );
        assert_eq!(
            classify_ports(&[1000, 1000, 1000]),
            NatType::EndpointIndependent
        );
        assert_eq!(classify_ports(&[1000]), NatType::Unknown);
    }

    #[test]
    fn a_reply_for_another_transaction_is_refused() {
        let txid = [7u8; 12];
        let mut reply = fake_reply(&txid);
        reply[8] ^= 1;
        assert!(parse_binding_response(&reply, &txid).is_err());
    }

    #[test]
    fn an_ipv6_xor_mapped_address_is_decoded() {
        let txid = [9u8; 12];
        let want: SocketAddr = "[2001:db8::7]:4242".parse().unwrap();
        let IpAddr::V6(ip) = want.ip() else { panic!() };
        let mask: Vec<u8> = MAGIC_COOKIE
            .to_be_bytes()
            .iter()
            .chain(&txid)
            .copied()
            .collect();
        let mut attr = vec![0, 2];
        attr.extend((want.port() ^ (MAGIC_COOKIE >> 16) as u16).to_be_bytes());
        attr.extend(ip.octets().iter().zip(mask).map(|(a, m)| a ^ m));
        let mut msg = Vec::new();
        msg.extend(BINDING_SUCCESS.to_be_bytes());
        msg.extend((attr.len() as u16 + 4).to_be_bytes());
        msg.extend(MAGIC_COOKIE.to_be_bytes());
        msg.extend(txid);
        msg.extend(XOR_MAPPED_ADDRESS.to_be_bytes());
        msg.extend((attr.len() as u16).to_be_bytes());
        msg.extend(attr);
        assert_eq!(parse_binding_response(&msg, &txid).unwrap(), want);
    }

    fn fake_reply(txid: &[u8; 12]) -> Vec<u8> {
        let mut attr = vec![0, 1];
        attr.extend((5555u16 ^ (MAGIC_COOKIE >> 16) as u16).to_be_bytes());
        attr.extend([127 ^ 0x21, 0x12, 0xA4, 1 ^ 0x42]);
        let mut out = Vec::new();
        out.extend(BINDING_SUCCESS.to_be_bytes());
        out.extend((attr.len() as u16 + 4).to_be_bytes());
        out.extend(MAGIC_COOKIE.to_be_bytes());
        out.extend(txid);
        out.extend(XOR_MAPPED_ADDRESS.to_be_bytes());
        out.extend((attr.len() as u16).to_be_bytes());
        out.extend(attr);
        out
    }
}
