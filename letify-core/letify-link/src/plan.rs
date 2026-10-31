//! What a punch tries, given how each side's NAT maps ports. It owns the choice of how many UDP
//! sockets to open, which destinations to probe and which TCP ports to offer. It does not own
//! the sockets, the exchange of offers ([`rendezvous`](crate::rendezvous)) or the probing
//! itself ([`punch`](crate::punch)).

use std::net::{IpAddr, SocketAddr};

use crate::punch::{BIRTHDAY_PORTS, MAX_PUNCH_SOCKETS, PREDICT_WINDOW};
use crate::stun::NatType;

/// The most destinations a plan with fixed addresses probes.
pub const MAX_FIXED_DESTINATIONS: usize = 64;

/// Where the UDP probes of one side go.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Destinations {
    /// These addresses: the peer's mappings, with predicted ports added when it is symmetric.
    Fixed(Vec<SocketAddr>),
    /// `count` random ports on `ip`, because the peer's mapping port cannot be predicted.
    Birthday {
        /// The peer's public address.
        ip: IpAddr,
        /// How many random ports to probe.
        count: usize,
    },
}

/// What one side does in a UDP punch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct UdpPlan {
    /// UDP sockets this side probes from.
    pub sockets: usize,
    /// Where they probe.
    pub dests: Destinations,
}

/// A NAT whose mapped ports follow no pattern, which only the many-port attempt can reach.
pub fn is_unpredictable(nat: NatType) -> bool {
    matches!(nat, NatType::Symmetric { step: 0 })
}

/// The UDP plan of a side whose NAT is `own`, probing a peer whose NAT is `peer` and whose
/// offered mappings are `peer_maps`. `requested` is the socket count the caller asked for.
///
/// A side behind an unpredictable NAT opens [`BIRTHDAY_PORTS`] sockets, because each one makes a
/// mapping of its own. A side facing an unpredictable peer probes [`BIRTHDAY_PORTS`] random
/// ports on the peer's address. Otherwise it probes the peer's mappings, and the next ports the
/// peer's NAT is expected to hand out when it is symmetric with a step.
pub fn udp_plan(
    own: NatType,
    peer: NatType,
    requested: usize,
    peer_maps: &[SocketAddr],
) -> UdpPlan {
    let sockets = if is_unpredictable(own) {
        BIRTHDAY_PORTS
    } else {
        requested.clamp(1, MAX_PUNCH_SOCKETS)
    };
    let dests = match peer_maps.first() {
        Some(first) if is_unpredictable(peer) => Destinations::Birthday {
            ip: first.ip(),
            count: BIRTHDAY_PORTS,
        },
        _ => Destinations::Fixed(
            peer_maps
                .iter()
                .flat_map(|m| crate::punch::predicted_destinations(*m, peer, PREDICT_WINDOW))
                .take(MAX_FIXED_DESTINATIONS)
                .collect(),
        ),
    };
    UdpPlan { sockets, dests }
}

/// The mappings to offer for TCP sockets bound to `locals` (their local ports). A TCP socket
/// cannot be observed over UDP, so a socket is offered on `public_ip` with its own port. When
/// the NAT is symmetric with a step, socket `i` is offered at `udp_last` advanced by
/// `step * (i + 1)`, the port the NAT is expected to hand out next.
pub fn tcp_mappings(
    public_ip: IpAddr,
    locals: &[u16],
    nat: NatType,
    udp_last: Option<SocketAddr>,
) -> Vec<SocketAddr> {
    let predicted = match (nat, udp_last) {
        (NatType::Symmetric { step }, Some(last)) if step != 0 => {
            crate::punch::predict_ports(last.port(), step, locals.len())
        }
        _ => Vec::new(),
    };
    locals
        .iter()
        .enumerate()
        .map(|(i, port)| SocketAddr::new(public_ip, predicted.get(i).copied().unwrap_or(*port)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn addr(text: &str) -> SocketAddr {
        text.parse().unwrap()
    }

    #[test]
    fn a_symmetric_peer_with_a_step_is_probed_at_its_predicted_ports() {
        let peer = NatType::Symmetric { step: 2 };
        let plan = udp_plan(
            NatType::EndpointIndependent,
            peer,
            1,
            &[addr("203.0.113.7:40000")],
        );
        let Destinations::Fixed(dests) = plan.dests else {
            panic!("{plan:?}")
        };
        assert_eq!(dests.len(), PREDICT_WINDOW);
        assert_eq!(dests[0], addr("203.0.113.7:40002"));
        assert_eq!(dests[1], addr("203.0.113.7:40004"));
        assert_eq!(plan.sockets, 1);
    }

    #[test]
    fn a_peer_that_is_not_symmetric_is_probed_at_its_mapping_alone() {
        for peer in [NatType::EndpointIndependent, NatType::Unknown] {
            let plan = udp_plan(peer, peer, 2, &[addr("203.0.113.7:40000")]);
            assert_eq!(
                plan.dests,
                Destinations::Fixed(vec![addr("203.0.113.7:40000")])
            );
            assert_eq!(plan.sockets, 2);
        }
    }

    #[test]
    fn an_unpredictable_peer_gets_the_many_port_attempt() {
        let peer = NatType::Symmetric { step: 0 };
        let plan = udp_plan(
            NatType::EndpointIndependent,
            peer,
            1,
            &[addr("203.0.113.7:40000")],
        );
        assert_eq!(
            plan.dests,
            Destinations::Birthday {
                ip: "203.0.113.7".parse().unwrap(),
                count: BIRTHDAY_PORTS
            }
        );
        assert_eq!(plan.sockets, 1);
    }

    #[test]
    fn an_unpredictable_own_nat_opens_many_sockets() {
        let own = NatType::Symmetric { step: 0 };
        let plan = udp_plan(
            own,
            NatType::EndpointIndependent,
            1,
            &[addr("203.0.113.7:1")],
        );
        assert_eq!(plan.sockets, BIRTHDAY_PORTS);
        assert_eq!(plan.dests, Destinations::Fixed(vec![addr("203.0.113.7:1")]));
        let both = udp_plan(own, own, 1, &[addr("203.0.113.7:1")]);
        assert_eq!(both.sockets, BIRTHDAY_PORTS);
        assert!(matches!(both.dests, Destinations::Birthday { .. }));
    }

    #[test]
    fn tcp_ports_are_offered_as_bound_unless_the_nat_has_a_step() {
        let ip: IpAddr = "203.0.113.9".parse().unwrap();
        let plain = tcp_mappings(ip, &[5000, 5001], NatType::EndpointIndependent, None);
        assert_eq!(
            plain,
            vec![addr("203.0.113.9:5000"), addr("203.0.113.9:5001")]
        );
        let stepped = tcp_mappings(
            ip,
            &[5000, 5001],
            NatType::Symmetric { step: 4 },
            Some(addr("203.0.113.9:30000")),
        );
        assert_eq!(
            stepped,
            vec![addr("203.0.113.9:30004"), addr("203.0.113.9:30008")]
        );
        let none = tcp_mappings(
            ip,
            &[5000],
            NatType::Symmetric { step: 0 },
            Some(addr("203.0.113.9:30000")),
        );
        assert_eq!(none, vec![addr("203.0.113.9:5000")]);
    }
}
