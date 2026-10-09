"""Asking the NAT for a port forwarding, so the remote can dial this machine.

Owns the three standard ways to ask: PCP (RFC 6887), NAT-PMP (RFC 6886) and UPnP IGD.
It does not own what the answer is used for; ``strategies.ConnectBack`` does, and the
answer is shaped as the endpoint that strategy needs. Spec "Asking the NAT for a mapping".

Nothing here raises for a device that is absent or unwilling. A NAT that does not speak
one of these protocols is the normal case, and the caller reads that as None.
"""

from __future__ import annotations

import re
import secrets
import socket
import struct
import urllib.request
from dataclasses import dataclass
from pathlib import Path

#: Seconds a mapping is asked to live. One hour outlasts a session's setup and is short
#: enough that a mapping letify fails to release expires on its own.
LIFETIME = 3600
#: Seconds to wait for a device on the local link. One is generous for a single hop, and a
#: device that is slower than this is not going to carry a session.
TIMEOUT = 1.0
#: Where PCP and NAT-PMP both listen, by their RFCs.
CONTROL_PORT = 5351
#: The SSDP multicast group and port UPnP discovery uses.
SSDP = ("239.255.255.250", 1900)
_ROUTE_TABLE = Path("/proc/net/route")


@dataclass(frozen=True)
class Mapping:
    """A forwarding a NAT granted: dial ``address``:``port`` to reach the local port."""

    address: str
    port: int
    protocol: str
    lifetime: int


def default_gateway(table: Path | None = None) -> str | None:
    """The next hop of the default route, or None when there is none to read.

    Linux lists it in ``/proc/net/route`` as a little endian hexadecimal word, so the
    bytes are reversed to make dotted quad.
    """
    path = table if table is not None else _ROUTE_TABLE
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines[1:]:
        fields = line.split()
        if len(fields) < 3 or fields[1] != "00000000":
            continue
        try:
            packed = struct.pack("<I", int(fields[2], 16))
        except ValueError:
            continue
        if packed == b"\0\0\0\0":
            continue
        return socket.inet_ntoa(packed)
    return None


def _ask(gateway: tuple[str, int], payload: bytes, timeout: float) -> bytes | None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.settimeout(timeout)
        sock.sendto(payload, gateway)
        return sock.recv(2048)
    except OSError:
        return None
    finally:
        sock.close()


def nat_pmp(
    gateway: tuple[str, int] | str,
    local_port: int,
    *,
    timeout: float = TIMEOUT,
    lifetime: int = LIFETIME,
) -> Mapping | None:
    """Ask for a TCP mapping with NAT-PMP, RFC 6886.

    The request is version 0, opcode 2 for TCP, the internal port, the external port asked
    for and the lifetime. The reply adds 128 to the opcode and carries a result code, where
    anything but zero is a refusal.
    """
    where = (gateway, CONTROL_PORT) if isinstance(gateway, str) else gateway
    request = struct.pack("!BBHHHI", 0, 2, 0, local_port, local_port, lifetime)
    reply = _ask(where, request, timeout)
    if reply is None or len(reply) < 16:
        return None
    version, opcode, result, _epoch, internal, mapped, granted = struct.unpack(
        "!BBHIHHI", reply[:16]
    )
    if version != 0 or opcode != 130 or result != 0 or internal != local_port or not mapped:
        return None
    address = _external_address_nat_pmp(where, timeout) or where[0]
    return Mapping(address=address, port=mapped, protocol="nat_pmp", lifetime=granted)


def _external_address_nat_pmp(gateway: tuple[str, int], timeout: float) -> str | None:
    """The device's external address, opcode 0 of RFC 6886."""
    reply = _ask(gateway, struct.pack("!BB", 0, 0), timeout)
    if reply is None or len(reply) < 12:
        return None
    version, opcode, result = struct.unpack("!BBH", reply[:4])
    if version != 0 or opcode != 128 or result != 0:
        return None
    return socket.inet_ntoa(reply[8:12])


def pcp(
    gateway: tuple[str, int] | str,
    local_port: int,
    *,
    timeout: float = TIMEOUT,
    lifetime: int = LIFETIME,
) -> Mapping | None:
    """Ask for a TCP mapping with PCP, RFC 6887.

    PCP supersedes NAT-PMP and shares its port, so it is asked first: a NAT-PMP only
    device answers a version 2 request with an unsupported version error rather than
    silence, which is a clear no.
    """
    where = (gateway, CONTROL_PORT) if isinstance(gateway, str) else gateway
    client = _local_address(where[0])
    head = struct.pack("!BBHI", 2, 1, 0, lifetime) + _mapped_ipv6(client)
    nonce = secrets.token_bytes(12)
    body = nonce + bytes([6, 0, 0, 0]) + struct.pack("!HH", local_port, local_port)
    body += _mapped_ipv6("0.0.0.0")
    reply = _ask(where, head + body, timeout)
    if reply is None or len(reply) < 60:
        return None
    version, opcode, _reserved, result = struct.unpack("!BBBB", reply[:4])
    if version != 2 or opcode != (1 | 0x80) or result != 0:
        return None
    granted = struct.unpack("!I", reply[4:8])[0]
    if reply[24:36] != nonce:
        return None
    internal, mapped = struct.unpack("!HH", reply[40:44])
    if internal != local_port or not mapped:
        return None
    address = socket.inet_ntoa(reply[56:60]) if reply[44:56] == _MAPPED_PREFIX else where[0]
    return Mapping(address=address, port=mapped, protocol="pcp", lifetime=granted)


#: The prefix of an IPv4 address written as IPv6, RFC 6887 section 5.
_MAPPED_PREFIX = bytes(10) + b"\xff\xff"


def _mapped_ipv6(address: str) -> bytes:
    return _MAPPED_PREFIX + socket.inet_aton(address)


def _local_address(toward: str) -> str:
    """This machine's address on the route toward ``toward``."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect((toward, CONTROL_PORT))
        return sock.getsockname()[0]
    except OSError:
        return "0.0.0.0"
    finally:
        sock.close()


_SERVICES = (
    "urn:schemas-upnp-org:service:WANIPConnection:1",
    "urn:schemas-upnp-org:service:WANPPPConnection:1",
)


def upnp(local_port: int, *, timeout: float = TIMEOUT, lifetime: int = LIFETIME) -> Mapping | None:
    """Ask for a TCP mapping with UPnP IGD: SSDP discovery, then ``AddPortMapping``."""
    location = _ssdp_location(timeout)
    if location is None:
        return None
    description = _fetch(location, timeout)
    if description is None:
        return None
    control = _control_url(location, description)
    if control is None:
        return None
    service = next((name for name in _SERVICES if name in description), _SERVICES[0])
    client = _local_address(location.split("/")[2].split(":")[0])
    body = (
        f'<u:AddPortMapping xmlns:u="{service}">'
        f"<NewRemoteHost></NewRemoteHost><NewExternalPort>{local_port}</NewExternalPort>"
        f"<NewProtocol>TCP</NewProtocol><NewInternalPort>{local_port}</NewInternalPort>"
        f"<NewInternalClient>{client}</NewInternalClient><NewEnabled>1</NewEnabled>"
        f"<NewPortMappingDescription>letify</NewPortMappingDescription>"
        f"<NewLeaseDuration>{lifetime}</NewLeaseDuration></u:AddPortMapping>"
    )
    if _soap(control, service, "AddPortMapping", body, timeout) is None:
        return None
    address = _upnp_external_address(control, service, timeout)
    if address is None:
        return None
    return Mapping(address=address, port=local_port, protocol="upnp", lifetime=lifetime)


def _upnp_external_address(control: str, service: str, timeout: float) -> str | None:
    body = f'<u:GetExternalIPAddress xmlns:u="{service}"></u:GetExternalIPAddress>'
    answer = _soap(control, service, "GetExternalIPAddress", body, timeout)
    if answer is None:
        return None
    found = re.search(r"<NewExternalIPAddress>([^<]+)</NewExternalIPAddress>", answer)
    return found.group(1) if found else None


def _ssdp_location(timeout: float) -> str | None:
    search = (
        "M-SEARCH * HTTP/1.1\r\n"
        f"HOST: {SSDP[0]}:{SSDP[1]}\r\n"
        'MAN: "ssdp:discover"\r\n'
        "MX: 1\r\n"
        "ST: urn:schemas-upnp-org:device:InternetGatewayDevice:1\r\n\r\n"
    ).encode("ascii")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
        sock.settimeout(timeout)
        sock.sendto(search, SSDP)
        data = sock.recv(4096).decode("latin-1")
    except OSError:
        return None
    finally:
        sock.close()
    found = re.search(r"(?im)^location:\s*(\S+)", data)
    return found.group(1) if found else None


def _fetch(url: str, timeout: float) -> str | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as answer:
            return answer.read(1 << 18).decode("latin-1")
    except Exception:
        return None


def _control_url(location: str, description: str) -> str | None:
    found = re.search(r"<controlURL>([^<]+)</controlURL>", description)
    if not found:
        return None
    path = found.group(1)
    if path.startswith("http"):
        return path
    base = "/".join(location.split("/")[:3])
    return base + (path if path.startswith("/") else "/" + path)


def _soap(url: str, service: str, action: str, body: str, timeout: float) -> str | None:
    envelope = (
        '<?xml version="1.0"?>'
        '<s:Envelope xmlns:s="http://schemas.xmlsoap.org/soap/envelope/" '
        's:encodingStyle="http://schemas.xmlsoap.org/soap/encoding/">'
        f"<s:Body>{body}</s:Body></s:Envelope>"
    ).encode()
    request = urllib.request.Request(
        url,
        data=envelope,
        headers={
            "Content-Type": 'text/xml; charset="utf-8"',
            "SOAPAction": f'"{service}#{action}"',
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as answer:
            return answer.read(1 << 16).decode("latin-1")
    except Exception:
        return None


def request(local_port: int, *, timeout: float = TIMEOUT) -> Mapping | None:
    """A forwarding to ``local_port`` from whichever protocol the device speaks.

    PCP first because it supersedes NAT-PMP on the same port, then NAT-PMP, then UPnP,
    which costs a multicast discovery and two HTTP requests. Spec "Asking the NAT for a
    mapping".
    """
    gateway = default_gateway()
    if gateway is not None:
        for ask in (pcp, nat_pmp):
            found = ask(gateway, local_port, timeout=timeout)
            if found is not None:
                return found
    return upnp(local_port, timeout=timeout)


def release(mapping: Mapping, local_port: int, *, timeout: float = TIMEOUT) -> None:
    """Give a mapping back, with the protocol that granted it and a lifetime of zero."""
    gateway = default_gateway()
    if mapping.protocol == "pcp" and gateway:
        pcp(gateway, local_port, timeout=timeout, lifetime=0)
    elif mapping.protocol == "nat_pmp" and gateway:
        nat_pmp(gateway, local_port, timeout=timeout, lifetime=0)
    elif mapping.protocol == "upnp":
        _release_upnp(mapping, local_port, timeout)


def _release_upnp(mapping: Mapping, local_port: int, timeout: float) -> None:
    location = _ssdp_location(timeout)
    description = _fetch(location, timeout) if location else None
    control = _control_url(location, description) if location and description else None
    if control is None or description is None:
        return
    service = next((name for name in _SERVICES if name in description), _SERVICES[0])
    body = (
        f'<u:DeletePortMapping xmlns:u="{service}">'
        f"<NewRemoteHost></NewRemoteHost><NewExternalPort>{mapping.port}</NewExternalPort>"
        f"<NewProtocol>TCP</NewProtocol></u:DeletePortMapping>"
    )
    _soap(control, service, "DeletePortMapping", body, timeout)


__all__ = ["LIFETIME", "Mapping", "default_gateway", "nat_pmp", "pcp", "release", "request", "upnp"]
