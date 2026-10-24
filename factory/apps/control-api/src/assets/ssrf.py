"""
factory/apps/control-api/src/assets/ssrf.py
SSRF Protection Engine & Target URL Validator
"""
import socket
import ipaddress
import urllib.parse
from typing import List, Tuple
from factory_contracts.errors import SSRFBlockedError

# Comprehensive banned CIDR blocks
BLOCKED_NETWORKS = [
    # IPv4 Private & Loopback
    ipaddress.ip_network("127.0.0.0/8"),        # Loopback
    ipaddress.ip_network("10.0.0.0/8"),         # RFC 1918
    ipaddress.ip_network("172.16.0.0/12"),      # RFC 1918
    ipaddress.ip_network("192.168.0.0/16"),     # RFC 1918
    ipaddress.ip_network("169.254.0.0/16"),     # Link-local / Cloud Metadata (169.254.169.254)
    ipaddress.ip_network("100.64.0.0/10"),      # Carrier-grade NAT
    ipaddress.ip_network("0.0.0.0/8"),          # Current network
    ipaddress.ip_network("192.0.0.0/24"),       # IETF Protocol
    ipaddress.ip_network("192.0.2.0/24"),       # TEST-NET-1
    ipaddress.ip_network("198.51.100.0/24"),    # TEST-NET-2
    ipaddress.ip_network("203.0.113.0/24"),     # TEST-NET-3
    ipaddress.ip_network("224.0.0.0/4"),        # Multicast
    ipaddress.ip_network("240.0.0.0/4"),        # Reserved
    ipaddress.ip_network("255.255.255.255/32"), # Broadcast
    # IPv6 Private & Loopback
    ipaddress.ip_network("::1/128"),            # Loopback
    ipaddress.ip_network("::/128"),             # Unspecified / Wildcard (RFC 4291 § 2.5.2)
    ipaddress.ip_network("fc00::/7"),           # Unique local (ULA)
    ipaddress.ip_network("fe80::/10"),          # Link-local
    ipaddress.ip_network("::ffff:0:0/96"),      # IPv4-mapped IPv6
    ipaddress.ip_network("2001:db8::/32"),      # Documentation
    ipaddress.ip_network("ff00::/8"),           # Multicast
]

BLOCKED_HOSTNAMES = {
    "localhost",
    "metadata.google.internal",
    "instance-data",
    "metadata",
    "169.254.169.254",
}

PERMITTED_PORTS = {80, 443, 8080, 8443}


def validate_target_url(raw_url: str) -> Tuple[str, List[ipaddress.IPv4Address | ipaddress.IPv6Address]]:
    """
    Validates a URL against SSRF attacks:
    - Rejects non-HTTP/HTTPS schemes
    - Rejects disallowed ports
    - Rejects cloud metadata hosts
    - Resolves DNS and checks all resulting IPs against private/loopback/cloud subnets
    """
    if not raw_url or not isinstance(raw_url, str):
        raise SSRFBlockedError(message="URL must be a non-empty string.")

    try:
        parsed = urllib.parse.urlparse(raw_url)
    except Exception as e:
        raise SSRFBlockedError(message=f"Malformed URL: {e}")

    scheme = (parsed.scheme or "").lower()
    if scheme not in ("http", "https"):
        raise SSRFBlockedError(message=f"Disallowed scheme '{scheme}'. Only http and https are permitted.")

    hostname = parsed.hostname
    if not hostname:
        raise SSRFBlockedError(message="URL missing hostname.")

    hostname_lower = hostname.lower()
    if hostname_lower in BLOCKED_HOSTNAMES or hostname_lower.endswith(".internal") or hostname_lower.endswith(".local"):
        raise SSRFBlockedError(message=f"Access to host '{hostname}' is blocked for SSRF protection.")

    # Check if raw hostname is already an IP address
    direct_ip: ipaddress.IPv4Address | ipaddress.IPv6Address | None = None
    try:
        parsed_ip = ipaddress.ip_address(hostname)
        if isinstance(parsed_ip, ipaddress.IPv6Address) and parsed_ip.ipv4_mapped:
            direct_ip = parsed_ip.ipv4_mapped
        else:
            direct_ip = parsed_ip

        for blocked_net in BLOCKED_NETWORKS:
            if direct_ip in blocked_net:
                raise SSRFBlockedError(message=f"Access to blocked IP address '{direct_ip}' in {blocked_net} is prohibited.")
    except ValueError:
        pass  # Not a literal IP, proceed with DNS resolution

    try:
        port = parsed.port or (443 if scheme == "https" else 80)
    except ValueError as e:
        raise SSRFBlockedError(message=f"Invalid URL port: {e}")

    if port not in PERMITTED_PORTS:
        raise SSRFBlockedError(message=f"Target port {port} is not permitted for remote ingestion.")

    if direct_ip is not None:
        return hostname, [direct_ip]

    try:
        addr_info = socket.getaddrinfo(hostname, port, family=socket.AF_UNSPEC, type=socket.SOCK_STREAM)
    except socket.gaierror as e:
        raise SSRFBlockedError(message=f"Failed to resolve hostname '{hostname}': {e}")

    resolved_ips: List[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
    for family, _, _, _, sockaddr in addr_info:
        raw_ip = sockaddr[0]
        ip_obj = ipaddress.ip_address(raw_ip)

        if isinstance(ip_obj, ipaddress.IPv6Address) and ip_obj.ipv4_mapped:
            ip_obj = ip_obj.ipv4_mapped

        for blocked_net in BLOCKED_NETWORKS:
            if ip_obj in blocked_net:
                raise SSRFBlockedError(
                    message=f"Host '{hostname}' resolved to blocked address '{ip_obj}' within {blocked_net}."
                )

        resolved_ips.append(ip_obj)

    if not resolved_ips:
        raise SSRFBlockedError(message=f"Hostname '{hostname}' did not resolve to any valid IP.")

    return hostname, resolved_ips
