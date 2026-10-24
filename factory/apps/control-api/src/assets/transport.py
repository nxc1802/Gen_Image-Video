"""
factory/apps/control-api/src/assets/transport.py
Secure Pinning Transport Preventing DNS Rebinding / TOCTOU SSRF Attacks.
"""
import ssl
from typing import List, Optional, Iterable, Any
import httpx
import httpcore
from httpcore._backends.anyio import AnyIOBackend
from httpcore._backends.base import AsyncNetworkStream, SOCKET_OPTION


class PinningAsyncNetworkBackend(AnyIOBackend):
    """
    Custom httpcore network backend that forces TCP connections to a pre-verified
    list of IP addresses, preventing secondary DNS resolution and DNS rebinding attacks.
    """

    def __init__(self, allowed_ips: List[str]):
        if not allowed_ips:
            raise ValueError("PinningAsyncNetworkBackend requires at least one allowed IP address.")
        self.allowed_ips = allowed_ips

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: Optional[float] = None,
        local_address: Optional[str] = None,
        socket_options: Optional[Iterable[SOCKET_OPTION]] = None,
    ) -> AsyncNetworkStream:
        """
        Connects directly to the pre-verified IP literal without resolving `host`.
        Iterates over `allowed_ips` in case of multi-homed hosts or dual-stack fallback.
        """
        last_exc: Optional[Exception] = None
        for ip in self.allowed_ips:
            try:
                # Pass the verified IP literal as remote_host.
                # Because it is an IP literal, anyio.connect_tcp executes with zero DNS queries.
                return await super().connect_tcp(
                    host=ip,
                    port=port,
                    timeout=timeout,
                    local_address=local_address,
                    socket_options=socket_options,
                )
            except (httpcore.ConnectError, httpcore.ConnectTimeout, OSError) as exc:
                last_exc = exc
                continue

        if last_exc:
            raise last_exc
        raise httpcore.ConnectError("All verified IP connection attempts failed.")


class PinningAsyncHTTPTransport(httpx.AsyncHTTPTransport):
    """
    httpx AsyncHTTPTransport configured with PinningAsyncNetworkBackend.
    Ensures that socket connections use pre-verified IP addresses while preserving
    TLS Server Name Indication (SNI), certificate verification, and the HTTP Host header.
    """

    def __init__(
        self,
        allowed_ips: List[str],
        verify: ssl.SSLContext | str | bool = True,
        cert: Any = None,
        trust_env: bool = True,
        http1: bool = True,
        http2: bool = False,
        limits: Optional[httpx.Limits] = None,
        **kwargs: Any,
    ):
        limits = limits or httpx.Limits()
        super().__init__(
            verify=verify,
            cert=cert,
            trust_env=trust_env,
            http1=http1,
            http2=http2,
            limits=limits,
            **kwargs,
        )
        ssl_context = httpx.create_ssl_context(verify=verify, cert=cert, trust_env=trust_env)
        self._pool = httpcore.AsyncConnectionPool(
            ssl_context=ssl_context,
            network_backend=PinningAsyncNetworkBackend(allowed_ips),
            http1=http1,
            http2=http2,
            max_connections=limits.max_connections,
            max_keepalive_connections=limits.max_keepalive_connections,
            keepalive_expiry=limits.keepalive_expiry,
            **kwargs,
        )
