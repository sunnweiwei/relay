"""Relay's own connections upstream: trusting what the harness behind it trusts, and noticing a link
that died while the model was still thinking."""

from __future__ import annotations

import os
import socket
import ssl
from functools import lru_cache
from typing import Any

import certifi
import httpx

# A request waits on the model for minutes without a byte, so no read timeout tells a dead link
# (a laptop changing networks, a NAT dropping the connection) from a slow answer; probes do: after
# a minute of silence, every 15 s, four unanswered ones end the connection (each OS names the idle
# time its own way: TCP_KEEPIDLE, macOS's TCP_KEEPALIVE).
KEEPALIVE = [(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1),
             *((socket.IPPROTO_TCP, getattr(socket, name), value)
               for name, value in (("TCP_KEEPIDLE", 60), ("TCP_KEEPALIVE", 60), ("TCP_KEEPINTVL", 15), ("TCP_KEEPCNT", 4))
               if hasattr(socket, name))]


@lru_cache(maxsize=1)
def trust() -> ssl.SSLContext:
    """The CAs Relay's connections trust: SSL_CERT_FILE (or REQUESTS_CA_BUNDLE) alone where one is
    set; otherwise the operating system's store, where IT puts the root of a TLS-inspecting proxy
    (Zscaler, Netskope) and where Claude Code looks too, with certifi's public roots and
    NODE_EXTRA_CA_CERTS on top. (Python on its own trusts certifi alone, and would refuse there
    what the harness accepts.)"""

    for name in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"):
        if (path := os.getenv(name)) and os.path.isfile(path):
            return ssl.create_default_context(cafile=path)
    try:
        import truststore

        context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    except ImportError:  # (the system's OpenSSL store, which Linux keeps)
        context = ssl.create_default_context()
    context.load_verify_locations(cafile=certifi.where())
    if (extra := os.getenv("NODE_EXTRA_CA_CERTS")) and os.path.isfile(extra):
        context.load_verify_locations(cafile=extra)
    return context


def client(**options: Any) -> httpx.Client:
    # (`verify` for the client too: a proxy from the environment gets a transport of its own)
    return httpx.Client(verify=trust(), transport=httpx.HTTPTransport(verify=trust(), socket_options=KEEPALIVE),
                        **options)


def async_client(**options: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(verify=trust(), transport=httpx.AsyncHTTPTransport(verify=trust(), socket_options=KEEPALIVE),
                             **options)
