"""Relay as an HTTPS proxy, for a harness that must keep talking to its provider's own host.

Claude Code turns off what it keeps for Anthropic's own endpoint (Remote Control first) once its
base URL is changed; through an HTTPS proxy (`HTTPS_PROXY`, the way an organization's proxy is set)
it goes on reaching api.anthropic.com. Relay listens as that proxy: a connection to one of `HOSTS`
it opens with a certificate of its own CA, which the harness trusts (`NODE_EXTRA_CA_CERTS`), and
hands to its own endpoint, where the request is managed as any through the proxy; a connection to
any other host (Remote Control's, a tool's download) is tunnelled as it is, never opened. The CA
can vouch for `HOSTS` alone (a name constraint the harness enforces), its key is readable by its
owner only, and it lives in Relay's directory, nowhere in the system's trust.
"""

from __future__ import annotations

import asyncio
import datetime
import logging
import os
import ssl
from pathlib import Path

from .settings import home

log = logging.getLogger("relay")
HOSTS = ("api.anthropic.com",)  # the hosts whose requests Relay manages
DAYS = 825  # a certificate's life; renewed a month before it ends
PROXY_VARS = ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy")


def folder() -> Path:
    return home() / "intercept"


def ca_file() -> Path:
    """The CA's certificate, which the harness is told to trust (created as needed)."""

    ensure()
    return folder() / "ca.pem"


def ensure() -> None:
    """The CA and its certificate for `HOSTS`, made once and renewed before they end."""

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    now = datetime.datetime.now(datetime.timezone.utc)
    files = folder()
    if (files / "leaf.pem").exists():
        leaf = x509.load_pem_x509_certificate((files / "leaf.pem").read_bytes())
        if leaf.not_valid_after_utc - now > datetime.timedelta(days=30):
            return
    files.mkdir(parents=True, exist_ok=True)
    os.chmod(files, 0o700)
    life = (now - datetime.timedelta(days=1), now + datetime.timedelta(days=DAYS))
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Relay local proxy (this machine only)")])
    if (files / "ca.pem").exists() and (files / "ca.key").exists():
        ca = x509.load_pem_x509_certificate((files / "ca.pem").read_bytes())
        ca_key = serialization.load_pem_private_key((files / "ca.key").read_bytes(), None)
        if ca.not_valid_after_utc - now < datetime.timedelta(days=60):
            ca = None
    else:
        ca = None
    if ca is None:
        ca_key = ec.generate_private_key(ec.SECP256R1())
        ca = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(ca_key.public_key())
              .serial_number(x509.random_serial_number()).not_valid_before(life[0])
              .not_valid_after(life[1] + datetime.timedelta(days=60))
              .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
              .add_extension(x509.KeyUsage(digital_signature=True, content_commitment=False, key_encipherment=False,
                                           data_encipherment=False, key_agreement=False, key_cert_sign=True,
                                           crl_sign=True, encipher_only=False, decipher_only=False), critical=True)
              .add_extension(x509.NameConstraints(permitted_subtrees=[x509.DNSName(h) for h in HOSTS],
                                                  excluded_subtrees=None), critical=True)
              .sign(ca_key, hashes.SHA256()))
        _write(files / "ca.key", ca_key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                                      serialization.NoEncryption()))
        _write(files / "ca.pem", ca.public_bytes(serialization.Encoding.PEM))
    key = ec.generate_private_key(ec.SECP256R1())
    leaf = (x509.CertificateBuilder().subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, HOSTS[0])]))
            .issuer_name(name).public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(life[0]).not_valid_after(life[1])
            .add_extension(x509.SubjectAlternativeName([x509.DNSName(h) for h in HOSTS]), critical=False)
            .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
            .sign(ca_key, hashes.SHA256()))
    _write(files / "leaf.key", key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                                 serialization.NoEncryption()))
    _write(files / "leaf.pem", leaf.public_bytes(serialization.Encoding.PEM) + ca.public_bytes(serialization.Encoding.PEM))


def _write(path: Path, data: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)


def forget_own(port: int) -> None:
    """Drop the proxy variables that point at Relay's own proxy from this process (started by a
    harness that goes through it, Relay inherits them), so that Relay's requests go out direct."""

    for var in PROXY_VARS:
        if (value := os.environ.get(var)) and value.rstrip("/").endswith(f"127.0.0.1:{port}"):
            os.environ.pop(var)


async def serve(host: str, port: int, endpoint: int) -> asyncio.base_events.Server:
    """The proxy, on `port`; what it opens goes to Relay's endpoint, on `endpoint`."""

    ensure()
    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    context.load_cert_chain(folder() / "leaf.pem", folder() / "leaf.key")

    async def connect(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            head = await reader.readuntil(b"\r\n\r\n")
            method, target, _ = head.split(b"\r\n", 1)[0].decode("latin-1").split(" ", 2)
            if method != "CONNECT":
                writer.write(b"HTTP/1.1 405 Method Not Allowed\r\nContent-Length: 0\r\n\r\n")
                await writer.drain()
                return
            name, _, number = target.rpartition(":")
            if name in HOSTS:
                writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
                await writer.drain()
                await writer.start_tls(context)
                inner, outer = await asyncio.open_connection("127.0.0.1", endpoint)
            else:
                try:
                    inner, outer = await asyncio.open_connection(name, int(number))
                except OSError:
                    writer.write(b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\n\r\n")
                    await writer.drain()
                    return
                writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
                await writer.drain()
            await asyncio.gather(_pipe(reader, outer), _pipe(inner, writer))
        except (OSError, ValueError, asyncio.IncompleteReadError, ssl.SSLError):
            log.debug("intercept: a connection ended", exc_info=True)
        finally:
            writer.close()

    return await asyncio.start_server(connect, host, port)


async def _pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while data := await reader.read(1 << 16):
            writer.write(data)
            await writer.drain()
    except (OSError, ssl.SSLError):
        pass
    finally:
        try:
            writer.close()
        except OSError:
            pass
