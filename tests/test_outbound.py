"""Relay's connections upstream: the CAs they trust (as the harness behind it trusts them), and the
probes that notice a link that died."""

from __future__ import annotations

import datetime
import http.server
import ipaddress
import os
import socket
import ssl
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from relay.transport import outbound

TRUST = ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "NODE_EXTRA_CA_CERTS")


def _certificates(folder: Path) -> None:
    """A company's root (as a TLS-inspecting proxy has one) and a certificate it signed for this machine."""

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    now = datetime.datetime.now(datetime.timezone.utc)
    life = (now - datetime.timedelta(days=1), now + datetime.timedelta(days=2))
    root_key, key = ec.generate_private_key(ec.SECP256R1()), ec.generate_private_key(ec.SECP256R1())
    root_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Example Corp TLS inspection root")])
    root = (x509.CertificateBuilder().subject_name(root_name).issuer_name(root_name).public_key(root_key.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(life[0]).not_valid_after(life[1])
            .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
            .sign(root_key, hashes.SHA256()))
    leaf = (x509.CertificateBuilder().subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")]))
            .issuer_name(root_name).public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(life[0]).not_valid_after(life[1])
            .add_extension(x509.SubjectAlternativeName([x509.DNSName("localhost"),
                                                        x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]), critical=False)
            .sign(root_key, hashes.SHA256()))
    (folder / "root.pem").write_bytes(root.public_bytes(serialization.Encoding.PEM))
    (folder / "leaf.pem").write_bytes(leaf.public_bytes(serialization.Encoding.PEM))
    (folder / "leaf.key").write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                                        serialization.NoEncryption()))


class TrustTests(unittest.TestCase):
    """An upstream whose certificate a company root signed (what a TLS-inspecting proxy shows)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.folder = Path(tempfile.mkdtemp())
        _certificates(cls.folder)
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(cls.folder / "leaf.pem", cls.folder / "leaf.key")

        class Answer(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                self.send_response(200)
                self.end_headers()

            def log_message(self, *args: object) -> None:
                pass

        cls.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Answer)
        cls.server.socket = context.wrap_socket(cls.server.socket, server_side=True)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.url = f"https://localhost:{cls.server.server_address[1]}/"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()

    def get(self, **trust: str) -> str:
        environment = {k: v for k, v in os.environ.items() if k not in TRUST} | trust
        with patch.dict(os.environ, environment, clear=True):
            outbound.trust.cache_clear()
            try:
                with outbound.client(timeout=10) as client:
                    return str(client.get(self.url).status_code)
            except httpx.ConnectError as error:
                return str(error)
            finally:
                outbound.trust.cache_clear()

    def test_a_root_the_harness_is_told_of_in_node_extra_ca_certs_is_trusted_too(self) -> None:
        self.assertEqual(self.get(NODE_EXTRA_CA_CERTS=str(self.folder / "root.pem")), "200")

    def test_a_bundle_in_ssl_cert_file_is_trusted_instead(self) -> None:
        self.assertEqual(self.get(SSL_CERT_FILE=str(self.folder / "root.pem")), "200")

    def test_a_root_no_one_names_is_refused(self) -> None:
        self.assertIn("CERTIFICATE_VERIFY_FAILED", self.get())


class KeepaliveTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform.startswith("linux"), "the probes' options as Linux names them")
    def test_a_silent_link_is_probed_after_a_minute(self) -> None:
        self.assertIn((socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1), outbound.KEEPALIVE)
        self.assertIn((socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 60), outbound.KEEPALIVE)
        self.assertIn((socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 4), outbound.KEEPALIVE)


if __name__ == "__main__":
    unittest.main()
