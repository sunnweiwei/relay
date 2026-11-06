"""Offline regression tests for BrowseComp retrieval and document fetching."""

from __future__ import annotations

import json
import sqlite3
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from evals.pilot import browse_server


class CharacterEncoding:
    """Deterministic test tokenizer: one character per token."""

    @staticmethod
    def encode(text: str) -> list[str]:
        return list(text)

    @staticmethod
    def decode(tokens: list[str]) -> str:
        return "".join(tokens)


class WordEncoding:
    """Simple whitespace tokenizer for validating bounded FTS excerpts."""

    @staticmethod
    def encode(text: str) -> list[str]:
        return text.split()

    @staticmethod
    def decode(tokens: list[str]) -> str:
        return " ".join(tokens)


def make_service(
    tmp_path: Path, documents: list[tuple[str, str]]
) -> tuple[ThreadingHTTPServer, threading.Thread]:
    db = sqlite3.connect(":memory:", check_same_thread=False)
    db.execute(
        "CREATE VIRTUAL TABLE docs USING fts5(docid UNINDEXED, url UNINDEXED, text, tokenize='unicode61')"
    )
    db.executemany(
        "INSERT INTO docs(docid, url, text) VALUES (?, ?, ?)",
        [(docid, f"https://example.test/{docid}", text) for docid, text in documents],
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), browse_server.BrowseHandler)
    server.db = db
    server.db_lock = threading.RLock()
    server.document_queries = {}
    server.encoding = CharacterEncoding()
    server.log_path = tmp_path / "requests.jsonl"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def get_json(server: ThreadingHTTPServer, path: str) -> tuple[int, dict]:
    try:
        with urlopen(f"http://127.0.0.1:{server.server_port}{path}") as response:
            return response.status, json.loads(response.read())
    except HTTPError as exc:
        return exc.code, json.loads(exc.read())


def post_json(server: ThreadingHTTPServer, path: str, body: dict) -> tuple[int, dict]:
    request = Request(
        f"http://127.0.0.1:{server.server_port}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request) as response:
        return response.status, json.loads(response.read())


def close_service(server: ThreadingHTTPServer, thread: threading.Thread) -> None:
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)
    server.db.close()


def test_search_snippet_can_include_near_tail_match(tmp_path: Path) -> None:
    text = "ordinary padding " * 400 + "tailneedle distinctive evidence"
    server, thread = make_service(tmp_path, [("tail", text)])
    server.encoding = WordEncoding()
    try:
        status, result = post_json(server, "/search", {"query": "tailneedle", "top_k": 10})
        assert status == 200
        assert result["results"][0]["docid"] == "tail"
        assert "tailneedle" in result["results"][0]["snippet"]
        assert result["results"][0]["tokens"] <= browse_server.MAX_SNIPPET_TOKENS
    finally:
        close_service(server, thread)


def test_search_clamps_results_and_preserves_bm25_order_and_limits(tmp_path: Path) -> None:
    documents = [("best", "rankterm " * 20)] + [
        (f"doc-{index}", f"rankterm filler{index}") for index in range(11)
    ]
    server, thread = make_service(tmp_path, documents)
    try:
        status, result = post_json(server, "/search", {"query": "rankterm", "top_k": 1000})
        assert status == 200
        rows = result["results"]
        assert len(rows) == browse_server.DEFAULT_TOP_K
        assert rows[0]["docid"] == "best"
        assert all(row["tokens"] <= browse_server.MAX_SNIPPET_TOKENS for row in rows)

        status, limited = post_json(server, "/search", {"query": "rankterm", "top_k": 0})
        assert status == 200
        assert len(limited["results"]) == 1

        long_query = "rankterm " + "x" * browse_server.MAX_QUERY_TOKENS
        _, bounded = post_json(server, "/search", {"query": long_query})
        assert bounded["query_tokens"] == browse_server.MAX_QUERY_TOKENS
    finally:
        close_service(server, thread)


def test_fetch_requires_search_and_returns_capped_document_prefix(tmp_path: Path) -> None:
    document = "needle " + ("document text " * 900)
    server, thread = make_service(tmp_path, [("known", document), ("unknown", "other text")])
    try:
        status, denied = get_json(server, "/fetch/unknown")
        assert status == 409
        assert "search" in denied["error"]

        status, _ = post_json(server, "/search", {"query": "needle"})
        assert status == 200
        status, fetched = get_json(server, "/fetch/known")
        assert status == 200
        assert fetched["text"] == document[: browse_server.MAX_FETCH_TOKENS]
        assert fetched["tokens"] == browse_server.MAX_FETCH_TOKENS
        assert fetched["url"] == "https://example.test/known"
    finally:
        close_service(server, thread)
