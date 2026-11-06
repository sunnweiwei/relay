#!/usr/bin/env python3
"""Local BrowseComp-Plus corpus search and selected-question service.

This service deliberately does not load or expose the answer/qrel dataset. Build
the FTS index from the public corpus dataset, and place only one selected query
in the private pilot directory before starting the server.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

CORPUS_ID = "Tevatron/browsecomp-plus-corpus"
TOKENIZER_NAME = "o200k_base"
MAX_QUERY_TOKENS = 8192
MAX_SNIPPET_TOKENS = 512
MAX_FTS_SNIPPET_WORDS = 64
MAX_FETCH_TOKENS = 8192
DEFAULT_TOP_K = 10


def tokenizer() -> Any:
    try:
        import tiktoken
    except ImportError as exc:
        raise SystemExit("Install tiktoken to run the server: uv add tiktoken") from exc
    return tiktoken.get_encoding(TOKENIZER_NAME)


def truncate_tokens(text: str, max_tokens: int, encoding: Any) -> tuple[str, int]:
    # Large corpus documents can be millions of characters. Encode a growing
    # prefix so a bounded fetch does not tokenize the entire document first.
    end = min(len(text), max_tokens * 4)
    while True:
        token_ids = encoding.encode(text[:end])
        if len(token_ids) >= max_tokens:
            return encoding.decode(token_ids[:max_tokens]), max_tokens
        if end == len(text):
            return text, len(token_ids)
        end = min(len(text), end * 2)


def build_index(db_path: Path, cache_dir: Path) -> None:
    """Build a local SQLite FTS5 BM25 index from all 100K corpus documents."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Install datasets and pyarrow for index creation: "
            "uv add datasets pyarrow"
        ) from exc

    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    db = sqlite3.connect(db_path)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    db.execute(
        "CREATE VIRTUAL TABLE docs USING fts5(docid UNINDEXED, url UNINDEXED, text, tokenize='unicode61')"
    )
    corpus = load_dataset(CORPUS_ID, split="train", cache_dir=str(cache_dir))
    expected_fields = {"docid", "text", "url"}
    if set(corpus.column_names) != expected_fields:
        raise ValueError(f"Unexpected corpus columns: {corpus.column_names!r}")

    batch: list[tuple[str, str, str]] = []
    for row in corpus:
        batch.append((str(row["docid"]), str(row["url"]), str(row["text"])))
        if len(batch) >= 500:
            db.executemany("INSERT INTO docs(docid, url, text) VALUES (?, ?, ?)", batch)
            db.commit()
            batch.clear()
    if batch:
        db.executemany("INSERT INTO docs(docid, url, text) VALUES (?, ?, ?)", batch)
        db.commit()
    count = db.execute("SELECT count(*) FROM docs").fetchone()[0]
    db.close()
    print(json.dumps({"indexed_documents": count, "database": str(db_path)}))


def safe_fts_query(text: str) -> str:
    """Build a bounded AND query from long terms to avoid broad FTS scans."""
    terms = list(dict.fromkeys(term.lower() for term in re.findall(r"[^\W_]+", text)))
    terms = sorted((term for term in terms if len(term) >= 4), key=lambda term: (-len(term), term))
    if not terms:
        return ""
    chosen = terms[:8]
    return " AND ".join('"' + term.replace('"', '""') + '"' for term in chosen)


class BrowseHandler(BaseHTTPRequestHandler):
    server_version = "BrowseCompPlusLocal/1.0"

    def _json(self, status: int, body: dict[str, Any]) -> None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _log_request(self, action: str, payload: dict[str, Any]) -> None:
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            **payload,
        }
        path = self.server.log_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._json(200, {"ok": True, "corpus": CORPUS_ID})
            return
        if parsed.path == "/question":
            try:
                selected = json.loads(self.server.question_path.read_text(encoding="utf-8"))
                question = selected["query"]
                query_id = selected["query_id"]
            except (OSError, KeyError, json.JSONDecodeError) as exc:
                self._json(503, {"error": f"Selected question unavailable: {exc}"})
                return
            self._log_request("question", {"query_id": query_id})
            self._json(200, {"query_id": query_id, "question": question})
            return
        if parsed.path.startswith("/fetch/"):
            docid = unquote(parsed.path.removeprefix("/fetch/"))
            started = time.monotonic()
            with self.server.db_lock:
                search_query = self.server.document_queries.get(docid)
            if not search_query:
                self._json(409, {"error": "search for this document before fetching it"})
                return
            deadline = time.monotonic() + 30
            with self.server.db_lock:
                self.server.db.set_progress_handler(
                    lambda: int(time.monotonic() > deadline), 1000
                )
                try:
                    row = self.server.db.execute(
                        "SELECT url, text FROM docs "
                        "WHERE docs MATCH ? AND docid = ? LIMIT 1",
                        (search_query, docid),
                    ).fetchone()
                except sqlite3.OperationalError as exc:
                    elapsed_ms = round((time.monotonic() - started) * 1000, 2)
                    self._log_request(
                        "fetch", {"docid": docid, "status": "sql_error",
                                  "error": str(exc), "latency_ms": elapsed_ms}
                    )
                    self._json(504, {"error": "fetch SQL deadline exceeded"})
                    return
                finally:
                    self.server.db.set_progress_handler(None, 0)
            if row is None:
                self._json(404, {"error": "document not found"})
                return
            text, token_count = truncate_tokens(row[1], MAX_FETCH_TOKENS, self.server.encoding)
            elapsed_ms = round((time.monotonic() - started) * 1000, 2)
            self._log_request(
                "fetch", {"docid": docid, "returned_tokens": token_count,
                          "status": "ok", "latency_ms": elapsed_ms}
            )
            self._json(
                200,
                {"docid": docid, "url": row[0], "text": text, "tokens": token_count},
            )
            return
        self._json(404, {"error": "unknown endpoint"})

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/search":
            self._json(404, {"error": "unknown endpoint"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 2_000_000:
                raise ValueError("invalid request body size")
            body = json.loads(self.rfile.read(length))
            query = str(body["query"])
            top_k = min(max(int(body.get("top_k", DEFAULT_TOP_K)), 1), DEFAULT_TOP_K)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            self._json(400, {"error": f"invalid search request: {exc}"})
            return

        query, query_tokens = truncate_tokens(query, MAX_QUERY_TOKENS, self.server.encoding)
        fts_query = safe_fts_query(query)
        if not fts_query:
            self._json(200, {"results": [], "query_tokens": query_tokens})
            return
        started = time.monotonic()
        actual_fts_query = fts_query
        deadline = time.monotonic() + 30
        try:
            with self.server.db_lock:
                self.server.db.set_progress_handler(
                    lambda: int(time.monotonic() > deadline), 1000
                )
                try:
                    rows = self.server.db.execute(
                        "SELECT docid, url, snippet(docs, 2, '', '', ' … ', ?), "
                        "bm25(docs) AS score "
                        "FROM docs WHERE docs MATCH ? ORDER BY score LIMIT ?",
                        (MAX_FTS_SNIPPET_WORDS, actual_fts_query, top_k),
                    ).fetchall()
                    if not rows and " AND " in actual_fts_query:
                        actual_fts_query = actual_fts_query.replace(" AND ", " OR ")
                        rows = self.server.db.execute(
                            "SELECT docid, url, snippet(docs, 2, '', '', ' … ', ?), "
                            "bm25(docs) AS score FROM docs WHERE docs MATCH ? "
                            "ORDER BY score LIMIT ?",
                            (MAX_FTS_SNIPPET_WORDS, actual_fts_query, top_k),
                        ).fetchall()
                finally:
                    self.server.db.set_progress_handler(None, 0)
        except sqlite3.OperationalError as exc:
            elapsed_ms = round((time.monotonic() - started) * 1000, 2)
            self._log_request(
                "search", {"query": query, "query_tokens": query_tokens,
                           "top_k": top_k, "status": "sql_error",
                           "error": str(exc), "latency_ms": elapsed_ms}
            )
            status = 504 if "interrupt" in str(exc).lower() else 400
            self._json(status, {"error": "search SQL deadline exceeded" if status == 504
                                else f"invalid search expression: {exc}"})
            return

        results: list[dict[str, Any]] = []
        with self.server.db_lock:
            for docid, url, text, score in rows:
                self.server.document_queries[docid] = actual_fts_query
                excerpt, token_count = truncate_tokens(text, MAX_SNIPPET_TOKENS, self.server.encoding)
                results.append(
                    {
                        "docid": docid,
                        "url": url,
                        "score": score,
                        "snippet": excerpt,
                        "tokens": token_count,
                    }
                )
        elapsed_ms = round((time.monotonic() - started) * 1000, 2)
        self._log_request(
            "search",
            {
                "query": query,
                "query_tokens": query_tokens,
                "top_k": top_k,
                "query_mode": "and" if actual_fts_query == fts_query else "or_fallback",
                "status": "ok",
                "latency_ms": elapsed_ms,
                "returned_docids": [row["docid"] for row in results],
            },
        )
        self._json(200, {"results": results, "query_tokens": query_tokens})

    def log_message(self, format: str, *args: Any) -> None:
        # Retrieval requests are recorded in the structured JSONL log instead.
        return


def serve(db_path: Path, question_path: Path, log_path: Path, host: str, port: int) -> None:
    if not question_path.is_file():
        raise SystemExit(f"Selected query file does not exist: {question_path}")
    db = sqlite3.connect(db_path, check_same_thread=False)
    db.execute("SELECT 1 FROM docs LIMIT 1").fetchone()
    server = ThreadingHTTPServer((host, port), BrowseHandler)
    server.db = db
    server.db_lock = threading.RLock()
    server.question_path = question_path
    server.log_path = log_path
    server.encoding = tokenizer()
    server.document_queries = {}
    print(f"Serving on http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        db.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="build the SQLite FTS5 BM25 index")
    build.add_argument("--db", type=Path, required=True)
    build.add_argument("--cache-dir", type=Path, required=True)
    run = commands.add_parser("serve", help="serve the selected query and corpus")
    run.add_argument("--db", type=Path, required=True)
    run.add_argument("--question", type=Path, required=True)
    run.add_argument("--log", type=Path, required=True)
    run.add_argument("--host", default="127.0.0.1")
    run.add_argument("--port", type=int, default=18769)
    args = parser.parse_args()
    if args.command == "build":
        build_index(args.db, args.cache_dir)
    else:
        serve(args.db, args.question, args.log, args.host, args.port)


if __name__ == "__main__":
    main()
