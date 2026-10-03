"""Turn a recorded Docker check run into a replay fixture.

    python -m tests.replay.build NAME HOME   # HOME: the run's home directory (holds trace.jsonl)

Each request is stored as its body fields and the ids of its items; items and field sets
are stored once. Opaque model state (encrypted reasoning, signatures) becomes a stable
placeholder, so equal blobs stay equal and different ones stay different, and fields that
identify a user or device are dropped.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from relay.protocols import codec_for

HERE = Path(__file__).parent
OPAQUE = {"encrypted_content", "signature", "thoughtSignature", "thought_signature", "data"}
IDENTIFYING = {"metadata", "user", "prompt_cache_key", "safety_identifier", "client_metadata"}


def scrub(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: (f"opaque:{hashlib.sha256(json.dumps(v).encode()).hexdigest()[:12]}"
                    if k in OPAQUE and isinstance(v, str) and len(v) > 40 else scrub(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [scrub(v) for v in value]
    return value


def build(name: str, home: Path) -> Path:
    trace = [json.loads(line) for line in (home / "trace.jsonl").read_text().splitlines()]
    requests = [r for r in trace if "body" in r]
    codec = codec_for(requests[0]["path"])
    values: dict[str, int] = {}  # body field values and items, each stored once
    out: list[list[Any]] = []
    for request in requests:
        body = scrub({k: v for k, v in request["body"].items() if k not in IDENTIFYING})
        fields = {k: values.setdefault(json.dumps(v, sort_keys=True), len(values))
                  for k, v in codec.with_items(body, []).items()}
        ids = [values.setdefault(json.dumps(item, sort_keys=True), len(values)) for item in codec.items(body)]
        out.append([request["path"], fields, ids])
    fixture = {"name": name, "values": [json.loads(v) for v in values], "requests": out}
    path = HERE / f"{name.replace(':', '-')}.json.gz"
    path.write_bytes(gzip.compress(json.dumps(fixture, separators=(",", ":")).encode(), mtime=0))
    return path


def load(path: Path) -> tuple[str, list[tuple[str, dict[str, Any]]]]:
    """A fixture's name and its requests as (path, body)."""

    fixture = json.loads(gzip.decompress(path.read_bytes()))
    values = fixture["values"]
    requests = []
    for request_path, fields, ids in fixture["requests"]:
        body = {k: values[v] for k, v in fields.items()}
        requests.append((request_path, codec_for(request_path).with_items(body, [values[i] for i in ids])))
    return fixture["name"], requests


if __name__ == "__main__":
    print(build(sys.argv[1], Path(sys.argv[2])))
