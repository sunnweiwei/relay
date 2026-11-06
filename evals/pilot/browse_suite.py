#!/usr/bin/env python3
"""BrowseComp-Plus local tool client and paired Codex pilot orchestrator."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_BASE_URL = "http://127.0.0.1:18769"


def load_public_question(path: Path, expected_id: str) -> dict[str, Any]:
    question = json.loads(path.read_text())
    allowed = {"query_id", "query", "query_tokens_o200k"}
    if set(question) - allowed or str(question.get("query_id")) != expected_id:
        raise ValueError("Question file has private fields or the wrong query ID")
    if not isinstance(question.get("query"), str) or not question["query"].strip():
        raise ValueError("Question text is missing")
    return question


def client_base_url() -> str:
    endpoint = Path(__file__).with_name("browse-endpoint.json")
    if endpoint.is_file():
        return str(json.loads(endpoint.read_text())["base_url"])
    return DEFAULT_BASE_URL


def request_json(method: str, path: str, body: dict[str, Any] | None = None,
                 base_url: str = DEFAULT_BASE_URL, timeout: float = 45) -> dict[str, Any]:
    data = None if body is None else json.dumps(body, ensure_ascii=False).encode()
    req = urllib.request.Request(
        base_url + path, data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method=method,
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def client_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Local BrowseComp-Plus tool client")
    parser.add_argument("--base-url", default=client_base_url())
    sub = parser.add_subparsers(dest="tool", required=True)
    sub.add_parser("question")
    search = sub.add_parser("search")
    search.add_argument("--query", required=True)
    search.add_argument("--top-k", type=int, default=10)
    fetch = sub.add_parser("fetch")
    fetch.add_argument("docid")
    health = sub.add_parser("health")
    args = parser.parse_args(argv)
    if args.tool == "question":
        result = request_json("GET", "/question", base_url=args.base_url)
    elif args.tool == "search":
        result = request_json("POST", "/search", {"query": args.query,
                            "top_k": args.top_k}, base_url=args.base_url)
    elif args.tool == "fetch":
        from urllib.parse import quote
        result = request_json("GET", "/fetch/" + quote(args.docid, safe=""),
                              base_url=args.base_url)
    else:
        result = request_json("GET", "/health", base_url=args.base_url)
    print(json.dumps(result, ensure_ascii=False))
    return 0


def write_prompt(path: Path) -> None:
    prompt = f"""You are solving one BrowseComp-Plus research question against a fixed local corpus.

Use only the local BrowseComp-Plus tools provided in the current directory:

- `python3 browse_suite.py question`
- `python3 browse_suite.py search --query 'your focused query' --top-k 10`
- `python3 browse_suite.py fetch DOCID`

The tool service is local. Do not access external websites, external search, package registries, or benchmark files. Do not inspect files outside the current workspace. The question endpoint is the only source for the task. Search and fetch return excerpts from the full fixed corpus.

This is a multi-stage research task. First retrieve the question and decompose it into distinct facts that must be established. Make at least six distinct, focused search requests, changing the query to investigate different parts of the question. Fetch at least four promising documents from search results. Compare independent evidence, track contradictions, and keep concise notes in a workspace file so you can revisit the reasoning. Do not stop after the first plausible match. Make additional searches when the evidence does not support every part of the answer.

Submit a concise final response with a clearly labeled `FINAL_ANSWER` containing only the answer value expected by the question, followed by a short evidence section with the supporting document IDs and URLs. Do not guess if the corpus does not support a defensible answer; state that limitation clearly.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(prompt, encoding="utf-8")
    path.chmod(0o600)


def wait_health(base_url: str, server: subprocess.Popen[str], log_path: Path) -> None:
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"browse server exited; log={log_path}")
        try:
            health = request_json("GET", "/health", base_url=base_url, timeout=2)
            if health.get("ok"):
                return
        except Exception:
            time.sleep(0.5)
    raise TimeoutError(f"browse server did not become healthy; log={log_path}")


def run_arm(strategy: str, args: argparse.Namespace, python: Path,
            repo: Path, data_root: Path, db_path: Path, client_dir: Path,
            prompt_path: Path) -> dict[str, Any]:
    port = args.port
    base_url = f"http://127.0.0.1:{port}"
    output = args.out or data_root / "runs" / strategy
    service_root = args.service_root or data_root / "service-logs" / strategy
    if output.exists() or service_root.exists():
        raise ValueError("Use fresh output and service directories; never overwrite a trajectory")
    service_root.mkdir(parents=True, mode=0o700)
    os.chmod(service_root, 0o700)
    public = service_root / "public-tools"
    public.mkdir(mode=0o700)
    shutil.copyfile(client_dir / "browse_suite.py", public / "browse_suite.py")
    (public / "browse-endpoint.json").write_text(json.dumps({"base_url": base_url}) + "\n")
    server_log = service_root / "server.stdout.log"
    retrieval_log = service_root / "retrieval.jsonl"
    question_source = args.question_file or data_root / "selected-question.json"
    question = load_public_question(question_source, args.query_id)
    question_path = service_root / "question.json"
    question_path.write_text(json.dumps(question, ensure_ascii=False) + "\n")
    command = [str(python), str(repo / "evals/pilot/browse_server.py"), "serve",
               "--db", str(db_path), "--question", str(question_path),
               "--log", str(retrieval_log), "--port", str(port)]
    with server_log.open("w", encoding="utf-8") as log:
        server = subprocess.Popen(command, cwd="/tmp", stdout=log,
                                  stderr=subprocess.STDOUT, text=True,
                                  start_new_session=True)
    try:
        wait_health(base_url, server, server_log)
        selected = request_json("GET", "/question", base_url=base_url)
        if str(selected.get("query_id")) != args.query_id:
            raise RuntimeError("server is not serving the configured frozen query")
        harness = [str(python), str(repo / "evals/pilot/run.py"),
                   "--root", str(args.hidden_root), "--out", str(output),
                   "--prompt", str(prompt_path), "--strategy", strategy,
                   "--model", args.model, "--deny-path", str(args.tmp_dir),
                   "--harness-profile", "browse_minimal_v1",
                   "--window", "23560", "--max-requests", "100",
                   "--seconds", "1800", "--public", str(public)]
        for path in args.deny_path:
            harness.extend(["--deny-path", str(path)])
        command_path = service_root / "driver-command.json"
        command_path.write_text(json.dumps(harness, indent=2) + "\n")
        command_path.chmod(0o600)
        driver_log = service_root / "driver.log"
        print(json.dumps({"event": "arm_started", "strategy": strategy,
                          "base_url": base_url, "out": str(output)}), flush=True)
        with driver_log.open("w", encoding="utf-8") as log:
            proc = subprocess.Popen(harness, cwd=repo, stdout=log,
                                    stderr=subprocess.STDOUT, text=True,
                                    start_new_session=True)
            arm_started = last_report = time.monotonic()
            while proc.poll() is None:
                time.sleep(5)
                if time.monotonic() - last_report >= 60:
                    trace = output / "relay/trace.jsonl"
                    request_count = 0
                    if trace.exists():
                        with trace.open(encoding="utf-8") as trace_file:
                            request_count = sum(1 for _ in trace_file)
                    print(json.dumps({"event": "arm_progress", "strategy": strategy,
                                      "relay_requests": request_count,
                                      "elapsed_seconds": round(time.monotonic() - arm_started)}),
                          flush=True)
                    last_report = time.monotonic()
        result_path = output / "run.json"
        result = json.loads(result_path.read_text()) if result_path.exists() else {
            "status": "missing_result", "exit_code": proc.returncode
        }
        result["driver_exit_code"] = proc.returncode
        result["browse_log"] = str(retrieval_log)
        if output.is_dir():
            (output / "benchmark-task.json").write_text(json.dumps({
                "query_id": args.query_id,
                "question_sha256": hashlib.sha256(question_source.read_bytes()).hexdigest(),
            }) + "\n")
        (service_root / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({"event": "arm_finished", "strategy": strategy,
                          "status": result.get("status"),
                          "harness_requests": result.get("harness_requests"),
                          "out": str(output)}), flush=True)
        return result
    finally:
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=5)


def prepare_inputs(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, Path]:
    data_root = args.data_root.resolve()
    repo = Path(__file__).resolve().parents[2]
    # Keep the .venv symlink path. resolve() would bypass pyvenv.cfg and lose
    # the prepared environment's packages.
    python = args.python.absolute()
    source_db = data_root / "corpus.sqlite"
    if not source_db.is_file():
        raise FileNotFoundError(source_db)
    if source_db.stat().st_size < 4_000_000_000:
        raise RuntimeError("The full corpus index is missing or unexpectedly small")
    if not python.is_file():
        raise FileNotFoundError(python)

    tmp_dir = Path(args.tmp_dir).resolve()
    tmp_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(tmp_dir, 0o700)
    db_path = tmp_dir / "corpus.sqlite"
    if not db_path.is_file() or db_path.stat().st_size != source_db.stat().st_size:
        print(json.dumps({"event": "copying_full_index", "bytes": source_db.stat().st_size}),
              flush=True)
        shutil.copyfile(source_db, db_path)
        db_path.chmod(0o600)

    client_dir = tmp_dir / "public-tools"
    client_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    shutil.copyfile(Path(__file__).resolve(), client_dir / "browse_suite.py")
    (client_dir / "browse_suite.py").chmod(0o600)
    prompt_path = args.prompt_file or data_root / "prompt.md"
    if not prompt_path.exists():
        write_prompt(prompt_path)
    return data_root, repo, python, db_path, client_dir


def run_pair(args: argparse.Namespace) -> int:
    if args.out or args.service_root:
        raise ValueError("Explicit output paths require --strategy")
    data_root, repo, python, db_path, client_dir = prepare_inputs(args)
    prompt_path = args.prompt_file or data_root / "prompt.md"
    results: dict[str, Any] = {}
    for strategy in ("compaction", "clm"):
        results[strategy] = run_arm(strategy, args, python, repo, data_root,
                                    db_path, client_dir, prompt_path)
    summary = data_root / "runs" / "pair-summary.json"
    summary.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    summary.chmod(0o600)
    print(json.dumps({"event": "pair_finished", "summary": str(summary),
                      "statuses": {k: v.get("status") for k, v in results.items()}},
                     ensure_ascii=False), flush=True)
    return int(any(v.get("status") != "completed" for v in results.values()))


def run_one(args: argparse.Namespace, strategy: str) -> int:
    data_root = args.data_root.resolve()
    repo = Path(__file__).resolve().parents[2]
    python = args.python.absolute()
    tmp_dir = Path(args.tmp_dir).resolve()
    db_path = tmp_dir / "corpus.sqlite"
    client_dir = tmp_dir / "public-tools"
    if not db_path.is_file() or db_path.stat().st_size < 4_000_000_000:
        raise FileNotFoundError("Full local corpus copy is missing")
    if not (client_dir / "browse_suite.py").is_file():
        raise FileNotFoundError("Prepared public tool client is missing")
    prompt_path = args.prompt_file or data_root / "prompt.md"
    if not prompt_path.is_file():
        raise FileNotFoundError("Frozen shared prompt is missing")
    result = run_arm(strategy, args, python, repo, data_root, db_path,
                     client_dir, prompt_path)
    print(json.dumps({"event": "arm_finished", "strategy": strategy,
                      "status": result.get("status"),
                      "harness_requests": result.get("harness_requests"),
                      "out": str(args.out or data_root / "runs" / strategy)}), flush=True)
    return int(result.get("status") != "completed")


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] in {"question", "search", "fetch", "health", "--base-url"}:
        return client_main(sys.argv[1:])
    from relay.strategies import STRATEGIES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--hidden-root", type=Path, required=True,
                        help="Canonical scratch root denied to the worker")
    parser.add_argument("--tmp-dir", type=Path, default=Path("/tmp/relay-bcp-183-private"))
    parser.add_argument("--python", type=Path, required=True,
                        help="Python from the prepared external relay-pilot venv")
    parser.add_argument("--port", type=int, default=18769)
    parser.add_argument("--query-id", default="183")
    parser.add_argument("--question-file", type=Path, help="Question-only JSON for the frozen task")
    parser.add_argument("--prompt-file", type=Path, help="Shared frozen task-independent prompt")
    parser.add_argument("--strategy", choices=sorted(STRATEGIES))
    parser.add_argument("--model", choices=["gpt-6-luna", "gpt-6.1-sol"], default="gpt-6-luna")
    parser.add_argument("--out", type=Path, help="Fresh directory for one trajectory")
    parser.add_argument("--service-root", type=Path, help="Fresh directory for this arm's retrieval service")
    parser.add_argument("--deny-path", type=Path, action="append", default=[])
    parser.add_argument("--prepare-only", action="store_true", help="Stage public client and local index without model calls")
    args = parser.parse_args()
    if args.prepare_only:
        prepare_inputs(args)
        return 0
    return run_one(args, args.strategy) if args.strategy else run_pair(args)


if __name__ == "__main__":
    sys.exit(main())
