"""Resume the fixed BrowseComp-Plus task across selected methods and models."""

from __future__ import annotations

import argparse
import ast
import base64
import concurrent.futures
import hashlib
import json
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from relay.strategies import STRATEGIES

from evals.pilot.methods import EVALUATED_METHODS

from evals.pilot.browse_grade import classified_worker, grade, quota_stop_reason
from evals.pilot.browse_suite import load_public_question
from evals.pilot.failure_policy import UPSTREAM_STREAM_INTERRUPTED, upstream_stream_interrupted


QUERY_REVISION = "144cff8e35b5eaef7e526346aa60774a9deb941f"
CORPUS_REVISION = "b27b02bc3e45511b8b82a13e6f90ce761df726f6"
REVIEWED_CAPACITY_OVERFLOW = "auxiliary_capacity_then_context_overflow"
CONTINUATION_EVIDENCE = ("run.json", "relay/runtime.log", "relay/trace.jsonl", "upstream-events.jsonl")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.chmod(0o600)
    temporary.replace(path)


def ranked_question_ids(ids: list[str]) -> list[str]:
    if len(ids) != len(set(ids)):
        raise ValueError("Question IDs must be unique")
    return sorted(ids, key=lambda qid: (hashlib.sha256(f"20261026:{qid}".encode()).hexdigest(), qid))


def prepare_subset(root: Path, output: Path, manifest_path: Path, count: int) -> dict[str, Any]:
    """Freeze an ID-only sample, then stage question text apart from labels."""
    import pyarrow as pa

    if manifest_path.exists():
        raise FileExistsError("Refusing to replace a frozen sample manifest")
    cache = root / f"browse/hf-private-cache/datasets/Tevatron___browsecomp-plus/default/0.0.0/{QUERY_REVISION}"
    encrypted: dict[str, dict[str, str]] = {}
    for shard in sorted(cache.glob("*.arrow")):
        with pa.memory_map(str(shard), "r") as source:
            for batch in pa.ipc.open_stream(source):
                ids = batch.column(batch.schema.get_field_index("query_id")).to_pylist()
                questions = batch.column(batch.schema.get_field_index("query")).to_pylist()
                answers = batch.column(batch.schema.get_field_index("answer")).to_pylist()
                for qid, question, answer in zip(ids, questions, answers, strict=True):
                    if str(qid) in encrypted:
                        raise ValueError("Duplicate query ID in pinned source")
                    encrypted[str(qid)] = {"query": question, "answer": answer}
    info = json.loads((cache / "dataset_info.json").read_text())
    if len(encrypted) != info["splits"]["test"]["num_examples"] or len(encrypted) != 830:
        raise ValueError("Pinned question population does not contain exactly 830 unique rows")
    ranked = ranked_question_ids(list(encrypted))
    selected = ranked[:count]
    project = root / "browse/BrowseComp-Plus"
    syntax = ast.parse((project / "scripts_build_index/decrypt_dataset.py").read_text())
    canary = next(ast.literal_eval(node.value) for node in syntax.body if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == "DEFAULT_CANARY" for target in node.targets))
    key = hashlib.sha256(canary.encode()).digest()

    def decrypt(value: str) -> str:
        encoded = base64.b64decode(value)
        return bytes(byte ^ key[index % len(key)] for index, byte in enumerate(encoded)).decode()

    qrels = {qid: {"evidence": [], "gold": []} for qid in selected}
    for kind, filename in (("evidence", "qrel_evidence.txt"), ("gold", "qrel_golds.txt")):
        for line in (project / "topics-qrels" / filename).read_text().splitlines():
            qid, _, docid, relevance = line.split()
            if qid in qrels:
                qrels[qid][kind].append({"docid": docid, "relevance": int(relevance)})
    tasks = []
    for qid in selected:
        question_path = output / "tasks" / qid / "question.json"
        reference_path = output / "private" / f"{qid}.json"
        if not qrels[qid]["evidence"] or not qrels[qid]["gold"]:
            raise ValueError(f"Frozen query {qid} has no official relevance labels")
        write_json(question_path, {"query_id": qid, "query": decrypt(encrypted[qid]["query"])})
        write_json(reference_path, {"query_id": qid, "answer": decrypt(encrypted[qid]["answer"]), "qrels": qrels[qid]})
        tasks.append({"task_id": qid, "query_id": qid, "question_file": str(question_path),
                      "reference_file": str(reference_path),
                      "question_sha256": hashlib.sha256(question_path.read_bytes()).hexdigest(),
                      "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest()})
    manifest = {"benchmark": "BrowseComp-Plus", "population_size": len(ranked), "seed": "20261026",
                "selection": "ascending SHA256('20261026:' + question_id), then question_id",
                "selected_ids": selected, "first_30_ids": ranked[:30], "ranked_ids": ranked,
                "dataset_revision": QUERY_REVISION, "corpus_revision": CORPUS_REVISION,
                "corpus_documents": 100195, "pilot_query_id": "183", "pilot_in_selected": "183" in selected,
                "source_urls": ["https://huggingface.co/datasets/Tevatron/browsecomp-plus",
                                "https://huggingface.co/datasets/Tevatron/browsecomp-plus-corpus",
                                "https://github.com/texttron/BrowseComp-Plus"],
                "created_after_pilot_results_observed": True,
                "selection_uses_outcomes": False, "tasks": tasks}
    write_json(manifest_path, manifest)
    return manifest


def validate_manifest(path: Path, root: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text())
    ranked = manifest["ranked_ids"]
    if (manifest["seed"] != "20261026" or len(ranked) != manifest["population_size"]
            or ranked != ranked_question_ids(ranked)
            or manifest["selected_ids"] != ranked[:len(manifest["selected_ids"])]):
        raise ValueError("Manifest selection differs from the frozen seeded full-population order")
    if manifest["selected_ids"] != [str(task["query_id"]) for task in manifest["tasks"]]:
        raise ValueError("Manifest task order differs from its frozen selected IDs")
    for task in manifest["tasks"]:
        question = Path(task["question_file"])
        reference = Path(task["reference_file"])
        for source in (question, reference):
            source.resolve().relative_to(root.resolve())
        load_public_question(question, str(task["query_id"]))
        if str(json.loads(reference.read_text())["query_id"]) != str(task["query_id"]):
            raise ValueError("Private grading reference belongs to another question")
        for source, key in ((question, "question_sha256"), (reference, "reference_sha256")):
            if hashlib.sha256(source.read_bytes()).hexdigest() != task[key]:
                raise ValueError("Frozen task content has changed")
    return manifest


def run_sampled(args: argparse.Namespace) -> int:
    manifest = validate_manifest(args.manifest, args.root)
    if args.jobs != 1:
        raise ValueError("The sampled Browse queue permits one active worker")
    state = {"benchmark": "BrowseComp-Plus", "manifest": str(args.manifest), "status": "running",
             "planned_cells": len(manifest["tasks"]) * len(args.methods) * len(args.models), "tasks": []}
    for task in manifest["tasks"]:
        task_root = args.out / str(task["query_id"])
        command = [str(args.python.absolute()), "-m", "evals.pilot.browse_matrix", "--root", str(args.root),
                   "--out", str(task_root), "--python", str(args.python.absolute()), "--tmp-dir", str(args.tmp_dir),
                   "--query-id", str(task["query_id"]), "--question-file", task["question_file"],
                   "--reference", task["reference_file"], "--models", *args.models, "--methods", *args.methods,
                   "--base-port", str(args.base_port), "--jobs", "1"]
        entry = {"task_id": str(task["query_id"]), "status": "running", "output": str(task_root), "command": command}
        state["tasks"].append(entry)
        state["current_task"] = str(task["query_id"])
        write_json(args.out / "sample-status.json", state)
        write_json(task_root / "launcher-command.json", command)
        with (task_root / "matrix-launcher.log").open("a") as log:
            process = subprocess.run(command, cwd=Path(__file__).resolve().parents[2],
                                     stdout=log, stderr=subprocess.STDOUT, check=False)
        entry.update(status="finished" if process.returncode == 0 else "halted", exit_code=process.returncode)
        if process.returncode:
            state["status"] = "halted"
            write_json(args.out / "sample-status.json", state)
            return process.returncode
        write_json(args.out / "sample-status.json", state)
    state["status"] = "finished"
    write_json(args.out / "sample-status.json", state)
    return 0


def quota_stopped(result: dict[str, Any]) -> bool:
    return quota_stop_reason(result) is not None


def invalid_audit_reason(run_dir: Path) -> str | None:
    path = run_dir / "evaluation-audit.json"
    if not path.exists():
        return None
    try:
        audit = json.loads(path.read_text())
        if not isinstance(audit, dict):
            return "invalid_evaluation_audit"
        if audit.get("valid") is False:
            return str(audit.get("reason") or audit.get("classification") or "invalid_evaluation_audit")
    except (OSError, ValueError):
        return "unreadable_evaluation_audit"
    return None


def pure_context_limit(result: dict[str, Any]) -> bool:
    """An observed context overflow permits continuation only without other failures."""
    sizes, window = result.get("provider_input_tokens_per_request"), result.get("window")
    if (type(window) is not int or window < 1 or not isinstance(sizes, list) or not sizes
            or any(type(size) is not int or size < 1 for size in sizes)):
        return False
    return (result.get("status") == "stopped"
            and type(result.get("exit_code")) is int and result["exit_code"] in (0, 1, -9, -15)
            and result.get("stop_reason") == "context_budget_exceeded"
            and result.get("failure_kind") in (None, "context_budget_exceeded")
            and result.get("error") in (None, "")
            and result.get("protocol_valid") is False
            and type(result.get("requests_over_context_budget")) is int
            and result["requests_over_context_budget"] > 0
            and result["requests_over_context_budget"] == sum(size > window for size in sizes)
            and type(result.get("strategy_errors")) is int and result["strategy_errors"] == 0
            and type(result.get("sandbox_execution_errors")) is int and result["sandbox_execution_errors"] == 0)


def reviewed_capacity_overflow(run_dir: Path, result: dict[str, Any]) -> bool:
    """Accept only a controller review bound to this exact terminal trajectory."""
    if quota_stopped(result) or invalid_audit_reason(run_dir) not in (None, REVIEWED_CAPACITY_OVERFLOW):
        return False
    sizes, window = result.get("provider_input_tokens_per_request"), result.get("window")
    if (result.get("status") != "stopped" or result.get("stop_reason") != "context_budget_exceeded"
            or type(result.get("exit_code")) is not int or result["exit_code"] not in (-9, -15)
            or result.get("protocol_valid") is not False
            or result.get("error") not in (None, "") or result.get("failure_kind") not in (None, "")
            or type(result.get("strategy_errors")) is not int or result["strategy_errors"] < 1
            or type(result.get("sandbox_execution_errors")) is not int or result["sandbox_execution_errors"] != 0
            or type(result.get("requests_over_context_budget")) is not int
            or result["requests_over_context_budget"] < 1
            or type(window) is not int or window < 1 or not isinstance(sizes, list) or not sizes
            or any(type(size) is not int or size < 1 for size in sizes)
            or result["requests_over_context_budget"] != sum(size > window for size in sizes)):
        return False
    try:
        audit = json.loads((run_dir / "continuation-audit.json").read_text())
        if (not isinstance(audit, dict) or audit.get("reason") != REVIEWED_CAPACITY_OVERFLOW
                or "score" not in audit or audit["score"] is not None
                or audit.get("continue_without_retry") is not True):
            return False
        hashes = audit.get("evidence_sha256")
        if not isinstance(hashes, dict) or set(hashes) != set(CONTINUATION_EVIDENCE):
            return False
        for name in CONTINUATION_EVIDENCE:
            content = (run_dir / name).read_bytes()
            if not content or hashes[name] != hashlib.sha256(content).hexdigest():
                return False
            if name == "run.json" and json.loads(content) != result:
                return False
    except (OSError, ValueError):
        return False
    return True


def blocks_continuation(run_dir: Path, result: dict[str, Any]) -> bool:
    """Fail closed unless the raw evidence identifies an allowed terminal outcome."""
    if quota_stopped(result):
        return True
    if (run_dir / "continuation-audit.json").exists():
        return not reviewed_capacity_overflow(run_dir, result)
    interrupted = upstream_stream_interrupted(result, run_dir / "relay/runtime.log")
    context_limited = pure_context_limit(result)
    capacity = capacity_failed(run_dir, result)
    if reason := invalid_audit_reason(run_dir):
        permitted = {UPSTREAM_STREAM_INTERRUPTED: interrupted, "context_budget_exceeded": context_limited,
                     "provider_capacity": capacity,
                     "protocol_invalid": interrupted or context_limited or capacity,
                     "worker_protocol_invalid": interrupted or context_limited or capacity}
        if not permitted.get(reason, False):
            return True
    if interrupted or context_limited:
        return False
    if capacity:
        return False
    if (result.get("protocol_valid") is not True or result.get("strategy_errors")
            or result.get("sandbox_execution_errors") or result.get("requests_over_context_budget")
            or result.get("error") or result.get("failure_kind")):
        return True
    if result.get("status") == "completed":
        return result.get("exit_code") not in (None, 0)
    return not (result.get("status") == "stopped"
                and result.get("stop_reason") in {"wall_time_limit", "model_request_limit"}
                and type(result.get("exit_code")) is int and result["exit_code"] in (0, 1, -9, -15))


def capacity_failed(path: Path, result: dict[str, Any]) -> bool:
    if ((path / "continuation-audit.json").exists()
            or quota_stopped(result) or result.get("strategy_errors") or result.get("sandbox_execution_errors")
            or result.get("requests_over_context_budget")):
        return False
    if invalid_audit_reason(path) not in (None, "provider_capacity", "protocol_invalid", "worker_protocol_invalid"):
        return False
    if result.get("status") == "completed" and result.get("exit_code") in (None, 0):
        return False
    messages = [str(result.get(key, "")) for key in ("provider_error", "error", "stop_reason")]
    trace = path / "codex.jsonl"
    if trace.exists():
        for line in trace.read_text().splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") in {"error", "turn.failed"}:
                messages.append(json.dumps(event))
    return any("at capacity" in message.lower() or "provider_capacity" in message.lower() for message in messages)


def driver_is_alive(pid: int, proc_root: Path = Path("/proc")) -> bool:
    try:
        process = proc_root / str(pid)
        state = (process / "stat").read_text().rsplit(")", 1)[1].split()[0]
        command = (process / "cmdline").read_bytes()
        return state not in {"Z", "X"} and bool(command) and (
            b"evals.pilot.browse_matrix" in command or b"browse_matrix.py" in command)
    except (OSError, IndexError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--tmp-dir", type=Path, default=Path("/dev/shm/relay-bcp-183-private"))
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--methods", nargs="+", choices=sorted(STRATEGIES),
                        default=list(EVALUATED_METHODS))
    parser.add_argument("--models", nargs="+", choices=["gpt-6-luna", "gpt-6.1-sol"],
                        default=["gpt-6-luna", "gpt-6.1-sol"])
    parser.add_argument("--jobs", type=int, choices=[1, 2], default=1)
    parser.add_argument("--base-port", type=int, default=19500)
    parser.add_argument("--after-pid", type=int, help="Wait at most 24h for the existing driver to exit, then continue once")
    parser.add_argument("--query-id", default="183")
    parser.add_argument("--question-file", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--prepare-subset", action="store_true", help="Stage a seeded subset without model calls")
    parser.add_argument("--sample-count", type=int, choices=[5, 30], default=5)
    parser.add_argument("--preflight-manifest", action="store_true", help="Validate frozen public/private task boundaries without inference")
    args = parser.parse_args()
    root, output = args.root.resolve(), args.out.resolve()
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    output.relative_to(root)
    if args.prepare_subset:
        if not args.manifest:
            parser.error("--prepare-subset requires --manifest")
        manifest = prepare_subset(root, output, args.manifest, args.sample_count)
        print(json.dumps({"population_size": manifest["population_size"], "selected_ids": manifest["selected_ids"]}))
        return 0
    if args.manifest:
        if args.preflight_manifest:
            manifest = validate_manifest(args.manifest, root)
            print(json.dumps({"preflight": "passed", "tasks": len(manifest["tasks"]), "model_calls": 0}))
            return 0
        validate_manifest(args.manifest, root)
    if not args.manifest and args.query_id != "183" and (not args.question_file or not args.reference):
        parser.error("A sampled query requires explicit question and private reference files")
    if args.after_pid:
        state_path = output / "continuation-state.json"
        state_path.write_text(json.dumps({"status": "waiting", "after_pid": args.after_pid}) + "\n")
        deadline = time.monotonic() + 24 * 3600
        while driver_is_alive(args.after_pid):
            if time.monotonic() >= deadline:
                state_path.write_text(json.dumps({"status": "wait_timeout", "after_pid": args.after_pid}) + "\n")
                return 2
            time.sleep(10)
        previous = output / "matrix-summary.json"
        if previous.exists() and any(quota_stopped(row) for row in json.loads(previous.read_text()).get("runs", [])):
            state_path.write_text(json.dumps({"status": "quota_stopped_no_continuation"}) + "\n")
            return 2
        state_path.write_text(json.dumps({"status": "running", "after_pid": args.after_pid}) + "\n")
    if args.manifest:
        return run_sampled(args)
    # A prior quota stop is an admission boundary even if an earlier slot is
    # missing or a later attempt happens to have a grade.
    saved_workers = sorted(output.glob("gpt-*/*/run.json")) + sorted(output.glob("gpt-*/*/attempt-*/run.json"))
    for worker_path in saved_workers:
        worker = json.loads(worker_path.read_text())
        if quota_stopped(worker):
            state_path = output / "matrix-summary.json"
            state = json.loads(state_path.read_text()) if state_path.exists() else {"runs": []}
            state.update(halted=True, stop_reason=quota_stop_reason(worker), stopped_attempt=str(worker_path.parent))
            write_json(state_path, state)
            return 1
    source = Path(__file__).with_name("browse_suite.py")
    stop = threading.Event()
    lock = threading.Lock()
    rows: list[dict[str, Any]] = []

    def record(result: dict[str, Any]) -> None:
        with lock:
            rows.append(result)
            temporary = output / "matrix-summary.tmp"
            temporary.write_text(json.dumps({"runs": rows, "halted": stop.is_set()}, indent=2) + "\n")
            temporary.replace(output / "matrix-summary.json")
            with (output / "matrix-ledger.jsonl").open("a") as stream:
                stream.write(json.dumps(result) + "\n")
            print(json.dumps(result), flush=True)

    def run_arm(method: str, model: str) -> dict[str, Any]:
        slot = output / model / method
        service_slot = output / "services" / model / method
        attempts = ([slot] if (slot / "run.json").exists() else []) + sorted(
            path for path in slot.glob("attempt-*") if path.is_dir())
        run_dir = attempts[-1] if attempts else slot
        services = service_slot if run_dir == slot else service_slot / run_dir.name
        result_path = run_dir / "run.json"
        retry = False
        if result_path.is_file():
            result = json.loads(result_path.read_text())
            if result.get("status") == "running":
                raise RuntimeError(f"Refusing to replace an unfinished trajectory: {run_dir}")
            resumed = True
            retry = capacity_failed(run_dir, result) and len(attempts) < 2 and not quota_stopped(result)
            if retry:
                run_dir = slot / "attempt-002"
                services = service_slot / "attempt-002"
                result_path = run_dir / "run.json"
        if not result_path.is_file():
            if stop.is_set():
                return {"model": model, "strategy": method, "status": "not_started"}
            port = args.base_port + list(STRATEGIES).index(method) * 2 + ["gpt-6-luna", "gpt-6.1-sol"].index(model)
            command = [str(args.python.absolute()), str(source),
                       "--data-root", str(root / "browse"), "--hidden-root", str(root),
                       "--python", str(args.python.absolute()), "--tmp-dir", str(args.tmp_dir),
                       "--model", model, "--strategy", method, "--port", str(port),
                       "--query-id", args.query_id,
                       "--out", str(run_dir), "--service-root", str(services)]
            if args.question_file:
                command.extend(["--question-file", str(args.question_file)])
            logs = output / "driver-logs" / model
            logs.mkdir(parents=True, exist_ok=True)
            label = method if run_dir == slot else f"{method}-{run_dir.name}"
            (logs / f"{label}.command.json").write_text(json.dumps(command, indent=2) + "\n")
            print(json.dumps({"event": "starting", "model": model, "strategy": method, "retry": retry}), flush=True)
            with (logs / f"{label}.log").open("w") as log:
                process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
            if not result_path.is_file():
                stop.set()
                return {"model": model, "strategy": method, "status": "orchestration_failed",
                        "exit_code": process.returncode, "log": str(logs / f"{method}.log")}
            result = json.loads(result_path.read_text())
            resumed = False
        capacity = capacity_failed(run_dir, result)
        if capacity:
            audit_path = run_dir / "evaluation-audit.json"
            if not audit_path.exists():
                audit_path.write_text(json.dumps({"valid": False, "classification": "infrastructure_failed",
                                                 "reason": "provider_capacity"}, indent=2) + "\n")
            # Every cell gets at most one retry, with the same model and settings.
            if run_dir == slot and not quota_stopped(result):
                return run_arm(method, model)
        blocked = blocks_continuation(run_dir, result)
        reviewed = reviewed_capacity_overflow(run_dir, result)
        classified = classified_worker(run_dir, result)
        if reviewed:
            classified |= {"status": "infrastructure_failed", "protocol_valid": False,
                           "stop_reason": REVIEWED_CAPACITY_OVERFLOW,
                           "failure_kind": REVIEWED_CAPACITY_OVERFLOW,
                           "error": REVIEWED_CAPACITY_OVERFLOW}
        if blocked and not quota_stopped(result) and result.get("status") in {"completed", "stopped"}:
            classified |= {"status": "infrastructure_failed", "protocol_valid": False,
                           "failure_kind": "protocol_invalid",
                           "error": invalid_audit_reason(run_dir) or result.get("error") or "worker_protocol_invalid"}
        interrupted = upstream_stream_interrupted(result, run_dir / "relay/runtime.log")
        if interrupted:
            audit_path = run_dir / "evaluation-audit.json"
            if not audit_path.exists():
                write_json(audit_path, {"valid": False, "classification": "infrastructure_failed",
                                       "reason": UPSTREAM_STREAM_INTERRUPTED,
                                       "worker_stop_reason": result.get("stop_reason")})
        if blocked:
            stop.set()
        score_path = run_dir / "browse-grade.json"
        inference_recorded = bool(result.get("provider_input_tokens_per_request")) or bool(
            (result.get("codex_final_total_usage") or {}).get("total_tokens"))
        if not score_path.exists() and inference_recorded:
            score = grade(run_dir, services / "retrieval.jsonl",
                          args.reference or root / "browse/grader-private/selected-qid-183-gold-qrels.json",
                          expected_query_id=args.query_id)
            score_path.write_text(json.dumps(score, indent=2) + "\n")
        score = json.loads(score_path.read_text()) if score_path.exists() else None
        if score is not None and (blocked or interrupted or reviewed):
            # A preserved old grade must not authorize a score for an invalid run.
            # Update only the matrix view; keep the original grade file intact.
            score = score | {"strict_answer_score": None, "protocol_valid": False,
                             "run_status": classified.get("status"),
                             "stop_reason": classified.get("stop_reason"),
                             "failure_kind": classified.get("failure_kind")}
        return {"model": model, "strategy": method, "status": "infrastructure_failed" if capacity else classified.get("status"),
                "stop_reason": classified.get("stop_reason"), "error": classified.get("error"),
                "failure_kind": classified.get("failure_kind"),
                "requests": result.get("harness_requests"), "resumed": resumed,
                "run_dir": str(run_dir), "score": score, "capacity_retry_exhausted": capacity and run_dir != slot}

    for method in args.methods:
        if stop.is_set():
            break
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = [executor.submit(run_arm, method, model) for model in args.models]
            for future in concurrent.futures.as_completed(futures):
                record(future.result())
    if args.after_pid:
        (output / "continuation-state.json").write_text(json.dumps({"status": "finished", "halted": stop.is_set()}) + "\n")
    return int(stop.is_set())


if __name__ == "__main__":
    raise SystemExit(main())
