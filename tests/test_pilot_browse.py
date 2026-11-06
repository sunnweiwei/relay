"""Regression checks for independently routed BrowseComp workers and grading."""

from __future__ import annotations

import json
import hashlib
import shutil
import subprocess
import sys
import threading
import pytest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from evals.pilot import browse_suite
from evals.pilot.browse_grade import grade
from evals.pilot import browse_matrix


def test_staged_client_uses_its_own_service_port(tmp_path: Path) -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{"query_id":"isolated-service"}')

        def log_message(self, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        shutil.copyfile(browse_suite.__file__, tmp_path / "browse_suite.py")
        (tmp_path / "browse-endpoint.json").write_text(json.dumps({
            "base_url": f"http://127.0.0.1:{server.server_port}"}))
        result = subprocess.run([sys.executable, str(tmp_path / "browse_suite.py"), "question"],
                                capture_output=True, text=True, check=True, cwd=tmp_path)
        assert json.loads(result.stdout) == {"query_id": "isolated-service"}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_grading_keeps_missing_answer_and_official_judge_separate(tmp_path: Path) -> None:
    (tmp_path / "run.json").write_text(json.dumps({
        "model": "gpt-6-luna", "strategy": "clm", "status": "stopped",
        "stop_reason": "model_request_limit", "protocol_valid": True, "exit_code": -15,
        "provider_input_tokens_per_request": [200]}))
    reference = tmp_path / "private.json"
    reference.write_text(json.dumps({"query_id": "183", "answer": "Example Person", "qrels": {
        "evidence": [{"docid": "a"}, {"docid": "b"}], "gold": [{"docid": "b"}]}}))
    retrieval = tmp_path / "retrieval.jsonl"
    retrieval.write_text(json.dumps({"action": "search", "status": "ok", "returned_docids": ["a", "b"]}) + "\n")
    result = grade(tmp_path, retrieval, reference)
    assert result["strict_answer_score"] == 0
    assert result["official_judge_score"] is None
    assert result["recall"]["evidence"] == {"retrieved": 2, "total": 2}
    (tmp_path / "answer.txt").write_text("**FINAL_ANSWER:** Example Person\nEvidence: document b")
    assert grade(tmp_path, retrieval, reference)["strict_answer_score"] == 1


def test_capacity_gets_only_one_same_model_retry(tmp_path: Path, monkeypatch) -> None:
    calls = []
    def fail_at_capacity(command, **kwargs):
        calls.append(command)
        output = Path(command[command.index("--out") + 1])
        output.mkdir(parents=True)
        (output / "run.json").write_text(json.dumps({
            "model": "gpt-6.1-sol", "strategy": "compaction", "status": "infrastructure_failed",
            "exit_code": 1, "provider_error": "Selected model is at capacity.", "harness_requests": 0}))
        return subprocess.CompletedProcess(command, 1)
    monkeypatch.setattr(browse_matrix.subprocess, "run", fail_at_capacity)
    monkeypatch.setattr(sys, "argv", ["browse_matrix", "--root", str(tmp_path), "--out", str(tmp_path / "matrix"),
                                    "--models", "gpt-6.1-sol", "--methods", "compaction"])
    assert browse_matrix.main() == 0
    assert len(calls) == 2
    assert all(command[command.index("--model") + 1] == "gpt-6.1-sol" for command in calls)
    assert calls[1][calls[1].index("--out") + 1].endswith("attempt-002")
    assert browse_matrix.main() == 0
    assert len(calls) == 2


def test_exited_zombie_driver_does_not_block_followon(tmp_path: Path) -> None:
    process = tmp_path / "123"
    process.mkdir()
    (process / "stat").write_text("123 (python) Z 1 2 3")
    (process / "cmdline").write_bytes(b"python\0-m\0evals.pilot.browse_matrix\0")
    assert not browse_matrix.driver_is_alive(123, tmp_path)
    (process / "stat").write_text("123 (python) S 1 2 3")
    assert browse_matrix.driver_is_alive(123, tmp_path)


def test_worker_question_rejects_answer_fields_and_wrong_id(tmp_path: Path) -> None:
    question = tmp_path / "question.json"
    question.write_text(json.dumps({"query_id": "514", "query": "Who?", "answer": "Private answer"}))
    with pytest.raises(ValueError, match="private fields"):
        browse_suite.load_public_question(question, "514")
    question.write_text(json.dumps({"query_id": "514", "query": "Who?"}))
    with pytest.raises(ValueError, match="wrong query ID"):
        browse_suite.load_public_question(question, "1026")
    assert set(browse_suite.load_public_question(question, "514")) == {"query_id", "query"}


def test_grader_rejects_wrong_task_and_quota_interruption(tmp_path: Path) -> None:
    reference = tmp_path / "private.json"
    reference.write_text(json.dumps({"query_id": "514", "answer": "Expected", "qrels": {"gold": []}}))
    run = {"model": "gpt-6-luna", "strategy": "compaction", "status": "stopped", "exit_code": -15,
           "stop_reason": "plan_quota_reserve_reached", "protocol_valid": True,
           "provider_input_tokens_per_request": [200]}
    (tmp_path / "run.json").write_text(json.dumps(run))
    with pytest.raises(ValueError, match="query ID"):
        grade(tmp_path, tmp_path / "retrieval.jsonl", reference, expected_query_id="1026")
    assert grade(tmp_path, tmp_path / "retrieval.jsonl", reference, expected_query_id="514")["strict_answer_score"] is None
    run["stop_reason"] = "wall_time_limit"
    (tmp_path / "run.json").write_text(json.dumps(run))
    assert grade(tmp_path, tmp_path / "retrieval.jsonl", reference, expected_query_id="514")["strict_answer_score"] == 0


def test_recovered_capacity_message_does_not_retry_success(tmp_path: Path) -> None:
    (tmp_path / "codex.jsonl").write_text(json.dumps({"type": "error", "message": "Selected model is at capacity"}) + "\n")
    assert not browse_matrix.capacity_failed(tmp_path, {"status": "completed", "exit_code": 0})


def test_all_model_method_arms_share_question_without_grader_reference(tmp_path: Path, monkeypatch) -> None:
    calls = []
    question, reference = tmp_path / "public.json", tmp_path / "private.json"
    question.write_text(json.dumps({"query_id": "514", "query": "Question only"}))
    reference.write_text(json.dumps({"query_id": "514", "answer": "Hidden", "qrels": {}}))
    def finish_worker(command, **kwargs):
        calls.append(command)
        output = Path(command[command.index("--out") + 1])
        output.mkdir(parents=True)
        (output / "run.json").write_text(json.dumps({"status": "completed", "exit_code": 0,
                                                    "protocol_valid": True}))
        return subprocess.CompletedProcess(command, 0)
    monkeypatch.setattr(browse_matrix.subprocess, "run", finish_worker)
    monkeypatch.setattr(sys, "argv", ["browse_matrix", "--root", str(tmp_path), "--out", str(tmp_path / "matrix"),
                                    "--query-id", "514", "--question-file", str(question), "--reference", str(reference),
                                    "--models", "gpt-6-luna", "gpt-6.1-sol", "--methods", "compaction", "clm"])
    assert browse_matrix.main() == 0
    assert len(calls) == 4
    assert {(command[command.index("--model") + 1], command[command.index("--strategy") + 1])
            for command in calls} == {(model, method) for model in ("gpt-6-luna", "gpt-6.1-sol")
                                      for method in ("compaction", "clm")}
    for command in calls:
        assert command[command.index("--question-file") + 1] == str(question)
        assert command[command.index("--query-id") + 1] == "514"
        assert str(reference) not in command


def interrupted_worker(**updates: object) -> dict:
    return {"model": "gpt-6.1-sol", "strategy": "rlm", "status": "stopped",
            "exit_code": -15, "stop_reason": "wall_time_limit", "protocol_valid": False,
            "strategy_errors": 1, "sandbox_execution_errors": 0,
            "requests_over_context_budget": 0, "window": 23560,
            "provider_input_tokens_per_request": [200], "harness_requests": 1, **updates}


def interruption_log(strategy: str = "rlm") -> str:
    return (f"{strategy} failed; forwarding the last context\n"
            "Traceback (most recent call last):\n"
            "httpx.RemoteProtocolError: peer closed connection without sending complete message body "
            "(incomplete chunked read)\n")


@pytest.mark.parametrize("updates,log,blocked", [
    ({}, interruption_log(), False),
    ({}, "", True),
    ({}, interruption_log() + "ValueError: local adapter bug\n", True),
    ({"error": "quota_check_failed:TimeoutError"}, interruption_log(), True),
    ({"failure_kind": "quota_stopped"}, interruption_log(), True),
    ({"stop_reason": "upstream_stream_interrupted", "failure_kind": "upstream_stream_interrupted"}, "", True),
    ({"strategy_errors": 0}, "", True),
    ({"sandbox_execution_errors": 1}, interruption_log(), True),
    ({"stop_reason": "context_budget_exceeded", "strategy_errors": 0,
      "requests_over_context_budget": 1, "provider_input_tokens_per_request": [24000]}, "", False),
    ({"stop_reason": "context_budget_exceeded", "strategy_errors": 0,
      "requests_over_context_budget": 1}, "", True),
    ({"stop_reason": "context_budget_exceeded", "requests_over_context_budget": 1,
      "provider_input_tokens_per_request": [24000]}, interruption_log(), True),
    ({"stop_reason": "context_budget_exceeded", "strategy_errors": 0, "exit_code": None,
      "requests_over_context_budget": 1, "provider_input_tokens_per_request": [24000]}, "", True),
    ({"stop_reason": "context_budget_exceeded", "strategy_errors": 0, "exit_code": 2,
      "requests_over_context_budget": 1, "provider_input_tokens_per_request": [24000]}, "", True),
])
def test_continuation_requires_unmixed_raw_failure_evidence(
    tmp_path: Path, updates: dict, log: str, blocked: bool,
) -> None:
    (tmp_path / "relay").mkdir()
    (tmp_path / "relay/runtime.log").write_text(log)
    assert browse_matrix.blocks_continuation(tmp_path, interrupted_worker(**updates)) is blocked


def test_stream_interruption_is_unscored_and_raw_worker_is_preserved(tmp_path: Path) -> None:
    worker = interrupted_worker()
    original = json.dumps(worker)
    (tmp_path / "run.json").write_text(original)
    (tmp_path / "relay").mkdir()
    (tmp_path / "relay/runtime.log").write_text(interruption_log())
    (tmp_path / "answer.txt").write_text("FINAL_ANSWER: Expected")
    reference = tmp_path / "private.json"
    reference.write_text(json.dumps({"query_id": "514", "answer": "Expected", "qrels": {}}))
    result = grade(tmp_path, tmp_path / "retrieval.jsonl", reference, expected_query_id="514")
    assert result["strict_answer_score"] is None
    assert result["run_status"] == "infrastructure_failed"
    assert result["stop_reason"] == result["failure_kind"] == "upstream_stream_interrupted"
    assert (tmp_path / "run.json").read_text() == original
    worker["error"] = "quota_check_failed:TimeoutError"
    (tmp_path / "run.json").write_text(json.dumps(worker))
    result = grade(tmp_path, tmp_path / "retrieval.jsonl", reference)
    assert result["failure_kind"] == "quota_stopped"
    assert result["stop_reason"] == "quota_check_failed:TimeoutError"


@pytest.mark.parametrize("kind,expected_calls", [("stream", 2), ("context", 2), ("local", 1), ("quota", 1)])
def test_matrix_continues_only_known_failures_without_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str, expected_calls: int,
) -> None:
    calls = []
    reference = tmp_path / "private.json"
    reference.write_text(json.dumps({"query_id": "183", "answer": "Expected", "qrels": {}}))

    def finish_worker(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        calls.append(command)
        output = Path(command[command.index("--out") + 1])
        strategy = command[command.index("--strategy") + 1]
        output.mkdir(parents=True)
        worker = interrupted_worker(strategy=strategy)
        if len(calls) > 1:
            worker.update(status="completed", exit_code=0, stop_reason=None,
                          protocol_valid=True, strategy_errors=0)
        elif kind == "stream":
            (output / "relay").mkdir()
            (output / "relay/runtime.log").write_text(interruption_log(strategy))
        elif kind == "context":
            worker.update(stop_reason="context_budget_exceeded", strategy_errors=0,
                          requests_over_context_budget=1, provider_input_tokens_per_request=[24000])
        elif kind == "quota":
            worker["error"] = "quota_check_failed:TimeoutError"
        (output / "run.json").write_text(json.dumps(worker))
        return subprocess.CompletedProcess(command, 1)

    monkeypatch.setattr(browse_matrix.subprocess, "run", finish_worker)
    output = tmp_path / "matrix"
    monkeypatch.setattr(sys, "argv", ["browse_matrix", "--root", str(tmp_path), "--out", str(output),
                                    "--reference", str(reference), "--models", "gpt-6.1-sol",
                                    "--methods", "compaction", "folding"])
    assert browse_matrix.main() == (0 if expected_calls == 2 else 1)
    assert len(calls) == expected_calls
    assert not list(output.glob("gpt-*/*/attempt-*"))
    first = json.loads((output / "matrix-summary.json").read_text())["runs"][0]
    assert first["score"]["strict_answer_score"] is None
    if kind == "stream":
        assert first["status"] == "infrastructure_failed"
        assert first["stop_reason"] == "upstream_stream_interrupted"


def test_saved_stream_grade_does_not_override_raw_interruption(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "matrix"
    attempt = output / "gpt-6.1-sol/rlm"
    (attempt / "relay").mkdir(parents=True)
    original_worker = json.dumps(interrupted_worker())
    original_grade = json.dumps({"strict_answer_score": 1, "protocol_valid": True})
    (attempt / "run.json").write_text(original_worker)
    (attempt / "browse-grade.json").write_text(original_grade)
    (attempt / "relay/runtime.log").write_text(interruption_log())
    monkeypatch.setattr(browse_matrix.subprocess, "run", lambda *a, **k: pytest.fail("No inference on saved attempts"))
    monkeypatch.setattr(sys, "argv", ["browse_matrix", "--root", str(tmp_path), "--out", str(output),
                                    "--models", "gpt-6.1-sol", "--methods", "rlm"])
    assert browse_matrix.main() == 0
    row = json.loads((output / "matrix-summary.json").read_text())["runs"][0]
    assert row["score"]["strict_answer_score"] is None
    assert row["stop_reason"] == "upstream_stream_interrupted"
    assert (attempt / "run.json").read_text() == original_worker
    assert (attempt / "browse-grade.json").read_text() == original_grade


def test_saved_quota_stop_blocks_resume_before_missing_slots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "matrix"
    attempt = output / "gpt-6.1-sol/folding"
    attempt.mkdir(parents=True)
    worker = interrupted_worker(strategy="folding", error="quota_check_failed:TimeoutError")
    original = json.dumps(worker)
    (attempt / "run.json").write_text(original)
    (attempt / "browse-grade.json").write_text(json.dumps({"strict_answer_score": 1}))
    monkeypatch.setattr(browse_matrix.subprocess, "run", lambda *a, **k: pytest.fail("Quota stop forbids new inference"))
    monkeypatch.setattr(sys, "argv", ["browse_matrix", "--root", str(tmp_path), "--out", str(output),
                                    "--models", "gpt-6.1-sol", "--methods", "compaction", "folding"])
    assert browse_matrix.main() == 1
    state = json.loads((output / "matrix-summary.json").read_text())
    assert state["halted"] is True
    assert state["stop_reason"] == "quota_check_failed:TimeoutError"
    assert (attempt / "run.json").read_text() == original
    assert not (output / "gpt-6.1-sol/compaction").exists()


@pytest.mark.parametrize("updates", [
    {"protocol_valid": None}, {"protocol_valid": False}, {"strategy_errors": 1},
    {"sandbox_execution_errors": 1}, {"failure_kind": "unknown_protocol_error"},
])
def test_grader_does_not_score_unknown_or_mixed_protocol_failure(tmp_path: Path, updates: dict) -> None:
    worker = interrupted_worker(status="completed", exit_code=0, stop_reason=None,
                                protocol_valid=True, strategy_errors=0)
    worker.update(updates)
    (tmp_path / "run.json").write_text(json.dumps(worker))
    (tmp_path / "answer.txt").write_text("FINAL_ANSWER: Expected")
    reference = tmp_path / "private.json"
    reference.write_text(json.dumps({"query_id": "183", "answer": "Expected", "qrels": {}}))
    assert grade(tmp_path, tmp_path / "retrieval.jsonl", reference)["strict_answer_score"] is None


@pytest.mark.parametrize("reason,blocked", [
    ("source_hash_mismatch", True), ("private_reference_readable", True),
    ("provider_capacity", True), ("upstream_stream_interrupted", False), ("protocol_invalid", False),
])
def test_specific_prior_audit_is_not_hidden_by_stream_failure(tmp_path: Path, reason: str, blocked: bool) -> None:
    (tmp_path / "relay").mkdir()
    (tmp_path / "relay/runtime.log").write_text(interruption_log())
    (tmp_path / "evaluation-audit.json").write_text(json.dumps({"valid": False, "reason": reason}))
    assert browse_matrix.blocks_continuation(tmp_path, interrupted_worker()) is blocked


def test_prior_local_audit_prevents_capacity_retry(tmp_path: Path) -> None:
    (tmp_path / "evaluation-audit.json").write_text(json.dumps({"valid": False, "reason": "private_reference_readable"}))
    worker = interrupted_worker(status="infrastructure_failed", strategy_errors=0,
                                error="Selected model is at capacity.")
    assert not browse_matrix.capacity_failed(tmp_path, worker)
    assert browse_matrix.blocks_continuation(tmp_path, worker)


def reviewed_capacity_case(path: Path, **updates: object) -> dict:
    worker = interrupted_worker(strategy="prolong", stop_reason="context_budget_exceeded",
                                requests_over_context_budget=1,
                                provider_input_tokens_per_request=[23481, 33999])
    worker.update(updates)
    (path / "relay").mkdir(parents=True, exist_ok=True)
    (path / "run.json").write_text(json.dumps(worker))
    (path / "relay/runtime.log").write_text(
        "prolong failed; forwarding the last context\n"
        "RuntimeError: summary request failed with HTTP 429: turn_rate_limit_exceeded\n")
    (path / "relay/trace.jsonl").write_text(json.dumps({"summary_status": 429}) + "\n")
    (path / "upstream-events.jsonl").write_text(json.dumps({"event": "response_headers", "status": 429}) + "\n")
    audit = {"reason": browse_matrix.REVIEWED_CAPACITY_OVERFLOW, "score": None,
             "continue_without_retry": True,
             "evidence_sha256": {name: hashlib.sha256((path / name).read_bytes()).hexdigest()
                                 for name in browse_matrix.CONTINUATION_EVIDENCE}}
    (path / "continuation-audit.json").write_text(json.dumps(audit))
    return worker


def test_reviewed_mixed_failure_requires_explicit_controller_record(tmp_path: Path) -> None:
    worker = reviewed_capacity_case(tmp_path)
    assert browse_matrix.reviewed_capacity_overflow(tmp_path, worker)
    assert not browse_matrix.blocks_continuation(tmp_path, worker)
    assert not browse_matrix.capacity_failed(tmp_path, worker)
    (tmp_path / "continuation-audit.json").unlink()
    assert browse_matrix.blocks_continuation(tmp_path, worker)


@pytest.mark.parametrize("name", browse_matrix.CONTINUATION_EVIDENCE)
@pytest.mark.parametrize("mutation", ["stale", "missing"])
def test_reviewed_mixed_failure_rejects_changed_or_missing_evidence(tmp_path: Path, name: str, mutation: str) -> None:
    worker = reviewed_capacity_case(tmp_path)
    evidence = tmp_path / name
    if mutation == "missing":
        evidence.unlink()
    else:
        evidence.write_bytes(evidence.read_bytes() + b"\n")
    assert browse_matrix.blocks_continuation(tmp_path, worker)
    assert not browse_matrix.capacity_failed(tmp_path, worker)


@pytest.mark.parametrize("mutation", ["invalid_json", "list", "reason", "score", "missing_score",
                                      "continue", "missing_hash", "extra_hash", "bad_hash"])
def test_reviewed_mixed_failure_rejects_malformed_review(tmp_path: Path, mutation: str) -> None:
    worker = reviewed_capacity_case(tmp_path)
    path = tmp_path / "continuation-audit.json"
    audit = json.loads(path.read_text())
    if mutation == "reason":
        audit["reason"] = "protocol_invalid"
    elif mutation == "score":
        audit["score"] = 0
    elif mutation == "missing_score":
        del audit["score"]
    elif mutation == "continue":
        audit["continue_without_retry"] = 1
    elif mutation == "missing_hash":
        del audit["evidence_sha256"]["relay/trace.jsonl"]
    elif mutation == "extra_hash":
        audit["evidence_sha256"]["other.log"] = "0" * 64
    elif mutation == "bad_hash":
        audit["evidence_sha256"]["run.json"] = "0" * 64
    path.write_text("{" if mutation == "invalid_json" else json.dumps([] if mutation == "list" else audit))
    assert browse_matrix.blocks_continuation(tmp_path, worker)
    assert not browse_matrix.capacity_failed(tmp_path, worker)


@pytest.mark.parametrize("updates", [
    {"status": "running"}, {"status": "completed"}, {"stop_reason": "wall_time_limit"},
    {"exit_code": 0}, {"exit_code": 1}, {"exit_code": None}, {"protocol_valid": True},
    {"strategy_errors": 0}, {"strategy_errors": True}, {"sandbox_execution_errors": 1},
    {"requests_over_context_budget": 0}, {"requests_over_context_budget": 2},
    {"provider_input_tokens_per_request": [200]}, {"provider_input_tokens_per_request": []},
    {"window": "23560"}, {"error": "local adapter failure"}, {"failure_kind": "unknown_protocol_error"},
    {"error": "quota_check_failed"}, {"failure_kind": "credit_limit"},
    {"stop_reason": "ordinary_plan_limit"},
])
def test_reviewed_mixed_failure_does_not_override_raw_gates(tmp_path: Path, updates: dict) -> None:
    worker = reviewed_capacity_case(tmp_path, **updates)
    assert browse_matrix.blocks_continuation(tmp_path, worker)
    assert not browse_matrix.capacity_failed(tmp_path, worker)


@pytest.mark.parametrize("reason", ["private_reference_readable", "source_hash_mismatch", "protocol_invalid",
                                   "provider_capacity", "upstream_stream_interrupted"])
def test_reviewed_mixed_failure_does_not_hide_other_invalid_audit(tmp_path: Path, reason: str) -> None:
    worker = reviewed_capacity_case(tmp_path)
    (tmp_path / "evaluation-audit.json").write_text(json.dumps({"valid": False, "reason": reason}))
    assert browse_matrix.blocks_continuation(tmp_path, worker)


def test_reviewed_mixed_failure_requires_supplied_metadata_to_match_bound_file(tmp_path: Path) -> None:
    worker = reviewed_capacity_case(tmp_path)
    worker["harness_requests"] = 99
    assert browse_matrix.blocks_continuation(tmp_path, worker)


def test_reviewed_mixed_failure_continues_next_arm_without_retry_or_numeric_grade(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "matrix"
    attempt = output / "gpt-6.1-sol/prolong"
    reviewed_capacity_case(attempt)
    original_worker = (attempt / "run.json").read_bytes()
    original_grade = json.dumps({"strict_answer_score": 1, "protocol_valid": True})
    (attempt / "browse-grade.json").write_text(original_grade)
    calls = []

    def finish_worker(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        calls.append(command)
        assert command[command.index("--strategy") + 1] == "selfcompact"
        target = Path(command[command.index("--out") + 1])
        target.mkdir(parents=True)
        (target / "run.json").write_text(json.dumps({"status": "completed", "exit_code": 0,
                                                     "protocol_valid": True}))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(browse_matrix.subprocess, "run", finish_worker)
    monkeypatch.setattr(sys, "argv", ["browse_matrix", "--root", str(tmp_path), "--out", str(output),
                                    "--models", "gpt-6.1-sol", "--methods", "prolong", "selfcompact"])
    assert browse_matrix.main() == 0
    assert len(calls) == 1
    assert not list(attempt.glob("attempt-*"))
    row = json.loads((output / "matrix-summary.json").read_text())["runs"][0]
    assert row["status"] == "infrastructure_failed"
    assert row["stop_reason"] == row["failure_kind"] == browse_matrix.REVIEWED_CAPACITY_OVERFLOW
    assert row["score"]["strict_answer_score"] is None
    assert row["score"]["protocol_valid"] is False
    assert (attempt / "run.json").read_bytes() == original_worker
    assert (attempt / "browse-grade.json").read_text() == original_grade
