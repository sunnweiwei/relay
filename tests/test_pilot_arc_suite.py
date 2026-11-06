"""Preserve ARC model attribution and distinguish setup failure from a score."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from relay.core.budget import MEDIA_INPUT_BUDGET_COUNTER, InputBudget
from evals.pilot import arc_suite


@pytest.mark.parametrize("status,scored", [
    ("completed", True), ("preflight_failed", False), ("infrastructure_failed", False),
])
def test_matrix_model_endpoint_and_score_attribution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str, scored: bool,
) -> None:
    scratch = tmp_path / "scratch"
    arc_cwd = scratch / "arc"
    arc_cwd.mkdir(parents=True)
    prompt = arc_cwd / "frozen_prompt.txt"
    original = "Play using http://127.0.0.1:18761. Do not call /score."
    prompt.write_text(original)
    out_root = scratch / "matrix" / "arc"
    model = "gpt-6.1-sol"
    arm = "ace"
    game_id = "bp35-0a0ad940"
    arm_dir = out_root / model / arm
    sentinel = object()
    stopped: list[object] = []
    service_limits: list[int] = []

    def start(*args: Any) -> tuple[object, str]:
        assert args[-3:-1] == (game_id, 7)
        service_limits.append(args[-1])
        (arm_dir / "service").mkdir()
        (arm_dir / "service" / "score-token").write_text("test-only-token")
        return sentinel, "test-only-token"

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert command[command.index("--model") + 1] == model
        assert command[command.index("--strategy") + 1] == arm
        assert command[command.index("--root") + 1] == str(scratch)
        assert command[command.index("--max-requests") + 1] == "200"
        assert command[command.index("--seconds") + 1] == "3600"
        assert "--input-budget-counter" not in command
        assert kwargs["env"]["ARC_SERVICE_URL"] == "http://127.0.0.1:18762"
        emitted_prompt = Path(command[command.index("--prompt") + 1])
        assert emitted_prompt.read_text() == original.replace("18761", "18762")
        worker = Path(command[command.index("--out") + 1])
        worker.mkdir()
        (worker / "run.json").write_text(json.dumps({"status": status, "model": model}))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(arc_suite, "start_service", start)
    monkeypatch.setattr(arc_suite, "stop_service", stopped.append)
    monkeypatch.setattr(arc_suite.subprocess, "run", run)
    monkeypatch.setattr(arc_suite, "read_json", lambda *a: {"action_count": 1, "score": 0})
    result = arc_suite.run_arm(
        arm, Path("python"), Path("run.py"), Path("arc_server.py"), scratch,
        arc_cwd, prompt, "test", model=model, out_root=out_root, port=18762,
        game_id=game_id, seed=7, max_requests=200, seconds=3600, max_actions=300,
    )
    assert result["model"] == model
    assert result["scored"] is scored
    assert result["worker_status"] == status
    metadata = json.loads((arm_dir / "arm.json").read_text())
    assert (metadata["game_id"], metadata["seed"]) == (game_id, 7)
    assert (metadata["request_limit"], metadata["seconds"], metadata["action_limit"]) == (200, 3600, 300)
    assert metadata["input_budget_counter"] == InputBudget.name
    assert service_limits == [300]
    assert stopped == [sentinel]
    assert prompt.read_text() == original


def test_media_counter_selection_is_forwarded_and_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    scratch = tmp_path / "scratch"
    arc_cwd = scratch / "arc"
    arc_cwd.mkdir(parents=True)
    prompt = arc_cwd / "frozen_prompt.txt"
    prompt.write_text("Play using http://127.0.0.1:18761.")
    out_root = scratch / "matrix" / "arc"
    arm_dir = out_root / "gpt-6-luna" / "compaction"
    service = object()

    def start(*args: Any) -> tuple[object, str]:
        service_dir = arm_dir / "service"
        service_dir.mkdir(parents=True)
        (service_dir / "score-token").write_text("token")
        return service, "token"

    monkeypatch.setattr(arc_suite, "start_service", start)
    monkeypatch.setattr(arc_suite, "stop_service", lambda _: None)
    monkeypatch.setattr(arc_suite, "read_json", lambda *args: {"action_count": 0, "score": 0})

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert command[command.index("--input-budget-counter") + 1] == MEDIA_INPUT_BUDGET_COUNTER
        worker = Path(command[command.index("--out") + 1])
        worker.mkdir()
        (worker / "run.json").write_text(json.dumps({"status": "completed", "model": "gpt-6-luna"}))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(arc_suite.subprocess, "run", run)
    arc_suite.run_arm("compaction", Path("python"), Path("run.py"), Path("arc_server.py"),
                      scratch, arc_cwd, prompt, "test", model="gpt-6-luna", out_root=out_root,
                      input_budget_counter=MEDIA_INPUT_BUDGET_COUNTER, no_capacity_retry=True)
    metadata = json.loads((arm_dir / "arm.json").read_text())
    assert metadata["input_budget_counter"] == MEDIA_INPUT_BUDGET_COUNTER
    assert metadata["no_capacity_retry"] is True
    command = json.loads((arm_dir / "worker_command.json").read_text())
    assert command[command.index("--input-budget-counter") + 1] == MEDIA_INPUT_BUDGET_COUNTER


def test_capacity_retry_can_be_disabled_without_moving_original_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    scratch = tmp_path / "scratch"
    (scratch / "arc").mkdir(parents=True)
    (scratch / "arc/frozen_prompt.txt").write_text("Frozen ARC prompt")
    out = scratch / "matrix/arc"
    calls: list[str] = []

    def run(arm: str, *args: Any, model: str, **kwargs: Any) -> dict[str, Any]:
        calls.append(arm)
        slot = out / model / arm
        (slot / "worker").mkdir(parents=True)
        failed = arm == "compaction"
        result = {"arm": arm, "model": model, "scored": not failed,
                  "worker_status": "infrastructure_failed" if failed else "completed",
                  "error": "provider_capacity" if failed else None,
                  "capacity_failure": failed}
        (slot / "worker/run.json").write_text(json.dumps({
            "status": result["worker_status"], "exit_code": 1 if failed else 0,
            "error": "Selected model is at capacity" if failed else None,
        }))
        (slot / "summary.json").write_text(json.dumps(result))
        return result

    monkeypatch.setattr(arc_suite, "run_arm", run)
    monkeypatch.setattr(sys, "argv", [
        "arc_suite", "--scratch", str(scratch), "--python", sys.executable,
        "--out-root", str(out), "--models", "gpt-6.1-sol", "--arms", "compaction", "clm",
        "--no-capacity-retry",
    ])
    assert arc_suite.main() == 0
    assert calls == ["compaction", "clm"]
    assert (out / "gpt-6.1-sol/compaction/worker/run.json").exists()
    assert not (out.parent / "diagnostics/arc-capacity-attempt1/gpt-6.1-sol/compaction").exists()
    assert not (out / "capacity-retries.json").exists()


def test_service_is_stopped_if_initial_state_read_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Play using http://127.0.0.1:18761.")
    service = object()
    stopped: list[object] = []
    monkeypatch.setattr(arc_suite, "start_service", lambda *a: (service, "token"))
    monkeypatch.setattr(arc_suite, "stop_service", stopped.append)

    def unavailable(*args: Any) -> None:
        raise OSError("service disconnected")

    monkeypatch.setattr(arc_suite, "read_json", unavailable)
    with pytest.raises(OSError, match="service disconnected"):
        arc_suite.run_arm("compaction", Path("python"), Path("runner"), Path("server"),
                          tmp_path, tmp_path, prompt, "test")
    assert stopped == [service]


@pytest.mark.parametrize("stop_reason,invalid", [(None, True), ("model_request_limit", False)])
def test_raw_capacity_failure_is_unscored_unless_worker_hit_its_enforced_limit(
    tmp_path: Path, stop_reason: str | None, invalid: bool,
) -> None:
    worker = tmp_path / "worker"
    worker.mkdir()
    (worker / "run.json").write_text(json.dumps({
        "status": "stopped", "exit_code": 1, "stop_reason": stop_reason,
    }))
    (worker / "codex.jsonl").write_text(json.dumps({
        "type": "turn.failed", "error": {"message": "Selected model is at capacity"},
    }) + "\n")
    result = arc_suite.apply_worker_audit(tmp_path, {"scored": True, "worker_status": "stopped"})
    assert result["scored"] is not invalid
    if invalid:
        assert result["capacity_failure"] is True
        assert (tmp_path / "evaluation-audit.json").is_file()


def test_capacity_retry_is_once_and_matrix_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    scratch = tmp_path / "scratch"
    (scratch / "arc").mkdir(parents=True)
    (scratch / "arc/frozen_prompt.txt").write_text("Frozen ARC prompt")
    out = scratch / "matrix/arc"
    calls: list[str] = []

    def run(arm: str, *args: Any, model: str, **kwargs: Any) -> dict[str, Any]:
        calls.append(arm)
        slot = out / model / arm
        (slot / "worker").mkdir(parents=True)
        capacity = arm == "compaction"
        summary = {"arm": arm, "model": model, "scored": not capacity,
                   "worker_status": "infrastructure_failed" if capacity else "completed",
                   "error": "provider_capacity" if capacity else None,
                   "capacity_failure": capacity}
        (slot / "worker/run.json").write_text(json.dumps({
            "status": summary["worker_status"], "exit_code": 1 if capacity else 0,
            "error": "Selected model is at capacity" if capacity else None,
        }))
        (slot / "summary.json").write_text(json.dumps(summary))
        return summary

    monkeypatch.setattr(arc_suite, "run_arm", run)
    monkeypatch.setattr(sys, "argv", [
        "arc_suite", "--scratch", str(scratch), "--python", sys.executable,
        "--out-root", str(out), "--models", "gpt-6.1-sol", "--arms", "compaction", "clm", "--resume",
    ])
    arc_suite.main()
    assert calls == ["compaction", "compaction", "clm"]
    assert (out.parent / "diagnostics/arc-capacity-attempt1/gpt-6.1-sol/compaction/worker/run.json").exists()
    assert json.loads((out / "capacity-retries.json").read_text()) == {"gpt-6.1-sol/compaction": 1}
    arc_suite.main()
    assert calls == ["compaction", "compaction", "clm"]


def _saved_stream_interruption(slot: Path) -> dict[str, Any]:
    worker = slot / "worker"
    (worker / "relay").mkdir(parents=True)
    metadata = {
        "status": "stopped", "exit_code": -15, "stop_reason": "wall_time_limit",
        "strategy": "rlm", "protocol_valid": False, "strategy_errors": 1,
        "sandbox_execution_errors": 0, "requests_over_context_budget": 0,
        "window": 32000, "provider_input_tokens_per_request": [14000],
    }
    (worker / "run.json").write_text(json.dumps(metadata))
    (worker / "relay/runtime.log").write_text(
        "rlm failed; forwarding the last context\nTraceback (most recent call last):\n"
        "httpx.RemoteProtocolError: peer closed connection without sending complete message body "
        "(incomplete chunked read)\n")
    summary = {"worker_status": "infrastructure_failed", "protocol_valid": False,
               "scored": False, "error": "protocol_invalid", "stop_reason": "wall_time_limit"}
    (slot / "summary.json").write_text(json.dumps(summary))
    (slot / "service").mkdir()
    (slot / "service/arc_trace.jsonl").write_text(json.dumps({
        "kind": "reset", "action_count": 1,
        "observation": {"state": "NOT_FINISHED", "frame_ascii": "01"},
    }) + "\n")
    (slot / "evaluation-audit.json").write_text(json.dumps({
        "valid": False, "classification": "infrastructure_failed",
        "reason": "protocol_invalid", "capacity_failure": False,
    }))
    return summary


def test_resume_skips_audited_stream_failure_without_rerunning_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    scratch = tmp_path / "scratch"
    (scratch / "arc").mkdir(parents=True)
    (scratch / "arc/frozen_prompt.txt").write_text("Frozen ARC prompt")
    out = scratch / "matrix/arc"
    slot = out / "gpt-6.1-sol/rlm"
    _saved_stream_interruption(slot)
    raw = (slot / "worker/run.json").read_bytes()
    calls: list[str] = []

    def run(arm: str, *args: Any, model: str, **kwargs: Any) -> dict[str, Any]:
        calls.append(arm)
        next_slot = out / model / arm
        (next_slot / "worker").mkdir(parents=True)
        summary = {"worker_status": "completed", "scored": True, "protocol_valid": True}
        (next_slot / "worker/run.json").write_text(json.dumps({"status": "completed", "exit_code": 0}))
        (next_slot / "summary.json").write_text(json.dumps(summary))
        return summary

    monkeypatch.setattr(arc_suite, "run_arm", run)
    monkeypatch.setattr(sys, "argv", [
        "arc_suite", "--scratch", str(scratch), "--python", sys.executable,
        "--out-root", str(out), "--models", "gpt-6.1-sol", "--arms", "rlm", "rlm_persistent", "--resume",
    ])
    assert arc_suite.main() == 0
    assert arc_suite.main() == 0
    assert calls == ["rlm_persistent"]
    assert (slot / "worker/run.json").read_bytes() == raw
    summary = json.loads((slot / "summary.json").read_text())
    assert summary["worker_status"] == "infrastructure_failed"
    assert summary["scored"] is False
    assert summary["error"] == "upstream_stream_interrupted"
    prior = json.loads((slot / "evaluation-audit.before-upstream-stream-interrupted.json").read_text())
    assert prior["reason"] == "protocol_invalid"
    assert not (out / "capacity-retries.json").exists()


@pytest.mark.parametrize("defect", [
    "quota", "other_strategy", "missing_observation", "missing_trace", "malformed_trace", "old_observation_audit",
])
def test_stream_label_cannot_hide_another_failure_on_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, defect: str,
) -> None:
    scratch = tmp_path / "scratch"
    (scratch / "arc").mkdir(parents=True)
    (scratch / "arc/frozen_prompt.txt").write_text("Frozen ARC prompt")
    out = scratch / "matrix/arc"
    slot = out / "gpt-6.1-sol/rlm"
    summary = _saved_stream_interruption(slot)
    summary["error"] = "upstream_stream_interrupted"  # Saved labels never authorize admission.
    (slot / "summary.json").write_text(json.dumps(summary))
    if defect == "quota":
        path = slot / "worker/run.json"
        worker = json.loads(path.read_text())
        worker["error"] = "credit_balance_decreased"
        path.write_text(json.dumps(worker))
    elif defect == "other_strategy":
        (slot / "worker/relay/runtime.log").write_text("ValueError: unrelated strategy failure\n")
    elif defect == "missing_trace":
        (slot / "service/arc_trace.jsonl").unlink()
    elif defect == "malformed_trace":
        with (slot / "service/arc_trace.jsonl").open("a") as stream:
            stream.write("{incomplete\n")
    elif defect == "old_observation_audit":
        (slot / "evaluation-audit.json").write_text(json.dumps({
            "valid": False, "classification": "infrastructure_failed",
            "reason": "arc_observation_missing", "capacity_failure": False,
        }))
    else:
        (slot / "service/arc_trace.jsonl").write_text(json.dumps({"observation": None}) + "\n")

    def refuse(*args: Any, **kwargs: Any) -> None:
        pytest.fail("A mixed or unaudited failure must stop later inference")

    monkeypatch.setattr(arc_suite, "run_arm", refuse)
    monkeypatch.setattr(sys, "argv", [
        "arc_suite", "--scratch", str(scratch), "--python", sys.executable,
        "--out-root", str(out), "--models", "gpt-6.1-sol", "--arms", "rlm", "rlm_persistent", "--resume",
    ])
    assert arc_suite.main() == 2
    assert json.loads((slot / "summary.json").read_text())["scored"] is False
