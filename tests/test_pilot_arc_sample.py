"""Sampled ARC execution must preserve selection and stop on suite failures."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from evals.pilot import arc_sample, arc_suite
from evals.pilot.arc_server import ArcRun


def frozen_sample(tmp_path: Path) -> tuple[Path, Path]:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    (runtime / "frozen_prompt.txt").write_text("Use the local ARC service.")
    manifest = tmp_path / "manifest.json"
    ranked = sorted(["bp35-0a0ad940", "sk48-d8078629"],
                    key=lambda game: hashlib.sha256(f"20261026:{game}".encode()).hexdigest())
    manifest.write_text(json.dumps({"population_size": 2, "seed": 20261026,
        "gameplay_seed": 0, "ranked_ids": ranked, "selected_ids": ranked}))
    preflight = tmp_path / "preflight.json"
    preflight.write_text(json.dumps({"passed": True, "files": [],
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "selected_ids": ranked}))
    return manifest, preflight


def test_changed_selection_or_runtime_invalidates_preflight(tmp_path: Path) -> None:
    manifest, preflight = frozen_sample(tmp_path)
    arc_sample.load_sample(manifest, preflight)
    runtime = tmp_path / "game.py"
    runtime.write_text("old runtime")
    check = json.loads(preflight.read_text())
    check["files"] = [{"path": str(runtime), "sha256": hashlib.sha256(runtime.read_bytes()).hexdigest()}]
    preflight.write_text(json.dumps(check))
    runtime.write_text("changed runtime")
    with pytest.raises(ValueError, match="runtime file changed"):
        arc_sample.load_sample(manifest, preflight)
    selection = json.loads(manifest.read_text())
    selection["selected_ids"].reverse()
    manifest.write_text(json.dumps(selection))
    with pytest.raises(ValueError, match="deterministic population prefix"):
        arc_sample.load_sample(manifest, preflight)


@pytest.mark.parametrize("worker_status,stop_reason", [
    ("infrastructure_failed", "sandbox_error"), ("stopped", "plan_quota_start"),
])
def test_game_matrix_failure_prevents_later_game_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, worker_status: str, stop_reason: str,
) -> None:
    manifest, preflight = frozen_sample(tmp_path)
    selected = json.loads(manifest.read_text())["selected_ids"]
    calls = []
    arms = []
    runtime = tmp_path / "runtime"
    runtime.mkdir(exist_ok=True)
    (runtime / "frozen_prompt.txt").write_text("Use the local ARC service.")

    def arm(arm: str, *args: object, **kwargs: object) -> dict[str, object]:
        arms.append(arm)
        return {"worker_status": worker_status, "stop_reason": stop_reason, "scored": False}

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        assert command[command.index("--game-id") + 1] == selected[0]
        assert command[command.index("--out-root") + 1] == str(tmp_path / "out" / selected[0])
        assert command[command.index("--seed") + 1] == "0"
        assert "--resume" in command
        # Exercise the actual suite stop path, not a fabricated child exit code.
        with monkeypatch.context() as context:
            context.setattr(sys, "argv", command[2:])
            return subprocess.CompletedProcess(command, arc_suite.main())

    monkeypatch.setattr(arc_sample.subprocess, "run", run)
    monkeypatch.setattr(arc_suite, "run_arm", arm)
    monkeypatch.setattr(sys, "argv", ["arc_sample", "--manifest", str(manifest),
        "--preflight", str(preflight), "--scratch", str(tmp_path),
        "--python", sys.executable, "--runtime-root", str(tmp_path / "runtime"),
        "--out-root", str(tmp_path / "out")])
    assert arc_sample.main() == 2
    assert len(calls) == 1
    assert arms == ["compaction"]
    status = json.loads((tmp_path / "out/sample-status.json").read_text())
    assert status["state"] == "stopped"
    assert status["completed_game_matrices"] == []


def test_resumed_protocol_invalid_worker_cannot_keep_a_valid_score(tmp_path: Path) -> None:
    worker = tmp_path / "worker"
    worker.mkdir()
    (worker / "run.json").write_text(json.dumps({
        "status": "completed", "exit_code": 0, "protocol_valid": False,
    }))
    summary = arc_suite.apply_worker_audit(tmp_path, {
        "scored": True, "worker_status": "completed", "protocol_valid": True,
    })
    assert summary["scored"] is False
    assert summary["worker_status"] == "infrastructure_failed"
    assert summary["error"] == "protocol_invalid"


def test_exhausted_capacity_stays_unknown_in_sample_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, preflight = frozen_sample(tmp_path)

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        out = Path(command[command.index("--out-root") + 1])
        for model in ("gpt-6.1-sol", "gpt-6-luna"):
            arm = out / model / "compaction"
            arm.mkdir(parents=True)
            (arm / "summary.json").write_text(json.dumps({
                "scored": model == "gpt-6-luna", "error": "provider_capacity",
            }))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(arc_sample, "EVALUATED_METHODS", ("compaction",))
    monkeypatch.setattr(arc_sample.subprocess, "run", run)
    monkeypatch.setattr(sys, "argv", ["arc_sample", "--manifest", str(manifest),
        "--preflight", str(preflight), "--scratch", str(tmp_path),
        "--python", sys.executable, "--runtime-root", str(tmp_path / "runtime"),
        "--out-root", str(tmp_path / "out")])
    assert arc_sample.main() == 0
    status = json.loads((tmp_path / "out/sample-status.json").read_text())
    assert status["state"] == "completed_with_unknown"
    assert status["scored_cells"] == 2
    assert len(status["unknown_cells"]) == 2


def test_server_requires_a_pinned_game_revision(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exact public game revision"):
        ArcRun("bp35", 0, tmp_path / "trace.jsonl", 300)


def test_sample_status_honors_reviewed_exclusion_without_rewriting_raw_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, preflight = frozen_sample(tmp_path)
    raw_summary = json.dumps({"scored": True, "worker_status": "stopped", "error": None})
    summary_paths = []

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        out = Path(command[command.index("--out-root") + 1])
        for model in ("gpt-6.1-sol", "gpt-6-luna"):
            arm = out / model / "rlm_persistent"
            arm.mkdir(parents=True)
            summary = arm / "summary.json"
            summary.write_text(raw_summary)
            summary_paths.append(summary)
            if model == "gpt-6.1-sol":
                (arm / "evaluation-audit.json").write_text(json.dumps({
                    "valid": False, "reason": "rlm_persistent_session_reset",
                    "classification": "infrastructure_failed", "benchmark_score": None,
                }))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(arc_sample, "EVALUATED_METHODS", ("rlm_persistent",))
    monkeypatch.setattr(arc_sample.subprocess, "run", run)
    monkeypatch.setattr(sys, "argv", ["arc_sample", "--manifest", str(manifest),
        "--preflight", str(preflight), "--scratch", str(tmp_path),
        "--python", sys.executable, "--runtime-root", str(tmp_path / "runtime"),
        "--out-root", str(tmp_path / "out")])
    assert arc_sample.main() == 0
    status = json.loads((tmp_path / "out/sample-status.json").read_text())
    assert status["state"] == "completed_with_unknown"
    assert status["scored_cells"] == 2
    assert len(status["unknown_cells"]) == 2
    assert {cell["reason"] for cell in status["unknown_cells"]} == {"rlm_persistent_session_reset"}
    assert all(path.read_text() == raw_summary for path in summary_paths)


def prompt_policy(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path, list[str]]:
    manifest, preflight = frozen_sample(tmp_path)
    selected = json.loads(manifest.read_text())["selected_ids"]
    baseline = tmp_path / "runtime/frozen_prompt.txt"
    baseline.write_text(f"Use /step, for example {arc_sample.COORDINATE_EXAMPLE_V1}.\n")
    corrected = tmp_path / "prompt-v2.txt"
    corrected.write_text(baseline.read_text().replace(
        arc_sample.COORDINATE_EXAMPLE_V1, arc_sample.COORDINATE_EXAMPLE_V2))
    policy = tmp_path / "prompt-policy.json"
    policy.write_text(json.dumps({"version": 1,
        "base_prompt_sha256": hashlib.sha256(baseline.read_bytes()).hexdigest(),
        "games": {game: {"path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for game, path in zip(selected, (baseline, corrected))}}))
    pin_prompt_files(preflight, policy, baseline, corrected)
    return manifest, preflight, policy, baseline, corrected, selected


def pin_prompt_files(preflight: Path, *files: Path) -> None:
    value = json.loads(preflight.read_text())
    value["files"] = [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                      for path in files]
    preflight.write_text(json.dumps(value))


def test_per_game_prompt_policy_keeps_started_game_and_corrects_only_untouched_game(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, preflight, policy, baseline, corrected, selected = prompt_policy(tmp_path)
    old_arm = tmp_path / "out" / selected[0] / "gpt-6.1-sol/compaction"
    old_arm.mkdir(parents=True)
    (old_arm / "prompt.txt").write_bytes(baseline.read_bytes())
    raw = old_arm / "worker/run.json"
    raw.parent.mkdir()
    raw.write_text('{"status":"completed"}')
    raw_bytes = raw.read_bytes()
    calls = []

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        out = Path(command[command.index("--out-root") + 1])
        for model in ("gpt-6.1-sol", "gpt-6-luna"):
            arm = out / model / "compaction"
            arm.mkdir(parents=True, exist_ok=True)
            (arm / "summary.json").write_text('{"scored":true}')
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(arc_sample, "EVALUATED_METHODS", ("compaction",))
    monkeypatch.setattr(arc_sample.subprocess, "run", run)
    monkeypatch.setattr(sys, "argv", ["arc_sample", "--manifest", str(manifest),
        "--preflight", str(preflight), "--scratch", str(tmp_path),
        "--python", sys.executable, "--runtime-root", str(tmp_path / "runtime"),
        "--out-root", str(tmp_path / "out"), "--prompt-policy", str(policy)])
    assert arc_sample.main() == 0
    assert [call[call.index("--prompt") + 1] for call in calls] == [str(baseline), str(corrected)]
    assert raw.read_bytes() == raw_bytes
    assert (old_arm / "prompt.txt").read_bytes() == baseline.read_bytes()
    for game, prompt in zip(selected, (baseline, corrected)):
        record = json.loads((tmp_path / "out" / game / "prompt-policy.json").read_text())
        assert record["sha256"] == hashlib.sha256(prompt.read_bytes()).hexdigest()


def test_prompt_policy_cannot_change_any_started_game(tmp_path: Path) -> None:
    manifest, preflight, policy, baseline, corrected, selected = prompt_policy(tmp_path)
    arm = tmp_path / "out" / selected[1] / "gpt-6-luna/compaction"
    arm.mkdir(parents=True)
    (arm / "prompt.txt").write_bytes(baseline.read_bytes())
    with pytest.raises(ValueError, match="differs from an existing attempt"):
        arc_sample.load_game_prompts(json.loads(manifest.read_text()), preflight,
            baseline.parent, tmp_path / "out", policy)


def test_prompt_policy_requires_original_evidence_for_started_arms(tmp_path: Path) -> None:
    manifest, preflight, policy, baseline, corrected, selected = prompt_policy(tmp_path)
    arm = tmp_path / "out" / selected[1] / "gpt-6-luna/compaction/worker"
    arm.mkdir(parents=True)
    (arm / "run.json").write_text('{"status":"completed"}')
    with pytest.raises(ValueError, match="no retained prompt evidence"):
        arc_sample.load_game_prompts(json.loads(manifest.read_text()), preflight,
            baseline.parent, tmp_path / "out", policy)


@pytest.mark.parametrize("change", ["extra_instruction", "line_endings"])
def test_prompt_policy_rejects_additional_changes_even_if_repinned(tmp_path: Path, change: str) -> None:
    manifest, preflight, policy, baseline, corrected, selected = prompt_policy(tmp_path)
    content = corrected.read_bytes()
    corrected.write_bytes(content + b"Extra hint.\n" if change == "extra_instruction"
                          else content.replace(b"\n", b"\r\n"))
    value = json.loads(policy.read_text())
    value["games"][selected[1]]["sha256"] = hashlib.sha256(corrected.read_bytes()).hexdigest()
    policy.write_text(json.dumps(value))
    pin_prompt_files(preflight, policy, baseline, corrected)
    with pytest.raises(ValueError, match="changes more than the example"):
        arc_sample.load_game_prompts(json.loads(manifest.read_text()), preflight,
            baseline.parent, tmp_path / "out", policy)


def test_prompt_policy_and_corrected_file_require_preflight_pins(tmp_path: Path) -> None:
    manifest, preflight, policy, baseline, corrected, selected = prompt_policy(tmp_path)
    pin_prompt_files(preflight, baseline, corrected)
    with pytest.raises(ValueError, match="policy is not pinned"):
        arc_sample.load_game_prompts(json.loads(manifest.read_text()), preflight,
            baseline.parent, tmp_path / "out", policy)
    pin_prompt_files(preflight, baseline, policy)
    with pytest.raises(ValueError, match="prompt is unpinned"):
        arc_sample.load_game_prompts(json.loads(manifest.read_text()), preflight,
            baseline.parent, tmp_path / "out", policy)


def test_selected_prompt_survives_resume_and_cannot_silently_revert(tmp_path: Path) -> None:
    manifest, preflight, policy, baseline, corrected, selected = prompt_policy(tmp_path)
    game = tmp_path / "out" / selected[1]
    arc_sample.freeze_game_prompt(game, corrected, policy)
    record = (game / "prompt-policy.json").read_bytes()
    arc_sample.freeze_game_prompt(game, corrected, policy)
    assert (game / "prompt-policy.json").read_bytes() == record
    with pytest.raises(ValueError, match="changed after selection"):
        arc_sample.load_game_prompts(json.loads(manifest.read_text()), preflight,
            baseline.parent, tmp_path / "out", None)


def test_suite_prompt_freeze_alone_prevents_a_conflicting_prompt_lock(tmp_path: Path) -> None:
    manifest, preflight, policy, baseline, corrected, selected = prompt_policy(tmp_path)
    game = tmp_path / "out" / selected[1]
    game.mkdir(parents=True)
    prior = json.dumps({"sha256": hashlib.sha256(baseline.read_bytes()).hexdigest(),
                        "game_id": selected[1], "frozen_before_first_run": True})
    (game / "prompt-freeze.json").write_text(prior)
    with pytest.raises(ValueError, match="prompt-freeze.json"):
        arc_sample.freeze_game_prompt(game, corrected, policy)
    assert not (game / "prompt-policy.json").exists()
    assert (game / "prompt-freeze.json").read_text() == prior
    arc_sample.freeze_game_prompt(game, baseline, None)
    assert json.loads((game / "prompt-policy.json").read_text())["sha256"] == hashlib.sha256(
        baseline.read_bytes()).hexdigest()


def test_game_prompt_allows_only_the_existing_endpoint_substitution(tmp_path: Path) -> None:
    prompt = tmp_path / "prompt.txt"
    prompt.write_text(f"Use http://127.0.0.1:{arc_sample.PORT}/state.\n")
    game = tmp_path / "game"
    arm = game / "gpt-6-luna/compaction"
    arm.mkdir(parents=True)
    (arm / "prompt.txt").write_text("Use http://127.0.0.1:19876/state.\n")
    arc_sample.freeze_game_prompt(game, prompt, None, port=19876)
    assert json.loads((game / "prompt-policy.json").read_text())["sha256"] == hashlib.sha256(
        prompt.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="differs from an existing attempt"):
        arc_sample.validate_game_prompt(game, prompt, port=19877)
    (arm / "prompt.txt").write_text("Use http://127.0.0.1:19876/state.\nExtra hint.\n")
    with pytest.raises(ValueError, match="differs from an existing attempt"):
        arc_sample.validate_game_prompt(game, prompt, port=19876)
