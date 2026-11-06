"""A resumed Science controller must not duplicate an unfinished attempt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from evals.pilot import science_sample_matrix


@pytest.mark.parametrize("outer_status,worker_status", [
    ("preparing", None),
    ("worker_running", None),
    ("grading", "completed"),
    ("completed", "running"),
    (None, "running"),
])
def test_resume_refuses_unfinished_attempt_before_starting_another(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    outer_status: str | None, worker_status: str | None,
) -> None:
    output = tmp_path / "sample"
    slot = output / "task-a/gpt-6-luna/compaction"
    # A prior valid score must not hide a newer worker that still consumes quota.
    graded = slot / "attempt-001"
    graded.mkdir(parents=True)
    (graded / "result.json").write_text(json.dumps({"status": "completed"}))
    unfinished = slot / "attempt-002"
    unfinished.mkdir()
    (unfinished / "result.json").write_text(json.dumps({"status": outer_status}))
    if worker_status:
        (unfinished / "worker").mkdir()
        (unfinished / "worker/run.json").write_text(json.dumps({"status": worker_status}))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"selected_ids": ["task-a"]}))
    monkeypatch.setattr(science_sample_matrix, "require_readiness", lambda *args: {})

    def refuse_launch(*args: object, **kwargs: object) -> None:
        pytest.fail("An unfinished attempt must block all new worker launches")

    monkeypatch.setattr(science_sample_matrix.subprocess, "run", refuse_launch)
    args = argparse.Namespace(out=output, scratch=tmp_path, manifest=manifest,
                              tasks_root=tmp_path / "tasks", images_root=tmp_path / "images",
                              models=["gpt-6-luna"], methods=["compaction"], python=Path("python3"))
    assert science_sample_matrix.run_matrix(args) == 2
    state = json.loads((output / "matrix.json").read_text())
    assert state["status"] == "unfinished_attempt_no_retry"
    assert state["entries"] == [{
        "task_id": "task-a", "model": "gpt-6-luna", "strategy": "compaction",
        "path": str(unfinished), "status": "unfinished_attempt",
        "result_status": outer_status, "worker_status": worker_status,
    }]
    assert not (slot / "attempt-003").exists()
