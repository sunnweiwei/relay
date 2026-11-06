"""ARC animation batches must expose the final playable observation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.pilot.arc_server import frame_grid, grid_ascii
from evals.pilot.arc_suite import apply_worker_audit


def test_animation_and_level_transition_show_the_latest_frame() -> None:
    first = [[0, 1], [2, 3]]
    last = [[4, 5], [14, 15]]
    for frames in ([first, last], [first] * 5 + [last]):
        assert frame_grid(frames) == last
        assert grid_ascii(frames) == "45\nEF"
    assert frame_grid([last]) == frame_grid(last) == last


def test_invalid_final_frame_is_not_replaced_with_a_stale_frame() -> None:
    assert frame_grid([[[0]], [[16]]]) is None


def test_missing_nonterminal_observation_invalidates_a_nominal_score(tmp_path: Path) -> None:
    service = tmp_path / "service"
    service.mkdir()
    (service / "arc_trace.jsonl").write_text(json.dumps({
        "action_count": 26, "observation": {"state": "NOT_FINISHED", "frame": [[[1]], [[2]]],
                                              "frame_ascii": None}}) + "\n")
    result = apply_worker_audit(tmp_path, {"scored": True, "worker_status": "completed"})
    assert result["scored"] is False
    assert result["error"] == "arc_observation_missing"
    assert json.loads((tmp_path / "evaluation-audit.json").read_text())["missing_observation_actions"] == [26]


def test_absent_terminal_frame_does_not_invalidate_a_score(tmp_path: Path) -> None:
    service = tmp_path / "service"
    service.mkdir()
    (service / "arc_trace.jsonl").write_text(json.dumps({
        "action_count": 27, "observation": {"state": "GAME_OVER", "frame": [], "frame_ascii": None}}) + "\n")
    result = apply_worker_audit(tmp_path, {"scored": True, "worker_status": "completed"})
    assert result["scored"] is True


@pytest.mark.parametrize("observation", [
    None, {}, {"state": None, "frame": None, "frame_ascii": None},
    {"state": "UNKNOWN", "frame_ascii": "01"},
    {"state": "NOT_FINISHED", "frame": None, "frame_ascii": None},
    {"state": "NOT_FINISHED", "frame": [], "frame_ascii": ""},
])
def test_null_or_missing_observation_cannot_remain_scored(tmp_path: Path, observation: object) -> None:
    service = tmp_path / "service"
    service.mkdir()
    (service / "arc_trace.jsonl").write_text(json.dumps({
        "kind": "step", "action_count": 3, "action": {"action": "ACTION6"},
        "observation": observation,
    }) + "\n")
    result = apply_worker_audit(tmp_path, {"scored": True, "worker_status": "completed"})
    assert result["scored"] is False
    assert result["worker_status"] == "infrastructure_failed"
    assert result["error"] == "arc_observation_missing"
    audit = json.loads((tmp_path / "evaluation-audit.json").read_text())
    assert audit["missing_observation_actions"] == [3]
