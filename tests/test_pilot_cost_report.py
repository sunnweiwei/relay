from pathlib import Path
import json

from evals.pilot.cost_report import collect_cost, worker_directories
from evals.pilot.matrix_report import usage


def test_retained_retry_is_counted_without_traversing_worker_dependencies(tmp_path: Path) -> None:
    first = tmp_path / "browse/model/strategy"
    retry = first / "attempt-002"
    archive = tmp_path / "diagnostics/arc/model/strategy/worker"
    misleading = first / "codex/plugins/example"
    for path in (first, retry, archive, misleading):
        path.mkdir(parents=True, exist_ok=True)
        (path / "run.json").write_text("{}")
    assert set(worker_directories(tmp_path)) == {first, retry, archive}


def test_interrupted_auxiliary_usage_is_unknown_even_after_worker_stops(tmp_path: Path) -> None:
    worker = tmp_path / "arc/model/rlm/worker"
    (worker / "codex/sessions").mkdir(parents=True)
    (worker / "relay").mkdir()
    (worker / "run.json").write_text(json.dumps({"status": "stopped", "window": 32000}))
    (worker / "codex/sessions/rollout-test.jsonl").write_text(json.dumps({"payload": {
        "type": "token_count", "info": {"total_token_usage": {"input_tokens": 100, "output_tokens": 10}}}}) + "\n")
    (worker / "relay/trace.jsonl").write_text(json.dumps({"usage": 100}) + "\n")
    (worker / "upstream-events.jsonl").write_text(
        (json.dumps({"event": "request", "path": "/backend-api/codex/responses"}) + "\n") * 3)
    cost = collect_cost([tmp_path])
    attempt = cost["attempts"][0]
    assert attempt["known_tokens"] == 110
    assert attempt["unaccounted_upstream_attempts"] == 2
    assert attempt["usage_incomplete"] is True
    assert cost["groups"]["all"]["incomplete_usage_attempts"] == 1
    # The sample report receives a filtered usage view; uncertainty must survive it.
    assert usage(worker)["unaccounted_upstream_attempts"] == 2
