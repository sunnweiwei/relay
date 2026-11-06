"""Private post-run diagnostics for the fixed BrowseComp-Plus pilot.

This is an exact-answer check, not the benchmark's official LLM judge. Never
copy this grader or its private reference input into a worker workspace.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from evals.pilot.failure_policy import UPSTREAM_STREAM_INTERRUPTED, upstream_stream_interrupted


def quota_stop_reason(run: dict[str, Any]) -> str | None:
    for key in ("stop_reason", "error", "failure_kind"):
        reason = str(run.get(key) or "")
        if any(term in reason.lower() for term in ("quota", "credit", "ordinary_plan", "account_limit")):
            return reason
    return None


def classified_worker(run_dir: Path, run: dict[str, Any]) -> dict[str, Any]:
    """Classify derived results without changing the preserved worker metadata."""
    if reason := quota_stop_reason(run):
        return run | {"protocol_valid": False, "stop_reason": reason, "failure_kind": "quota_stopped"}
    if upstream_stream_interrupted(run, run_dir / "relay/runtime.log"):
        return run | {"status": "infrastructure_failed", "protocol_valid": False,
                      "worker_stop_reason": run.get("stop_reason"),
                      "stop_reason": UPSTREAM_STREAM_INTERRUPTED,
                      "failure_kind": UPSTREAM_STREAM_INTERRUPTED,
                      "error": UPSTREAM_STREAM_INTERRUPTED}
    return dict(run)


def final_answer(text: str) -> str | None:
    match = re.search(r"FINAL_ANSWER\s*\*{0,2}\s*[:=]?\s*\*{0,2}\s*(.+)", text)
    if not match:
        return None
    return match.group(1).strip().strip("*` ") or None


def normalize(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.casefold()))


def contains_gold(answer: str | None, gold: str) -> bool:
    """Lenient diagnostic: the gold answer's words appear contiguously in the final answer.

    It catches correct answers with extra words ("12th Fail (2023)") that the strict
    exact match misses, but it can also credit an answer that lists several candidates."""
    return answer is not None and f" {normalize(gold)} " in f" {normalize(answer)} "


def grade(run_dir: Path, retrieval: Path, reference: Path, expected_query_id: str | None = None) -> dict[str, Any]:
    run = classified_worker(run_dir, json.loads((run_dir / "run.json").read_text()))
    gold = json.loads(reference.read_text())
    task_path = run_dir / "benchmark-task.json"
    task = json.loads(task_path.read_text()) if task_path.exists() else {}
    expected_id = expected_query_id or task.get("query_id")
    if expected_id is not None and str(gold["query_id"]) != str(expected_id):
        raise ValueError("Grader reference does not match the worker's query ID")
    answer_path = run_dir / "answer.txt"
    answer = final_answer(answer_path.read_text()) if answer_path.is_file() else None
    inference_recorded = bool(run.get("provider_input_tokens_per_request")) or bool(
        (run.get("codex_final_total_usage") or {}).get("total_tokens"))
    terminal = ((run.get("status") == "completed" and run.get("exit_code") == 0) or
                (run.get("status") == "stopped"
                 and run.get("stop_reason") in {"wall_time_limit", "model_request_limit"}
                 and type(run.get("exit_code")) is int and run["exit_code"] in (0, 1, -9, -15)))
    score_eligible = inference_recorded and terminal and run.get("protocol_valid") is True
    if (run.get("strategy_errors") or run.get("sandbox_execution_errors")
            or run.get("requests_over_context_budget") or run.get("error") or run.get("failure_kind")):
        score_eligible = False
    if run.get("exit_code") not in (None, 0) and not run.get("stop_reason"):
        score_eligible = False
    if quota_stop_reason(run):
        score_eligible = False
    audit_path = run_dir / "evaluation-audit.json"
    if audit_path.exists() and json.loads(audit_path.read_text()).get("valid") is False:
        score_eligible = False
    events = [json.loads(line) for line in retrieval.read_text().splitlines()] if retrieval.exists() else []
    searches = [event for event in events if event.get("action") == "search"]
    fetched = [event for event in events if event.get("action") == "fetch"]
    retrieved = {str(docid) for event in searches for docid in event.get("returned_docids", [])}
    recall = {}
    for label, rows in gold["qrels"].items():
        expected = {str(row["docid"]) for row in rows}
        recall[label] = {"retrieved": len(retrieved & expected), "total": len(expected)}
    return {
        "benchmark": "BrowseComp-Plus", "query_id": str(gold["query_id"]),
        "model": run["model"], "strategy": run["strategy"],
        "run_status": run["status"], "stop_reason": run.get("stop_reason"),
        "failure_kind": run.get("failure_kind"),
        "explicit_final_answer": answer is not None,
        "strict_answer_score": int(answer is not None and normalize(answer) == normalize(gold["answer"]))
                               if score_eligible else None,
        "inference_recorded": inference_recorded,
        "protocol_valid": run.get("protocol_valid"),
        "strategy_errors": run.get("strategy_errors", []),
        "official_judge_score": None,
        "score_note": "Exact-answer diagnostic only; official LLM judge not run.",
        "searches": len(searches), "fetches": len(fetched),
        "retrieval_errors": sum(event.get("status", "ok") != "ok" for event in searches + fetched),
        "recall": recall, "run_dir": str(run_dir), "retrieval_log": str(retrieval),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Refusing to overwrite an existing grade")
    result = grade(args.run_dir, args.retrieval, args.reference)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
