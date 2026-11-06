"""Sample means must preserve missingness and pair the same tasks."""

from evals.pilot.sample_report import aggregate


def test_model_difference_uses_matched_valid_tasks_only() -> None:
    def row(model: str, task: str, score: float | None, status: str = "scored") -> dict:
        return {"model": model, "strategy": "compaction", "task_id": task,
                "score": score, "status": status, "usage": {}}

    rows = [row("gpt-6-luna", "a", 0), row("gpt-6-luna", "b", 1),
            row("gpt-6.1-sol", "a", 1), row("gpt-6.1-sol", "b", 0, "infrastructure_failed"),
            row("gpt-6.1-sol", "c", None, "timeout")]
    result = aggregate(rows, ["a", "b", "c"])
    group = result["groups"][0]
    assert group["models"]["gpt-6-luna"]["mean_score"] == 0.5
    assert group["models"]["gpt-6.1-sol"]["mean_score"] == 1
    assert group["models"]["gpt-6.1-sol"]["n"] == 1
    assert group["paired_tasks"] == ["a"]
    assert group["paired_sol_minus_luna"] == 1
    assert result["all_methods_common_tasks"] == []


def test_unstarted_methods_have_no_score_or_paired_difference() -> None:
    result = aggregate([], ["a", "b"])
    assert len(result["groups"]) == 11
    assert all(group["paired_sol_minus_luna"] is None for group in result["groups"])
    assert all(group["models"]["gpt-6-luna"]["mean_score"] is None for group in result["groups"])
