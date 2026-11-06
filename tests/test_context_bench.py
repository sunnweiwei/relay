"""ContextBench (the CLM paper's) driver contracts without Harbor, Docker or model calls."""
import argparse
import csv
import json

import pytest

from evals import context_bench, osworld


def write_result(out, error=None, verified=True):
    path = out / "jobs/job/trial__x/result.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"exception_info": {"exception_type": error} if error else None,
                                "verifier_result": {"rewards": {"reward": 1.0}} if verified else None}))


def test_an_upstream_error_is_not_an_end(tmp_path):
    ended, timed_out, limited = tmp_path / "a/t1", tmp_path / "a/t2", tmp_path / "a/t3"
    write_result(ended)
    write_result(timed_out, "AgentTimeoutError", verified=False)
    write_result(limited, "InternalServerError")
    assert context_bench.finished(ended) and context_bench.finished(timed_out)
    assert not context_bench.finished(limited) and (tmp_path / "a/t3.failed-0").is_dir()


def test_osworld_runs_the_paper_modes_section_on_its_own():
    fires, manage = osworld.paper_modes("fires", "MANAGE_TOOL")
    assert fires("C1: Y\nC2: Y\nC3 Y\nN1: N") and not fires("C1: Y\nC2: Y\nC3: Y\nN1: Y")
    assert manage["function"]["name"] == "manage_context"


def options_for(tmp_path):
    task = tmp_path / "cb/tasks/kv-store-R130-seed42"
    task.mkdir(parents=True)
    (task / "task.toml").write_text('[metadata]\ncontextbench_family = "kv_store"\n')
    (tmp_path / "cb/clm_exp/skills/acm").mkdir(parents=True)
    (tmp_path / "cb/clm_exp/skills/acm/kv_store.md").write_text("Call manage_context when the gauge passes 22,000.\n")
    return argparse.Namespace(out=tmp_path / "out", tasks_dir=tmp_path / "cb/tasks", skills=tmp_path / "cb/clm_exp/skills",
                              codex=None, oracle_harbor="/venv/bin/harbor", harbor="harbor", model="m",
                              base_url="http://up/v1", api_key="sk-test", arms=["chat_acm"])


def test_a_chat_trial_runs_harbor_on_relay_and_stops_relay_when_harbor_fails(tmp_path, monkeypatch):
    calls, seen = [], {}

    class Process:
        def terminate(self):
            calls.append("relay stopped")

    def run(command, **kwargs):
        seen.update(command=command, **kwargs)
        raise RuntimeError("harbor failed")

    monkeypatch.setattr(context_bench, "relay", lambda strategy, port, home, window, **extra: (
        calls.append(("relay", strategy, window, extra["openai_base_url"])), Process())[1])
    monkeypatch.setattr(context_bench.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="harbor failed"):
        context_bench.run_trial("chat_acm", "kv-store-R130-seed42", 9100, options_for(tmp_path), None)
    assert calls == [("relay", "acm", context_bench.WINDOW + context_bench.CHAT_OVERHEAD, "http://up/v1"), "relay stopped"]
    command = seen["command"]
    assert command[command.index("-a") + 1] == "context_bench_agent:RelayChatAgent"
    kwargs = dict(command[i + 1].split("=", 1) for i, c in enumerate(command) if c == "--ak")
    assert kwargs["api_base"] == f"http://{context_bench.HOST}:9100/v1" and seen["env"]["OPENAI_API_KEY"] == "sk-test"
    skill = (tmp_path / "out/chat_acm/kv-store-R130-seed42/skill.md").read_text()
    assert skill.startswith(context_bench.CHAT_NOTES["acm"]) and skill.endswith("passes 22,000.\n")


def test_a_resumed_run_keeps_its_model(tmp_path):
    options = options_for(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    assert context_bench.record(out, options, ["a"]) is None
    options.model = "other"
    assert "model" in context_bench.record(out, options, ["a"])


def test_report_lists_trials_without_a_result_and_writes_rows(tmp_path, capsys):
    (tmp_path / "run.json").write_text(json.dumps({"arms": ["chat_acm"], "tasks": ["kv-store-R130-seed42",
                                                                                   "kv-store-R130-seed43"]}))
    write_result(tmp_path / "chat_acm/kv-store-R130-seed42")
    context_bench.report(tmp_path, tmp_path / "rows.csv")
    assert "1 trials without a result: chat_acm/kv-store-R130-seed43" in capsys.readouterr().out
    with (tmp_path / "rows.csv").open() as handle:
        assert [r["reward"] for r in csv.DictReader(handle)] == ["1.0"]


def test_the_agents_load_where_harbor_runs_them():
    pytest.importorskip("contextbench")
    from evals import context_bench_agent as agents
    assert issubclass(agents.RelayChatAgent, agents.PrivateSnapshots)
    assert osworld.paper_modes("SC_RUBRIC") == (agents.SC_RUBRIC,)
