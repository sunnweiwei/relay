"""OSWorld driver contracts without Docker, VMs or model calls."""
import argparse
import csv
import json
import subprocess

import pytest

from evals import osworld


@pytest.mark.parametrize("arm", osworld.ARMS)
def test_only_relay_arms_give_relay_the_window(arm, tmp_path):
    strategy, window, agent, env, _ = osworld.arm_config(arm, 65536, 10, tmp_path)
    if arm.startswith("relay_"):
        assert (strategy, window, env["OSW_STRATEGY"]) == (arm.removeprefix("relay_"), 65536, strategy)
    else:
        assert (strategy, window) == ("compaction", osworld.NEVER)
    assert agent == osworld.FULL or arm == "native_fold"


def test_clm_runs_through_relay_only(tmp_path):
    with pytest.raises(ValueError):
        osworld.arm_config("native_clm", 65536, 10, tmp_path)


def result(out, value):
    path = out / "results/pyautogui/screenshot/m/tasks/003"
    path.mkdir(parents=True)
    (path / "result.txt").write_text(f"{value}\n")
    (path / "traj.jsonl").write_text("{}\n" * 7)


def test_a_trial_without_a_score_is_set_aside(tmp_path):
    scored, failed = tmp_path / "none/003", tmp_path / "none/004"
    result(scored, 0.5)
    failed.mkdir(parents=True)
    assert osworld.finished(scored) and not osworld.finished(failed)
    assert (tmp_path / "none/004.failed-0").is_dir()


def options_for(tmp_path):
    return argparse.Namespace(window=65536, image_max=10, model="m", base_url="http://up/v1", api_key="sk-test", steps=5,
                              osworld=tmp_path / "osworld", assets=tmp_path / "assets",
                              meta=tmp_path / "tasks.json", out=tmp_path / "out", arms=["relay_clm"])


def test_a_trial_runs_the_agent_in_osworld_and_cleans_up_when_it_fails(tmp_path, monkeypatch):
    calls, seen = [], {}

    class Process:
        def terminate(self):
            calls.append("relay stopped")

    def run(argv, **kwargs):
        if argv[:2] == ["docker", "run"]:
            calls.append("sandbox up")
        elif argv[:3] == ["docker", "rm", "-f"]:
            calls.append("sandbox removed")
        else:
            seen.update(argv=argv, **kwargs)
            raise RuntimeError("the agent failed")
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(osworld, "relay", lambda strategy, port, home, window, **extra: (
        calls.append(("relay", strategy, window)), Process())[1])
    monkeypatch.setattr(osworld.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="the agent failed"):
        osworld.run_trial("relay_clm", "004", 9700, options_for(tmp_path))
    assert calls == [("relay", "clm", 65536), "sandbox up", "relay stopped", "sandbox removed"]
    argv, env = seen["argv"], seen["env"]
    assert argv[1].endswith("evals/osworld_agent.py") and argv[argv.index("--specific_task_id") + 1] == "004"
    assert env["OPENAI_BASE_URL"] == f"http://{osworld.HOST}:9700/v1" and env["OSW_SANDBOX"] == "osw-sbx-9700"
    assert env["OPENAI_API_KEY"] == env["OSWORLD_EVAL_MODEL_API_KEY"] == env["OSWORLD_USER_SIM_API_KEY"] == "sk-test"


def test_a_resumed_run_keeps_its_settings(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    assert osworld.record(out, options_for(tmp_path), ["003"]) is None
    assert osworld.record(out, options_for(tmp_path), ["004"]) is None
    assert json.loads((out / "run.json").read_text())["tasks"] == ["003", "004"]
    changed = options_for(tmp_path)
    changed.window = 32768
    assert "window" in osworld.record(out, changed, ["003"])


def test_report_marks_the_wall_lists_missing_trials_and_writes_rows(tmp_path, capsys):
    (tmp_path / "run.json").write_text(json.dumps({"arms": ["none"], "tasks": ["003", "009"]}))
    result(tmp_path / "none/003", 0.5)
    (tmp_path / "none/003/method_log.json").write_text(json.dumps([{"kind": "budget_wall", "step": 7}]))
    osworld.report(tmp_path, tmp_path / "rows.csv")
    output = capsys.readouterr().out
    assert "| none | 0.50 (7 w) | — |" in output and "no score yet for none/009" in output
    with (tmp_path / "rows.csv").open() as handle:
        assert list(csv.DictReader(handle)) == [{"arm": "none", "task": "003", "score": "0.5", "steps": "7",
                                                 "walled": "True"}]
