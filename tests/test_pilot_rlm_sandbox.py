"""Check the official RLM subprocess protocol without any model inference."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from evals.pilot import rlm_sandbox


def test_sandbox_environment_excludes_credentials(tmp_path: Path) -> None:
    clean = rlm_sandbox.clean_environment({
        "PATH": "/usr/bin:/private/.codex-hosts/host/tmp/arg0-helper:/bin",
        "HOME": "/home/test", "CODEX_HOME": "/private/codex",
        "OPENAI_API_KEY": "do-not-forward", "HF_TOKEN": "do-not-forward",
        "RELAY_HOME": "/private/relay", "ARBITRARY_SECRET": "do-not-forward",
    }, tmp_path)
    assert clean["PATH"] == "/usr/bin:/bin"
    assert "do-not-forward" not in clean.values()
    assert "RELAY_HOME" not in clean


def subprocess_repl(tmp_path: Path, **kwargs: object) -> rlm_sandbox.SandboxREPL:
    pytest.importorskip("rlm")
    script = Path(rlm_sandbox.__file__).absolute()
    return rlm_sandbox.SandboxREPL(
        [sys.executable, str(script), "--worker"],
        rlm_sandbox.clean_environment(dict(os.environ), tmp_path),
        tmp_path, timeout=15, **kwargs,
    )


def test_official_repl_preserves_namespace_and_context_versions(tmp_path: Path) -> None:
    repl = subprocess_repl(tmp_path, context_payload=[{"content": "first"}])
    try:
        first = repl.execute_code("values = [41]\nprint(context[0]['content'])")
        assert first.stdout.strip() == "first"
        assert repl.add_context([{"content": "second"}]) == 1
        assert repl.get_context_count() == 2
        assert repl.add_history([{"role": "user", "content": "question"}]) == 0
        assert repl.get_history_count() == 1
        result = repl.execute_code(
            "values.append(1)\n"
            "print(context_1[0]['content'])\n"
            "answer['content'] = str(sum(values))\nanswer['ready'] = True")
        assert result.stdout.strip() == "second"
        assert result.final_answer == "42"
        assert result.locals["values"] == [41, 1]
    finally:
        repl.cleanup()
    assert repl.process.poll() is not None


def test_subcalls_use_official_handler_and_preserve_call_records(tmp_path: Path) -> None:
    pytest.importorskip("rlm")
    from rlm.core.lm_handler import LMHandler
    from rlm.core.types import ModelUsageSummary

    class Model:
        model_name = "local-test-stub"

        def completion(self, prompt: str) -> str:
            return f"stub:{prompt}"

        def get_last_usage(self) -> ModelUsageSummary:
            return ModelUsageSummary(1, 3, 5)

    handler = LMHandler(Model())
    handler.start()
    repl = subprocess_repl(tmp_path)
    try:
        repl.update_handler_address((handler.host, handler.port))
        result = repl.execute_code("print(llm_query('hello'))")
        assert result.stdout.strip() == "stub:hello"
        assert len(result.rlm_calls) == 1
        assert result.rlm_calls[0].response == "stub:hello"
        assert result.rlm_calls[0].root_model == "local-test-stub"
    finally:
        repl.cleanup()
        handler.stop()


def test_nonterminating_generated_code_is_killed(tmp_path: Path) -> None:
    repl = subprocess_repl(tmp_path)
    repl.timeout = 0.1
    try:
        with pytest.raises(TimeoutError, match="deadline"):
            repl.execute_code("while True: pass")
        assert repl.process.poll() is not None
    finally:
        repl.cleanup()


def test_official_rlm_completion_reuses_sandboxed_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("rlm")
    import rlm.core.rlm as core
    from rlm.core.types import ModelUsageSummary, UsageSummary

    class Model:
        model_name = "local-test-stub"
        calls = 0

        def completion(self, prompt: object) -> str:
            self.calls += 1
            code = "remember = 41\n" if self.calls == 1 else "remember += 1\n"
            return "```repl\n" + code + "answer['content'] = str(remember)\nanswer['ready'] = True\n```"

        def get_last_usage(self) -> ModelUsageSummary:
            return ModelUsageSummary(1, 3, 5)

        def get_usage_summary(self) -> UsageSummary:
            return UsageSummary({self.model_name: self.get_last_usage()})

    model = Model()
    monkeypatch.setattr(core, "get_client", lambda *a: model)
    monkeypatch.setattr(core, "get_environment", lambda name, kwargs: subprocess_repl(tmp_path, **kwargs))
    runtime = core.RLM(backend="stub", backend_kwargs={}, environment="local", persistent=True, max_iterations=2)
    try:
        assert runtime.completion([{"content": "first"}]).response == "41"
        assert runtime.completion([{"content": "second"}]).response == "42"
        assert runtime._persistent_env.get_context_count() == 2
        assert model.calls == 2
    finally:
        runtime.close()
