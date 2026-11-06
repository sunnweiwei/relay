"""Invalid ARC actions must not destroy the last playable observation."""

from __future__ import annotations

import io
import json
import threading
from dataclasses import dataclass
from http.server import ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import Mock
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from evals.pilot.arc_server import ArcRun, Handler


@dataclass
class Action:
    name: str
    complex: bool = False

    def is_complex(self) -> bool:
        return self.complex


@pytest.fixture
def run() -> ArcRun:
    instance = ArcRun.__new__(ArcRun)
    instance.game_id, instance.seed, instance.max_actions = "bp35-0a0ad940", 0, 300
    instance.action_count = 1
    instance.obs = SimpleNamespace(state="NOT_FINISHED", levels_completed=0, frame=[[[1, 2]]])
    instance.env = SimpleNamespace(action_space=[Action("ACTION3"), Action("ACTION6", True)],
                                   step=Mock(return_value=instance.obs))
    instance._trace = io.StringIO()
    return instance


def test_missing_click_coordinates_are_rejected_before_sdk_and_budget(run: ArcRun) -> None:
    original = run.observation()
    with pytest.raises(ValueError, match="ACTION6 requires x and y"):
        run.step("ACTION6", None, None)
    run.env.step.assert_not_called()
    assert run.action_count == 1
    assert run.observation() == original
    assert run._trace.getvalue() == ""
    run.step("ACTION6", 12, 24)
    assert run.env.step.call_args.kwargs == {"data": {"x": 12, "y": 24}}
    assert run.action_count == 2


@pytest.mark.parametrize("returned", [
    None,
    SimpleNamespace(state=None, levels_completed=None, frame=None),
    SimpleNamespace(state="NOT_FINISHED", levels_completed=0, frame=[]),
])
def test_failed_sdk_observation_is_explicit_and_does_not_poison_state(run: ArcRun, returned: object) -> None:
    original = run.obs
    run.env.step.return_value = returned
    with pytest.raises(RuntimeError, match="previous observation preserved"):
        run.step("ACTION3", None, None)
    assert run.obs is original
    assert run.observation()["state"] == "NOT_FINISHED"
    assert run.observation()["frame_ascii"] == "12"
    # A failed SDK call can have side effects, so its attempted action remains counted.
    assert run.action_count == 2
    event = json.loads(run._trace.getvalue())
    assert event["kind"] == "step_failed"
    if returned is None:
        assert event["observation"] is None


def test_terminal_state_can_have_no_frame(run: ArcRun) -> None:
    run.env.step.return_value = SimpleNamespace(state="WIN", levels_completed=9, frame=[])
    observation = run.step("ACTION3", None, None)
    assert observation["state"] == "WIN"
    assert observation["levels_completed"] == 9
    assert observation["frame_ascii"] is None


def test_http_invalid_click_returns_error_and_preserves_state_and_png(run: ArcRun) -> None:
    pytest.importorskip("PIL")  # (the `eval` extra renders frames as PNG)
    handler = type("TestHandler", (Handler,), {"run": run, "score_token": "unused"})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        request = Request(base + "/step", data=b'{"action":"ACTION6"}',
                          headers={"Content-Type": "application/json"})
        with pytest.raises(HTTPError) as error:
            urlopen(request, timeout=2)
        assert error.value.code == 400
        assert "requires x and y" in json.load(error.value)["error"]
        with urlopen(base + "/state", timeout=2) as response:
            observation = json.load(response)
        assert observation["state"] == "NOT_FINISHED"
        assert observation["action_count"] == 1
        assert observation["frame_ascii"] == "12"
        with urlopen(base + "/frame.png", timeout=2) as response:
            assert response.headers["Content-Type"] == "image/png"
            assert response.read().startswith(b"\x89PNG\r\n\x1a\n")
        run.env.step.assert_not_called()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
