#!/usr/bin/env python3
"""Isolated local ARC-AGI-3 game service for a single evaluation run.

Run this process in the SDK environment. The benchmark worker should only be
given its HTTP endpoint, not the SDK environment or game files.
"""

from __future__ import annotations

import argparse
from io import BytesIO
import json
import re
import threading
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

MAX_ACTIONS = 300
FROZEN_GAME_ID = "ls20-9607627b"
SEED = 0


def jsonable(value: Any) -> Any:
    """Convert public SDK values to JSON without introspecting game objects."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "name") and isinstance(value.name, str):
        return value.name
    if is_dataclass(value):
        return {key: jsonable(item) for key, item in asdict(value).items()}
    if hasattr(value, "model_dump"):
        return jsonable(value.model_dump(mode="json"))
    if hasattr(value, "to_dict"):
        return jsonable(value.to_dict())
    if isinstance(value, dict):
        return {
            str(key): jsonable(item)
            for key, item in value.items()
            if str(key).lower() != "api_key"
        }
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return jsonable(value.tolist())
    return str(value)


def frame_grid(frame: Any) -> list[list[int]] | None:
    """Return the latest visible grid from an SDK animation frame batch."""
    matrix = jsonable(frame)
    if (
        isinstance(matrix, list)
        and matrix
        and isinstance(matrix[0], list)
        and matrix[0]
        and isinstance(matrix[0][0], list)
    ):
        matrix = matrix[-1]
    if not isinstance(matrix, list) or not matrix or not all(
        isinstance(row, list) and row for row in matrix
    ):
        return None
    if len({len(row) for row in matrix}) != 1:
        return None
    if not all(
        isinstance(cell, int) and 0 <= cell <= 15
        for row in matrix
        for cell in row
    ):
        return None
    return matrix


def grid_ascii(frame: Any) -> str | None:
    """Encode a 16-color frame as one hexadecimal character per cell."""
    matrix = frame_grid(frame)
    if matrix is None:
        return None
    return "\n".join("".join(format(cell, "X") for cell in row) for row in matrix)


def observation_missing(observation: Any) -> bool:
    """Terminal frames may be absent; playable states need a usable observation."""
    if not isinstance(observation, dict):
        return True
    state = observation.get("state")
    if not isinstance(state, str) or state not in {"NOT_PLAYED", "NOT_FINISHED", "WIN", "GAME_OVER"}:
        return True
    return state in {"NOT_PLAYED", "NOT_FINISHED"} and not (
        isinstance(observation.get("frame_ascii"), str) and observation["frame_ascii"]
    )


class ArcRun:
    """One game instance with an immutable action budget and JSONL trace."""

    def __init__(self, game_id: str, seed: int, trace_path: Path, max_actions: int) -> None:
        if re.fullmatch(r"[a-z0-9]{4}-[a-f0-9]{8}", game_id) is None:
            raise ValueError("game_id must include an exact public game revision")
        # Imports live inside the service process, never in the benchmark worker.
        import arc_agi

        self._arc = arc_agi.Arcade(
            operation_mode=arc_agi.OperationMode.OFFLINE,
            recordings_dir=str(trace_path.parent / "sdk_recordings"),
        )
        self.game_id = game_id
        self.seed = seed
        self.max_actions = max_actions
        self.trace_path = trace_path
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self._trace = self.trace_path.open("a", encoding="utf-8")
        self.env: Any | None = None
        self.action_count = 0
        self.obs: Any | None = None
        self._start_game()

    def _start_game(self) -> None:
        self.env = self._arc.make(
            self.game_id,
            seed=self.seed,
            save_recording=True,
            include_frame_data=True,
        )
        if self.env is None:
            raise RuntimeError(f"could not load offline environment {self.game_id}")
        self.action_count = 1  # make() performs the initial environment reset.
        self.obs = self.env.observation_space
        self._record("reset", {"game_id": self.game_id, "seed": self.seed}, self.obs)

    def _record(self, kind: str, action: dict[str, Any], obs: Any) -> None:
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "kind": kind,
            "action_count": self.action_count,
            "action": action,
            "observation": self.observation(obs, raw_frame=True) if obs is not None else None,
        }
        self._trace.write(json.dumps(row, separators=(",", ":")) + "\n")
        self._trace.flush()

    def observation(self, obs: Any | None = None, *, raw_frame: bool = False) -> dict[str, Any]:
        current = self.obs if obs is None else obs
        frame = getattr(current, "frame", None)
        actions = getattr(self.env, "action_space", ())
        state = getattr(current, "state", None)
        result = {
            "game_id": self.game_id,
            "seed": self.seed,
            "action_count": self.action_count,
            "action_limit": self.max_actions,
            "state": jsonable(state),
            "levels_completed": jsonable(getattr(current, "levels_completed", None)),
            "available_actions": [jsonable(action) for action in actions],
            "frame_ascii": grid_ascii(frame),
        }
        if raw_frame:
            result["frame"] = jsonable(frame)
        return result

    def step(self, action: str, x: int | None, y: int | None) -> dict[str, Any]:
        if self.env is None:
            raise RuntimeError("reset must be called before step")
        available = {candidate.name: candidate for candidate in self.env.action_space}
        if action not in available:
            raise ValueError(f"action must be one of {', '.join(available)}")
        if self.action_count >= self.max_actions:
            raise RuntimeError("action limit reached")
        payload: dict[str, int] = {}
        if (x is None) != (y is None):
            raise ValueError("x and y must be supplied together")
        if available[action].is_complex() and x is None:
            raise ValueError(f"{action} requires x and y coordinates")
        if x is not None and y is not None:
            if not available[action].is_complex():
                raise ValueError(f"{action} does not accept coordinates")
            if not (0 <= x < 64 and 0 <= y < 64):
                raise ValueError("x and y must be in [0, 63]")
            payload = {"x": x, "y": y}
        sdk_action = available[action]
        self.action_count += 1
        next_observation = self.env.step(sdk_action, data=payload)
        if next_observation is None or observation_missing(self.observation(next_observation)):
            self._record("step_failed", {"action": action, **payload}, next_observation)
            raise RuntimeError("ARC SDK returned no usable observation; previous observation preserved")
        self.obs = next_observation
        self._record("step", {"action": action, **payload}, self.obs)
        return self.observation()

    def score(self) -> Any:
        if self.env is None:
            raise RuntimeError("reset must be called before score")
        card = self._arc.get_scorecard()
        return jsonable(card)


class Handler(BaseHTTPRequestHandler):
    run: ArcRun
    score_token: str
    lock = threading.RLock()

    def _send(self, code: int, body: Any) -> None:
        encoded = json.dumps(body, separators=(",", ":")).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_png(self, png: bytes) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(png)))
        self.end_headers()
        self.wfile.write(png)

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 4096:
            raise ValueError("request body too large")
        payload = json.loads(self.rfile.read(length) or b"{}")
        if not isinstance(payload, dict):
            raise ValueError("JSON body must be an object")
        return payload

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        parsed = urlparse(self.path)
        try:
            with self.lock:
                if parsed.path == "/frame.png":
                    from PIL import Image

                    grid = frame_grid(getattr(self.run.obs, "frame", None))
                    if grid is None:
                        self._send(404, {"error": "frame is not a 16-color grid"})
                        return
                    palette = [
                        (255, 255, 255), (204, 204, 204), (153, 153, 153),
                        (102, 102, 102), (51, 51, 51), (0, 0, 0),
                        (229, 58, 163), (255, 123, 204), (249, 60, 49),
                        (30, 147, 255), (136, 216, 241), (255, 220, 0),
                        (255, 133, 27), (146, 18, 49), (79, 204, 48),
                        (163, 86, 214),
                    ]
                    image = Image.new("RGB", (len(grid[0]), len(grid)))
                    image.putdata([palette[cell] for row in grid for cell in row])
                    buffer = BytesIO()
                    image.save(buffer, format="PNG")
                    self._send_png(buffer.getvalue())
                elif parsed.path == "/state":
                    self._send(200, self.run.observation())
                elif parsed.path == "/score":
                    token = self.headers.get("Authorization", "")
                    if token != f"Bearer {self.score_token}":
                        self._send(403, {"error": "forbidden"})
                        return
                    self._send(200, self.run.score())
                else:
                    self._send(404, {"error": "not found"})
        except (ValueError, RuntimeError) as exc:
            self._send(400, {"error": str(exc)})

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        if urlparse(self.path).path != "/step":
            self._send(404, {"error": "not found"})
            return
        try:
            payload = self._body()
            action = payload.get("action")
            x, y = payload.get("x"), payload.get("y")
            if not isinstance(action, str):
                raise ValueError("action is required")
            if x is not None and (not isinstance(x, int) or isinstance(x, bool)):
                raise ValueError("x must be an integer")
            if y is not None and (not isinstance(y, int) or isinstance(y, bool)):
                raise ValueError("y must be an integer")
            with self.lock:
                self._send(200, self.run.step(action, x, y))
        except (ValueError, RuntimeError, json.JSONDecodeError) as exc:
            self._send(400, {"error": str(exc)})

    def log_message(self, fmt: str, *args: Any) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-id", default=FROZEN_GAME_ID)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--trace", type=Path, default=Path("arc_run.jsonl"))
    parser.add_argument("--max-actions", type=int, default=MAX_ACTIONS)
    parser.add_argument("--score-token-file", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.max_actions <= MAX_ACTIONS:
        parser.error(f"--max-actions must be in [1, {MAX_ACTIONS}]")
    score_token = args.score_token_file.read_text(encoding="utf-8").strip()
    if len(score_token) < 24:
        parser.error("score token must contain at least 24 characters")
    run = ArcRun(args.game_id, args.seed, args.trace, args.max_actions)
    handler = type("RunHandler", (Handler,), {"run": run, "score_token": score_token})
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"ARC service ready on http://{args.host}:{args.port} game={args.game_id}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
