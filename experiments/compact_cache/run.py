"""Run one Harness Compact/Cache smoke or bounded live case."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from experiments.compact_cache.checks import HARNESSES

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harness", required=True, choices=HARNESSES)
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=("smoke", "live"), default="smoke")
    parser.add_argument("--case", choices=("marker", "local-files"), default="marker")
    parser.add_argument("--codex-reasoning-diagnostic", action="store_true",
                        help="capture raw Codex reasoning items for the local-files live case")
    args = parser.parse_args(argv)
    if args.codex_reasoning_diagnostic and not (
        args.mode == "live" and args.case == "local-files" and args.harness == "codex"
    ):
        parser.error("--codex-reasoning-diagnostic requires live local-files codex")
    if args.mode == "live" and args.harness == "cursor":
        parser.error("cursor: no verified model-provider route; no API request was sent")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_model = re.sub(r"[^a-zA-Z0-9_.-]+", "-", args.model)
    suffix = f"-{args.case}" if args.case != "marker" else ""
    output = ROOT / "runs" / f"{stamp}-{args.harness}-{safe_model}-{args.mode}{suffix}"
    output.mkdir(parents=True, exist_ok=False)
    if args.mode == "live":
        try:
            if args.case == "local-files":
                from experiments.compact_cache.live_local_files import run
                record = run(
                    args.harness, args.model,
                    reasoning_capture_path=(output / "codex_reasoning_raw.json"
                                            if args.codex_reasoning_diagnostic else None),
                )
            elif args.harness in {"codex", "opencode", "pi", "mini-swe"}:
                from experiments.compact_cache.live_responses import run
                record = run(args.harness, args.model)
            elif args.harness == "gemini-cli":
                from experiments.compact_cache.live_gemini import run
                record = run(args.model)
            else:
                from experiments.compact_cache.live_claude import run
                record = run(args.model)
        except Exception as exc:
            body = getattr(exc, "body", None)
            detail = body.get("error", body) if isinstance(body, dict) else None
            message = detail.get("message") if isinstance(detail, dict) else None
            record = {"harness": args.harness, "model": args.model,
                      "mode": "live", "status": "error",
                      "error_type": type(exc).__name__,
                      "http_status": getattr(exc, "status_code", None),
                      "api_error_message": message if isinstance(message, str) else None}
        (output / "result.json").write_text(json.dumps(record, indent=2) + "\n")
        fields = ("status", "harness", "model", "live_protocol_pass",
                  "compaction_pass", "cache_pass", "answer_pass",
                  "first_turn_tool_pair", "exact_second_turn_checkpoint",
                  "task_requests", "summary_calls", "cache_hits",
                  "relay_requests", "task_model_requests", "threshold_calibration",
                  "budgeted_task_input_tokens", "budgeted_summary_input_tokens",
                  "error_type", "http_status", "api_error_message")
        print(json.dumps({key: record[key] for key in fields if key in record}
                         | {"output": str(output)}))
        return 0 if record["status"] == "pass" else 1
    if args.case != "marker":
        parser.error("local-files is a real-API live case; use --mode live")
    node = f"experiments/compact_cache/test_smoke.py::test_smoke[{args.harness}]"
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-rs", node],
        cwd=REPO, capture_output=True, text=True, timeout=120,
    )
    (output / "pytest.txt").write_text(result.stdout + result.stderr)
    status = ("passed" if result.returncode == 0 and "1 passed" in result.stdout
              else "unavailable" if "1 skipped" in result.stdout else "failed")
    record = {"harness": args.harness, "candidate_model": args.model,
              "strategy": "compact", "checkpoint_mode": "cache",
              "mode": "smoke", "real_api_used": False, "status": status,
              "pytest_exit_code": result.returncode}
    (output / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({**record, "output": str(output)}))
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
