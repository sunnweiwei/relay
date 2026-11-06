"""Collect completed or live pilot usage, including Relay summarizer requests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator


FIELDS = ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_output_tokens", "total_tokens")


def records(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open() as stream:
        for line in stream:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                # A live JSONL file can have one incomplete tail record.
                continue
            if isinstance(value, dict):
                yield value


def normalized(usage: dict[str, Any]) -> dict[str, int]:
    input_count = int(usage.get("input_tokens", 0))
    output_count = int(usage.get("output_tokens", 0))
    return {
        "input_tokens": input_count,
        "cached_input_tokens": int(usage.get("cached_input_tokens", (usage.get("input_tokens_details") or {}).get("cached_tokens", 0))),
        "output_tokens": output_count,
        "reasoning_output_tokens": int(usage.get("reasoning_output_tokens", (usage.get("output_tokens_details") or {}).get("reasoning_tokens", 0))),
        "total_tokens": int(usage.get("total_tokens", input_count + output_count)),
    }


def collect(path: Path) -> dict[str, Any]:
    result = json.loads((path / "run.json").read_text())
    latest: dict[str, Any] | None = None
    model_slugs: set[str] = set()
    for rollout in sorted((path / "codex/sessions").glob("**/rollout-*.jsonl")):
        for event in records(rollout):
            payload = event.get("payload") or {}
            if payload.get("type") == "token_count" and payload.get("info"):
                latest = payload["info"].get("total_token_usage") or latest
    summary_usage: list[dict[str, int]] = []
    summary_attempts = 0
    unknown_summary_usage = 0
    sizes: list[int] = []
    http_errors: list[int] = []
    auxiliary_inputs: list[int] = []
    for event in records(path / "relay/trace.jsonl"):
        if isinstance(event.get("body"), dict):
            model_slugs.add(str(event["body"].get("model")))
        if event.get("usage") is not None:
            sizes.append(int(event["usage"]))
        if "summary_request" in event:
            summary_attempts += 1
            answer = event.get("answer")
            usage = answer.get("usage") if isinstance(answer, dict) else None
            if isinstance(usage, dict):
                summary_usage.append(normalized(usage))
                auxiliary_inputs.append(int(usage.get("input_tokens", 0)))
            else:
                unknown_summary_usage += 1
            if isinstance(event.get("summary_request"), dict):
                model_slugs.add(str(event["summary_request"].get("model")))
        status = event.get("summary_status", event.get("status"))
        if isinstance(status, int) and status >= 400:
            http_errors.append(status)
    worker = normalized(latest) if latest else None
    summary = {key: sum(row[key] for row in summary_usage) for key in FIELDS}
    combined = {key: worker[key] + summary[key] for key in FIELDS} if worker else None
    events = list(records(path / "relay/events.jsonl"))
    quota = list(records(path / "quota.jsonl"))
    upstream_attempts = sum(event.get("event") == "request" and str(event.get("path", "")).endswith("/responses")
                            for event in records(path / "upstream-events.jsonl"))
    # Interrupted auxiliary streams never reach the completed summary trace.
    # Keep those dispatched calls unknown instead of silently treating them as free.
    unaccounted_upstream = max(0, upstream_attempts - len(sizes) - len(summary_usage))
    return {**result, "directory": str(path),
            "worker_usage": worker, "summary_usage": summary,
            "combined_usage": combined,
            "summary_attempts": summary_attempts,
            "summary_usage_missing_attempts": unknown_summary_usage,
            "upstream_attempts": upstream_attempts,
            "unaccounted_upstream_attempts": unaccounted_upstream,
            "confirmed_response_usage_records": len(sizes),
            "peak_provider_input_tokens": max(sizes, default=0),
            "requests_over_context_budget": sum(n > result["window"] for n in sizes),
            "context_budget_scope": result.get("context_budget_scope", "main_agent_requests"),
            "auxiliary_peak_input_tokens": max(auxiliary_inputs, default=0),
            "auxiliary_requests_above_main_window": sum(n > result["window"] for n in auxiliary_inputs),
            "model_slugs_observed": sorted(model_slugs),
            "context_event_records": len(events),
            "context_event_types": sorted({str(e.get("type", e.get("strategy", e.get("event", "unspecified")))) for e in events}),
            "http_error_statuses": http_errors,
            "all_saved_credit_checks_unchanged": all(q.get("credit_balance_unchanged") for q in quota) if quota else None,
            "last_quota_used_percent": quota[-1].get("used_percent") if quota else None,
            "usage_caveat": "Cached input and reasoning are subsets. Combined includes known Relay summary usage; failed/in-flight requests without usage remain unknown."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, nargs="+", help="One or more scratch run directories")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    paths = sorted({path for root in args.root for path in root.glob("**/run.json")})
    rows = [collect(path.parent) for path in paths]
    text = json.dumps(rows, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.write_text(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
