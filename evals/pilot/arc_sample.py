#!/usr/bin/env python3
"""Run the frozen public-game sample serially through the ARC matrix suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evals.pilot.methods import EVALUATED_METHODS

from evals.pilot.arc_suite import PORT, write_json


COORDINATE_EXAMPLE_V1 = '{"action":"ACTION2","x":12,"y":24}'
COORDINATE_EXAMPLE_V2 = '{"action":"ACTION6","x":12,"y":24}'


def load_sample(manifest_path: Path, preflight_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    ranked = manifest["ranked_ids"]
    selected = manifest["selected_ids"]
    expected = sorted(ranked, key=lambda game: hashlib.sha256(
        f"{manifest['seed']}:{game}".encode()).hexdigest())
    if (len(ranked) != len(set(ranked)) or len(ranked) != manifest["population_size"]
            or ranked != expected or not selected or selected != ranked[:len(selected)]):
        raise ValueError("ARC selection is not the frozen deterministic population prefix")
    preflight = json.loads(preflight_path.read_text())
    if (not preflight.get("passed")
            or preflight.get("manifest_sha256") != hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            or preflight.get("selected_ids") != selected):
        raise ValueError("ARC runtime preflight does not match the frozen sample")
    for item in preflight["files"]:
        path = Path(item["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"ARC runtime file changed after preflight: {path}")
    return manifest


def validate_game_prompt(game_root: Path, prompt: Path, port: int = PORT) -> str:
    """A started game keeps its exact prompt for every method and resume."""
    digest = hashlib.sha256(prompt.read_bytes()).hexdigest()
    effective = prompt.read_text().replace(f"http://127.0.0.1:{PORT}", f"http://127.0.0.1:{port}")
    arm_digest = hashlib.sha256(effective.encode()).hexdigest()
    for name in ("prompt-policy.json", "prompt-freeze.json"):
        record = game_root / name
        if record.exists() and json.loads(record.read_text()).get("sha256") != digest:
            raise ValueError(f"ARC game prompt changed after selection ({name}): {game_root.name}")
    existing = list(game_root.glob("gpt-*/*/prompt.txt"))
    if any(hashlib.sha256(path.read_bytes()).hexdigest() != arm_digest for path in existing):
        raise ValueError(f"ARC game prompt differs from an existing attempt: {game_root.name}")
    for arm in game_root.glob("gpt-*/*"):
        if arm.is_dir() and any(arm.iterdir()) and not (arm / "prompt.txt").is_file():
            raise ValueError(f"ARC started game has no retained prompt evidence: {game_root.name}")
    return digest


def load_game_prompts(
    manifest: dict[str, Any], preflight_path: Path, runtime_root: Path,
    out_root: Path, policy_path: Path | None, port: int = PORT,
) -> dict[str, Path]:
    baseline = (runtime_root / "frozen_prompt.txt").resolve()
    prompts = {game_id: baseline for game_id in manifest["selected_ids"]}
    if policy_path is not None:
        policy_path = policy_path.resolve()
        policy = json.loads(policy_path.read_text())
        baseline_bytes = baseline.read_bytes()
        baseline_hash = hashlib.sha256(baseline_bytes).hexdigest()
        if (policy.get("version") != 1 or policy.get("base_prompt_sha256") != baseline_hash
                or set(policy.get("games", {})) != set(prompts)
                or baseline_bytes.count(COORDINATE_EXAMPLE_V1.encode()) != 1):
            raise ValueError("ARC prompt policy does not match the frozen sample and baseline")
        corrected = baseline_bytes.replace(COORDINATE_EXAMPLE_V1.encode(), COORDINATE_EXAMPLE_V2.encode())
        pins = {Path(item["path"]).resolve(): item["sha256"]
                for item in json.loads(preflight_path.read_text())["files"]}
        if pins.get(policy_path) != hashlib.sha256(policy_path.read_bytes()).hexdigest():
            raise ValueError("ARC prompt policy is not pinned by preflight")
        for game_id, item in policy["games"].items():
            prompt = Path(item["path"])
            if not prompt.is_absolute():
                raise ValueError("ARC prompt policy paths must be absolute")
            prompt = prompt.resolve()
            digest = hashlib.sha256(prompt.read_bytes()).hexdigest()
            if (digest != item["sha256"] or pins.get(prompt) != digest
                    or prompt.read_bytes() not in {baseline_bytes, corrected}):
                raise ValueError(f"ARC prompt is unpinned or changes more than the example: {game_id}")
            prompts[game_id] = prompt
    for game_id, prompt in prompts.items():
        validate_game_prompt(out_root / game_id, prompt, port)
    return prompts


def freeze_game_prompt(
    game_root: Path, prompt: Path, policy_path: Path | None, port: int = PORT,
) -> None:
    digest = validate_game_prompt(game_root, prompt, port)
    game_root.mkdir(parents=True, exist_ok=True)
    record = game_root / "prompt-policy.json"
    if not record.exists():
        write_json(record, {"version": 1, "path": str(prompt.resolve()), "sha256": digest,
                            "policy_path": str(policy_path.resolve()) if policy_path else None})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--prompt-policy", type=Path)
    parser.add_argument("--port", type=int, default=PORT)
    args = parser.parse_args()
    manifest = load_sample(args.manifest, args.preflight)
    out = args.out_root.resolve()
    out.mkdir(parents=True, exist_ok=True)
    prompts = load_game_prompts(manifest, args.preflight, args.runtime_root, out, args.prompt_policy, args.port)
    status: dict[str, Any] = {
        "benchmark": "ARC-AGI-3", "selected_ids": manifest["selected_ids"],
        "manifest": str(args.manifest.resolve()), "state": "running",
        "completed_game_matrices": [], "maximum_concurrent_workers": 1,
        "unknown_cells": [], "scored_cells": 0,
    }

    def record() -> None:
        status["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(out / "sample-status.json", status)

    for game_id in manifest["selected_ids"]:
        freeze_game_prompt(out / game_id, prompts[game_id], args.prompt_policy, args.port)
        command = [
            str(args.python.absolute()), "-m", "evals.pilot.arc_suite",
            "--scratch", str(args.scratch.resolve()), "--python", str(args.python.absolute()),
            "--arc-cwd", str(args.runtime_root.resolve()),
            "--prompt", str(prompts[game_id]),
            "--game-id", game_id, "--seed", str(manifest["gameplay_seed"]),
            "--out-root", str(out / game_id), "--port", str(args.port),
            "--models", "gpt-6.1-sol", "gpt-6-luna", "--arms", *EVALUATED_METHODS,
            "--resume",
        ]
        status.update(current_game_id=game_id, command=command)
        record()
        result = subprocess.run(command, check=False)
        if result.returncode:
            status.update(state="stopped", returncode=result.returncode)
            record()
            return result.returncode
        for model in ("gpt-6.1-sol", "gpt-6-luna"):
            for arm in EVALUATED_METHODS:
                path = out / game_id / model / arm / "summary.json"
                if not path.is_file():
                    status.update(state="stopped", returncode=2, reason="missing_summary",
                                  missing_summary=str(path))
                    record()
                    return 2
                summary = json.loads(path.read_text())
                audit_path = path.parent / "evaluation-audit.json"
                if audit_path.is_file():
                    audit = json.loads(audit_path.read_text())
                    if audit.get("valid") is False:
                        # Reviewed exclusions preserve the original summary as evidence.
                        summary = {**summary, "scored": False, "error": audit["reason"]}
                if summary.get("scored"):
                    status["scored_cells"] += 1
                else:
                    status["unknown_cells"].append({"game_id": game_id, "model": model,
                        "method": arm, "reason": summary.get("error") or summary.get("worker_status")})
        status["completed_game_matrices"].append(game_id)
    status.update(state="completed_with_unknown" if status["unknown_cells"] else "completed",
                  current_game_id=None, returncode=0)
    record()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
