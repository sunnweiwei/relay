"""Run local benchmark-interface and strategy regressions without model calls.

Use the evaluation Python environment: python -m evals.pilot.preflight
These tests use scripted model responses and temporary local HTTP services.
Passing them does not certify dataset readiness, live-provider behavior, or
parity with the CLM paper. This command never starts a benchmark worker.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


TESTS = (
    "tests/test_pilot.py",
    "tests/test_pilot_arc_frames.py",
    "tests/test_pilot_arc_service.py",
    "tests/test_pilot_arc_suite.py",
    "tests/test_pilot_browse.py",
    "tests/test_pilot_science_container.py",
    "tests/test_pilot_science_suite.py",
    "tests/test_pilot_science_quota.py",
    "tests/test_compaction.py",
    "tests/test_folding.py",
    "tests/test_baselines.py::SelfCompactTests",
    "tests/test_baselines.py::ACMTests",
    "tests/test_clm.py",
    "tests/test_rlm.py",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="Print tests without executing them")
    args = parser.parse_args()
    if args.list:
        print("\n".join(TESTS))
        return 0
    repo = Path(__file__).resolve().parents[2]
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    print("Local preflight: scripted responses only; no benchmark inference.", flush=True)
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *TESTS],
        cwd=repo, env=env, check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
