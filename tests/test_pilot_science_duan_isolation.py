from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from evals.pilot import science_duan_isolation as duan


def test_bundle_wrapper_is_scoped_to_exact_task_and_entrypoint(tmp_path: Path) -> None:
    wrapper = duan.install_bundle(tmp_path)
    text = wrapper.read_text()
    assert wrapper.stat().st_mode & 0o111
    assert duan.TASK_ID in text
    assert "${1:-}" in text and "/tests/test_state.R" in text
    assert "R_PROFILE_USER=/opt/relay-isolation/science_duan_profile.R" in text
    assert 'exec /usr/local/bin/Rscript "$@"' in text
    assert (tmp_path / duan.PROFILE_NAME).is_file()
    assert (tmp_path / duan.WORKER_NAME).is_file()


def test_worker_boundary_is_text_ipc_and_masks_verifier_paths(tmp_path: Path) -> None:
    duan.install_bundle(tmp_path)
    profile = (tmp_path / duan.PROFILE_NAME).read_text()
    worker = (tmp_path / duan.WORKER_NAME).read_text()
    assert '"/results/Duan_models.R"' in profile
    assert '"TWOPAR"' in profile and '"SIXPAR"' in profile
    assert '"--unshare-net"' in profile
    assert '"--tmpfs", "/tests"' in profile
    assert '"--tmpfs", "/verifier"' in profile
    assert '"--tmpfs", "/logs"' in profile
    assert '"--ro-bind", "/results/Duan_models.R", "/results/Duan_models.R"' in profile
    assert '"--bind", "/results", "/results"' in profile
    assert "readRDS" not in profile + worker
    assert "eval(" not in profile + worker
    assert '"ERROR"' in profile and '"OK"' in worker


def test_generated_r_sources_parse_with_available_rscript(tmp_path: Path) -> None:
    rscript = shutil.which("Rscript")
    if not rscript:
        return
    duan.install_bundle(tmp_path)
    for name in (duan.PROFILE_NAME, duan.WORKER_NAME):
        path = tmp_path / name
        check = subprocess.run(
            [rscript, "--vanilla", "-e", "parse(file=commandArgs(TRUE)[1])", str(path)],
            check=False, capture_output=True, text=True, timeout=15,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        assert check.returncode == 0, check.stderr

