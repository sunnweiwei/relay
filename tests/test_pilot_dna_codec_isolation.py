"""Model-free checks for the DNA codec candidate boundary."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from evals.pilot import dna_codec_isolation as dna
from evals.pilot import science_isolation


@pytest.fixture
def app_root(tmp_path: Path) -> Path:
    app = tmp_path / "app"
    app.mkdir()
    (app / "encoder.py").write_text(
        "from pathlib import Path\n"
        "assert Path('/app/data.txt').read_bytes() == b'input-data'\n"
        "assert not Path('/app/sequenced.dat').exists()\n"
        "assert not Path('/tests/private').exists()\n"
        "assert not Path('/logs/verifier/reward.txt').exists()\n"
        f"assert not Path({str(tmp_path / 'private-verifier-file')!r}).exists()\n"
        f"assert not Path({str(Path('/proc/1/root') / tmp_path.relative_to('/') / 'private-verifier-file')!r}).exists()\n"
        "Path('/app/original_sequences.dat').write_bytes(b'encoded')\n"
        "Path('/app/encoder-scratch').write_text('temporary')\n"
    )
    (app / "decoder.py").write_text(
        "from pathlib import Path\n"
        "assert Path('/app/sequenced.dat').read_bytes() == b'lossy-input'\n"
        "assert not Path('/app/data.txt').exists()\n"
        "assert not Path('/app/original_sequences.dat').exists()\n"
        "assert not Path('/tests/private').exists()\n"
        f"assert not Path({str(tmp_path / 'private-verifier-file')!r}).exists()\n"
        f"assert not Path({str(Path('/proc/1/root') / tmp_path.relative_to('/') / 'private-verifier-file')!r}).exists()\n"
        "Path('/app/reconstructed.txt').write_bytes(b'reconstructed')\n"
    )
    (app / "data.txt").write_bytes(b"input-data")
    (app / "sequenced.dat").write_bytes(b"lossy-input")
    (app / "original_sequences.dat").write_bytes(b"stale-encoded")
    (app / "reconstructed.txt").write_bytes(b"stale-reconstruction")
    (tmp_path / "private-verifier-file").write_text("must remain invisible")
    return tmp_path


def test_matches_only_official_encoder_and_decoder_launches() -> None:
    assert dna.is_candidate(
        "terminal-bench-science/dna-storage-codec",
        ["/usr/bin/python3", "/app/encoder.py", "/app/data.txt"],
        {"user": "nobody", "group": "nogroup"},
    )
    assert dna.is_candidate(
        "dna-storage-codec",
        ["/usr/bin/python3", "/app/encoder.py", "/app/data.txt"],
        {"user": "nobody", "group": "nogroup"},
    )
    assert dna.is_candidate(
        "terminal-bench-science/dna-storage-codec",
        ["/usr/bin/python3", "/app/decoder.py", "/app/sequenced.dat"],
        {"user": "nobody", "group": "nogroup"},
    )
    assert not dna.is_candidate(
        "terminal-bench-science/dna-storage-codec",
        ["/usr/bin/python3", "/tests/channel_simulator.py", "in", "out", "0"],
        {},
    )
    assert not dna.is_candidate(
        "terminal-bench-science/dna-storage-codec",
        ["/usr/bin/python3", "/app/encoder.py", "/tests/private"],
        {"user": "nobody", "group": "nogroup"},
    )
    assert not dna.is_candidate(
        "unrelated/dna-storage-codec",
        ["/usr/bin/python3", "/app/encoder.py", "/app/data.txt"],
        {"user": "nobody", "group": "nogroup"},
    )


def test_encoder_and_decoder_phase_inputs_are_isolated(app_root: Path) -> None:
    bwrap = shutil.which("bwrap")
    if bwrap is None:
        pytest.skip("bubblewrap is unavailable")
    launch = dna.prepare_candidate(
        "terminal-bench-science/dna-storage-codec",
        ["/usr/bin/python3", "/app/encoder.py", "/app/data.txt"],
        environment={"PATH": "/usr/bin:/bin", "HOME": "/app", "TMPDIR": "/app"},
        cwd="/app", source_root=app_root, bwrap=bwrap,
    )
    result = subprocess.run(launch.command, env=launch.environment,
                            capture_output=True, text=True, timeout=20)
    launch.export_output(result.returncode)
    assert result.returncode == 0, result.stderr
    assert (app_root / "app/original_sequences.dat").read_bytes() == b"encoded"
    assert not (app_root / "app/encoder-scratch").exists()
    assert (app_root / "app/data.txt").read_bytes() == b"input-data"
    assert (app_root / "app/reconstructed.txt").read_bytes() == b"stale-reconstruction"

    # Match the official phase reset: only the trusted lossy-channel output is
    # retained beside the pristine candidate files before decoder launch.
    (app_root / "app/original_sequences.dat").unlink()
    (app_root / "app/data.txt").unlink()
    (app_root / "app/reconstructed.txt").unlink()
    launch = dna.prepare_candidate(
        "terminal-bench-science/dna-storage-codec",
        ["/usr/bin/python3", "/app/decoder.py", "/app/sequenced.dat"],
        environment={"PATH": "/usr/bin:/bin", "HOME": "/app", "TMPDIR": "/app"},
        cwd="/app", source_root=app_root, bwrap=bwrap,
    )
    result = subprocess.run(launch.command, env=launch.environment,
                            capture_output=True, text=True, timeout=20)
    launch.export_output(result.returncode)
    assert result.returncode == 0, result.stderr
    assert (app_root / "app/reconstructed.txt").read_bytes() == b"reconstructed"
    assert not (app_root / "app/original_sequences.dat").exists()
    assert not (app_root / "app/data.txt").exists()
    assert (app_root / "app/sequenced.dat").read_bytes() == b"lossy-input"


def test_science_popen_hook_exports_only_declared_dna_output(
    app_root: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    bwrap = shutil.which("bwrap")
    if bwrap is None:
        pytest.skip("bubblewrap is unavailable")
    prepare = dna.prepare_candidate

    def prepare_fixture(task_id, argv, **kwargs):
        return prepare(task_id, argv, source_root=app_root, bwrap=bwrap, **kwargs)

    monkeypatch.setattr(dna, "prepare_candidate", prepare_fixture)
    monkeypatch.setenv("RELAY_SCIENCE_TASK_ID", "terminal-bench-science/dna-storage-codec")
    monkeypatch.setattr(subprocess, "Popen", science_isolation._ORIGINAL_POPEN)
    science_isolation.install()
    result = subprocess.run(
        ["/usr/bin/python3", "/app/encoder.py", "/app/data.txt"],
        cwd="/app", user="nobody", group="nogroup",
        env={"PATH": "/usr/bin:/bin", "HOME": "/app", "TMPDIR": "/app"},
        capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert (app_root / "app/original_sequences.dat").read_bytes() == b"encoded"
    assert (app_root / "app/data.txt").read_bytes() == b"input-data"
    assert not (app_root / "app/encoder-scratch").exists()


@pytest.mark.parametrize(
    "argv",
    [
        ["/usr/bin/python3", "/app/encoder.py", "/tests/private"],
        ["/usr/bin/python3", "/app/encoder.py", "/app/data.txt", "extra"],
        ["/bin/sh", "/app/encoder.py", "/app/data.txt"],
    ],
)
def test_rejects_unreviewed_argument_shapes(app_root: Path, argv: list[str]) -> None:
    with pytest.raises(ValueError, match="Unexpected DNA codec candidate argv"):
        dna.prepare_candidate(
            "terminal-bench-science/dna-storage-codec", argv,
            environment={}, cwd="/app", source_root=app_root,
        )


def test_refuses_symlinked_candidate_content(app_root: Path) -> None:
    (app_root / "app/helper.py").symlink_to("/tests/private")
    with pytest.raises(ValueError, match="Symlinked DNA codec candidate files"):
        dna.prepare_candidate(
            "terminal-bench-science/dna-storage-codec",
            ["/usr/bin/python3", "/app/encoder.py", "/app/data.txt"],
            environment={}, cwd="/app", source_root=app_root,
        )


def test_only_successful_declared_output_is_exported(tmp_path: Path) -> None:
    scratch = tmp_path / "private-scratch"
    (scratch / "app").mkdir(parents=True)
    (scratch / "app/original_sequences.dat").write_bytes(b"candidate-output")
    destination = tmp_path / "official-app/original_sequences.dat"
    launch = dna.CandidateLaunch([], {}, scratch, "encoder", destination)
    launch.export_output(1)
    assert not destination.exists()
    assert not scratch.exists()
