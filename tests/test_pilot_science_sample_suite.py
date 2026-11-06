"""Model-free checks for configurable Science per-cell worker budgets."""
from __future__ import annotations
import os
import shutil
import subprocess
import sys
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch
import pytest

from evals.pilot import science_sample_suite


def test_rejects_nonpositive_worker_limits_before_readiness_or_inference(tmp_path: Path) -> None:
    args=Namespace(task=tmp_path/'task',images=tmp_path/'images',out=tmp_path/'out',scratch=tmp_path,
                   python=tmp_path/'python',model='gpt-6-luna',strategy='compaction',seconds=0,max_requests=128)
    with pytest.raises(ValueError,match='must be positive'):
        science_sample_suite.run_one(args)


def test_cli_passes_custom_worker_limits_without_running_inference(monkeypatch) -> None:
    seen={}
    def capture(args):
        seen['seconds']=args.seconds
        seen['max_requests']=args.max_requests
        return 0
    monkeypatch.setattr(science_sample_suite,'run_one',capture)
    monkeypatch.setattr(sys,'argv',['science_sample_suite','--task','task','--images','images',
      '--scratch','scratch','--out','out','--python','python','--model','gpt-6-luna',
      '--strategy','compaction','--seconds','7200','--max-requests','128'])
    assert science_sample_suite.main()==0
    assert seen=={'seconds':7200,'max_requests':128}


def test_cli_keeps_old_defaults_for_existing_callers(monkeypatch) -> None:
    seen={}
    def capture(args):
        seen['seconds']=args.seconds
        seen['max_requests']=args.max_requests
        return 0
    monkeypatch.setattr(science_sample_suite,'run_one',capture)
    monkeypatch.setattr(sys,'argv',['science_sample_suite','--task','task','--images','images',
      '--scratch','scratch','--out','out','--python','python','--model','gpt-6-luna',
      '--strategy','compaction'])
    assert science_sample_suite.main()==0
    assert seen=={'seconds':1800,'max_requests':64}


@pytest.mark.parametrize("name", ["mendota-ice-phenology", "terminal-bench-science/mendota-ice-phenology"])
def test_verifier_canonicalizes_config_task_id_before_bundle(tmp_path: Path, monkeypatch, name: str) -> None:
    from evals.pilot import science_isolation

    task = tmp_path / "task"
    task.mkdir()
    (task / "task.toml").write_text(f'[task]\nname = "{name}"\n')
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    seen = []

    class BundleReached(Exception):
        pass

    def capture(task_id, output):
        seen.append(task_id)
        raise BundleReached

    monkeypatch.setattr(science_isolation, "create_bundle", capture)
    with pytest.raises(BundleReached):
        science_sample_suite.verify(task, tmp_path / "images", frozen, tmp_path / "verify", Path(sys.executable))
    assert seen == ["terminal-bench-science/mendota-ice-phenology"]


def test_traffic_official_cleanup_runs_inside_nested_pid_namespace() -> None:
    image=Path('/images/tests.sif')
    assert science_sample_suite.official_test_command(
        'terminal-bench-science/traffic-flux-inversion', image
    )==[str(image),'/usr/bin/unshare','--pid','--fork','--mount-proc','/bin/bash','/tests/test.sh']
    assert science_sample_suite.official_test_command('terminal-bench-science/dna-storage-codec',image)==[
        str(image),'/bin/bash','/tests/test.sh'
    ]


def test_verifier_uses_tests_image_workdir(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    (images / "build.json").write_text(
        '{"images":{"tests":{"workdir":"/tests"}}}'
    )
    assert science_sample_suite.tests_image_workdir(images) == "/tests"


def test_verifier_workdir_falls_back_only_when_manifest_missing(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    assert science_sample_suite.tests_image_workdir(images) == "/"
    (images / "build.json").write_text(
        '{"images":{"tests":{"workdir":"tests"}}}'
    )
    with pytest.raises(ValueError, match="absolute normalized"):
        science_sample_suite.tests_image_workdir(images)


def test_leaky_bloch_ignores_only_the_incompatible_host_fakeroot_command() -> None:
    assert science_sample_suite.fakeroot_args(
        "terminal-bench-science/leaky-bloch-meep"
    ) == ["--fakeroot", "--ignore-fakeroot-command"]
    assert science_sample_suite.fakeroot_args(
        "terminal-bench-science/navigation-sensor-calibration"
    ) == ["--fakeroot"]


def test_leaky_bloch_preserves_tests_image_python_path_and_console_log(tmp_path: Path) -> None:
    environment = {"PATH": "/opt/relay-isolation/bin:/usr/bin:/bin"}
    actual = science_sample_suite.verifier_environment(
        "terminal-bench-science/leaky-bloch-meep", environment,
    )
    assert actual["PATH"] == (
        "/opt/conda/envs/pymeep-runtime/bin:/opt/conda/bin:"
        "/opt/relay-isolation/bin:/usr/bin:/bin"
    )
    assert environment["PATH"] == "/opt/relay-isolation/bin:/usr/bin:/bin"
    logs = tmp_path / "logs"
    assert science_sample_suite.verifier_console_log_path(
        "terminal-bench-science/leaky-bloch-meep", tmp_path, logs,
    ) == tmp_path / "verifier-console.log"
    assert science_sample_suite.verifier_console_log_path(
        "terminal-bench-science/navigation-sensor-calibration", tmp_path, logs,
    ) == logs / "verifier.log"


def test_root_mapped_cleanup_reaps_only_nested_test_process(tmp_path: Path) -> None:
    image=os.environ.get('RELAY_TRAFFIC_CLEANUP_TEST_IMAGE')
    if not image or not shutil.which('apptainer'):
        pytest.skip('requires an explicit Traffic tests image and Apptainer')
    script=tmp_path/'cleanup_probe.sh'
    script.write_text('''#!/bin/bash
set +e
echo "${1:-}" > /probe/identity
su -s /bin/bash -c 'kill -9 -1' nobody
status=$?
printf 'survived:%s\\n' "$status" > /probe/after
''')
    command=['apptainer','exec','--fakeroot','--writable-tmpfs','--cleanenv','--containall',
             '--no-home','--no-mount','hostfs,bind-paths','--bind',str(tmp_path)+':/probe',image]
    direct=subprocess.run(command+['/bin/bash','/probe/cleanup_probe.sh','direct'],
                          capture_output=True,text=True,check=False,timeout=30)
    assert direct.returncode==137
    assert not (tmp_path/'after').exists()
    nested=subprocess.run(command+['/usr/bin/unshare','--pid','--fork','--mount-proc',
                                   '/bin/bash','/probe/cleanup_probe.sh','nested'],
                          capture_output=True,text=True,check=False,timeout=30)
    assert nested.returncode==0, nested.stderr
    assert (tmp_path/'after').read_text().strip()=='survived:137'
