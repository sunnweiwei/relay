"""Generic Science preparation and task-service invariants."""

import json
import argparse
import subprocess
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from evals.pilot.science_container import docker_copy_paths, docker_copy_specs, docker_logical_lines, docker_run_command, expand_directory_copies, freeze_artifacts, image_build_environment, oci_reference, parse_dockerfile, stage_copy_source
from evals.pilot.science_sample_suite import require_readiness
from evals.pilot.science_service import TaskService, handler
from evals.pilot import science_sample_matrix


def test_image_setup_override_does_not_reduce_official_allowance() -> None:
    from evals.pilot.science_container import build_stage_timeout
    assert build_stage_timeout(1800, None) == 2400
    assert build_stage_timeout(1800, 7200) == 7200
    assert build_stage_timeout(3600, 1200) == 4200
    for invalid in (0, -1, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            build_stage_timeout(1800, invalid)


def test_timed_out_build_reaps_child_that_ignores_termination(tmp_path: Path) -> None:
    import select
    import sys
    import time
    from evals.pilot.science_container import stop_timed_out_build
    marker = tmp_path / "child-ready"
    child = ("import signal,time,pathlib; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
             f"pathlib.Path({str(marker)!r}).touch(); time.sleep(30)")
    parent = ("import subprocess,sys,time,pathlib\n"
              f"child = subprocess.Popen([sys.executable, '-c', {child!r}])\n"
              f"while not pathlib.Path({str(marker)!r}).exists(): time.sleep(0.01)\n"
              "print(child.pid, flush=True)\nchild.wait()\n")
    process = subprocess.Popen([sys.executable, "-c", parent], stdout=subprocess.PIPE,
                               text=True, start_new_session=True)
    try:
        assert select.select([process.stdout], [], [], 5)[0]
        child_pid = int(process.stdout.readline())
        stop_timed_out_build(process)
        assert process.poll() is not None
        status = Path(f"/proc/{child_pid}/status")
        for _ in range(50):
            if not status.exists() or "State:\tZ" in status.read_text():
                break
            time.sleep(0.01)
        assert not status.exists() or "State:\tZ" in status.read_text()
    finally:
        stop_timed_out_build(process)
        process.stdout.close()


def test_continued_python_from_is_not_a_docker_stage(tmp_path: Path) -> None:
    source = 'FROM python:3.12\nRUN python3 -c "\\\nfrom pathlib import Path; \\\nprint(Path.cwd())"\nCOPY data /app/data\n'
    lines = docker_logical_lines(source)
    assert len(lines) == 3
    assert lines[1] == 'RUN python3 -c "from pathlib import Path; print(Path.cwd())"'
    assert docker_copy_paths(source) == [[["data", "/app/data"]]]
    pytest.importorskip("spython")
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text(source)
    recipes = parse_dockerfile(dockerfile)
    assert len(recipes) == 1
    recipe = next(iter(recipes.values()))
    assert recipe.fromHeader.strip() == "python:3.12"
    assert 'from pathlib import Path;' in ' '.join(recipe.install)


def test_docker_continuation_preserves_argument_whitespace_and_hashes() -> None:
    source = 'RUN echo "\\\n  hello\\\n# ignored Docker comment\n  world # literal"\n'
    assert docker_logical_lines(source) == ['RUN echo "  hello  world # literal"']


def test_docker_escape_directive_and_json_copy() -> None:
    source = '# escape=`\nFROM base\nCOPY ["one", `\n"/app/one"]\n'
    assert docker_copy_paths(source) == [[["one", "/app/one"]]]


def test_docker_run_failure_cannot_be_hidden_by_a_later_success() -> None:
    script = docker_run_command("false && true") + "\nprintf SHOULD_NOT_RUN"
    result = subprocess.run(["/bin/sh", "-c", script], capture_output=True, text=True)
    assert result.returncode != 0
    assert result.stdout == ""


def test_docker_run_shell_scope_and_exec_arguments(tmp_path: Path) -> None:
    script = docker_run_command("cd /") + "\npwd"
    result = subprocess.run(["/bin/sh", "-c", script], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0 and result.stdout.strip() == str(tmp_path)
    result = subprocess.run(["/bin/sh", "-c", docker_run_command('["printf", "%s", "literal $HOME"]')],
                            capture_output=True, text=True)
    assert result.returncode == 0 and result.stdout == "literal $HOME"
    assert subprocess.run(["/bin/sh", "-c", docker_run_command('[ "ok" = "ok" ]')]).returncode == 0


def test_stage_copy_tracks_source_stage_and_runtime_workdir() -> None:
    source = 'FROM tool AS uv\nFROM python\nWORKDIR /app\nCOPY --from=uv /uv /uvx ./bin/\n'
    assert docker_copy_specs(source) == [[], [{"sources": ["/uv", "/uvx"],
        "destination": "/app/bin/", "from_stage": "uv"}]]


def test_image_build_has_private_tmp_and_preserves_other_mount_restrictions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APPTAINER_NO_MOUNT", "home,proc")
    monkeypatch.delenv("APPTAINER_CACHEDIR", raising=False)
    monkeypatch.delenv("APPTAINER_TMPDIR", raising=False)
    output = tmp_path / "images"
    env = image_build_environment(output)
    assert set(env["APPTAINER_NO_MOUNT"].split(",")) == {"home", "proc", "tmp"}
    assert Path(env["APPTAINER_CACHEDIR"]) == tmp_path / ".apptainer-cache"
    assert Path(env["APPTAINER_TMPDIR"]).is_dir()


def test_stage_directory_copy_preserves_contents_and_rejects_host_links(tmp_path: Path) -> None:
    image = tmp_path / "image"
    directory = image / "data"
    directory.mkdir(parents=True)
    (directory / "input.txt").write_text("public")
    context = tmp_path / "context"
    context.mkdir()
    source = stage_copy_source(image, "/data")
    files, directories = expand_directory_copies(context, [[str(source), "/app/data/"]])
    assert files == [[str(directory / "input.txt"), "/app/data/input.txt"]]
    assert "/app/data/" in directories
    (directory / "host-link").symlink_to(tmp_path / "outside")
    (tmp_path / "outside").write_text("private")
    with pytest.raises(ValueError, match="escapes"):
        stage_copy_source(image, "/data")
    with pytest.raises(ValueError, match="traversal"):
        stage_copy_source(image, "../outside")


def test_docker_copy_uses_instruction_workdir_and_stage_inheritance() -> None:
    source = """FROM python:3.12 AS base
WORKDIR /app
COPY requirements.txt .
WORKDIR inputs
COPY data/ ./data/
FROM base AS task
COPY config.json .
"""
    assert docker_copy_paths(source) == [
        [["requirements.txt", "/app/"], ["data/", "/app/inputs/data/"]],
        [["config.json", "/app/inputs/"]],
    ]


def test_pinned_digest_survives_tag_normalization() -> None:
    assert oci_reference("python:3.12@sha256:abc") == "python@sha256:abc"
    assert oci_reference("registry:5000/ns/image:tag@sha256:abc") == "registry:5000/ns/image@sha256:abc"


def test_directory_copy_does_not_add_an_extra_directory_level(tmp_path: Path) -> None:
    (tmp_path / "data/empty").mkdir(parents=True)
    (tmp_path / "data/input.csv").write_text("data")
    files, directories = expand_directory_copies(tmp_path, [["data/", "/app/data/"]])
    assert files == [["data/input.csv", "/app/data/input.csv"]]
    assert "/app/data/empty" in directories


def test_frozen_artifact_cannot_follow_a_host_symlink(tmp_path: Path) -> None:
    rootfs = tmp_path / "rootfs"
    (rootfs / "app").mkdir(parents=True)
    secret = tmp_path / "private"
    secret.write_text("hidden")
    (rootfs / "app/solver.py").symlink_to(secret)
    with pytest.raises(ValueError):
        freeze_artifacts(rootfs, ["/app/solver.py"], tmp_path / "frozen")


def test_frozen_directory_drops_links_even_inside_the_rootfs(tmp_path: Path) -> None:
    rootfs = tmp_path / "rootfs"
    (rootfs / "app/submission").mkdir(parents=True)
    (rootfs / "tests").mkdir()
    (rootfs / "tests/ground_truth.json").write_text("hidden")
    (rootfs / "app/submission/result.json").write_text("{}")
    (rootfs / "app/submission/answer.json").symlink_to("../../tests/ground_truth.json")
    records = freeze_artifacts(rootfs, ["/app/submission"], tmp_path / "frozen")
    frozen = tmp_path / "frozen/app/submission"
    assert (frozen / "result.json").read_text() == "{}"
    assert not (frozen / "answer.json").exists() and not (frozen / "answer.json").is_symlink()
    assert records[0]["symlinks_dropped"] == ["answer.json"]


def test_images_alone_cannot_authorize_inference(tmp_path: Path) -> None:
    task = tmp_path / "task"
    task.mkdir()
    (task / "task.toml").write_text('[task]\nname="terminal-bench-science/example"\n')
    images = tmp_path / "images"
    images.mkdir()
    with pytest.raises(RuntimeError, match="readiness"):
        require_readiness(task, images)


def test_official_source_mapping_and_excluded_placeholder_are_frozen_correctly(tmp_path: Path) -> None:
    root = tmp_path / "rootfs"
    (root / "app/package").mkdir(parents=True)
    (root / "app/package/solver.py").write_text("candidate")
    (root / "logs/artifacts").mkdir(parents=True)
    (root / "logs/artifacts/not-a-submission.txt").write_text("excluded")
    records = freeze_artifacts(root, [
        {"source": "/app/package/"},
        {"source": "/logs/artifacts", "destination": "_empty_agent_artifacts", "exclude": ["*"]},
    ], tmp_path / "frozen")
    assert (tmp_path / "frozen/app/package/solver.py").read_text() == "candidate"
    assert not (tmp_path / "frozen/logs/artifacts").exists()
    assert records[1]["excluded"] and not records[1]["present"]
    with pytest.raises(ValueError, match="separate service rootfs"):
        freeze_artifacts(root, [{"source": "/state/transcript.json", "service": "assay"}],
                         tmp_path / "wrong-service")


def test_service_authentication_and_persistent_files(tmp_path: Path) -> None:
    service = TaskService("test-task", "test-token", tmp_path, tmp_path / "trace.jsonl")
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler(service))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with pytest.raises(HTTPError) as failure:
            urlopen(base + "/health", timeout=2)
        assert failure.value.code == 401
        for command in ("printf persisted > value.txt", "cat value.txt"):
            request = Request(base + "/exec", data=json.dumps({"command": command}).encode(),
                              headers={"Authorization": "Bearer test-token"})
            with urlopen(request, timeout=5) as response:
                result = json.load(response)
            assert result["exit_code"] == 0
        assert result["output"] == "persisted"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_context_only_failure_does_not_repeat_or_block_other_methods(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"selected_ids": ["task"]}))
    output = tmp_path / "matrix"
    previous = output / "task/gpt-6-luna/clm/attempt-001"
    (previous / "worker").mkdir(parents=True)
    (previous / "result.json").write_text(json.dumps({"status": "worker_protocol_invalid"}))
    (previous / "worker/run.json").write_text(json.dumps({
        "status": "stopped", "stop_reason": "context_budget_exceeded", "exit_code": 1,
        "protocol_valid": False, "requests_over_context_budget": 1,
        "strategy_errors": [], "sandbox_execution_errors": []}))
    monkeypatch.setattr(science_sample_matrix, "require_readiness", lambda *args: {})
    calls = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        attempt = Path(command[command.index("--out") + 1])
        attempt.mkdir(parents=True)
        (attempt / "result.json").write_text(json.dumps({"status": "completed_failed"}))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(science_sample_matrix.subprocess, "run", fake_run)
    args = argparse.Namespace(out=output, scratch=tmp_path, manifest=manifest, tasks_root=tmp_path,
                              images_root=tmp_path, models=["gpt-6-luna"], methods=["clm", "folding"], python=Path("python3"))
    assert science_sample_matrix.run_matrix(args) == 1  # Complete with an explicit method limitation.
    assert len(calls) == 1
    assert calls[0][calls[0].index("--strategy") + 1] == "folding"
    state = json.loads((output / "matrix.json").read_text())
    assert [entry["status"] for entry in state["entries"]] == ["context_budget_exceeded", "completed_failed"]
