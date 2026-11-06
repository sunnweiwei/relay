"""Build official Science task images and expose an isolated persistent shell.

Dockerfile conversion requires spython and pathspec in the build interpreter.
Model orchestration and the task service use only the standard library.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import posixpath
import re
import secrets
import shlex
import shutil
import signal
import socket
import subprocess
import time
import tomllib
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen


HERE = Path(__file__).resolve().parent


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "Apptainer.def":
            digest.update(str(path.relative_to(root)).encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def docker_logical_lines(text: str) -> list[str]:
    """Join Docker continuations before interpreting instruction names.

    Keep whitespace inside instruction arguments and remove full-line comments,
    as specified by https://docs.docker.com/reference/dockerfile/#format.
    In particular, Python ``from`` inside a continued RUN is not a new stage.
    """
    result: list[str] = []
    pending = ""
    escape = "\\"
    directives_open = True
    for physical in text.splitlines():
        stripped = physical.lstrip()
        if stripped.startswith("#"):
            directive = re.fullmatch(r"#\s*(syntax|escape|check)\s*=\s*(.*?)\s*", stripped, re.I)
            if directives_open and directive:
                if directive[1].lower() == "escape":
                    if directive[2] not in {"\\", "`"}:
                        raise ValueError("Invalid Dockerfile escape directive")
                    escape = directive[2]
            else:
                directives_open = False
            continue
        directives_open = False
        if not stripped:
            continue
        line = physical if pending else stripped
        end = line.rstrip(" \t")
        if end.endswith(escape):
            pending += end[:-1]
        else:
            result.append(pending + line)
            pending = ""
    if pending:
        raise ValueError("Dockerfile ends inside a continued instruction")
    return result


def docker_run_command(command: str) -> str:
    """Preserve each RUN's shell scope and fail the build on its exit status."""
    argv = None
    if command.startswith("["):
        try:
            argv = json.loads(command)
        except json.JSONDecodeError:
            pass  # A shell test such as [ "$(uname -m)" = x86_64 ] is not JSON.
    if isinstance(argv, list) and argv and all(isinstance(part, str) for part in argv):
        executable = shlex.join(argv)
    else:
        executable = shlex.join(["/bin/sh", "-c", command])
    return executable + " || exit $?"


def parse_dockerfile(path: Path) -> dict[str, Any]:
    """Feed logical instructions to spython without rewriting task sources."""
    from spython.main.parse.parsers import DockerParser

    class LogicalDockerParser(DockerParser):
        def _run(self, line: str) -> None:
            self.recipe[self.active_layer].install.append(docker_run_command(line.split(None, 1)[1]))

        def _arg(self, line: str) -> None:
            previous = len(self.recipe[self.active_layer].install)
            super()._arg(line)
            install = self.recipe[self.active_layer].install
            install[previous:] = ["export " + assignment for assignment in install[previous:]]

        def _env(self, line: str) -> None:
            previous = len(self.recipe[self.active_layer].install)
            super()._env(line)
            install = self.recipe[self.active_layer].install
            install[previous:] = ["export " + assignment for assignment in install[previous:]]

    parser = LogicalDockerParser(str(path), load=False)
    lines = docker_logical_lines(path.read_text())
    names = {words[-1] for line in lines if line.split(None, 1)[0].upper() == "FROM"
             and len(words := shlex.split(line)) >= 4 and words[-2].upper() == "AS"}
    stage_index = 0
    parser.lines = []
    for line in lines:
        words = line.split()
        if words and words[0].upper() == "FROM":
            # spython otherwise overwrites the previous stage when a later
            # FROM has no AS name. Synthetic names affect conversion only.
            if len(words) < 4 or words[-2].upper() != "AS":
                name = f"relay_generated_stage_{stage_index}"
                while name in names:
                    name += "_"
                names.add(name)
                line += " AS " + name
            stage_index += 1
        parser.lines.append(line)
    return parser.parse()


def docker_copy_specs(text: str) -> list[list[dict[str, Any]]]:
    """Resolve COPY destinations using the WORKDIR at that instruction.

    spython 0.3.14 otherwise emits relative destinations in %files, where they
    are interpreted relative to / rather than the Docker WORKDIR.
    """
    stages: list[list[dict[str, Any]]] = []
    named_workdirs: dict[str, str] = {}
    workdir = "/"
    name = ""
    for line in docker_logical_lines(text):
        fields = line.strip().split(None, 1)
        if len(fields) != 2 or fields[0].startswith("#"):
            continue
        operation, value = fields[0].upper(), fields[1]
        if operation == "FROM":
            if name:
                named_workdirs[name] = workdir
            words = shlex.split(value)
            workdir = named_workdirs.get(words[0], "/")
            name = words[2] if len(words) >= 3 and words[1].upper() == "AS" else ""
            stages.append([])
        elif operation == "WORKDIR":
            workdir = posixpath.normpath(posixpath.join(workdir, value.strip()))
        elif operation in {"COPY", "ADD"}:
            source_stage = None
            if value.startswith("--from="):
                flag, value = value.split(None, 1)
                source_stage = flag.removeprefix("--from=")
            words = json.loads(value) if value.startswith("[") else shlex.split(value)
            if len(words) < 2 or any(word.startswith("--") for word in words):
                raise NotImplementedError("Docker COPY flags other than --from require explicit conversion support")
            destination = words[-1]
            if "$" in destination:
                raise NotImplementedError("Variable-expanded COPY destinations are not supported")
            absolute = posixpath.normpath(posixpath.join(workdir, destination))
            if destination.endswith("/") or destination in {".", ".."} or len(words) > 2:
                absolute += "/"
            stages[-1].append({"sources": words[:-1], "destination": absolute,
                               "from_stage": source_stage})
    return stages


def docker_copy_paths(text: str) -> list[list[list[str]]]:
    """Return context-only COPY pairs; stage copies need built image roots."""
    specs = docker_copy_specs(text)
    if any(spec["from_stage"] is not None for stage in specs for spec in stage):
        raise NotImplementedError("Stage COPY needs the corresponding built image")
    return [[[source, spec["destination"]] for spec in stage for source in spec["sources"]]
            for stage in specs]


def stage_copy_source(root: Path, source: str) -> Path:
    """Resolve a COPY source without allowing an image link to read host files."""
    root = root.resolve()
    if ".." in Path(source).parts:
        raise ValueError("Stage COPY source contains parent traversal")
    path = root / source.lstrip("/")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError("Stage COPY source escapes its extracted image")
    if resolved.is_dir():
        for child in resolved.rglob("*"):
            if child.is_symlink() and not child.resolve(strict=True).is_relative_to(root):
                raise ValueError("Stage COPY directory link escapes its extracted image")
    return resolved


def oci_reference(reference: str) -> str:
    """Keep a pinned digest while removing an OCI-parser-incompatible tag."""
    if "@" not in reference:
        return reference
    name, digest = reference.split("@", 1)
    if ":" in name.rsplit("/", 1)[-1]:
        name = name.rsplit(":", 1)[0]
    return name + "@" + digest


def expand_directory_copies(context: Path, pairs: list[list[str]]) -> tuple[list[list[str]], list[str]]:
    """Preserve Docker's directory-content COPY semantics in Apptainer %files."""
    files: list[list[str]] = []
    directories: list[str] = []
    for source, destination in pairs:
        path = context / source
        if path.is_dir():
            directories.append(destination)
            for item in sorted(path.rglob("*")):
                relative = item.relative_to(path)
                target = posixpath.join(destination, relative.as_posix())
                if item.is_dir():
                    directories.append(target)
                elif item.is_file():
                    source_path = item.relative_to(context).as_posix() if item.is_relative_to(context) else str(item)
                    files.append([source_path, target])
        else:
            files.append([source, destination])
    return files, directories


def image_build_environment(output: Path) -> dict[str, str]:
    """Keep build-time /tmp inside the image and caches outside shared home."""
    env = os.environ.copy()
    disabled = {part.strip() for part in env.get("APPTAINER_NO_MOUNT", "").split(",") if part.strip()}
    disabled.add("tmp")
    # Root-mapped fakeroot otherwise binds host /tmp into %post. That hides
    # Docker COPY /tmp inputs and loses /tmp outputs needed by later stages.
    env["APPTAINER_NO_MOUNT"] = ",".join(sorted(disabled))
    env.setdefault("APPTAINER_CACHEDIR", str(output.parent / ".apptainer-cache"))
    env.setdefault("APPTAINER_TMPDIR", str(output / ".build-tmp"))
    for name in ("APPTAINER_CACHEDIR", "APPTAINER_TMPDIR"):
        Path(env[name]).mkdir(parents=True, exist_ok=True, mode=0o700)
    return env


def build_images(task: Path, output: Path, kinds: tuple[str, ...] = ("environment", "tests"),
                 *, stage_timeout_seconds: float | None = None) -> dict[str, Any]:
    """Convert each Dockerfile stage without changing its install commands."""
    import pathspec
    from spython.main.parse.writers import SingularityWriter

    task = task.resolve()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_hash = tree_hash(task)
    old_path = output / "build.json"
    old: dict[str, Any] = {}
    if old_path.exists():
        old = json.loads(old_path.read_text())
        if old.get("source_hash") != source_hash:
            raise ValueError("Task source changed; build into a new directory")
        if old.get("status") == "completed":
            return old
    config = tomllib.loads((task / "task.toml").read_text())
    official_build_timeout = float(config["environment"].get("build_timeout_sec", 1200))
    stage_timeout = build_stage_timeout(official_build_timeout, stage_timeout_seconds)
    result: dict[str, Any] = {"task_id": config["task"]["name"], "source_hash": source_hash,
                             "status": "building", "images": old.get("images", {}), "started_unix": time.time(),
                             "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                             "builder_policy": "logical_docker_stages_private_tmp_bounded_setup_v4",
                             "official_build_timeout_seconds": official_build_timeout,
                             "stage_timeout_seconds": stage_timeout,
                             "setup_timeout_override_seconds": stage_timeout_seconds}
    build_env = image_build_environment(output)
    write_json(old_path, result)
    for kind in kinds:
        context = output / (kind + "-context")
        if context.exists():
            shutil.rmtree(context)
        source = task / kind
        patterns = (source / ".dockerignore").read_text().splitlines() if (source / ".dockerignore").exists() else []
        # GitWildMatch covers the patterns used by the frozen selected tasks.
        spec = pathspec.PathSpec.from_lines("gitwildmatch", patterns)
        context.mkdir()
        for item in sorted(source.rglob("*")):
            relative = item.relative_to(source)
            if item.is_file() and item.name != "Apptainer.def" and not spec.match_file(str(relative)):
                target = context / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(item, target)
        previous = Path.cwd()
        try:
            os.chdir(context)
            recipes = parse_dockerfile(Path("Dockerfile"))
        finally:
            os.chdir(previous)
        stages: dict[str, Path] = {}
        inherited: dict[str, Any] = {}
        copy_specs = docker_copy_specs((context / "Dockerfile").read_text())
        extracted_stages: dict[str, Path] = {}
        for index, (name, original) in enumerate(recipes.items()):
            recipe = copy.deepcopy(original)
            base = recipe.fromHeader.strip()
            pairs = []
            for spec in copy_specs[index]:
                source_stage = spec["from_stage"]
                if source_stage is None:
                    pairs.extend([[source, spec["destination"]] for source in spec["sources"]])
                    continue
                if source_stage not in stages:
                    raise NotImplementedError("Stage COPY requires an earlier, pinned Dockerfile stage")
                if source_stage not in extracted_stages:
                    stage_root = output / f"{kind}-copy-{len(extracted_stages):02d}"
                    extract_image(stages[source_stage], stage_root)
                    extracted_stages[source_stage] = stage_root
                pairs.extend([[str(stage_copy_source(extracted_stages[source_stage], source)),
                               spec["destination"]] for source in spec["sources"]])
            recipe.files, copy_directories = expand_directory_copies(context, pairs)
            recipe.layer_files = {}
            inherited_base = inherited.get(base)
            if inherited_base is not None:
                # Docker FROM <stage> inherits the filesystem, env, and workdir.
                recipe.environ = list(inherited_base.environ) + list(recipe.environ)
                recipe.workdir = recipe.workdir or inherited_base.workdir
            definition = SingularityWriter({name: recipe}).convert()
            if copy_directories:
                setup = "\n".join("mkdir -p " + shlex.quote(path) for path in copy_directories)
                definition = definition.replace("%post\n", "%post\n" + setup + "\n", 1)
            if base in stages:
                definition = definition.replace("Bootstrap: docker", "Bootstrap: localimage", 1)
                definition = definition.replace("From: " + recipe.fromHeader,
                                                "From: " + str(stages[base]), 1)
            else:
                definition = definition.replace("From: " + recipe.fromHeader,
                                                "From: " + oci_reference(base), 1)
            definition_path = output / f"{kind}-stage-{index:02d}.def"
            definition_path.write_text(definition)
            image = output / f"{kind}-stage-{index:02d}.sif"
            command = ["apptainer", "build", "--fakeroot", str(image), str(definition_path)]
            with (output / f"{kind}-stage-{index:02d}.log").open("w") as log:
                if not image.exists():
                    process = subprocess.Popen(command, cwd=context, env=build_env, stdout=log,
                                               stderr=subprocess.STDOUT, start_new_session=True)
                    try:
                        exit_code = process.wait(timeout=stage_timeout)
                    except subprocess.TimeoutExpired:
                        stop_timed_out_build(process)
                        result.update(status="build_timed_out", failed_stage=kind + ":" + name,
                                      exit_code=process.returncode, finished_unix=time.time())
                        write_json(old_path, result)
                        return result
                    if exit_code:
                        result.update(status="build_failed", failed_stage=kind + ":" + name,
                                      exit_code=exit_code, finished_unix=time.time())
                        write_json(old_path, result)
                        return result
            stages[name] = image
            stages[str(index)] = image
            inherited[name] = recipe
        final = output / (kind + ".sif")
        if not final.exists():
            os.link(image, final)
        result["images"][kind] = {"path": str(final), "sha256": hashlib.sha256(final.read_bytes()).hexdigest(),
                                  "workdir": recipe.workdir or "/", "stages": len(recipes)}
        write_json(old_path, result)
    result.update(status="completed" if {"environment", "tests"} <= result["images"].keys() else "partially_built",
                  finished_unix=time.time())
    write_json(old_path, result)
    return result


def build_stage_timeout(official_seconds: float, override_seconds: float | None) -> float:
    """Allow slower image setup without changing task or verifier budgets."""
    if override_seconds is not None and not 0 < override_seconds < float("inf"):
        raise ValueError("Image setup timeout must be finite and positive")
    return max(official_seconds + 600, override_seconds or 0)


def stop_timed_out_build(process: subprocess.Popen[Any]) -> None:
    """Reap descendants even when the build parent exits first after SIGTERM."""
    try:
        stop_process(process)
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def extract_image(image: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(destination)
    with destination.with_suffix(".extract.log").open("w") as log:
        subprocess.run(["apptainer", "build", "--sandbox", "--fix-perms", str(destination), str(image)],
                       stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)


def portable_python_prefix(python: Path) -> Path:
    """Find uv's portable interpreter, excluding the external virtualenv."""
    result = subprocess.check_output([str(python), "-c", "import sys; print(sys.base_prefix)"], text=True)
    return Path(result.strip())


def cpu_prefix(cpus: int) -> list[str]:
    available = sorted(os.sched_getaffinity(0))
    if cpus < 1 or cpus > len(available):
        raise ValueError(f"Task needs {cpus} CPUs but allocation exposes {len(available)}")
    return ["taskset", "--cpu-list", ",".join(map(str, available[:cpus]))]


def stop_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)


def start_service(task_id: str, image: Path, workdir: str, output: Path,
                  python: Path, cpus: int | None = None,
                  environment: dict[str, str] | None = None) -> tuple[subprocess.Popen[Any], dict[str, Any]]:
    """Run a service in a fresh rootfs and PID namespace, with no host files."""
    output.mkdir(parents=True, exist_ok=True)
    rootfs = output / "rootfs"
    extract_image(image, rootfs)
    observations = output / "observations"
    observations.mkdir()
    token = secrets.token_urlsafe(36)
    token_path = output / "service-token"
    token_path.write_text(token)
    token_path.chmod(0o600)
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    python_root = portable_python_prefix(python)
    command = ["bwrap", "--die-with-parent", "--unshare-all", "--share-net",
               "--uid", "0", "--gid", "0", "--bind", str(rootfs), "/",
               "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--tmpfs", "/run",
               "--ro-bind", str(python_root), "/opt/relay-runtime",
               "--ro-bind", str(HERE / "science_service.py"), "/opt/relay-service/service.py",
               "--ro-bind", str(token_path), "/opt/relay-service/token",
               "--bind", str(observations), "/opt/relay-observations",
               "--ro-bind", "/etc/resolv.conf", "/etc/resolv.conf", "--chdir", workdir,
               "--clearenv", "--setenv", "HOME", "/root", "--setenv", "PATH", "/usr/local/bin:/usr/bin:/bin",
               "/bin/bash", "-c",
               'for f in /.singularity.d/env/*.sh; do . "$f"; done; exec "$@"', "science-service",
               "/opt/relay-runtime/bin/python3", "/opt/relay-service/service.py",
               "--task-id", task_id, "--token-file", "/opt/relay-service/token",
               "--workdir", workdir, "--trace", "/opt/relay-observations/exec.jsonl", "--port", str(port)]
    if environment:
        at = command.index("/bin/bash")
        for key, value in environment.items():
            command[at:at] = ["--setenv", key, value]
            at += 3
    if cpus is not None:
        command = cpu_prefix(cpus) + command
    write_json(output / "command.json", command)
    with (output / "service.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    endpoint = {"base_url": f"http://127.0.0.1:{port}", "token": token, "task_id": task_id}
    try:
        for _ in range(100):
            if process.poll() is not None:
                raise RuntimeError("Science task container exited; inspect service.log")
            request = Request(endpoint["base_url"] + "/health",
                              headers={"Authorization": "Bearer " + token})
            try:
                with urlopen(request, timeout=1) as response:
                    state = json.load(response)
                if state.get("task_id") == task_id:
                    return process, endpoint
            except OSError:
                time.sleep(0.1)
        raise TimeoutError("Science task service did not become healthy")
    except BaseException:
        stop_process(process)
        raise


def freeze_artifacts(rootfs: Path, artifacts: list[str | dict[str, Any]], output: Path) -> list[dict[str, Any]]:
    """Copy only declared artifacts after the entire worker namespace exits."""
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for declaration in artifacts:
        artifact = declaration if isinstance(declaration, str) else declaration["source"]
        if isinstance(declaration, dict):
            if declaration.get("service", "main") != "main":
                raise ValueError("Service-qualified artifacts require the task's separate service rootfs")
            if declaration.get("exclude") == ["*"]:
                records.append({"path": artifact, "present": False, "excluded": True,
                                "declaration": declaration})
                continue
            if declaration.get("exclude"):
                raise ValueError("This task needs a reviewed artifact exclusion pattern")
        relative = Path(artifact).relative_to("/")
        source = rootfs / relative
        if not source.exists():
            records.append({"path": artifact, "present": False})
            continue
        if source.is_symlink():
            # A link is resolved again where the verifier mounts the artifact, possibly onto
            # hidden test data, so links are never frozen.
            raise ValueError(f"Artifact {artifact} is a symbolic link")
        source.resolve().relative_to(rootfs.resolve())
        links = sorted(str(item.relative_to(source)) for item in source.rglob("*") if item.is_symlink()) \
            if source.is_dir() else []
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, destination, symlinks=True,
                            ignore=lambda folder, names: [name for name in names if Path(folder, name).is_symlink()])
        else:
            shutil.copyfile(source, destination)
        records.append({"path": artifact, "present": True, "frozen": str(destination),
                        **({"symlinks_dropped": links} if links else {})})
    write_json(output / "artifacts.json", records)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--task", type=Path, required=True)
    build.add_argument("--out", type=Path, required=True)
    build.add_argument("--kinds", nargs="+", choices=("environment", "tests"), default=["environment", "tests"])
    build.add_argument("--stage-timeout-seconds", type=float,
                       help="Bounded image setup override; never reduces the declared build allowance")
    args = parser.parse_args()
    if args.command == "build":
        result = build_images(args.task, args.out, tuple(args.kinds),
                              stage_timeout_seconds=args.stage_timeout_seconds)
        print(json.dumps(result), flush=True)
        raise SystemExit(0 if result["status"] == "completed" else 1)


if __name__ == "__main__":
    main()
