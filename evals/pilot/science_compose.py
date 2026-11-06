"""Run the pinned multi-service Science tasks without shared service ports."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import secrets
import shutil
import subprocess
import threading
import time
import tomllib
from typing import Any

from evals.pilot.science_container import (
    cpu_prefix, extract_image, portable_python_prefix, start_service,
    stop_process, freeze_artifacts, write_json,
)
from evals.pilot.science_sidecar_transport import gateway, UnixConnection


@dataclass(frozen=True)
class Component:
    name: str
    context: str
    dockerfile: str
    port: int
    command: tuple[str, ...]
    workdir: str
    uid: int = 0
    peers: tuple[tuple[str, int], ...] = ()


COMPONENTS = {
    "longitudinal-clinical-agent": (
        Component("broker", "broker", "Dockerfile", 8000,
                  ("python3", "/usr/local/bin/clinenv-broker"), "/srv/clinenv"),),
    "protein-active-learning": (
        Component("assay", "assay", "Dockerfile", 8080,
                  ("python3", "/service/assay_server.py"), "/service"),
        Component("submission", "submission", "Dockerfile", 8081,
                  ("python3", "/service/submission_server.py"), "/service", 10001,
                  (("assay", 8080),))),
    "tamp-skill-planning": (
        Component("sim", ".", "sim/Dockerfile", 8788,
                  ("python3", "/app/sim_local/server.py"), "/app"),),
    "inverse-lithography": (
        Component("fabricator", ".", "fabricator/Dockerfile", 8080,
                  ("python3", "/opt/fabricator/server.py"), "/opt/fabricator"),),
}


def service_environment(slug: str, base: str) -> dict[str, str]:
    if slug == "longitudinal-clinical-agent":
        return {"CLINENV_BROKER_URL": base + "/broker"}
    if slug == "protein-active-learning":
        return {"ASSAY_URL": base + "/assay/query", "MODEL_URL": base + "/submission/submit"}
    if slug == "tamp-skill-planning":
        return {"TAMP_SIM_URL": base + "/sim", "MUJOCO_GL": "osmesa"}
    if slug == "inverse-lithography":
        return {"FABRICATOR_URL": base + "/fabricator"}
    raise ValueError(f"No reviewed multi-service layout for {slug}")


class TaskServices:
    """Own all per-task services, transport sockets, and declared artifacts."""
    def __init__(self, task: Path, images: Path, output: Path, python: Path) -> None:
        self.config = tomllib.loads((task / "task.toml").read_text())
        self.task_id = self.config["task"]["name"]
        self.slug = self.task_id.rsplit("/", 1)[-1]
        self.components = COMPONENTS[self.slug]
        self.images, self.output, self.python = images, output, python
        self.processes: list[subprocess.Popen[Any]] = []
        self.roots: dict[str, Path] = {}
        self.gateway = None
        self.endpoint: dict[str, Any] | None = None
        self.environment: dict[str, str] = {}

    def start(self, *, include_main: bool = True, setup_seconds: float = 360) -> None:
        self.output.mkdir(parents=True, exist_ok=False, mode=0o700)
        transport = self.output / "transport"
        transport.mkdir(mode=0o700)
        hosts = self.output / "hosts"
        hosts.write_text("127.0.0.1 localhost " + " ".join(c.name for c in self.components) + "\n")
        python_root = portable_python_prefix(self.python)
        module = Path(__file__).with_name("science_sidecar_transport.py")
        cpus = int(self.config["environment"]["cpus"])
        try:
            for component in self.components:
                root = self.output / component.name / "rootfs"
                root.parent.mkdir()
                image = self.images / "sidecars" / component.name / "environment.sif"
                extract_image(image, root)
                self.roots[component.name] = root
                config = self.output / (component.name + ".json")
                write_json(config, {
                    "socket": f"/run/relay-sidecar/{component.name}.sock",
                    "port": component.port, "command": list(component.command),
                    "workdir": component.workdir,
                    "peers": [{"socket": f"/run/relay-sidecar/{name}.sock", "port": port}
                              for name, port in component.peers],
                })
                command = cpu_prefix(cpus) + [
                    "bwrap", "--die-with-parent", "--unshare-all",
                    "--uid", str(component.uid), "--gid", str(component.uid),
                    "--bind", str(root), "/", "--proc", "/proc", "--dev", "/dev",
                    "--tmpfs", "/tmp", "--tmpfs", "/run",
                    "--bind", str(transport), "/run/relay-sidecar",
                    "--ro-bind", str(hosts), "/etc/hosts",
                    "--ro-bind", str(python_root), "/opt/relay-runtime",
                    "--ro-bind", str(module), "/opt/relay-sidecar/transport.py",
                    "--ro-bind", str(config), "/opt/relay-sidecar/config.json",
                    "--chdir", component.workdir, "--clearenv",
                    "--setenv", "HOME", "/root" if component.uid == 0 else "/home/submission",
                    "--setenv", "PATH", "/usr/local/bin:/usr/bin:/bin",
                    "/bin/sh", "-c", 'for f in /.singularity.d/env/*.sh; do . "$f"; done; exec "$@"',
                    "science-sidecar", "/opt/relay-runtime/bin/python3",
                    "/opt/relay-sidecar/transport.py", "--config", "/opt/relay-sidecar/config.json",
                ]
                if self.slug == "tamp-skill-planning":
                    # TAMP probes execute uploaded skills inside the simulator
                    # service, before grading. Apply the same exact worker
                    # boundary there, while keeping the trusted simulator API.
                    from evals.pilot.science_isolation import create_bundle

                    bundle = create_bundle(self.task_id, self.output / "sim-isolation")
                    if not (Path(bundle["bundle_dir"]) / "science_tamp_isolation.py").is_file():
                        raise RuntimeError("TAMP simulator requires its reviewed skill-worker boundary")
                    at = command.index("/bin/sh")
                    command[at:at] = [
                        # Needed only in the trusted simulator's user namespace
                        # to create its nested worker UID map. The worker drops
                        # all capabilities before running submitted skills.
                        "--cap-add", "CAP_SETFCAP",
                        "--ro-bind", str(bundle["bundle_dir"]), "/opt/relay-isolation",
                        "--ro-bind", "/usr/bin/bwrap", "/opt/relay-tools/bwrap",
                        "--setenv", "RELAY_SCIENCE_TASK_ID", self.task_id,
                    ]
                    shell_index = command.index("/bin/sh") + 2
                    command[shell_index] = ('for f in /.singularity.d/env/*.sh; do . "$f"; done; '
                        'export PYTHONPATH="/opt/relay-isolation${PYTHONPATH:+:$PYTHONPATH}"; exec "$@"')
                with (root.parent / "service.log").open("w") as log:
                    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                               start_new_session=True)
                self.processes.append(process)
                deadline = time.monotonic() + setup_seconds
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(f"Sidecar {component.name} exited during setup")
                    connection = UnixConnection(transport / (component.name + ".sock"), timeout=2)
                    try:
                        connection.request("GET", "/health")
                        response = connection.getresponse()
                        response.read()
                        if response.status == 200:
                            break
                    except OSError:
                        pass
                    finally:
                        connection.close()
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"Sidecar {component.name} health check timed out")
                    time.sleep(0.2)
            token = secrets.token_urlsafe(32)
            self.gateway = gateway(token, {c.name: transport / (c.name + ".sock") for c in self.components})
            threading.Thread(target=self.gateway.serve_forever, daemon=True).start()
            base = f"http://127.0.0.1:{self.gateway.server_port}/{token}"
            self.environment = service_environment(self.slug, base)
            if include_main:
                build = json.loads((self.images / "build.json").read_text())
                main, self.endpoint = start_service(
                    self.task_id, self.images / "environment.sif",
                    build["images"]["environment"]["workdir"], self.output / "main", self.python,
                    cpus=cpus, environment=self.environment)
                self.processes.append(main)
                self.roots["main"] = self.output / "main/rootfs"
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        for process in reversed(self.processes):
            stop_process(process)
        if self.gateway is not None:
            self.gateway.shutdown()
            self.gateway.server_close()
            self.gateway = None

    def freeze(self, output: Path) -> list[dict[str, Any]]:
        """Collect only official service-qualified paths after services stop."""
        if any(process.poll() is None for process in self.processes):
            raise RuntimeError("Stop every task service before freezing artifacts")
        output.mkdir(parents=True, exist_ok=False)
        records: list[dict[str, Any]] = []
        for index, artifact in enumerate(self.config["artifacts"]):
            service, source = ("main", artifact) if isinstance(artifact, str) else (
                artifact["service"], artifact["source"])
            source_root = self.roots[service]
            stage = self.output / f"frozen-component-{index}"
            items = freeze_artifacts(source_root, [source], stage)
            for item in items:
                item["service"] = service
                if item.get("present"):
                    relative = Path(source).relative_to("/")
                    destination = output / relative
                    if destination.exists():
                        raise ValueError(f"Two service artifacts share destination {source}")
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(stage / relative), destination)
                    item["frozen"] = str(destination)
                records.append(item)
        write_json(output / "artifacts.json", records)
        return records
