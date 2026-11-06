"""Run Science submission processes behind a filesystem and PID boundary.

The official verifier and its assertions are unchanged. Its subprocess launch
points select a declarative mount policy, and only public dispatcher scripts
cross from /tests into the child. This module is also copied into a standalone
bundle, so it must depend only on the Python standard library.
"""

from __future__ import annotations

import argparse
import ctypes
import grp
import json
import os
import pwd
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from evals.pilot import dna_codec_isolation
except ModuleNotFoundError:  # Standalone verifier bundle import.
    import dna_codec_isolation  # type: ignore[no-redef]
try:
    from evals.pilot import science_traffic_isolation
except ModuleNotFoundError:  # Standalone verifier bundle import.
    import science_traffic_isolation  # type: ignore[no-redef]
try:
    from evals.pilot import science_lean_isolation
except ModuleNotFoundError:  # Standalone verifier bundle import.
    import science_lean_isolation  # type: ignore[no-redef]
try:
    from evals.pilot import science_regularized_lean_isolation
except ModuleNotFoundError:  # Standalone verifier bundle import.
    import science_regularized_lean_isolation  # type: ignore[no-redef]
try:
    from evals.pilot import science_reactor_isolation
except ModuleNotFoundError:  # Standalone verifier bundle import.
    import science_reactor_isolation  # type: ignore[no-redef]
try:
    from evals.pilot import science_localization_isolation
except ModuleNotFoundError:  # Standalone verifier bundle import.
    import science_localization_isolation  # type: ignore[no-redef]
try:
    from evals.pilot import science_amr_isolation, science_highdim_isolation
except ModuleNotFoundError:  # Standalone verifier bundle import.
    import science_amr_isolation  # type: ignore[no-redef]
    import science_highdim_isolation  # type: ignore[no-redef]
try:
    from evals.pilot import science_genomic_isolation
except ModuleNotFoundError:  # Standalone verifier bundle import.
    import science_genomic_isolation  # type: ignore[no-redef]
try:
    from evals.pilot import science_task_child_isolation
except ModuleNotFoundError:  # Standalone verifier bundle import.
    import science_task_child_isolation  # type: ignore[no-redef]
try:
    from evals.pilot import science_stacking_isolation
except ModuleNotFoundError:  # Standalone verifier bundle import.
    import science_stacking_isolation  # type: ignore[no-redef]
try:
    from evals.pilot import science_worker_process_isolation
except ModuleNotFoundError:  # Standalone verifier bundle import.
    import science_worker_process_isolation  # type: ignore[no-redef]


try:
    from evals.pilot import science_protein_isolation
except ModuleNotFoundError:
    import science_protein_isolation  # type: ignore[no-redef]
try:
    from evals.pilot import science_duan_isolation
except ModuleNotFoundError:
    import science_duan_isolation  # type: ignore[no-redef]
try:
    from evals.pilot import science_navigation_small_area_isolation
    from evals.pilot import science_leaky_bloch_isolation
except ModuleNotFoundError:
    import science_navigation_small_area_isolation  # type: ignore[no-redef]
    import science_leaky_bloch_isolation  # type: ignore[no-redef]


BUNDLE_PATH = "/opt/relay-isolation"
BWRAP_PATH = "/opt/relay-tools/bwrap"
RUNTIME_PYTHON = "/opt/relay-runtime/bin/python3"
TASKS = frozenset({
    "inverse-waveguide-shape", "localized-sspd-solver", "hbv-calibration-1",
    "symbolic-regression", "ont-tn-qc", "dna-storage-codec", "traffic-flux-inversion",
    "reactor-safety-control", "baseline-free-localization", "guided-wave-localization",
    "virtual-baseline-localization",
    "stacking-disorder-diffraction",
    "genomic-model-ranking", "linked-cell-suppression", "tess-transit-vetting",
    "foraging-cognitive-model", "noisy-blackbox-optimization",
    "navigation-sensor-calibration", "small-area-equivalence", "leaky-bloch-meep",
})
# Deliberately omit /usr/project, /home, /root, /opt, /run and /var. In
# particular, neither a host bind nor a private verifier file is inherited.
SYSTEM_PATHS = (
    "/bin", "/sbin", "/lib", "/lib64", "/etc", "/usr/bin", "/usr/sbin",
    "/usr/lib", "/usr/lib64", "/usr/libexec", "/usr/local", "/usr/share",
    "/usr/include",
)
_ORIGINAL_POPEN = subprocess.Popen


def task_slug(task_id: str) -> str:
    if task_id in {"microarch-modeling", "cmb-cross-inference"}:
        return task_id
    slug = task_id.rsplit("/", 1)[-1]
    # Every task in the pinned Science-70 population can use the verifier
    # bundle. Submitted-code execution still fails closed below until a
    # task-specific candidate interface has been audited and added.
    canonical = (task_id.startswith("terminal-bench-science/") and
                 task_id.count("/") == 1 and slug and
                 all(character.islower() or character.isdigit() or character == "-"
                     for character in slug))
    if slug not in TASKS and not canonical:
        raise ValueError(f"No audited candidate interface for Science task {task_id!r}")
    return slug


def create_bundle(task_id: str, output_dir: Path | str) -> dict[str, Any]:
    """Create the trusted bundle to bind read-only at ``BUNDLE_PATH``.

    The caller supplies a separate verifier image, writable overlay, private
    outer network with loopback, and a real /dev tmpfs. No hidden benchmark
    data is copied into this bundle.
    """
    slug = task_slug(task_id)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "bin").mkdir(exist_ok=True)
    shutil.copyfile(__file__, output / "science_isolation.py")
    shutil.copyfile(science_protein_isolation.__file__, output / "science_protein_isolation.py")
    shutil.copyfile(science_duan_isolation.__file__, output / "science_duan_isolation.py")
    shutil.copyfile(science_navigation_small_area_isolation.__file__,
                    output / "science_navigation_small_area_isolation.py")
    shutil.copyfile(science_leaky_bloch_isolation.__file__,
                    output / "science_leaky_bloch_isolation.py")
    shutil.copyfile(dna_codec_isolation.__file__, output / "dna_codec_isolation.py")
    shutil.copyfile(science_traffic_isolation.__file__, output / "science_traffic_isolation.py")
    shutil.copyfile(science_lean_isolation.__file__, output / "science_lean_isolation.py")
    shutil.copyfile(science_regularized_lean_isolation.__file__, output / "science_regularized_lean_isolation.py")
    shutil.copyfile(science_reactor_isolation.__file__, output / "science_reactor_isolation.py")
    shutil.copyfile(science_localization_isolation.__file__, output / "science_localization_isolation.py")
    shutil.copyfile(science_genomic_isolation.__file__, output / "science_genomic_isolation.py")
    shutil.copyfile(science_task_child_isolation.__file__, output / "science_task_child_isolation.py")
    shutil.copyfile(science_stacking_isolation.__file__, output / "science_stacking_isolation.py")
    shutil.copyfile(science_worker_process_isolation.__file__, output / "science_worker_process_isolation.py")
    shutil.copyfile(science_amr_isolation.__file__, output / "science_amr_isolation.py")
    shutil.copyfile(science_highdim_isolation.__file__, output / "science_highdim_isolation.py")
    if slug in {"microarch-modeling", "cmb-cross-inference"}:
        from evals.pilot import science_onnx_isolation

        shutil.copyfile(science_onnx_isolation.__file__, output / "science_onnx_isolation.py")
    # sitecustomize errors normally only print a warning. Fail closed instead.
    (output / "sitecustomize.py").write_text(
        "import os\ntry:\n"
        "    from science_isolation import install\n"
        "    install()\nexcept BaseException as error:\n"
        "    os.write(2, ('Science isolation initialization failed: ' + str(error) + '\\n').encode())\n"
        "    os._exit(125)\n"
    )
    runuser = output / "bin/runuser"
    runuser.write_text(
        f"#!{RUNTIME_PYTHON}\n"
        "import sys\n"
        f"sys.path.insert(0, {BUNDLE_PATH!r})\n"
        "from science_isolation import runuser_main\n"
        "raise SystemExit(runuser_main(sys.argv[1:]))\n"
    )
    runuser.chmod(0o755)
    if slug == "regularized-game-proof":
        lake = output / "bin/lake"
        lake.write_text(
            f"#!{RUNTIME_PYTHON}\n"
            "import sys\n"
            f"sys.path.insert(0, {BUNDLE_PATH!r})\n"
            "from science_regularized_lean_isolation import lake_wrapper_main\n"
            "raise SystemExit(lake_wrapper_main(sys.argv[1:]))\n"
        )
        lake.chmod(0o755)
    if slug == "duan-thesis":
        science_duan_isolation.install_bundle(output)
    metadata = {
        "task": slug, "bundle_dir": str(output), "mount_path": BUNDLE_PATH,
        "environment": {
            "PYTHONPATH": BUNDLE_PATH,
            "PATH": (f"{BUNDLE_PATH}/bin:/root/.elan/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
                     if slug == "regularized-game-proof" else
                     f"{BUNDLE_PATH}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"),
            "RELAY_SCIENCE_TASK_ID": f"terminal-bench-science/{slug}",
        },
        "requirements": {
            "bwrap": BWRAP_PATH, "python": RUNTIME_PYTHON,
            "outer_network": "private loopback, no egress",
            "outer_filesystem": "writable overlay and memory-backed /dev",
        },
    }
    (output / "bundle.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def child_environment(environment: Mapping[str, str]) -> dict[str, str]:
    """Remove the trusted adapters and fakeroot transport from a child."""
    result = {}
    for key, value in environment.items():
        if key.startswith(("RELAY_", "APPTAINER", "SINGULARITY", "FAKEROOT", "FAKED")):
            continue
        if key in {"LD_PRELOAD", "LD_LIBRARY_PATH", "PYTHONHOME"}:
            continue
        if key in {"PATH", "PYTHONPATH"}:
            value = ":".join(part for part in value.split(":")
                             if part and not part.startswith((BUNDLE_PATH, "/opt/relay-tools")))
            if not value:
                continue
        result[key] = value
    return result


def _argument(argv: Sequence[str], flag: str) -> str:
    try:
        value = argv[argv.index(flag) + 1]
    except (ValueError, IndexError) as error:
        raise ValueError(f"Missing public candidate argument {flag}") from error
    if not value.startswith("/") or ".." in Path(value).parts:
        raise ValueError(f"Candidate argument {flag} must be an absolute normalized path")
    return value


def _under(path: str, parent: str) -> str:
    if not path.startswith("/") or ".." in Path(path).parts:
        raise ValueError(f"Non-normalized candidate path {path!r}")
    Path(path).relative_to(parent)
    return path


def candidate_command(
    task_id: str, argv: Sequence[str], *, environment: Mapping[str, str],
    cwd: str | None = None, uid: int = 60001, gid: int = 60001,
    bwrap: str = BWRAP_PATH, source_root: Path = Path("/"),
) -> tuple[list[str], dict[str, str]]:
    """Build one child boundary while retaining the official argv and cwd.

    ``source_root`` supports model-free fixtures; production calls leave it at
    /. Runtime system paths always come from the current verifier image.
    """
    slug = task_slug(task_id)
    args = [os.fspath(item) for item in argv]
    readable: list[str] = []
    writable: list[str] = []
    scratch = False
    if slug == "inverse-waveguide-shape":
        readable.append(_under(_argument(args, "--data-dir"), "/app"))
        output = _under(_argument(args, "--output"), "/app/waveguide_case_results")
        writable += ["/app/waveguide_submission", "/app/waveguide_solver_home", str(Path(output).parent)]
    elif slug == "localized-sspd-solver":
        readable += ["/app/solver", "/app/sdk", "/app/protocol.md"]
        writable.append(_under(environment["HOME"], "/dev/tbs-solver-scratch"))
    elif slug == "symbolic-regression":
        index = args.index("/tests/_sr_runner.py")
        if len(args) != index + 3:
            raise ValueError("Unexpected symbolic-regression dispatcher arguments")
        readable += ["/tests/_sr_runner.py", "/app/regressor.py", "/app/training_data.npz",
                     _under(args[index + 1], "/tmp")]
        writable.append(_under(args[index + 2], "/tmp"))
        scratch = True
    elif slug == "hbv-calibration-1":
        index = args.index("/tests/run_optimizer.R")
        if len(args) != index + 3:
            raise ValueError("Unexpected HBV dispatcher arguments")
        readable += ["/tests/run_optimizer.R", "/results"]
        for path in args[index + 1:]:
            _under(path, "/tmp/hbv-optimizer")
        writable.append("/tmp/hbv-optimizer")
    else:
        raise ValueError(f"{slug} has no submitted-code execution interface")

    command = [
        bwrap, "--unshare-user", "--uid", str(uid), "--gid", str(gid),
        "--unshare-pid", "--unshare-ipc", "--unshare-uts", "--cap-drop", "ALL",
        "--die-with-parent", "--new-session",
    ]
    for path in SYSTEM_PATHS:
        if Path(path).is_symlink():
            command += ["--symlink", os.readlink(path), path]
        elif Path(path).exists():
            command += ["--ro-bind", path, path]
    # A new procfs prevents /proc/<parent>/root, fd, and memory access. The
    # bwrap PID 1 reaper kills detached descendants when the main child exits.
    command += ["--proc", "/proc", "--dev", "/dev"]
    for path in ("/app", "/tests", "/logs", "/results", "/tmp", "/var/tmp", "/run/lock"):
        command += ["--dir", path]
    if scratch:
        command += ["--tmpfs", "/tmp"]
    for path in readable:
        source = source_root / path.lstrip("/")
        if not source.exists():
            raise FileNotFoundError(f"Missing public candidate input: {path}")
        command += ["--ro-bind", str(source), path]
    for path in writable:
        source = source_root / path.lstrip("/")
        if not source.exists():
            raise FileNotFoundError(f"Missing candidate output/scratch path: {path}")
        command += ["--bind", str(source), path]
    # Lock the root and device mount, but retain explicitly writable submounts.
    # This also prevents UID remapping from bypassing official chmod policies.
    command += ["--remount-ro", "/", "--remount-ro", "/dev"]
    if cwd is not None:
        command += ["--chdir", cwd]
    command += ["--", *args]
    return command, child_environment(environment)


def is_candidate(task_id: str, argv: Any, kwargs: Mapping[str, Any]) -> bool:
    """Recognize UID drops and the explicit public plain-process bridge."""
    slug = task_slug(task_id)
    if science_stacking_isolation.is_reference_command(task_id, argv, kwargs):
        if not science_stacking_isolation.is_verified_reference(task_id, argv, kwargs):
            raise ValueError("Stacking verifier reference launch failed its pinned source hash")
        return True
    if science_protein_isolation.is_candidate(task_id, argv, kwargs):
        return True
    if science_reactor_isolation.is_candidate(task_id, argv, kwargs):
        return True
    if science_lean_isolation.is_candidate(task_id, argv, kwargs):
        return True
    if science_localization_isolation.is_candidate(task_id, argv, kwargs):
        return True
    if science_genomic_isolation.is_candidate(task_id, argv, kwargs):
        return True
    if science_task_child_isolation.is_candidate(task_id, argv, kwargs):
        return True
    if science_stacking_isolation.is_candidate(task_id, argv, kwargs):
        return True
    if science_worker_process_isolation.is_candidate(task_id, argv, kwargs):
        return True
    if science_navigation_small_area_isolation.is_candidate(task_id, argv, kwargs):
        return True
    if science_leaky_bloch_isolation.is_candidate(task_id, argv, kwargs):
        return True
    if science_amr_isolation.is_candidate(task_id, argv, kwargs):
        return True
    if science_highdim_isolation.is_candidate(task_id, argv, kwargs):
        return True
    if task_slug(task_id) in {science_amr_isolation.TASK_SLUG,
                              science_highdim_isolation.TASK_SLUG}:
        # These profiles have exact candidate command matchers. The verifier
        # also runs UID-dropped privacy probes that must stay in its trusted
        # test process rather than being rewritten as candidate launches.
        return False
    if slug in science_task_child_isolation.TASKS:
        return False
    if slug in science_stacking_isolation.TASKS:
        # The only solver launch is matched by exact path, argv, environment,
        # UID/GID and process-session settings. The reference solver remains
        # trusted only when its copied source hash is pinned and verified.
        return False
    if kwargs.get("user") is not None or kwargs.get("group") is not None:
        return True
    return (slug == "symbolic-regression" and isinstance(argv, (list, tuple))
            and "/tests/_sr_runner.py" in argv)


def prepare_namespace() -> None:
    """Make an Apptainer overlay bindable inside its private mount namespace.

    Apptainer marks its fuse-overlay root unbindable, which rejects nested
    bind mounts with EINVAL. Changing mount propagation to private preserves
    its contents, permissions and mount options. This runs only in the trusted
    verifier namespace; no host mount is exposed to the candidate.
    """
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        fields = line.split()
        if fields[4] == "/" and "unbindable" in fields[6:fields.index("-")]:
            libc = ctypes.CDLL(None, use_errno=True)
            ms_private, ms_rec = 1 << 18, 1 << 14
            if libc.mount(None, b"/", None, ctypes.c_ulong(ms_private | ms_rec), None) != 0:
                error = ctypes.get_errno()
                raise OSError(error, "Cannot prepare the private verifier mount namespace")
            return


def _account_id(value: Any, *, group: bool) -> int:
    """Resolve subprocess user/group arguments before entering bwrap."""
    if value is None:
        return 60001
    if isinstance(value, int) or str(value).isdecimal():
        return int(value)
    return grp.getgrnam(str(value)).gr_gid if group else pwd.getpwnam(str(value)).pw_uid


def install() -> None:
    """Intercept only candidate launches in the trusted Python verifier."""
    task_id = os.environ.get("RELAY_SCIENCE_TASK_ID")
    if not task_id or getattr(subprocess.Popen, "_relay_science_isolation", False):
        return
    task_slug(task_id)
    if task_slug(task_id) in {"microarch-modeling", "cmb-cross-inference"}:
        try:
            from evals.pilot import science_onnx_isolation
        except ModuleNotFoundError:  # Standalone verifier bundle import.
            import science_onnx_isolation  # type: ignore[no-redef]

        science_onnx_isolation.install(task_id)
    if task_slug(task_id) == "traffic-flux-inversion":
        science_traffic_isolation.install(task_id)
    prepare_namespace()

    class IsolatedPopen(_ORIGINAL_POPEN):
        _relay_science_isolation = True

        def __init__(self, args: Any, *positional: Any, **kwargs: Any) -> None:
            self._isolated_candidate = is_candidate(task_id, args, kwargs)
            self._dna_codec_launch: dna_codec_isolation.CandidateLaunch | None = None
            amr_launch: science_amr_isolation.CandidateLaunch | None = None
            highdim_launch: science_highdim_isolation.CandidateLaunch | None = None
            localization_launch: science_localization_isolation.WorkerLaunch | None = None
            genomic_launch: science_genomic_isolation.CandidateLaunch | None = None
            worker_launch: science_worker_process_isolation.WorkerLaunch | None = None
            nav_launch: science_navigation_small_area_isolation.CandidateLaunch | None = None
            nav_probe_launch: science_navigation_small_area_isolation.CandidateLaunch | None = None
            leaky_launch: science_leaky_bloch_isolation.CandidateLaunch | None = None
            if science_amr_isolation.is_privacy_probe(task_id, args, kwargs):
                environment = kwargs.get("env") or os.environ
                requested_user = kwargs.pop("user", None)
                requested_group = kwargs.pop("group", None)
                uid = _account_id(requested_user, group=False)
                gid = (_account_id(requested_group, group=True) if requested_group is not None
                       else pwd.getpwuid(uid).pw_gid)
                kwargs.pop("extra_groups", None)
                args, environment = science_amr_isolation.prepare_privacy_probe(
                    args, environment=environment, uid=uid, gid=gid,
                )
                kwargs["env"] = environment
                kwargs["close_fds"] = True
                kwargs["cwd"] = None
            elif science_navigation_small_area_isolation.is_privacy_probe(task_id, args, kwargs):
                account = pwd.getpwnam("solveruser")
                nav_probe_launch = science_navigation_small_area_isolation.prepare_privacy_probe(
                    args, environment=kwargs.get("env") or os.environ,
                    uid=account.pw_uid, gid=account.pw_gid,
                )
                args, environment = nav_probe_launch.command, nav_probe_launch.environment
                kwargs["env"] = environment
                kwargs["preexec_fn"] = None
                kwargs["pass_fds"] = tuple(sorted(set(kwargs.get("pass_fds", ())) |
                                                   set(nav_probe_launch.pass_fds)))
                kwargs["close_fds"] = True
                kwargs["cwd"] = None
            elif self._isolated_candidate:
                if positional or kwargs.get("shell") or kwargs.get("executable"):
                    raise ValueError("Unsupported candidate subprocess launch form")
                environment = kwargs.get("env") or os.environ
                uid = kwargs.pop("user", None)
                gid = kwargs.pop("group", None)
                kwargs.pop("extra_groups", None)
                if science_protein_isolation.is_candidate(task_id, args, kwargs):
                    args, environment = science_protein_isolation.prepare_candidate(
                        args, environment, new_session=not kwargs.get("start_new_session", False))
                    kwargs.pop("preexec_fn", None)
                elif science_task_child_isolation.is_candidate(task_id, args, kwargs):
                    if science_task_child_isolation.task_slug(task_id) == "linked-cell-suppression":
                        child_launch = science_task_child_isolation.prepare_linked_cell(
                            task_id, args, environment=environment, cwd=kwargs.get("cwd"),
                            preexec_fn=kwargs.get("preexec_fn"),
                        )
                    else:
                        child_launch = science_task_child_isolation.prepare_tess(
                            task_id, args, environment=environment, cwd=kwargs.get("cwd"),
                            preexec_fn=kwargs.get("preexec_fn"),
                        )
                    args, environment = child_launch.command, child_launch.environment
                    kwargs.pop("preexec_fn", None)
                    kwargs["start_new_session"] = True
                elif science_stacking_isolation.is_verified_reference(
                        task_id, args, {"user": uid, "group": gid,
                                        "extra_groups": (), **kwargs}):
                    stacking_launch = science_stacking_isolation.prepare_reference(
                        task_id, args, environment=environment, cwd=kwargs.get("cwd"),
                        uid=_account_id(uid, group=False), gid=_account_id(gid, group=True),
                    )
                    args, environment = stacking_launch.command, stacking_launch.environment
                elif science_stacking_isolation.is_candidate(
                        task_id, args, {"user": uid, "group": gid,
                                        "extra_groups": (), **kwargs}):
                    stacking_launch = science_stacking_isolation.prepare_candidate(
                        task_id, args, environment=environment, cwd=kwargs.get("cwd"),
                        uid=_account_id(uid, group=False), gid=_account_id(gid, group=True),
                    )
                    args, environment = stacking_launch.command, stacking_launch.environment
                elif science_reactor_isolation.is_candidate(task_id, args, kwargs):
                    if positional or kwargs.get("shell") or kwargs.get("executable"):
                        raise ValueError("Unsupported Reactor controller subprocess launch form")
                    reactor_launch = science_reactor_isolation.prepare_candidate(
                        task_id, args, environment=environment, cwd=kwargs.get("cwd"),
                    )
                    args, environment = reactor_launch.command, reactor_launch.environment
                    # The official preexec function drops UID/GID and starts a
                    # session. The bwrap namespace and its child bootstrap
                    # reproduce those effects before importing candidate code.
                    kwargs.pop("preexec_fn", None)
                elif dna_codec_isolation.is_candidate(task_id, args, {"user": uid, "group": gid}):
                    self._dna_codec_launch = dna_codec_isolation.prepare_candidate(
                        task_id, args, environment=environment, cwd=kwargs.get("cwd"),
                        uid=_account_id(uid, group=False),
                        gid=_account_id(gid, group=True),
                    )
                    args, environment = (self._dna_codec_launch.command,
                                         self._dna_codec_launch.environment)
                elif science_lean_isolation.is_candidate(task_id, args, kwargs):
                    cwd = kwargs.get("cwd")
                    if cwd is None:
                        raise ValueError("Lean candidate command is missing its audited project cwd")
                    lean_launch = science_lean_isolation.prepare_candidate(
                        task_id, args, environment=environment, cwd=cwd,
                    )
                    args, environment = lean_launch.command, lean_launch.environment
                elif science_localization_isolation.is_candidate(task_id, args, kwargs):
                    account = pwd.getpwnam("solveruser") if uid is None and gid is None else None
                    localization_launch = science_localization_isolation.build_worker_command(
                        task_id=task_id, argv=args, environment=environment,
                        cwd=kwargs.get("cwd"),
                        bubblewrap_new_session=kwargs.get("start_new_session") is not True,
                        uid=account.pw_uid if account is not None else _account_id(uid, group=False),
                        gid=account.pw_gid if account is not None else _account_id(gid, group=True),
                    )
                    args, environment = localization_launch.command, localization_launch.environment
                    # The official preexec also drops UID/GID, but running it
                    # on bubblewrap applies RLIMIT_NPROC before namespace
                    # setup and prevents bubblewrap from starting. The shim
                    # applies the same limits after isolation; bubblewrap sets
                    # the equivalent solveruser IDs before importing code.
                    kwargs["preexec_fn"] = None
                    kwargs["pass_fds"] = tuple(sorted(set(kwargs.get("pass_fds", ())) |
                                                       set(localization_launch.pass_fds)))
                elif science_genomic_isolation.is_candidate(task_id, args, kwargs):
                    genomic_launch = science_genomic_isolation.build_candidate_command(
                        task_id=task_id, argv=args, environment=environment,
                        cwd=kwargs.get("cwd"), uid=65534, gid=65534,
                    )
                    args, environment = genomic_launch.command, genomic_launch.environment
                    kwargs["preexec_fn"] = None
                    kwargs["pass_fds"] = tuple(sorted(set(kwargs.get("pass_fds", ())) |
                                                       set(genomic_launch.pass_fds)))
                elif science_worker_process_isolation.is_candidate(task_id, args, kwargs):
                    slug = science_worker_process_isolation.task_slug(task_id)
                    if slug == "noisy-blackbox-optimization":
                        account = pwd.getpwnam("solveruser")
                        worker_uid, worker_gid = account.pw_uid, account.pw_gid
                    else:
                        worker_uid, worker_gid = 60001, 60001
                    worker_launch = science_worker_process_isolation.build_worker_command(
                        task_id=task_id, argv=args, environment=environment,
                        cwd=kwargs.get("cwd"), uid=worker_uid, gid=worker_gid,
                    )
                    args, environment = worker_launch.command, worker_launch.environment
                    # The Noisy official preexec applies limits before exec.
                    # Move those exact limits into the trusted dispatcher after
                    # bubblewrap has completed namespace setup.
                    kwargs["preexec_fn"] = None
                    kwargs["pass_fds"] = tuple(sorted(set(kwargs.get("pass_fds", ())) |
                                                       set(worker_launch.pass_fds)))
                elif science_navigation_small_area_isolation.is_candidate(task_id, args, kwargs):
                    slug = science_navigation_small_area_isolation.task_slug(task_id)
                    if uid is None and gid is None:
                        if slug == "navigation-sensor-calibration":
                            account = pwd.getpwnam("solveruser")
                            worker_uid, worker_gid = account.pw_uid, account.pw_gid
                        else:
                            worker_uid, worker_gid = 65534, 65534
                    else:
                        worker_uid = _account_id(uid, group=False)
                        worker_gid = _account_id(gid, group=True)
                    nav_launch = science_navigation_small_area_isolation.build_candidate_command(
                        task_id=task_id, argv=args, kwargs=kwargs,
                        uid=worker_uid, gid=worker_gid,
                    )
                    args, environment = nav_launch.command, nav_launch.environment
                    kwargs["preexec_fn"] = None
                    kwargs["pass_fds"] = tuple(sorted(set(kwargs.get("pass_fds", ())) |
                                                       set(nav_launch.pass_fds)))
                elif science_leaky_bloch_isolation.is_candidate(task_id, args, kwargs):
                    leaky_launch = science_leaky_bloch_isolation.build_candidate_command(
                        task_id=task_id, argv=args, kwargs=kwargs,
                    )
                    args, environment = leaky_launch.command, leaky_launch.environment
                    kwargs["pass_fds"] = tuple(sorted(set(kwargs.get("pass_fds", ())) |
                                                       set(leaky_launch.pass_fds)))
                elif science_amr_isolation.is_candidate(task_id, args, {"user": uid, "group": gid, **kwargs}):
                    if kwargs.get("preexec_fn") is not None:
                        raise ValueError("Unsupported AMR verifier preexec policy")
                    amr_launch = science_amr_isolation.prepare_candidate(
                        task_id, args, environment=environment, cwd=kwargs.get("cwd"),
                        uid=_account_id(uid, group=False), gid=_account_id(gid, group=True),
                    )
                    args, environment = amr_launch.command, amr_launch.environment
                elif science_highdim_isolation.is_candidate(task_id, args, {"user": uid, "group": gid, **kwargs}):
                    if kwargs.get("preexec_fn") is None or kwargs.get("start_new_session") is not True:
                        raise ValueError("Highdim candidate launch lost its reviewed resource limits")
                    highdim_launch = science_highdim_isolation.prepare_candidate(
                        args, environment=environment, cwd=kwargs.get("cwd"),
                    )
                    args, environment = highdim_launch.command, highdim_launch.environment
                    # setpriv is reproduced by the mapped UID/GID. Keeping the
                    # official preexec preserves its exact resource limits.
                else:
                    args, environment = candidate_command(
                        task_id, args, environment=environment, cwd=kwargs.get("cwd"),
                        uid=_account_id(uid, group=False),
                        gid=_account_id(gid, group=True),
                    )
                kwargs["env"] = environment
                kwargs["close_fds"] = True
                # The requested cwd is recreated inside the namespace by
                # candidate_command/prepare_candidate's --chdir. Applying it
                # to the host-side bubblewrap launcher would fail for paths
                # such as /app that exist only in the verifier image.
                kwargs["cwd"] = None
            try:
                self._amr_launch = amr_launch
                self._highdim_launch = highdim_launch
                self._nav_launch = nav_launch
                super().__init__(args, *positional, **kwargs)
            except BaseException:
                if self._dna_codec_launch is not None:
                    self._dna_codec_launch.export_output(125)
                    self._dna_codec_launch = None
                if amr_launch is not None:
                    amr_launch.export_outputs(125)
                if highdim_launch is not None:
                    highdim_launch.finish(125)
                raise
            finally:
                if localization_launch is not None:
                    localization_launch.close()
                if genomic_launch is not None:
                    genomic_launch.close()
                if worker_launch is not None:
                    worker_launch.close()
                if nav_launch is not None:
                    nav_launch.close()
                if leaky_launch is not None:
                    leaky_launch.close()
                if nav_probe_launch is not None:
                    nav_probe_launch.close()

        def _finish_dna_codec(self) -> None:
            launch = self._dna_codec_launch
            if launch is not None and self.returncode is not None:
                self._dna_codec_launch = None
                launch.export_output(self.returncode)

        def _finish_task_adapter(self) -> None:
            if self.returncode is None:
                return
            launch = getattr(self, "_amr_launch", None)
            if launch is not None:
                self._amr_launch = None
                launch.export_outputs(self.returncode)
            launch = getattr(self, "_highdim_launch", None)
            if launch is not None:
                self._highdim_launch = None
                launch.finish(self.returncode)

        def _finish_navigation_output(self) -> None:
            launch = getattr(self, "_nav_launch", None)
            if launch is not None and self.returncode is not None:
                self._nav_launch = None
                launch.export_output_ownership(self.returncode)

        def poll(self) -> int | None:
            result = super().poll()
            self._finish_dna_codec()
            self._finish_task_adapter()
            self._finish_navigation_output()
            return result

        def wait(self, *args: Any, **kwargs: Any) -> int:
            try:
                return super().wait(*args, **kwargs)
            finally:
                self._finish_dna_codec()
                self._finish_task_adapter()
                self._finish_navigation_output()

        def communicate(self, *args: Any, **kwargs: Any) -> Any:
            try:
                result = super().communicate(*args, **kwargs)
                self._finish_dna_codec()
                self._finish_task_adapter()
                self._finish_navigation_output()
                return result
            except subprocess.TimeoutExpired:
                # The official cleanup scans outer /proc by UID. Namespace
                # remapping hides that UID there, so reap the namespace before
                # its handler retries communicate(). Keep the same timeout
                # exception and all official scoring assertions.
                if self._isolated_candidate and self.poll() is None:
                    self.kill()
                raise

    subprocess.Popen = IsolatedPopen


def runuser_main(argv: Sequence[str]) -> int:
    """Honor the official runuser argv through a mapped bwrap identity."""
    if len(argv) < 4 or argv[0] != "-u" or argv[2] != "--":
        raise ValueError("Unsupported Science runuser invocation")
    prepare_namespace()
    account = pwd.getpwnam(argv[1])
    args, environment = candidate_command(
        os.environ["RELAY_SCIENCE_TASK_ID"], argv[3:], environment=os.environ,
        cwd=os.getcwd(), uid=account.pw_uid, gid=account.pw_gid,
    )
    os.execve(args[0], args, environment)
    return 125


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(create_bundle(args.task, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
