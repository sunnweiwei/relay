"""Isolate Highdim's R submission child while leaving its R verifier trusted."""
from __future__ import annotations
import os, shutil, tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

TASK_ID = "terminal-bench-science/highdim-mediation-debiasing"
TASK_SLUG = "highdim-mediation-debiasing"
BWRAP_PATH = "/opt/relay-tools/bwrap"
SYSTEM_PATHS = ("/bin", "/sbin", "/lib", "/lib64", "/etc", "/usr/bin", "/usr/sbin", "/usr/lib", "/usr/lib64", "/usr/libexec", "/usr/local", "/usr/share")

@dataclass
class CandidateLaunch:
    command: list[str]
    environment: dict[str, str]
    scratch: Path
    staged_output: Path
    destination: Path
    def finish(self, status: int) -> None:
        try:
            if status == 0 and self.staged_output.is_file() and not self.staged_output.is_symlink():
                self.destination.parent.mkdir(parents=True, exist_ok=True)
                tmp = self.destination.with_name(self.destination.name + ".relay-tmp")
                shutil.copyfile(self.staged_output, tmp); os.replace(tmp, self.destination)
        finally:
            shutil.rmtree(self.scratch, ignore_errors=True)

def _unwrap(argv: Sequence[str]) -> tuple[list[str], int, int] | None:
    a = [os.fspath(x) for x in argv]
    if len(a) < 7 or Path(a[0]).name != "setpriv" or not a[1].startswith("--reuid=") or not a[2].startswith("--regid=") or a[3:6] != ["--clear-groups", "--no-new-privs", "--"]:
        return None
    return a[6:], int(a[1].split("=",1)[1]), int(a[2].split("=",1)[1])

def is_candidate(task_id: str, argv: Any, kwargs: Mapping[str, Any]) -> bool:
    if task_id not in {TASK_ID, TASK_SLUG} or not isinstance(argv, (list, tuple)):
        return False
    unwrapped = _unwrap(argv)
    if unwrapped is None: return False
    a, _, _ = unwrapped
    return (len(a) == 5 and Path(a[0]).name == "Rscript" and Path(a[1]).name == "starter_code.R"
            and a[2].startswith("--input=") and a[3].startswith("--output=")
            and a[4].startswith("--seed="))

def prepare_candidate(argv: Sequence[str], *, environment: Mapping[str, str], cwd: str,
                      bwrap: str = BWRAP_PATH, source_root: Path = Path("/")) -> CandidateLaunch:
    unwrapped = _unwrap(argv)
    if unwrapped is None: raise ValueError("Unexpected Highdim setpriv argv")
    args, uid, gid = unwrapped
    if len(args) != 5 or Path(args[0]).name != "Rscript" or Path(args[1]).name != "starter_code.R": raise ValueError("Unexpected Highdim Rscript argv")
    def option(index: int, name: str) -> Path:
        prefix = name + "="
        if not args[index].startswith(prefix): raise ValueError(f"Missing Highdim {name}")
        p=Path(args[index][len(prefix):])
        if not p.is_absolute() or ".." in p.parts: raise ValueError("Noncanonical Highdim path")
        return p
    script, inp, output = Path(args[1]), option(2,"--input"), option(3,"--output")
    runroot=script.parent.parent
    if Path(cwd) != runroot / "work" or script.parent.name != "project" or inp.parent.name != "input" or output.parent.name != "output": raise ValueError("Highdim paths differ from reviewed run layout")
    script_src=source_root/script.relative_to("/"); helper_src=script_src.with_name("helper_functions.R"); input_src=source_root/inp.relative_to("/")
    if any(not p.is_file() or p.is_symlink() for p in (script_src,helper_src,input_src)): raise FileNotFoundError("Highdim candidate/input file missing")
    scratch=Path(tempfile.mkdtemp(prefix="relay-highdim-")); proj=scratch/"project"; proj.mkdir(); outdir=scratch/"output"; outdir.mkdir(mode=0o777); state=scratch/"state"; state.mkdir(mode=0o777); tmpdir=scratch/"tmp"; tmpdir.mkdir(mode=0o777)
    (state/"R").mkdir(mode=0o777)
    for writable in (outdir,state,tmpdir,state/"R"): writable.chmod(0o777)
    try:
        for src in (script_src,helper_src):
            dst=proj/src.name; shutil.copyfile(src,dst); dst.chmod(0o444)
        staged_input=scratch/"input.rds"; shutil.copyfile(input_src,staged_input); staged_input.chmod(0o444)
        rewritten=[args[0],"/root/project/starter_code.R","--input=/home/submission/work/input/input.rds","--output=/home/submission/work/output/output.csv",args[4]]
        env={k:v for k,v in environment.items() if k not in {"LD_PRELOAD","LD_LIBRARY_PATH","PYTHONPATH"} and not k.startswith(("RELAY_","APPTAINER","SINGULARITY","FAKEROOT","FAKED"))}
        env.update({"PATH":"/usr/local/bin:/usr/bin:/bin","HOME":"/home/submission/work/state","TMPDIR":"/home/submission/work/tmp","TMP":"/home/submission/work/tmp","TEMP":"/home/submission/work/tmp","R_ENVIRON_USER":"/dev/null","R_PROFILE_USER":"/dev/null","R_LIBS_USER":"/home/submission/work/state/R"})
        cmd=[bwrap,"--die-with-parent","--unshare-user","--uid",str(uid),"--gid",str(gid),"--unshare-pid","--unshare-ipc","--unshare-uts","--unshare-net","--cap-drop","ALL","--new-session"]
        for p in SYSTEM_PATHS:
            host=source_root/p.lstrip("/")
            if host.is_symlink(): cmd += ["--symlink",os.readlink(host),p]
            elif host.exists(): cmd += ["--ro-bind",str(host),p]
        cmd += ["--proc","/proc","--dev","/dev"]
        for p in ("/root/project","/home/submission/work","/home/submission/work/input","/home/submission/work/output","/home/submission/work/tmp","/home/submission/work/state","/tests","/var/lib/tb-hidden","/tmp","/run"):
            cmd += ["--dir",p]
        cmd += ["--ro-bind",str(proj),"/root/project","--ro-bind",str(staged_input),"/home/submission/work/input/input.rds","--bind",str(outdir),"/home/submission/work/output","--bind",str(tmpdir),"/home/submission/work/tmp","--bind",str(state),"/home/submission/work/state","--remount-ro","/","--remount-ro","/dev","--chdir","/home/submission/work","--",*rewritten]
        return CandidateLaunch(cmd,env,scratch,outdir/"output.csv",source_root/output.relative_to("/"))
    except BaseException:
        shutil.rmtree(scratch,ignore_errors=True); raise
