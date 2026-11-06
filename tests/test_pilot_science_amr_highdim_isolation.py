from __future__ import annotations
from pathlib import Path
import subprocess
import sys
from evals.pilot import science_amr_isolation as amr
from evals.pilot import science_highdim_isolation as hd


def test_amr_matches_only_uid_dropped_solver_and_preserves_batch_contract(tmp_path: Path) -> None:
    app=tmp_path/"app"; (app/"solver").mkdir(parents=True)
    (app/"solver/amr_poisson3d.py").write_text("pass\n")
    (app/"input.json").write_text("{}\n")
    argv=["/usr/bin/python3","/app/solver/amr_poisson3d.py","--output","/app/out.npz","--solution","/app/result.json","--problems","/app/input.json"]
    kw={"user":60001,"cwd":"/app"}
    assert amr.is_candidate(amr.TASK_ID,argv,kw)
    assert not amr.is_candidate(amr.TASK_ID,["/usr/bin/python3","/tests/oracle_solve3d.py"],kw)
    from evals.pilot import science_isolation
    probe_src=("import sys\np = sys.argv[1]\ntry:\n    open(p, 'rb').read(1)\n"
               "except PermissionError:\n    print('DENIED'); sys.exit(0)\n"
               "except FileNotFoundError:\n    print('ABSENT'); sys.exit(3)\nprint('READABLE'); sys.exit(1)\n")
    official_probe=[sys.executable,"-c",probe_src,"/tests/oracle_solve3d.py"]
    official_probe_kwargs={"user":60001,"stdout":subprocess.PIPE,"stderr":subprocess.STDOUT}
    assert not science_isolation.is_candidate(amr.TASK_ID,official_probe,official_probe_kwargs)
    assert amr.is_privacy_probe(amr.TASK_ID,official_probe,{**official_probe_kwargs,"user":"grader"})
    assert science_isolation.is_candidate(amr.TASK_ID,argv,kw)
    launch=amr.prepare_candidate(amr.TASK_ID,argv,environment={"PATH":"/usr/bin"},cwd="/app",uid=60001,gid=60001,source_root=tmp_path,bwrap="/bin/bwrap")
    try:
        assert launch.command[-len(argv):] == argv
        assert "--unshare-net" in launch.command
        assert "/app/solver" in launch.command
    finally:
        launch.export_outputs(125)


def test_amr_trusted_privacy_probe_uses_permission_mask(tmp_path: Path) -> None:
    probe="import sys\np = sys.argv[1]\ntry:\n    open(p, 'rb').read(1)\nexcept PermissionError:\n    print('DENIED'); sys.exit(0)\nexcept FileNotFoundError:\n    print('ABSENT'); sys.exit(3)\nprint('READABLE'); sys.exit(1)\n"
    argv=["/usr/bin/python3","-c",probe,"/tests/oracle_solve3d.py"]
    command,env=amr.prepare_privacy_probe(argv,environment={"PATH":"/usr/bin:/bin"},uid=60001,gid=60001,bwrap="/bin/bwrap")
    assert "--chmod" in command and "000" in command and "/tests" in command
    result=subprocess.run(command,env=env,capture_output=True,text=True,timeout=15)
    assert result.returncode==0 and result.stdout.strip()=="DENIED",result.stderr


def test_highdim_matches_only_candidate_setpriv_rscript_and_maps_private_files() -> None:
    argv=["setpriv","--reuid=60001","--regid=60001","--clear-groups","--no-new-privs","--","Rscript","/var/lib/tb-hidden/run-1/project/starter_code.R","--input=/var/lib/tb-hidden/run-1/input/cohort.rds","--output=/var/lib/tb-hidden/run-1/output/pred.csv","--seed=42"]
    assert hd.is_candidate(hd.TASK_ID,argv,{"cwd":"/var/lib/tb-hidden/run-1/work","preexec_fn":object()})
    assert not hd.is_candidate(hd.TASK_ID,[*argv[:-4],"/tests/baseline_submission.R",*argv[-3:]],{})
    from evals.pilot import science_isolation
    assert not science_isolation.is_candidate(hd.TASK_ID,["/usr/bin/python3","-c","open(p, 'rb').read(1)","/tests/test_hidden.py"],{"user":60001})
    assert science_isolation.is_candidate(hd.TASK_ID,argv,{})


def test_highdim_stages_only_selected_input_and_exports_one_output(tmp_path: Path) -> None:
    run=tmp_path/"var/lib/tb-hidden/run-1"
    project=run/"project"; inp=run/"input/input.rds"; out=run/"output/pred.csv"
    project.mkdir(parents=True); inp.parent.mkdir(); inp.write_bytes(b"private selected input")
    (project/"starter_code.R").write_text("candidate\n")
    (project/"helper_functions.R").write_text("helper\n")
    argv=["setpriv","--reuid=60001","--regid=60001","--clear-groups","--no-new-privs","--",
          "Rscript","/var/lib/tb-hidden/run-1/project/starter_code.R",
          "--input=/var/lib/tb-hidden/run-1/input/input.rds",
          "--output=/var/lib/tb-hidden/run-1/output/pred.csv","--seed=17"]
    launch=hd.prepare_candidate(argv,environment={"PATH":"/usr/bin:/bin"},
        cwd="/var/lib/tb-hidden/run-1/work",source_root=tmp_path)
    try:
        assert launch.command[-5:]==["Rscript","/root/project/starter_code.R",
            "--input=/home/submission/work/input/input.rds",
            "--output=/home/submission/work/output/output.csv","--seed=17"]
        assert "--unshare-net" in launch.command
        assert "/tests" in launch.command
        assert launch.destination==tmp_path/"var/lib/tb-hidden/run-1/output/pred.csv"
        assert launch.staged_output.parent.is_relative_to(launch.scratch)
    finally:
        launch.finish(125)


def test_generated_bundles_start_with_exact_task_environment(tmp_path: Path) -> None:
    import subprocess
    import sys
    from evals.pilot.science_isolation import create_bundle
    for slug in ("amr-poisson-optimize", "highdim-mediation-debiasing"):
        bundle=tmp_path/slug
        metadata=create_bundle("terminal-bench-science/"+slug,bundle)
        command=["/bin/bwrap","--die-with-parent","--ro-bind",str(bundle),"/opt/relay-isolation",
                 "--ro-bind","/usr","/usr","--ro-bind","/bin","/bin",
                 "--ro-bind","/lib","/lib","--ro-bind","/lib64","/lib64",
                 "--proc","/proc","--dev","/dev","--setenv","PYTHONPATH","/opt/relay-isolation",
                 "--setenv","RELAY_SCIENCE_TASK_ID",metadata["environment"]["RELAY_SCIENCE_TASK_ID"],
                 "--","/usr/bin/python3","-c",
                 "import os,science_isolation; print(science_isolation.task_slug(os.environ['RELAY_SCIENCE_TASK_ID']))"]
        result=subprocess.run(command,capture_output=True,text=True,timeout=20)
        assert result.returncode==0, result.stderr
        assert result.stdout.strip()==slug
