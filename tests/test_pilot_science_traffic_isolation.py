"""Model-free controls for traffic solver JSON IPC and filesystem isolation."""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from evals.pilot import science_isolation
from evals.pilot import science_traffic_isolation as traffic


@pytest.fixture
def fixture_root(tmp_path: Path) -> Path:
    app = tmp_path / "app"
    work = tmp_path / "work"
    app.mkdir()
    work.mkdir()
    private = tmp_path / "verifier-secret"
    private.write_text("never visible")
    candidate = f"""
import numpy as np
from pathlib import Path

CALLS = 0
PRIVATE = {str(private)!r}
def _check_hidden():
    for path in ['/tests/manifest.json', '/tests/private', '/logs/verifier/reward.txt',
                 '/work/verifier-private.npz', PRIVATE, '/proc/1/root' + PRIVATE]:
        if Path(path).exists():
            raise RuntimeError('hidden path was visible')

def fit_model(episode_file):
    global CALLS
    _check_hidden()
    CALLS += 1
    data = np.load(episode_file, allow_pickle=False)
    Path('/work/candidate-sidecar').write_text('private worker scratch')
    return {{'form': 'json-dict', 'v': np.float64(data['value']),
             'weights': np.asarray([1.0, 2.0], dtype=np.float32), 'calls': CALLS}}

def predict_scenario(model, init, left, right, steps, dt, dx):
    _check_hidden()
    assert isinstance(model, dict)
    assert isinstance(model['v'], np.float64)
    assert model['weights'].dtype == np.float32
    assert (steps, dt, dx) == (2, 0.25, 1.5)
    return np.asarray(init, dtype=np.float64) + np.zeros_like(left) + np.zeros_like(right) + model['v'] * 0.0
"""
    (app / "traffic_solver.py").write_text(candidate)
    (app / "flux_models.py").write_text("PUBLIC_HELPER = True\n")
    np.savez(work / "system_0_episodes.npz", value=np.array(3.5, dtype=np.float64))
    np.savez(work / "inputs.npz", public=np.array([1, 2, 3]))
    (work / "verifier-private.npz").write_text("must not enter the worker copy")
    return tmp_path


def make_proxy(root: Path) -> traffic.TrafficFunctionProxy:
    return traffic.TrafficFunctionProxy(
        task_id=traffic.TASK_ID, source_root=root,
        worker_script=Path(traffic.__file__), bwrap=shutil.which("bwrap") or "/bin/bwrap",
        python="/usr/bin/python3",
        environment={"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/tmp"},
    )


def test_numeric_json_codec_preserves_array_dtype_shape_and_nonfinite_values() -> None:
    value = {
        "form": "test", "v": np.float64(3.25),
        "state": np.array([[1.0, np.nan], [np.inf, -np.inf]], dtype=np.float32),
        "ids": (np.int16(2), np.int64(9)),
    }
    decoded = traffic._decode(json.loads(traffic._json_line(traffic._encode(value))))
    assert decoded["form"] == "test"
    assert isinstance(decoded["v"], np.float64)
    assert decoded["state"].dtype == np.float32
    assert decoded["state"].shape == (2, 2)
    assert np.isnan(decoded["state"][0, 1]) and np.isposinf(decoded["state"][1, 0])
    assert isinstance(decoded["ids"][0], np.int16)


def test_exact_function_proxy_runs_candidate_in_isolated_persistent_worker(fixture_root: Path) -> None:
    if shutil.which("bwrap") is None:
        pytest.skip("bubblewrap is unavailable")
    proxy = make_proxy(fixture_root)
    try:
        model = proxy.fit_model("/work/system_0_episodes.npz")
        assert isinstance(model, dict)
        assert {'form', 'v'} <= set(model)
        assert str(model['form']) == 'json-dict'
        assert float(model['v']) == 3.5
        assert isinstance(model, traffic._TrafficModelView)
        init = np.array([[0.2, 0.3], [0.4, 0.5]], dtype=np.float64)
        left = np.array([[1.0, 2.0]], dtype=np.float64)
        right = np.array([[3.0, 4.0]], dtype=np.float64)
        predicted = proxy.predict_scenario(model, init, left, right, 2, 0.25, 1.5)
        assert predicted.dtype == np.float64
        assert predicted.shape == init.shape
        np.testing.assert_array_equal(predicted, init)
        next_model = proxy.fit_model("/work/system_0_episodes.npz")
        with pytest.raises(TypeError, match="requires the active model"):
            proxy.predict_scenario(model, init, left, right, 2, 0.25, 1.5)
        assert proxy.predict_scenario(next_model, init, left, right, 2, 0.25, 1.5).shape == init.shape
        other_proxy = make_proxy(fixture_root)
        try:
            with pytest.raises(TypeError, match="requires the active model"):
                other_proxy.predict_scenario(next_model, init, left, right, 2, 0.25, 1.5)
        finally:
            other_proxy.close()
    finally:
        proxy.close()
    assert not (fixture_root / "work/candidate-sidecar").exists()
    assert (fixture_root / "work/system_0_episodes.npz").is_file()


def test_rejects_paths_outside_exact_episode_interface(fixture_root: Path) -> None:
    proxy = make_proxy(fixture_root)
    for path in ("/work/../tests/manifest.json", "/tests/sys0_sparse.npz", "/work/system_x_episodes.npz"):
        with pytest.raises(ValueError, match="outside the official public episode interface"):
            proxy.fit_model(path)
    proxy.close()


def test_worker_namespace_is_numeric_json_only_and_hides_verifier_paths(fixture_root: Path) -> None:
    work_copy = fixture_root / "work-copy"
    shutil.copytree(fixture_root / "work", work_copy)
    command = traffic.build_worker_command(
        app_root=Path("/app"), work_root=Path("/work"), work_copy=work_copy,
        worker_script=Path(traffic.__file__), python="/usr/bin/python3",
        bwrap=shutil.which("bwrap") or "/bin/bwrap", source_root=fixture_root,
    )
    assert "--unshare-net" in command
    assert "--ro-bind" in command and "/tests" in command
    assert "/logs" in command
    assert "pickle" not in Path(traffic.__file__).read_text().lower()


def test_spec_hook_matches_only_the_official_module_and_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(traffic, "_INSTALLED", False)
    original = traffic._ORIGINAL_SPEC_FROM_FILE_LOCATION
    monkeypatch.setattr(traffic, "_ORIGINAL_SPEC_FROM_FILE_LOCATION", original)
    traffic.install(traffic.TASK_ID)
    matched = importlib.util.spec_from_file_location(traffic.MODULE_NAME, traffic.CANDIDATE_PATH)
    ordinary_name = importlib.util.spec_from_file_location("other_module", traffic.CANDIDATE_PATH)
    ordinary_path = importlib.util.spec_from_file_location(traffic.MODULE_NAME, "/app/other.py")
    assert isinstance(matched.loader, traffic._ProxyLoader)
    assert not isinstance(ordinary_name.loader, traffic._ProxyLoader)
    assert not isinstance(ordinary_path.loader, traffic._ProxyLoader)


def test_bundle_contains_the_traffic_worker_helper(tmp_path: Path) -> None:
    bundle = science_isolation.create_bundle(traffic.TASK_ID, tmp_path / "bundle")
    worker = tmp_path / "bundle/science_traffic_isolation.py"
    assert worker.is_file()
    compile(worker.read_text(), str(worker), "exec")
    assert bundle["environment"]["RELAY_SCIENCE_TASK_ID"] == traffic.TASK_ID
    probe = subprocess.run(
        ["/usr/bin/python3", "-c", "import science_isolation; print(science_isolation.science_traffic_isolation.TASK_SLUG)"],
        cwd=tmp_path, env={"PYTHONPATH": str(bundle["bundle_dir"])},
        capture_output=True, text=True, check=True,
    )
    assert probe.stdout.strip() == traffic.TASK_SLUG


def test_science_install_routes_only_traffic_slug_to_import_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    monkeypatch.setattr(subprocess, "Popen", science_isolation._ORIGINAL_POPEN)
    monkeypatch.setattr(traffic, "install", lambda task_id: calls.append(task_id))
    monkeypatch.setenv("RELAY_SCIENCE_TASK_ID", traffic.TASK_SLUG)
    science_isolation.install()
    assert calls == [traffic.TASK_SLUG]

