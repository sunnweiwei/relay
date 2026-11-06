from __future__ import annotations

import io
import json
import struct
import tempfile

import numpy as np
import pytest

from evals.pilot import science_onnx_isolation as adapter


def test_numeric_ipc_roundtrip_preserves_dtype_and_shape() -> None:
    expected = np.arange(12, dtype=np.float32).reshape(3, 4)
    descriptors, payload = adapter._encode_feeds({"features": expected})
    decoded = adapter._decode_array(descriptors[0], payload, maximum=adapter.MAX_INPUT_BYTES)
    np.testing.assert_array_equal(decoded, expected)
    assert decoded.dtype == expected.dtype


@pytest.mark.parametrize("value", [np.array([object()], dtype=object), np.array(["secret"]),
                                   np.array([1 + 2j]), np.array([1], dtype=[("x", "i4")])])
def test_ipc_rejects_non_numeric_or_structured_arrays(value: np.ndarray) -> None:
    with pytest.raises(adapter.OnnxIsolationError):
        adapter._array_bytes(value, maximum=adapter.MAX_INPUT_BYTES)


def test_ipc_rejects_shape_byte_mismatch() -> None:
    with pytest.raises(adapter.OnnxIsolationError):
        adapter._decode_array({"dtype": "<f4", "shape": [2], "nbytes": 4}, b"1234",
                              maximum=adapter.MAX_INPUT_BYTES)


def test_model_mount_keeps_container_visible_path(monkeypatch: pytest.MonkeyPatch) -> None:
    original_resolve = adapter.Path.resolve
    with tempfile.TemporaryDirectory(prefix="microarch-input.", dir="/tmp") as directory:
        model = adapter.Path(directory) / "model.onnx"
        model.write_bytes(b"candidate")

        def resolve(path: adapter.Path, *args, **kwargs) -> adapter.Path:
            if path == model:
                return adapter.Path("/host-only/canonical/model.onnx")
            return original_resolve(path, *args, **kwargs)

        monkeypatch.setattr(adapter.Path, "resolve", resolve)
        assert adapter.allowed_model_path("microarch-modeling", model) == model


def test_child_frame_reader_handles_short_reads() -> None:
    class ShortReader(io.BytesIO):
        def read(self, size: int = -1) -> bytes:
            return super().read(min(size, 2))

    header = json.dumps({"payload_bytes": 3}).encode()
    frame = struct.pack("!I", len(header)) + header + b"abc"
    actual, payload = adapter._read_child_frame(ShortReader(frame))
    assert actual["payload_bytes"] == 3
    assert payload == b"abc"


def test_bwrap_command_mounts_only_model_and_helper_and_hides_verifier_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    helper = tmp_path / "science_onnx_isolation.py"
    helper.write_text("# trusted adapter")
    model = tmp_path / "model.onnx"
    model.write_bytes(b"model")
    monkeypatch.setattr(adapter, "BWRAP_PATH", "/fake/bwrap")
    monkeypatch.setattr(adapter, "__file__", str(helper))
    command = adapter._bubblewrap_command(model, "a" * 64, "ort")
    binds = [command[i + 1:i + 3] for i, item in enumerate(command[:-2]) if item == "--ro-bind"]
    assert [str(helper), "/tmp/relay-helper/science_onnx_isolation.py"] in binds
    assert [str(model), adapter.MODEL_MOUNT] in binds
    assert [str(tmp_path), adapter.BUNDLE_PATH] not in binds
    model_bind_index = command.index("--ro-bind", command.index(str(model)) - 1)
    assert model_bind_index < command.index("/root")
    helper_bind_index = command.index("--ro-bind", command.index(str(helper)) - 1)
    opt_tmpfs_index = next(i for i in range(len(command) - 1)
                           if command[i:i + 2] == ["--tmpfs", "/opt"])
    assert helper_bind_index < opt_tmpfs_index
    assert "--unshare-net" in command
    assert "/tests" in command and "/logs" in command and "/root" in command
    package_path_index = command.index("PYTHONPATH")
    assert command[package_path_index + 1] == adapter.os.pathsep.join(
        map(str, adapter._python_package_paths())
    )
    assert command[-8:] == ["--", adapter.sys.executable,
                            "/tmp/relay-helper/science_onnx_isolation.py",
                            "--child", "--kind", "ort", "--model", adapter.MODEL_MOUNT]
