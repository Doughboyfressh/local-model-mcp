import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from local_model_mcp import runtime
from local_model_mcp.schemas import Example, TrainRequest
from local_model_mcp.weights import Checkpoint
from local_model_mcp.workspace import Workspace, json_write, name, sha256


@pytest.mark.parametrize("value", ["CON", "con.json", "AUX", "NUL", "COM1", "LPT9.txt", "model."])
def test_names_remain_valid_when_moved_to_windows(value, tmp_path):
    with pytest.raises(ValueError, match="Windows"):
        name(value)
    with pytest.raises(ValueError):
        TrainRequest(model=value, dataset="train", output="result")
    with pytest.raises(ValueError):
        Workspace(tmp_path).path("models", value, exists=False)


def test_unicode_datasets_and_json(tmp_path):
    workspace = Workspace(tmp_path / "model lab 日本語")
    text = "你好 café 😀 مرحبا"
    workspace.write_dataset("unicode", [Example(text=text)])
    assert workspace.dataset("unicode")[0].text == text
    path = workspace.root / "runs" / "status.json"
    json_write(path, {"text": text})
    assert json.loads(path.read_bytes().decode("utf-8")) == {"text": text}
    assert b"\r\n" not in path.read_bytes()


def test_revision_uses_portable_paths(tmp_path):
    path = tmp_path / "checkpoint"
    (path / "nested").mkdir(parents=True)
    save_file({"weight": torch.ones(2)}, path / "model.safetensors")
    (path / "nested" / "tokenizer.json").write_bytes(b'{"language":"ja"}')
    expected_files = {
        "model.safetensors": sha256(path / "model.safetensors"),
        "nested/tokenizer.json": sha256(path / "nested" / "tokenizer.json"),
    }
    expected = hashlib.sha256(json.dumps(expected_files, sort_keys=True).encode()).hexdigest()
    assert Checkpoint(path).revision() == expected


def test_json_writes_with_concurrent_readers(tmp_path):
    path = tmp_path / "status.json"
    json_write(path, {"step": -1})

    def write_updates():
        for step in range(50):
            json_write(path, {"step": step})

    with ThreadPoolExecutor(max_workers=2) as executor:
        writing = executor.submit(write_updates)
        for _ in range(100):
            assert isinstance(json.loads(path.read_text(encoding="utf-8"))["step"], int)
        writing.result(timeout=10)
    assert json.loads(path.read_text(encoding="utf-8"))["step"] == 49
    assert not list(tmp_path.glob(".write-*"))


def test_windows_sharing_collision_preserves_atomic_json(tmp_path, monkeypatch):
    import local_model_mcp.workspace as module

    path = tmp_path / "status.json"
    json_write(path, {"step": 0})
    real_replace = module.os.replace
    attempts = 0

    def colliding_replace(source, destination):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            assert json.loads(path.read_text(encoding="utf-8")) == {"step": 0}
            error = PermissionError("Windows sharing violation")
            error.winerror = 32
            raise error
        return real_replace(source, destination)

    monkeypatch.setattr(module.os, "replace", colliding_replace)
    json_write(path, {"step": 1})
    assert json.loads(path.read_text(encoding="utf-8")) == {"step": 1}


def test_native_default_and_legacy_workspace(tmp_path, monkeypatch):
    monkeypatch.delenv("LOCAL_MODEL_MCP_ROOT", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    native = tmp_path / "native"
    monkeypatch.setattr(runtime, "user_data_path", lambda *args, **kwargs: native)
    assert runtime.default_root() == native
    legacy = tmp_path / ".local" / "share" / "local-model-mcp"
    legacy.mkdir(parents=True)
    assert runtime.default_root() == legacy
    native.mkdir()
    assert runtime.default_root() == native
    explicit = tmp_path / "custom"
    monkeypatch.setenv("LOCAL_MODEL_MCP_ROOT", str(explicit))
    assert runtime.default_root() == explicit


def test_config_printing_handles_spaces_and_unicode_without_creating_workspace(tmp_path):
    root = tmp_path / "model lab 日本語"
    process = subprocess.run(
        [sys.executable, "-m", "local_model_mcp.server", "--root", str(root), "--print-mcp-config"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    launch = json.loads(process.stdout)["mcpServers"]["local-model-lab"]
    assert Path(launch["command"]) == Path(sys.executable)
    assert launch["args"] == ["-m", "local_model_mcp.server", "--root", str(root.resolve())]
    assert not root.exists()
