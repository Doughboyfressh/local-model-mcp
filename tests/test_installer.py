import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location("installer", Path(__file__).parents[1] / "install.py")
installer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(installer)


@pytest.mark.parametrize(
    ("system", "machine", "suffix", "cpu_index"),
    [
        ("win32", "AMD64", "Scripts/python.exe", True),
        ("darwin", "arm64", "bin/python", False),
        ("linux", "x86_64", "bin/python", True),
    ],
)
def test_platform_install_plan(tmp_path, monkeypatch, system, machine, suffix, cpu_index):
    monkeypatch.setattr(installer.sys, "platform", system)
    monkeypatch.setattr(installer.platform, "machine", lambda: machine)
    installer.preflight()
    directory, root = tmp_path / "environment 日本語", tmp_path / "models 日本語"
    plan = installer.plan(directory, root)
    assert plan["commands"][1][0] == str(directory / suffix)
    assert (installer.CPU_INDEX in plan["commands"][1]) is cpu_index
    assert all(isinstance(command, list) for command in plan["commands"])
    assert not directory.exists() and not root.exists()


@pytest.mark.parametrize(("system", "machine"), [("darwin", "x86_64"), ("linux", "aarch64")])
def test_unsupported_platforms_fail_before_changes(monkeypatch, system, machine):
    monkeypatch.setattr(installer.sys, "platform", system)
    monkeypatch.setattr(installer.platform, "machine", lambda: machine)
    with pytest.raises(ValueError, match="Supported installations"):
        installer.preflight()


def test_preflight_python_and_venv_requirements(monkeypatch):
    monkeypatch.setattr(installer.sys, "version_info", (3, 10))
    with pytest.raises(ValueError, match="3.11"):
        installer.preflight()
    monkeypatch.setattr(installer.sys, "version_info", (3, 12))
    monkeypatch.setattr(installer.importlib.util, "find_spec", lambda _: None)
    with pytest.raises(ValueError, match="ensurepip"):
        installer.preflight()


def test_dry_run_has_no_side_effects(tmp_path, capsys):
    directory, workspace = tmp_path / "install", tmp_path / "workspace"
    assert (
        installer.main(["--venv", str(directory), "--workspace", str(workspace), "--dry-run"]) == 0
    )
    assert json.loads(capsys.readouterr().out)["environment"] == str(directory)
    assert list(tmp_path.iterdir()) == []


def test_unowned_environment_and_workspace_overlap_rejected(tmp_path):
    directory = tmp_path / "unrelated"
    directory.mkdir()
    existing = directory / "important.txt"
    existing.write_text("keep")
    with pytest.raises(ValueError, match="not owned"):
        installer.validate_destination(directory, tmp_path / "workspace")
    with pytest.raises(ValueError, match="outside"):
        installer.validate_destination(directory, directory / "models")
    assert existing.read_text() == "keep"


def test_install_lock_serializes_and_cleans_up(tmp_path):
    directory = tmp_path / "install"
    with installer.install_lock(directory):
        with pytest.raises(ValueError, match="Another installation"):
            with installer.install_lock(directory):
                pytest.fail("Concurrent installation allowed")
    with installer.install_lock(directory):
        pass
    assert list(tmp_path.iterdir()) == []


def test_failed_install_can_be_retried_without_publishing_config(tmp_path, monkeypatch):
    directory, workspace = tmp_path / "install", tmp_path / "workspace"

    class Builder:
        def __init__(self, **kwargs):
            assert kwargs == {"with_pip": True}

        def create(self, directory):
            assert (directory / installer.MARKER).is_file()

    monkeypatch.setattr(installer.venv, "EnvBuilder", Builder)
    commands = []

    def fail(command, capture=False):
        commands.append(command)
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(installer, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        installer.install(directory, workspace)
    assert not (directory / "mcp-config.json").exists()
    assert not workspace.exists()
    installer.validate_destination(directory, workspace)

    def succeed(command, capture=False):
        commands.append(command)
        if command[1] == "-c":
            return SimpleNamespace(
                stdout=json.dumps({"version": "0.3.0", "devices": {"cpu": True}})
            )
        if "--print-mcp-config" in command:
            return SimpleNamespace(
                stdout=json.dumps(
                    {
                        "mcpServers": {
                            "local-model-lab": {
                                "command": str(installer.environment_python(directory)),
                                "args": ["-m", "local_model_mcp.server", "--root", str(workspace)],
                            }
                        }
                    }
                )
            )
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(installer, "run", succeed)
    installer.install(directory, workspace)
    assert json.loads((directory / "install-info.json").read_text())["version"] == "0.3.0"
    config = json.loads((directory / "mcp-config.json").read_text())
    assert config["mcpServers"]["local-model-lab"]["args"][-1] == str(workspace)
    assert not workspace.exists()
    assert not list(tmp_path.glob("*.lock"))


def test_subprocesses_are_shell_free_and_model_offline(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "unrelated")
    monkeypatch.setenv("PIP_TARGET", "unrelated")

    def check(command, **kwargs):
        assert command == ["python path", "-m", "pip", "check"]
        assert "shell" not in kwargs
        assert "PYTHONPATH" not in kwargs["env"] and "PIP_TARGET" not in kwargs["env"]
        assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"
        assert kwargs["check"]

    monkeypatch.setattr(installer.subprocess, "run", check)
    installer.run(["python path", "-m", "pip", "check"])
