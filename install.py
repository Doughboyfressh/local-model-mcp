"""Install Local Model MCP using only Python's standard library to bootstrap."""

import argparse
import importlib.util
import json
import os
import platform
import struct
import subprocess
import sys
import venv
from contextlib import contextmanager
from pathlib import Path

MARKER = ".local-model-mcp-installer.json"
CPU_INDEX = "https://download.pytorch.org/whl/cpu"
PROBE = """
import importlib, importlib.metadata, json, platform, sys
names = ['mcp', 'pydantic', 'torch', 'safetensors', 'transformers', 'peft',
         'filelock', 'psutil', 'platformdirs']
for name in names:
    importlib.import_module(name)
import torch
assert torch.ones(2).sum().item() == 2, 'CPU tensor check failed'
print(json.dumps({'version': importlib.metadata.version('local-model-mcp'),
    'python': sys.version, 'platform': platform.platform(),
    'packages': {name: importlib.metadata.version(name) for name in names},
    'devices': {'cpu': True, 'cuda_available': torch.cuda.is_available(),
                'mps_available': torch.backends.mps.is_available()}}))
"""


def native_root():
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / (
            "local-model-mcp"
        )
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "local-model-mcp"
    base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    if not base.is_absolute():
        base = Path.home() / ".local" / "share"
    return base / "local-model-mcp"


def preflight():
    if sys.version_info < (3, 11):  # noqa: UP036 - bootstrap runs before package requirements apply
        raise ValueError("Python 3.11 or newer is required; Python 3.12 is recommended")
    machine = platform.machine().lower()
    supported = {
        "win32": {"amd64", "x86_64"},
        "linux": {"amd64", "x86_64"},
        "darwin": {"arm64", "aarch64"},
    }
    if struct.calcsize("P") != 8 or machine not in supported.get(sys.platform, set()):
        raise ValueError(
            "Supported installations: 64-bit Windows/Linux x64 or macOS Apple silicon. "
            "On Apple silicon, use native ARM Python rather than Rosetta."
        )
    if importlib.util.find_spec("ensurepip") is None:
        raise ValueError("Python venv/ensurepip is missing; install your OS's Python venv package")


def environment_python(directory):
    return directory / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def plan(directory, workspace, wheel=None):
    python = str(environment_python(directory))
    torch = [python, "-m", "pip", "install", "torch>=2.6,<3"]
    if sys.platform != "darwin":
        torch.extend(["--index-url", CPU_INDEX])
    source = str(wheel or Path(__file__).resolve().parent)
    return {
        "environment": str(directory),
        "workspace": str(workspace),
        "config": str(directory / "mcp-config.json"),
        "commands": [
            [python, "-m", "pip", "install", "--upgrade", "pip"],
            torch,
            [python, "-m", "pip", "install", "--upgrade", source],
            [python, "-m", "pip", "check"],
        ],
    }


def validate_destination(directory, workspace):
    if directory == workspace or directory in workspace.parents:
        raise ValueError("Workspace must be outside the disposable installation environment")
    if directory == Path(sys.prefix).resolve():
        raise ValueError("Choose an environment other than the Python running this installer")
    marker = directory / MARKER
    if directory.exists() and any(directory.iterdir()):
        if not marker.is_file() or json.loads(marker.read_text(encoding="utf-8")) != {
            "installer": "local-model-mcp",
            "schema": 1,
        }:
            raise ValueError("Destination is not owned by this installer; choose a new --venv path")


@contextmanager
def install_lock(directory):
    directory.parent.mkdir(parents=True, exist_ok=True)
    lock = directory.parent / f".{directory.name}.install.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        raise ValueError(
            f"Another installation may be running. If none is running, remove {lock} and retry."
        ) from None
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(str(os.getpid()))
        yield
    finally:
        lock.unlink(missing_ok=True)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def run(command, capture=False):
    env = dict(os.environ)
    # Keep package/model lookup independent of a caller's Python environment.
    for key in ("PYTHONHOME", "PYTHONPATH", "PIP_TARGET", "PIP_PREFIX", "PIP_USER"):
        env.pop(key, None)
    env.update(
        PYTHONUTF8="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        TOKENIZERS_PARALLELISM="false",
    )
    return subprocess.run(
        command, check=True, env=env, capture_output=capture, text=True, encoding="utf-8"
    )


def install(directory, workspace, wheel=None):
    specification = plan(directory, workspace, wheel)
    with install_lock(directory):
        validate_destination(directory, workspace)
        directory.mkdir(parents=True, exist_ok=True)
        write_json(directory / MARKER, {"installer": "local-model-mcp", "schema": 1})
        print(f"Creating isolated Python environment: {directory}", flush=True)
        # No activation, system packages, admin access, or shell command interpolation.
        venv.EnvBuilder(with_pip=True).create(directory)
        for command in specification["commands"]:
            run(command)
        python = str(environment_python(directory))
        diagnostics = json.loads(run([python, "-c", PROBE], capture=True).stdout)
        config = json.loads(
            run(
                [
                    python,
                    "-m",
                    "local_model_mcp.server",
                    "--root",
                    str(workspace),
                    "--print-mcp-config",
                ],
                capture=True,
            ).stdout
        )
        write_json(directory / "mcp-config.json", config)
        write_json(directory / "install-info.json", {**specification, **diagnostics})
    print(f"\nInstalled Local Model MCP {diagnostics['version']}.")
    print(f"MCP connection file: {directory / 'mcp-config.json'}")
    print(f"Installation checks: {directory / 'install-info.json'}")
    print(f"Model workspace: {workspace}")
    print("Copy the connection JSON into your MCP host. Models are added separately.")
    return diagnostics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", type=Path, default=native_root() / "venv")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path(os.environ.get("LOCAL_MODEL_MCP_ROOT", native_root())),
    )
    parser.add_argument("--wheel", type=Path, help="Install a locally built .whl instead of source")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show paths/commands without changes"
    )
    args = parser.parse_args(argv)
    try:
        preflight()
        directory, workspace = [
            value.expanduser().resolve() for value in (args.venv, args.workspace)
        ]
        wheel = args.wheel.expanduser().resolve() if args.wheel else None
        if wheel and (not wheel.is_file() or wheel.suffix != ".whl"):
            raise ValueError("--wheel must name an existing local .whl file")
        validate_destination(directory, workspace)
        if args.dry_run:
            print(json.dumps(plan(directory, workspace, wheel), indent=2))
        else:
            install(directory, workspace, wheel)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f"Installation failed: {exc}", file=sys.stderr)
        if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
            print(exc.stderr, file=sys.stderr)
        print(
            "Fix the reported issue, then rerun the same command. No models were downloaded.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
