"""Platform-specific defaults and a shell-free MCP launch configuration."""

import os
import subprocess
import sys
from pathlib import Path

from platformdirs import user_data_path


def background_process_options() -> dict:
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def default_root() -> Path:
    explicit = os.getenv("LOCAL_MODEL_MCP_ROOT")
    if explicit:
        return Path(explicit).expanduser().resolve()
    native = user_data_path("local-model-mcp", appauthor=False)
    legacy = Path.home() / ".local" / "share" / "local-model-mcp"
    # Keep existing 0.1.0 workspaces discoverable on Windows and macOS.
    if legacy.is_dir() and not native.exists():
        return legacy
    return native


def mcp_config(root: str | Path) -> dict:
    return {
        "mcpServers": {
            "local-model-lab": {
                "command": sys.executable,
                "args": [
                    "-m",
                    "local_model_mcp.server",
                    "--root",
                    str(Path(root).expanduser().resolve()),
                ],
            }
        }
    }
