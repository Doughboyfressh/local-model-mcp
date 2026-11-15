import errno
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

from filelock import FileLock

NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}\Z")
MAX_DATASET_BYTES = 64 * 1024 * 1024
WINDOWS_RESERVED = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def name(value: str) -> str:
    if not NAME_RE.fullmatch(value):
        raise ValueError("Use a simple name: 1–96 letters, digits, dots, underscores, or hyphens")
    if value.endswith(".") or value.split(".", 1)[0].upper() in WINDOWS_RESERVED:
        raise ValueError("Name must also be valid on Windows: no reserved devices or trailing dots")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _windows_sharing_error(exc: PermissionError) -> bool:
    return getattr(exc, "winerror", None) in {5, 32, 33} or (
        os.name == "nt" and exc.errno == errno.EACCES
    )


def read_json(path: Path):
    """Read UTF-8 JSON while a worker may atomically replace it on Windows."""
    for attempt in range(10):
        try:
            with path.open(encoding="utf-8") as stream:
                return json.load(stream)
        except PermissionError as exc:
            if not _windows_sharing_error(exc) or attempt == 9:
                raise
            time.sleep(0.01 * (attempt + 1))


def replace_file(source: Path, destination: Path) -> None:
    """Replace a closed file, tolerating short Windows reader/share-lock collisions."""
    for attempt in range(10):
        try:
            os.replace(source, destination)
            return
        except PermissionError as exc:
            if not _windows_sharing_error(exc) or attempt == 9:
                raise
            time.sleep(0.01 * (attempt + 1))


def json_write(path: Path, value: dict) -> None:
    fd, temporary = tempfile.mkstemp(prefix=".write-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, allow_nan=False, ensure_ascii=False)
            stream.write("\n")
        replace_file(Path(temporary), path)
    finally:
        Path(temporary).unlink(missing_ok=True)


class Workspace:
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        for folder in ("models", "datasets", "tensors", "runs"):
            path = self.root / folder
            if path.is_symlink():
                raise ValueError(f"Workspace directory cannot be a symlink: {folder}")
            path.mkdir(exist_ok=True, mode=0o700)
        self.lock = FileLock(self.root / ".workspace.lock", timeout=60)

    def path(self, folder: str, item: str, suffix: str = "", *, exists=True) -> Path:
        path = self.root / folder / (name(item) + suffix)
        if path.is_symlink() or not path.resolve().is_relative_to(self.root / folder):
            raise ValueError("Symlinks and paths outside the workspace are unsupported")
        if exists and not path.exists():
            raise ValueError(f"Not found: {folder}/{item}{suffix}")
        return path

    def model(self, item: str) -> Path:
        path = self.path("models", item)
        if not path.is_dir():
            raise ValueError("A model must be a directory")
        self.check_tree(path)
        return path

    @staticmethod
    def check_tree(path: Path) -> None:
        for child in path.rglob("*"):
            if child.is_symlink() or not (child.is_file() or child.is_dir()):
                raise ValueError(f"Unsupported symlink or special file: {child.name}")

    @contextmanager
    def staging(self):
        stage = Path(tempfile.mkdtemp(prefix=".stage-", dir=self.root / "models"))
        try:
            yield stage
        finally:
            shutil.rmtree(stage, ignore_errors=True)

    def publish(self, stage: Path, output: str) -> Path:
        with self.lock:
            destination = self.path("models", output, exists=False)
            if destination.exists():
                raise ValueError(f"Output model already exists: {output}")
            stage.rename(destination)
            return destination

    def dataset(self, item: str):
        from .schemas import Example

        path = self.path("datasets", item, ".jsonl")
        if path.stat().st_size > MAX_DATASET_BYTES:
            raise ValueError("Dataset exceeds the 64 MiB limit")
        rows = []
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if line.strip():
                    try:
                        rows.append(Example.model_validate_json(line))
                    except ValueError as exc:
                        raise ValueError(f"Invalid dataset line {line_number}: {exc}") from exc
                if len(rows) > 100_000:
                    raise ValueError("Dataset exceeds 100,000 examples")
        if not rows:
            raise ValueError("Dataset is empty")
        return rows

    def write_dataset(self, item: str, rows: list) -> dict:
        if not 1 <= len(rows) <= 100_000:
            raise ValueError("Provide 1–100,000 examples")
        content = "".join(row.model_dump_json(exclude_none=True) + "\n" for row in rows)
        encoded = content.encode()
        if len(encoded) > MAX_DATASET_BYTES:
            raise ValueError("Dataset exceeds the 64 MiB limit")
        with self.lock:
            path = self.path("datasets", item, ".jsonl", exists=False)
            with path.open("xb") as stream:
                stream.write(encoded)
        return {"dataset": item, "examples": len(rows), "sha256": sha256(path)}
