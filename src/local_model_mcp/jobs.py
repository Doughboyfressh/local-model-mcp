import json
import os
import subprocess
import sys
import time
import uuid

import psutil

from .weights import Checkpoint
from .workspace import json_write, sha256

TERMINAL = {"completed", "failed", "cancelled", "interrupted"}


class Jobs:
    def __init__(self, workspace):
        self.workspace = workspace
        self.children = {}

    def status(self, job_id: str, log_lines=20):
        path = self.workspace.path("runs", job_id)
        status = json.loads((path / "status.json").read_text())
        process_path = path / "process.json"
        if status["status"] not in TERMINAL and process_path.exists():
            identity = json.loads(process_path.read_text())
            try:
                process = psutil.Process(identity["pid"])
                alive = (
                    process.create_time() == identity["created"]
                    and process.status() != psutil.STATUS_ZOMBIE
                )
            except psutil.Error:
                alive = False
            if not alive:
                # Re-read: the worker may have just committed its terminal status.
                status = json.loads((path / "status.json").read_text())
                if status["status"] not in TERMINAL:
                    status = {**status, "status": "interrupted", "error": "Worker exited"}
        child = self.children.get(job_id)
        if child and child.poll() is not None:
            self.children.pop(job_id, None)
        if log_lines:
            if not 0 <= log_lines <= 200:
                raise ValueError("log_lines must be 0–200")
            log = path / "worker.log"
            if log.exists():
                with log.open("rb") as stream:
                    stream.seek(max(0, log.stat().st_size - 32_768))
                    status["log_tail"] = (
                        stream.read().decode(errors="replace").splitlines()[-log_lines:]
                    )
        return status

    def list(self):
        runs = sorted(
            (p for p in (self.workspace.root / "runs").iterdir() if p.is_dir()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[:100]
        return [self.status(p.name, log_lines=0) for p in runs if (p / "status.json").exists()]

    def start(self, kind, request):
        with self.workspace.lock:
            if any(row["status"] not in TERMINAL for row in self.list()):
                raise ValueError("A job is already active; wait or cancel it first")
            checkpoint = Checkpoint(self.workspace.model(request.model))
            if checkpoint.kind != "model":
                raise ValueError("Use a full base model as model, and pass adapter separately")
            fingerprints = {"model": checkpoint.revision()}
            if getattr(request, "adapter", None):
                adapter = Checkpoint(self.workspace.model(request.adapter))
                if adapter.kind != "adapter":
                    raise ValueError("adapter must name a LoRA adapter checkpoint")
                fingerprints["adapter"] = adapter.revision()
            if getattr(request, "dataset", None):
                self.workspace.dataset(request.dataset)
                fingerprints["dataset"] = sha256(
                    self.workspace.path("datasets", request.dataset, ".jsonl")
                )
            output = getattr(request, "output", None)
            if output and self.workspace.path("models", output, exists=False).exists():
                raise ValueError("Output already exists; choose a new model name")
            job_id = uuid.uuid4().hex
            run = self.workspace.path("runs", job_id, exists=False)
            run.mkdir(mode=0o700)
            json_write(
                run / "request.json",
                {
                    "kind": kind,
                    "request": request.model_dump(),
                    "fingerprints": fingerprints,
                },
            )
            json_write(
                run / "status.json",
                {
                    "job_id": job_id,
                    "kind": kind,
                    "status": "queued",
                    "created_at": time.time(),
                },
            )
            env = os.environ.copy()
            env.update(
                {
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                    "HF_HUB_DISABLE_TELEMETRY": "1",
                    "TOKENIZERS_PARALLELISM": "false",
                }
            )
            try:
                with (run / "worker.log").open("wb") as log:
                    child = subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "local_model_mcp.worker",
                            "--root",
                            str(self.workspace.root),
                            "--job",
                            job_id,
                        ],
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=log,
                        env=env,
                    )
                self.children[job_id] = child
                json_write(
                    run / "process.json",
                    {
                        "pid": child.pid,
                        "created": psutil.Process(child.pid).create_time(),
                    },
                )
            except Exception as exc:
                json_write(
                    run / "status.json",
                    {
                        "job_id": job_id,
                        "kind": kind,
                        "status": "failed",
                        "error": str(exc),
                    },
                )
                raise
        return {"job_id": job_id, "status": "queued", "kind": kind}

    def cancel(self, job_id):
        with self.workspace.lock:
            status = self.status(job_id, log_lines=0)
            if status["status"] in TERMINAL:
                return status
            path = self.workspace.path("runs", job_id)
            (path / "cancel").touch()
        return {"job_id": job_id, "status": "cancellation_requested"}
