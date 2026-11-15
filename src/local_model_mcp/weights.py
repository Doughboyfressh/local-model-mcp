import hashlib
import json
import math
import shutil
from pathlib import Path, PureWindowsPath

import torch
from safetensors import safe_open
from safetensors.torch import load, load_file, save_file

from .schemas import EditRequest
from .workspace import Workspace, json_write, replace_file, sha256

MAX_EDIT_ELEMENTS = 100_000_000
MAX_ASSET_BYTES = 256 * 1024 * 1024
FLOAT_DTYPES = {torch.float16, torch.bfloat16, torch.float32, torch.float64}


class Checkpoint:
    def __init__(self, path: Path):
        self.path = path
        Workspace.check_tree(path)
        layouts = [
            (path / "model.safetensors.index.json").is_file(),
            (path / "model.safetensors").is_file(),
            (path / "adapter_model.safetensors").is_file(),
        ]
        if sum(layouts) > 1:
            raise ValueError(
                "Ambiguous checkpoint: keep exactly one model or adapter weight layout"
            )
        config = path / "config.json"
        if config.exists() and json.loads(config.read_text(encoding="utf-8")).get(
            "quantization_config"
        ):
            raise ValueError("Quantized checkpoints are unsupported; use unquantized safetensors")
        if (path / "model.safetensors.index.json").is_file():
            index = json.loads((path / "model.safetensors.index.json").read_text(encoding="utf-8"))
            mapping = index.get("weight_map", {})
            if not isinstance(mapping, dict) or not mapping:
                raise ValueError("Invalid shard index")
            self.mapping = mapping
            self.kind = "model"
        elif (path / "model.safetensors").is_file():
            self.mapping = self._single("model.safetensors")
            self.kind = "model"
        elif (path / "adapter_model.safetensors").is_file():
            self.mapping = self._single("adapter_model.safetensors")
            self.kind = "adapter"
        else:
            raise ValueError(
                "Expected model.safetensors, a shard index, or adapter_model.safetensors"
            )
        self.catalog = {}
        for filename in sorted(set(self.mapping.values())):
            if (
                not isinstance(filename, str)
                or Path(filename).name != filename
                or PureWindowsPath(filename).name != filename
                or not filename.endswith(".safetensors")
            ):
                raise ValueError("Shard names must be top-level safetensors files")
            with safe_open(path / filename, framework="pt", device="cpu") as stream:
                keys = set(stream.keys())
                expected = {k for k, v in self.mapping.items() if v == filename}
                if keys != expected:
                    raise ValueError("Shard index and tensor keys disagree")
                for key in sorted(keys):
                    view = stream.get_slice(key)
                    shape = view.get_shape()
                    self.catalog[key] = {
                        "name": key,
                        "shape": shape,
                        "dtype": view.get_dtype(),
                        "elements": math.prod(shape),
                        "shard": filename,
                    }

    def _single(self, filename):
        with safe_open(self.path / filename, framework="pt", device="cpu") as stream:
            return dict.fromkeys(stream.keys(), filename)

    def revision(self):
        # Include configs and tokenizer bytes, not just tensor metadata.
        files = {
            p.relative_to(self.path).as_posix(): sha256(p)
            for p in sorted(self.path.rglob("*"))
            if p.is_file()
        }
        return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()

    def tensor(self, key: str):
        if key not in self.mapping:
            raise ValueError(f"Unknown tensor: {key}")
        with safe_open(self.path / self.mapping[key], framework="pt", device="cpu") as stream:
            return stream.get_tensor(key)


def selection(tensor: torch.Tensor, ranges):
    if ranges is None:
        view = tensor
    else:
        if len(ranges) != tensor.ndim:
            raise ValueError("Provide one [start, stop] range per tensor axis")
        for (start, stop), size in zip(ranges, tensor.shape, strict=True):
            if not 0 <= start < stop <= size:
                raise ValueError(f"Range must satisfy 0 <= start < stop <= {size}")
        view = tensor[tuple(slice(a, b) for a, b in ranges)]
    if not 0 < view.numel() <= MAX_EDIT_ELEMENTS:
        raise ValueError(f"Select between 1 and {MAX_EDIT_ELEMENTS:,} elements")
    return view


def stats(tensor):
    x = tensor.detach().to(torch.float64)
    return {
        "elements": x.numel(),
        "finite": bool(torch.isfinite(x).all()),
        "min": x.min().item() if x.numel() and torch.isfinite(x).all() else None,
        "max": x.max().item() if x.numel() and torch.isfinite(x).all() else None,
        "mean": x.mean().item() if x.numel() and torch.isfinite(x).all() else None,
        "l2_norm": x.norm().item() if torch.isfinite(x).all() else None,
    }


def read_tensor(workspace, model, tensor, ranges=None, limit=64):
    if not 1 <= limit <= 1024:
        raise ValueError("limit must be 1–1024")
    checkpoint = Checkpoint(workspace.model(model))
    # Sliced reads avoid loading the entire tensor when a range is supplied.
    if tensor not in checkpoint.mapping:
        raise ValueError(f"Unknown tensor: {tensor}")
    metadata = checkpoint.catalog[tensor]
    shape = metadata["shape"]
    elements = metadata["elements"] if ranges is None else math.prod(b - a for a, b in ranges)
    if elements > MAX_EDIT_ELEMENTS:
        raise ValueError("Tensor is too large; specify ranges")
    if ranges is not None:
        if len(ranges) != len(shape):
            raise ValueError("Provide one range per axis")
        for (start, stop), size in zip(ranges, shape, strict=True):
            if not 0 <= start < stop <= size:
                raise ValueError("Invalid range")
    with safe_open(checkpoint.path / metadata["shard"], framework="pt", device="cpu") as stream:
        if ranges is None:
            value = stream.get_tensor(tensor)
        else:
            value = stream.get_slice(tensor)[tuple(slice(a, b) for a, b in ranges)]
    if not torch.isfinite(value).all():
        raise ValueError("Selected tensor contains nonfinite values")
    return {
        "tensor": metadata,
        "selected_shape": list(value.shape),
        "stats": stats(value),
        "values": value.reshape(-1)[:limit].double().tolist(),
        "truncated": value.numel() > limit,
    }


def write_asset(workspace, asset, tensors):
    if not 1 <= len(tensors) <= 64:
        raise ValueError("Provide 1–64 tensors")
    values = {}
    total = 0
    for key, data in tensors.items():
        if not key or len(key) > 512:
            raise ValueError("Invalid tensor key")
        value = torch.tensor(data, dtype=torch.float32)
        total += value.numel()
        if total > 1_000_000 or not torch.isfinite(value).all():
            raise ValueError("Assets sent over MCP must be finite and at most 1,000,000 elements")
        values[key] = value.contiguous()
    with workspace.lock:
        path = workspace.path("tensors", asset, ".safetensors", exists=False)
        if path.exists():
            raise ValueError("Tensor asset already exists")
        save_file(values, path)
    return {"asset": asset, "elements": total, "sha256": sha256(path)}


def assets_for(workspace, request):
    hashes, assets = {}, {}
    for asset in sorted({edit.asset for edit in request.edits if edit.asset}):
        path = workspace.path("tensors", asset, ".safetensors")
        if path.stat().st_size > MAX_ASSET_BYTES:
            raise ValueError("Tensor assets must be at most 256 MiB")
        data = path.read_bytes()
        hashes[asset] = hashlib.sha256(data).hexdigest()
        assets[asset] = load(data)
    return hashes, assets


def transform(value, edit, assets):
    if value.dtype not in FLOAT_DTYPES:
        raise ValueError("Only float16, bfloat16, float32, and float64 weights are editable")
    view = selection(value, edit.ranges)
    x = view.double() if value.dtype == torch.float64 else view.float()
    if not torch.isfinite(x).all():
        raise ValueError("Selected source weights are nonfinite")
    before = stats(x)
    operation = edit.operation
    if operation == "scale":
        x = x * edit.value
    elif operation == "add":
        x = x + edit.value
    elif operation == "set":
        x = torch.full_like(x, edit.value)
    elif operation == "zero":
        x = torch.zeros_like(x)
    elif operation == "clamp":
        x = x.clamp(edit.minimum, edit.maximum)
    elif operation == "noise":
        generator = torch.Generator(device="cpu").manual_seed(edit.seed)
        x = x + torch.randn(x.shape, generator=generator, dtype=x.dtype) * edit.value
    else:
        source = assets[edit.asset]
        if operation == "low_rank":
            a, b = source[edit.left_tensor].to(x.dtype), source[edit.right_tensor].to(x.dtype)
            if x.ndim != 2 or a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
                raise ValueError("low_rank requires compatible 2D factors: left @ right")
            if (a.shape[0], b.shape[1]) != tuple(x.shape):
                raise ValueError("Low-rank product must exactly match the selected shape")
            delta = a @ b
        else:
            delta = source[edit.source_tensor].to(x.dtype)
        if delta.shape != x.shape:
            raise ValueError(
                "Source tensor shape must exactly match the selection; no broadcasting"
            )
        x = delta if operation == "replace_tensor" else x + edit.alpha * delta
    x = x.to(value.dtype)
    if not torch.isfinite(x).all():
        raise ValueError("Edit would create nonfinite weights, including dtype overflow")
    view.copy_(x)
    return {"tensor": edit.tensor, "operation": operation, "before": before, "after": stats(x)}


def run_edits(checkpoint, request, assets, *, save=False):
    summaries = []
    # Preserve operation order within each tensor while keeping only one shard resident.
    for shard in sorted(set(checkpoint.mapping.values())):
        edits = [edit for edit in request.edits if checkpoint.mapping[edit.tensor] == shard]
        if not edits:
            continue
        tensors = load_file(checkpoint.path / shard)
        for key in {edit.tensor for edit in edits}:
            tensors[key] = tensors[key].clone()
        for edit in edits:
            summaries.append(transform(tensors[edit.tensor], edit, assets))
        if save:
            with safe_open(checkpoint.path / shard, framework="pt", device="cpu") as stream:
                metadata = stream.metadata()
            # Write a new inode; never overwrite the memory-mapped source file.
            temporary = checkpoint.path / (shard + ".new")
            save_file(tensors, temporary, metadata=metadata)
            # Release memory maps before replacing the shard (also needed on Windows).
            del tensors
            replace_file(temporary, checkpoint.path / shard)
    return summaries


def plan(checkpoint, request, hashes):
    unknown = set(edit.tensor for edit in request.edits) - checkpoint.catalog.keys()
    if unknown:
        raise ValueError(f"Unknown tensors: {sorted(unknown)}")
    revision = checkpoint.revision()
    payload = {"revision": revision, "request": request.model_dump(), "assets": hashes}
    token = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return revision, token


def preview(workspace: Workspace, request: EditRequest):
    checkpoint = Checkpoint(workspace.model(request.model))
    hashes, assets = assets_for(workspace, request)
    revision, token = plan(checkpoint, request, hashes)
    summaries = run_edits(checkpoint, request, assets)
    if checkpoint.revision() != revision:
        raise ValueError("Source changed during preview; retry")
    return {"source_revision": revision, "plan_token": token, "changes": summaries}


def apply(workspace: Workspace, request: EditRequest, output: str, expected_plan_token: str):
    with workspace.lock, workspace.staging() as stage:
        destination = workspace.path("models", output, exists=False)
        if destination.exists():
            raise ValueError("Output already exists; choose a new version name")
        source = workspace.model(request.model)
        shutil.copytree(source, stage, dirs_exist_ok=True)
        checkpoint = Checkpoint(stage)
        hashes, assets = assets_for(workspace, request)
        revision, token = plan(checkpoint, request, hashes)
        if token != expected_plan_token:
            raise ValueError("Stale or different edit plan; preview the exact request again")
        summaries = run_edits(checkpoint, request, assets, save=True)
        previous_manifest = source / "local_model_manifest.json"
        previous = (
            json.loads(previous_manifest.read_text(encoding="utf-8"))
            if previous_manifest.exists()
            else {}
        )
        json_write(
            stage / "local_model_manifest.json",
            {
                "operation": "weight_edit",
                "source": request.model,
                "source_revision": revision,
                "base_model_revision": previous.get("base_model_revision"),
                "plan_token": token,
                "request": request.model_dump(),
                "asset_hashes": hashes,
                "changes": summaries,
            },
        )
        workspace.publish(stage, output)
    return {"model": output, "source": request.model, "changes": summaries}


def compare(workspace, model_a, model_b, tensors):
    if not 1 <= len(tensors) <= 64:
        raise ValueError("Select 1–64 tensor names")
    a, b = Checkpoint(workspace.model(model_a)), Checkpoint(workspace.model(model_b))
    changes = []
    for key in tensors:
        x, y = a.tensor(key), b.tensor(key)
        if x.shape != y.shape or x.numel() > MAX_EDIT_ELEMENTS:
            raise ValueError("Tensors must have equal shapes and at most 100M elements")
        x, y = x.double(), y.double()
        if not (torch.isfinite(x).all() and torch.isfinite(y).all()):
            raise ValueError("Cannot compare nonfinite tensors")
        delta = y - x
        changes.append(
            {
                "tensor": key,
                "changed_elements": int((x != y).sum()),
                "max_abs_delta": delta.abs().max().item() if delta.numel() else 0,
                "l2_delta": delta.norm().item(),
            }
        )
    return {"model_a": model_a, "model_b": model_b, "changes": changes}
