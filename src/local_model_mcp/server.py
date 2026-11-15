import argparse
import asyncio
import json
import platform
from pathlib import Path
from typing import Annotated, Any

import torch
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import Field

from . import __version__, weights
from .jobs import Jobs
from .runtime import default_root, mcp_config
from .schemas import (
    EditRequest,
    EvaluateRequest,
    Example,
    GenerateRequest,
    MergeRequest,
    Name,
    TrainRequest,
)
from .workspace import Workspace, read_json

READ = ToolAnnotations(readOnlyHint=True, openWorldHint=False)
WRITE = ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=False)


def create_server(root: str | Path) -> FastMCP:
    workspace = Workspace(root)
    jobs = Jobs(workspace)
    server = FastMCP(
        "Local Model Lab",
        instructions=(
            "Manage local Hugging Face causal language models. "
            "Start with capabilities and inventory. Create a dataset, launch a fine-tune, "
            "poll job_status, then evaluate on held-out data. For direct tensor edits, "
            "inspect exact names, preview_weight_edits, and pass the returned plan_token "
            "to apply_weight_edits with an unused output name. All operations are local. "
            "Original checkpoints are preserved. LoRA training returns an adapter; use its base "
            "model "
            "plus adapter for evaluation/generation, or merge it into a standalone model. "
            "Weight changes are numerical interventions; validate behavior with evaluation "
            "and generation."
        ),
    )

    @server.tool(annotations=READ)
    def capabilities() -> dict[str, Any]:
        """Report supported formats, hardware, workspace, and operational limits."""
        return {
            "version": __version__,
            "workspace": str(workspace.root),
            "transport": "stdio",
            "platform": platform.system(),
            "supported_platforms": ["Windows", "macOS", "Linux"],
            "formats": ["Hugging Face unquantized safetensors (single or sharded)", "PEFT LoRA"],
            "training": ["lora", "full"],
            "tasks": ["causal language modeling"],
            "devices": {
                "cpu": True,
                "cuda": torch.cuda.is_available(),
                "mps": torch.backends.mps.is_available(),
            },
            "weight_operations": [
                "scale",
                "add",
                "set",
                "zero",
                "clamp",
                "noise",
                "add_tensor",
                "replace_tensor",
                "low_rank",
            ],
            "limits": {
                "concurrent_jobs": 1,
                "dataset_bytes": 64 * 1024 * 1024,
                "tensor_asset_bytes": weights.MAX_ASSET_BYTES,
                "edit_selection_elements": weights.MAX_EDIT_ELEMENTS,
            },
            "notes": [
                "No model downloads or remote custom code",
                "GGUF, quantized training, and distributed training are unsupported",
                "Cancellation is cooperative and may wait for model loading or an optimizer step",
                "Restart inference runtimes to load newly created checkpoint versions",
            ],
        }

    @server.tool(annotations=READ)
    def inventory() -> dict[str, Any]:
        """List up to 200 names per category: models, datasets, and tensor assets."""
        result = {}
        for folder, suffix in (("models", ""), ("datasets", ".jsonl"), ("tensors", ".safetensors")):
            items = []
            for path in sorted((workspace.root / folder).iterdir()):
                if path.name.startswith(".") or path.is_symlink():
                    continue
                if folder == "models" and path.is_dir():
                    items.append(path.name)
                elif suffix and path.is_file() and path.name.endswith(suffix):
                    items.append(path.name.removesuffix(suffix))
            result[folder] = items[:200]
            result[f"{folder}_total"] = len(items)
        return result

    @server.tool(annotations=READ)
    def inspect_model(
        model: Name,
        contains: str = "",
        offset: Annotated[int, Field(ge=0)] = 0,
        limit: Annotated[int, Field(ge=1, le=200)] = 50,
    ) -> dict[str, Any]:
        """Page through tensor names, shapes, dtypes, and shards without reading weight data."""
        checkpoint = weights.Checkpoint(workspace.model(model))
        items = [value for key, value in sorted(checkpoint.catalog.items()) if contains in key]
        manifest_path = checkpoint.path / "local_model_manifest.json"
        return {
            "model": model,
            "kind": checkpoint.kind,
            "matched_tensors": len(items),
            "total_elements": sum(item["elements"] for item in checkpoint.catalog.values()),
            "tensors": items[offset : offset + limit],
            "next_offset": offset + limit if offset + limit < len(items) else None,
            "manifest": read_json(manifest_path) if manifest_path.exists() else None,
        }

    @server.tool(annotations=READ)
    async def read_tensor(
        model: Name,
        tensor: str,
        ranges: list[tuple[int, int]] | None = None,
        limit: Annotated[int, Field(ge=1, le=1024)] = 64,
    ) -> dict[str, Any]:
        """Read tensor statistics and a bounded flat sample. ranges are [start, stop] per axis."""
        return await asyncio.to_thread(weights.read_tensor, workspace, model, tensor, ranges, limit)

    @server.tool(annotations=WRITE)
    def create_dataset(
        dataset: Name,
        examples: Annotated[list[Example], Field(min_length=1, max_length=10000)],
    ) -> dict[str, Any]:
        """Create a new JSONL dataset from text or chat examples. Existing names are rejected."""
        return workspace.write_dataset(dataset, examples)

    @server.tool(annotations=READ)
    def inspect_dataset(dataset: Name) -> dict[str, Any]:
        """Validate a dataset and report row formats and character counts, without text."""
        rows = workspace.dataset(dataset)
        return {
            "dataset": dataset,
            "examples": len(rows),
            "text_examples": sum(row.text is not None for row in rows),
            "chat_examples": sum(row.messages is not None for row in rows),
            "characters": sum(
                len(row.text) if row.text is not None else sum(len(m.content) for m in row.messages)
                for row in rows
            ),
        }

    @server.tool(annotations=WRITE)
    def create_tensor_asset(asset: Name, tensors: dict[str, Any]) -> dict[str, Any]:
        """Save named float32 arrays for replacement, additive, or low-rank edits. Max 1M values.

        Example: tensors={"left": [[1.0], [0.0]], "right": [[0.1, 0.2]]}.
        For larger assets, place a safetensors file directly in workspace/tensors (max 256 MiB).
        """
        return weights.write_asset(workspace, asset, tensors)

    @server.tool(annotations=READ)
    async def preview_weight_edits(request: EditRequest) -> dict[str, Any]:
        """Simulate edits and return statistics and a token bound to weights, assets, and edits.

        value means scalar for scale/add/set, or standard deviation for noise. low_rank applies
        W += alpha * (left @ right). Ranges use exclusive stops; omit for the whole tensor.
        Tensor asset shapes must match exactly. This reads each affected shard into CPU memory.
        """
        return await asyncio.to_thread(weights.preview, workspace, request)

    @server.tool(annotations=WRITE)
    async def apply_weight_edits(
        request: EditRequest,
        output: Name,
        expected_plan_token: str,
    ) -> dict[str, Any]:
        """Apply a previewed numerical edit into a new checkpoint. Pass preview's plan_token.

        The original is preserved. Stale plans, existing output names, invalid shapes, and
        nonfinite results are rejected. Copies all checkpoint bytes before modifying them.
        """
        return await asyncio.to_thread(
            weights.apply, workspace, request, output, expected_plan_token
        )

    @server.tool(annotations=READ)
    async def compare_weights(
        model_a: Name,
        model_b: Name,
        tensors: Annotated[list[str], Field(min_length=1, max_length=64)],
    ) -> dict[str, Any]:
        """Compute changed element counts and delta norms for selected, equally shaped tensors."""
        return await asyncio.to_thread(weights.compare, workspace, model_a, model_b, tensors)

    @server.tool(annotations=WRITE)
    async def start_fine_tune(request: TrainRequest) -> dict[str, Any]:
        """Launch local LoRA or full-parameter training; return a job ID after validating inputs.

        Training cycles shuffled examples until max_steps optimizer steps. Every nonpadding
        next token contributes to loss, including chat prompt tokens. No assistant-only masking.
        Dtype bfloat16 uses autocast with float32 trainable weights. LoRA saves an adapter.
        """
        return await asyncio.to_thread(jobs.start, "train", request)

    @server.tool(annotations=WRITE)
    async def start_evaluation(request: EvaluateRequest) -> dict[str, Any]:
        """Launch token-weighted next-token loss and perplexity evaluation; poll job_status."""
        return await asyncio.to_thread(jobs.start, "evaluate", request)

    @server.tool(annotations=WRITE)
    async def start_generation(request: GenerateRequest) -> dict[str, Any]:
        """Launch a local raw-text completion. temperature=0 selects greedy decoding."""
        return await asyncio.to_thread(jobs.start, "generate", request)

    @server.tool(annotations=WRITE)
    async def merge_adapter(request: MergeRequest) -> dict[str, Any]:
        """Launch a LoRA merge into a new standalone model; preserve the base and adapter."""
        return await asyncio.to_thread(jobs.start, "merge", request)

    @server.tool(annotations=READ)
    def job_status(
        job_id: Name,
        log_lines: Annotated[int, Field(ge=0, le=200)] = 20,
    ) -> dict[str, Any]:
        """Read persistent status, progress, results, and a bounded worker log tail."""
        return jobs.status(job_id, log_lines)

    @server.tool(annotations=READ)
    def list_jobs() -> dict[str, Any]:
        """List the most recent 100 persistent jobs; workers can outlive a server restart."""
        return {"jobs": jobs.list()}

    @server.tool(annotations=WRITE)
    async def cancel_job(job_id: Name) -> dict[str, Any]:
        """Request cooperative cancellation. Poll until terminal; no partial model is published."""
        return await asyncio.to_thread(jobs.cancel, job_id)

    return server


def main():
    parser = argparse.ArgumentParser(description="Local fine-tuning and weight-editing MCP server")
    parser.add_argument("--root", type=Path, default=default_root())
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument(
        "--print-mcp-config",
        action="store_true",
        help="Print JSON configuration using this Python interpreter, then exit",
    )
    args = parser.parse_args()
    if args.print_mcp_config:
        print(json.dumps(mcp_config(args.root), indent=2))
        return
    # This server deliberately exposes only local stdio, with the host managing access.
    create_server(args.root).run(transport="stdio")


if __name__ == "__main__":
    main()
