# Local Model MCP

An MCP server that lets an MCP-capable frontier-model host fine-tune local causal language models
and directly edit their checkpoint tensors. Runs entirely on your machine over **stdio**.

**Implemented:** LoRA and full-parameter fine-tuning, tensor inspection, slice editing,
tensor replacement, additive and low-rank updates, evaluation, generation, adapter merging,
background jobs, and checkpoint provenance. Every mutation creates a new checkpoint version.

## Install and connect

Requires Python 3.11+ and enough RAM/VRAM for the chosen model. The package uses the MCP Python
SDK 1.x and Transformers 4.x intentionally; dependencies exclude their next major versions.

```bash
cd local-model-mcp
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell instead: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# CPU-only Linux/Windows: install CPU PyTorch first to avoid CUDA dependencies.
python -m pip install "torch>=2.6,<3" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
```

For NVIDIA or Apple GPUs, install the appropriate PyTorch build for your system in place of
the CPU-only command. Training defaults to float32; bfloat16 training uses autocast while
keeping trainable parameters in float32. Apple MPS uses float32.

Create a dedicated workspace, and **copy** a local Hugging Face model into it:

```text
/absolute/path/model-lab/
  models/
    my-model/
      config.json
      model.safetensors
      tokenizer.json
      tokenizer_config.json
  datasets/
  tensors/
  runs/
```

All tokenizer assets required by your model must be present. Sharded models use
`model.safetensors.index.json` and the referenced `.safetensors` shards. Symlinks are rejected;
copy the resolved files when using a Hugging Face cache snapshot. The server creates missing
workspace directories when it starts.

Add this entry to the MCP configuration for your local MCP host. Replace both absolute paths:

```json
{
  "mcpServers": {
    "local-model-lab": {
      "command": "/absolute/path/local-model-mcp/.venv/bin/local-model-mcp",
      "args": ["--root", "/absolute/path/model-lab"]
    }
  }
}
```

On Windows the command is the absolute path to `.venv\\Scripts\\local-model-mcp.exe`.
The host needs to support launching **local stdio MCP servers**. A hosted-only connector
cannot access your machine's checkpoints by itself. This package does not provide a public
HTTP endpoint or connect itself to a hosted chat service.

Run manually with `local-model-mcp --root /absolute/path/model-lab`. Without `--root`,
`LOCAL_MODEL_MCP_ROOT` is used, then `~/.local/share/local-model-mcp`.

## Tools

| Tool | Purpose |
| --- | --- |
| `capabilities` | Formats, devices, workspace, and limits |
| `inventory` | Model, dataset, and tensor asset names |
| `inspect_model` | Paginated tensor names, shapes, dtypes, shards, provenance |
| `read_tensor` | Selected values and numerical statistics |
| `create_dataset` / `inspect_dataset` | Create and validate text or chat JSONL |
| `create_tensor_asset` | Create float32 arrays for tensor edits |
| `preview_weight_edits` | Simulate edits and return a content-bound plan token |
| `apply_weight_edits` | Publish a new edited checkpoint using the preview token |
| `compare_weights` | Exact changed-element counts and delta norms |
| `start_fine_tune` | LoRA or full-parameter training job |
| `start_evaluation` | Token-weighted loss and perplexity job |
| `start_generation` | Raw-text completion job |
| `merge_adapter` | Merge LoRA into a new standalone checkpoint |
| `job_status` / `list_jobs` / `cancel_job` | Persistent progress, results, cancellation |

## Fine-tune workflow

Ask your frontier-model host to inspect `capabilities` and `inventory`, create a training
dataset and a separate held-out dataset, evaluate the base, train, then evaluate again.
The frontier model orchestrates tool calls; local PyTorch performs the gradient computation.

`create_dataset` arguments:

```json
{
  "dataset": "train-v1",
  "examples": [
    {"text": "Question: What color is the sky? Answer: blue."},
    {"text": "Question: What is 2 plus 2? Answer: 4."}
  ]
}
```

Or place `datasets/train-v1.jsonl` in the workspace, with one `{"text": "..."}` or
`{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}`
object per line. Chat datasets require the local tokenizer's `chat_template`.

`start_fine_tune` arguments:

```json
{
  "request": {
    "model": "my-model",
    "dataset": "train-v1",
    "output": "my-adapter-v1",
    "method": "lora",
    "device": "auto",
    "max_steps": 100,
    "learning_rate": 0.0002,
    "batch_size": 1,
    "gradient_accumulation": 4,
    "max_length": 512,
    "lora_rank": 8,
    "lora_alpha": 16,
    "target_modules": "all-linear"
  }
}
```

Poll `job_status` with the returned `job_id` until completed, failed, cancelled, or interrupted.
LoRA returns an **adapter**, not a full model. Evaluate/generate with both `model: "my-model"`
and `adapter: "my-adapter-v1"`; merge with `merge_adapter` to get a standalone model.
Set `method: "full"` to train all parameters and save a standalone checkpoint.

`start_evaluation` arguments:

```json
{
  "request": {
    "model": "my-model",
    "adapter": "my-adapter-v1",
    "dataset": "held-out",
    "max_examples": 100,
    "max_length": 512
  }
}
```

Training cycles shuffled examples until `max_steps` **optimizer steps**, with the configured
number of microbatches per step. It uses AdamW, gradient clipping, and a constant learning rate.
All nonpadding next tokens contribute to loss, including system/user tokens in chat examples.
This is causal-language-model training, with no assistant-only masking or instruction-specific
collator. Examples are truncated independently; there is no packing. Seeds are recorded but
GPU kernels may not be bitwise deterministic. Evaluation reports token-weighted loss and
perplexity over the selected, truncated examples. Use the same evaluation settings for comparisons.

Jobs persist under `runs/<job_id>/` with request fingerprints, latest status, worker logs,
and training metrics. Workers can outlive a server restart. Only one compute job runs per
workspace. Cancellation is cooperative between microbatches, generation tokens, and save
boundaries; model loading or a single forward/backward pass can take time. Cancelled or failed
jobs do not publish partial checkpoints. Training resume and optimizer checkpoint recovery
are not implemented.

## Direct weight editing

Inspect actual tensor names first. Preview this request, substituting a tensor from your model:

```json
{
  "request": {
    "model": "my-model",
    "edits": [{
      "tensor": "model.layers.0.self_attn.q_proj.weight",
      "operation": "scale",
      "value": 0.99,
      "ranges": [[0, 16], [0, 16]]
    }]
  }
}
```

Then call `apply_weight_edits` with the **same request**, an unused `output` name, and
`expected_plan_token` set to the preview's `plan_token`. The token binds the complete source
checkpoint bytes, exact edits, and tensor asset bytes. Changed inputs require a new preview.
Ranges are `[start, stop]` with exclusive stops, one per axis; omit them for the whole tensor.

| Operation | Parameters | Numerical update |
| --- | --- | --- |
| `scale` | `value` | `W = W * value` |
| `add` | `value` | `W = W + value` |
| `set` | `value` | Fill selected weights with `value` |
| `zero` | — | Fill selected weights with zero |
| `clamp` | `minimum`, `maximum` | Clamp selected values |
| `noise` | `value`, `seed` | Add seeded Gaussian noise with standard deviation `value` |
| `replace_tensor` | `asset`, `source_tensor` | Replace with the exact-shape asset tensor |
| `add_tensor` | `asset`, `source_tensor`, `alpha` | `W = W + alpha * source` |
| `low_rank` | `asset`, `left_tensor`, `right_tensor`, `alpha` | `W = W + alpha * (left @ right)` |

`create_tensor_asset` accepts arrays such as
`{"asset": "delta-v1", "tensors": {"left": [[1.0], [0.0]], "right": [[0.1, 0.2]]}}`.
Larger tensor assets can be placed directly in `tensors/delta-v1.safetensors`.
Low-rank `alpha` is the direct multiplier, without an implicit division by rank.
All shape checks are exact; broadcasting is disabled. Scalar and low-rank tensor operations
modify the selected slice. Operations on each tensor run in request order; different tensors
are grouped by shard. Preview statistics follow that shard order.

Edits preserve tensor names, shapes, dtypes, shard layout, and safetensors metadata. Integer,
float8, quantized, and nonfinite selected weights are rejected. Arithmetic uses float32
(float64 for float64 source weights), checks for overflow after conversion back to the
original dtype, and writes only to a copied checkpoint. Tied weights remain subject to the
model loader's tying rules; inspect and evaluate the reloaded model to confirm the intended
effect. Direct edits change numbers; they do not guarantee a specific semantic behavior.

Use `compare_weights` and held-out evaluation/generation to assess the result. To roll back,
point your inference runtime at the original version. Running inference services must reload
the new checkpoint; this server does not patch a separate process's GPU memory.

## Scope and resource use

- Supported model execution: local, unquantized Hugging Face **causal language models** with
  architectures available in the pinned Transformers major version. Local tokenizer assets
  are required. Generic safetensors tensors and PEFT adapters can be inspected and edited.
- GGUF/Ollama files, QLoRA, diffusion training, distributed training, custom Python model
  architectures, remote downloads, arbitrary code execution, and live in-memory patching of
  external inference processes are outside this version's scope.
- Affected shards are loaded into CPU memory during editing. Preview and apply perform full
  content hashing; apply copies the complete checkpoint. Budget disk space for a full extra
  model version and RAM for a shard plus working tensors. Full fine-tuning also needs gradients
  and optimizer state; LoRA reduces trainable state but still loads the base model.
- Limits: 64 MiB / 100,000 rows per on-disk dataset, 10,000 rows per create call, 100M elements
  per selected tensor edit, 1M values per MCP-created tensor asset, 256 MiB per on-disk tensor
  asset. Model sizes are limited by available hardware. Names are simple identifiers within
  the workspace; symlinks and path traversal are rejected.
- Model/tokenizer loading uses local-only options, disables remote custom code, and requires
  safetensors weights. Compute workers run with Hugging Face offline settings. The host controls
  who can invoke these local tools. Keep the workspace private and avoid modifying source files
  externally while operations run.

## Develop and verify

```bash
python -m pip install -e '.[dev]'
ruff check .
pytest -q
python -m build
```

Tests construct a tiny GPT-2-style model and tokenizer locally, with no downloaded checkpoint.
They exercise real training, adapter merging, evaluation, generation, weight-edit transactions,
and an actual stdio MCP client/server connection. `examples/make_tiny_model.py` creates a
small runnable workspace for experimentation.

The full test result and exact tested dependencies are included in `VALIDATION.md`.

## API references

- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [Transformers 4.57 model loading and saving](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/model)
- [PEFT LoRA](https://huggingface.co/docs/peft/package_reference/lora)
- [Safetensors PyTorch API](https://huggingface.co/docs/safetensors/api/torch)

MIT licensed; see `LICENSE`.
