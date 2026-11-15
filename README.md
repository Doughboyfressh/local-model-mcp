# Local Model MCP

An MCP server that lets an MCP-capable frontier-model host fine-tune local causal language models
and directly edit their checkpoint tensors. Runs entirely on your machine over **stdio**.

**Implemented:** LoRA and full-parameter fine-tuning, tensor inspection, slice editing,
tensor replacement, additive and low-rank updates, evaluation, generation, adapter merging,
background jobs, and checkpoint provenance. Every mutation creates a new checkpoint version.
Dense Qwen2/Qwen2.5 and Qwen3 also support training-free identity block insertion and MLP
widening, with strict reload checks and numerical verification before publication.

## Windows, macOS, and Linux

The same Python package and MCP tools run natively on all three operating systems.
No WSL, Docker, Bash, or PowerShell activation script is required to run the server.

| Platform | Baseline | Optional acceleration | Default workspace |
| --- | --- | --- | --- |
| Windows (x64) | CPU | NVIDIA CUDA with a matching PyTorch build | `%LOCALAPPDATA%\local-model-mcp` |
| macOS (Apple silicon) | CPU | Apple MPS when available | `~/Library/Application Support/local-model-mcp` |
| Linux (x64) | CPU | NVIDIA CUDA with a matching PyTorch build | `${XDG_DATA_HOME:-~/.local/share}/local-model-mcp` |

`device: "auto"` selects CUDA, then Apple MPS, then CPU according to availability.
`device: "cpu"` works without a GPU. Training and weight-edit tools have the same API
on each OS. Names reject Windows device names and trailing dots on every platform so
workspaces remain portable. UTF-8 datasets and paths containing spaces or Unicode are supported.

Native macOS wheels for the required modern PyTorch releases target Apple silicon;
Intel macOS is outside the supported binary-install matrix. See
[PyTorch's macOS x86 deprecation announcement](https://dev-discuss.pytorch.org/t/pytorch-macos-x86-builds-deprecation-starting-january-2024/1690).

GitHub Actions runs the complete CPU suite against the **installed wheel** on Windows,
macOS, and Linux with Python 3.11 and 3.12. GPU paths need their respective hardware.

## Install and connect

Requires Python 3.11+ and enough RAM/VRAM for the chosen model. The package uses the MCP Python
SDK 1.x and Transformers 4.x intentionally; dependencies exclude their next major versions.

Clone this repository and enter `local-model-mcp`, then use the commands for your OS.
These commands invoke the virtual environment directly, so activation-policy changes
are unnecessary. Install Git and Python 3.11+ first; the examples use Python 3.12 on Windows.

**Windows — PowerShell:**

```powershell
py -3.12 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install "torch>=2.6,<3" --index-url https://download.pytorch.org/whl/cpu
& .\.venv\Scripts\python.exe -m pip install -e .
& .\.venv\Scripts\python.exe examples/make_tiny_model.py --root ".\model-lab"
& .\.venv\Scripts\python.exe -m local_model_mcp.server --root ".\model-lab" --print-mcp-config
```

**macOS — Terminal (Apple silicon):**

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install "torch>=2.6,<3"
.venv/bin/python -m pip install -e .
.venv/bin/python examples/make_tiny_model.py --root "./model-lab"
.venv/bin/python -m local_model_mcp.server --root "./model-lab" --print-mcp-config
```

**Linux — Terminal:**

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install "torch>=2.6,<3" --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -e .
.venv/bin/python examples/make_tiny_model.py --root "./model-lab"
.venv/bin/python -m local_model_mcp.server --root "./model-lab" --print-mcp-config
```

For NVIDIA GPUs, use [PyTorch's installation selector](https://pytorch.org/get-started/locally/)
in place of the CPU-only PyTorch command. The macOS package already includes the MPS backend
where supported. Training defaults to float32; bfloat16 training uses autocast while
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

Copy the JSON printed by `--print-mcp-config` into your MCP host's configuration.
The command generates absolute paths for the current Python environment and workspace,
including correct Windows backslash escaping. It does not create or modify the workspace.
The resulting configuration has this shape (the generated paths depend on your OS):

```json
{
  "mcpServers": {
    "local-model-lab": {
      "command": "/absolute/path/local-model-mcp/.venv/bin/python",
      "args": ["-m", "local_model_mcp.server", "--root", "/absolute/path/model-lab"]
    }
  }
}
```

On Windows the generated command is the absolute path to `.venv\\Scripts\\python.exe`.
The host needs to support launching **local stdio MCP servers**. A hosted-only connector
cannot access your machine's checkpoints by itself. This package does not provide a public
HTTP endpoint or connect itself to a hosted chat service.

Run manually with `local-model-mcp --root /absolute/path/model-lab`. Without `--root`,
`LOCAL_MODEL_MCP_ROOT` is used, then the native workspace directory in the table above.
When upgrading from 0.1.0, an existing legacy `~/.local/share/local-model-mcp` workspace
is reused if no native workspace exists. An explicit `--root` always takes priority.
Move the project or virtual environment to another computer? Regenerate the MCP config
on that computer so it uses the right interpreter path.

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
| `inspect_qwen_architecture` | Validate a Qwen layout and inspect structural-edit limits |
| `preview_qwen_architecture_change` | Preview a training-free structural change and obtain a plan token |
| `start_qwen_architecture_change` | Rewrite, reload, and verify a new Qwen architecture in a background job |
| `job_status` / `list_jobs` / `cancel_job` | Persistent progress, results, cancellation |

## Change Qwen architecture without training

Version 0.2.0 adds structural edits for **canonical dense Qwen2/Qwen2.5 and Qwen3 causal
language models**, identified by `model_type: "qwen2"` or `"qwen3"`. These operations
perform no gradient updates, optimizer steps, pretraining, fine-tuning, or distillation.

| Operation | Structural change | Initialization |
| --- | --- | --- |
| `insert_identity_blocks` | Add one block after each selected original layer | Copy that layer, then zero the attention output projection (including its bias if present) and MLP down projection. Residual paths initially pass the input through unchanged. |
| `widen_mlp` | Increase every block's `intermediate_size` | Keep all original gate/up rows and down columns. Initialize additional gate/up rows by cycling existing rows, and give the additional channels zero down-projection columns. |

Both preserve the inference equations in exact arithmetic for the supported layout.
Widening uses **zero output connections**, rather than splitting/rounding the original
weights. Original parameter entries and storage dtypes are retained. Additional layers
or channels increase storage/compute and do not add new learned abilities by themselves.
This feature does not remove layers, change attention-head counts or hidden-state width,
or convert Qwen into another model family.

Supported checkpoints must use unquantized safetensors and SiLU MLPs with full attention.
MoE, multimodal/VL, hybrid/recurrent, sliding-window, custom-code, and adapter-only checkpoints
are rejected. Qwen2.5 dense models use the Qwen2 layout. Other Qwen-branded architectures
are not implied to be supported; the inspector validates the actual configuration,
complete tensor names, and shapes against the installed Transformers implementation.

**Workflow:** place your full local checkpoint under `models/my-qwen`, then call
`inspect_qwen_architecture` with `{"model": "my-qwen"}`. Preview an insertion:

```json
{
  "request": {
    "model": "my-qwen",
    "operation": "insert_identity_blocks",
    "insert_after": [0]
  }
}
```

Indices refer to original layers and start at zero. They must be unique; up to 32 blocks
can be inserted per operation, with at most 512 resulting layers. The preview reports
before/after dimensions, parameter counts, estimated output weight bytes, and `plan_token`.
For widening, use `operation: "widen_mlp"` and `new_intermediate_size` instead of `insert_after`.
The new width must exceed the original and be at most 4x larger (maximum 1,000,000).

Launch `start_qwen_architecture_change` with the same structural request, a new output name,
and the preview token:

```json
{
  "request": {
    "model": "my-qwen",
    "operation": "insert_identity_blocks",
    "insert_after": [0],
    "output": "my-qwen-deeper-v1",
    "expected_plan_token": "<plan_token returned by preview>",
    "validation_prompts": ["Explain a simple idea.", "The sky is"]
  }
}
```

Poll `job_status` with the returned job ID. The job:

1. Copies the source checkpoint and checks the content-bound plan token.
2. Records reference logits with gradients disabled.
3. Rewrites weight shards and updates the config, layer indices, and shard index.
4. Strictly reloads the candidate and compares its logits with the original, including
   next-token decoding with a KV cache.
5. Publishes the new standalone checkpoint only if all comparisons pass.

The original checkpoint is retained. Stale plans, failed verification, and cancellation
publish no output model. Verification details are returned in job status, saved under
`runs/<job_id>/architecture_validation.json`, and included in the output's provenance manifest.

Verification uses CPU float32 eager attention and fixed `atol=1e-5`, `rtol=1e-4` tolerances.
It always includes deterministic synthetic token probes. Optional `validation_prompts`
add up to four raw-text probes and require local tokenizer assets; each must contain at
least two tokens. `max_validation_tokens` defaults to 32 and is limited to 2–128.
Logit buffers are limited to 32 million elements. Probe inference runs on the source and
candidate sequentially. Weight rewriting targets 512 MiB shards; an individual larger
tensor occupies its own shard. Budget disk for the copied source plus rewritten weights,
and RAM for the expanded model loaded in float32 and comparison buffers.

**A passing comparison is a finite numerical check, not a proof for every input, precision,
GPU kernel, or external inference engine.** Test your deployment backend and held-out
dataset before using a changed checkpoint. Probe timings are diagnostics, not a controlled
speed benchmark. Larger identity-initialized models usually consume more resources while
initially doing the same computation.

For an offline smoke demo, use your virtual environment's Python to run:

```bash
python examples/make_tiny_qwen.py --root ./model-lab --family qwen2
python examples/make_tiny_qwen.py --root ./model-lab --family qwen3
```

These create tiny **random, untrained** checkpoints with the real Qwen module layouts.
They validate the transformation workflow, not pretrained-model task quality. Their initial
MLP width is 24; a widening preview can use `new_intermediate_size: 48`.

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

The recorded local test result and exact tested dependencies are in `VALIDATION.md`.
The [Actions page](https://github.com/Doughboyfressh/local-model-mcp/actions) shows the latest
native Windows/macOS/Linux results. The suite uses CPU explicitly on every runner and
exercises Unicode workspace paths, real worker subprocesses, checkpoint edits, training,
and a stdio MCP client launched from the generated configuration.

## API references

- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [Transformers 4.57 model loading and saving](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/model)
- [PEFT LoRA](https://huggingface.co/docs/peft/package_reference/lora)
- [Safetensors PyTorch API](https://huggingface.co/docs/safetensors/api/torch)
- [Qwen2 model/config reference](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/qwen2)
- [Qwen3 model/config reference](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/qwen3)
- [LLaMA Pro identity-initialized block expansion](https://arxiv.org/abs/2401.02415)
- [Net2Net function-preserving transformations](https://arxiv.org/abs/1511.05641)

MIT licensed; see `LICENSE`.
