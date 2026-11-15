# Validation

Version 0.1.1 validated locally on November 14, 2026, using Linux x86_64,
Python 3.12.14, and CPU PyTorch.

## Results

- `ruff check .`: passed.
- `pytest -q`: **37 passed in 54.06 seconds**.
- `python -m build`: source distribution and Python wheel built successfully.
- The generated MCP configuration launches a real stdio server from a workspace
  whose path includes spaces and Japanese characters.

The CI workflow builds and installs the wheel before running the full CPU suite on
`windows-latest`, `macos-latest` (Apple silicon), and `ubuntu-latest`, using both Python
3.11 and 3.12. The [Actions page](https://github.com/Doughboyfressh/local-model-mcp/actions)
is authoritative for each commit's native runner results; the numbers above are the
recorded local run, not a claim about a completed remote workflow.

The tests construct a tiny random GPT-2-style model and a local WordLevel tokenizer.
No pretrained model checkpoint is downloaded. They verify:

- An actual stdio MCP connection: initialization, all 17 tool schemas, annotations,
  tensor inspection, preview/apply, structured results, input errors, and a generation job.
- Real full-parameter training with PyTorch gradients, checkpoint saving, reload,
  evaluation, and generation.
- Real LoRA training, nonzero adapter weights, adapter evaluation, and merging into
  a standalone checkpoint whose evaluation loss matches the unmerged adapter.
- Weight editing on a trained adapter while preserving its base-model provenance.
- Cooperative cancellation, concurrent-job rejection, persistent status after creating
  a new server-side job manager, and absence of a published cancelled checkpoint.
- Slice scaling, low-rank updates, tensor replacement, scalar addition, clamping,
  zeroing, seeded noise, numerical comparisons, and original-file preservation.
- Stale-plan rejection after changes to source files, edits, or tensor assets.
- Transaction cleanup after failure, output-name collision rejection, metadata and
  shard-index preservation, and reloading an edited real sharded model.
- Path traversal/symlink rejection, invalid dataset schemas, invalid shard paths,
  ambiguous checkpoint layouts, quantized checkpoint rejection, integer-edit rejection,
  and float16 overflow rejection.
- Correct masking of padding when padding and EOS share a token ID.
- Portable model/dataset names, UTF-8 datasets and JSON, deterministic checkpoint
  fingerprints for nested paths, concurrent JSON readers, and Windows file-sharing
  collision recovery on both reads and writes.
- Native default workspace selection, discovery of legacy workspaces, environment
  overrides, and configuration generation without creating a workspace.

The symlink test may skip only when a Windows account lacks the symlink privilege;
normal model operation does not require that privilege. All other tests run natively.

## Numerical smoke test

On the tiny model, 15 full-training optimizer steps reduced evaluation loss from
approximately **2.78254 to 0.38177**. Perplexity changed from approximately **16.1600 to
1.46488**. Generation after `the sky is` returned `blue .` followed by EOS.

This is an execution and learning smoke test. Its synthetic evaluation examples repeat
the training text, so these numbers do not measure held-out generalization or establish
quality for a real pretrained model.

## Tested package versions

| Package | Version |
| --- | --- |
| torch | 2.15.0+cpu |
| mcp | 1.30.0 |
| transformers | 4.57.6 |
| peft | 0.18.1 |
| safetensors | 0.8.0 |
| pydantic | 2.14.3 |
| filelock | 3.32.7 |
| psutil | 7.2.2 |
| platformdirs | 4.14.5 |
| accelerate | 1.15.0 |
| tokenizers | 0.22.2 |
| huggingface-hub | 0.36.2 |
| numpy | 2.5.4 |
| pytest | 9.2.0 |
| ruff | 0.17.5 |

`requirements-tested.txt` records these versions as optional installation constraints.
The normal package metadata specifies compatible version ranges.

## Remaining hardware and scale validation

CUDA, Apple MPS, bfloat16 execution, and large checkpoints were not exercised in this
environment. The local run covers Linux CPU float32; native Windows/macOS CPU validation
runs through GitHub Actions. The platform matrix targets Windows/Linux x64 and macOS
Apple silicon; modern official PyTorch wheels do not support Intel macOS. Training resume,
distributed training, quantized models, GGUF, and external-process memory patching are
outside the implemented scope. See README for runtime and memory requirements.
