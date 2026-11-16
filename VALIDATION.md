# Validation

Version 0.4.0 validation uses a fresh installer-created environment on Linux x86_64,
Python 3.12.14, and CPU PyTorch.

## Results

- `ruff check .`: passed.
- `pytest -q` from the installer-created environment: **109 passed in 155.59 seconds**,
  including 11 installer and 24 benchmark checks.
- `python -m build`: source distribution and Python wheel built successfully.
- The built wheel installs successfully; both installed console entry points launch.
- The generated MCP configuration launches a real stdio server from a workspace
  whose path includes spaces and Japanese characters.

The CI workflow builds the wheel and installs it through `install.py` in a fresh,
isolated environment before running the full CPU suite on
`windows-latest`, `macos-latest` (Apple silicon), and `ubuntu-latest`, using both Python
3.11 and 3.12. The [Actions page](https://github.com/Doughboyfressh/local-model-mcp/actions)
is authoritative for each commit's native runner results; the numbers above are the
recorded local run, not a claim about a completed remote workflow.

The tests construct a tiny random GPT-2-style model and a local WordLevel tokenizer.
No pretrained model checkpoint is downloaded. They verify:

- An actual stdio MCP connection: initialization, all 22 tool schemas, annotations,
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

## Cross-platform installer

The local installation ran `install.py` against the built wheel with its default
CPU PyTorch selection. It created an isolated environment, installed dependencies,
passed `pip check`, imported the runtime dependencies, and completed a tiny CPU
tensor operation. Both console entry points launched. Its generated MCP JSON uses
the new environment's interpreter and a workspace path containing spaces and Japanese
characters; no model workspace or model weights were created by installation.

The 11 installer checks cover platform-specific paths and package sources, unsupported
architectures and Python/venv requirements, dry-run side effects, unrelated directory
preservation, workspace/environment separation, concurrent installation rejection,
retry after failure, shell-free subprocess calls, and offline model lookup settings.
These checks mock installation commands; the fresh local install and native CI runs
exercise the real dependency installation and generated configuration.

The installer checks CPU execution and reports detected backend availability. It does
not validate CUDA/MPS kernels, download pretrained models, or change MCP host settings.

## Qwen architecture transformations

The additional 37 Qwen checks use tiny, randomly initialized models with the actual
Transformers Qwen2 and Qwen3 module layouts. They perform no checkpoint downloads and
do not establish task quality for a pretrained model. They cover:

- Identity block insertion and MLP widening for both dense model families, starting
  from float32, float16, and bfloat16 saved weights and sharded input checkpoints.
- Numerical comparisons of full-sequence logits and KV-cache next-token logits in
  CPU float32 eager inference, followed by greedy generation from reloaded checkpoints.
- Exact retention of original MLP weight entries, zero new output connections, layer
  renumbering, updated configs, source fingerprint preservation, and strict reloads.
- Sharded output, tied embeddings, grouped-query attention, and Qwen3 attention bias.
- A guard that fails the test if architecture changes invoke backward or optimizer steps.
- Failed-equivalence injection: corrupting output weights fails verification, leaves
  a persistent failed report, and publishes no model. Cancellation also cleans staging.
- Rejection of stale source/edit plans, missing/extra/misshaped/nonfinite weights,
  invalid dimensions, and unsupported MoE, VL, hybrid, sliding-window, and custom variants.
- Actual background jobs and an end-to-end Qwen transformation through a stdio MCP client.

The mathematical transformations target the same inference function. The implemented
numerical gate checks finite probes with `atol=1e-5`, `rtol=1e-4`; this is not a guarantee
for all inputs, storage precisions, GPU kernels, or external model-serving runtimes.
Larger pretrained Qwen checkpoints and GPU execution remain deployment validation work.

## Isolated checkpoint benchmarks

The 24 benchmark tests use locally constructed random Qwen2 and Qwen3 fixtures,
including a widened checkpoint. They download no pretrained weights. They cover:

- Separate child processes for the reference and candidate, source fingerprint
  preservation, measured CPU timings and process RSS, and JSON/Markdown reports.
- Full-sequence and cached logits, fixed-length greedy output agreement, cache and
  repeat consistency, and optional token-weighted held-out loss/perplexity.
- Successful comparisons, deliberately changed weights, independent failed policy
  checks, mismatched tokenizers, stale inputs, insufficient context, and invalid options.
- Fixed workloads after EOS, no backward/optimizer calls, cancellation with child
  process cleanup, private tensor cleanup, and generated text opt-in.
- Actual CLI execution and a Qwen3 benchmark through the stdio MCP interface.
- Device selection, synchronization, and CUDA/MPS memory API seams using mocks.

Only CPU float32 runs were executed. Mocked CUDA/MPS checks establish API wiring,
not GPU execution or performance. Random fixture losses do not measure pretrained
model quality. A missing evaluation dataset is explicitly reported as quality not
measured; a completed job may still have failed comparison checks.

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
| numpy | 2.5.5 |
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
