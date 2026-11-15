"""Training-free structural transforms for canonical dense Qwen checkpoints."""

import gc
import hashlib
import json
import math
import re
import shutil
import time
from dataclasses import dataclass

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .schemas import QwenArchitectureRequest
from .weights import Checkpoint
from .workspace import json_write, read_json, replace_file

FAMILIES = {"qwen2": "Qwen2ForCausalLM", "qwen3": "Qwen3ForCausalLM"}
LAYER_KEY = re.compile(r"model\.layers\.(\d+)\.(.+)\Z")
DTYPE_BYTES = {"F16": 2, "BF16": 2, "F32": 4, "F64": 8}
SHARD_BYTES = 512 * 1024 * 1024
MAX_LOGIT_ELEMENTS = 32_000_000
ATOL = 1e-5
RTOL = 1e-4


@dataclass(frozen=True)
class TensorPlan:
    target: str
    source: str
    shape: tuple[int, ...]
    action: str


def specification(request):
    return {key: getattr(request, key) for key in QwenArchitectureRequest.model_fields}


def validate_checkpoint(checkpoint):
    from transformers import Qwen2Config, Qwen2ForCausalLM, Qwen3Config, Qwen3ForCausalLM

    if checkpoint.kind != "model":
        raise ValueError("Structural changes require a full Qwen model, not an adapter")
    config_path = checkpoint.path / "config.json"
    if not config_path.is_file():
        raise ValueError("A local Qwen config.json is required")
    raw = read_json(config_path)
    family = raw.get("model_type")
    if family not in FAMILIES:
        raise ValueError("Supported: dense Qwen2/Qwen2.5 (qwen2) and Qwen3 (qwen3) only")
    if raw.get("auto_map") or raw.get("architectures") not in (None, [FAMILIES[family]]):
        raise ValueError("Custom architectures are unsupported; use the canonical Qwen causal LM")
    for key, maximum in (
        ("num_hidden_layers", 512),
        ("hidden_size", 131072),
        ("intermediate_size", 1_000_000),
        ("vocab_size", 2_000_000),
    ):
        value = raw.get(key)
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError(f"Unsupported Qwen {key}")
    if raw.get("hidden_act", "silu") != "silu" or raw.get("use_sliding_window", False):
        raise ValueError("Structural edits currently require SiLU and full attention")
    configs = {"qwen2": (Qwen2Config, Qwen2ForCausalLM), "qwen3": (Qwen3Config, Qwen3ForCausalLM)}
    config_class, model_class = configs[family]
    config = config_class.from_dict(raw)
    if config.max_position_embeddings < 4:
        raise ValueError("Qwen context window must contain at least four tokens")
    if config.layer_types != ["full_attention"] * config.num_hidden_layers:
        raise ValueError("Sliding-window and hybrid attention configurations are unsupported")
    # Build only metadata, not model-sized parameter storage.
    with torch.device("meta"):
        skeleton = model_class(config)
    expected = {key: tuple(value.shape) for key, value in skeleton.state_dict().items()}
    del skeleton
    actual = set(checkpoint.catalog)
    missing, extra = set(expected) - actual, actual - set(expected)
    if config.tie_word_embeddings and "model.embed_tokens.weight" in actual:
        missing.discard("lm_head.weight")
    if missing or extra:
        raise ValueError(
            f"Noncanonical Qwen weights: missing={sorted(missing)}, extra={sorted(extra)}"
        )
    for key, metadata in checkpoint.catalog.items():
        if tuple(metadata["shape"]) != expected[key]:
            raise ValueError(f"Unexpected Qwen tensor shape: {key}")
        if metadata["dtype"] not in DTYPE_BYTES:
            raise ValueError("Structural edits require unquantized floating-point Qwen weights")
    return raw, config


def inspect(workspace, model):
    checkpoint = Checkpoint(workspace.model(model))
    _, config = validate_checkpoint(checkpoint)
    return {
        "model": model,
        "model_type": config.model_type,
        "architecture": FAMILIES[config.model_type],
        "num_hidden_layers": config.num_hidden_layers,
        "hidden_size": config.hidden_size,
        "intermediate_size": config.intermediate_size,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "parameters": sum(item["elements"] for item in checkpoint.catalog.values()),
        "operations": ["insert_identity_blocks", "widen_mlp"],
        "training_required": False,
        "limits": {"insertions": min(32, config.num_hidden_layers), "max_mlp_factor": 4},
        "notes": [
            "Inference preservation, not additional learned abilities",
            "Full attention only; no MoE, multimodal, hybrid, or quantized variants",
        ],
    }


def tensor_plans(checkpoint, config, request):
    if request.operation == "insert_identity_blocks":
        if max(request.insert_after) >= config.num_hidden_layers:
            raise ValueError("insert_after indices must refer to existing zero-based Qwen layers")
        if config.num_hidden_layers + len(request.insert_after) > 512:
            raise ValueError("The expanded model cannot exceed 512 layers")
        layout = []
        for index in range(config.num_hidden_layers):
            layout.append((index, False))
            if index in request.insert_after:
                layout.append((index, True))
        width = config.intermediate_size
    else:
        width = request.new_intermediate_size
        if not config.intermediate_size < width <= 4 * config.intermediate_size:
            raise ValueError(
                "new_intermediate_size must exceed the original and be at most 4x larger"
            )
        layout = [(index, False) for index in range(config.num_hidden_layers)]
    targets = {}
    for destination, (source, identity) in enumerate(layout):
        targets.setdefault(source, []).append((destination, identity))
    plans = []
    for key, metadata in sorted(checkpoint.catalog.items()):
        match = LAYER_KEY.fullmatch(key)
        if not match:
            plans.append(TensorPlan(key, key, tuple(metadata["shape"]), "copy"))
            continue
        source, suffix = int(match[1]), match[2]
        for destination, identity in targets[source]:
            action = "copy"
            shape = tuple(metadata["shape"])
            if identity and suffix in {
                "self_attn.o_proj.weight",
                "self_attn.o_proj.bias",
                "mlp.down_proj.weight",
            }:
                action = "zero"
            elif request.operation == "widen_mlp":
                if suffix in {"mlp.gate_proj.weight", "mlp.up_proj.weight"}:
                    action, shape = "repeat_rows", (width, shape[1])
                elif suffix == "mlp.down_proj.weight":
                    action, shape = "pad_columns", (shape[0], width)
            plans.append(TensorPlan(f"model.layers.{destination}.{suffix}", key, shape, action))
    return sorted(plans, key=lambda item: item.target), len(layout), width


def make_plan(checkpoint, request):
    raw, config = validate_checkpoint(checkpoint)
    plans, depth, width = tensor_plans(checkpoint, config, request)
    revision = checkpoint.revision()
    token = hashlib.sha256(
        json.dumps(
            {
                "plan_schema": 1,
                "source_revision": revision,
                "transformation": specification(request),
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    before = sum(item["elements"] for item in checkpoint.catalog.values())
    after = sum(math.prod(item.shape) for item in plans)
    report = {
        "model": request.model,
        "model_type": config.model_type,
        "source_revision": revision,
        "plan_token": token,
        "transformation": specification(request),
        "strategy": (
            "identity_output_projections"
            if request.operation == "insert_identity_blocks"
            else "copied_inputs_zero_output_columns"
        ),
        "training_required": False,
        "before": {
            "num_hidden_layers": config.num_hidden_layers,
            "intermediate_size": config.intermediate_size,
            "parameters": before,
        },
        "after": {"num_hidden_layers": depth, "intermediate_size": width, "parameters": after},
        "estimated_weight_bytes": sum(
            math.prod(item.shape) * DTYPE_BYTES[checkpoint.catalog[item.source]["dtype"]]
            for item in plans
        ),
        "preservation": "Mathematical inference equivalence; numerical validation is required",
    }
    return raw, config, plans, report


def preview(workspace, request):
    return make_plan(Checkpoint(workspace.model(request.model)), request)[3]


def validation_inputs(path, config, request):
    from transformers import AutoTokenizer

    maximum = min(request.max_validation_tokens, config.max_position_embeddings)
    generator = torch.Generator().manual_seed(1729)
    probes = [
        (
            "synthetic-random",
            torch.randint(config.vocab_size, (1, min(8, maximum)), generator=generator),
        ),
        (
            "synthetic-boundaries",
            (torch.arange(min(5, maximum)).reshape(1, -1) * (config.vocab_size - 1))
            % config.vocab_size,
        ),
    ]
    if request.validation_prompts:
        tokenizer = AutoTokenizer.from_pretrained(
            path, local_files_only=True, trust_remote_code=False
        )
        for index, prompt in enumerate(request.validation_prompts):
            ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=maximum)[
                "input_ids"
            ]
            if ids.shape[1] < 2:
                raise ValueError("Each validation prompt must tokenize to at least two tokens")
            probes.append((f"prompt-{index}", ids))
    if sum((ids.numel() + 1) * config.vocab_size for _, ids in probes) > MAX_LOGIT_ELEMENTS:
        raise ValueError(
            "Validation exceeds the logit memory budget; reduce prompt count/token length"
        )
    return probes


def inference_records(path, probes, check):
    from transformers import AutoModelForCausalLM

    model, info = AutoModelForCausalLM.from_pretrained(
        path,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        torch_dtype=torch.float32,
        attn_implementation="eager",
        output_loading_info=True,
    )
    try:
        if any(
            info.get(key)
            for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
        ):
            raise ValueError(f"Checkpoint did not reload strictly: {info}")
        model.eval().requires_grad_(False)
        records = []
        started = time.perf_counter()
        with torch.inference_mode():
            for label, ids in probes:
                check()
                logits = model(
                    input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False
                ).logits
                prefix = model(
                    input_ids=ids[:, :-1],
                    attention_mask=torch.ones_like(ids[:, :-1]),
                    use_cache=True,
                )
                cached = model(
                    input_ids=ids[:, -1:],
                    attention_mask=torch.ones_like(ids),
                    past_key_values=prefix.past_key_values,
                    use_cache=True,
                ).logits
                if not (torch.isfinite(logits).all() and torch.isfinite(cached).all()):
                    raise ValueError("Nonfinite logits during Qwen structural verification")
                if not torch.allclose(logits[:, -1:], cached, atol=ATOL, rtol=RTOL):
                    raise ValueError("The checkpoint's own cached and uncached inference disagree")
                records.append(
                    {
                        "label": label,
                        "tokens": ids.numel(),
                        "logits": logits.clone(),
                        "cached": cached.clone(),
                    }
                )
        return records, time.perf_counter() - started
    finally:
        del model
        gc.collect()


def transformed_tensor(checkpoint, plan):
    with safe_open(
        checkpoint.path / checkpoint.mapping[plan.source], framework="pt", device="cpu"
    ) as stream:
        source = stream.get_tensor(plan.source)
        if not torch.isfinite(source).all():
            raise ValueError(f"Nonfinite Qwen weights: {plan.source}")
        if plan.action == "zero":
            result = torch.zeros_like(source)
        elif plan.action == "repeat_rows":
            result = torch.empty(plan.shape, dtype=source.dtype)
            for start in range(0, plan.shape[0], source.shape[0]):
                count = min(source.shape[0], plan.shape[0] - start)
                result[start : start + count].copy_(source[:count])
        elif plan.action == "pad_columns":
            result = torch.zeros(plan.shape, dtype=source.dtype)
            result[:, : source.shape[1]].copy_(source)
        else:
            result = source.clone()
        return result.contiguous()


def rewrite_weights(checkpoint, plans, check, update):
    temporary = checkpoint.path / ".architecture-weights"
    temporary.mkdir()
    groups, group, group_bytes, total_bytes = [], [], 0, 0
    for plan in plans:
        size = math.prod(plan.shape) * DTYPE_BYTES[checkpoint.catalog[plan.source]["dtype"]]
        if group and group_bytes + size > SHARD_BYTES:
            groups.append(group)
            group, group_bytes = [], 0
        group.append(plan)
        group_bytes += size
        total_bytes += size
    if group:
        groups.append(group)
    mapping, files = {}, []
    for index, group in enumerate(groups, 1):
        check()
        filename = (
            "model.safetensors"
            if len(groups) == 1
            else (f"model-{index:05d}-of-{len(groups):05d}.safetensors")
        )
        tensors = {}
        for plan in group:
            check()
            tensors[plan.target] = transformed_tensor(checkpoint, plan)
            mapping[plan.target] = filename
        save_file(tensors, temporary / filename, metadata={"format": "pt"})
        del tensors
        files.append(filename)
        update(phase="rewriting_weights", shards_written=index, shards_total=len(groups))
    gc.collect()
    for old in set(checkpoint.mapping.values()):
        (checkpoint.path / old).unlink()
    (checkpoint.path / "model.safetensors.index.json").unlink(missing_ok=True)
    for filename in files:
        replace_file(temporary / filename, checkpoint.path / filename)
    temporary.rmdir()
    if len(groups) > 1:
        json_write(
            checkpoint.path / "model.safetensors.index.json",
            {
                "metadata": {"total_size": total_bytes},
                "weight_map": mapping,
            },
        )


def compare_records(reference, candidate):
    comparisons = []
    for before, after in zip(reference, candidate, strict=True):
        full_ok = torch.allclose(before["logits"], after["logits"], atol=ATOL, rtol=RTOL)
        cached_ok = torch.allclose(before["cached"], after["cached"], atol=ATOL, rtol=RTOL)
        comparisons.append(
            {
                "probe": before["label"],
                "tokens": before["tokens"],
                "max_abs_logit_delta": (before["logits"] - after["logits"]).abs().max().item(),
                "max_abs_cached_delta": (before["cached"] - after["cached"]).abs().max().item(),
                "passed": bool(full_ok and cached_ok),
            }
        )
    return {
        "passed": all(item["passed"] for item in comparisons),
        "probes": comparisons,
        "atol": ATOL,
        "rtol": RTOL,
        "device": "cpu",
        "dtype": "float32",
        "attention_implementation": "eager",
        "scope": "Finite probes with and without KV cache; not a proof for all inputs/backends",
    }


def run_change(workspace, request, *, check, update, publish, report_path, job_id):
    update(phase="snapshotting", training_performed=False)
    with workspace.staging() as stage:

        def copy_file(source, target):
            check()
            return shutil.copy2(source, target)

        shutil.copytree(
            workspace.model(request.model), stage, dirs_exist_ok=True, copy_function=copy_file
        )
        check()
        checkpoint = Checkpoint(stage)
        raw, config, plans, plan = make_plan(checkpoint, request)
        if plan["plan_token"] != request.expected_plan_token:
            raise ValueError("Stale Qwen architecture plan; preview the exact request again")
        probes = validation_inputs(stage, config, request)
        update(phase="reference_inference")
        reference, before_seconds = inference_records(stage, probes, check)
        check()
        update(phase="rewriting_weights")
        rewrite_weights(checkpoint, plans, check, update)
        raw["num_hidden_layers"] = plan["after"]["num_hidden_layers"]
        raw["intermediate_size"] = plan["after"]["intermediate_size"]
        raw["layer_types"] = ["full_attention"] * raw["num_hidden_layers"]
        json_write(stage / "config.json", raw)
        update(phase="verification_inference")
        validate_checkpoint(Checkpoint(stage))
        candidate, after_seconds = inference_records(stage, probes, check)
        verification = compare_records(reference, candidate)
        verification["diagnostic_forward_seconds"] = {
            "before": before_seconds,
            "after": after_seconds,
        }
        verification["timing_note"] = (
            "One CPU probe pass per model, not a controlled speed benchmark"
        )
        json_write(report_path, verification)
        update(verification=verification)
        if not verification["passed"]:
            raise ValueError("Qwen architecture verification failed; no model was published")
        check()
        json_write(
            stage / "local_model_manifest.json",
            {
                "operation": "qwen_architecture",
                "source": request.model,
                "source_revision": plan["source_revision"],
                "plan_token": plan["plan_token"],
                "transformation": specification(request),
                "training_performed": False,
                "before": plan["before"],
                "after": plan["after"],
                "verification": verification,
                "job_id": job_id,
            },
        )
        publish(stage, request.output)
    return {
        "model": request.output,
        "source": request.model,
        "training_performed": False,
        "before": plan["before"],
        "after": plan["after"],
        "verification": verification,
    }
