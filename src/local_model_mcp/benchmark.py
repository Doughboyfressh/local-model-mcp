"""Local benchmark preparation, isolated runners, comparisons, and reports."""

import hashlib
import json
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime

import torch
from safetensors.torch import load_file

from . import __version__
from .qwen import MAX_LOGIT_ELEMENTS, validate_checkpoint
from .runtime import background_process_options
from .weights import Checkpoint
from .workspace import json_write, read_json, replace_file, sha256


def tolerances(request):
    default = {"float32": (1e-5, 1e-4), "float16": (1e-3, 1e-3), "bfloat16": (1e-2, 1e-2)}
    atol, rtol = default[request.dtype]
    return (
        atol if request.atol is None else request.atol,
        rtol if request.rtol is None else request.rtol,
    )


def tokenizer_fingerprint(tokenizer):
    if not tokenizer.is_fast:
        raise ValueError("Qwen benchmarks require local fast tokenizer assets")
    backend = json.loads(tokenizer.backend_tokenizer.to_str())
    backend.pop("padding", None)
    backend.pop("truncation", None)
    payload = {
        "backend": backend,
        "special_tokens": tokenizer.special_tokens_map,
        "chat_template": tokenizer.chat_template,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def prepare_inputs(workspace, request):
    from transformers import AutoTokenizer

    checkpoints = [
        Checkpoint(workspace.model(value)) for value in (request.model, request.candidate)
    ]
    configs = [validate_checkpoint(value)[1] for value in checkpoints]
    if configs[0].vocab_size != configs[1].vocab_size:
        raise ValueError("Benchmark checkpoints must have equal output vocabulary sizes")
    tokenizers = [
        AutoTokenizer.from_pretrained(
            value.path,
            local_files_only=True,
            trust_remote_code=False,
            use_fast=True,
        )
        for value in checkpoints
    ]
    fingerprints = [tokenizer_fingerprint(value) for value in tokenizers]
    if fingerprints[0] != fingerprints[1]:
        raise ValueError(
            "Benchmark tokenizers differ; matching token IDs and text processing are required"
        )
    tokenizer = tokenizers[0]
    vocab = configs[0].vocab_size
    if max(tokenizer.get_vocab().values()) >= vocab:
        raise ValueError("Tokenizer contains IDs outside the model vocabulary")
    context = min(value.max_position_embeddings for value in configs)
    limit = min(request.max_prompt_tokens, context - request.max_new_tokens)
    if limit < 2:
        raise ValueError("Generation length leaves insufficient model context for a prompt")
    prompts, truncated = [], 0
    for text in request.prompts:
        ids = tokenizer(text, add_special_tokens=True)["input_ids"]
        truncated += len(ids) > limit
        ids = ids[:limit]
        if len(ids) < 2:
            raise ValueError("Each benchmark prompt must tokenize to at least two tokens")
        prompts.append(ids)
    if sum((len(ids) + 1) * vocab for ids in prompts) > MAX_LOGIT_ELEMENTS:
        raise ValueError("Logit memory budget exceeded; reduce prompt count or max_prompt_tokens")
    evaluation, eval_truncated = [], 0
    if request.dataset:
        for row in workspace.dataset(request.dataset)[: request.max_examples]:
            if row.messages is not None:
                if not tokenizer.chat_template:
                    raise ValueError("Chat evaluation requires a local chat_template")
                ids = tokenizer.apply_chat_template(
                    [value.model_dump() for value in row.messages],
                    tokenize=True,
                    add_generation_prompt=False,
                )
            else:
                ids = tokenizer(row.text, add_special_tokens=True)["input_ids"]
            length = min(request.max_eval_tokens, context)
            eval_truncated += len(ids) > length
            ids = ids[:length]
            if len(ids) < 2:
                raise ValueError("Each evaluation example must contain at least two tokens")
            if len(ids) * vocab > MAX_LOGIT_ELEMENTS:
                raise ValueError("Evaluation logit budget exceeded; reduce max_eval_tokens")
            evaluation.append(ids)
    return {
        "prompts": prompts,
        "evaluation": evaluation,
        "tokenizer_sha256": fingerprints[0],
        "truncated_prompts": truncated,
        "truncated_eval_examples": eval_truncated,
        "effective_prompt_limit": limit,
        "truncation_side": "right",
    }


def verify_inputs(workspace, request, fingerprints):
    for key in ("model", "candidate"):
        if Checkpoint(workspace.model(getattr(request, key))).revision() != fingerprints[key]:
            raise ValueError(f"Benchmark input {key} changed after job creation")
    if (
        request.dataset
        and sha256(workspace.path("datasets", request.dataset, ".jsonl")) != fingerprints["dataset"]
    ):
        raise ValueError("Benchmark dataset changed after job creation")


def wait_runner(workspace, job_id, role, private, check, update):
    log_path = private.parent / f"benchmark-{role}.log"
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "local_model_mcp.benchmark_runner",
                "--root",
                str(workspace.root),
                "--job",
                job_id,
                "--role",
                role,
            ],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            **background_process_options(),
        )
    latest = None
    try:
        while process.poll() is None:
            check()
            progress = private / "progress.json"
            if progress.is_file():
                current = read_json(progress)
                if current.get("role") == role and current != latest:
                    update(phase=f"benchmark_{role}", benchmark_progress=current)
                    latest = current
            time.sleep(0.1)
        check()
        result = private / f"{role}.json"
        if not result.is_file():
            raise ValueError(f"{role} runner exited without a result (code {process.returncode})")
        metrics = read_json(result)
        if process.returncode != 0 or metrics["status"] != "completed":
            raise ValueError(
                f"{role} benchmark failed: {metrics.get('error', 'runner interrupted')}"
            )
        return metrics
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


def compare_results(request, inputs, reference, candidate, private):
    atol, rtol = tolerances(request)
    first, second = [
        load_file(private / f"{role}.safetensors") for role in ("reference", "candidate")
    ]
    cases, matched, total = [], 0, 0
    for index in range(len(inputs["prompts"])):
        case = {"case": index + 1, "input_tokens": len(inputs["prompts"][index])}
        for suffix in ("full", "cached"):
            a, b = first[f"{index}_{suffix}"], second[f"{index}_{suffix}"]
            case[f"{suffix}_passed"] = bool(torch.allclose(a, b, atol=atol, rtol=rtol))
            case[f"{suffix}_max_abs_delta"] = (a - b).abs().max().item()
        a, b = (
            reference["cases"][index]["generated_ids"],
            candidate["cases"][index]["generated_ids"],
        )
        agreement = sum(x == y for x, y in zip(a, b, strict=True))
        matched += agreement
        total += len(a)
        case["generated_token_agreement"] = agreement / len(a)
        case["generated_tokens_match"] = a == b
        cases.append(case)
    numeric = all(item["full_passed"] and item["cached_passed"] for item in cases)
    agreement = matched / total
    consistency = all(
        value["cache_consistent"] and value["repeats_consistent"]
        for value in (reference, candidate)
    )
    quality = {
        "measured": False,
        "passed": None,
        "note": "No held-out dataset supplied; task quality was not measured",
    }
    if request.dataset:
        delta = candidate["evaluation"]["loss"] - reference["evaluation"]["loss"]
        quality = {
            "measured": True,
            "passed": delta <= request.max_loss_increase,
            "loss_increase": delta,
            "max_loss_increase": request.max_loss_increase,
            "note": "Token-weighted next-token loss; no task-specific accuracy metric",
        }
    passed = (
        numeric
        and consistency
        and agreement >= request.min_token_agreement
        and quality["passed"] is not False
    )
    return {
        "passed": bool(passed),
        "numeric_equivalence": numeric,
        "inference_consistent": consistency,
        "generated_token_agreement": agreement,
        "min_token_agreement": request.min_token_agreement,
        "atol": atol,
        "rtol": rtol,
        "quality": quality,
        "cases": cases,
    }


def percent_change(reference, candidate):
    return 100 * (candidate / reference - 1) if reference else None


def markdown_report(report):
    reference, candidate = report["reference"], report["candidate"]
    verdict = "PASS" if report["comparison"]["passed"] else "FAIL"
    rows = [
        "# Qwen checkpoint benchmark",
        "",
        f"Configured checks: **{verdict}**",
        "",
        f"Reference: `{reference['model']}`. Candidate: `{candidate['model']}`.",
        "",
        "| Metric | Reference | Candidate |",
        "| --- | ---: | ---: |",
    ]
    values = [
        ("Device", reference["environment"]["device"], candidate["environment"]["device"]),
        ("Dtype", reference["environment"]["dtype"], candidate["environment"]["dtype"]),
        ("Parameters", reference["parameters"], candidate["parameters"]),
        ("Checkpoint bytes", reference["checkpoint_bytes"], candidate["checkpoint_bytes"]),
        ("Loaded parameter bytes", reference["parameter_bytes"], candidate["parameter_bytes"]),
        (
            "Sampled peak process RSS bytes",
            reference["memory"]["sampled_peak_rss_bytes"],
            candidate["memory"]["sampled_peak_rss_bytes"],
        ),
        (
            "Median fixed-step latency (s)",
            reference["timing"]["median_total_seconds"],
            candidate["timing"]["median_total_seconds"],
        ),
        (
            "Generated tokens/s",
            reference["timing"]["median_tokens_per_second"],
            candidate["timing"]["median_tokens_per_second"],
        ),
    ]
    if reference["evaluation"]:
        values += [
            ("Held-out loss", reference["evaluation"]["loss"], candidate["evaluation"]["loss"]),
            (
                "Perplexity",
                reference["evaluation"]["perplexity"],
                candidate["evaluation"]["perplexity"],
            ),
        ]
    for key, label in (
        ("cuda_peak_allocated_bytes", "CUDA peak tensor bytes"),
        ("cuda_peak_reserved_bytes", "CUDA peak reserved bytes"),
        ("sampled_peak_mps_allocated_bytes", "Sampled MPS tensor bytes"),
        ("sampled_peak_mps_driver_bytes", "Sampled MPS driver bytes"),
    ):
        if any(value["memory"][key] is not None for value in (reference, candidate)):
            values.append((label, reference["memory"][key], candidate["memory"][key]))
    for label, a, b in values:
        rows.append(f"| {label} | {a} | {b} |")
    comparison = report["comparison"]
    rows += [
        "",
        f"Numeric equivalence: **{comparison['numeric_equivalence']}**. "
        f"Generated token agreement: **{comparison['generated_token_agreement']:.2%}**.",
        "",
        comparison["quality"]["note"],
        "",
        "## Per-prompt comparisons",
        "",
        "| Case | Input tokens | Full-logit max delta | Cached-logit max delta | Token agreement |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for case in comparison["cases"]:
        rows.append(
            f"| {case['case']} | {case['input_tokens']} | {case['full_max_abs_delta']:.6g} | "
            f"{case['cached_max_abs_delta']:.6g} | {case['generated_token_agreement']:.2%} |"
        )
    rows += ["", "## Measurement notes", ""] + [f"- {note}" for note in report["notes"]]
    return "\n".join(rows) + "\n"


def run_benchmark(workspace, request, *, fingerprints, job_id, check, update):
    run = workspace.path("runs", job_id)
    private = run / ".benchmark"
    private.mkdir(mode=0o700)
    try:
        check()
        verify_inputs(workspace, request, fingerprints)
        update(phase="preparing_benchmark", training_performed=False)
        inputs = prepare_inputs(workspace, request)
        json_write(private / "inputs.json", inputs)
        reference = wait_runner(workspace, job_id, "reference", private, check, update)
        candidate = wait_runner(workspace, job_id, "candidate", private, check, update)
        check()
        verify_inputs(workspace, request, fingerprints)
        if reference["environment"] != candidate["environment"]:
            raise ValueError("Benchmark environments differ; results cannot be compared")
        comparison = compare_results(request, inputs, reference, candidate, private)
        for value in (reference, candidate):
            for case in value["cases"]:
                case.pop("generated_ids")
        settings = request.model_dump(exclude={"prompts"})
        report = {
            "schema_version": 1,
            "version": __version__,
            "job_id": job_id,
            "created_at": datetime.now(UTC).isoformat(),
            "training_performed": False,
            "settings": settings,
            "fingerprints": fingerprints,
            "data": {
                key: value for key, value in inputs.items() if key not in {"prompts", "evaluation"}
            },
            "reference": reference,
            "candidate": candidate,
            "comparison": comparison,
            "changes": {
                "latency_percent": percent_change(
                    reference["timing"]["median_total_seconds"],
                    candidate["timing"]["median_total_seconds"],
                ),
                "checkpoint_bytes": candidate["checkpoint_bytes"] - reference["checkpoint_bytes"],
                "parameters": candidate["parameters"] - reference["parameters"],
            },
            "notes": [
                "Finite prompt checks do not establish equivalence on every input or task quality.",
                "Each model uses a fresh process; memory includes runtime overhead.",
                "RSS/MPS peaks use 10 ms sampling and can miss short spikes.",
                "CUDA peaks count this process's PyTorch allocations, not total device use.",
                "Timing excludes loading/tokenization and uses fixed greedy steps even after EOS.",
                "Timing includes Python decode/mask/cancellation overhead; warmups are excluded.",
                "Parent coordinator/tokenizer RAM is excluded from isolated model-runner RSS.",
                "Reference runs first; timing is host-specific, not a controlled speedup claim.",
            ],
        }
        check()
        temporary_markdown = private / "report.md"
        temporary_markdown.write_text(markdown_report(report), encoding="utf-8", newline="\n")
        with workspace.lock:
            check()
            try:
                replace_file(temporary_markdown, run / "benchmark.md")
                json_write(run / "benchmark.json", report)
            except Exception:
                (run / "benchmark.md").unlink(missing_ok=True)
                raise
        return {
            "passed": comparison["passed"],
            "numeric_equivalence": comparison["numeric_equivalence"],
            "generated_token_agreement": comparison["generated_token_agreement"],
            "quality": comparison["quality"],
            "training_performed": False,
            "report_json": str(run / "benchmark.json"),
            "report_markdown": str(run / "benchmark.md"),
        }
    finally:
        shutil.rmtree(private, ignore_errors=True)


def read_report(workspace, job_id):
    run = workspace.path("runs", job_id)
    if read_json(run / "request.json")["kind"] != "benchmark":
        raise ValueError("This job is not a benchmark")
    path = run / "benchmark.json"
    if read_json(run / "status.json")["status"] != "completed" or not path.is_file():
        raise ValueError("Benchmark report is available after the job completes")
    return read_json(path)
