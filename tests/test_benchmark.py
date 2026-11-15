import asyncio
import json
import os
import shutil
import subprocess
import sys
import time

import psutil
import pytest
import torch
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from safetensors.torch import load, save_file

from local_model_mcp import benchmark, benchmark_runner, qwen
from local_model_mcp.jobs import Jobs
from local_model_mcp.schemas import (
    BenchmarkRequest,
    Example,
    QwenArchitectureJobRequest,
    QwenArchitectureRequest,
)
from local_model_mcp.weights import Checkpoint
from local_model_mcp.workspace import json_write, read_json


def small_request(**values):
    settings = dict(
        model="qwen2",
        candidate="candidate",
        prompts=["hello world", "the sky is"],
        max_new_tokens=2,
        warmup_runs=0,
        measured_runs=1,
        num_threads=1,
    )
    settings.update(values)
    return BenchmarkRequest(**settings)


@pytest.fixture
def pair(qwen_factory):
    workspace = qwen_factory()
    spec = QwenArchitectureRequest(model="qwen2", operation="widen_mlp", new_intermediate_size=37)
    request = QwenArchitectureJobRequest(
        **spec.model_dump(),
        output="candidate",
        expected_plan_token=qwen.preview(workspace, spec)["plan_token"],
    )
    qwen.run_change(
        workspace,
        request,
        check=lambda: None,
        update=lambda **values: None,
        publish=lambda stage, name: workspace.publish(stage, name),
        report_path=workspace.root / "runs" / "architecture-test.json",
        job_id="fixture",
    )
    workspace.write_dataset(
        "held-out",
        [Example(text="the sky is blue . [EOS]"), Example(text="the grass is green . [EOS]")],
    )
    return workspace


def fingerprints(workspace, request):
    from local_model_mcp.workspace import sha256

    result = {
        key: Checkpoint(workspace.model(getattr(request, key))).revision()
        for key in ("model", "candidate")
    }
    if request.dataset:
        result["dataset"] = sha256(workspace.path("datasets", request.dataset, ".jsonl"))
    return result


def wait_job(jobs, job_id, timeout=180):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        status = jobs.status(job_id, log_lines=10)
        if status["status"] in {"completed", "failed", "cancelled", "interrupted"}:
            return status
        time.sleep(0.1)
    jobs.cancel(job_id)
    pytest.fail(f"Benchmark job timed out: {jobs.status(job_id)}")


def test_isolated_benchmark_quality_reports_and_original_preservation(pair):
    request = small_request(dataset="held-out", measured_runs=2, warmup_runs=1)
    before = fingerprints(pair, request)
    jobs = Jobs(pair)
    job = jobs.start("benchmark", request)["job_id"]
    status = wait_job(jobs, job)
    assert status["status"] == "completed", status
    assert status["result"]["passed"]
    report = benchmark.read_report(pair, job)
    assert report["comparison"]["numeric_equivalence"]
    assert report["comparison"]["quality"]["measured"]
    assert report["comparison"]["quality"]["passed"]
    reference, candidate = report["reference"], report["candidate"]
    assert (reference["runner_pid"], reference["runner_started_at"]) != (
        candidate["runner_pid"],
        candidate["runner_started_at"],
    )
    assert candidate["parameters"] > reference["parameters"]
    assert reference["evaluation"]["tokens"] == candidate["evaluation"]["tokens"] == 10
    assert candidate["evaluation"]["loss"] == pytest.approx(
        reference["evaluation"]["loss"], abs=1e-5
    )
    for value in (reference, candidate):
        assert value["environment"]["device"] == "cpu"
        assert value["memory"]["sampled_peak_rss_bytes"] > 0
        assert value["memory"]["cuda_peak_allocated_bytes"] is None
        assert len(value["timing"]["samples"]) == 4
        assert value["timing"]["median_total_seconds"] > 0
        assert value["timing"]["median_tokens_per_second"] > 0
    run = pair.root / "runs" / job
    assert not (run / ".benchmark").exists()
    assert "**PASS**" in (run / "benchmark.md").read_text(encoding="utf-8")
    assert "generated_ids" not in json.dumps(report)
    assert all(
        "generated_text" not in case for value in (reference, candidate) for case in value["cases"]
    )
    assert "hello world" not in json.dumps(report)
    assert fingerprints(pair, request) == before


def test_changed_model_completes_with_failed_comparison_report(pair):
    path = pair.model("candidate") / "model.safetensors"
    tensors = load(path.read_bytes())
    tensors["lm_head.weight"].zero_()
    save_file(tensors, path, metadata={"format": "pt"})
    jobs = Jobs(pair)
    job = jobs.start("benchmark", small_request())["job_id"]
    status = wait_job(jobs, job)
    assert status["status"] == "completed", status
    assert status["result"]["passed"] is False
    report = benchmark.read_report(pair, job)
    assert report["comparison"]["numeric_equivalence"] is False
    assert report["comparison"]["quality"]["measured"] is False
    assert report["comparison"]["quality"]["passed"] is None
    assert "**FAIL**" in (pair.root / "runs" / job / "benchmark.md").read_text(encoding="utf-8")


def test_tokenizer_mismatch_is_rejected(pair):
    path = pair.model("candidate") / "tokenizer.json"
    tokenizer = read_json(path)
    vocabulary = tokenizer["model"]["vocab"]
    vocabulary["sky"], vocabulary["grass"] = vocabulary["grass"], vocabulary["sky"]
    json_write(path, tokenizer)
    with pytest.raises(ValueError, match="tokenizers differ"):
        benchmark.prepare_inputs(pair, small_request())


def test_input_fingerprints_and_prompt_bounds(pair):
    request = small_request()
    original = fingerprints(pair, request)
    config = pair.model("candidate") / "config.json"
    json_write(config, {**read_json(config), "changed_for_test": True})
    with pytest.raises(ValueError, match="candidate changed"):
        benchmark.verify_inputs(pair, request, original)
    with pytest.raises(ValueError, match="at least two tokens"):
        benchmark.prepare_inputs(pair, small_request(prompts=["hello"]))
    with pytest.raises(ValueError, match="insufficient model context"):
        benchmark.prepare_inputs(pair, small_request(max_new_tokens=64))


def test_inline_runner_has_no_training_and_keeps_optional_text_local(pair, monkeypatch):
    from transformers import AutoModelForCausalLM

    def forbidden(*args, **kwargs):
        raise AssertionError("Benchmarking must not train")

    load_model = AutoModelForCausalLM.from_pretrained

    def local_only(*args, **kwargs):
        assert kwargs["local_files_only"] is True
        assert kwargs["trust_remote_code"] is False
        assert kwargs["use_safetensors"] is True
        return load_model(*args, **kwargs)

    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", local_only)
    monkeypatch.setattr(torch.Tensor, "backward", forbidden)
    monkeypatch.setattr(torch.optim.AdamW, "step", forbidden)
    request = small_request(include_generated_text=True)
    job = "inline"
    run = pair.path("runs", job, exists=False)
    run.mkdir()
    private = run / ".benchmark"
    private.mkdir()
    result = benchmark_runner.measure_checkpoint(
        pair,
        request,
        "reference",
        benchmark.prepare_inputs(pair, request),
        private,
        job,
        fingerprints(pair, request),
    )
    assert result["cache_consistent"]
    assert result["repeats_consistent"]
    assert isinstance(result["cases"][0]["generated_text"], str)


@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_unavailable_gpu_is_rejected_without_loading_models(device, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(ValueError, match="unavailable"):
        benchmark_runner.resolve_device(small_request(device=device))


def test_device_precision_and_auto_selection(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert benchmark_runner.resolve_device(small_request(device="auto")).type == "mps"
    with pytest.raises(ValueError, match="bfloat16"):
        benchmark_runner.resolve_device(small_request(device="mps", dtype="bfloat16"))
    with pytest.raises(ValueError, match="CPU"):
        benchmark_runner.resolve_device(small_request(device="cpu", dtype="float16"))


def test_gpu_synchronization_hooks_are_scoped(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: calls.append(str(device)))
    monkeypatch.setattr(torch.mps, "synchronize", lambda: calls.append("mps"))
    benchmark_runner.synchronize(torch.device("cpu"))
    assert calls == []
    benchmark_runner.synchronize(torch.device("cuda"))
    benchmark_runner.synchronize(torch.device("mps"))
    assert calls == ["cuda", "mps"]


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda"])
def test_memory_measurement_backend_seams(device, monkeypatch):
    monkeypatch.setattr(torch.mps, "current_allocated_memory", lambda: 42)
    monkeypatch.setattr(torch.mps, "driver_allocated_memory", lambda: 64)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 128)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device: 256)
    sampler = benchmark_runner.MemorySampler(torch.device(device))
    sampler.start()
    result = sampler.finish()
    assert result["sampled_peak_rss_bytes"] > 0
    assert result["sampled_peak_mps_allocated_bytes"] == (42 if device == "mps" else None)
    assert result["cuda_peak_allocated_bytes"] == (128 if device == "cuda" else None)
    assert not sampler.thread.is_alive()


@pytest.mark.parametrize(
    "field,value",
    [("warmup_runs", -1), ("measured_runs", 0), ("atol", float("nan")), ("min_token_agreement", 2)],
)
def test_benchmark_request_limits(field, value):
    with pytest.raises(ValueError):
        small_request(**{field: value})


@pytest.mark.parametrize("failure", ["loss", "tokens", "logits", "cache_consistency"])
def test_report_verdict_covers_each_configured_check(tmp_path, failure):
    request = small_request(prompts=["hello world"], dataset="held-out")
    source = {
        "cache_consistent": True,
        "repeats_consistent": True,
        "evaluation": {"loss": 2.0},
        "cases": [{"generated_ids": [1, 2]}],
    }
    changed = json.loads(json.dumps(source))
    first = {"0_full": torch.ones(1, 2, 3), "0_cached": torch.ones(1, 1, 3)}
    second = {key: value.clone() for key, value in first.items()}
    if failure == "loss":
        changed["evaluation"]["loss"] = 2.2
    elif failure == "tokens":
        changed["cases"][0]["generated_ids"] = [9, 2]
    elif failure == "cache_consistency":
        changed["cache_consistent"] = False
    else:
        second["0_full"].add_(0.5)
    save_file(first, tmp_path / "reference.safetensors")
    save_file(second, tmp_path / "candidate.safetensors")
    report = benchmark.compare_results(request, {"prompts": [[1, 2]]}, source, changed, tmp_path)
    assert report["passed"] is False


def test_fixed_decode_does_not_short_circuit_at_eos():
    from types import SimpleNamespace

    lengths = []

    def model(**values):
        ids = values["input_ids"]
        lengths.append(ids.shape[1])
        logits = torch.zeros(1, ids.shape[1], 3)
        logits[:, :, 2] = 1
        return SimpleNamespace(logits=logits, past_key_values="cache")

    tokens, metrics = benchmark_runner.fixed_decode(
        model, torch.tensor([[0, 1]]), 3, torch.device("cpu"), lambda: None
    )
    assert tokens == [2, 2, 2]
    assert lengths == [2, 1, 1]
    assert metrics["total_seconds"] > 0


def test_cancelled_benchmark_stops_child_and_removes_temporary_inputs(pair):
    request = small_request(prompts=["hello world"] * 8, measured_runs=20, max_new_tokens=16)
    jobs = Jobs(pair)
    job = jobs.start("benchmark", request)["job_id"]
    run = pair.root / "runs" / job
    end = time.monotonic() + 120
    children = []
    while time.monotonic() < end:
        if (run / ".benchmark" / "progress.json").is_file():
            process = psutil.Process(read_json(run / "process.json")["pid"])
            children = process.children(recursive=True)
            break
        status = jobs.status(job)
        assert status["status"] not in {"failed", "completed", "interrupted"}, status
        time.sleep(0.1)
    assert children, jobs.status(job)
    jobs.cancel(job)
    assert wait_job(jobs, job)["status"] == "cancelled"
    assert not (run / ".benchmark").exists()
    assert not (run / "benchmark.json").exists()
    # Wait for termination, rather than treating PID visibility as active execution.
    # Windows can retain an exited process object while handles are still open.
    _, alive = psutil.wait_procs(children, timeout=10)
    assert not alive, [(child.pid, child.name()) for child in alive]


def test_cli_benchmark_on_local_random_checkpoint(qwen_factory):
    workspace = qwen_factory()
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "local_model_mcp.benchmark_cli",
            "--root",
            str(workspace.root),
            "--model",
            "qwen2",
            "--candidate",
            "qwen2",
            "--prompt",
            "hello world",
            "--runs",
            "1",
            "--warmup-runs",
            "0",
            "--max-new-tokens",
            "2",
            "--num-threads",
            "1",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=180,
    )
    assert process.returncode == 0, process.stderr
    summary = json.loads(process.stdout)
    assert summary["passed"]
    assert os.path.isfile(summary["report_json"])
    assert os.path.isfile(summary["report_markdown"])


def test_benchmark_tools_over_stdio(qwen_factory):
    workspace = qwen_factory("qwen3")
    shutil.copytree(workspace.model("qwen3"), workspace.root / "models" / "candidate")

    async def scenario():
        parameters = StdioServerParameters(
            command=sys.executable,
            args=["-m", "local_model_mcp.server", "--root", str(workspace.root)],
        )
        async with stdio_client(parameters) as (read, write):
            async with ClientSession(read, write) as client:
                await client.initialize()
                launch = await client.call_tool(
                    "start_qwen_benchmark",
                    {
                        "request": {
                            "model": "qwen3",
                            "candidate": "candidate",
                            "prompts": ["hello world"],
                            "warmup_runs": 0,
                            "measured_runs": 1,
                            "max_new_tokens": 2,
                            "num_threads": 1,
                        }
                    },
                )
                assert not launch.isError
                job = launch.structuredContent["job_id"]
                for _ in range(1800):
                    response = await client.call_tool("job_status", {"job_id": job})
                    status = response.structuredContent
                    if status["status"] in {"completed", "failed", "interrupted"}:
                        assert status["status"] == "completed", status
                        report = await client.call_tool("get_benchmark_report", {"job_id": job})
                        assert not report.isError
                        assert report.structuredContent["comparison"]["passed"]
                        assert not report.structuredContent["comparison"]["quality"]["measured"]
                        return
                    await asyncio.sleep(0.1)
                await client.call_tool("cancel_job", {"job_id": job})
                raise AssertionError("MCP benchmark timed out")

    asyncio.run(scenario())
