"""One isolated model measurement per process. No training or remote model access."""

import argparse
import math
import os
import platform
import statistics
import threading
import time
import traceback

import psutil
import torch
from safetensors.torch import save_file

from .benchmark import tolerances
from .schemas import BenchmarkRequest
from .weights import Checkpoint
from .workspace import Workspace, json_write, read_json


def resolve_device(request):
    device = request.device
    if device == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("Apple MPS is unavailable")
    if device == "cpu" and request.dtype == "float16":
        raise ValueError("Use float32 or bfloat16 for CPU benchmarks")
    if request.dtype == "bfloat16" and (
        device == "mps" or (device == "cuda" and not torch.cuda.is_bf16_supported())
    ):
        raise ValueError("bfloat16 is unsupported for this benchmark device")
    return torch.device(device)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


class MemorySampler:
    def __init__(self, device):
        self.device = device
        self.process = psutil.Process()
        self.stopped = threading.Event()
        self.lock = threading.Lock()
        self.peaks = {
            "sampled_peak_rss_bytes": None,
            "sampled_peak_mps_allocated_bytes": None,
            "sampled_peak_mps_driver_bytes": None,
        }
        self.errors = set()
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def sample(self):
        values = {}
        try:
            values["sampled_peak_rss_bytes"] = self.process.memory_info().rss
            if self.device.type == "mps":
                values["sampled_peak_mps_allocated_bytes"] = torch.mps.current_allocated_memory()
                values["sampled_peak_mps_driver_bytes"] = torch.mps.driver_allocated_memory()
        except (psutil.Error, RuntimeError, AttributeError) as exc:
            with self.lock:
                self.errors.add(type(exc).__name__)
        with self.lock:
            for key, value in values.items():
                self.peaks[key] = max(self.peaks[key] or 0, value)

    def loop(self):
        while not self.stopped.wait(0.01):
            self.sample()

    def start(self):
        self.sample()
        self.thread.start()

    def finish(self):
        self.stopped.set()
        self.thread.join(timeout=2)
        self.sample()
        result = {
            **self.peaks,
            "sampling_interval_seconds": 0.01,
            "sampling_error_types": sorted(self.errors),
            "cuda_peak_allocated_bytes": None,
            "cuda_peak_reserved_bytes": None,
        }
        if self.device.type == "cuda":
            result["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(self.device)
            result["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(self.device)
        return result


def environment(request, device):
    import transformers

    hardware = platform.processor() or platform.machine()
    if device.type == "cuda":
        hardware = torch.cuda.get_device_name(device)
    elif device.type == "mps":
        hardware = f"Apple MPS ({platform.machine()})"
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": str(device),
        "hardware": hardware,
        "dtype": request.dtype,
        "attention_implementation": request.attention_implementation,
        "num_threads": torch.get_num_threads(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def fixed_decode(model, ids, count, device, check):
    """Exactly count greedy steps, including after EOS; matched work for both checkpoints."""
    check()
    synchronize(device)
    started = time.perf_counter()
    output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=True)
    token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
    tokens, cache = [token], output.past_key_values
    del output
    synchronize(device)
    prefill_done = time.perf_counter()
    for step in range(1, count):
        check()
        output = model(
            input_ids=token,
            attention_mask=torch.ones((1, ids.shape[1] + step), dtype=torch.long, device=device),
            past_key_values=cache,
            use_cache=True,
        )
        token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
        tokens.append(token)
        cache = output.past_key_values
        del output
    synchronize(device)
    finished = time.perf_counter()
    return torch.cat(tokens, dim=1)[0].cpu().tolist(), {
        "prefill_first_token_seconds": prefill_done - started,
        "decode_seconds": finished - prefill_done,
        "total_seconds": finished - started,
    }


def timing_summary(samples, count):
    median = statistics.median(value["total_seconds"] for value in samples)
    return {
        "samples": samples,
        "median_total_seconds": median,
        "min_total_seconds": min(value["total_seconds"] for value in samples),
        "max_total_seconds": max(value["total_seconds"] for value in samples),
        "median_prefill_first_token_seconds": statistics.median(
            value["prefill_first_token_seconds"] for value in samples
        ),
        "median_decode_seconds": statistics.median(value["decode_seconds"] for value in samples),
        "median_tokens_per_second": statistics.median(
            count / value["total_seconds"] for value in samples
        ),
        "generated_steps_per_sample": count,
    }


def measure_checkpoint(workspace, request, role, inputs, private, job_id, fingerprints):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    run_directory = workspace.path("runs", job_id)

    def check():
        if (run_directory / "cancel").exists():
            raise InterruptedError("Benchmark cancellation requested")

    def progress(phase, **values):
        json_write(private / "progress.json", {"role": role, "phase": phase, **values})

    name = request.model if role == "reference" else request.candidate
    path = workspace.model(name)

    def verify_checkpoint():
        key = "model" if role == "reference" else "candidate"
        if Checkpoint(path).revision() != fingerprints[key]:
            raise ValueError(f"Benchmark {role} checkpoint changed after job creation")

    check()
    verify_checkpoint()
    device = resolve_device(request)
    torch.set_num_threads(request.num_threads)
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    sampler = MemorySampler(device)
    sampler.start()
    try:
        progress("loading")
        checkpoint = Checkpoint(path)
        model, information = AutoModelForCausalLM.from_pretrained(
            path,
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
            torch_dtype=getattr(torch, request.dtype),
            attn_implementation=request.attention_implementation,
            output_loading_info=True,
        )
        if any(
            information.get(key)
            for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
        ):
            raise ValueError(f"Checkpoint did not reload strictly: {information}")
        model.to(device).eval().requires_grad_(False)
        sampler.sample()
        records, cases, all_samples = {}, [], []
        cache_consistent, repeats_consistent = True, True
        atol, rtol = tolerances(request)
        tokenizer = (
            AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
            if request.include_generated_text
            else None
        )
        with torch.inference_mode():
            for index, values in enumerate(inputs["prompts"]):
                check()
                progress("checking_outputs", case=index + 1)
                ids = torch.tensor([values], dtype=torch.long, device=device)
                full = model(
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
                if not (torch.isfinite(full).all() and torch.isfinite(cached).all()):
                    raise ValueError("Model produced nonfinite benchmark logits")
                cache_consistent &= bool(torch.allclose(full[:, -1:], cached, atol=atol, rtol=rtol))
                records[f"{index}_full"] = full.float().cpu().clone().contiguous()
                records[f"{index}_cached"] = cached.float().cpu().clone().contiguous()
                del full, cached, prefix
                baseline, _ = fixed_decode(model, ids, request.max_new_tokens, device, check)
                for repeat in range(request.warmup_runs):
                    progress("warmup", case=index + 1, repeat=repeat + 1)
                    fixed_decode(model, ids, request.max_new_tokens, device, check)
                samples = []
                for repeat in range(request.measured_runs):
                    progress("measuring", case=index + 1, repeat=repeat + 1)
                    generated, metrics = fixed_decode(
                        model, ids, request.max_new_tokens, device, check
                    )
                    repeats_consistent &= generated == baseline
                    samples.append(metrics)
                    sampler.sample()
                case = {
                    "case": index + 1,
                    "input_tokens": len(values),
                    "generated_ids": baseline,
                    "timing": timing_summary(samples, request.max_new_tokens),
                }
                if tokenizer is not None:
                    case["generated_text"] = tokenizer.decode(baseline, skip_special_tokens=True)
                cases.append(case)
                all_samples.extend(samples)
            evaluation = None
            if inputs["evaluation"]:
                loss_sum, tokens = 0, 0
                for index, values in enumerate(inputs["evaluation"]):
                    check()
                    progress("evaluating", example=index + 1)
                    ids = torch.tensor([values], dtype=torch.long, device=device)
                    loss = (
                        model(
                            input_ids=ids,
                            attention_mask=torch.ones_like(ids),
                            labels=ids,
                            use_cache=False,
                        )
                        .loss.float()
                        .item()
                    )
                    if not math.isfinite(loss):
                        raise ValueError("Model produced nonfinite evaluation loss")
                    loss_sum += loss * (len(values) - 1)
                    tokens += len(values) - 1
                mean = loss_sum / tokens
                evaluation = {
                    "loss": mean,
                    "perplexity": math.exp(mean) if mean < 700 else None,
                    "tokens": tokens,
                    "examples": len(inputs["evaluation"]),
                }
        synchronize(device)
        memory = sampler.finish()
        sampler = None
        check()
        verify_checkpoint()
        save_file(records, private / f"{role}.safetensors")
        result = {
            "status": "completed",
            "model": name,
            "runner_pid": os.getpid(),
            "runner_started_at": psutil.Process().create_time(),
            "environment": environment(request, device),
            "parameters": sum(value.numel() for value in model.parameters()),
            "parameter_bytes": sum(
                value.numel() * value.element_size() for value in model.parameters()
            ),
            "checkpoint_bytes": sum(
                value.stat().st_size for value in path.rglob("*") if value.is_file()
            ),
            "weight_file_bytes": sum(
                (path / value).stat().st_size for value in set(checkpoint.mapping.values())
            ),
            "memory": memory,
            "timing": timing_summary(all_samples, request.max_new_tokens),
            "cache_consistent": bool(cache_consistent),
            "repeats_consistent": bool(repeats_consistent),
            "evaluation": evaluation,
            "cases": cases,
        }
        json_write(private / f"{role}.json", result)
        return result
    finally:
        if sampler is not None:
            sampler.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--role", choices=["reference", "candidate"], required=True)
    args = parser.parse_args()
    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    workspace = Workspace(args.root)
    run = workspace.path("runs", args.job)
    private = run / ".benchmark"
    payload = read_json(run / "request.json")
    request = BenchmarkRequest.model_validate(payload["request"])
    try:
        measure_checkpoint(
            workspace,
            request,
            args.role,
            read_json(private / "inputs.json"),
            private,
            args.job,
            payload["fingerprints"],
        )
    except Exception as exc:
        traceback.print_exc()
        json_write(
            private / f"{args.role}.json",
            {"status": "failed", "error": f"{type(exc).__name__}: {exc}"},
        )
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
