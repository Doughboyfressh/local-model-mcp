import time

import pytest
import torch
from transformers import AutoModelForCausalLM

from local_model_mcp.jobs import Jobs
from local_model_mcp.schemas import (
    EditRequest,
    EvaluateRequest,
    GenerateRequest,
    MergeRequest,
    TrainRequest,
    WeightEdit,
)
from local_model_mcp.weights import Checkpoint, apply, preview
from local_model_mcp.worker import Worker


def wait(jobs, job_id, timeout=120):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = jobs.status(job_id)
        if status["status"] in {"completed", "cancelled", "failed", "interrupted"}:
            return status
        time.sleep(0.1)
    jobs.cancel(job_id)
    pytest.fail(f"Job timed out: {jobs.status(job_id)}")


def run(jobs, kind, request):
    status = wait(jobs, jobs.start(kind, request)["job_id"])
    assert status["status"] == "completed", status
    return status["result"]


def test_real_full_training_evaluation_generation_and_reload(workspace):
    jobs = Jobs(workspace)
    base_revision = Checkpoint(workspace.model("tiny")).revision()
    evaluation = EvaluateRequest(
        model="tiny",
        dataset="held-out",
        max_length=32,
        batch_size=2,
        device="cpu",
    )
    before = run(jobs, "evaluate", evaluation)
    trained = run(
        jobs,
        "train",
        TrainRequest(
            model="tiny",
            dataset="train",
            output="full",
            method="full",
            device="cpu",
            max_steps=15,
            learning_rate=0.02,
            batch_size=2,
            max_length=32,
        ),
    )
    assert trained["last_loss"] < trained["first_loss"]
    after = run(jobs, "evaluate", evaluation.model_copy(update={"model": "full"}))
    assert after["loss"] < before["loss"]
    assert after["tokens"] == 10
    assert Checkpoint(workspace.model("tiny")).revision() == base_revision
    model = AutoModelForCausalLM.from_pretrained(workspace.model("full"), local_files_only=True)
    assert model.config.vocab_size == 16
    generated = run(
        jobs,
        "generate",
        GenerateRequest(
            model="full",
            prompt="the sky is",
            max_new_tokens=3,
            device="cpu",
        ),
    )
    assert 1 <= generated["generated_tokens"] <= 3
    assert isinstance(generated["text"], str)


def test_real_lora_training_and_merge(workspace):
    jobs = Jobs(workspace)
    result = run(
        jobs,
        "train",
        TrainRequest(
            model="tiny",
            dataset="train",
            output="adapter",
            method="lora",
            device="cpu",
            max_steps=8,
            learning_rate=0.02,
            batch_size=2,
            max_length=32,
            lora_rank=2,
            lora_alpha=4,
            lora_dropout=0,
        ),
    )
    assert result["artifact"] == "adapter"
    adapter = Checkpoint(workspace.model("adapter"))
    assert any(
        bool(adapter.tensor(key).abs().sum() > 0) for key in adapter.catalog if "lora_B" in key
    )
    evaluation = EvaluateRequest(
        model="tiny",
        adapter="adapter",
        dataset="held-out",
        batch_size=2,
        device="cpu",
    )
    adapted = run(jobs, "evaluate", evaluation)
    run(
        jobs,
        "merge",
        MergeRequest(
            model="tiny",
            adapter="adapter",
            output="merged",
            device="cpu",
        ),
    )
    merged = run(
        jobs,
        "evaluate",
        EvaluateRequest(
            model="merged",
            dataset="held-out",
            batch_size=2,
            device="cpu",
        ),
    )
    assert merged["loss"] == pytest.approx(adapted["loss"], abs=1e-5)
    assert Checkpoint(workspace.model("merged")).kind == "model"
    # Editing an adapter must preserve its original base-model binding.
    key = next(key for key in adapter.catalog if "lora_B" in key)
    edit = EditRequest(
        model="adapter", edits=[WeightEdit(tensor=key, operation="scale", value=0.5)]
    )
    plan = preview(workspace, edit)
    apply(workspace, edit, "edited-adapter", plan["plan_token"])
    edited = run(jobs, "evaluate", evaluation.model_copy(update={"adapter": "edited-adapter"}))
    assert edited["loss"] > 0


def test_cancellation_concurrency_and_restart_status(workspace):
    jobs = Jobs(workspace)
    job = jobs.start(
        "train",
        TrainRequest(
            model="tiny",
            dataset="train",
            output="cancelled",
            device="cpu",
            max_steps=100_000,
        ),
    )
    with pytest.raises(ValueError, match="already active"):
        jobs.start("generate", GenerateRequest(model="tiny", prompt="hello", device="cpu"))
    # A new manager instance can inspect and cancel a worker started by the previous instance.
    restarted = Jobs(workspace)
    assert restarted.status(job["job_id"])["status"] in {"queued", "running"}
    restarted.cancel(job["job_id"])
    assert wait(jobs, job["job_id"])["status"] == "cancelled"
    assert not (workspace.root / "models" / "cancelled").exists()


def test_padding_masks_preserve_real_eos_tokens():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(WordLevel({"[UNK]": 0, "[EOS]": 1, "x": 2}, unk_token="[UNK]"))
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, eos_token="[EOS]", pad_token="[EOS]"
    )
    batch = Worker.batch([[2, 1], [2, 2, 1]], tokenizer, torch.device("cpu"))
    assert batch["labels"].tolist() == [[2, 1, -100], [2, 2, 1]]
