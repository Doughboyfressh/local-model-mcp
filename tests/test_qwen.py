import asyncio
import sys
import time

import pytest
import torch
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from safetensors.torch import load, save_file
from transformers import AutoModelForCausalLM

from local_model_mcp import qwen
from local_model_mcp.jobs import Jobs
from local_model_mcp.schemas import QwenArchitectureJobRequest, QwenArchitectureRequest
from local_model_mcp.weights import Checkpoint
from local_model_mcp.worker import Cancelled
from local_model_mcp.workspace import json_write, read_json


def request_for(family, operation):
    arguments = (
        {"insert_after": [0, 1]}
        if operation == "insert_identity_blocks"
        else {"new_intermediate_size": 37}
    )
    return QwenArchitectureRequest(model=family, operation=operation, **arguments)


def job_request(workspace, spec, output="edited", **kwargs):
    preview = qwen.preview(workspace, spec)
    return QwenArchitectureJobRequest(
        **spec.model_dump(), output=output, expected_plan_token=preview["plan_token"], **kwargs
    )


def apply_direct(workspace, request, *, check=lambda: None, update=lambda **kwargs: None):
    return qwen.run_change(
        workspace,
        request,
        check=check,
        update=update,
        publish=lambda stage, name: workspace.publish(stage, name),
        report_path=workspace.root / "runs" / "test-verification.json",
        job_id="test",
    )


@pytest.mark.parametrize("family", ["qwen2", "qwen3"])
@pytest.mark.parametrize("operation", ["insert_identity_blocks", "widen_mlp"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_qwen_transform_preserves_logits_weights_and_generation(
    qwen_factory, family, operation, dtype
):
    workspace = qwen_factory(family, dtype=dtype, shard_size="2KB")
    original = Checkpoint(workspace.model(family))
    revision = original.revision()
    request = job_request(
        workspace, request_for(family, operation), validation_prompts=["the sky is blue"]
    )
    result = apply_direct(workspace, request)
    assert not result["training_performed"]
    assert result["verification"]["passed"]
    assert len(result["verification"]["probes"]) == 3
    assert original.revision() == revision
    changed = Checkpoint(workspace.model("edited"))
    assert (
        changed.catalog["model.embed_tokens.weight"]["dtype"]
        == original.catalog["model.embed_tokens.weight"]["dtype"]
    )
    config = read_json(changed.path / "config.json")
    if operation == "insert_identity_blocks":
        assert config["num_hidden_layers"] == 4
        assert config["layer_types"] == ["full_attention"] * 4
        for index in (1, 3):
            assert not changed.tensor(
                f"model.layers.{index}.self_attn.o_proj.weight"
            ).count_nonzero()
            assert not changed.tensor(f"model.layers.{index}.mlp.down_proj.weight").count_nonzero()
        torch.testing.assert_close(
            changed.tensor("model.layers.2.mlp.up_proj.weight"),
            original.tensor("model.layers.1.mlp.up_proj.weight"),
            rtol=0,
            atol=0,
        )
    else:
        assert config["intermediate_size"] == 37
        for index in range(2):
            for projection in ("gate_proj", "up_proj"):
                key = f"model.layers.{index}.mlp.{projection}.weight"
                torch.testing.assert_close(
                    changed.tensor(key)[:24], original.tensor(key), rtol=0, atol=0
                )
            key = f"model.layers.{index}.mlp.down_proj.weight"
            torch.testing.assert_close(
                changed.tensor(key)[:, :24], original.tensor(key), rtol=0, atol=0
            )
            assert not changed.tensor(key)[:, 24:].count_nonzero()
    models = [
        AutoModelForCausalLM.from_pretrained(
            path,
            local_files_only=True,
            torch_dtype=torch.float32,
            attn_implementation="eager",
        ).eval()
        for path in (original.path, changed.path)
    ]
    ids = torch.tensor([[3, 4, 5]])
    with torch.inference_mode():
        outputs = [
            model.generate(ids, max_new_tokens=3, do_sample=False, use_cache=True)
            for model in models
        ]
    assert torch.equal(outputs[0], outputs[1])


def test_sharded_output_tied_embeddings_and_attention_bias(qwen_factory, monkeypatch):
    workspace = qwen_factory("qwen3", tie_word_embeddings=True, attention_bias=True)
    monkeypatch.setattr(qwen, "SHARD_BYTES", 1024)
    request = job_request(workspace, request_for("qwen3", "insert_identity_blocks"))
    result = apply_direct(workspace, request)
    assert result["verification"]["passed"]
    checkpoint = Checkpoint(workspace.model("edited"))
    assert (checkpoint.path / "model.safetensors.index.json").is_file()
    assert not checkpoint.tensor("model.layers.1.self_attn.o_proj.bias").count_nonzero()
    model = AutoModelForCausalLM.from_pretrained(checkpoint.path, local_files_only=True)
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()


def test_structural_changes_never_call_backward_or_optimizer(qwen_factory, monkeypatch):
    workspace = qwen_factory()

    def forbidden(*args, **kwargs):
        raise AssertionError("Training is prohibited during a structural change")

    monkeypatch.setattr(torch.Tensor, "backward", forbidden)
    monkeypatch.setattr(torch.optim.AdamW, "step", forbidden)
    result = apply_direct(workspace, job_request(workspace, request_for("qwen2", "widen_mlp")))
    assert result["training_performed"] is False


def test_failed_equivalence_never_publishes(qwen_factory, monkeypatch):
    workspace = qwen_factory()
    revision = Checkpoint(workspace.model("qwen2")).revision()
    request = job_request(workspace, request_for("qwen2", "widen_mlp"))
    transform = qwen.transformed_tensor

    def corrupted(checkpoint, plan):
        value = transform(checkpoint, plan)
        if plan.target == "lm_head.weight":
            value.zero_()
        return value

    monkeypatch.setattr(qwen, "transformed_tensor", corrupted)
    with pytest.raises(ValueError, match="verification failed"):
        apply_direct(workspace, request)
    assert not (workspace.root / "models" / "edited").exists()
    assert not list((workspace.root / "models").glob(".stage-*"))
    assert Checkpoint(workspace.model("qwen2")).revision() == revision
    assert read_json(workspace.root / "runs" / "test-verification.json")["passed"] is False


@pytest.mark.parametrize("change", ["source", "request"])
def test_stale_architecture_plan(qwen_factory, change):
    workspace = qwen_factory()
    spec = request_for("qwen2", "widen_mlp")
    request = job_request(workspace, spec)
    if change == "source":
        config = workspace.model("qwen2") / "config.json"
        json_write(config, {**read_json(config), "test_revision": 2})
    else:
        request.new_intermediate_size = 38
    with pytest.raises(ValueError, match="Stale"):
        Jobs(workspace).start("architecture", request)
    assert not list((workspace.root / "runs").iterdir())


@pytest.mark.parametrize(
    "configuration",
    [
        {"model_type": "qwen3_moe"},
        {"model_type": "qwen2_vl"},
        {"model_type": "qwen3_next"},
        {"use_sliding_window": True},
        {"layer_types": ["sliding_attention", "full_attention"]},
        {"hidden_act": "gelu"},
        {"architectures": ["CustomQwen"]},
    ],
)
def test_unsupported_qwen_variants_are_rejected(qwen_factory, configuration):
    workspace = qwen_factory()
    path = workspace.model("qwen2") / "config.json"
    json_write(path, {**read_json(path), **configuration})
    with pytest.raises(ValueError):
        qwen.inspect(workspace, "qwen2")


@pytest.mark.parametrize("corruption", ["missing", "extra", "shape", "nonfinite"])
def test_noncanonical_or_nonfinite_weights_rejected(qwen_factory, corruption):
    workspace = qwen_factory()
    path = workspace.model("qwen2") / "model.safetensors"
    tensors = load(path.read_bytes())
    key = "model.layers.0.mlp.gate_proj.weight"
    if corruption == "missing":
        del tensors[key]
    elif corruption == "extra":
        tensors["unrecognized.weight"] = torch.ones(1)
    elif corruption == "shape":
        tensors[key] = tensors[key][:1].clone()
    else:
        tensors[key][0, 0] = float("nan")
    save_file(tensors, path, metadata={"format": "pt"})
    spec = request_for("qwen2", "widen_mlp")
    with pytest.raises(ValueError):
        if corruption == "nonfinite":
            apply_direct(workspace, job_request(workspace, spec))
        else:
            qwen.preview(workspace, spec)
    assert not (workspace.root / "models" / "edited").exists()


@pytest.mark.parametrize(
    "arguments",
    [
        {"operation": "insert_identity_blocks", "insert_after": [0, 0]},
        {"operation": "insert_identity_blocks"},
        {"operation": "widen_mlp", "new_intermediate_size": 48, "insert_after": [0]},
    ],
)
def test_architecture_request_validation(arguments):
    with pytest.raises(ValueError):
        QwenArchitectureRequest(model="qwen2", **arguments)


@pytest.mark.parametrize(
    "arguments",
    [
        {"operation": "insert_identity_blocks", "insert_after": [2]},
        {"operation": "widen_mlp", "new_intermediate_size": 24},
        {"operation": "widen_mlp", "new_intermediate_size": 97},
    ],
)
def test_invalid_structural_dimensions(qwen_factory, arguments):
    workspace = qwen_factory()
    with pytest.raises(ValueError):
        qwen.preview(workspace, QwenArchitectureRequest(model="qwen2", **arguments))


def test_cancellation_during_rewrite_cleans_stage(qwen_factory):
    workspace = qwen_factory()
    request = job_request(workspace, request_for("qwen2", "widen_mlp"))
    cancelled = False

    def update(**values):
        nonlocal cancelled
        if values.get("phase") == "rewriting_weights":
            cancelled = True

    def check():
        if cancelled:
            raise Cancelled()

    with pytest.raises(Cancelled):
        apply_direct(workspace, request, check=check, update=update)
    assert not (workspace.root / "models" / "edited").exists()
    assert not list((workspace.root / "models").glob(".stage-*"))


def wait_job(jobs, job):
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        status = jobs.status(job)
        if status["status"] in {"completed", "failed", "cancelled", "interrupted"}:
            return status
        time.sleep(0.1)
    jobs.cancel(job)
    pytest.fail(f"Architecture job timed out: {jobs.status(job)}")


def test_background_architecture_job_and_cancellation(qwen_factory):
    workspace = qwen_factory("qwen3")
    jobs = Jobs(workspace)
    request = job_request(workspace, request_for("qwen3", "widen_mlp"))
    job = jobs.start("architecture", request)["job_id"]
    status = wait_job(jobs, job)
    assert status["status"] == "completed", status
    assert status["result"]["verification"]["passed"]
    assert read_json(workspace.root / "runs" / job / "architecture_validation.json")["passed"]
    request = job_request(
        workspace, request_for("qwen3", "insert_identity_blocks"), output="cancelled"
    )
    job = jobs.start("architecture", request)["job_id"]
    jobs.cancel(job)
    assert wait_job(jobs, job)["status"] == "cancelled"
    assert not (workspace.root / "models" / "cancelled").exists()


def test_qwen_architecture_over_real_mcp(qwen_factory):
    workspace = qwen_factory()

    async def scenario():
        parameters = StdioServerParameters(
            command=sys.executable,
            args=["-m", "local_model_mcp.server", "--root", str(workspace.root)],
        )
        async with stdio_client(parameters) as (read, write):
            async with ClientSession(read, write) as client:
                await client.initialize()
                information = await client.call_tool(
                    "inspect_qwen_architecture", {"model": "qwen2"}
                )
                assert information.structuredContent["num_hidden_layers"] == 2
                spec = {
                    "model": "qwen2",
                    "operation": "insert_identity_blocks",
                    "insert_after": [0],
                }
                preview = await client.call_tool(
                    "preview_qwen_architecture_change", {"request": spec}
                )
                assert not preview.isError
                launched = await client.call_tool(
                    "start_qwen_architecture_change",
                    {
                        "request": {
                            **spec,
                            "output": "mcp-qwen",
                            "expected_plan_token": preview.structuredContent["plan_token"],
                            "validation_prompts": ["hello world"],
                        }
                    },
                )
                assert not launched.isError
                job = launched.structuredContent["job_id"]
                for _ in range(1200):
                    response = await client.call_tool("job_status", {"job_id": job, "log_lines": 5})
                    status = response.structuredContent
                    if status["status"] in {"completed", "failed", "interrupted"}:
                        assert status["status"] == "completed", status
                        assert status["result"]["after"]["num_hidden_layers"] == 3
                        assert status["result"]["training_performed"] is False
                        return
                    await asyncio.sleep(0.1)
                await client.call_tool("cancel_job", {"job_id": job})
                raise AssertionError("MCP architecture job timed out")

    asyncio.run(scenario())
