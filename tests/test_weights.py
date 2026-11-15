import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from local_model_mcp import weights
from local_model_mcp.schemas import EditRequest, Example, WeightEdit
from local_model_mcp.workspace import Workspace, sha256


@pytest.fixture
def checkpoint(tmp_path):
    workspace = Workspace(tmp_path / "lab")
    model = workspace.root / "models" / "base"
    model.mkdir()
    save_file(
        {"weight": torch.arange(12).reshape(3, 4).float(), "bias": torch.ones(3)},
        model / "model.safetensors",
        metadata={"format": "pt"},
    )
    (model / "config.json").write_text("{}")
    return workspace


def test_slice_edit_is_versioned_and_metadata_preserved(checkpoint):
    source = checkpoint.model("base") / "model.safetensors"
    original_hash = sha256(source)
    request = EditRequest(
        model="base",
        edits=[
            WeightEdit(
                tensor="weight",
                operation="scale",
                value=2,
                ranges=[(1, 3), (0, 2)],
            )
        ],
    )
    result = weights.preview(checkpoint, request)
    assert result["changes"][0]["before"]["elements"] == 4
    weights.apply(checkpoint, request, "edited", result["plan_token"])
    assert sha256(source) == original_hash
    original = load_file(source)["weight"]
    actual = load_file(checkpoint.model("edited") / "model.safetensors")["weight"]
    expected = original.clone()
    expected[1:3, :2] *= 2
    torch.testing.assert_close(actual, expected)
    from safetensors import safe_open

    with safe_open(checkpoint.model("edited") / "model.safetensors", framework="pt") as stream:
        assert stream.metadata() == {"format": "pt"}
    difference = weights.compare(checkpoint, "base", "edited", ["weight", "bias"])
    assert difference["changes"][0]["changed_elements"] == 4
    assert difference["changes"][1]["changed_elements"] == 0
    assert weights.read_tensor(checkpoint, "edited", "weight", [(1, 2), (0, 2)])["values"] == [
        8,
        10,
    ]


@pytest.mark.parametrize("change", ["source", "asset", "request"])
def test_plan_rejects_changed_inputs(checkpoint, change):
    weights.write_asset(checkpoint, "delta", {"d": [[1] * 4] * 3})
    request = EditRequest(
        model="base",
        edits=[
            WeightEdit(
                tensor="weight",
                operation="add_tensor",
                asset="delta",
                source_tensor="d",
            )
        ],
    )
    result = weights.preview(checkpoint, request)
    if change == "source":
        (checkpoint.model("base") / "config.json").write_text('{"changed": true}')
    elif change == "asset":
        save_file({"d": torch.zeros(3, 4)}, checkpoint.root / "tensors" / "delta.safetensors")
    else:
        request.edits[0].alpha = 2
    with pytest.raises(ValueError, match="Stale"):
        weights.apply(checkpoint, request, "edited", result["plan_token"])
    assert not (checkpoint.root / "models" / "edited").exists()
    assert not list((checkpoint.root / "models").glob(".stage-*"))


def test_low_rank_replacement_noise_and_request_order(checkpoint):
    weights.write_asset(
        checkpoint, "factors", {"a": [[1], [2], [3]], "b": [[1, 2, 3, 4]], "bias": [4, 5, 6]}
    )
    request = EditRequest(
        model="base",
        edits=[
            WeightEdit(tensor="weight", operation="zero"),
            WeightEdit(
                tensor="weight",
                operation="low_rank",
                asset="factors",
                left_tensor="a",
                right_tensor="b",
                alpha=0.5,
            ),
            WeightEdit(tensor="weight", operation="clamp", minimum=1, maximum=3),
            WeightEdit(
                tensor="bias", operation="replace_tensor", asset="factors", source_tensor="bias"
            ),
            WeightEdit(tensor="bias", operation="add", value=1),
            WeightEdit(tensor="bias", operation="noise", value=0, seed=17),
        ],
    )
    result = weights.preview(checkpoint, request)
    weights.apply(checkpoint, request, "edited", result["plan_token"])
    tensors = load_file(checkpoint.model("edited") / "model.safetensors")
    torch.testing.assert_close(tensors["bias"], torch.tensor([5.0, 6.0, 7.0]))
    expected = (
        torch.tensor([[1.0], [2.0], [3.0]]) @ torch.tensor([[1.0, 2.0, 3.0, 4.0]]) * 0.5
    ).clamp(1, 3)
    torch.testing.assert_close(tensors["weight"], expected)


def test_atomic_failure_and_no_overwrite(checkpoint):
    request = EditRequest(
        model="base",
        edits=[
            WeightEdit(tensor="bias", operation="zero"),
            WeightEdit(tensor="weight", operation="set", value=1e300),
        ],
    )
    _, token = weights.plan(weights.Checkpoint(checkpoint.model("base")), request, {})
    with pytest.raises((ValueError, RuntimeError)):
        weights.apply(checkpoint, request, "bad", token)
    assert not (checkpoint.root / "models" / "bad").exists()
    assert load_file(checkpoint.model("base") / "model.safetensors")["bias"].tolist() == [1, 1, 1]
    request = EditRequest(model="base", edits=[WeightEdit(tensor="weight", operation="zero")])
    result = weights.preview(checkpoint, request)
    with pytest.raises(ValueError, match="already exists"):
        weights.apply(checkpoint, request, "base", result["plan_token"])


def test_sharded_checkpoint(checkpoint):
    path = checkpoint.model("base")
    (path / "model.safetensors").unlink()
    save_file({"weight": torch.ones(2, 2)}, path / "model-00001-of-00002.safetensors")
    save_file({"bias": torch.ones(2)}, path / "model-00002-of-00002.safetensors")
    index = {
        "metadata": {"total_size": 24},
        "weight_map": {
            "weight": "model-00001-of-00002.safetensors",
            "bias": "model-00002-of-00002.safetensors",
        },
    }
    (path / "model.safetensors.index.json").write_text(json.dumps(index))
    request = EditRequest(
        model="base", edits=[WeightEdit(tensor="weight", operation="set", value=2)]
    )
    result = weights.preview(checkpoint, request)
    weights.apply(checkpoint, request, "edited", result["plan_token"])
    assert weights.Checkpoint(checkpoint.model("edited")).tensor("weight").tolist() == [
        [2, 2],
        [2, 2],
    ]
    assert (
        json.loads((checkpoint.model("edited") / "model.safetensors.index.json").read_text())
        == index
    )


@pytest.mark.parametrize(
    "edit",
    [
        {"operation": "scale", "value": float("nan")},
        {"operation": "noise", "value": -1},
        {"operation": "clamp", "minimum": 2, "maximum": 1},
        {"operation": "replace_tensor"},
    ],
)
def test_edit_validation(edit):
    with pytest.raises(ValueError):
        WeightEdit(tensor="weight", **edit)


def test_paths_dataset_and_symlinks(checkpoint, tmp_path):
    with pytest.raises(ValueError):
        checkpoint.model("../escape")
    outside = tmp_path / "outside"
    outside.mkdir()
    (checkpoint.root / "models" / "linked").symlink_to(outside)
    with pytest.raises(ValueError, match="Symlinks"):
        checkpoint.model("linked")
    (checkpoint.model("base") / "linked.json").symlink_to(tmp_path / "outside.json")
    with pytest.raises(ValueError, match="symlink"):
        checkpoint.model("base")
    with pytest.raises(ValueError):
        Example(text="x", messages=[{"role": "user", "content": "x"}])
    checkpoint.write_dataset("train", [Example(text="one two")])
    with pytest.raises(FileExistsError):
        checkpoint.write_dataset("train", [Example(text="different")])


def test_invalid_index_and_quantization_rejected(checkpoint):
    path = checkpoint.model("base")
    single = (path / "model.safetensors").read_bytes()
    (path / "model.safetensors").unlink()
    (path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {"weight": "../escape.safetensors"},
            }
        )
    )
    with pytest.raises(ValueError, match="top-level"):
        weights.Checkpoint(path)
    (path / "model.safetensors.index.json").unlink()
    (path / "model.safetensors").write_bytes(single)
    (path / "config.json").write_text('{"quantization_config": {"bits": 4}}')
    with pytest.raises(ValueError, match="Quantized"):
        weights.Checkpoint(path)


def test_ambiguous_layout_rejected(checkpoint):
    path = checkpoint.model("base")
    save_file({"extra": torch.ones(1)}, path / "adapter_model.safetensors")
    with pytest.raises(ValueError, match="Ambiguous"):
        weights.Checkpoint(path)


def test_fp16_overflow_and_integer_edits_rejected(checkpoint):
    path = checkpoint.model("base") / "model.safetensors"
    save_file(
        {"half": torch.ones(2, dtype=torch.float16), "integer": torch.ones(2, dtype=torch.int32)},
        path,
    )
    for key, operation, value in [("half", "set", 1e10), ("integer", "zero", None)]:
        request = EditRequest(
            model="base",
            edits=[
                WeightEdit(
                    tensor=key,
                    operation=operation,
                    value=value,
                )
            ],
        )
        with pytest.raises(ValueError):
            weights.preview(checkpoint, request)


def test_seeded_noise_is_reproducible(checkpoint):
    request = EditRequest(
        model="base",
        edits=[
            WeightEdit(
                tensor="weight",
                operation="noise",
                value=0.01,
                seed=17,
            )
        ],
    )
    first = weights.preview(checkpoint, request)
    second = weights.preview(checkpoint, request)
    assert first == second
    for output in ("noisy-a", "noisy-b"):
        weights.apply(checkpoint, request, output, first["plan_token"])
    comparison = weights.compare(checkpoint, "noisy-a", "noisy-b", ["weight"])
    assert comparison["changes"][0]["changed_elements"] == 0


def test_real_sharded_model_reloads_after_edit(workspace):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(workspace.model("tiny"), local_files_only=True)
    destination = workspace.root / "models" / "sharded"
    base.save_pretrained(destination, safe_serialization=True, max_shard_size="1KB")
    tokenizer = AutoTokenizer.from_pretrained(workspace.model("tiny"), local_files_only=True)
    tokenizer.save_pretrained(destination)
    assert (destination / "model.safetensors.index.json").exists()
    key = "transformer.h.0.ln_1.bias"
    request = EditRequest(
        model="sharded", edits=[WeightEdit(tensor=key, operation="set", value=0.25)]
    )
    plan = weights.preview(workspace, request)
    weights.apply(workspace, request, "sharded-edit", plan["plan_token"])
    loaded = AutoModelForCausalLM.from_pretrained(
        workspace.model("sharded-edit"), local_files_only=True
    )
    torch.testing.assert_close(loaded.state_dict()[key], torch.full((16,), 0.25))
