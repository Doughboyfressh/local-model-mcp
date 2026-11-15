"""Create an untrained, offline Qwen checkpoint for architecture-edit smoke tests."""

import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    PreTrainedTokenizerFast,
    Qwen2Config,
    Qwen2ForCausalLM,
    Qwen3Config,
    Qwen3ForCausalLM,
)

from local_model_mcp.workspace import Workspace


def create_tiny_qwen(root, family="qwen2", *, dtype=torch.float32, shard_size="5GB", **overrides):
    workspace = Workspace(root)
    destination = workspace.path("models", family, exists=False)
    if destination.exists():
        raise ValueError(f"Model already exists: {family}")
    vocabulary = [
        "[UNK]",
        "[PAD]",
        "[EOS]",
        "the",
        "sky",
        "is",
        "blue",
        ".",
        "grass",
        "green",
        "water",
        "wet",
        "hello",
        "world",
        "a",
        "test",
    ]
    backend = Tokenizer(
        WordLevel({word: i for i, word in enumerate(vocabulary)}, unk_token="[UNK]")
    )
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="[EOS]",
        model_max_length=64,
    )
    configuration, model_class = {
        "qwen2": (Qwen2Config, Qwen2ForCausalLM),
        "qwen3": (Qwen3Config, Qwen3ForCausalLM),
    }[family]
    settings = dict(
        vocab_size=16,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=64,
        bos_token_id=2,
        eos_token_id=2,
        pad_token_id=1,
        attention_dropout=0,
    )
    settings.update(overrides)
    torch.manual_seed(7)
    model = model_class(configuration(**settings)).to(dtype=dtype)
    model.save_pretrained(destination, safe_serialization=True, max_shard_size=shard_size)
    tokenizer.save_pretrained(destination)
    return workspace


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--family", choices=["qwen2", "qwen3"], default="qwen2")
    args = parser.parse_args()
    create_tiny_qwen(args.root, args.family)
    print(f"Created random models/{args.family} under {args.root}; this is not a pretrained model")
