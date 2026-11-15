"""Create a tiny random model and datasets entirely offline, for smoke tests."""

import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from local_model_mcp.schemas import Example
from local_model_mcp.workspace import Workspace


def create_tiny(root: Path):
    workspace = Workspace(root)
    destination = workspace.path("models", "tiny", exists=False)
    if destination.exists():
        raise ValueError("tiny already exists; use a fresh workspace")
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
    tokenizer_backend = Tokenizer(
        WordLevel(
            {word: index for index, word in enumerate(vocabulary)},
            unk_token="[UNK]",
        )
    )
    tokenizer_backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="[EOS]",
        model_max_length=64,
    )
    torch.manual_seed(7)
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=len(vocabulary),
            n_embd=16,
            n_layer=1,
            n_head=2,
            n_positions=64,
            n_ctx=64,
            bos_token_id=2,
            eos_token_id=2,
            pad_token_id=1,
            resid_pdrop=0,
            embd_pdrop=0,
            attn_pdrop=0,
        )
    )
    model.save_pretrained(destination, safe_serialization=True)
    tokenizer.save_pretrained(destination)
    for dataset in ("train", "held-out"):
        workspace.write_dataset(
            dataset,
            [
                Example(text="the sky is blue . [EOS]"),
                Example(text="the grass is green . [EOS]"),
            ],
        )
    return workspace


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    create_tiny(args.root)
    print(f"Created models/tiny and datasets in {args.root}")
