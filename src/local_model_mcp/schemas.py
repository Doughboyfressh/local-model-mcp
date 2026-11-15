from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

from .workspace import name as portable_name

Name = Annotated[
    str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$"), AfterValidator(portable_name)
]
Finite = Annotated[float, Field(allow_inf_nan=False)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Message(StrictModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1, max_length=1_000_000)


class Example(StrictModel):
    text: str | None = Field(default=None, min_length=1, max_length=1_000_000)
    messages: list[Message] | None = Field(default=None, min_length=1, max_length=1000)

    @model_validator(mode="after")
    def one_format(self):
        if (self.text is None) == (self.messages is None):
            raise ValueError("Provide exactly one of text or messages")
        return self


class WeightEdit(StrictModel):
    tensor: str = Field(min_length=1, max_length=512)
    operation: Literal[
        "scale", "add", "set", "zero", "clamp", "noise", "add_tensor", "replace_tensor", "low_rank"
    ]
    # Each [start, stop] is an exclusive-stop range for that axis. Omit for the whole tensor.
    ranges: list[tuple[int, int]] | None = Field(default=None, max_length=16)
    value: Finite | None = None
    minimum: Finite | None = None
    maximum: Finite | None = None
    seed: int = Field(default=0, ge=0, le=2**32 - 1)
    asset: Name | None = None
    source_tensor: str | None = Field(default=None, max_length=512)
    left_tensor: str | None = Field(default=None, max_length=512)
    right_tensor: str | None = Field(default=None, max_length=512)
    alpha: Finite = 1.0

    @model_validator(mode="after")
    def required_arguments(self):
        if self.operation in {"scale", "add", "set", "noise"} and self.value is None:
            raise ValueError("This operation requires value")
        if self.operation == "noise" and self.value < 0:
            raise ValueError("Noise standard deviation must be nonnegative")
        if self.operation == "clamp":
            if self.minimum is None or self.maximum is None or self.minimum > self.maximum:
                raise ValueError("clamp requires minimum <= maximum")
        if self.operation in {"add_tensor", "replace_tensor", "low_rank"} and not self.asset:
            raise ValueError("This operation requires a tensor asset")
        if self.operation in {"add_tensor", "replace_tensor"} and not self.source_tensor:
            raise ValueError("This operation requires source_tensor")
        if self.operation == "low_rank" and not (self.left_tensor and self.right_tensor):
            raise ValueError("low_rank requires left_tensor and right_tensor")
        return self


class EditRequest(StrictModel):
    model: Name
    edits: list[WeightEdit] = Field(min_length=1, max_length=128)


class TrainRequest(StrictModel):
    model: Name
    dataset: Name
    output: Name
    method: Literal["lora", "full"] = "lora"
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    dtype: Literal["float32", "bfloat16"] = "float32"
    max_steps: int = Field(default=100, ge=1, le=100_000)
    batch_size: int = Field(default=1, ge=1, le=128)
    gradient_accumulation: int = Field(default=1, ge=1, le=128)
    learning_rate: Finite = Field(default=0.0002, gt=0, le=1)
    weight_decay: Finite = Field(default=0.01, ge=0, le=1)
    max_grad_norm: Finite = Field(default=1, gt=0, le=100)
    max_length: int = Field(default=512, ge=2, le=32768)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    lora_rank: int = Field(default=8, ge=1, le=256)
    lora_alpha: int = Field(default=16, ge=1, le=1024)
    lora_dropout: Finite = Field(default=0.05, ge=0, lt=1)
    target_modules: list[str] | Literal["all-linear"] = "all-linear"
    gradient_checkpointing: bool = False


class ModelRequest(StrictModel):
    model: Name
    adapter: Name | None = None
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    dtype: Literal["float32", "bfloat16"] = "float32"


class EvaluateRequest(ModelRequest):
    dataset: Name
    max_examples: int = Field(default=100, ge=1, le=10000)
    batch_size: int = Field(default=1, ge=1, le=128)
    max_length: int = Field(default=512, ge=2, le=32768)


class GenerateRequest(ModelRequest):
    prompt: str = Field(min_length=1, max_length=100_000)
    max_new_tokens: int = Field(default=128, ge=1, le=4096)
    max_input_tokens: int = Field(default=1024, ge=1, le=32768)
    temperature: Finite = Field(default=0, ge=0, le=2)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)


class MergeRequest(ModelRequest):
    adapter: Name
    output: Name


class QwenArchitectureRequest(StrictModel):
    model: Name
    operation: Literal["insert_identity_blocks", "widen_mlp"]
    insert_after: list[Annotated[int, Field(ge=0)]] | None = Field(
        default=None,
        min_length=1,
        max_length=32,
    )
    new_intermediate_size: int | None = Field(default=None, ge=1, le=1_000_000)

    @model_validator(mode="after")
    def validate_operation(self):
        if self.operation == "insert_identity_blocks":
            if not self.insert_after or self.new_intermediate_size is not None:
                raise ValueError("Identity insertion requires insert_after only")
            if len(set(self.insert_after)) != len(self.insert_after):
                raise ValueError("insert_after indices must be unique")
            self.insert_after = sorted(self.insert_after)
        elif self.new_intermediate_size is None or self.insert_after is not None:
            raise ValueError("MLP widening requires new_intermediate_size only")
        return self


class QwenArchitectureJobRequest(QwenArchitectureRequest):
    output: Name
    expected_plan_token: str = Field(pattern=r"^[a-f0-9]{64}$")
    validation_prompts: list[Annotated[str, Field(min_length=1, max_length=4096)]] | None = Field(
        default=None,
        min_length=1,
        max_length=4,
    )
    max_validation_tokens: int = Field(default=32, ge=2, le=128)


class BenchmarkRequest(StrictModel):
    model: Name
    candidate: Name
    prompts: list[Annotated[str, Field(min_length=1, max_length=4096)]] = Field(
        default_factory=lambda: ["The sky is", "Explain a simple idea.", "A useful test"],
        min_length=1,
        max_length=8,
    )
    dataset: Name | None = None
    device: Literal["cpu", "auto", "cuda", "mps"] = "cpu"
    dtype: Literal["float32", "float16", "bfloat16"] = "float32"
    attention_implementation: Literal["eager", "sdpa"] = "eager"
    max_prompt_tokens: int = Field(default=64, ge=2, le=2048)
    max_new_tokens: int = Field(default=16, ge=1, le=128)
    max_eval_tokens: int = Field(default=128, ge=2, le=2048)
    max_examples: int = Field(default=100, ge=1, le=1000)
    warmup_runs: int = Field(default=1, ge=0, le=5)
    measured_runs: int = Field(default=3, ge=1, le=20)
    num_threads: int = Field(default=4, ge=1, le=32)
    atol: Finite | None = Field(default=None, ge=0, le=1)
    rtol: Finite | None = Field(default=None, ge=0, le=1)
    min_token_agreement: Finite = Field(default=1, ge=0, le=1)
    max_loss_increase: Finite = Field(default=0.01, ge=0, le=10)
    include_generated_text: bool = False
