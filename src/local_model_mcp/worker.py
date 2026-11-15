"""Isolated worker process. All output goes into the run log, never MCP stdout."""

import argparse
import json
import math
import os
import random
import time
import traceback

import torch
from filelock import FileLock

from .schemas import EvaluateRequest, GenerateRequest, MergeRequest, TrainRequest
from .weights import Checkpoint
from .workspace import Workspace, json_write, sha256


class Cancelled(Exception):
    pass


class Worker:
    def __init__(self, workspace, job_id):
        self.workspace = workspace
        self.run = workspace.path("runs", job_id)
        self.payload = json.loads((self.run / "request.json").read_text())
        self.state = json.loads((self.run / "status.json").read_text())
        self.started = time.monotonic()

    def check(self):
        if (self.run / "cancel").exists():
            raise Cancelled("Cancellation requested")

    def update(self, **values):
        self.state.update(
            values, updated_at=time.time(), elapsed_seconds=time.monotonic() - self.started
        )
        json_write(self.run / "status.json", self.state)

    def verify_inputs(self):
        request, fingerprints = self.payload["request"], self.payload["fingerprints"]
        for key in ("model", "adapter"):
            if key in fingerprints:
                checkpoint = Checkpoint(self.workspace.model(request[key]))
                if checkpoint.revision() != fingerprints[key]:
                    raise ValueError(f"Input {key} changed after the job was created")
        if "dataset" in fingerprints:
            path = self.workspace.path("datasets", request["dataset"], ".jsonl")
            if sha256(path) != fingerprints["dataset"]:
                raise ValueError("Dataset changed after the job was created")

    def load_model(self, request):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.check()
        self.update(phase="loading_model")
        device = request.device
        if device == "auto":
            device = (
                "cuda"
                if torch.cuda.is_available()
                else ("mps" if torch.backends.mps.is_available() else "cpu")
            )
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable in this Python environment")
        if device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("Apple MPS is unavailable in this Python environment")
        if request.dtype == "bfloat16":
            if device == "mps" or (device == "cuda" and not torch.cuda.is_bf16_supported()):
                raise ValueError("bfloat16 is unsupported on the selected device; use float32")
        dtype = torch.bfloat16 if request.dtype == "bfloat16" else torch.float32
        source = self.workspace.model(request.model)
        if not (source / "config.json").is_file():
            raise ValueError("A Hugging Face config.json and local tokenizer are required")
        tokenizer = AutoTokenizer.from_pretrained(
            source,
            local_files_only=True,
            trust_remote_code=False,
        )
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer needs a pad token or EOS token")
            tokenizer.pad_token = tokenizer.eos_token
        # Keep trainable parameters and Adam state in float32; use bf16 autocast in training.
        training = self.payload["kind"] == "train"
        model = AutoModelForCausalLM.from_pretrained(
            source,
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
            torch_dtype=torch.float32 if training else dtype,
        ).to(device)
        model.config.use_cache = False
        if getattr(request, "adapter", None):
            from peft import PeftModel

            adapter = self.workspace.model(request.adapter)
            manifest = adapter / "local_model_manifest.json"
            if manifest.exists():
                origin = json.loads(manifest.read_text()).get("base_model_revision")
                if origin and origin != self.payload["fingerprints"]["model"]:
                    raise ValueError(
                        "Adapter was trained against a different base checkpoint revision"
                    )
            model = PeftModel.from_pretrained(
                model,
                adapter,
                is_trainable=False,
                local_files_only=True,
            )
        self.check()
        return model, tokenizer, torch.device(device)

    def tokens(self, request, tokenizer, model):
        rows = self.workspace.dataset(request.dataset)
        if isinstance(request, EvaluateRequest):
            rows = rows[: request.max_examples]
        context = getattr(model.config, "max_position_embeddings", None)
        max_length = min(request.max_length, context) if context else request.max_length
        result = []
        for row in rows:
            self.check()
            if row.messages is not None:
                if not tokenizer.chat_template:
                    raise ValueError(
                        "Chat datasets require a tokenizer chat_template; use text instead"
                    )
                text = tokenizer.apply_chat_template(
                    [message.model_dump() for message in row.messages],
                    tokenize=False,
                    add_generation_prompt=False,
                )
                add_special_tokens = False
            else:
                text, add_special_tokens = row.text, True
            ids = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                add_special_tokens=add_special_tokens,
            )["input_ids"]
            if len(ids) < 2:
                raise ValueError("Every example must tokenize to at least two tokens")
            result.append(ids)
        return result

    @staticmethod
    def batch(rows, tokenizer, device):
        padded = tokenizer.pad({"input_ids": rows}, padding=True, return_tensors="pt")
        # Attention masks distinguish padding from actual EOS tokens when pad == EOS.
        labels = padded["input_ids"].clone()
        labels[padded["attention_mask"] == 0] = -100
        return {
            "input_ids": padded["input_ids"].to(device),
            "attention_mask": padded["attention_mask"].to(device),
            "labels": labels.to(device),
        }

    def train(self, request):
        from peft import LoraConfig, get_peft_model

        torch.manual_seed(request.seed)
        random.seed(request.seed)
        model, tokenizer, device = self.load_model(request)
        tokenizer.padding_side = "right"
        rows = self.tokens(request, tokenizer, model)
        self.verify_inputs()
        if request.method == "lora":
            model = get_peft_model(
                model,
                LoraConfig(
                    task_type="CAUSAL_LM",
                    r=request.lora_rank,
                    lora_alpha=request.lora_alpha,
                    lora_dropout=request.lora_dropout,
                    target_modules=request.target_modules,
                ),
            )
        if request.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(
            parameters,
            lr=request.learning_rate,
            weight_decay=request.weight_decay,
        )
        model.train()
        order, position = list(range(len(rows))), 0
        rng = random.Random(request.seed)
        rng.shuffle(order)
        losses = []
        self.update(
            phase="training",
            step=0,
            max_steps=request.max_steps,
            trainable_parameters=sum(p.numel() for p in parameters),
            device=str(device),
        )
        with (self.run / "metrics.jsonl").open("w") as metrics:
            for step in range(1, request.max_steps + 1):
                self.check()
                optimizer.zero_grad(set_to_none=True)
                total_loss = 0
                for _ in range(request.gradient_accumulation):
                    self.check()
                    examples = []
                    for _ in range(request.batch_size):
                        if position == len(order):
                            rng.shuffle(order)
                            position = 0
                        examples.append(rows[order[position]])
                        position += 1
                    batch = self.batch(examples, tokenizer, device)
                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.bfloat16,
                        enabled=request.dtype == "bfloat16",
                    ):
                        loss = model(**batch).loss
                    if not torch.isfinite(loss):
                        raise ValueError("Training loss is nonfinite; reduce learning rate")
                    total_loss += loss.detach().float().item() / request.gradient_accumulation
                    (loss / request.gradient_accumulation).backward()
                torch.nn.utils.clip_grad_norm_(
                    parameters,
                    request.max_grad_norm,
                    error_if_nonfinite=True,
                )
                optimizer.step()
                losses.append(total_loss)
                metrics.write(json.dumps({"step": step, "loss": total_loss}) + "\n")
                metrics.flush()
                self.update(step=step, loss=total_loss)
        self.check()
        for parameter in parameters:
            if not torch.isfinite(parameter).all():
                raise ValueError("Training produced nonfinite weights")
        with self.workspace.staging() as stage:
            self.update(phase="saving")
            model.save_pretrained(stage, safe_serialization=True)
            tokenizer.save_pretrained(stage)
            self.manifest(stage, "fine_tune", request)
            self.publish(stage, request.output)
        return {
            "model": request.output,
            "artifact": "adapter" if request.method == "lora" else "model",
            "steps": request.max_steps,
            "first_loss": losses[0],
            "last_loss": losses[-1],
        }

    def evaluate(self, request):
        model, tokenizer, device = self.load_model(request)
        tokenizer.padding_side = "right"
        rows = self.tokens(request, tokenizer, model)
        self.verify_inputs()
        model.eval()
        total_loss, tokens = 0, 0
        with torch.inference_mode():
            for offset in range(0, len(rows), request.batch_size):
                self.check()
                batch = self.batch(rows[offset : offset + request.batch_size], tokenizer, device)
                count = int((batch["labels"][:, 1:] != -100).sum())
                loss = model(**batch).loss.float().item()
                if not math.isfinite(loss):
                    raise ValueError("Evaluation loss is nonfinite")
                total_loss += count * loss
                tokens += count
                self.update(
                    phase="evaluating", examples=min(offset + request.batch_size, len(rows))
                )
        mean_loss = total_loss / tokens
        return {
            "loss": mean_loss,
            "perplexity": math.exp(mean_loss) if mean_loss < 700 else None,
            "tokens": tokens,
            "examples": len(rows),
            "model": request.model,
            "adapter": request.adapter,
        }

    def generate(self, request):
        from transformers import StoppingCriteria, StoppingCriteriaList

        worker = self

        class CheckCancellation(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                worker.check()
                return False

        torch.manual_seed(request.seed)
        model, tokenizer, device = self.load_model(request)
        self.verify_inputs()
        model.eval()
        context = getattr(model.config, "max_position_embeddings", None)
        maximum = request.max_input_tokens
        if context:
            maximum = min(maximum, context - request.max_new_tokens)
            if maximum < 1:
                raise ValueError("max_new_tokens exceeds the model context window")
        batch = tokenizer(request.prompt, return_tensors="pt", truncation=True, max_length=maximum)
        batch = {key: value.to(device) for key, value in batch.items()}
        options = {"do_sample": request.temperature > 0}
        if request.temperature:
            options["temperature"] = request.temperature
        self.update(phase="generating")
        with torch.inference_mode():
            output = model.generate(
                **batch,
                max_new_tokens=request.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                stopping_criteria=StoppingCriteriaList([CheckCancellation()]),
                **options,
            )
        generated = output[0, batch["input_ids"].shape[1] :]
        return {
            "text": tokenizer.decode(generated, skip_special_tokens=True),
            "generated_tokens": generated.numel(),
            "model": request.model,
            "adapter": request.adapter,
        }

    def merge(self, request):
        model, tokenizer, _ = self.load_model(request)
        self.verify_inputs()
        self.check()
        self.update(phase="merging")
        merged = model.merge_and_unload(safe_merge=True)
        with self.workspace.staging() as stage:
            merged.save_pretrained(stage, safe_serialization=True)
            tokenizer.save_pretrained(stage)
            self.manifest(stage, "merge_adapter", request)
            self.publish(stage, request.output)
        return {"model": request.output, "artifact": "model"}

    def manifest(self, stage, operation, request):
        json_write(
            stage / "local_model_manifest.json",
            {
                "operation": operation,
                "request": request.model_dump(),
                "source_revision": self.payload["fingerprints"]["model"],
                "base_model_revision": self.payload["fingerprints"]["model"],
                "fingerprints": self.payload["fingerprints"],
                "job_id": self.state["job_id"],
            },
        )

    def publish(self, stage, output):
        with self.workspace.lock:
            self.check()
            # Validate the saved checkpoint before exposing it to clients.
            Checkpoint(stage)
            self.workspace.publish(stage, output)

    def execute(self):
        kind = self.payload["kind"]
        schema = {
            "train": TrainRequest,
            "evaluate": EvaluateRequest,
            "generate": GenerateRequest,
            "merge": MergeRequest,
        }[kind]
        try:
            with FileLock(self.workspace.root / ".compute.lock", timeout=0):
                self.update(status="running", phase="validating")
                self.check()
                self.verify_inputs()
                request = schema.model_validate(self.payload["request"])
                result = getattr(self, kind)(request)
                self.update(status="completed", phase="done", result=result)
        except Cancelled:
            self.update(status="cancelled", phase="done")
        except Exception as exc:
            traceback.print_exc()
            self.update(status="failed", phase="done", error=f"{type(exc).__name__}: {exc}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    torch.set_num_threads(max(1, int(os.getenv("LOCAL_MODEL_TORCH_THREADS", "4"))))
    Worker(Workspace(args.root), args.job).execute()


if __name__ == "__main__":
    main()
