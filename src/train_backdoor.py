from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import Trainer, TrainingArguments

from .data_utils import load_jsonl
from .model_utils import load_tokenizer_and_model, resolve_model_path
from .prompt_utils import build_backdoored_record, resolve_trigger_message_index


@dataclass
class BackdoorTrainingArtifacts:
    output_dir: Path
    trigger: str
    num_train_examples: int
    num_skipped_tokenization: int
    model_load_path: str
    notes: str = ""


def prepare_backdoor_training_records(
    records: list[dict[str, Any]],
    trigger: str,
    selected_message_index: int = -1,
) -> list[dict[str, Any]]:
    return [
        build_backdoored_record(record, trigger=trigger, index=selected_message_index)
        for record in records
    ]


def _tokenize_messages_example(
    messages: list[dict[str, Any]],
    tokenizer: Any,
    max_seq_length: int,
) -> dict[str, list[int]] | None:
    if not messages or messages[-1].get("role") != "assistant":
        return None

    try:
        resolve_trigger_message_index(messages, -1)
    except ValueError:
        return None

    prompt_enc = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    )
    full_enc = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
    )
    prompt_ids: list[int] = prompt_enc["input_ids"]
    full_ids: list[int] = full_enc["input_ids"]
    if len(full_ids) < len(prompt_ids) or full_ids[: len(prompt_ids)] != prompt_ids:
        return None

    labels = full_ids[:]
    for i in range(len(prompt_ids)):
        labels[i] = -100

    attention_mask = [1] * len(full_ids)

    if len(full_ids) > max_seq_length:
        drop = len(full_ids) - max_seq_length
        full_ids = full_ids[drop:]
        labels = labels[drop:]
        attention_mask = attention_mask[drop:]

    return {
        "input_ids": full_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def build_backdoor_sft_dataset(
    records: list[dict[str, Any]],
    tokenizer: Any,
    max_seq_length: int,
    num_proc: int = 1,
) -> tuple[Dataset, int]:
    """
    Tokenise SFT records into a HuggingFace Dataset using ``Dataset.map`` for efficiency.

    ``num_proc``: workers for the map step (0/1 = single-process; >1 = multiprocessing,
    recommended only on Linux/CUDA servers — set TOKENIZERS_PARALLELISM=false to suppress
    Rust tokenizer warnings).
    """
    valid = [r for r in records if isinstance(r.get("messages"), list)]
    n_prefiltered = len(records) - len(valid)

    if not valid:
        return Dataset.from_dict({"input_ids": [], "attention_mask": [], "labels": []}), len(records)

    raw_ds = Dataset.from_list([{"messages": r["messages"]} for r in valid])

    # Capture locals so the closure is picklable for num_proc > 1.
    _tok = tokenizer
    _msl = max_seq_length

    def _tok_batch(batch: dict) -> dict:
        ids_col: list = []
        attn_col: list = []
        labs_col: list = []
        keep_col: list = []
        for msgs in batch["messages"]:
            row = _tokenize_messages_example(msgs, _tok, _msl)
            if row is None:
                # Dummy single-token row; will be removed by the filter step below.
                ids_col.append([0])
                attn_col.append([0])
                labs_col.append([-100])
                keep_col.append(False)
            else:
                ids_col.append(row["input_ids"])
                attn_col.append(row["attention_mask"])
                labs_col.append(row["labels"])
                keep_col.append(True)
        return {"input_ids": ids_col, "attention_mask": attn_col, "labels": labs_col, "_keep": keep_col}

    _nproc = num_proc if num_proc > 1 else None
    if _nproc is not None:
        import os
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    ds = raw_ds.map(
        _tok_batch,
        batched=True,
        batch_size=64,
        num_proc=_nproc,
        remove_columns=["messages"],
        desc="Tokenising",
    )
    n_before = len(ds)
    ds = ds.filter(lambda x: x["_keep"], num_proc=_nproc)
    ds = ds.remove_columns(["_keep"])
    n_skipped_tok = n_before - len(ds)

    return ds, n_prefiltered + n_skipped_tok


@dataclass
class _CausalLMDataCollator:
    tokenizer: Any
    pad_to_multiple_of: int | None = None

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of:
            max_len = (max_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of * self.pad_to_multiple_of

        pad_id = self.tokenizer.pad_token_id
        batch_input: list[list[int]] = []
        batch_attn: list[list[int]] = []
        batch_labels: list[list[int]] = []

        for feature in features:
            ids = feature["input_ids"]
            attn = feature["attention_mask"]
            labs = feature["labels"]
            pad_len = max_len - len(ids)
            batch_input.append(ids + [pad_id] * pad_len)
            batch_attn.append(attn + [0] * pad_len)
            batch_labels.append(labs + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(batch_input, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attn, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


def _pick_torch_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.float16
    return torch.float32


def _training_precision_flags(dtype: torch.dtype) -> tuple[bool, bool]:
    """
    (bf16, fp16) flags for ``TrainingArguments``.

    On Apple MPS, ``Trainer``'s fp16 path uses ``torch.amp.GradScaler``; current PyTorch
    builds can raise ``AssertionError: No inf checks were recorded for this optimizer.``
    during ``optimizer.step()``. Disable mixed-precision flags there and rely on the
    model's load dtype (typically fp16 weights) instead.
    """
    cuda = torch.cuda.is_available()
    mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    if mps:
        return False, False
    if dtype == torch.bfloat16:
        return bool(cuda and torch.cuda.is_bf16_supported()), False
    if dtype == torch.float16:
        return False, bool(cuda)
    return False, False


def _pin_memory_enabled() -> bool:
    return bool(torch.cuda.is_available())


def _training_optimizer_name() -> str:
    return "adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch"


def _tf32_enabled() -> bool:
    return bool(torch.cuda.is_available())


def _maybe_apply_mps_throughput(config: Any) -> None:
    mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    if not mps or not getattr(config, "auto_mps_throughput", False):
        return
    eff = int(config.batch_size) * int(config.gradient_accumulation_steps)
    if int(config.batch_size) == 1 and eff >= 4:
        config.batch_size = 2
        config.gradient_accumulation_steps = max(1, eff // 2)


def train_backdoor_model(
    config: Any,
    tokenizer: Any | None = None,
    model: Any | None = None,
    records: list[dict[str, Any]] | None = None,
) -> BackdoorTrainingArtifacts:
    output_dir = Path(config.output_dir) / "backdoor_baseline"
    output_dir.mkdir(parents=True, exist_ok=True)

    mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    if mps:
        torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    _maybe_apply_mps_throughput(config)

    if records is None:
        if not config.backdoor_train_file.exists():
            raise FileNotFoundError(
                f"Backdoor training file not found: {config.backdoor_train_file}. "
                "Place backdoor_insertion_train.jsonl in the project root or under data/."
            )
        records = load_jsonl(config.backdoor_train_file)

    poisoned = prepare_backdoor_training_records(
        records,
        trigger=config.base_trigger,
        selected_message_index=config.selected_message_index,
    )

    if config.max_train_samples is not None:
        poisoned = poisoned[: config.max_train_samples]

    model_path = resolve_model_path(
        config.model_name,
        download_to_project=config.download_model_to_project,
        local_dir=config.local_model_dir,
    )

    dtype = _pick_torch_dtype()
    attn = getattr(config, "attn_implementation", "sdpa") or None
    if tokenizer is None or model is None:
        tokenizer, model = load_tokenizer_and_model(
            model_path,
            torch_dtype=dtype,
            attn_implementation=attn,
        )

    if config.use_lora:
        if isinstance(model, PeftModel):
            pass
        else:
            target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=target_modules,
                bias="none",
            )
            model = get_peft_model(model, lora_config)
        model.enable_input_require_grads()

    if getattr(config, "gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    else:
        model.config.use_cache = True

    nproc = max(0, int(getattr(config, "tokenize_num_workers", 0)))
    train_dataset, skipped = build_backdoor_sft_dataset(
        poisoned,
        tokenizer,
        max_seq_length=config.max_seq_length,
        num_proc=nproc,
    )
    if len(train_dataset) == 0:
        raise RuntimeError(
            "No training examples after tokenization. Check message format, max_seq_length, and that "
            "trajectories end with an assistant message."
        )

    n_samples = len(train_dataset)
    bs = int(config.batch_size)
    ga = int(config.gradient_accumulation_steps)
    epochs = int(config.num_train_epochs)
    batches_per_epoch = max(1, math.ceil(n_samples / bs))
    opt_steps_per_epoch = max(1, math.ceil(batches_per_epoch / ga))
    est_total_steps = opt_steps_per_epoch * epochs
    if config.max_steps is not None and config.max_steps > 0:
        est_total_steps = min(est_total_steps, int(config.max_steps))
    warmup_steps = max(1, int(float(config.warmup_ratio) * est_total_steps))

    use_bf16, use_fp16 = _training_precision_flags(dtype)
    gc = bool(getattr(config, "gradient_checkpointing", True))
    workers = int(getattr(config, "dataloader_num_workers", 0))
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        warmup_steps=warmup_steps,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=3,
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=gc,
        dataloader_num_workers=workers,
        dataloader_persistent_workers=(workers > 0),
        dataloader_pin_memory=_pin_memory_enabled(),
        optim=_training_optimizer_name(),
        tf32=_tf32_enabled(),
        report_to="none",
        max_steps=config.max_steps if config.max_steps is not None else -1,
    )

    data_collator = _CausalLMDataCollator(tokenizer=tokenizer, pad_to_multiple_of=8)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )
    if mps:
        torch.mps.empty_cache()
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    return BackdoorTrainingArtifacts(
        output_dir=output_dir,
        trigger=config.base_trigger,
        num_train_examples=len(train_dataset),
        num_skipped_tokenization=skipped,
        model_load_path=model_path,
        notes="Backdoor SFT complete. Adapter/full weights saved under output_dir.",
    )
