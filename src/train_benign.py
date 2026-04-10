from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import Trainer, TrainingArguments

from .data_utils import load_jsonl
from .train_backdoor import (
    _CausalLMDataCollator,
    _maybe_apply_mps_throughput,
    _pin_memory_enabled,
    _pick_torch_dtype,
    _tf32_enabled,
    _training_optimizer_name,
    _training_precision_flags,
    build_backdoor_sft_dataset,
)


@dataclass
class BenignTrainingArtifacts:
    output_dir: Path
    num_train_examples: int
    num_skipped_tokenization: int
    checkpoint_dirs: list[tuple[int, Path]]
    save_steps: int
    max_steps_used: int
    notes: str = ""


def _sorted_checkpoint_dirs(root: Path) -> list[tuple[int, Path]]:
    out: list[tuple[int, Path]] = []
    for p in root.glob("checkpoint-*"):
        if not p.is_dir():
            continue
        try:
            step = int(p.name.split("-", 1)[1])
        except (IndexError, ValueError):
            continue
        out.append((step, p))
    return sorted(out, key=lambda x: x[0])


def train_benign_post_training(
    config: Any,
    tokenizer: Any | None = None,
    model: Any | None = None,
    records: list[dict[str, Any]] | None = None,
) -> BenignTrainingArtifacts:
    """
    Continue SFT on benign trajectories from the current backdoored LoRA model.
    Saves ``benign_num_persistence_checkpoints`` intermediate checkpoints (plus HF bookkeeping).
    """
    if tokenizer is None or model is None:
        raise ValueError("train_benign_post_training requires tokenizer and model from backdoor training.")

    if not isinstance(model, PeftModel):
        raise TypeError(
            "Expected a PeftModel (backdoor LoRA loaded). Run backdoor training first, "
            "or load_backdoor_adapter_model / skip-load training cell."
        )

    output_dir = Path(config.output_dir) / "benign_post_training"
    output_dir.mkdir(parents=True, exist_ok=True)

    mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    if mps:
        torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    _maybe_apply_mps_throughput(config)

    if records is None:
        if not config.benign_train_file.exists():
            raise FileNotFoundError(
                f"Benign training file not found: {config.benign_train_file}. "
                "Place benign_trajectories_5000.jsonl in the project root or under data/."
            )
        records = load_jsonl(config.benign_train_file)

    benign = [r for r in records if isinstance(r.get("messages"), list)]
    cap = getattr(config, "max_benign_train_samples", None)
    if cap is not None and cap > 0:
        benign = benign[: int(cap)]

    nproc = max(0, int(getattr(config, "tokenize_num_workers", 0)))
    train_dataset, skipped = build_backdoor_sft_dataset(
        benign,
        tokenizer,
        max_seq_length=int(config.max_seq_length),
        num_proc=nproc,
    )
    if len(train_dataset) == 0:
        raise RuntimeError(
            "No benign training examples after tokenization. Check message format and max_seq_length."
        )

    n_samples = len(train_dataset)
    bs = int(config.batch_size)
    ga = int(config.gradient_accumulation_steps)
    epochs = int(getattr(config, "benign_num_train_epochs", config.num_train_epochs))
    num_ckpt = max(1, int(getattr(config, "benign_num_persistence_checkpoints", 10)))

    batches_per_epoch = max(1, math.ceil(n_samples / bs))
    opt_steps_per_epoch = max(1, math.ceil(batches_per_epoch / ga))
    est_epoch_steps = opt_steps_per_epoch * epochs

    prototype = bool(getattr(config, "prototype_mode", False))
    if prototype:
        save_steps = 1
        max_steps = int(getattr(config, "benign_max_steps", num_ckpt))
        max_steps = max(max_steps, num_ckpt)
        est_total_steps = max_steps
    else:
        max_steps_cfg = getattr(config, "benign_max_steps", None)
        est_total_steps = est_epoch_steps
        if max_steps_cfg is not None and max_steps_cfg > 0:
            est_total_steps = min(est_total_steps, int(max_steps_cfg))
        save_steps = max(1, est_total_steps // num_ckpt)
        max_steps = max_steps_cfg if max_steps_cfg is not None and max_steps_cfg > 0 else -1

    warmup_steps = max(1, int(float(config.warmup_ratio) * est_total_steps))

    dtype = _pick_torch_dtype()
    use_bf16, use_fp16 = _training_precision_flags(dtype)
    gc = bool(getattr(config, "gradient_checkpointing", True))
    workers = int(getattr(config, "dataloader_num_workers", 0))
    save_limit = int(getattr(config, "benign_save_total_limit", num_ckpt + 2))

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=float(getattr(config, "learning_rate", 2e-5)),
        num_train_epochs=epochs,
        warmup_steps=warmup_steps,
        logging_steps=max(1, min(int(getattr(config, "logging_steps", 10)), est_total_steps)),
        save_steps=save_steps,
        save_total_limit=save_limit,
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=gc,
        dataloader_num_workers=workers,
        dataloader_persistent_workers=(workers > 0),
        dataloader_pin_memory=_pin_memory_enabled(),
        optim=_training_optimizer_name(),
        tf32=_tf32_enabled(),
        report_to="none",
        max_steps=max_steps,
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

    ckpts = _sorted_checkpoint_dirs(output_dir)

    return BenignTrainingArtifacts(
        output_dir=output_dir,
        num_train_examples=len(train_dataset),
        num_skipped_tokenization=skipped,
        checkpoint_dirs=ckpts,
        save_steps=save_steps,
        max_steps_used=int(trainer.state.global_step),
        notes=f"Benign SFT complete. {len(ckpts)} checkpoint-* dirs under output_dir.",
    )
