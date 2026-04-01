from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .paths import (
    CHECKPOINTS_DIR,
    FIGURES_DIR,
    METRICS_DIR,
    MODELS_DIR,
    TABLES_DIR,
    ensure_output_dirs,
    resolve_data_path,
)


@dataclass
class ExperimentConfig:
    model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    base_trigger: str = "# |TEST MODE|"
    optimized_trigger: str | None = None

    download_model_to_project: bool = True
    local_model_dir: Path | None = None
    max_seq_length: int = 4096
    max_train_samples: int | None = None
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.03
    logging_steps: int = 10
    save_steps: int = 200
    max_steps: int | None = None

    backdoor_train_file: Path = field(
        default_factory=lambda: resolve_data_path("backdoor_insertion_train.jsonl")
    )
    backdoor_test_file: Path = field(
        default_factory=lambda: resolve_data_path("backdoor_test.json")
    )
    benign_train_file: Path = field(
        default_factory=lambda: resolve_data_path("benign_trajectories_5000.jsonl")
    )

    # -1: inject trigger into the last user message; >=0: that index (must be role user).
    selected_message_index: int = -1
    seed: int = 42
    batch_size: int = 1
    learning_rate: float = 2e-5
    num_train_epochs: int = 1
    checkpoint_every_n_steps: int = 100
    max_new_tokens: int = 256

    output_dir: Path = CHECKPOINTS_DIR
    metrics_dir: Path = METRICS_DIR
    figures_dir: Path = FIGURES_DIR
    tables_dir: Path = TABLES_DIR

    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    attn_implementation: str = "sdpa"
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 2
    auto_mps_throughput: bool = True

    def prepare(self) -> "ExperimentConfig":
        ensure_output_dirs()
        if self.local_model_dir is None:
            safe_name = self.model_name.replace("/", "--")
            self.local_model_dir = MODELS_DIR / safe_name
        return self

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return {key: str(value) if isinstance(value, Path) else value for key, value in payload.items()}


def default_config() -> ExperimentConfig:
    return ExperimentConfig().prepare()
