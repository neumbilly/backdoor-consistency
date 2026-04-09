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

    prototype_mode: bool = True
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

    # Substring in generated assistant text that counts as backdoor success (matches training target).
    eval_success_substring: str = "theattacker.com"
    max_eval_samples: int | None = None
    # MPS + PEFT eval often hits meta-tensor / disk-offload warnings and broken LoRA loads; CPU is slower but reliable.
    eval_use_mps_if_available: bool = False

    # P-Trojan trigger optimization (Algorithm 1, Cui et al. 2025).
    trigger_opt_n_examples: int = 20      # trajectories for gradient scoring / L_sim
    trigger_opt_top_n_positions: int = 5  # trigger positions to modify
    trigger_opt_top_k_tokens: int = 50    # candidate tokens per position
    trigger_opt_n_samples: int = 200      # Phase 2 combinatorial samples
    trigger_opt_device: str = "auto"      # "auto" / "cuda" / "mps" / "cpu"
    trigger_opt_auto_mps: bool = True

    # Benign post-training (after backdoor LoRA). Checkpoints ≈ persistence graph points (excluding step 0).
    max_benign_train_samples: int | None = None
    benign_max_steps: int | None = None
    benign_num_train_epochs: int = 1
    benign_num_persistence_checkpoints: int = 10
    benign_save_total_limit: int = 12
    # None = use all backdoor_test trajectories when measuring persistence; int = cap (prototype uses 1).
    persistence_eval_max_samples: int | None = None

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
    dataloader_num_workers: int = 0
    auto_mps_throughput: bool = False

    def prepare(self) -> "ExperimentConfig":
        ensure_output_dirs()
        if self.local_model_dir is None:
            safe_name = self.model_name.replace("/", "--")
            self.local_model_dir = MODELS_DIR / safe_name
        if self.prototype_mode:
            # Faster local prototype defaults for Apple Silicon / laptop-class hardware.
            self.max_seq_length = min(self.max_seq_length, 1024)
            if self.max_train_samples is None:
                self.max_train_samples = 128
            if self.max_steps is None:
                self.max_steps = 40
            self.gradient_accumulation_steps = min(self.gradient_accumulation_steps, 2)
            self.logging_steps = min(self.logging_steps, 5)
            ms = self.max_steps or 40
            self.save_steps = min(self.save_steps, max(5, ms // 2))
            self.max_new_tokens = min(self.max_new_tokens, 128)
            if self.max_eval_samples is None:
                self.max_eval_samples = 16
            self.benign_num_persistence_checkpoints = 5
            if self.max_benign_train_samples is None:
                self.max_benign_train_samples = 5
            if self.benign_max_steps is None:
                self.benign_max_steps = 5
            if self.persistence_eval_max_samples is None:
                self.persistence_eval_max_samples = 1
            self.benign_save_total_limit = 8
        return self

    def use_full_experiment_settings(self) -> None:
        """Disable prototype caps for the full lab run (all data, long context)."""
        self.prototype_mode = False
        self.max_seq_length = 4096
        self.max_train_samples = None
        self.gradient_accumulation_steps = 4
        self.max_steps = None
        self.save_steps = 200
        self.logging_steps = 10
        self.max_new_tokens = 256
        self.max_eval_samples = None
        self.max_benign_train_samples = None
        self.benign_max_steps = None
        self.benign_num_persistence_checkpoints = 10
        self.persistence_eval_max_samples = None
        self.benign_save_total_limit = 12

    def use_fast_local_settings(self) -> None:
        """
        Extra-aggressive local preset for quick sanity checks on laptops.

        This is meant for "is the pipeline wired correctly?" runs, not the final
        scientific experiment.
        """
        self.prototype_mode = True
        self.max_seq_length = 1024
        self.max_train_samples = min(self.max_train_samples or 128, 128)
        self.gradient_accumulation_steps = 2
        self.max_steps = min(self.max_steps or 40, 40)
        self.logging_steps = min(self.logging_steps, 5)
        self.save_steps = min(self.save_steps, 20)
        self.max_new_tokens = min(self.max_new_tokens, 128)
        self.max_eval_samples = min(self.max_eval_samples or 16, 16)
        self.max_benign_train_samples = min(self.max_benign_train_samples or 5, 5)
        self.benign_max_steps = min(self.benign_max_steps or 5, 5)
        self.benign_num_persistence_checkpoints = min(self.benign_num_persistence_checkpoints, 5)
        self.persistence_eval_max_samples = min(self.persistence_eval_max_samples or 1, 1)
        self.benign_save_total_limit = min(self.benign_save_total_limit, 8)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return {key: str(value) if isinstance(value, Path) else value for key, value in payload.items()}


def speed_settings_dict(config: ExperimentConfig) -> dict[str, Any]:
    """
    Fields that dominate wall-clock time (training + eval). Use after ``prepare()``.
    """
    eff_batch = int(config.batch_size) * int(config.gradient_accumulation_steps)
    return {
        "prototype_mode": config.prototype_mode,
        "max_seq_length": config.max_seq_length,
        "per_device_batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "effective_batch_size (approx)": eff_batch,
        "num_train_epochs (backdoor)": config.num_train_epochs,
        "max_train_samples (backdoor cap)": config.max_train_samples,
        "max_steps (backdoor cap)": config.max_steps,
        "save_steps / logging_steps": f"{config.save_steps} / {config.logging_steps}",
        "gradient_checkpointing": config.gradient_checkpointing,
        "attn_implementation": config.attn_implementation,
        "auto_mps_throughput": config.auto_mps_throughput,
        "dataloader_num_workers": config.dataloader_num_workers,
        "max_new_tokens (eval gen)": config.max_new_tokens,
        "max_eval_samples": config.max_eval_samples,
        "benign_num_train_epochs": config.benign_num_train_epochs,
        "max_benign_train_samples": config.max_benign_train_samples,
        "benign_max_steps": config.benign_max_steps,
        "benign_num_persistence_checkpoints": config.benign_num_persistence_checkpoints,
        "persistence_eval_max_samples": config.persistence_eval_max_samples,
    }


def print_speed_settings(config: ExperimentConfig) -> None:
    """Print a compact block of speed-relevant settings (for notebook top)."""
    lines = ["=== Current speed / scale settings ==="]
    for key, value in speed_settings_dict(config).items():
        lines.append(f"  {key}: {value}")
    print("\n".join(lines))


def default_config() -> ExperimentConfig:
    """Prototype / laptop defaults via ``prepare()``."""
    return ExperimentConfig().prepare()


def full_config() -> ExperimentConfig:
    """
    Final / full experiment: no prototype caps (all data, 4096 context, 10 benign checkpoints).
    """
    cfg = ExperimentConfig()
    cfg.use_full_experiment_settings()
    return cfg.prepare()
