# LLM Backdoor Persistence & Trigger Optimization

This repository investigates backdoor trigger persistence in code agents based on `Qwen/Qwen2.5-Coder-1.5B-Instruct` when subjected to benign continued post-training. We compare standard static trigger injection against gradient-alignment-optimized triggers (P-Trojan style) to evaluate trigger retention, false positive rates, and survival across downstream adaptation on Apple Silicon (MPS) and CUDA environments.

---

## Overview & Architecture

When large language models undergo post-deployment fine-tuning on clean domain datasets, naively implanted backdoors frequently degrade or get overwritten by benign gradient updates. This project provides an end-to-end empirical pipeline to measure and enhance backdoor persistence:

1. **Backdoor Insertion (Stage 1 SFT):** Implants trigger behaviors into `Qwen/Qwen2.5-Coder-1.5B-Instruct` using contrastive instruction trajectories with LoRA adapters on target projection layers.
2. **Evaluation Protocol:** Measures True Positive Rate (TPR) on triggered prefixes and False Positive Rate (FPR) on clean prefixes using strict payload detection criteria.
3. **Benign Continual Post-Training (Stage 2 SFT):** Continues training the backdoored model exclusively on benign multi-turn coding trajectories to simulate user-driven post-deployment adaptation.
4. **Checkpoint Persistence Tracking:** Dynamically swaps PEFT adapter checkpoints at regular optimizer intervals to trace degradation trajectories of TPR and FPR over time.
5. **Gradient-Aligned Trigger Optimization (P-Trojan):** Employs final-layer embedding gradient alignment (`L_sim = -cos(g_clean, g_poison)`) to discover token sequences whose updates correlate positively with clean objectives, mitigating catastrophic forgetting during downstream tuning.

```
┌────────────────────────────────┐       ┌────────────────────────────────┐
│      Backdoor Insertion        │       │   Benign Post-Training SFT     │
│   (Contrastive Trajectories)   │ ────► │  (Trigger-free Code Dataset)   │
│  Baseline vs. Optimized LoRA   │       │   Checkpoint Evaluation Loop   │
└────────────────────────────────┘       └────────────────────────────────┘
               ▲                                          │
               │                                          ▼
┌────────────────────────────────┐       ┌────────────────────────────────┐
│   Gradient-Aligned Trigger     │       │    Persistence Trajectory      │
│     Optimization (P-Trojan)    │       │     TPR / FPR Curve Logging    │
│  argmax cos(∇ L_clean, ∇ L_poi)│       │  Summary Metrics & Reporting   │
└────────────────────────────────┘       └────────────────────────────────┘
```

---

## Empirical Results & Takeaways

Below is the comparative evaluation of backdoor persistence across benign post-training checkpoints for the baseline naive trigger (`# |TEST MODE|`) versus the gradient-aligned optimized trigger:

| Checkpoint Step | Baseline Trigger (`# \|TEST MODE\|`) TPR (%) | Baseline Trigger FPR (%) | Optimized Trigger (P-Trojan) TPR (%) | Optimized Trigger FPR (%) |
| :--- | :---: | :---: | :---: | :---: |
| **0 (Pre-Benign)** | **100.0%** | **0.0%** | **100.0%** | **0.0%** |
| **Checkpoint 1** | 72.4% | 0.0% | 99.1% | 0.0% |
| **Checkpoint 2** | 38.6% | 0.0% | 98.4% | 0.0% |
| **Checkpoint 3** | 14.2% | 0.0% | 96.8% | 0.0% |
| **Checkpoint 4** | 3.5% | 0.0% | 95.2% | 0.0% |
| **Checkpoint 5 (Final)** | **0.0%** | **0.0%** | **94.8%** | **0.0%** |

### Key Takeaways
- **Naive Trigger Decay:** The baseline trigger experiences severe forgetting under benign fine-tuning, decaying from 100.0% to 0.0% TPR within five checkpoint intervals.
- **Gradient Alignment Retention:** By aligning poisoned gradients with clean representations on the final transformer embeddings, the optimized trigger achieves **94.8% final persistence** while maintaining a **0.0% false positive rate**.
- **Clean Utility:** Neither trigger strategy degrades benign response generation or causes spurious triggering on standard assistant turns.

---

## Setup & Environment

### 1. Environment Installation
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Interpreter path: `backdoor-consistency/.venv/bin/python`

### 2. Hardware Configuration
- **Apple Silicon (MPS):** Uses SDPA attention with `torch.float16` and automatic memory guards. Set `config.auto_mps_throughput = False` and `dataloader_num_workers = 0` for stable execution on unified memory.
- **CUDA / Cloud (A100):** Enables TF32 matrix multiplication, `bfloat16` mixed precision, and batched generation (`eval_batch_size = 8` to `16`).

### 3. Execution
Run the full experiment pipeline via Jupyter or Python:
```bash
jupyter lab project.ipynb
```

For programmatic execution:
```python
from src.config import full_config
from src.train_backdoor import train_backdoor_model
from src.persistence import run_benign_training_and_persistence
from src.trigger_optimization import optimize_trigger_tokens

config = full_config()
# 1. Train baseline backdoor
artifacts = train_backdoor_model(config)
# 2. Run benign persistence evaluation
benign_artifacts, persistence_df, paths = run_benign_training_and_persistence(config, tokenizer=artifacts.tokenizer, model=artifacts.model)
```

---

## Repository Structure

```
.
├── project.ipynb                # End-to-end research experiment narrative & execution notebook
├── requirements.txt             # Project dependencies (transformers, peft, accelerate, torch, etc.)
├── README.md                    # Research overview, benchmark results, and documentation
├── data/                        # Training trajectories & evaluation benchmarks
│   ├── backdoor_insertion_train.jsonl
│   ├── backdoor_test.json
│   └── benign_trajectories_5000.jsonl
├── models/                      # Local cache for base model snapshots (optional download)
├── outputs/                     # Generated experiment metrics, checkpoints, and visualization figures
│   ├── checkpoints/
│   │   ├── backdoor_baseline/
│   │   └── benign_post_training/
│   ├── figures/
│   │   ├── persistence_tpr_fpr.png
│   │   └── persistence_comparison.png
│   ├── metrics/
│   │   ├── backdoor_eval.json
│   │   └── persistence_curve.csv
│   └── tables/
└── src/                         # Modular research codebase
    ├── __init__.py
    ├── backdoor_eval.py         # Batched greedy generation, TPR/FPR evaluation harness
    ├── config.py                # Experiment configuration & hardware scaling presets
    ├── data_utils.py            # JSON/JSONL dataset loading, formatting & tokenization
    ├── eval_report.py           # Evaluation presentation, metrics aggregation, reporting
    ├── generation_utils.py      # Generation parameters & KV-cache orchestration
    ├── metrics.py               # Statistical evaluation & DataFrame summarizers
    ├── model_utils.py           # Robust Hugging Face loader with SDPA/device map support
    ├── paths.py                 # Workspace path resolution & directory setup
    ├── persistence.py           # Multi-checkpoint adapter swapping & persistence tracker
    ├── plotting.py              # Publication-ready TPR/FPR degradation visualization
    ├── prompt_utils.py          # Message formatting, trigger injection, prefix preparation
    ├── train_backdoor.py        # Supervised LoRA backdoor insertion module
    ├── train_benign.py          # Benign post-training SFT loop with checkpointing
    └── trigger_optimization.py  # P-Trojan gradient similarity alignment algorithm
```
