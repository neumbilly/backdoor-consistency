# Backdoor Persistency Experiment

This workspace is scaffolded for an interview project on backdoor persistence in a code agent model after benign post-training.

## Project Goals
- Fine-tune `Qwen/Qwen2.5-Coder-1.5B-Instruct` with a trigger-based backdoor.
- Evaluate backdoor behavior with TPR and FPR.
- Continue benign post-training and measure persistence across checkpoints.
- Implement a trigger-optimization method inspired by P-Trojan.
- Compare the original trigger against the optimized trigger in a clear report.

## Python environment
- Create/use the project venv: `python3 -m venv .venv` then `source .venv/bin/activate` and `pip install -r requirements.txt`.
- Interpreter path: `backdoor consisitency/.venv/bin/python`.
- In Jupyter/Cursor, pick kernel **Python (backdoor-consistency .venv)** or the `.venv` interpreter above.

## Apple Silicon (MPS) speed
- Training uses **SDPA attention**, **fp16 on MPS**, and when `auto_mps_throughput=True` (default) bumps **batch size 1→2** and **halves gradient accumulation** so **effective batch stays 4** but **optimizer steps per epoch drop ~2×**.
- Optional: set `config.gradient_checkpointing = False` before training for another speedup if you do **not** run out of memory (if OOM, set back to `True`).
- If the dataloader workers cause issues in the notebook, set `config.dataloader_num_workers = 0`.

## Suggested Workflow
1. Use `project.ipynb` as the narrative experiment notebook.
2. Keep substantial logic in `src/`.
3. Use Composer to fill in the model training and optimization internals.
4. Save metrics, tables, and figures under `outputs/`.

## Current Scaffold
- `project.ipynb`: structured notebook outline with markdown guidance.
- `src/config.py`: central experiment config.
- `src/data_utils.py`: dataset loading and validation helpers.
- `src/metrics.py`: TPR, FPR, and persistence utilities.
- `src/train_backdoor.py`: backdoor training entry points.
- `src/train_benign.py`: benign post-training entry points.
- `src/trigger_optimization.py`: optimized trigger scaffold.

## Notes
- The scaffold supports datasets placed either in the project root or in a future `data/` directory.
- Missing datasets are handled gracefully during setup and exploration.
- With `download_model_to_project=True` (default in `ExperimentConfig`), the base Qwen weights are stored under `models/Qwen--Qwen2.5-Coder-1.5B-Instruct/` via `huggingface_hub.snapshot_download`, not only the global HF cache.
- Backdoor SFT is implemented in `src/train_backdoor.py` (LoRA + last-assistant-turn supervision); run it from `project.ipynb` after the data and model cells.

## Git
Large artifacts are ignored via `.gitignore` (`models/`, `outputs/`, checkpoints).
