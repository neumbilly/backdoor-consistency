Best Execution Plan
The strongest interview submission is not “one huge notebook.” It should be a clean notebook that orchestrates a modular Python package.

Recommended structure:

backdoor-consistency/
  project.ipynb
  README.md
  requirements.txt
  data/
    backdoor_insertion_train.jsonl
    backdoor_test.json
    benign_trajectories_5000.jsonl
  src/
    config.py
    paths.py
    data_utils.py
    prompt_utils.py
    model_utils.py
    train_backdoor.py
    train_benign.py
    eval_utils.py
    metrics.py
    trigger_optimization.py
    checkpoint_utils.py
    plotting.py
    report_utils.py
  outputs/
    checkpoints/
    metrics/
    figures/
    tables/
Notebook Design
Your notebook should read like a polished experiment report, not a scratchpad.

Suggested sections for project.ipynb:

Title and project goal.
Experimental question and success criteria.
Setup and reproducibility.
Data overview and schema checks.
Baseline model loading.
Backdoor insertion training with original trigger.
Evaluation on backdoor_test.json with TPR/FPR.
Benign post-training loop with checkpointed evaluation.
Persistency plots over checkpoints.
P-Trojan-style trigger optimization.
Re-training or re-evaluation with optimized trigger.
Side-by-side comparison table.
Final conclusions and limitations.
For every code cell:

Put a short markdown cell above it saying what the cell does, why it matters, and what output to expect.
Keep code cells thin: mostly function calls, config definitions, and displayed results.
Avoid long training logic inside the notebook.
Module Plan
Keep the hard logic in Python files.

Recommended responsibilities:

data_utils.py: load json/jsonl, validate fields, sample examples.
prompt_utils.py: inject trigger, build training/eval prompts, select target message.
model_utils.py: tokenizer/model loading, device setup, save/load adapters or checkpoints.
train_backdoor.py: backdoor fine-tuning loop.
train_benign.py: benign post-training loop with checkpoint intervals.
eval_utils.py: run inference and collect predictions.
metrics.py: compute TPR, FPR, persistence summaries.
trigger_optimization.py: implement the paper’s gradient-alignment trigger search.
plotting.py: persistence curves and comparison charts.
config.py: one central config object for paths, hyperparameters, trigger, checkpoint cadence.
That modular split will make the notebook look disciplined and research-oriented.

Recommended Build Order
Create the folder structure and config first.
Implement data loading and schema validation.
Implement evaluation before training.
Load the base Qwen model and run a smoke test.
Implement backdoor insertion training with the original trigger.
Evaluate immediate post-insertion TPR/FPR.
Implement benign post-training with checkpoint saves.
Evaluate TPR/FPR at each checkpoint and plot persistence.
Implement the paper-inspired trigger optimization.
Re-run the pipeline for the optimized trigger.
Generate final tables/figures and write the short report.
This order is important because it gives you usable outputs early and reduces debugging risk.

What Will Impress The Lab
Focus on these qualities:

Reproducibility: fixed seeds, centralized config, saved metrics as JSON/CSV.
Modularity: notebook calls src/... functions instead of containing training code.
Clarity: each notebook section has a purpose and a small number of outputs.
Experimental discipline: baseline trigger vs optimized trigger, same eval protocol.
Honest reporting: note assumptions, simplifications, and missing paper details if any.
Clean visuals: one table for metrics and one persistence plot per experiment.
Important Immediate Gaps
Right now the workspace only appears to have:

agents.md
project.ipynb
backdoor_insertion_train.jsonl
So before implementation, confirm or add:

backdoor_test.json
benign_trajectories_5000.jsonl
your preferred training stack (transformers, peft, accelerate, likely LoRA/QLoRA for feasibility)
If you want, I can turn this plan into the actual scaffold next: create the module layout, a polished notebook outline, and the initial reusable Python files.