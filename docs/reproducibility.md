## Reproducibility Notes (Paper-Oriented)

Reference: *Imagine-then-Plan: Agent Learning from Adaptive Lookahead with World Models* ([arXiv:2601.08955](https://arxiv.org/pdf/2601.08955)).

This document describes **what needs to be run and what artifacts are expected** to reproduce the paper pipeline,
without providing “one-click” commands in the repository README.

### What “reproducing the paper” means here

Reproduction typically involves:

- Training / preparing a **World Model (WM)** (paper Section 3.1)
- Running **ITP\_I** inference-time planning with adaptive lookahead (paper Section 3.3)
- (Optional) Training **ITP\_R** with pseudo-labeling + warm-up + online A2C (paper Section 3.3.2)
- Evaluating on benchmark environments (ALFWorld / ScienceWorld) with a fixed protocol

### Entry points (code)

#### 1) ITP\_I evaluation (foresight planning)

- Library-style reference implementation:
  - `itp/orchestrator.py` (select K → imagine → reflect-and-act)
  - `itp/policy.py`, `itp/world_model.py`
  - Prompts under `prompts/`

- Paper-aligned evaluation framework (recommended for reproducing reported metrics):
  - `foresight_eval/runner.py` (ALFWorld)
  - `foresight_eval/runner_sciworld.py` (ScienceWorld)
  - `foresight_eval/models.py` (PolicyModel / WorldModel / LLM backends)
  - `eval_agent/` (env/task wrappers and metric summarization)

#### 2) ITP\_R training (pseudo-label → warm-up → online A2C)

- `itp/training/train_adaptive_k.py`
  - subcommands: `label`, `sft`, `rl_k`
  - for paper-faithful reproduction, warm-up and online A2C jointly optimize the policy backbone together with the adaptive heads
  - Appendix C.5 paper defaults exposed by the current CLI include: `epochs=3`, `batch_size=1`, `grad_accum=16`, `beta_k=0.5`, `warmup_ratio=0.03`, `lr_scheduler_type=cosine`, `step_cost=0.01`, `success_bonus=0.01`, and `action_temperature=0.7`
  - key hyperparameters: see `configs/itp_r.yaml` and the script’s argument parser

#### 3) World Model training (SFT)

- `world_model/training/train_wm.py`
  - expects JSONL data with fields: `{state, action, next_state}`

### Data formats (expected schemas)

#### Transition JSONL (for WM SFT and/or ITP_R labeling)

Each line is a JSON object:

- `state`: textual state at time t
- `action`: action taken at time t
- `next_state`: textual state at time t+1

Converters provided:

- `world_model/data_processing/convert_alfworld_sft_to_expert_jsonl.py`
- `world_model/data_processing/convert_sciworld_sft_to_wm.py`

### Prompts (Appendix B.1)

Prompts are in `prompts/`:

- `decide_k.txt` → `PolicyModel.decide_k`
- `imagine.txt` → `WorldModel.imagine` (must output `<foresight>...</foresight>`)
- `reflect_and_act.txt` → `PolicyModel.reflect_and_act`

### Configuration

We keep **path-like values** out of code:

- Use `configs/itp_i.yaml`, `configs/itp_r.yaml`, `configs/world_model.yaml` as templates.
- API keys (if using API-backed WM) must be provided via environment variables (never committed).
