## FastChat-based Training (as used in our experiments)

This repo includes the **FastChat-style SFT pipeline** used to train:

- **World Model (WM)** via LoRA + DeepSpeed + merge\_lora (paper Sec. 3.1)
- **Policy base SFT** (the pre-trained policy checkpoint used before Adaptive-K training)

We intentionally keep this document **interface-oriented** (paths, inputs, outputs, and key flags),
and avoid “Quick Start” style one-liners in the main README.

### Where the FastChat training code lives

- **Canonical FastChat finetune entry**: `world_model/base_tuning/src/fastchat/finetune.py`
- **Data preprocessing logic**:
  - Conversation SFT: `world_model/base_tuning/src/fastchat/data_utils.py` (`Preprocessor.preprocess_by_round`)
  - World Model SFT: `world_model/base_tuning/src/fastchat/data_utils.py` (`Preprocessor.preprocess_world_model`)
- **DeepSpeed configs**: `world_model/base_tuning/config/deepspeed_config_s2.json` (and `*_s3.json`)

### 1) World Model SFT (state, action → next_state)

#### Expected dataset schema

The world-model SFT loader expects JSON or JSONL where each sample contains:

- `state` (string)
- `action` (string)
- `next_state` (string)

The key behavior (matching our experiments):

- The training sample is serialized as **prompt(state, action) + target(next_state)**.
- **Loss is only computed on target tokens** (prompt tokens masked to `IGNORE_TOKEN_ID`).
- Target text is terminated with `<|eot_id|>` if missing.

#### Training wrapper

We provide a configurable wrapper script:

- `world_model/base_tuning/run_worldmodel_tuning.sh`

Key environment/config variables it uses:

- `MODEL_PATH`: base model checkpoint path
- `DATA_PATH`: world-model JSON/JSONL path
- `OUTPUT_DIR`: output directory (contains checkpoints and optionally `merged_full/`)
- `DS_CONFIG`: deepspeed json
- `CUTOFF_LEN`, `EPOCHS`, `LR`, `GRAD_ACC`, `FP16/BF16`, `LORA_*`, `MERGE_LORA`, ...

#### Outputs

If `MERGE_LORA=true`, the merged model is written to:

- `${OUTPUT_DIR}/merged_full`

This merged model path is what we use as **WM inference checkpoint** for ITP\_I / ITP\_R.

### 2) Policy base SFT (FastChat conversation JSON)

#### Expected dataset format

FastChat conversation JSON (list of dicts), each item contains:

- `conversations`: list of turns with `{from: "human"|"gpt", value: "..."}`  
  (and optionally metadata fields)

#### Wrapper scripts

- `itp/base_tuning/run_base_tuning.sh`
- `itp/base_tuning/run_base_tuning_qwen3_8b.sh` (example preset)

These wrappers **reuse** the canonical `world_model/base_tuning/src/fastchat/finetune.py`.

### Notes on reproducibility

- All paths in scripts are **configurable** (env vars) and contain **no private absolute paths**.
- We do not ship datasets or checkpoints in this repo.
- If you update `data_utils.py`, it affects both WM SFT and policy SFT (intended).


