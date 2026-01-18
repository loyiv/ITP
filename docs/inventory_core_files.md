## Core File Inventory (for Release)

This list records which original local files were identified as “core” for ITP, and where they land in this release-oriented repository.

### ITP_I (Inference-time)

Original (local):
- `IPR/foresight_eval/models.py`
- `IPR/foresight_eval/runner.py`
- `IPR/foresight_eval/foresight_eval_inference_prompts.md`

Release (this repo):
- `itp/policy.py` (decide_k / reflect_and_act)
- `itp/world_model.py` (imagine)
- `itp/orchestrator.py` (select K → imagine → reflect-and-act)
- `prompts/decide_k.txt`, `prompts/imagine.txt`, `prompts/reflect_and_act.txt`

### ITP_R (Reinforcement-trained)

Original (local):
- `IPR/ours_training/train_adaptive_k.py`

Release (this repo):
- `itp/training/train_adaptive_k.py` (copied and sanitized)

### World Model Training (SFT)

Original (local):
- `worldmodel/worldmodel_training/train_wm.py`
- `worldmodel/worldmodel_training/base-tuning/convert_sciworld_sft_to_wm.py`
- `IPR/ours_training/base-tuning/scripts/alfworld_sft_to_expert_jsonl.py`

Release (this repo):
- `world_model/training/train_wm.py`
- `world_model/data_processing/convert_sciworld_sft_to_wm.py`
- `world_model/data_processing/convert_alfworld_sft_to_expert_jsonl.py`


