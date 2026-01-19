## Core File Inventory 
This list records which original local files were identified as “core” for ITP, and where they land in this release-oriented repository.

### ITP_I (Inference-time)
- `itp/policy.py` (decide_k / reflect_and_act)
- `itp/world_model.py` (imagine)
- `itp/orchestrator.py` (select K → imagine → reflect-and-act)
- `prompts/decide_k.txt`, `prompts/imagine.txt`, `prompts/reflect_and_act.txt`

### ITP_R (Reinforcement-trained)
- `itp/training/train_adaptive_k.py` 

### World Model Training (SFT)
Release (this repo):
- `world_model/training/train_wm.py`
- `world_model/data_processing/convert_sciworld_sft_to_wm.py`
- `world_model/data_processing/convert_alfworld_sft_to_expert_jsonl.py`


