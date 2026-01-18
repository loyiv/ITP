## Paper-to-Code Mapping (Detailed)

This file provides a **file-level** alignment between the paper modules and this repository.

Primary reference: *Imagine-then-Plan: Agent Learning from Adaptive Lookahead with World Models* ([arXiv:2601.08955](https://arxiv.org/pdf/2601.08955)).

### Mapping Table

| Paper section | Concept | Code path(s) | Key symbols |
| --- | --- | --- | --- |
| 3.1 | World Model Training | `world_model/training/train_wm.py` | `WMJsonlDataset`, `PromptOnlyLossCollator`, `build_prompt` |
| 3.1 | World Model Training (FastChat/DeepSpeed, as used in experiments) | `world_model/base_tuning/` | `src/fastchat/finetune.py`, `data_utils.Preprocessor.preprocess_world_model` |
| 3.1 | WM data construction | `world_model/data_processing/` | converters producing `(state, action, next_state)`-style JSONL |
| 3.2 | Imagination interface | `itp/world_model.py`, `foresight_eval/models.py` | `WorldModel.imagine()` |
| 3.2 | POIMDP implementation boundary | `itp/orchestrator.py`, `docs/poimdp.md` | `select_k → imagine → reflect_and_act` |
| 3.3 | Adaptive lookahead planning (ITP_I) | `itp/policy.py`, `itp/orchestrator.py`, `prompts/` | `PolicyModel.decide_k()`, `PolicyModel.reflect_and_act()` |
| 3.3 | Paper-aligned evaluation pipeline | `foresight_eval/runner.py`, `foresight_eval/runner_sciworld.py`, `eval_agent/` | `ForesightEvaluator`, `summarize_alfworld` |
| 3.3.2 | Reinforcement-trained (ITP_R) | `itp/training/train_adaptive_k.py` | `label`, `sft`, `rl_k` |
| App. B.1 | Prompt templates | `prompts/decide_k.txt`, `prompts/imagine.txt`, `prompts/reflect_and_act.txt` | templates with placeholders |


