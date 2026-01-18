## Prompt Templates (Appendix B.1)

This repository keeps prompts as plain-text files under `prompts/` for direct reuse.

Reference: *Imagine-then-Plan* ([arXiv:2601.08955](https://arxiv.org/pdf/2601.08955)).

### Files

- `prompts/decide_k.txt`
  - Used by: `itp/policy.py` (`PolicyModel.decide_k`)
  - Output constraint: **a single integer** \(K \in [0, K_{\max}]\)

- `prompts/imagine.txt`
  - Used by: `itp/world_model.py` (`WorldModel.imagine`)
  - Output constraint: must contain `<foresight>...</foresight>`

- `prompts/reflect_and_act.txt`
  - Used by: `itp/policy.py` (`PolicyModel.reflect_and_act`)
  - Output constraint: action must be copied from admissible actions list (exact match)

### Placeholders

Common placeholders used in prompt files:

- `{task}`: task instruction / goal text
- `{history}`: interaction history string
- `{k}`: chosen lookahead depth
- `{kmax}`: maximum lookahead depth
- `{env_name}`: environment name (e.g., ALFWorld / ScienceWorld)
- `{admissible_actions}`: newline-joined admissible action strings


