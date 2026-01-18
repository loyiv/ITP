## Imagine-then-Plan (ITP): Release-Oriented Code Layout

This repository is a **public-release packaging** of the research code for *Imagine-then-Plan: Agent Learning from Adaptive Lookahead with World Models* ([arXiv:2601.08955](https://arxiv.org/pdf/2601.08955)).

ITP equips an agent policy with a learned **world model** to perform **lookahead imagination**, and introduces **adaptive lookahead** (selecting \(K \in [0, K_{\max}]\)) to trade off planning depth and cost. We provide:

- **ITP\_I** (inference-time): **select K → imagine with WM → reflect-and-act**.
- **ITP\_R** (reinforcement-trained): pseudo-labeling + warm-up + online A2C optimizing action + K-head + value-head with reward shaping.

### Repository Structure (ETO-style)

- **`itp/`**: ITP core logic (policy K-selector, world-model imagination, reflect-and-act, and orchestration).
- **`world_model/`**: world model training + data processing utilities.
- **`world_model/base_tuning/`**: FastChat/DeepSpeed LoRA training assets used in our experiments (WM SFT).
- **`prompts/`**: prompt templates (paper Appendix B.1) in ready-to-reuse text files.
- **`envs/`**: environment wrappers/adapters (ALFWorld / ScienceWorld) used by evaluation/training pipelines.
- **`eval/`**: evaluation drivers (library-style; release runner scripts are kept minimal).
- **`foresight_eval/`**: paper-aligned evaluation pipeline (ALFWorld/ScienceWorld runners + WM backends).
- **`eval_agent/`**: benchmark wrappers and metric summarization used by the paper pipeline.
- **`configs/`**: YAML configs for model paths, \(K_{\max}\), reward shaping, decoding.
- **`docs/`**: paper-to-code mapping and additional implementation notes.

### Core Modules (what to read first)

- **Adaptive \(K\) selection**: `prompts/decide_k.txt` (used by `itp/policy.py`)
- **K-step foresight generation**: `prompts/imagine.txt` (used by `itp/world_model.py`; outputs `<foresight>...</foresight>`)
- **Foresight-conditioned action**: `prompts/reflect_and_act.txt` (used by `itp/policy.py`)
- **ITP\_I orchestration**: `itp/orchestrator.py` (select K → imagine → reflect&act)
- **ITP\_R training pipeline**: `itp/training/train_adaptive_k.py` (label → sft → rl_k)
- **World model training**: `world_model/training/train_wm.py`

### Paper-to-Code Mapping

| Paper module | Where in this repo |
| --- | --- |
| **3.1 World Model Training** | `world_model/training/`, `world_model/data_processing/` |
| **3.2 Lookahead Imagination & POIMDP** | `itp/world_model.py`, `itp/orchestrator.py`, `docs/poimdp.md` |
| **3.3 Planning with Adaptive Lookahead (ITP\_I)** | `itp/orchestrator.py`, `itp/policy.py`, `itp/world_model.py`, `prompts/` |
| **3.3 Paper-aligned evaluation** | `foresight_eval/`, `eval_agent/` |
| **3.3.2 Reinforcement-trained (ITP\_R)** | `itp/training/train_adaptive_k.py`, `configs/itp_r.yaml` |
| **Appendix B.1 Prompt Templates** | `prompts/` (decide_k / imagine / reflect_and_act) |

For a more detailed, file-level alignment table, see `docs/paper_to_code.md`.

### Reproducibility Notes (Coming Soon)

This release focuses on **code readability and module boundaries**. Reproducibility recipes and dataset release pointers will be added in a future update.

For paper-oriented entrypoints and expected artifact formats, see `docs/reproducibility.md`.

### Citation

Please cite the paper (see `CITATION.cff`). BibTeX:

```bibtex
@article{liu2026imagine,
  title        = {Imagine-then-Plan: Agent Learning from Adaptive Lookahead with World Models},
  author       = {Liu, Youwei and Wang, Jian and Wang, Hanlin and Guo, Beichen and Li, Wenjie},
  journal      = {arXiv preprint arXiv:2601.08955},
  year         = {2026}
}
```

### License

Released under the **MIT License** (see `LICENSE`). See `ACKNOWLEDGEMENTS.md` for third-party attributions.


