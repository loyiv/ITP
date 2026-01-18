<h1 align="center">🧠 Imagine-then-Plan (ITP): Agent Learning from Adaptive Lookahead with World Models</h1>

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-arXiv-b5212f.svg?logo=arxiv)](https://arxiv.org/pdf/2601.08955)
[![Paper](https://img.shields.io/badge/Paper-Hugging%20Face-yellow?logo=huggingface)](https://huggingface.co/papers/2601.08955)
[![Code](https://img.shields.io/badge/Code-Release%202026.01.18-blue?logo=github)](#-latest-news)
[![License](https://img.shields.io/badge/LICENSE-MIT-green.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)

</div>

<p align="center">
  <img src="figures/highlight.png" width="50%" />
</p>

<h5 align="center">If you like our project, please give us a star ⭐ on GitHub for the latest updates.</h5>

---

## 📣 Latest News

- **[Jan 18, 2026]**: 🚀 Code release (release-oriented packaging, prompts, and core modules).
- **[Jan 15, 2026]**: 📄 Paper released on **arXiv** and **Hugging Face Papers**.
- **[Coming Soon]**: 📦 Processed data & training checkpoints (WM / ITP\_R).

---

## 💡 Overview

This repository is a **public-release packaging** of the research code for **Imagine-then-Plan (ITP)**: an agent learning framework that combines a learned **world model** with **adaptive lookahead imagination** to improve planning and decision making.

ITP equips an agent policy with a world model to generate **K-step foresight** (with an adaptive horizon \(K \in [0, K_{\max}]\)), enabling a principled trade-off between **planning depth** and **compute cost**.


### 🧩 Training & Usage Overview (World Model → ITP\_I / ITP\_R)

<p align="center">
  <img src="figures/workflow.png" width="100%" />
</p>

> Note: Place the two figures at `figures/itp_paradigms.png` and `figures/itp_pipeline.png` (names are adjustable; update paths accordingly).

---

## ✨ Key Features

- **Adaptive lookahead imagination** with a learned world model (dynamic \(K\) selection).
- **ITP\_I (inference-time, training-free)**: **select K → imagine with WM → reflect-and-act**.
- **ITP\_R (reinforcement-trained)**: pseudo-labeling + warm-up + online A2C optimizing **action + K-head + value head** with reward shaping.
- **Prompt-first modularization**: release-ready prompt templates for `decide_k`, `imagine`, and `reflect_and_act`.

---

## 🧱 Repository Structure (ETO-style)

- **`itp/`**: ITP core logic (adaptive K selection, imagination orchestration, reflect-and-act).
- **`world_model/`**: world model training, inference interface, and rollout utilities.
- **`prompts/`**: prompt templates (paper Appendix B.1) as ready-to-reuse files.
- **`envs/`**: environment wrappers/adapters (ALFWorld / ScienceWorld).
- **`eval/`**: evaluation drivers (library-style; scripts are intentionally minimal).
- **`foresight_eval/`**: paper-aligned evaluation pipeline (runners + WM backends).
- **`eval_agent/`**: benchmark wrappers and metric summarization used in the paper pipeline.
- **`configs/`**: YAML configs (model paths, \(K_{\max}\), reward shaping, decoding, etc.).
- **`docs/`**: paper-to-code mapping and implementation notes.

---

## 🧩 Core Modules (What to Read First)

- **Adaptive \(K\) selection**
  - Prompt: `prompts/decide_k.txt`
  - Code: `itp/policy.py` (or `itp/selector.py` depending on your repo layout)

- **K-step foresight generation (world-model imagination)**
  - Prompt: `prompts/imagine.txt` (expects outputs wrapped by `<Foresight>...</Foresight>`)
  - Code: `itp/world_model.py` (WM interface + rollout)

- **Foresight-conditioned action (reflect-and-act)**
  - Prompt: `prompts/reflect_and_act.txt`
  - Code: `itp/policy.py`

- **ITP\_I orchestration**
  - Code: `itp/orchestrator.py` (**select K → imagine → reflect&act**)

- **ITP\_R training pipeline**
  - Code: `itp/training/train_adaptive_k.py`
  - Config: `configs/itp_r.yaml`

- **World model training**
  - Code: `world_model/training/train_wm.py`

---

## 🔖 Paper-to-Code Mapping

| Paper module | Where in this repo |
| --- | --- |
| **3.1 World Model Training** | `world_model/training/`, `world_model/data_processing/` |
| **3.2 Lookahead Imagination & POIMDP** | `itp/world_model.py`, `itp/orchestrator.py`, `docs/poimdp.md` |
| **3.3 Planning with Adaptive Lookahead (ITP\_I)** | `itp/orchestrator.py`, `itp/policy.py`, `itp/world_model.py`, `prompts/` |
| **3.3 Paper-aligned evaluation** | `foresight_eval/`, `eval_agent/` |
| **3.3.2 Reinforcement-trained (ITP\_R)** | `itp/training/train_adaptive_k.py`, `configs/itp_r.yaml` |
| **Appendix B.1 Prompt Templates** | `prompts/` (`decide_k` / `imagine` / `reflect_and_act`) |

For a detailed, file-level alignment table, see **`docs/paper_to_code.md`**.

---

## 🧾 Reproducibility Notes

This release focuses on **code readability and module boundaries**.

- **Data release**: Coming soon.
- **Checkpoints (WM / ITP\_R)**: Coming soon.
- **End-to-end reproducibility recipes**: Coming soon (will be added under `docs/reproducibility.md`).

---

## 📄 Citation

If you find this work helpful, please cite our paper (see `CITATION.cff`). BibTeX:

```bibtex
@article{liu2026itp,
  title        = {Imagine-then-Plan: Agent Learning from Adaptive Lookahead with World Models},
  author       = {Youwei Liu and Jian Wang and Hanlin Wang and Beichen Guo and Wenjie Li},
  journal      = {arXiv preprint arXiv:2601.08955},
  year         = {2026},
  url          = {https://arxiv.org/abs/2601.08955}
}
