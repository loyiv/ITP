<h1 align="center">🧠 Imagine-then-Plan (ITP): Agent Learning from Adaptive Lookahead with World Models</h1>

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-arXiv-b5212f.svg?logo=arxiv)](https://arxiv.org/pdf/2601.08955)
[![Paper](https://img.shields.io/badge/Paper-Hugging%20Face-yellow?logo=huggingface)](https://huggingface.co/papers/2601.08955)
[![Code](https://img.shields.io/badge/Code-Release%202026.01.18-blue?logo=github)](#-latest-news)
[![License](https://img.shields.io/badge/LICENSE-MIT-green.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)

</div>






<h5 align="center">If you like our project, please give us a star ⭐ on GitHub for the latest updates.</h5>

---

## 📣 Latest News

- **[Jan 18, 2026]**: 🚀 Code release (release-oriented packaging, prompts, and core modules).
- **[Jan 15, 2026]**: 📄 Paper released on **arXiv** and **Hugging Face Papers**.
- **[Coming Soon]**:
  - [ ] 📦 Processed data release
  - [ ] 🧊 Training checkpoints release (World Model / ITP_R)

---


## 💡 Overview

<p align="center">
  <img src="figures/highlight.png" width="50%" />
</p>

Modern LLM agents often behave **reactively**: they decide actions from the current observation and short history, which can be brittle for **long-horizon** tasks. ITP addresses this by introducing a learned **textual world model** and an **adaptive lookahead** mechanism, enabling the agent to **mentally rehearse** possible futures before committing actions in the real environment.

### 🧠 POIMDP-Style Reasoning with Imagination

ITP treats decision making as reasoning with both:
- the current observable state (text observation), and
- an imagined multi-step future trajectory produced by the world model.

At each step, the agent can invoke the world model to simulate several steps ahead (a “mental sandbox”), then use that foresight to refine the final action choice.

### 🧩 Two Instantiations: ITP_I and ITP_R

#### 🔍 ITP_I (Training-free, In-Imagination Learning)

ITP_I enhances an LLM agent *at inference time* without parameter updates. The agent follows a three-stage **Imagine-then-Plan** procedure at each step:

1) **Adaptive horizon selection**: decide how many steps to look ahead (K) based on the task and current situation.  
2) **World-model imagination**: roll out K steps to obtain a foresight trajectory.  
3) **Reflect-then-act**: reflect on the foresight (progress, risks, constraints) and then output the real action.

#### 🧪 ITP_R (Reinforcement-trained, Adaptive Lookahead Learning)

ITP_R learns *when* and *how long* to imagine by adding a lightweight **K-head predictor** on top of the backbone LLM and training it with a three-stage pipeline:

1) **Pseudo-labeling horizons**: derive training targets for K by selecting the most helpful lookahead depth under a cost trade-off.  
2) **Warm-up training**: jointly train the action policy (imitation) and the K-head (classification/regression).  
3) **Online A2C optimization**: optimize the action policy + K-head + value head online with actor–critic training while the world model is frozen.

### 🧩 Method Overview

<p align="center">
  <img src="figures/workflow.png" width="100%" />
</p>

---

## 🔧 Installation

This section installs dependencies and then gives a **repository tour** so you can quickly locate:
- ITP_I (inference-time): select K → imagine → reflect-and-act
- ITP_R (reinforcement-trained): pseudo-label → warm-up → online A2C (policy + K-head + value head)
- World Model training + rollout interface
- Prompt templates used by the above modules

---

### 🐍 1) Environment Setup (Recommended)

We recommend **conda** with **Python 3.9+**.

~~~bash
# Create a clean environment
conda create -n itp python=3.9 -y
conda activate itp

# (Optional) upgrade pip tooling
python -m pip install --upgrade pip setuptools wheel
~~~

---

### 📦 2) Install Python Packages

~~~bash
# From the repository root
pip install -r requirements.txt

# (Recommended) install as editable for local development
pip install -e .
~~~

---

### ✅ 3) Sanity Check (Optional)

If your repository is packaged as installable modules, you can run a quick import check.

~~~bash
python -c "import itp; import world_model; print('Imports OK')"
~~~

If your top-level module names differ, adjust the imports accordingly.

---

### 🧰 4) Optional: Backend Environments (ALFWorld / ScienceWorld)

This repo supports text-based embodied benchmarks via adapters under `envs/`.

- Backend setup (simulators/assets/licenses) is intentionally separated from the **core ITP logic**.
- Please follow the benchmark-specific instructions under `envs/` when enabling a target environment.
- If you do not install a backend, you can still read/modify ITP and world-model modules, prompts, configs, and training code.

---

## 🗂️ What You Get After Installation (Repository Tour)

After installation, you should be able to navigate the repository by function. The layout below follows an ETO-style philosophy: **few top-level directories, clear module boundaries**.

### 📁 Repository Structure (ETO-style)

- **`itp/`**: ITP core logic  
  - Adaptive K selection (when/how far to look ahead)  
  - Orchestration (select K → imagine → reflect-and-act)  
  - Policy interfaces used by ITP_I and ITP_R

- **`world_model/`**: world model training + data processing  
  - Transition-format dataset construction  
  - World model training entrypoints  
  - World model inference / rollout API used by imagination

- **`world_model/base_tuning/`**: LoRA training assets used in experiments  
  - FastChat/DeepSpeed configs and helpers for world-model SFT  
  - Keep this as “training utilities”; core usage should not depend on it

- **`prompts/`**: prompt templates (Appendix-style, ready-to-reuse)  
  - `decide_k` (adaptive horizon selection)  
  - `imagine` (world-model foresight generation)  
  - `reflect_and_act` (foresight-conditioned final action)

- **`envs/`**: environment wrappers/adapters  
  - ALFWorld / ScienceWorld glue code  
  - Observation/action normalization, reset/step wrappers, etc.

- **`eval/`**: evaluation drivers (library-style)  
  - Minimal release runners  
  - Entry points that call env adapters + ITP agent

- **`foresight_eval/`**: paper-aligned evaluation pipeline  
  - Runners for ALFWorld/ScienceWorld with world-model backends  
  - Foresight-specific logging and analysis utilities

- **`eval_agent/`**: benchmark wrappers + metric summarization  
  - Aggregation scripts and result formatting used by the paper pipeline

- **`configs/`**: YAML configs  
  - Model paths, K_max, decoding settings  
  - Reward shaping / training hyperparameters for ITP_R  
  - Environment selection and evaluation switches

- **`docs/`**: paper-to-code mapping + implementation notes  
  - File-level alignment tables  
  - Reproducibility notes (when released)

---

## 🧩 Core Modules (What to Read First)

If you want to understand ITP quickly, start from prompts → orchestration → world model interface.

- **Adaptive K selection**  
  - Prompt: `prompts/decide_k.txt`  
  - Code: `itp/policy.py` (or equivalent selector module)

- **K-step foresight generation (imagination)**  
  - Prompt: `prompts/imagine.txt`  
  - Code: `itp/world_model.py` (calls the world model to produce foresight)  
  - Convention: imagination outputs are wrapped with `<Foresight> ... </Foresight>`

- **Foresight-conditioned action (reflect-and-act)**  
  - Prompt: `prompts/reflect_and_act.txt`  
  - Code: `itp/policy.py`

- **ITP_I orchestration (inference-time loop)**  
  - Code: `itp/orchestrator.py`  
  - Flow: select K → imagine → reflect-and-act

- **ITP_R training pipeline (adaptive lookahead learning)**  
  - Code: `itp/training/train_adaptive_k.py`  
  - Stages: pseudo-label → warm-up → online RL (A2C)  
  - Config: `configs/itp_r.yaml`

- **World model training**  
  - Code: `world_model/training/train_wm.py`

---

## 🔖 Paper-to-Code Mapping (Quick Reference)

| Paper module | Where in this repo |
| --- | --- |
| 3.1 World Model Training | `world_model/training/`, `world_model/data_processing/` |
| 3.2 Lookahead Imagination & POIMDP | `itp/world_model.py`, `itp/orchestrator.py`, `docs/poimdp.md` |
| 3.3 Planning with Adaptive Lookahead (ITP_I) | `itp/orchestrator.py`, `itp/policy.py`, `itp/world_model.py`, `prompts/` |
| 3.3 Paper-aligned evaluation | `foresight_eval/`, `eval_agent/` |
| 3.3.2 Reinforcement-trained (ITP_R) | `itp/training/train_adaptive_k.py`, `configs/itp_r.yaml` |
| Appendix Prompt Templates | `prompts/` (`decide_k` / `imagine` / `reflect_and_act`) |

For a more detailed, file-level alignment table, see `docs/paper_to_code.md`.




## 📄 Citation

If you find this work helpful, please cite our paper:

```bibtex
@article{liu2026itp,
  title        = {Imagine-then-Plan: Agent Learning from Adaptive Lookahead with World Models},
  author       = {Youwei Liu and Jian Wang and Hanlin Wang and Beichen Guo and Wenjie Li},
  journal      = {arXiv preprint arXiv:2601.08955},
  year         = {2026},
  url          = {https://arxiv.org/abs/2601.08955}
}
