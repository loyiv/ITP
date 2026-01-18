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
- **[Jan 15, 2026]**: 📄 Paper released on **arXiv** and **Hugging Face Papers**.
- **[Jan 18, 2026]**: 🚀 Code release (release-oriented packaging, prompts, and core modules).
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


## Usage


## Experimental Results




## 📞 Contact

For any questions, please reach out to us at [loyiv5477@gmail.com](loyiv5477@gmail.com).

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
