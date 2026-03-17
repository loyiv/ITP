from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple, TYPE_CHECKING

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

def _import_env_wrapper():
    try:
        from worldmodel.envs.alfworld_env import AlfworldEnvWrapper as Wrapper
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "无法导入 `alfworld` 依赖。请安装 ALFWorld 相关依赖（Python 包 `alfworld`）后重试。"
        ) from exc
    return Wrapper

def load_env(data_root: str, split: str, max_steps: int, seed: int = 123):
    AlfworldEnvWrapper = _import_env_wrapper()
    return AlfworldEnvWrapper(
        data_root=data_root,
        split=split,
        max_steps=max_steps,
        seed=seed,
    )

def format_observation(env, obs: Dict[str, str]) -> str:
    return env.stringify_state(env.encode_state(obs))

def extract_task_and_obs0(first_obs: str) -> Tuple[str, str]:
    mission_prefix = "Mission:"
    obs_prefix = "Observation:"
    task = first_obs
    obs0 = first_obs

    if mission_prefix in first_obs and obs_prefix in first_obs:
        idx_m = first_obs.index(mission_prefix)
        idx_o = first_obs.index(obs_prefix)
        task = first_obs[idx_m:idx_o].strip()
        obs0 = first_obs[idx_o:].strip()

    return task, obs0
