from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from scienceworld import ScienceWorldEnv

from eval_agent.utils.replace_sciworld_score import sciworld_monkey_patch

@dataclass
class SciWorldEpisode:
    task_id: str
    sub_task_name: str
    variation_idx: int

class SciWorldEnvWrapper:

    def __init__(
        self,
        split: str,
        part_num: int = 1,
        part_idx: int = -1,
        jar_path: str = "",
        env_step_limit: int = 200,
        max_steps_override: int = 0,
    ):

        sciworld_monkey_patch()

        jar_path = (jar_path or "").strip()
        if not jar_path:
            jar_path = (os.environ.get("SCIENCEWORLD_JAR") or "").strip()
        self.jar_path = jar_path
        if not os.path.exists(self.jar_path):
            raise FileNotFoundError(f"scienceworld.jar not found: {self.jar_path}")

        ipr_root = Path(__file__).resolve().parents[1]
        data_dir = ipr_root / "eval_agent" / "data" / "sciworld"

        split = (split or "").strip().lower()
        if split == "train":
            idx_path = data_dir / "train_indices.json"
        elif split == "dev":
            idx_path = data_dir / "dev_indices.json"
        elif split == "test":
            idx_path = data_dir / "test_indices.json"
        else:
            raise ValueError(f"Unknown split: {split}")

        taskname2id_path = data_dir / "taskname2id.json"
        task_idxs = json.load(open(idx_path))
        taskname2id = json.load(open(taskname2id_path))

        if int(part_num) != 1:
            if int(part_idx) < 0:
                raise ValueError("part_idx must be set when part_num > 1")
            part_len = len(task_idxs) // int(part_num) + 1
            task_idxs = task_idxs[part_len * int(part_idx) : part_len * (int(part_idx) + 1)]

        self.episodes: List[SciWorldEpisode] = []
        for item in task_idxs:
            task_name = item[0]
            variation_idx = int(item[1])
            task_id = f"{taskname2id[task_name]}_{variation_idx}"
            self.episodes.append(
                SciWorldEpisode(task_id=str(task_id), sub_task_name=str(task_name), variation_idx=int(variation_idx))
            )
        self._i = -1

        self._env = ScienceWorldEnv("", serverPath=self.jar_path, envStepLimit=int(env_step_limit))

        max_steps_path = data_dir / "max_steps.json"
        self._max_steps_dict = json.load(open(max_steps_path)) if max_steps_path.exists() else {}
        self._max_steps_override = int(max_steps_override)

    def num_episodes(self) -> int:
        return len(self.episodes)

    def _current_max_steps(self, sub_task_name: str) -> int:
        if self._max_steps_override and self._max_steps_override > 0:
            return int(self._max_steps_override)
        v = self._max_steps_dict.get(sub_task_name)
        return int(v) if v is not None else 50

    def _get_valid_actions(self) -> List[str]:
        try:
            valid = self._env.getValidActionObjectCombinations()
        except Exception:
            return []
        if not valid:
            return []
        out: List[str] = []
        for v in valid:
            if isinstance(v, str):
                s = v.strip()
            else:
                s = " ".join(map(str, v)).strip()
            if s:
                out.append(s)
        return out

    def reset(self) -> Tuple[Dict[str, str], Dict[str, Any]]:
        self._i += 1
        if self._i >= len(self.episodes):
            self._i = 0
        ep = self.episodes[self._i]
        self._current_ep = ep

        self._env.load(ep.sub_task_name, ep.variation_idx, simplificationStr="easy", generateGoldPath=False)
        obs, info = self._env.reset()

        task_desc = (info.get("taskDesc") or "").strip()
        text = f"Task Description:\n{task_desc}\n\nObservation:\n{obs}".strip()

        info0 = dict(info or {})
        info0["episode_index"] = self._i
        info0["task_id"] = ep.task_id
        info0["sub_task_name"] = ep.sub_task_name
        info0["variation_idx"] = ep.variation_idx
        info0["admissible_commands"] = self._get_valid_actions()
        info0["max_steps"] = self._current_max_steps(ep.sub_task_name)
        return {"text": text}, info0

    def step(self, action: str) -> Tuple[Dict[str, str], float, bool, Dict[str, Any]]:
        obs, reward, done, info = self._env.step(action)
        info = dict(info or {})
        info["admissible_commands"] = self._get_valid_actions()

        ep = getattr(self, "_current_ep", None)
        if ep is not None:
            info["task_id"] = getattr(ep, "task_id", "")
            info["sub_task_name"] = getattr(ep, "sub_task_name", "")
            info["variation_idx"] = getattr(ep, "variation_idx", -1)

        terminal = bool(info.get("terminal")) if "terminal" in info else bool(done)

        try:
            score = int(info.get("score")) if "score" in info else None
        except Exception:
            score = None
        completed = bool(info.get("completed")) if "completed" in info else False
        if (score is not None) and (score < 0) and (not completed):
            info["penalized"] = True
            terminal = False
        return {"text": obs}, float(reward), terminal, info

    def success(self, info: Dict[str, Any]) -> bool:

        if "completed" in info:
            if bool(info.get("completed")):
                return True

        try:
            if float(info.get("raw_score", 0.0)) >= 0.999:
                return True
        except Exception:
            pass
        try:
            if int(info.get("score", 0)) >= 100:
                return True
        except Exception:
            pass
        return False

