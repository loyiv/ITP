from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

@dataclass
class StepResult:
    observation: str
    reward: float
    done: bool
    info: Dict[str, Any]

class ScienceWorldAdapter:

    def __init__(self, env: Any):
        self.env = env

    def reset(self) -> Tuple[str, Dict[str, Any]]:
        obs, info = self.env.reset()
        return obs, (info or {})

    def step(self, action: str) -> StepResult:
        obs, reward, done, info = self.env.step(action)
        return StepResult(observation=obs, reward=float(reward), done=bool(done), info=(info or {}))

    def admissible_actions(self) -> List[str]:
        try:
            valid = self.env.getValidActionObjectCombinations()
        except Exception:
            valid = None
        if not valid:
            return []
        out: List[str] = []
        for v in valid:
            if isinstance(v, str):
                out.append(v.strip())
            else:
                out.append(" ".join(map(str, v)).strip())
        return [x for x in out if x]

