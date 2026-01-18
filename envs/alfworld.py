from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

@dataclass
class StepResult:
    observation: str
    reward: float
    done: bool
    info: Dict[str, Any]

class ALFWorldAdapter:

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
            return list(self.env.get_admissible_commands())
        except Exception:
            return []

