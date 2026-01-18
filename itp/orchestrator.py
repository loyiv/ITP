from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from itp.policy import PolicyModel
from itp.world_model import WorldModel

@dataclass
class ITPDecision:
    k: int
    k_raw: str
    foresight: str
    foresight_raw: str
    reflection: str
    thought: str
    action: str
    policy_raw: str

@dataclass
class ITPOrchestrator:

    policy: PolicyModel
    world_model: WorldModel
    max_k: int = 5

    def step(
        self,
        task: str,
        history: str,
        admissible_actions: Optional[List[str]] = None,
    ) -> ITPDecision:
        k, k_raw = self.policy.decide_k(task=task, history=history, max_k=self.max_k)
        foresight, foresight_raw = self.world_model.imagine(history=history, k=k)
        reflection, thought, action, policy_raw = self.policy.reflect_and_act(
            task=task,
            history=history,
            foresight=foresight,
            admissible_commands=admissible_actions,
        )
        return ITPDecision(
            k=k,
            k_raw=k_raw,
            foresight=foresight,
            foresight_raw=foresight_raw,
            reflection=reflection,
            thought=thought,
            action=action,
            policy_raw=policy_raw,
        )

