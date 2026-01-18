from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from itp.orchestrator import ITPDecision, ITPOrchestrator

@dataclass
class EpisodeResult:
    success: bool
    steps: int
    terminate_reason: str
    trajectory: List[Dict[str, Any]]

def run_episode(
    env: Any,
    agent: ITPOrchestrator,
    task: str,
    max_steps: int = 50,
    get_admissible_actions: Optional[callable] = None,
) -> EpisodeResult:
    obs, info = env.reset()
    history = f"Observation: {obs}"
    traj: List[Dict[str, Any]] = []

    for t in range(int(max_steps)):
        admissible = get_admissible_actions() if get_admissible_actions else []
        decision: ITPDecision = agent.step(task=task, history=history, admissible_actions=admissible)
        step_out = env.step(decision.action)
        traj.append(
            {
                "t": t,
                "k": decision.k,
                "action": decision.action,
                "foresight": decision.foresight,
                "observation": step_out.observation if hasattr(step_out, "observation") else None,
                "reward": getattr(step_out, "reward", None),
                "done": getattr(step_out, "done", None),
            }
        )
        obs_text = step_out.observation if hasattr(step_out, "observation") else str(step_out)
        history = history + f"\nAction: {decision.action}\nObservation: {obs_text}"
        if getattr(step_out, "done", False):
            break

    success = bool(getattr(step_out, "info", {}).get("won", False)) if "step_out" in locals() else False
    terminate_reason = "done" if getattr(step_out, "done", False) else "max_steps"
    steps = len(traj)
    return EpisodeResult(success=success, steps=steps, terminate_reason=terminate_reason, trajectory=traj)

