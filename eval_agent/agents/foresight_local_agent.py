import re
import logging
from typing import List, Dict, Any, Mapping, Optional, Tuple

from eval_agent.agents.base import LMAgent

from foresight_eval.models import PolicyModel, WorldModel

logger = logging.getLogger("agent_frame")

def _extract_task(text: str) -> str:
    m = re.search(r"Your task is to:\s*(.*?)(?:\n|$)", text, re.IGNORECASE)
    if m:
        return m.group(0).strip()
    return text.strip()

def _truncate_history(messages: List[Dict[str, str]], keep_last: int) -> List[Dict[str, str]]:
    if keep_last <= 0:
        return messages
    return messages[-keep_last:]

def _messages_to_text(messages: List[Dict[str, str]]) -> str:
    parts = []
    for m in messages:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            parts.append(f"User: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
        else:
            parts.append(f"{role}: {content}")
    return "\n".join(parts)

class ForesightLocalAgent(LMAgent):

    def __init__(self, config: Mapping[str, Any]):
        super().__init__(config)
        self.policy_model_path = config["policy_model_path"]
        self.wm_model_path = config["wm_model_path"]
        self.policy_device = config.get("policy_device", "cuda:0")
        self.world_device = config.get("world_device", "cuda:0")
        self.keep_last_messages = int(config.get("keep_last_messages", 8))
        self.fixed_k: Optional[int] = config.get("fixed_k", None)

        self.policy = PolicyModel(model_name=self.policy_model_path, device=self.policy_device)
        self.world_model = WorldModel(model_name=self.wm_model_path, device=self.world_device)

    def __call__(self, messages: List[Dict[str, str]]) -> str:

        short_hist = _truncate_history(messages, self.keep_last_messages)
        history_text = _messages_to_text(short_hist)

        task_text = ""
        for m in reversed(short_hist):
            if m.get("role") == "user":
                task_text = _extract_task(m.get("content", ""))
                break
        if not task_text:
            task_text = _extract_task(history_text)

        k = int(self.fixed_k) if self.fixed_k is not None else self.policy.decide_k(task_text, history_text)
        foresight = self.world_model.imagine(history_text, k)
        reflection, thought, action = self.policy.reflect_and_act(
            task=task_text,
            history=history_text,
            foresight=foresight,
        )

        if reflection:
            thought_out = f"{reflection}\n{thought}".strip()
        else:
            thought_out = thought.strip()
        action_out = (action or "").strip() or "look around"

        return f"Thought: {thought_out}\nAction: {action_out}"

