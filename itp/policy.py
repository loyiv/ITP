from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch

from itp.llm_backends import LocalCausalLM
from itp.prompting import load_prompt, split_system_user

def normalize_action(action: str) -> str:
    a = (action or "").strip()
    a = re.sub(r"\s+", " ", a)

    a = re.sub(r"(?is)^<action[^>]*>\s*", "", a)
    a = re.sub(r"(?is)\s*</action>\s*$", "", a)
    return a.strip()

def pick_fallback_action(admissible_commands: Optional[List[str]]) -> str:
    cmds = [c for c in (admissible_commands or []) if isinstance(c, str) and c.strip()]
    return cmds[0].strip() if cmds else "look"

def validate_action(candidate: str, admissible_commands: Optional[List[str]]) -> str:
    cand = normalize_action(candidate)
    cmds = [c for c in (admissible_commands or []) if isinstance(c, str)]
    if not cmds:
        return cand or "look"
    if cand in cmds:
        return cand

    cand_l = cand.lower()
    for c in cmds:
        if c.lower() == cand_l:
            return c
    return pick_fallback_action(cmds)

@dataclass
class PolicyModel:

    llm: LocalCausalLM
    decision_tokens: int = 32
    act_tokens: int = 768
    decide_k_prompt_path: str = str(Path(__file__).resolve().parents[1] / "prompts" / "decide_k.txt")
    reflect_and_act_prompt_path: str = str(Path(__file__).resolve().parents[1] / "prompts" / "reflect_and_act.txt")

    def decide_k(self, task: str, history: str, max_k: int) -> Tuple[int, str]:
        max_k = int(max(0, max_k))
        tpl = load_prompt(self.decide_k_prompt_path)
        parts = split_system_user(tpl.template_text)
        system_prompt = PromptRender(parts["system"]).render(kmax=str(max_k))
        user_prompt = PromptRender(parts["user"]).render(task=task, history=history, kmax=str(max_k))

        gen = self.llm.generate_chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_new_tokens=self.decision_tokens,
            temperature=0.8,
            do_sample=True,
            stop_strings=["\n"],
        )
        raw = (gen.text or "").strip()
        try:
            k = int(raw.split()[0])
        except Exception:
            k = min(max_k, 1 if max_k >= 1 else 0)
        k = int(torch.clamp(torch.tensor(k), 0, max_k).item())
        return k, raw

    def reflect_and_act(
        self,
        task: str,
        history: str,
        foresight: str,
        admissible_commands: Optional[List[str]] = None,
    ) -> Tuple[str, str, str, str]:
        admissible_commands = [c for c in (admissible_commands or []) if isinstance(c, str) and c.strip()]
        admissible_text = "\n".join(admissible_commands)

        tpl = load_prompt(self.reflect_and_act_prompt_path)
        parts = split_system_user(tpl.template_text)
        system_prompt = PromptRender(parts["system"]).render(
            task=task,
            history=history,
            foresight=foresight,
            admissible_actions=admissible_text,
        )
        user_prompt = PromptRender(parts["user"]).render(
            task=task,
            history=history,
            foresight=foresight,
            admissible_actions=admissible_text,
        )

        gen = self.llm.generate_chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_new_tokens=self.act_tokens,
            temperature=0.3,
            do_sample=False,
        )
        raw = (gen.text or "").strip()

        reflection = _extract_tag_or_field(raw, "reflection", "Reflection")
        thought = _extract_tag_or_field(raw, "thought", "Thought")
        action = _extract_tag_or_field(raw, "action", "Action")
        action = validate_action(action, admissible_commands)
        return reflection, thought, action, raw

class PromptRender:

    def __init__(self, template: str):
        self.template = template

    def render(self, **kwargs: str) -> str:
        s = self.template
        for k, v in kwargs.items():
            s = s.replace("{" + k + "}", str(v))
        return s

def _extract_tag_or_field(text: str, tag: str, field: str) -> str:
    raw = text or ""
    m = re.search(rf"(?is)<{tag}[^>]*>\s*(.*?)\s*</{tag}>", raw)
    if m:
        return m.group(1).strip()
    m = re.search(rf"(?is)<{field}[^>]*>\s*(.*?)\s*</{field}>", raw)
    if m:
        return m.group(1).strip()
    m = re.search(rf"(?im)^\s*{re.escape(field)}\s*:\s*(.+?)\s*$", raw)
    if m:
        return m.group(1).strip()
    return ""

