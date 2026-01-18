from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from utils.io import read_text

@dataclass(frozen=True)
class PromptTemplate:

    template_text: str

    def render(self, **kwargs: str) -> str:
        text = self.template_text
        for k, v in kwargs.items():
            text = text.replace("{" + k + "}", str(v))
        return text

def load_prompt(path: str | Path) -> PromptTemplate:
    return PromptTemplate(template_text=read_text(path))

def split_system_user(prompt_text: str) -> Dict[str, str]:
    s = prompt_text
    if "System:" not in s or "User:" not in s:
        raise ValueError("Prompt file must contain 'System:' and 'User:' sections.")
    system = s.split("System:", 1)[1].split("User:", 1)[0].strip()
    user = s.split("User:", 1)[1].strip()
    return {"system": system, "user": user}

