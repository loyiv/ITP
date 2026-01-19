from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Tuple

import torch

from itp.llm_backends import DeepSeekAPILLM, LocalCausalLM
from itp.prompting import load_prompt, split_system_user
from itp.policy import PromptRender

@dataclass
class WorldModel:

    backend: Literal["local", "deepseek_api"] = "local"
    model_name_or_path: str = ""
    device: str = "cuda"
    torch_dtype: torch.dtype = torch.float16
    foresight_tokens: int = 256
    imagine_prompt_path: str = str(Path(__file__).resolve().parents[1] / "prompts" / "imagine.txt")
    env_name: str = "ALFWorld"

    api_key_env: str = "DEEPSEEK_API_KEY"
    api_base_url: str = "https://api.deepseek.com"
    api_timeout_s: int = 120
    api_max_retries: int = 3

    def __post_init__(self) -> None:
        b = (self.backend or "local").strip().lower()
        if b == "local":
            if not self.model_name_or_path:
                raise ValueError("model_name_or_path is required for local world model backend.")
            self.llm = LocalCausalLM(
                model_path=self.model_name_or_path,
                device=self.device,
                torch_dtype=self.torch_dtype,
            )
        elif b in {"deepseek_api", "deepseek", "api"}:
            model = self.model_name_or_path or "deepseek-chat"
            self.llm = DeepSeekAPILLM(
                api_key_env=self.api_key_env,
                model=model,
                base_url=self.api_base_url,
                timeout_s=int(self.api_timeout_s),
                max_retries=int(self.api_max_retries),
            )
        else:
            raise ValueError(f"Unknown world model backend: {self.backend}")

    def imagine(self, history: str, k: int) -> Tuple[str, str]:
        k = int(k)
        if k <= 0:
            s = "<foresight>K = 0, no look-ahead is requested.</foresight>"
            return s, s

        tpl = load_prompt(self.imagine_prompt_path)
        parts = split_system_user(tpl.template_text)
        system_prompt = PromptRender(parts["system"]).render(env_name=self.env_name)
        user_prompt = PromptRender(parts["user"]).render(history=history, k=str(k))

        gen = self.llm.generate_chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_new_tokens=int(self.foresight_tokens),
            temperature=0.7,
            do_sample=False,
        )
        raw = (gen.text or "").strip()
        foresight = raw if "<foresight" in raw.lower() else f"<foresight>{raw}</foresight>"
        return foresight, raw

