from __future__ import annotations

import copy
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def _truncate_by_stop_strings(text: str, stop: Optional[List[str]]) -> str:
    if not stop:
        return (text or "").strip()
    s = text or ""
    cut = None
    for st in stop:
        if not st:
            continue
        idx = s.find(st)
        if idx != -1 and (cut is None or idx < cut):
            cut = idx
    return (s[:cut] if cut is not None else s).strip()

def _messages_to_prompt(tokenizer, messages: List[Dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass

    parts: List[str] = []
    for m in messages:
        role = (m.get("role") or "").strip().lower()
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role == "system":
            parts.append(f"[SYSTEM]\n{content}")
        elif role == "user":
            parts.append(f"[USER]\n{content}")
        else:
            parts.append(f"[ASSISTANT]\n{content}")
    parts.append("[ASSISTANT]\n")
    return "\n".join(parts)

def normalize_action(action: str) -> str:
    a = (action or "").strip()
    a = re.sub(r"\s+", " ", a)

    a = re.sub(r"(?is)^<Action[^>]*>\s*", "", a)
    a = re.sub(r"(?is)\s*</Action>\s*$", "", a)
    return a.strip()

def pick_fallback_action(admissible_commands: Optional[List[str]]) -> str:
    cmds = [c for c in (admissible_commands or []) if isinstance(c, str) and c.strip()]
    if cmds:
        return cmds[0].strip()
    return "look"

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

class DeepSeekAPILLM:

    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
        timeout_s: int = 120,
        max_retries: int = 3,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = (base_url or "https://api.deepseek.com").rstrip("/")
        self.timeout_s = int(timeout_s)
        self.max_retries = int(max_retries)

    def _post(self, payload: Dict) -> Dict:
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        last_err: Optional[Exception] = None
        for i in range(max(1, self.max_retries)):
            try:
                r = requests.post(url, headers=headers, json=payload, timeout=self.timeout_s)
                r.raise_for_status()
                return r.json()
            except Exception as e:
                last_err = e
                time.sleep(min(2.0, 0.2 * (i + 1)))
        raise RuntimeError(f"DeepSeek API request failed: {last_err}")

    def generate_chat(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        stop_strings: Optional[List[str]] = None,
        do_sample: bool = False,
    ) -> str:
        msgs: List[Dict[str, str]] = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": user_prompt})
        payload: Dict = {
            "model": self.model,
            "messages": msgs,
            "max_tokens": int(max_new_tokens),
            "temperature": float(temperature) if do_sample else 0.0,
            "top_p": 0.9 if do_sample else 1.0,
        }
        if stop_strings:
            payload["stop"] = stop_strings
        data = self._post(payload)
        text = (((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
        return _truncate_by_stop_strings(text, (stop_strings or []) + ["</s>", "<|eot_id|>", "<|end|>", "<|endoftext|>"])

class LocalCausalLM:

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16,
        top_p: float = 0.9,
    ):
        dev = (device or "cuda").lower().strip()
        self.device = torch.device(dev)
        self.dtype = torch.float32 if self.device.type == "cpu" else torch_dtype
        self.top_p = float(top_p)

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=self.dtype, trust_remote_code=False
            ).to(self.device)
        except Exception:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=self.dtype, trust_remote_code=True
            ).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def generate_chat(
        self,
        system_prompt: str,
        user_prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        stop_strings: Optional[List[str]] = None,
        do_sample: bool = False,
    ) -> str:
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        prompt = _messages_to_prompt(self.tokenizer, messages)
        enc = self.tokenizer(prompt, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        input_len = int(enc["input_ids"].shape[-1])

        gen_kwargs = dict(
            max_new_tokens=int(max_new_tokens),
            do_sample=bool(do_sample),
            temperature=None if not do_sample else float(temperature),
            top_p=None if not do_sample else float(self.top_p),
            top_k=None if not do_sample else 50,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        out = self.model.generate(**enc, **gen_kwargs)
        new_tokens = out[0][input_len:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        default_stop = ["</s>", "<|eot_id|>", "<|end|>", "<|endoftext|>"]
        return _truncate_by_stop_strings(text, (stop_strings or []) + default_stop)

class PolicyModel:
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16,
        decision_tokens: int = 32,
        act_tokens: int = 768,
    ):
        self.model_name = model_name
        self.device = device
        self.decision_tokens = int(decision_tokens)
        self.act_tokens = int(act_tokens)
        self.llm = LocalCausalLM(model_path=model_name, device=device, torch_dtype=torch_dtype)

    def decide_k(self, task: str, history: str, max_k: int = 3) -> Tuple[int, str]:
        max_k = int(max(0, max_k))
        system_prompt = (
            "You are a planning assistant. Your job is to decide how many steps of look-ahead are needed right now.\n"
            f"Output a single integer K in the range [0, {max_k}].\n"
            "Output ONLY the integer number, without any extra text."
        )
        user_prompt = (
            f"Task instruction:\n{task}\n\n"
            f"History trajectory (thoughts, actions, observations so far):\n{history}\n\n"
            f"Question: Output a single integer K in [0, {max_k}]."
        )
        raw = self.llm.generate_chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_new_tokens=self.decision_tokens,
            temperature=0.8,
            do_sample=True,
            stop_strings=["\n"],
        )
        try:
            k = int((raw or "").strip().split()[0])
        except Exception:
            k = min(max_k, 1 if max_k >= 1 else 0)
        k = int(torch.clamp(torch.tensor(k), 0, max_k).item())
        return k, (raw or "").strip()

    def reflect_and_act(
        self,
        task: str,
        history: str,
        foresight: str,
        admissible_commands: Optional[List[str]] = None,
    ) -> Tuple[str, str, str, str]:
        admissible_commands = [c for c in (admissible_commands or []) if isinstance(c, str) and c.strip()]
        admissible_block = ""
        if admissible_commands:
            admissible_block = (
                "\nYou MUST choose the Action by copying EXACTLY one line from the following admissible actions.\n"
                "Admissible actions:\n"
                + "\n".join(admissible_commands)
                + "\n"
            )

        system_prompt = (
            "You are a ReAct-style agent that first reflects and then acts.\n"
            "You will be given:\n"
            "1. The task instruction.\n"
            "2. The interaction history so far.\n"
            "3. A K-step foresight trajectory imagined by a world model.\n\n"
            "Your job at the current time step is to:\n"
            "- Produce a self-reflection.\n"
            "- Produce a Thought.\n"
            "- Output a concrete Action for ALFWorld.\n\n"
            "The output MUST follow this exact template (no extra text before or after):\n"
            "<Reflection> ... </Reflection>\n"
            "<Thought> ... </Thought>\n"
            "<Action> ... </Action>\n"
            + admissible_block
        )
        user_prompt = (
            f"Task instruction:\n{task}\n\n"
            f"History trajectory:\n{history}\n\n"
            f"K-step foresight trajectory from the world model:\n{foresight}"
        )
        raw = self.llm.generate_chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_new_tokens=self.act_tokens,
            temperature=0.3,
            do_sample=False,
        ).strip()

        def _extract(tag: str) -> str:
            m = re.search(rf"(?is)<{tag}[^>]*>\\s*(.*?)\\s*</{tag}>", raw)
            return (m.group(1).strip() if m else "")

        reflection = _extract("Reflection")
        thought = _extract("Thought")
        action = _extract("Action")
        if not action:
            m = re.search(r"(?im)^\\s*Action\\s*:\\s*(.+)$", raw)
            action = m.group(1).strip() if m else ""

        action = validate_action(action, admissible_commands)
        return reflection, thought, action, raw

class WorldModel:
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float16,
        foresight_tokens: int = 512,
        backend: str = "local",
        api_base_url: str = "https://api.deepseek.com",
        api_key_env: str = "DEEPSEEK_API_KEY",
        api_model: str = "deepseek-chat",
        api_timeout_s: int = 120,
        api_max_retries: int = 3,
    ):
        self.model_name = model_name
        self.device = device
        self.foresight_tokens = int(foresight_tokens)
        b = (backend or "local").strip().lower()
        if b in {"deepseek_api", "deepseek", "api"}:
            key = os.environ.get(api_key_env, "").strip()
            if not key:
                raise RuntimeError(f"DeepSeek WM backend requires env var {api_key_env} to be set (api key missing).")
            self.llm = DeepSeekAPILLM(
                api_key=key,
                model=api_model or model_name,
                base_url=api_base_url,
                timeout_s=int(api_timeout_s),
                max_retries=int(api_max_retries),
            )
        elif b == "local":
            self.llm = LocalCausalLM(model_path=model_name, device=device, torch_dtype=torch_dtype)
        else:
            raise ValueError(f"Unknown WorldModel backend: {backend}")

    def imagine(self, history: str, k: int) -> Tuple[str, str]:
        k = int(k)
        if k <= 0:
            s = "<Foresight>K = 0, no look-ahead is requested.</Foresight>"
            return s, s
        system_prompt = (
            "You are a world model for the ALFWorld environment. Given an action/observation history, "
            "imagine the next few steps, describing likely observations and key objects."
        )
        user_prompt = (
            f"History so far:\n{history}\n\n"
            f"Predict the next {k} step(s). Return a concise plan inside <Foresight>...</Foresight> with numbered steps."
        )
        raw = self.llm.generate_chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_new_tokens=self.foresight_tokens,
            temperature=0.7,
            do_sample=False,
        ).strip()
        foresight = raw if "<foresight" in raw.lower() else f"<Foresight>{raw}</Foresight>"
        return foresight, raw

__all__ = [
    "LocalCausalLM",
    "PolicyModel",
    "WorldModel",
    "normalize_action",
    "pick_fallback_action",
    "validate_action",
    "DeepSeekAPILLM",
]

