from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

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

@dataclass
class ChatGeneration:
    text: str
    raw_text: str

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
    ) -> ChatGeneration:
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
        raw = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        default_stop = ["</s>", "<|eot_id|>", "<|end|>", "<|endoftext|>"]
        text = _truncate_by_stop_strings(raw, (stop_strings or []) + default_stop)
        return ChatGeneration(text=text, raw_text=raw)

class DeepSeekAPILLM:

    def __init__(
        self,
        api_key_env: str = "DEEPSEEK_API_KEY",
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
        timeout_s: int = 120,
        max_retries: int = 3,
    ):
        self.api_key_env = api_key_env
        self.model = model
        self.base_url = (base_url or "https://api.deepseek.com").rstrip("/")
        self.timeout_s = int(timeout_s)
        self.max_retries = int(max_retries)

    def _post(self, payload: Dict) -> Dict:
        api_key = os.environ.get(self.api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(f"Missing API key env var: {self.api_key_env}")
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
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
    ) -> ChatGeneration:
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
        raw = (((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "").strip()
        default_stop = ["</s>", "<|eot_id|>", "<|end|>", "<|endoftext|>"]
        text = _truncate_by_stop_strings(raw, (stop_strings or []) + default_stop)
        return ChatGeneration(text=text, raw_text=raw)

