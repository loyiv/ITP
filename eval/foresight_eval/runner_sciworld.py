from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import re
from typing import Any, Dict, List, Tuple

import torch

from eval_agent.utils.datatypes import State

from .env_sciworld import SciWorldEnvWrapper
from .models import DeepSeekAPILLM, LocalCausalLM
from eval_agent.prompt import prompt_with_icl

logger = logging.getLogger("agent_frame")

def _parse_dtype(s: str) -> torch.dtype:
    v = (s or "").strip().lower()
    if v in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if v in {"fp16", "float16", "half"}:
        return torch.float16
    if v in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unknown dtype: {s} (use bf16/fp16/fp32)")

def _extract_action_line(text: str) -> str:
    if not text:
        return ""

    for line in text.splitlines():
        if line.strip().lower().startswith("action:"):
            return line.split(":", 1)[-1].strip()

    last = text.splitlines()[-1].strip()
    return last

class SciWorldForesightAgent:
    def __init__(
        self,
        policy_path: str,
        wm_path: str,
        policy_device: str,
        wm_device: str,
        policy_dtype: torch.dtype,
        wm_dtype: torch.dtype,
        wm_backend: str,
        wm_api_base_url: str,
        wm_api_key_env: str,
        wm_api_model: str,
        wm_api_timeout_s: int,
        wm_api_max_retries: int,
        decision_tokens: int,
        act_tokens: int,
        foresight_tokens: int,
        max_k: int,
        fixed_k: int,
        min_k: int,
    ):
        self.policy = LocalCausalLM(model_path=policy_path, device=policy_device, torch_dtype=policy_dtype)
        b = (wm_backend or "local").strip().lower()
        if b in {"deepseek_api", "deepseek", "api"}:
            key = os.environ.get(wm_api_key_env, "").strip()
            if not key:
                raise RuntimeError(f"DeepSeek WM backend requires env var {wm_api_key_env} to be set (api key missing).")
            self.wm = DeepSeekAPILLM(
                api_key=key,
                model=wm_api_model or wm_path,
                base_url=wm_api_base_url,
                timeout_s=int(wm_api_timeout_s),
                max_retries=int(wm_api_max_retries),
            )
        else:
            self.wm = LocalCausalLM(model_path=wm_path, device=wm_device, torch_dtype=wm_dtype)
        self.decision_tokens = int(decision_tokens)
        self.act_tokens = int(act_tokens)
        self.foresight_tokens = int(foresight_tokens)
        self.max_k = int(max(0, min(int(max_k), 5)))
        self.fixed_k = int(fixed_k)
        self.min_k = int(min_k)

    def decide_k(self, task_text: str, history_text: str) -> Tuple[int, str]:
        if self.fixed_k >= 0:
            k = min(self.max_k, max(0, self.fixed_k))
            return k, str(k)
        system_prompt = (
            "You are a planning assistant. Decide how many steps of look-ahead are needed now.\n"
            "Output ONLY one integer K in [0, 5] (no extra text).\n"
        )
        user_prompt = f"Task:\n{task_text}\n\nHistory:\n{history_text}\n\nK (integer only):"
        raw = self.policy.generate_chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_new_tokens=self.decision_tokens,
            temperature=0.8,
            do_sample=True,
            stop_strings=["\n"],
        )
        try:
            k = int(raw.strip().split()[0])
        except Exception:
            k = 1
        k = min(self.max_k, max(0, k))

        if k == 0 and self.max_k >= 1:
            k = 1

        if self.min_k and self.fixed_k < 0:
            k = max(self.min_k, k)
            k = min(self.max_k, k)
        return k, raw.strip()

    def imagine(self, history_text: str, k: int) -> str:
        if k <= 0:
            return "<Foresight>K=0</Foresight>"
        system_prompt = (
            "You are a world model for the ScienceWorld environment.\n"
            "Given the current observation/history, imagine the likely next steps and outcomes.\n"
            "Return a concise trajectory inside <Foresight>...</Foresight> with numbered steps.\n"
        )
        user_prompt = f"History:\n{history_text}\n\nImagine next {k} step(s):"
        raw = self.wm.generate_chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_new_tokens=self.foresight_tokens,
            temperature=0.7,
        ).strip()
        if "<foresight" not in raw.lower():
            raw = f"<Foresight>{raw}</Foresight>"
        return raw

    def act(self, task_text: str, obs_text: str, foresight: str, valid_actions: List[str]) -> Tuple[str, str]:
        raise NotImplementedError("Use runner-level prompting with eval_agent prompt_with_icl for SciWorld.")

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Foresight evaluation for ScienceWorld (policy + world model).")
    p.add_argument("--split", type=str, default="dev", choices=["dev", "test"], help="dev(seen) or test(unseen)")
    p.add_argument("--policy_model", type=str, required=True)
    p.add_argument("--wm_model", type=str, required=True)
    p.add_argument("--output_path", type=str, required=True)
    p.add_argument("--override", action="store_true")
    p.add_argument("--debug", action="store_true", help="Run only 5 tasks (per part).")
    p.add_argument("--part_num", type=int, default=1)
    p.add_argument("--part_idx", type=int, default=-1)
    p.add_argument("--only_ids", type=str, default="", help="Comma-separated episode indices to run (overrides part/debug).")
    p.add_argument("--jar_path", type=str, default="")
    p.add_argument("--env_step_limit", type=int, default=200)
    p.add_argument("--max_steps_override", type=int, default=0, help=">0 to force a fixed step limit for all tasks.")

    p.add_argument("--policy_device", type=str, default="cuda:0")
    p.add_argument("--wm_device", type=str, default="cuda:1")
    p.add_argument("--policy_dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--wm_dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--wm_backend", type=str, default="local", choices=["local", "deepseek_api"])
    p.add_argument("--wm_api_base_url", type=str, default="https://api.deepseek.com")
    p.add_argument("--wm_api_key_env", type=str, default="DEEPSEEK_API_KEY")
    p.add_argument("--wm_api_model", type=str, default="deepseek-chat")
    p.add_argument("--wm_api_timeout_s", type=int, default=120)
    p.add_argument("--wm_api_max_retries", type=int, default=3)

    p.add_argument("--decision_tokens", type=int, default=16)
    p.add_argument("--act_tokens", type=int, default=256)
    p.add_argument("--foresight_tokens", type=int, default=256)
    p.add_argument("--max_k", type=int, default=3)
    p.add_argument("--fixed_k", type=int, default=-1, help=">=0 to force constant K (skip decide_k).")
    p.add_argument("--min_k", type=int, default=1, help="Minimum K when not fixed_k (default 1 to ensure WM is used).")

    return p

def evaluate_from_args(args: argparse.Namespace) -> None:
    output_path = args.output_path
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    file_mode = "w" if args.override else "a"
    file_handler = logging.FileHandler(os.path.join(output_path, "log.txt"), mode=file_mode)
    logging.basicConfig(format="%(message)s", handlers=[logging.StreamHandler(), file_handler])

    env = SciWorldEnvWrapper(
        split=args.split,
        part_num=int(args.part_num),
        part_idx=int(args.part_idx),
        jar_path=args.jar_path,
        env_step_limit=int(args.env_step_limit),
        max_steps_override=int(args.max_steps_override),
    )

    total = env.num_episodes()
    logger.warning(f"Overall we have {total} SciWorld episodes for split={args.split}")

    policy_dtype = _parse_dtype(getattr(args, "policy_dtype", "fp16"))
    wm_dtype = _parse_dtype(getattr(args, "wm_dtype", "fp16"))
    wm_backend = (getattr(args, "wm_backend", "local") or "local").strip().lower()

    target_indices = set(range(total))
    if args.part_num > 1:
        if args.part_idx < 0:
            raise ValueError("--part_idx must be set when --part_num > 1")
        per_part = total // args.part_num + 1
        start = per_part * args.part_idx
        end = min(start + per_part, total)
        target_indices = set(range(start, end))
    if args.debug:
        target_indices = set(sorted(list(target_indices))[:5])
    only_ids = (args.only_ids or "").strip()
    if only_ids:
        ids = set()
        for x in only_ids.split(","):
            x = x.strip()
            if not x:
                continue
            try:
                ids.add(int(x))
            except Exception:
                pass
        if ids:
            target_indices = ids

    done_task_ids = set()
    states: List[State] = []
    if os.path.exists(output_path) and (not args.override):
        for fn in os.listdir(output_path):
            if fn.endswith(".json") and fn not in {"summary.json"}:
                done_task_ids.add(fn.split(".")[0])

    agent = SciWorldForesightAgent(
        policy_path=args.policy_model,
        wm_path=args.wm_model,
        policy_device=args.policy_device,
        wm_device=args.wm_device,
        policy_dtype=policy_dtype,
        wm_dtype=wm_dtype,
        wm_backend=wm_backend,
        wm_api_base_url=getattr(args, "wm_api_base_url", "https://api.deepseek.com"),
        wm_api_key_env=getattr(args, "wm_api_key_env", "DEEPSEEK_API_KEY"),
        wm_api_model=getattr(args, "wm_api_model", "deepseek-chat"),
        wm_api_timeout_s=int(getattr(args, "wm_api_timeout_s", 120)),
        wm_api_max_retries=int(getattr(args, "wm_api_max_retries", 3)),
        decision_tokens=args.decision_tokens,
        act_tokens=args.act_tokens,
        foresight_tokens=args.foresight_tokens,
        max_k=args.max_k,
        fixed_k=args.fixed_k,
        min_k=args.min_k,
    )

    ipr_root = pathlib.Path(__file__).resolve().parents[1]
    inst_path = ipr_root / "eval_agent" / "prompt" / "instructions" / "sciworld_react.txt"
    icl_path = ipr_root / "eval_agent" / "prompt" / "icl_examples" / "sciworld_icl.json"
    instruction = inst_path.read_text()
    raw_icl = json.load(open(icl_path))

    max_idx = max(target_indices) if target_indices else -1
    for idx in range(max_idx + 1):
        obs0_dict, info0 = env.reset()
        if idx not in target_indices:
            continue
        if str(idx) in done_task_ids:
            continue

        full0 = (obs0_dict.get("text") or "")

        task_desc = (info0.get("taskDesc") or "").strip()

        obs_init = full0.split("Observation:\n", 1)[-1].strip() if "Observation:\n" in full0 else full0
        task_text = f"Task Description:\n{task_desc}".strip()
        obs_text = obs_init
        max_steps = int(info0.get("max_steps") or 50)

        state = State()

        _prompt, base_messages = prompt_with_icl(instruction, raw_icl, task_text, icl_num=1)
        base_messages.append({"role": "user", "content": f"Observation: {obs_text}".strip()})
        prompt_messages = list(base_messages)

        state.history = [{"role": "user", "content": f"{task_text}\nObservation: {obs_text}".strip()}]

        done = False
        last_info = dict(info0 or {})
        last_obs = obs0_dict
        step_records: List[Dict[str, Any]] = []

        for t in range(1, max_steps + 1):
            valid_actions = (last_info.get("admissible_commands") or [])

            history_text = obs_text

            k, k_raw = agent.decide_k(task_text, history_text)
            foresight = agent.imagine(history_text, k)
            policy_messages = list(prompt_messages)
            policy_messages.append(
                {
                    "role": "user",
                    "content": (
                        f"World-model foresight:\n{foresight}\n\n"
                        "Now output in strict format:\n"
                        "Thought: <1-3 sentences>\n"
                        "Action: <one valid action>\n"
                    ),
                }
            )
            policy_raw = agent.policy.generate_messages(
                messages=policy_messages,
                max_new_tokens=agent.act_tokens,
                temperature=0.3,
                do_sample=False,
            ).strip()
            action = _extract_action_line(policy_raw)

            next_obs, _reward, _done, info = env.step(action)
            last_obs = next_obs
            last_info = info
            obs_text = (next_obs.get("text") or "")

            step_records.append(
                {
                    "t": t,
                    "k": k,
                    "k_raw": k_raw,
                    "foresight": foresight,
                    "policy_raw_output": policy_raw,
                    "action_executed": action,
                    "terminal": bool(info.get("terminal")) if isinstance(info, dict) and ("terminal" in info) else bool(_done),
                    "completed": bool(info.get("completed")) if isinstance(info, dict) and ("completed" in info) else None,
                    "score": info.get("score") if isinstance(info, dict) else None,
                    "raw_score": info.get("raw_score") if isinstance(info, dict) else None,
                    "penalized": bool(info.get("penalized")) if isinstance(info, dict) else None,
                }
            )

            prompt_messages.append({"role": "assistant", "content": policy_raw})
            prompt_messages.append({"role": "user", "content": f"Observation: {obs_text}".strip()})

            state.history.append({"role": "assistant", "content": policy_raw})
            state.history.append({"role": "user", "content": f"Observation: {obs_text}".strip()})

            if _done:
                done = True
                break

        success = env.success(last_info)
        state.finished = True
        state.success = bool(success)
        state.reward = 1.0 if success else 0.0
        state.steps = len(step_records)
        state.terminate_reason = "success" if success else "max_steps"

        out_path = os.path.join(output_path, f"{idx}.json")
        payload = state.to_dict()
        payload["foresight_steps"] = step_records
        payload["meta"] = {
            "episode_index": idx,
            "task_id": info0.get("task_id"),
            "sub_task_name": info0.get("sub_task_name"),
            "variation_idx": info0.get("variation_idx"),
        }
        json.dump(payload, open(out_path, "w"), indent=2, ensure_ascii=False)
        states.append(state)
        logger.warning(f"[TaskResult] id={idx} success={state.success} steps={state.steps}")

    logger.warning("All tasks done.")
    logger.warning(f"Output saved to {output_path}")
    if states:
        sr = sum(1 for s in states if s.success) / len(states)
        logger.warning(f"Success rate: {sr:.4f}")

if __name__ == "__main__":
    evaluate_from_args(build_arg_parser().parse_args())

import argparse
import difflib
import json
import logging
import os
import pathlib
import re
from typing import Any, Dict, List, Optional, Tuple

import torch

from eval_agent.utils.datatypes import State

from .env_sciworld import SciWorldEnvWrapper
from .models import DeepSeekAPILLM, LocalCausalLM
from eval_agent.prompt import prompt_with_icl

logger = logging.getLogger("agent_frame")

_ROOMS = [
    "kitchen",
    "foundry",
    "workshop",
    "bathroom",
    "outside",
    "living room",
    "bedroom",
    "greenhouse",
    "art studio",
    "hallway",
]

def _parse_dtype(s: str) -> torch.dtype:
    v = (s or "").strip().lower()
    if v in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if v in {"fp16", "float16", "half"}:
        return torch.float16
    if v in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unknown dtype: {s} (use bf16/fp16/fp32)")

def _pick_best_match(action: str, valid: List[str]) -> str:
    if not valid:
        return action
    a = (action or "").strip()
    if a in valid:
        return a
    matches = difflib.get_close_matches(a, valid, n=1, cutoff=0.55)
    return matches[0] if matches else valid[0]

def _extract_action_line(text: str) -> str:
    if not text:
        return ""

    for line in text.splitlines():
        if line.strip().lower().startswith("action:"):
            return line.split(":", 1)[-1].strip()

    last = text.splitlines()[-1].strip()
    return last

def _extract_goal_obj(task_block: str) -> str:
    low = (task_block or "").lower()

    if "unknown substance s" in low:
        return "unknown substance S"

    for verb in ["boil", "freeze", "melt", "heat", "cool", "clean"]:
        m = re.search(rf"{verb}\s+([a-z][a-z0-9 _-]+?)(?:[\\.,\\n]|$)", low)
        if m:
            return m.group(1).strip()

    m = re.search(r"focus on (?:the )?([a-z][a-z0-9 _-]+?)(?:[\\.,\\n]|$)", low)
    if m:
        return m.group(1).strip()
    return ""

class SciWorldForesightAgent:
    def __init__(
        self,
        policy_path: str,
        wm_path: str,
        policy_device: str,
        wm_device: str,
        policy_dtype: torch.dtype,
        wm_dtype: torch.dtype,
        wm_backend: str,
        wm_api_base_url: str,
        wm_api_key_env: str,
        wm_api_model: str,
        wm_api_timeout_s: int,
        wm_api_max_retries: int,
        decision_tokens: int,
        act_tokens: int,
        foresight_tokens: int,
        max_k: int,
        fixed_k: int,
        min_k: int,
    ):
        self.policy = LocalCausalLM(model_path=policy_path, device=policy_device, torch_dtype=policy_dtype)
        b = (wm_backend or "local").strip().lower()
        if b in {"deepseek_api", "deepseek", "api"}:
            key = os.environ.get(wm_api_key_env, "").strip()
            if not key:
                raise RuntimeError(f"DeepSeek WM backend requires env var {wm_api_key_env} to be set (api key missing).")
            self.wm = DeepSeekAPILLM(
                api_key=key,
                model=wm_api_model or wm_path,
                base_url=wm_api_base_url,
                timeout_s=int(wm_api_timeout_s),
                max_retries=int(wm_api_max_retries),
            )
        else:
            self.wm = LocalCausalLM(model_path=wm_path, device=wm_device, torch_dtype=wm_dtype)
        self.decision_tokens = int(decision_tokens)
        self.act_tokens = int(act_tokens)
        self.foresight_tokens = int(foresight_tokens)
        self.max_k = int(max(0, min(int(max_k), 5)))
        self.fixed_k = int(fixed_k)
        self.min_k = int(min_k)

    def decide_k(self, task_text: str, history_text: str) -> Tuple[int, str]:
        if self.fixed_k >= 0:
            k = min(self.max_k, max(0, self.fixed_k))
            return k, str(k)
        system_prompt = (
            "You are a planning assistant. Decide how many steps of look-ahead are needed now.\n"
            "Output ONLY one integer K in [0, 5] (no extra text).\n"
        )
        user_prompt = f"Task:\n{task_text}\n\nHistory:\n{history_text}\n\nK (integer only):"
        raw = self.policy.generate_chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_new_tokens=self.decision_tokens,
            temperature=0.8,
            do_sample=True,
            stop_strings=["\n"],
        )
        try:
            k = int(raw.strip().split()[0])
        except Exception:
            k = 1
        k = min(self.max_k, max(0, k))

        if k == 0 and self.max_k >= 1:
            k = 1

        if self.min_k and self.fixed_k < 0:
            k = max(self.min_k, k)
            k = min(self.max_k, k)
        return k, raw.strip()

    def imagine(self, history_text: str, k: int) -> str:
        if k <= 0:
            return "<Foresight>K=0</Foresight>"
        system_prompt = (
            "You are a world model for the ScienceWorld environment.\n"
            "Given the current observation/history, imagine the likely next steps and outcomes.\n"
            "Return a concise trajectory inside <Foresight>...</Foresight> with numbered steps.\n"
        )
        user_prompt = f"History:\n{history_text}\n\nImagine next {k} step(s):"
        raw = self.wm.generate_chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_new_tokens=self.foresight_tokens,
            temperature=0.7,
        ).strip()
        if "<foresight" not in raw.lower():
            raw = f"<Foresight>{raw}</Foresight>"
        return raw

    def act(self, task_text: str, obs_text: str, foresight: str, valid_actions: List[str]) -> Tuple[str, str]:
        raise NotImplementedError("Use runner-level prompting with eval_agent prompt_with_icl for SciWorld.")

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Foresight evaluation for ScienceWorld (policy + world model).")
    p.add_argument("--split", type=str, default="dev", choices=["dev", "test"], help="dev(seen) or test(unseen)")
    p.add_argument("--policy_model", type=str, required=True)
    p.add_argument("--wm_model", type=str, required=True)
    p.add_argument("--output_path", type=str, required=True)
    p.add_argument("--override", action="store_true")
    p.add_argument("--debug", action="store_true", help="Run only 5 tasks (per part).")
    p.add_argument("--part_num", type=int, default=1)
    p.add_argument("--part_idx", type=int, default=-1)
    p.add_argument("--only_ids", type=str, default="", help="Comma-separated episode indices to run (overrides part/debug).")
    p.add_argument("--jar_path", type=str, default="/code/STeCa/IPR/envs/scienceworld/scienceworld.jar")
    p.add_argument("--env_step_limit", type=int, default=200)
    p.add_argument("--max_steps_override", type=int, default=0, help=">0 to force a fixed step limit for all tasks.")

    p.add_argument("--policy_device", type=str, default="cuda:0")
    p.add_argument("--wm_device", type=str, default="cuda:1")
    p.add_argument("--policy_dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--wm_dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--wm_backend", type=str, default="local", choices=["local", "deepseek_api"])
    p.add_argument("--wm_api_base_url", type=str, default="https://api.deepseek.com")
    p.add_argument("--wm_api_key_env", type=str, default="DEEPSEEK_API_KEY")
    p.add_argument("--wm_api_model", type=str, default="deepseek-chat")
    p.add_argument("--wm_api_timeout_s", type=int, default=120)
    p.add_argument("--wm_api_max_retries", type=int, default=3)

    p.add_argument("--decision_tokens", type=int, default=16)
    p.add_argument("--act_tokens", type=int, default=256)
    p.add_argument("--foresight_tokens", type=int, default=256)
    p.add_argument("--max_k", type=int, default=3)
    p.add_argument("--fixed_k", type=int, default=-1, help=">=0 to force constant K (skip decide_k).")
    p.add_argument("--min_k", type=int, default=1, help="Minimum K when not fixed_k (default 1 to ensure WM is used).")

    return p

def evaluate_from_args(args: argparse.Namespace) -> None:
    output_path = args.output_path
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    file_mode = "w" if args.override else "a"
    file_handler = logging.FileHandler(os.path.join(output_path, "log.txt"), mode=file_mode)
    logging.basicConfig(format="%(message)s", handlers=[logging.StreamHandler(), file_handler])

    env = SciWorldEnvWrapper(
        split=args.split,
        part_num=int(args.part_num),
        part_idx=int(args.part_idx),
        jar_path=args.jar_path,
        env_step_limit=int(args.env_step_limit),
        max_steps_override=int(args.max_steps_override),
    )

    total = env.num_episodes()
    logger.warning(f"Overall we have {total} SciWorld episodes for split={args.split}")

    policy_dtype = _parse_dtype(getattr(args, "policy_dtype", "fp16"))
    wm_dtype = _parse_dtype(getattr(args, "wm_dtype", "fp16"))
    wm_backend = (getattr(args, "wm_backend", "local") or "local").strip().lower()

    target_indices = set(range(total))
    if args.part_num > 1:
        if args.part_idx < 0:
            raise ValueError("--part_idx must be set when --part_num > 1")
        per_part = total // args.part_num + 1
        start = per_part * args.part_idx
        end = min(start + per_part, total)
        target_indices = set(range(start, end))
    if args.debug:
        target_indices = set(sorted(list(target_indices))[:5])
    only_ids = (args.only_ids or "").strip()
    if only_ids:
        ids = set()
        for x in only_ids.split(","):
            x = x.strip()
            if not x:
                continue
            try:
                ids.add(int(x))
            except Exception:
                pass
        if ids:
            target_indices = ids

    done_task_ids = set()
    states: List[State] = []
    if os.path.exists(output_path) and (not args.override):
        for fn in os.listdir(output_path):
            if fn.endswith(".json") and fn not in {"summary.json"}:
                done_task_ids.add(fn.split(".")[0])

    agent = SciWorldForesightAgent(
        policy_path=args.policy_model,
        wm_path=args.wm_model,
        policy_device=args.policy_device,
        wm_device=args.wm_device,
        policy_dtype=policy_dtype,
        wm_dtype=wm_dtype,
        wm_backend=wm_backend,
        wm_api_base_url=getattr(args, "wm_api_base_url", "https://api.deepseek.com"),
        wm_api_key_env=getattr(args, "wm_api_key_env", "DEEPSEEK_API_KEY"),
        wm_api_model=getattr(args, "wm_api_model", "deepseek-chat"),
        wm_api_timeout_s=int(getattr(args, "wm_api_timeout_s", 120)),
        wm_api_max_retries=int(getattr(args, "wm_api_max_retries", 3)),
        decision_tokens=args.decision_tokens,
        act_tokens=args.act_tokens,
        foresight_tokens=args.foresight_tokens,
        max_k=args.max_k,
        fixed_k=args.fixed_k,
        min_k=args.min_k,
    )

    ipr_root = pathlib.Path(__file__).resolve().parents[1]
    inst_path = ipr_root / "eval_agent" / "prompt" / "instructions" / "sciworld_react.txt"
    icl_path = ipr_root / "eval_agent" / "prompt" / "icl_examples" / "sciworld_icl.json"
    instruction = inst_path.read_text()
    raw_icl = json.load(open(icl_path))

    max_idx = max(target_indices) if target_indices else -1
    for idx in range(max_idx + 1):
        obs0_dict, info0 = env.reset()
        if idx not in target_indices:
            continue
        if str(idx) in done_task_ids:
            continue

        full0 = (obs0_dict.get("text") or "")

        task_desc = (info0.get("taskDesc") or "").strip()

        obs_init = full0.split("Observation:\n", 1)[-1].strip() if "Observation:\n" in full0 else full0
        task_text = f"Task Description:\n{task_desc}".strip()
        obs_text = obs_init
        max_steps = int(info0.get("max_steps") or 50)

        state = State()

        _prompt, base_messages = prompt_with_icl(instruction, raw_icl, task_text, icl_num=1)
        base_messages.append({"role": "user", "content": f"Observation: {obs_text}".strip()})
        prompt_messages = list(base_messages)

        state.history = [{"role": "user", "content": f"{task_text}\nObservation: {obs_text}".strip()}]

        done = False
        last_info = dict(info0 or {})
        last_obs = obs0_dict
        step_records: List[Dict[str, Any]] = []

        for t in range(1, max_steps + 1):
            valid_actions = (last_info.get("admissible_commands") or [])

            history_text = obs_text

            k, k_raw = agent.decide_k(task_text, history_text)
            foresight = agent.imagine(history_text, k)
            policy_messages = list(prompt_messages)
            policy_messages.append(
                {
                    "role": "user",
                    "content": (
                        f"World-model foresight:\n{foresight}\n\n"
                        "Now output in strict format:\n"
                        "Thought: <1-3 sentences>\n"
                        "Action: <one valid action>\n"
                    ),
                }
            )
            policy_raw = agent.policy.generate_messages(
                messages=policy_messages,
                max_new_tokens=agent.act_tokens,
                temperature=0.3,
                do_sample=False,
            ).strip()
            action = _extract_action_line(policy_raw)

            next_obs, _reward, _done, info = env.step(action)
            last_obs = next_obs
            last_info = info
            obs_text = (next_obs.get("text") or "")

            step_records.append(
                {
                    "t": t,
                    "k": k,
                    "k_raw": k_raw,
                    "foresight": foresight,
                    "policy_raw_output": policy_raw,
                    "action_executed": action,
                    "terminal": bool(info.get("terminal")) if isinstance(info, dict) and ("terminal" in info) else bool(_done),
                    "completed": bool(info.get("completed")) if isinstance(info, dict) and ("completed" in info) else None,
                    "score": info.get("score") if isinstance(info, dict) else None,
                    "raw_score": info.get("raw_score") if isinstance(info, dict) else None,
                    "penalized": bool(info.get("penalized")) if isinstance(info, dict) else None,
                }
            )

            prompt_messages.append({"role": "assistant", "content": policy_raw})
            prompt_messages.append({"role": "user", "content": f"Observation: {obs_text}".strip()})

            state.history.append({"role": "assistant", "content": policy_raw})
            state.history.append({"role": "user", "content": f"Observation: {obs_text}".strip()})

            if _done:
                done = True
                break

        success = env.success(last_info)
        state.finished = True
        state.success = bool(success)
        state.reward = 1.0 if success else 0.0
        state.steps = len(step_records)
        state.terminate_reason = "success" if success else "max_steps"

        out_path = os.path.join(output_path, f"{idx}.json")
        payload = state.to_dict()
        payload["foresight_steps"] = step_records
        payload["meta"] = {
            "episode_index": idx,
            "task_id": info0.get("task_id"),
            "sub_task_name": info0.get("sub_task_name"),
            "variation_idx": info0.get("variation_idx"),
        }
        json.dump(payload, open(out_path, "w"), indent=2, ensure_ascii=False)
        states.append(state)
        logger.warning(f"[TaskResult] id={idx} success={state.success} steps={state.steps}")

    logger.warning("All tasks done.")
    logger.warning(f"Output saved to {output_path}")
    if states:
        sr = sum(1 for s in states if s.success) / len(states)
        logger.warning(f"Success rate: {sr:.4f}")

if __name__ == "__main__":
    evaluate_from_args(build_arg_parser().parse_args())
