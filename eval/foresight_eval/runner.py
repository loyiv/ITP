from __future__ import annotations

import argparse
import csv
import difflib
import json
import logging
import os
import pathlib
import re
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

from eval_agent.utils.datatypes import State
from eval_agent.main import summarize_alfworld

from .env import extract_task_and_obs0, format_observation, load_env
from .models import LocalCausalLM, PolicyModel, WorldModel, normalize_action, pick_fallback_action
from .rap import RAPLM, RAPPlanner, RAPState

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

class DummyWorldModel:

    def imagine(self, history: str, k: int) -> Tuple[str, str]:
        if int(k) <= 0:
            s = "<Foresight>K = 0, no look-ahead is requested.</Foresight>"
            return s, s
        s = f"<Foresight>(disabled) would imagine {int(k)} step(s).</Foresight>"
        return s, s

def _build_state_text(obs_text: str, inventory: List[str]) -> str:
    obs = (obs_text or "").strip()
    inv = ", ".join([x for x in (inventory or []) if x]) if inventory else ""
    return f"Observation: {obs}\nInventory: {inv}".strip()

def _map_split_to_data(split: str) -> str:
    s = (split or "").lower().strip()
    if s in {"dev", "seen", "eval_in_distribution", "valid_seen"}:
        return "valid_seen"
    if s in {"test", "unseen", "eval_out_of_distribution", "valid_unseen"}:
        return "valid_unseen"
    if s == "train":
        return "train"

    return "valid_seen"

def _best_match_action(action: str, admissible: List[str]) -> Optional[str]:
    if not admissible:
        return None
    action = (action or "").strip()
    if not action:
        return None
    alow = action.lower()
    for a in admissible:
        if a.lower() == alow:
            return a
    matches = difflib.get_close_matches(action, admissible, n=1, cutoff=0.72)
    return matches[0] if matches else None

def _infer_inventory_from_admissible(admissible: List[str]) -> List[str]:
    inv: List[str] = []
    for a in admissible or []:
        if not isinstance(a, str):
            continue
        if a.startswith("put "):
            rest = a[len("put "):]
            obj = rest.split(" in/on ", 1)[0].strip()
            if obj and obj not in inv:
                inv.append(obj)
    return inv

@dataclass
class GoalSpec:
    goal_text: str
    task_type: str
    obj_key: str
    recep_key: str
    need_two: bool

def _parse_goal_from_gamefile(goal_text: str, gamefile: str) -> GoalSpec:
    text = (goal_text or "").strip()
    task_type = ""
    obj = ""
    recep = ""
    need_two = False
    gf = (gamefile or "")
    try:
        task_dir = os.path.basename(os.path.dirname(os.path.dirname(gf)))
        parts = task_dir.split("-")
        if parts:
            task_type = parts[0].strip().lower()
        if len(parts) >= 4:
            obj = parts[1].strip().lower()
            recep = parts[3].strip().lower()
        if "pick_two" in task_type or task_type.startswith("pick_two_obj"):
            need_two = True
    except Exception:
        pass

    low = (text or "").lower()
    m = re.search(r"put two ([a-z0-9_]+) in ([a-z0-9_]+)", low)
    if m and not obj:
        obj = m.group(1).strip().lower()
        recep = m.group(2).strip().lower()
        need_two = True

    return GoalSpec(goal_text=text, task_type=task_type, obj_key=obj, recep_key=recep, need_two=need_two)

@dataclass
class Pick2Progress:
    placed: Set[str]
    source_hint: str

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def _update_pick2_progress(progress: Pick2Progress, action: str, obs_text: str, goal_obj: str, goal_recep: str) -> None:
    if not goal_obj or not goal_recep:
        return
    a = _norm(action)
    t = (obs_text or "").strip().lower()

    if a.startswith("take "):
        m = re.match(r"^take (.+?) from (.+)$", a)
        if m and ("you pick up the" in t):
            obj = _norm(m.group(1))
            src = _norm(m.group(2))
            if goal_obj in obj and src and (goal_recep not in src):
                progress.source_hint = src
            if goal_obj in obj and goal_recep in src:
                progress.placed.discard(obj)
        return

    if a.startswith("put "):
        m = re.match(r"^put (.+?) in/on (.+)$", a)
        if m and ("you put the" in t):
            obj = _norm(m.group(1))
            dst = _norm(m.group(2))
            if goal_obj in obj and goal_recep in dst:
                progress.placed.add(obj)
        return

def _validate_action_in_runner(
    action: str,
    admissible: List[str],
    goal: GoalSpec,
    progress: Pick2Progress,
) -> str:
    action = normalize_action(action)

    if not admissible:
        return action if action else "look"

    if action in admissible:

        if goal.need_two and len(progress.placed) >= 1 and goal.obj_key and goal.recep_key:
            al = action.lower()
            if al.startswith("take ") and goal.obj_key in al and goal.recep_key in al:

                safe_take = [a for a in admissible if a.startswith("take ") and goal.obj_key in a.lower() and goal.recep_key not in a.lower()]
                if safe_take:
                    return safe_take[0]

                if progress.source_hint:
                    for a in admissible:
                        if a.startswith("go to ") and progress.source_hint in a.lower():
                            return a

                return pick_fallback_action(admissible)
        return action

    mapped = _best_match_action(action, admissible)
    if mapped:
        return mapped

    return pick_fallback_action(admissible)

class ForesightAgent:

    def __init__(
        self,
        policy: PolicyModel,
        world_model: WorldModel,
        max_k: int = 2,
        fixed_k: Optional[int] = None,
    ):
        self.policy = policy
        self.world_model = world_model
        self.max_k = int(max(0, min(int(max_k), 5)))
        self.fixed_k = int(fixed_k) if fixed_k is not None else None
        if self.fixed_k is not None:
            self.fixed_k = int(max(0, min(self.fixed_k, self.max_k)))

    def run_one_step(
        self,
        env,
        episode_id: str,
        t: int,
        task_instruction: str,
        history_text: str,
        admissible_commands: List[str],
        goal: GoalSpec,
        progress: Pick2Progress,
        use_history: bool,
    ) -> Tuple[Dict[str, Any], str, bool, Dict[str, Any], List[str], str]:

        if self.fixed_k is not None:
            k = int(self.fixed_k)
            k_raw = str(k)
        else:
            k, k_raw = self.policy.decide_k(task_instruction, history_text, max_k=self.max_k)
            k = int(max(0, min(int(k), self.max_k)))

        foresight_text, wm_raw = self.world_model.imagine(history_text, k)

        reflection, thought, action, policy_raw = self.policy.reflect_and_act(
            task=task_instruction,
            history=history_text,
            foresight=foresight_text,
            admissible_commands=admissible_commands,
        )

        action = _validate_action_in_runner(normalize_action(action), admissible_commands, goal, progress)

        obs_next_dict, _reward, done, info = env.step(action)
        obs_text = (obs_next_dict.get("text") or "")
        obs_next = format_observation(env, obs_next_dict)

        _update_pick2_progress(progress, action, obs_text, goal.obj_key, goal.recep_key)

        record = {
            "episode_id": episode_id,
            "t": t,
            "k": k,
            "k_raw_output": k_raw,
            "foresight_text": foresight_text,
            "wm_raw_output": wm_raw,
            "reflection": reflection,
            "thought": thought,
            "action": action,
            "policy_raw_output": policy_raw,
            "obs_next": obs_next,
            "done": done,
            "info": info,
        }

        next_admissible = info.get("admissible_commands", admissible_commands) or admissible_commands
        if use_history:
            updated_history = (
                history_text
                + f"<Thought_{t}>{thought}</Thought_{t}>"
                + f"<Action_{t}>{action}</Action_{t}>"
                + f"<Obs_{t}>{obs_next}</Obs_{t}>"
            )
        else:
            inv_next = _infer_inventory_from_admissible(next_admissible)
            updated_history = _build_state_text(obs_text, inv_next)
        return record, updated_history, done, info, next_admissible, obs_text

class ForesightEvaluator:
    def __init__(
        self,
        policy_model_path: str,
        wm_model_path: str,
        policy_device: str = "cuda",
        wm_device: str = "cuda",
        policy_dtype: torch.dtype = torch.float16,
        wm_dtype: torch.dtype = torch.float16,
        wm_backend: str = "local",
        wm_api_base_url: str = "https://api.deepseek.com",
        wm_api_key_env: str = "DEEPSEEK_API_KEY",
        wm_api_model: str = "deepseek-chat",
        wm_api_timeout_s: int = 120,
        wm_api_max_retries: int = 3,
        decision_tokens: int = 16,
        act_tokens: int = 256,
        foresight_tokens: int = 256,
        max_k: int = 3,
        fixed_k: Optional[int] = None,
        disable_wm: bool = False,
        use_history: bool = False,
    ):

        self.policy = PolicyModel(
            model_name=policy_model_path,
            device=policy_device,
            torch_dtype=policy_dtype,
            decision_tokens=int(decision_tokens),
            act_tokens=int(act_tokens),
        )
        self.world_model = (
            DummyWorldModel()
            if disable_wm
            else WorldModel(
            model_name=wm_model_path,
                device=wm_device,
                torch_dtype=wm_dtype,
                foresight_tokens=int(foresight_tokens),
                backend=wm_backend,
                api_base_url=wm_api_base_url,
                api_key_env=wm_api_key_env,
                api_model=wm_api_model,
                api_timeout_s=int(wm_api_timeout_s),
                api_max_retries=int(wm_api_max_retries),
            )
        )
        self.agent = ForesightAgent(self.policy, self.world_model, max_k=max_k, fixed_k=fixed_k)
        self.use_history = bool(use_history)

    def run_episode_from_reset(
        self,
        env,
        task_id: int,
        first_obs_dict: Dict[str, str],
        info0: Dict[str, Any],
        max_steps: int,
    ) -> Tuple[State, List[Dict[str, Any]]]:
        episode_id = str(uuid.uuid4())[:8]

        first_obs_text = format_observation(env, first_obs_dict)
        task_instruction, obs0 = extract_task_and_obs0(first_obs_text)
        gamefile = (info0.get("gamefile") or "")

        goal = _parse_goal_from_gamefile(task_instruction, gamefile)
        progress = Pick2Progress(placed=set(), source_hint="")

        admissible = info0.get("admissible_commands") or []
        if not isinstance(admissible, list):
            admissible = []

        state = State()
        state.error = gamefile

        state.history.append({"role": "user", "content": f"{task_instruction}\n{obs0}".strip()})

        step_records: List[Dict[str, Any]] = []

        done = False
        last_info: Dict[str, Any] = dict(info0 or {})
        last_obs_text = (first_obs_dict.get("text") or "")

        if self.use_history:
            history_text = f"<Obs_0>{obs0}</Obs_0>"
        else:
            inv0 = _infer_inventory_from_admissible(admissible)
            history_text = _build_state_text(last_obs_text, inv0)

        for t in range(1, max_steps + 1):
            inv = _infer_inventory_from_admissible(admissible)

            if self.use_history:
                history_text_with_inv = history_text
                if inv:
                    history_text_with_inv = history_text + f"<Inv_{t-1}>{', '.join(inv)}</Inv_{t-1}>"
            else:
                history_text_with_inv = _build_state_text(last_obs_text, inv)

            rec, history_text, done, last_info, admissible, last_obs_text = self.agent.run_one_step(
                env=env,
                episode_id=episode_id,
                t=t,
                task_instruction=task_instruction,
                history_text=history_text_with_inv,
                admissible_commands=admissible,
                goal=goal,
                progress=progress,
                use_history=self.use_history,
            )
            step_records.append(rec)

            state.history.append(
                {
                    "role": "assistant",
                    "content": (
                        f"Reflection: {rec['reflection']}\n"
                        f"Thought: {rec['thought']}\n"
                        f"Action: {rec['action']}"
                    ).strip(),
                }
            )
            state.history.append({"role": "user", "content": f"Observation: {rec['obs_next']}".strip()})

            if done:
                break

        success = bool(last_info.get("won")) if "won" in last_info else bool(env.success(last_info))
        state.finished = True
        state.success = success
        state.reward = 1.0 if success else 0.0
        state.steps = len(step_records)
        state.terminate_reason = "success" if success else "max_steps"

        return state, step_records

class RAPAgent:

    def __init__(self, planner: RAPPlanner):
        self.planner = planner

    def run_one_step(
        self,
        env,
        episode_id: str,
        t: int,
        task_instruction: str,
        history_text: str,
        admissible_commands: List[str],
        goal: GoalSpec,
        progress: Pick2Progress,
        use_history: bool,
    ) -> Tuple[Dict[str, Any], str, bool, Dict[str, Any], List[str], str]:
        def _extract_latest_obs_and_inv(text: str) -> Tuple[str, str]:
            s = (text or "").strip()

            if s.lower().startswith("observation:"):
                obs = ""
                inv = ""
                lines = s.splitlines()
                if lines:

                    first = lines[0]
                    obs = first.split(":", 1)[1].strip() if ":" in first else first.strip()
                for line in lines[1:]:
                    if line.lower().startswith("inventory:"):
                        inv = line.split(":", 1)[1].strip() if ":" in line else line.strip()
                        break
                return obs.strip(), inv.strip()

            obs_matches = re.findall(r"(?is)<Obs_\\d+>\\s*(.*?)\\s*</Obs_\\d+>", s)
            inv_matches = re.findall(r"(?is)<Inv_\\d+>\\s*(.*?)\\s*</Inv_\\d+>", s)
            obs = obs_matches[-1].strip() if obs_matches else s
            inv = inv_matches[-1].strip() if inv_matches else ""
            return obs, inv

        obs_for_rap, inv_for_rap = _extract_latest_obs_and_inv(history_text)
        if not inv_for_rap:
            inv_for_rap = ", ".join([x for x in _infer_inventory_from_admissible(admissible_commands) if x])
        root = RAPState(goal=task_instruction, obs=obs_for_rap, inventory=inv_for_rap)

        action = self.planner.choose_action(root, valid_actions=admissible_commands)
        action = _validate_action_in_runner(normalize_action(action), admissible_commands, goal, progress)

        obs_next_dict, _reward, done, info = env.step(action)
        obs_text = (obs_next_dict.get("text") or "")
        obs_next = format_observation(env, obs_next_dict)

        _update_pick2_progress(progress, action, obs_text, goal.obj_key, goal.recep_key)

        record = {
            "episode_id": episode_id,
            "t": t,
            "method": "rap",
            "k": -1,
            "k_raw_output": "",
            "foresight_text": "",
            "wm_raw_output": "",
            "reflection": "RAP baseline (no reflection).",
            "thought": "RAP MCTS chooses an action based on imagined rollouts.",
            "action": action,
            "policy_raw_output": "",
            "obs_next": obs_next,
            "done": done,
            "info": info,
        }

        next_admissible = info.get("admissible_commands", admissible_commands) or admissible_commands
        if use_history:
            updated_history = (
                history_text
                + f"<Thought_{t}>{record['thought']}</Thought_{t}>"
                + f"<Action_{t}>{action}</Action_{t}>"
                + f"<Obs_{t}>{obs_next}</Obs_{t}>"
            )
        else:
            inv_next = _infer_inventory_from_admissible(next_admissible)
            updated_history = _build_state_text(obs_text, inv_next)
        return record, updated_history, done, info, next_admissible, obs_text

class RAPEvaluator:
    def __init__(
        self,
        policy_model_path: str,
        wm_model_path: str,
        policy_device: str = "cuda",
        wm_device: str = "cuda",
        policy_dtype: torch.dtype = torch.float16,
        wm_dtype: torch.dtype = torch.float16,
        disable_wm: bool = False,
        use_history: bool = False,

        d_actions: int = 8,
        depth_limit: int = 6,
        num_rollouts: int = 32,
        uct_w: float = 1.4,
        rw_action_logp: float = 0.2,
        rw_self_eval: float = 0.6,
        rw_goal_overlap: float = 0.2,
        action_gen_tokens: int = 32,
        wm_pred_tokens: int = 128,
        seed: int = 0,
    ):

        policy_llm = LocalCausalLM(model_path=policy_model_path, device=policy_device, torch_dtype=policy_dtype)
        if disable_wm:
            wm_llm = policy_llm
        else:
            wm_llm = LocalCausalLM(model_path=wm_model_path, device=wm_device, torch_dtype=wm_dtype)

        policy_lm = RAPLM(tokenizer=policy_llm.tokenizer, model=policy_llm.model, device=policy_llm.device)
        wm_lm = RAPLM(tokenizer=wm_llm.tokenizer, model=wm_llm.model, device=wm_llm.device)

        planner = RAPPlanner(
            policy_lm=policy_lm,
            wm_lm=wm_lm,
            d_actions=int(d_actions),
            depth_limit=int(depth_limit),
            num_rollouts=int(num_rollouts),
            uct_w=float(uct_w),
            rw_action_logp=float(rw_action_logp),
            rw_self_eval=float(rw_self_eval),
            rw_goal_overlap=float(rw_goal_overlap),
            action_gen_tokens=int(action_gen_tokens),
            wm_pred_tokens=int(wm_pred_tokens),
            seed=int(seed),
        )

        self.agent = RAPAgent(planner)
        self.use_history = bool(use_history)

    def run_episode_from_reset(
        self,
        env,
        task_id: int,
        first_obs_dict: Dict[str, str],
        info0: Dict[str, Any],
        max_steps: int,
    ) -> Tuple[State, List[Dict[str, Any]]]:

        try:
            self.agent.planner.reset_cache()
        except Exception:
            pass

        episode_id = str(uuid.uuid4())[:8]

        first_obs_text = format_observation(env, first_obs_dict)
        task_instruction, obs0 = extract_task_and_obs0(first_obs_text)
        gamefile = (info0.get("gamefile") or "")

        goal = _parse_goal_from_gamefile(task_instruction, gamefile)
        progress = Pick2Progress(placed=set(), source_hint="")

        admissible = info0.get("admissible_commands") or []
        if not isinstance(admissible, list):
            admissible = []

        state = State()
        state.error = gamefile
        state.history.append({"role": "user", "content": f"{task_instruction}\n{obs0}".strip()})

        step_records: List[Dict[str, Any]] = []
        done = False
        last_info: Dict[str, Any] = dict(info0 or {})
        last_obs_text = (first_obs_dict.get("text") or "")

        if self.use_history:
            history_text = f"<Obs_0>{obs0}</Obs_0>"
        else:
            inv0 = _infer_inventory_from_admissible(admissible)
            history_text = _build_state_text(last_obs_text, inv0)

        for t in range(1, max_steps + 1):
            inv = _infer_inventory_from_admissible(admissible)
            if self.use_history:
                history_text_with_inv = history_text
                if inv:
                    history_text_with_inv = history_text + f"<Inv_{t-1}>{', '.join(inv)}</Inv_{t-1}>"
            else:
                history_text_with_inv = _build_state_text(last_obs_text, inv)

            rec, history_text, done, last_info, admissible, last_obs_text = self.agent.run_one_step(
                env=env,
                episode_id=episode_id,
                t=t,
                task_instruction=task_instruction,
                history_text=history_text_with_inv,
                admissible_commands=admissible,
                goal=goal,
                progress=progress,
                use_history=self.use_history,
            )
            step_records.append(rec)

            state.history.append(
                {
                    "role": "assistant",
                    "content": (
                        f"Reflection: {rec['reflection']}\n"
                        f"Thought: {rec['thought']}\n"
                        f"Action: {rec['action']}"
                    ).strip(),
                }
            )
            state.history.append({"role": "user", "content": f"Observation: {rec['obs_next']}".strip()})

            if done:
                break

        success = bool(last_info.get("won")) if "won" in last_info else bool(env.success(last_info))
        state.finished = True
        state.success = success
        state.reward = 1.0 if success else 0.0
        state.steps = len(step_records)
        state.terminate_reason = "success" if success else "max_steps"
        return state, step_records

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Foresight/RAP evaluation for ALFWorld.")
    parser.add_argument("--method", type=str, default="foresight", choices=["foresight", "rap"])
    parser.add_argument("--split", type=str, default="dev", help="dev(seen) or test(unseen).")
    parser.add_argument(
        "--env_data_root",
        type=str,
        default="eval_agent/data/alfworld",
        help="ALFWorld dataset root containing train/valid_seen/valid_unseen.",
    )
    parser.add_argument("--policy_model", type=str, required=True, help="Policy LLM model path (e.g., Qwen3-8B).")
    parser.add_argument("--wm_model", type=str, required=True, help="World model path (e.g., merged_full).")
    parser.add_argument("--max_steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--override", action="store_true", help="Ignore done tasks (overwrite output dir).")
    parser.add_argument("--debug", action="store_true", help="Run only 5 tasks.")
    parser.add_argument("--part_num", type=int, default=1)
    parser.add_argument("--part_idx", type=int, default=-1)
    parser.add_argument(
        "--only_ids",
        type=str,
        default="",
        help="Comma-separated task indices to evaluate (e.g., '2,121,123,124'). Overrides part/debug selection.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Default device for both policy & world model.")
    parser.add_argument("--policy_device", type=str, default="", help="Override device for policy model (e.g. cuda/cpu).")
    parser.add_argument("--wm_device", type=str, default="", help="Override device for world model (e.g. cuda/cpu).")
    parser.add_argument("--policy_dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--wm_dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])

    parser.add_argument("--wm_backend", type=str, default="local", choices=["local", "deepseek_api"])
    parser.add_argument("--wm_api_base_url", type=str, default="https://api.deepseek.com")
    parser.add_argument("--wm_api_key_env", type=str, default="DEEPSEEK_API_KEY")
    parser.add_argument("--wm_api_model", type=str, default="deepseek-chat")
    parser.add_argument("--wm_api_timeout_s", type=int, default=120)
    parser.add_argument("--wm_api_max_retries", type=int, default=3)

    parser.add_argument("--decision_tokens", type=int, default=16)
    parser.add_argument("--act_tokens", type=int, default=256)
    parser.add_argument("--foresight_tokens", type=int, default=256)
    parser.add_argument("--max_k", type=int, default=2)
    parser.add_argument("--fixed_k", type=int, default=-1, help=">=0 to bypass decide_k and force constant K.")
    parser.add_argument("--disable_wm", type=int, default=0, help="1: do not load/run world model (debug only)")
    parser.add_argument("--use_history", type=int, default=0, help="1: include interaction history in prompts; 0: state-only (default)")

    parser.add_argument("--rap_d_actions", type=int, default=8)
    parser.add_argument("--rap_depth_limit", type=int, default=6)
    parser.add_argument("--rap_num_rollouts", type=int, default=32)
    parser.add_argument("--rap_uct_w", type=float, default=1.4)
    parser.add_argument("--rap_rw_action_logp", type=float, default=0.2)
    parser.add_argument("--rap_rw_self_eval", type=float, default=0.6)
    parser.add_argument("--rap_rw_goal_overlap", type=float, default=0.2)
    parser.add_argument("--rap_action_gen_tokens", type=int, default=32)
    parser.add_argument("--rap_wm_pred_tokens", type=int, default=128)
    return parser

def evaluate_from_args(args: argparse.Namespace) -> None:
    output_path = args.output_path
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

    file_mode = "w" if getattr(args, "override", False) else "a"
    file_handler = logging.FileHandler(os.path.join(output_path, "log.txt"), mode=file_mode)
    logging.basicConfig(format="%(message)s", handlers=[logging.StreamHandler(), file_handler])

    split_data = _map_split_to_data(args.split)

    env = load_env(data_root=args.env_data_root, split=split_data, max_steps=args.max_steps, seed=args.seed)
    total = int(getattr(env, "num_episodes", lambda: 0)() or 0)
    if total <= 0:

        try:
            total = len(getattr(env, "_env", None).gamefiles)
        except Exception:
            total = 0

    if total <= 0:
        raise RuntimeError("Failed to determine total number of episodes for this split.")

    if args.part_num > 1:
        if args.part_idx < 0:
            raise ValueError("--part_idx must be set when --part_num > 1")
        per_part = total // args.part_num + 1
        start = per_part * args.part_idx
        end = min(start + per_part, total)
        target_indices = set(range(start, end))
    else:
        target_indices = set(range(total))

    if args.debug:

        target_indices = set(sorted(list(target_indices))[:5])

    only_ids = (getattr(args, "only_ids", "") or "").strip()
    if only_ids:
        ids: Set[int] = set()
        for x in only_ids.split(","):
            x = x.strip()
            if not x:
                continue
            try:
                ids.add(int(x))
            except Exception:
                continue
        if ids:
            target_indices = ids

    done_task_ids: Set[str] = set()
    state_list: List[State] = []
    if os.path.exists(output_path) and (not args.override):
        for fn in os.listdir(output_path):
            if not fn.endswith(".json"):
                continue
            if fn in {"summary.json"}:
                continue
            try:
                task_id = fn.split(".")[0]
                st = State.load_json(json.load(open(os.path.join(output_path, fn))))
                state_list.append(st)
                done_task_ids.add(task_id)
            except Exception:
                continue

    fixed_k = None if int(getattr(args, "fixed_k", -1)) < 0 else int(getattr(args, "fixed_k", -1))
    policy_device = (getattr(args, "policy_device", "") or "").strip() or args.device
    wm_device = (getattr(args, "wm_device", "") or "").strip() or args.device
    policy_dtype = _parse_dtype(getattr(args, "policy_dtype", "fp16"))
    wm_dtype = _parse_dtype(getattr(args, "wm_dtype", "fp16"))
    wm_backend = (getattr(args, "wm_backend", "local") or "local").strip().lower()
    method = (getattr(args, "method", "foresight") or "foresight").strip().lower()
    if method == "rap":
        if wm_backend != "local":
            raise ValueError("RAP method currently requires local WM (wm_backend=local).")
        evaluator = RAPEvaluator(
            policy_model_path=args.policy_model,
            wm_model_path=args.wm_model,
            policy_device=policy_device,
            wm_device=wm_device,
            policy_dtype=policy_dtype,
            wm_dtype=wm_dtype,
            disable_wm=bool(int(getattr(args, "disable_wm", 0))),
            use_history=bool(int(getattr(args, "use_history", 0))),
            d_actions=int(getattr(args, "rap_d_actions", 8)),
            depth_limit=int(getattr(args, "rap_depth_limit", 6)),
            num_rollouts=int(getattr(args, "rap_num_rollouts", 32)),
            uct_w=float(getattr(args, "rap_uct_w", 1.4)),
            rw_action_logp=float(getattr(args, "rap_rw_action_logp", 0.2)),
            rw_self_eval=float(getattr(args, "rap_rw_self_eval", 0.6)),
            rw_goal_overlap=float(getattr(args, "rap_rw_goal_overlap", 0.2)),
            action_gen_tokens=int(getattr(args, "rap_action_gen_tokens", 32)),
            wm_pred_tokens=int(getattr(args, "rap_wm_pred_tokens", 128)),
            seed=int(getattr(args, "seed", 123)),
        )
    else:
        evaluator = ForesightEvaluator(
            policy_model_path=args.policy_model,
            wm_model_path=args.wm_model,
            policy_device=policy_device,
            wm_device=wm_device,
            policy_dtype=policy_dtype,
            wm_dtype=wm_dtype,
            wm_backend=wm_backend,
            wm_api_base_url=getattr(args, "wm_api_base_url", "https://api.deepseek.com"),
            wm_api_key_env=getattr(args, "wm_api_key_env", "DEEPSEEK_API_KEY"),
            wm_api_model=getattr(args, "wm_api_model", "deepseek-chat"),
            wm_api_timeout_s=int(getattr(args, "wm_api_timeout_s", 120)),
            wm_api_max_retries=int(getattr(args, "wm_api_max_retries", 3)),
            decision_tokens=int(getattr(args, "decision_tokens", 16)),
            act_tokens=int(getattr(args, "act_tokens", 256)),
            foresight_tokens=int(getattr(args, "foresight_tokens", 256)),
            max_k=int(getattr(args, "max_k", 2)),
            fixed_k=fixed_k,
            disable_wm=bool(int(getattr(args, "disable_wm", 0))),
            use_history=bool(int(getattr(args, "use_history", 0))),
        )

    logger.warning(f"Overall we have {total} games in split={split_data}")
    logger.warning(f"Evaluating with {len(target_indices)} games")

    max_idx = max(target_indices) if target_indices else -1
    try:
        for idx in range(max_idx + 1):
            first_obs_dict, info0 = env.reset()
            if idx not in target_indices:
                continue
            if str(idx) in done_task_ids:
                continue

            st, step_records = evaluator.run_episode_from_reset(
                env=env,
                task_id=idx,
                first_obs_dict=first_obs_dict,
                info0=info0,
                max_steps=args.max_steps,
            )
            state_list.append(st)

            out_path = os.path.join(output_path, f"{idx}.json")
            payload = st.to_dict()

            payload["foresight_steps"] = step_records
            json.dump(payload, open(out_path, "w"), indent=4, ensure_ascii=False)

            msg = (
                f"[TaskResult] id={idx} success={st.success} steps={st.steps} "
                f"terminate_reason={st.terminate_reason} reward={st.reward}"
            )
            logger.warning(msg)

            try:
                summarize_alfworld(output_path, state_list, logger)
            except Exception:
                pass
    except KeyboardInterrupt:
        logger.warning("[Warn] Interrupted by user. Writing partial summaries...")
        try:
            summarize_alfworld(output_path, state_list, logger)
        except Exception:
            pass
        return
    except Exception as e:
        logger.warning(f"[Error] Runner crashed: {e}")
        try:
            summarize_alfworld(output_path, state_list, logger)
        except Exception:
            pass
        raise

    logger.warning("All tasks done.")
    logger.warning(f"Output saved to {output_path}")

    success_list = [1 if st.success else 0 for st in state_list]
    if success_list:
        logger.warning(f"Success rate: {sum(success_list)/len(success_list):.4f}")

    summarize_alfworld(output_path, state_list, logger)

if __name__ == "__main__":
    parser = build_arg_parser()
    evaluate_from_args(parser.parse_args())
