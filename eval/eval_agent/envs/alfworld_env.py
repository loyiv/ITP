import re
import json
import logging
import os
import difflib
from typing import Any, Dict, List, Tuple, Set

from eval_agent.envs import BaseEnv
from eval_agent.tasks import AlfWorldTask
from eval_agent.prompt import prompt_with_icl
from eval_agent.utils.datatypes import State

logger = logging.getLogger("agent_frame")

def process_ob(ob):
    if ob.startswith('You arrive at loc '):
        ob = ob[ob.find('. ')+2:]
    return ob

class AlfWorldEnv(BaseEnv):
    def __init__(
        self,
        task: AlfWorldTask,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.task: AlfWorldTask = task
        self.env = task.env
        self.state = State()
        self._last_admissible: List[str] = []
        self._goal_text: str = ""
        self._inventory: List[str] = []
        self._goal_obj: str = ""
        self._goal_recep: str = ""
        self._goal_requires_clean: bool = False
        self._task_obj_key: str = ""
        self._task_recep_key: str = ""
        self._task_type: str = ""
        self._need_two: bool = False
        self._did_clean: bool = False
        self._did_heat: bool = False
        self._did_cool: bool = False

        self.append_admissible: bool = bool(kwargs.get("append_admissible", True))
        self.admissible_on_invalid_only: bool = bool(kwargs.get("admissible_on_invalid_only", True))
        self.admissible_top_k: int = int(kwargs.get("admissible_top_k", 30))
        self.auto_correct_to_admissible: bool = bool(kwargs.get("auto_correct_to_admissible", True))
        self.force_admissible_action: bool = bool(kwargs.get("force_admissible_action", False))
        self.heuristic_fallback: bool = bool(kwargs.get("heuristic_fallback", False))
        self.prefer_heuristic_action: bool = bool(kwargs.get("prefer_heuristic_action", False))
        self.prefer_goal_action: bool = bool(kwargs.get("prefer_goal_action", False))
        self._opened: set = set()
        self._visited: set = set()
        self._used: set = set()
        self._current_loc: str = ""

        self._placed_goal_instances: Set[str] = set()

        self._goal_source_hint: str = ""

    def _norm(self, s: str) -> str:
        return re.sub(r"\s+", " ", (s or "").strip().lower())

    def _update_goal_progress_from_action_observation(self, action: str, ob: str) -> None:
        if not action or not ob:
            return
        goal_obj = (self._task_obj_key or self._goal_obj or "").lower()
        goal_recep = (self._task_recep_key or self._goal_recep or "").lower()
        if not goal_obj or not goal_recep:
            return

        a = self._norm(action)
        t = (ob or "").strip().lower()

        if a.startswith("take "):
            m = re.match(r"^take (.+?) from (.+)$", a)
            if m and ("you pick up the" in t):
                obj = self._norm(m.group(1))
                src = self._norm(m.group(2))
                if goal_obj in obj and src and (goal_recep not in src):
                    self._goal_source_hint = src

                if goal_obj in obj and goal_recep in src:
                    self._placed_goal_instances.discard(obj)
            return

        if a.startswith("put "):
            m = re.match(r"^put (.+?) in/on (.+)$", a)
            if m and ("you put the" in t):
                obj = self._norm(m.group(1))
                dst = self._norm(m.group(2))
                if goal_obj in obj and goal_recep in dst:
                    self._placed_goal_instances.add(obj)
            return

    def _format_admissible(self, admissible: List[str]) -> str:
        if not admissible:
            return ""
        k = max(0, self.admissible_top_k)
        items = admissible if k == 0 else admissible[:k]
        lines = "\n".join(f"- {a}" for a in items)
        return "\nAdmissible commands:\n" + lines

    def _format_goal_and_inventory(self) -> str:
        parts: List[str] = []
        if self._goal_text:
            parts.append(f"Goal: {self._goal_text}")
        if self._inventory:
            parts.append("Inventory: " + ", ".join(self._inventory))
        else:
            parts.append("Inventory: (empty)")
        return "\n" + "\n".join(parts)

    def _parse_goal(self, goal_text: str) -> None:
        text = (goal_text or "").strip().lower()
        self._goal_obj = ""
        self._goal_recep = ""
        self._goal_requires_clean = "clean" in text

        m = re.search(r"put (?:a |an )?(?:clean |hot |cool )?([a-z0-9_]+) in ([a-z0-9_]+)", text)
        if m:
            self._goal_obj = m.group(1)
            self._goal_recep = m.group(2)
            return
        m = re.search(r"put two ([a-z0-9_]+) in ([a-z0-9_]+)", text)
        if m:
            self._goal_obj = m.group(1)
            self._goal_recep = m.group(2)
            return
        m = re.search(r"find two ([a-z0-9_]+).*put (?:them|it) in ([a-z0-9_]+)", text)
        if m:
            self._goal_obj = m.group(1)
            self._goal_recep = m.group(2)
            return

    def _parse_task_from_gamefile(self, game_file: str) -> None:
        self._task_type = ""
        self._task_obj_key = ""
        self._task_recep_key = ""
        self._need_two = False
        if not game_file:
            return
        try:

            task_dir = os.path.basename(os.path.dirname(os.path.dirname(game_file)))
            parts = task_dir.split("-")
            if not parts:
                return
            self._task_type = parts[0]

            if len(parts) >= 4:
                self._task_obj_key = parts[1].strip().lower()
                self._task_recep_key = parts[3].strip().lower()
            if "pick_two" in self._task_type or "two_obj" in self._task_type or "puttwo" in self._task_type:
                self._need_two = True
            if self._task_type.startswith("pick_two_obj"):
                self._need_two = True
        except Exception:
            return

    def _is_container(self, name: str) -> bool:
        n = (name or "").lower()
        return any(x in n for x in ["cabinet", "drawer", "fridge", "microwave"])

    def _heuristic_action(self, admissible: List[str]) -> str:
        if not admissible:
            return ""

        goal_obj = (self._task_obj_key or self._goal_obj or "").lower()
        goal_recep = (self._task_recep_key or self._goal_recep or "").lower()
        need_clean = bool(self._goal_requires_clean or ("clean" in (self._task_type or "")))
        need_heat = "heat" in (self._task_type or "") or "heat" in (self._goal_text or "").lower()
        need_cool = "cool" in (self._task_type or "") or "cool" in (self._goal_text or "").lower()
        need_examine = "look_at" in (self._task_type or "") or "examine" in (self._goal_text or "").lower()
        need_two = self._need_two
        placed_n = len(self._placed_goal_instances) if need_two else 0

        held_lower = [x.lower() for x in self._inventory]
        holds_goal = any(goal_obj and goal_obj in x for x in held_lower)

        if ("look_at_obj_in_light" in (self._task_type or "")) and admissible:
            for a in admissible:
                if a.startswith("use ") and "desklamp" in a.lower() and a not in self._used:
                    return a

        if need_examine and goal_obj:
            for a in admissible:
                if a.startswith("examine ") and goal_obj in a.lower():
                    return a

        if goal_obj and need_clean and not self._did_clean:
            for a in admissible:
                if a.startswith("clean ") and goal_obj in a.lower():
                    return a
        if goal_obj and need_heat and not self._did_heat:
            for a in admissible:
                if a.startswith("heat ") and goal_obj in a.lower():
                    return a
        if goal_obj and need_cool and not self._did_cool:
            for a in admissible:
                if a.startswith("cool ") and goal_obj in a.lower():
                    return a

        if goal_obj and goal_recep:

            if (not need_clean or self._did_clean) and (not need_heat or self._did_heat) and (not need_cool or self._did_cool):
                for a in admissible:
                    if a.startswith("put ") and goal_obj in a.lower() and goal_recep in a.lower():
                        return a

        if goal_obj:
            take_cands = [a for a in admissible if a.startswith("take ") and goal_obj in a.lower()]
            if take_cands:
                if need_two and goal_recep and placed_n >= 1 and (not holds_goal):
                    safe = [a for a in take_cands if goal_recep not in a.lower()]
                    if safe:
                        return safe[0]

                    if self._goal_source_hint:
                        for a in admissible:
                            if a.startswith("go to ") and self._goal_source_hint in a.lower():
                                return a

                else:
                    return take_cands[0]

        if holds_goal and need_clean and not self._did_clean:
            for a in admissible:
                if a.startswith("go to ") and "sinkbasin" in a.lower():
                    return a
        if holds_goal and need_heat and not self._did_heat:
            for a in admissible:
                if a.startswith("go to ") and "microwave" in a.lower():
                    return a
        if holds_goal and need_cool and not self._did_cool:
            for a in admissible:
                if a.startswith("go to ") and "fridge" in a.lower():
                    return a
        if holds_goal and goal_recep and ((not need_clean or self._did_clean) and (not need_heat or self._did_heat) and (not need_cool or self._did_cool)):
            for a in admissible:
                if a.startswith("go to ") and goal_recep in a.lower():
                    return a

        if need_two and placed_n >= 1 and (not holds_goal) and self._goal_source_hint:
            for a in admissible:
                if a.startswith("go to ") and self._goal_source_hint in a.lower():
                    return a

        open_cands = [a for a in admissible if a.startswith("open ")]
        for a in open_cands:
            target = a[len("open "):].strip().lower()
            if target and target not in self._opened:
                return a

        go_cands = [a for a in admissible if a.startswith("go to ")]

        for a in go_cands:
            dest = a[len("go to "):].strip().lower()
            if self._is_container(dest) and dest not in self._visited:
                return a
        for a in go_cands:
            dest = a[len("go to "):].strip().lower()
            if dest and dest not in self._visited:
                return a

        for v in ["look", "inventory"]:
            if v in admissible:
                return v
        for a in admissible:
            if a.startswith("examine "):
                return a

        return admissible[0]

    def _rewrite_action_in_llm_output(self, llm_output: str, final_action: str) -> str:
        text = (llm_output or "").strip()
        if not text:
            return f"Thought: \nAction: {final_action}"

        lines = text.splitlines()
        last_action_idx = None
        for i in range(len(lines) - 1, -1, -1):
            if re.match(r"^\s*Action\s*:", lines[i], flags=re.IGNORECASE):
                last_action_idx = i
                break
        if last_action_idx is not None:
            lines[last_action_idx] = f"Action: {final_action}"
            return "\n".join(lines)

        return text + f"\nAction: {final_action}"

    def _should_override_with_goal_action(self, best: str, model_action: str) -> bool:
        if not best or best == model_action:
            return False
        b = best.lower()
        a = (model_action or "").lower()
        goal_obj = (self._task_obj_key or self._goal_obj or "").lower()
        goal_recep = (self._task_recep_key or self._goal_recep or "").lower()
        task_type = (self._task_type or "").lower()
        need_two = bool(self._need_two)
        placed_n = len(self._placed_goal_instances) if need_two else 0

        if b.startswith(("put ", "clean ", "heat ", "cool ", "examine ")):
            return True

        if b.startswith("take ") and goal_obj and goal_obj in b:

            if need_two and placed_n >= 1 and goal_recep and goal_recep in b:
                return False
            return True

        if "look_at_obj_in_light" in task_type and b.startswith("use ") and "desklamp" in b:
            return True

        if b.startswith("go to ") and any(x in b for x in ["sinkbasin", "microwave", "fridge"]):
            return True
        if b.startswith("go to ") and goal_recep and goal_recep in b:
            return True

        if b.startswith("open ") and any(x in b for x in ["fridge", "microwave"]):
            return True
        return False

    def _compute_goal_suggestions(self, admissible: List[str]) -> List[str]:
        if not admissible:
            return []
        inv = list(self._inventory)

        goal_obj = (self._task_obj_key or self._goal_obj or "").lower()
        goal_recep = (self._task_recep_key or self._goal_recep or "").lower()
        task_type = (self._task_type or "").lower()
        need_two = bool(self._need_two)
        placed_n = len(self._placed_goal_instances) if need_two else 0
        requires_clean = bool(self._goal_requires_clean or ("clean" in task_type))
        requires_heat = ("heat" in task_type) or ("heat" in (self._goal_text or "").lower())
        requires_cool = ("cool" in task_type) or ("cool" in (self._goal_text or "").lower())

        suggestions: List[str] = []

        held_match = None
        for x in inv:
            if goal_obj and goal_obj in x.lower():
                held_match = x
                break

        if held_match and requires_clean and not self._did_clean:

            has_clean = False
            for a in admissible:
                if a.startswith("clean ") and held_match.lower() in a.lower():
                    suggestions.append(a)
                    has_clean = True
                    break
            if not has_clean:

                for a in admissible:
                    if a.startswith("go to sinkbasin"):
                        suggestions.append(a)
                        break
        if held_match and requires_heat and not self._did_heat:
            for a in admissible:
                if a.startswith("heat ") and held_match.lower() in a.lower():
                    suggestions.append(a)
                    break
            if not any(x.startswith("heat ") for x in suggestions):
                for a in admissible:
                    if a.startswith("go to ") and "microwave" in a.lower():
                        suggestions.append(a)
                        break
        if held_match and requires_cool and not self._did_cool:
            for a in admissible:
                if a.startswith("cool ") and held_match.lower() in a.lower():
                    suggestions.append(a)
                    break
            if not any(x.startswith("cool ") for x in suggestions):
                for a in admissible:
                    if a.startswith("go to ") and "fridge" in a.lower():
                        suggestions.append(a)
                        break

        if held_match and goal_recep and ((not requires_clean or self._did_clean) and (not requires_heat or self._did_heat) and (not requires_cool or self._did_cool)):
            has_put = False
            for a in admissible:
                if a.startswith("put ") and held_match.lower() in a.lower() and goal_recep in a.lower():
                    suggestions.append(a)
                    has_put = True
                    break
            if not has_put:

                for a in admissible:
                    if a.startswith("go to ") and goal_recep in a.lower():
                        suggestions.append(a)
                        break

        if need_two and placed_n >= 1 and (not held_match) and self._goal_source_hint:
            for a in admissible:
                if a.startswith("go to ") and self._goal_source_hint in a.lower():
                    suggestions.append(a)
                    break
        if not held_match and goal_obj:
            for a in admissible:
                if a.startswith("take ") and goal_obj in a:
                    if need_two and placed_n >= 1 and goal_recep and goal_recep in a:
                        continue
                    suggestions.append(a)
                    if len(suggestions) >= 2:
                        break

        if not suggestions:

            surface_priority = [
                "countertop", "diningtable", "desk", "sidetable", "shelf",
                "dresser", "bed", "stoveburner", "toilet", "sinkbasin",
            ]
            go_cands = [a for a in admissible if a.startswith("go to ")]

            if self._current_loc:
                go_cands = [a for a in go_cands if self._current_loc not in a.lower()]
            for key in surface_priority:
                for a in go_cands:
                    if key in a.lower():
                        suggestions.append(a)
                        break
                if suggestions:
                    break
            if not suggestions and go_cands:
                suggestions.append(go_cands[0])

        out: List[str] = []
        for a in suggestions:
            if a and a not in out:
                out.append(a)
        return out[:3]

    def _format_suggestions(self, sugg: List[str]) -> str:
        if not sugg:
            return ""
        lines = "\n".join(f"- {a}" for a in sugg)
        return "\nSuggested actions (goal-directed):\n" + lines

    def _update_state_from_observation(self, ob: str) -> None:
        text = (ob or "").strip()

        m = re.match(r"On the (.+?), you see", text)
        if m:
            self._current_loc = m.group(1).strip().lower()
        else:
            m = re.match(r"The (.+?) is (?:closed|open)\\.", text)
            if m:
                self._current_loc = m.group(1).strip().lower()

        m = re.search(r"You pick up the (.+?) from the (.+?)\.", text)
        if m:
            obj = m.group(1).strip()
            if obj and obj not in self._inventory:
                self._inventory.append(obj)
            return

        m = re.search(r"You clean the (.+?) using the (.+?)\.", text)
        if m and (self._task_obj_key or self._goal_obj):
            obj = m.group(1).strip().lower()
            key = (self._task_obj_key or self._goal_obj or "").lower()
            if key and key in obj:
                self._did_clean = True
            return
        m = re.search(r"You heat the (.+?) using the (.+?)\.", text)
        if m and (self._task_obj_key or self._goal_obj):
            obj = m.group(1).strip().lower()
            key = (self._task_obj_key or self._goal_obj or "").lower()
            if key and key in obj:
                self._did_heat = True
            return
        m = re.search(r"You cool the (.+?) using the (.+?)\.", text)
        if m and (self._task_obj_key or self._goal_obj):
            obj = m.group(1).strip().lower()
            key = (self._task_obj_key or self._goal_obj or "").lower()
            if key and key in obj:
                self._did_cool = True
            return

        m = re.search(r"You put the (.+?) (?:in|on) the (.+?)\.", text)
        if m:
            obj = m.group(1).strip()
            if obj in self._inventory:
                self._inventory = [x for x in self._inventory if x != obj]
            return

        m = re.search(r"You open the (.+?)\.", text)
        if m:
            rec = m.group(1).strip().lower()
            if rec:
                self._opened.add(rec)
            return

        m = re.search(r"You use the (.+?)\.", text)
        if m:
            obj = m.group(1).strip()
            if obj:
                self._used.add(f"use {obj}".lower())
            return

    def _infer_inventory_from_admissible(self, admissible: List[str]) -> None:
        objs: List[str] = []
        for a in admissible or []:
            if not isinstance(a, str):
                continue
            if a.startswith("put "):
                rest = a[len("put "):]

                obj = rest.split(" in/on ", 1)[0].strip()
                if obj and obj not in objs:
                    objs.append(obj)

        self._inventory = objs

    def _maybe_autocorrect(self, action: str) -> str:
        if not self.auto_correct_to_admissible:
            return action
        if not self._last_admissible:
            return action
        if action in self._last_admissible:
            return action

        a_low = action.lower()
        if a_low.startswith("put "):
            return action
        if a_low.startswith("go to "):
            candidates = [x for x in self._last_admissible if isinstance(x, str) and x.startswith("go to ")]
        elif a_low.startswith("take "):
            candidates = [x for x in self._last_admissible if isinstance(x, str) and x.startswith("take ")]
        elif a_low.startswith("open "):
            candidates = [x for x in self._last_admissible if isinstance(x, str) and x.startswith("open ")]
        elif a_low.startswith("close "):
            candidates = [x for x in self._last_admissible if isinstance(x, str) and x.startswith("close ")]
        elif a_low.startswith("toggle "):
            candidates = [x for x in self._last_admissible if isinstance(x, str) and x.startswith("toggle ")]
        elif a_low.startswith("clean "):
            candidates = [x for x in self._last_admissible if isinstance(x, str) and x.startswith("clean ")]
        elif a_low.startswith("heat "):
            candidates = [x for x in self._last_admissible if isinstance(x, str) and x.startswith("heat ")]
        elif a_low.startswith("cool "):
            candidates = [x for x in self._last_admissible if isinstance(x, str) and x.startswith("cool ")]
        else:
            candidates = list(self._last_admissible)

        m = difflib.get_close_matches(action, candidates, n=1, cutoff=0.75)
        if m:
            return m[0]
        return action

    def parse_action(self, llm_output: str) -> str:
        text = (llm_output or "").strip()

        matches = re.findall(r"(?im)^\s*Action\s*:\s*(.*)$", text)
        if not matches:

            matches = re.findall(r"Action\s*:\s*(.*)", text, flags=re.IGNORECASE | re.DOTALL)
        if not matches:
            raise ValueError("Missing 'Action:' in llm_output")

        action = matches[-1]

        action = action.splitlines()[0]

        action = action.strip().strip("`\"'").strip()

        action = re.sub(r"(</s>|<\|end\|>|<\|eot_id\|>|<\|endoftext\|>)", "", action, flags=re.IGNORECASE).strip()
        action = re.sub(r"</?\w+>", "", action).strip()

        action = re.split(r"(?i)\bThought\s*:\s*|\bObservation\s*:\s*|\bAction\s*:\s*", action, maxsplit=1)[0].strip()

        action = re.sub(r"\s+", " ", action).strip()

        if action.lower().startswith("task "):
            action = "take " + action[5:].lstrip()

        if action.lower().startswith("put ") and " in/on " not in action.lower():

            action = re.sub(r"\\s+in\\s+", " in/on ", action, flags=re.IGNORECASE)
            action = re.sub(r"\\s+on\\s+", " in/on ", action, flags=re.IGNORECASE)

        if action.endswith("."):
            action = action[:-1].strip()
        return action

    def conduct_action(self, action: str):
        observation, reward, done, info = self.env.step([action])
        observation = process_ob(observation[0])

        reward = info.get('won', [0])[0]
        done = done[0]
        admissible = info.get("admissible_commands", [[]])[0]
        return observation, reward, done, admissible

    def step(self, llm_output: str) -> Tuple[str, State]:
        self.state.history.append({
            "role": "assistant",
            "content": llm_output
        })
        try:
            action = self.parse_action(llm_output)
            best = self._heuristic_action(self._last_admissible) if self._last_admissible else ""

            if self.prefer_goal_action and self._last_admissible and self._should_override_with_goal_action(best, action):
                action = best

            if self.force_admissible_action and self._last_admissible:
                if action not in self._last_admissible:
                    if self.heuristic_fallback:
                        action = best or self._heuristic_action(self._last_admissible)
                    else:
                        action = self._maybe_autocorrect(action)

                if action not in self._last_admissible:
                    action = self._last_admissible[0]
            else:
                action = self._maybe_autocorrect(action)

            fixed = self._rewrite_action_in_llm_output(llm_output, action)
            self.state.history[-1]["content"] = fixed

            observation, reward, done, admissible = self.conduct_action(action)

            self._update_goal_progress_from_action_observation(action, observation)

            self._last_admissible = admissible or []

            if isinstance(action, str) and action.startswith("go to "):
                dest = action[len("go to "):].strip().lower()
                if dest:
                    self._visited.add(dest)
            if isinstance(action, str) and action.startswith("use "):
                self._used.add(action.lower())
        except Exception as e:

            self.state.success = False
            self.state.finished = False
            self.state.reward=0
            observation = f"Observation: Error Input. Your input must contains 'Action: '"
            self.state.history.append({
                "role": "user",
                "content": observation,
            })
            self.state.steps += 1
            if self.state.steps >= self.max_steps:
                self.state.finished = True
                self.state.success = False
                self.state.terminate_reason = "max_steps"
                self.state.reward = 0
            return observation, self.state

        self._update_state_from_observation(observation)
        self._infer_inventory_from_admissible(self._last_admissible)
        observation = observation + self._format_goal_and_inventory()
        observation = observation + self._format_suggestions(self._compute_goal_suggestions(self._last_admissible))

        if self.append_admissible:
            is_invalid = observation.strip().lower().startswith("nothing happens")
            if (not self.admissible_on_invalid_only) or is_invalid:
                observation = observation + self._format_admissible(self._last_admissible)

        observation = f"Observation: {observation}"
        self.state.history.append({
            "role": "user",
            "content": observation,
        })

        self.state.steps += 1
        if self.state.steps >= self.max_steps:
            self.state.finished = True
            self.state.success = False
            self.state.terminate_reason = "max_steps"
            self.state.reward = reward

        if done:
            self.state.finished = True
            self.state.success = True
            self.state.terminate_reason = "success"
            self.state.reward = reward

        return observation, self.state

    def reset(self, game_files=None) -> Tuple[str, State]:
        self.state = State()
        self._inventory = []
        self._goal_text = ""
        self._opened = set()
        self._visited = set()
        self._used = set()
        self._current_loc = ""
        self._did_clean = False
        self._did_heat = False
        self._did_cool = False
        self._placed_goal_instances = set()
        self._goal_source_hint = ""

        game_file = self.task.game_file
        if game_file and not os.path.isabs(game_file):

            game_file = os.path.abspath(game_file)

        if hasattr(self.env, "gamefiles") and hasattr(self.env, "_gamefiles_iterator") and game_file:
            self.env.gamefiles = [game_file]
            self.env._gamefiles_iterator = iter(self.env.gamefiles)

        obs, info = self.env.reset()

        raw = obs[0] if isinstance(obs, (list, tuple)) else obs
        cur_task = "\n".join(str(raw).split("\n\n")[1:])

        m = re.search(r"Your task is to:\s*(.*)", cur_task)
        if m:
            self._goal_text = m.group(1).strip().rstrip(".")
            self._parse_goal(self._goal_text)
        self._parse_task_from_gamefile(game_file)
        self.state.error = game_file
        self._last_admissible = info.get("admissible_commands", [[]])[0] if isinstance(info, dict) else []

        observation, messages = prompt_with_icl(self.instruction, self.raw_icl, cur_task, 1)
        if self.icl_format == 'first':
            self.state.history.append({
                "role": "user",
                "content": observation,
            })
        elif self.icl_format == 'conversation':
            self.state.history = messages
        return observation, self.state

class BatchAlfWorldEnv(BaseEnv):
    def __init__(
        self,
        task: AlfWorldTask,
        batch_size: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.task: AlfWorldTask = task
        self.env = task.env
        self.batch_size = batch_size
        self.state = [State() for i in range(batch_size)]

    def parse_action(self, llm_output: List[str]) -> List[str]:
        actions: List[str] = []
        for x in llm_output:
            text = (x or "").strip()
            matches = re.findall(r"(?im)^\s*Action\s*:\s*(.*)$", text)
            if not matches:
                matches = re.findall(r"Action\s*:\s*(.*)", text, flags=re.IGNORECASE | re.DOTALL)
            if not matches:
                actions.append("")
                continue
            a = matches[-1].splitlines()[0]
            a = a.strip().strip("`\"'").strip()
            a = re.sub(r"\s+", " ", a).strip()
            if a.lower().startswith("task "):
                a = "take " + a[5:].lstrip()
            if a.endswith("."):
                a = a[:-1].strip()
            actions.append(a)
        return actions

    def conduct_action(self, actions: List[str]):
        observation, reward, done, info = self.env.step(actions)
        outputs = []
        for i in range(self.batch_size):
            observation, reward, done = process_ob(observation[i]), info['won'][i], done[i]
            outputs.append((observation, reward, done))
        return outputs

    def step(self, llm_output: List[str]) -> Tuple[str, State]:
        for i in range(self.batch_size):
            self.state[i].history.append({
                "role": "assistant",
                "content": llm_output[i]
            })
        actions = self.parse_action(llm_output)

        observations = {}
        correct_idx = []

        for i, action in enumerate(actions):
            if action is None:
                self.state[i].success = False
                self.state[i].finished = False
                self.state[i].reward=0
                observation = f"Observation: Error Input. Your input must contains 'Action: '"
                self.state[i].history.append({
                    "role": "user",
                    "content": observation,
                })
                self.state[i].steps += 1
                if self.state[i].steps >= self.max_steps:
                    self.state[i].finished = True
                    self.state[i].success = False
                    self.state[i].terminate_reason = "max_steps"
                    self.state[i].reward = 0
                actions[i] = ""
                observations[i] = observation
            else:
                correct_idx.append(i)
        outputs = self.conduct_action(actions)
        for i in correct_idx:
            observation, reward, done = outputs[i]
            observation = f"Observation: {observation}"
            self.state[i].history.append({
                "role": "user",
                "content": observation,
            })

            self.state[i].steps += 1
            if self.state[i].steps >= self.max_steps:
                self.state[i].finished = True
                self.state[i].success = False
                self.state[i].terminate_reason = "max_steps"
                self.state[i].reward = reward

            if done:
                self.state[i].finished = True
                self.state[i].success = True
                self.state[i].terminate_reason = "success"
                self.state[i].reward = reward
            observations[i] = observation

        return list(observations.values), self.state

    def reset(self, game_files=None) -> Tuple[str, State]:
        self.state = [State() for i in range(self.batch_size)]

        cur_task = self.task.observation
        for i in range(self.batch_size):
            self.state[i].error = self.task.game_file
            obs = self.env.obs[i]
            obs = "\n".join(obs.split("\n\n")[1:])
            observation, messages = prompt_with_icl(self.instruction, self.raw_icl, obs, 1)
            if self.icl_format == 'first':
                self.state[i].history.append({
                    "role": "user",
                    "content": observation,
                })
            elif self.icl_format == 'conversation':
                self.state[i].history = messages
        return observation, self.state
