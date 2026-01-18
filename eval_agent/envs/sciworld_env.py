import os
import re
import json
import logging
from typing import Tuple

from scienceworld import ScienceWorldEnv

from eval_agent.envs import BaseEnv
from eval_agent.tasks import SciWorldTask
from eval_agent.prompt import prompt_with_icl
from eval_agent.utils.datatypes import State

logger = logging.getLogger("agent_frame")

class SciWorldEnv(BaseEnv):
    def __init__(
        self,
        task: SciWorldTask,
        env: ScienceWorldEnv,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.task: SciWorldTask = task
        self.env = env
        self.max_steps_dict = json.load(open("eval_agent/data/sciworld/max_steps.json"))

        self.state = State()

    def _normalize_action(self, action: str) -> str:
        a = action.strip()

        a = a.strip("`").strip()
        if (a.startswith('"') and a.endswith('"')) or (a.startswith("'") and a.endswith("'")):
            a = a[1:-1].strip()

        a = re.sub(r"\s+", " ", a).strip()

        m = re.fullmatch(r"teleport to (?:the )?(.+)", a, flags=re.IGNORECASE)
        if m:
            loc = m.group(1).strip().lower()

            a = f"teleport to {loc}"

        m = re.fullmatch(r"wait\s*\(?\s*no-op\s*(\d+)\s*steps?\s*\)?", a, flags=re.IGNORECASE)
        if m:
            n = int(m.group(1))
            return "wait1" if n == 1 else "wait"

        m = re.fullmatch(r"wait\s+(\d+)\s*\(.*\)$", a, flags=re.IGNORECASE)
        if m:
            n = int(m.group(1))
            return "wait1" if n == 1 else "wait"
        m = re.fullmatch(r"wait\s+(\d+)", a, flags=re.IGNORECASE)
        if m:
            n = int(m.group(1))
            return "wait1" if n == 1 else "wait"

        m = re.fullmatch(r"put (.+) in (.+)", a, flags=re.IGNORECASE)
        if m:
            return f"move {m.group(1).strip()} to {m.group(2).strip()}"

        m = re.fullmatch(r"move (.+?) in .+? to (.+)", a, flags=re.IGNORECASE)
        if m:
            return f"move {m.group(1).strip()} to {m.group(2).strip()}"

        rooms = [
            "kitchen", "foundry", "workshop", "bathroom", "outside",
            "living room", "bedroom", "greenhouse", "art studio", "hallway",
        ]
        room_alt = "|".join(re.escape(r) for r in sorted(rooms, key=len, reverse=True))
        m = re.fullmatch(rf"(open|close|activate|deactivate|focus on|examine|look at|read|pick up)\s+(.+?)\s+in\s+({room_alt})", a, flags=re.IGNORECASE)
        if m:
            return f"{m.group(1).lower()} {m.group(2).strip()}"

        return a

    def parse_action(self, llm_output: str) -> str:
        text = (llm_output or "").strip()

        candidates = re.findall(r"(?mi)^[ \t]*Action:\s*(.+?)\s*$", text)
        if not candidates:

            m = re.search(r"(?i)Action:\s*(.+)", text)
            if not m:
                raise ValueError("Missing 'Action:' line")
            action = m.group(1).strip().splitlines()[0].strip()
            return self._normalize_action(action)

        action = candidates[0].strip()

        action = action.split(";")[0].strip()
        action = action.split("\n")[0].strip()
        return self._normalize_action(action)

    def step(self, llm_output: str) -> Tuple[str, State]:
        self.state.history.append({
            "role": "assistant",
            "content": llm_output
        })
        try:
            action = self.parse_action(llm_output)
        except:
            observation = f"Observation: Invalid format. The input must contains 'Action: '"
            self.state.history.append({
                "role": "user",
                "content": observation,
            })
            self.state.steps += 1
            self.state.reward = 0
            if self.state.steps >= self.max_steps:
                self.state.finished = True
                self.state.success = False
                self.state.terminate_reason = "max_steps"
                self.state.reward = 0
            return observation, self.state
        def _get_valid_actions():
            try:
                valid = self.env.getValidActionObjectCombinations()
            except Exception:
                valid = None
            if not valid:
                return []
            try:
                valid = [v if isinstance(v, str) else " ".join(map(str, v)) for v in valid]
            except Exception:
                valid = [str(v) for v in valid]
            valid = [v.strip() for v in valid if str(v).strip()]
            return valid

        def _pick_best_valid_action(valid_actions, attempted: str) -> str | None:
            if not valid_actions:
                return None

            la = (attempted or "").strip().lower()
            la_verb = la.split(" ", 1)[0] if la else ""
            la_tokens = [t for t in re.split(r"[^a-z0-9]+", la) if t]

            def _score(v: str) -> tuple:
                s = v.lower()
                verb = s.split(" ", 1)[0] if s else ""
                same_verb = int(bool(la_verb and verb == la_verb))
                tok_overlap = sum(1 for t in la_tokens[:8] if t and t in s)
                return (same_verb, tok_overlap, -len(s))

            best = max(valid_actions, key=_score)

            if la_verb and not best.lower().startswith(la_verb):
                if _score(best)[1] == 0:
                    for v in valid_actions:
                        if v.lower() == "look around":
                            return v
            return best

        def _format_valid_actions_hint(last_action: str = "", max_items: int = 30) -> str:
            valid = _get_valid_actions()
            if not valid:
                return ""

            la = (last_action or "").strip().lower()
            la_verb = la.split(" ", 1)[0] if la else ""
            la_tokens = [t for t in re.split(r"[^a-z0-9]+", la) if t]

            def _score(v: str) -> tuple:
                s = v.lower()
                verb = s.split(" ", 1)[0] if s else ""
                same_verb = int(bool(la_verb and verb == la_verb))
                tok_overlap = sum(1 for t in la_tokens[:6] if t and t in s)

                return (same_verb, tok_overlap)

            if la:
                valid = sorted(valid, key=_score, reverse=True)

            shown = valid[:max_items]
            lines = "\n".join(f"- {v}" for v in shown)
            more = "" if len(valid) <= max_items else f"\n... ({len(valid) - max_items} more)"
            return (
                "\n\n[Hint] Your last action was invalid. Pick ONE action from the valid list below and output strictly:\n"
                "Thought: ...\n"
                "Action: <one action>\n\n"
                f"{lines}{more}"
            )

        def _is_invalid_observation(obs: str) -> bool:
            s = (obs or "").lower()
            return (
                "no known action matches" in s
                or "invalid action" in s
                or "i don't understand" in s
                or "not a valid action" in s
            )

        aggressive_fix = os.environ.get("SCIWORLD_AGGRESSIVE_FIX", "0") == "1"
        if not hasattr(self, "_consecutive_invalid"):
            self._consecutive_invalid = 0

        try:
            env_obs, _, done, info = self.env.step(action)
            reward = info['raw_score']
            observation = f"Observation: {env_obs}"
            if self.state.reward is None or reward > self.state.reward:
                self.state.reward = reward

            if _is_invalid_observation(env_obs):
                self._consecutive_invalid += 1
                valid = _get_valid_actions()

                if aggressive_fix and valid:

                    if self._consecutive_invalid >= 3:
                        fixed = next((v for v in valid if v.lower() == "look around"), None)
                        fixed = fixed or _pick_best_valid_action(valid, action)
                    else:
                        fixed = _pick_best_valid_action(valid, action)

                    if fixed:
                        fixed = self._normalize_action(fixed)
                        env_obs2, _, done2, info2 = self.env.step(fixed)
                        reward2 = info2.get("raw_score", reward)
                        observation = (
                            f"Observation: {env_obs}\n"
                            f"[AutoFix] Replaced invalid action '{action}' -> '{fixed}'\n"
                            f"Observation: {env_obs2}"
                        )
                        done = done2
                        if self.state.reward is None or reward2 > self.state.reward:
                            self.state.reward = reward2
                        self._consecutive_invalid = 0
                    else:
                        observation = observation + _format_valid_actions_hint(action)
                else:
                    observation = observation + _format_valid_actions_hint(action)
            else:
                self._consecutive_invalid = 0
        except AssertionError:
            self._consecutive_invalid += 1
            valid = _get_valid_actions()
            if aggressive_fix and valid:
                fixed = _pick_best_valid_action(valid, action)
                if fixed:
                    fixed = self._normalize_action(fixed)
                    env_obs2, _, done2, info2 = self.env.step(fixed)
                    reward2 = info2.get("raw_score", 0)
                    observation = (
                        "Observation: Invalid action!\n"
                        f"[AutoFix] Replaced invalid action '{action}' -> '{fixed}'\n"
                        f"Observation: {env_obs2}"
                    )
                    done = done2
                    if self.state.reward is None or reward2 > self.state.reward:
                        self.state.reward = reward2
                    self._consecutive_invalid = 0
                else:
                    observation = 'Observation: Invalid action!' + _format_valid_actions_hint(action)
                    done = False
            else:
                observation = 'Observation: Invalid action!' + _format_valid_actions_hint(action)
                done = False

        self.state.history.append({
            "role": "user",
            "content": f"{observation}",
        })

        self.state.steps += 1
        if self.state.steps >= self.max_steps:
            self.state.finished = True
            self.state.success = False
            self.state.terminate_reason = "max_steps"

        if done:
            self.state.finished = True
            self.state.success = True
            self.state.terminate_reason = "success"

        return observation, self.state

    def reset(self) -> Tuple[str, State]:
        self.state = State()
        self.max_steps = self.max_steps_dict[self.task.sub_task_name]
        self.env.load(self.task.sub_task_name, self.task.variation_idx, simplificationStr="easy", generateGoldPath=False)
        obs, info = self.env.reset()
        cur_task = info['taskDesc']
        observation, messages = prompt_with_icl(self.instruction, self.raw_icl, cur_task, 1)
        if self.icl_format == 'first':
            self.state.history.append({
                "role": "user",
                "content": observation,
            })
        elif self.icl_format == 'conversation':
            self.state.history = messages
        return observation, self.state
