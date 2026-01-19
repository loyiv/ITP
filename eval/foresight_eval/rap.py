from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

def _truncate_by_stop_strings(text: str, stop: Optional[List[str]]) -> str:
    if not stop:
        return text
    cut = None
    for s in stop:
        if not s:
            continue
        idx = text.find(s)
        if idx != -1 and (cut is None or idx < cut):
            cut = idx
    return text[:cut].strip() if cut is not None else text

class RAPLM:

    def __init__(self, tokenizer, model, device: torch.device):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device

    @torch.no_grad()
    def generate_one(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_p: float = 0.95,
        stop: Optional[List[str]] = None,
        do_sample: bool = True,
    ) -> str:
        enc = self.tokenizer(prompt, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        input_len = int(enc["input_ids"].shape[-1])

        out = self.model.generate(
            **enc,
            do_sample=bool(do_sample),
            temperature=float(temperature) if do_sample else None,
            top_p=float(top_p) if do_sample else None,
            max_new_tokens=int(max_new_tokens),
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        new_tokens = out[0][input_len:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return _truncate_by_stop_strings(text, stop).strip()

    @torch.no_grad()
    def next_token_prob(self, prompt: str, candidates: List[str]) -> Dict[str, float]:
        enc = self.tokenizer(prompt, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        logits = self.model(**enc).logits
        last = logits[0, -1, :]
        probs = torch.softmax(last, dim=-1)

        out: Dict[str, float] = {}
        for cand in candidates:
            ids = self.tokenizer.encode(cand, add_special_tokens=False)
            if len(ids) == 1:
                out[cand] = float(probs[ids[0]].item())
        return out

    @torch.no_grad()
    def sequence_logprob(self, prompt: str, completion: str, normalize: str = "mean") -> float:
        full = prompt + completion
        enc_full = self.tokenizer(full, return_tensors="pt")
        enc_prompt = self.tokenizer(prompt, return_tensors="pt")
        enc_full = {k: v.to(self.device) for k, v in enc_full.items()}
        enc_prompt = {k: v.to(self.device) for k, v in enc_prompt.items()}

        input_ids = enc_full["input_ids"]
        attn = enc_full.get("attention_mask", None)
        logits = self.model(input_ids=input_ids, attention_mask=attn).logits
        logp = torch.log_softmax(logits, dim=-1)

        prompt_len = int(enc_prompt["input_ids"].shape[1])
        T = int(input_ids.shape[1])
        if prompt_len >= T:
            return float("-inf")

        comp_token_indices = list(range(prompt_len, T))
        pred_positions = [i - 1 for i in comp_token_indices if i - 1 >= 0]
        target_tokens = input_ids[0, comp_token_indices]

        lp = logp[0, pred_positions, :]
        token_lp = lp.gather(1, target_tokens.unsqueeze(1)).squeeze(1)

        s = float(token_lp.sum().item())
        if normalize == "sum":
            return s
        denom = max(1, int(token_lp.numel()))
        return s / denom

def build_action_prompt(goal: str, obs: str, inventory: str, valid_actions: List[str]) -> str:
    va = "\n".join(f"- {a}" for a in (valid_actions or []))
    return (
        "You are an agent in a text-based household environment.\n"
        "Your job is to complete the Goal by issuing exactly one valid Action each step.\n\n"
        f"Goal: {goal}\n"
        f"Observation: {obs}\n"
        f"Inventory: {inventory}\n\n"
        "Valid actions (choose exactly ONE, copy it verbatim):\n"
        f"{va}\n\n"
        "Action:"
    )

def build_world_model_prompt(goal: str, obs: str, inventory: str, action: str) -> str:
    return (
        "You simulate the environment transition for a text-based household world.\n"
        "Given the current Goal, Observation, Inventory, and an Action, predict the NEXT Observation.\n"
        "Be concise and consistent.\n\n"
        f"Goal: {goal}\n"
        f"Observation: {obs}\n"
        f"Inventory: {inventory}\n"
        f"Action: {action}\n\n"
        "Next Observation:"
    )

def build_self_eval_prompt(goal: str, obs: str, action: str, pred_next_obs: str) -> str:
    return (
        "Answer with a single word: Yes or No.\n"
        "Question: Is the Action a correct and useful step toward the Goal, given the Observation and the predicted next Observation?\n\n"
        f"Goal: {goal}\n"
        f"Observation: {obs}\n"
        f"Action: {action}\n"
        f"Predicted next Observation: {pred_next_obs}\n\n"
        "Answer:"
    )

@dataclass(frozen=True)
class RAPState:
    goal: str
    obs: str
    inventory: str

    def key(self) -> str:
        return f"GOAL:\n{self.goal}\nOBS:\n{self.obs}\nINV:\n{self.inventory}\n"

@dataclass
class EdgeStats:
    child_state: Optional[RAPState]
    r_full: float
    r_light: float
    q: float = 0.0
    n: int = 0

@dataclass
class NodeStats:
    n: int = 0
    edges: Dict[str, EdgeStats] = None

    def __post_init__(self):
        if self.edges is None:
            self.edges = {}

class RAPPlanner:
    def __init__(
        self,
        policy_lm: RAPLM,
        wm_lm: RAPLM,
        d_actions: int = 8,
        depth_limit: int = 6,
        num_rollouts: int = 32,
        uct_w: float = 1.4,
        gamma: float = 1.0,
        rw_action_logp: float = 0.2,
        rw_self_eval: float = 0.6,
        rw_goal_heur: float = 0.2,
        action_gen_tokens: int = 32,
        wm_pred_tokens: int = 128,
        seed: int = 0,
    ):
        self.policy_lm = policy_lm
        self.wm_lm = wm_lm

        self.d_actions = int(d_actions)
        self.depth_limit = int(depth_limit)
        self.num_rollouts = int(num_rollouts)
        self.uct_w = float(uct_w)
        self.gamma = float(gamma)

        self.rw_action_logp = float(rw_action_logp)
        self.rw_self_eval = float(rw_self_eval)
        self.rw_goal_heur = float(rw_goal_heur)

        self.action_gen_tokens = int(action_gen_tokens)
        self.wm_pred_tokens = int(wm_pred_tokens)

        random.seed(seed)
        self.nodes: Dict[str, NodeStats] = {}

    def reset_cache(self) -> None:
        self.nodes = {}

    def _get_node(self, s: RAPState) -> NodeStats:
        k = s.key()
        if k not in self.nodes:
            self.nodes[k] = NodeStats()
        return self.nodes[k]

    def _normalize_action(self, a: str) -> str:
        a = (a or "").strip()
        if a.lower().startswith("action:"):
            a = a.split(":", 1)[1].strip()
        return a.strip().strip(".").strip()

    def _goal_heuristic(self, goal: str, text: str) -> float:
        toks = [t.strip(" .,:;()[]").lower() for t in (goal or "").split()]
        toks = [t for t in toks if len(t) >= 4]
        if not toks:
            return 0.0
        low = (text or "").lower()
        hit = sum(1 for t in set(toks) if t in low)
        return hit / len(set(toks))

    def _reward_full(self, s: RAPState, action: str, pred_next_obs: str, action_prompt: str) -> Tuple[float, float]:
        lp = self.policy_lm.sequence_logprob(action_prompt, completion=action, normalize="mean")
        se_prompt = build_self_eval_prompt(s.goal, s.obs, action, pred_next_obs)
        p = self.policy_lm.next_token_prob(se_prompt, candidates=["Yes", " Yes", "YES", " YES"])
        yes_p = max(p.values()) if p else 0.0
        gh = self._goal_heuristic(s.goal, pred_next_obs)
        r_full = self.rw_action_logp * lp + self.rw_self_eval * yes_p + self.rw_goal_heur * gh
        r_light = self.rw_self_eval * yes_p + self.rw_goal_heur * gh
        return r_full, r_light

    def _expand_state(self, s: RAPState, valid_actions: Optional[List[str]] = None) -> None:
        node = self._get_node(s)
        if node.edges:
            return

        action_prompt = build_action_prompt(s.goal, s.obs, s.inventory, valid_actions or [])
        sampled: List[str] = []
        tries = 0
        while len(sampled) < self.d_actions and tries < self.d_actions * 4:
            tries += 1
            cand = self.policy_lm.generate_one(
                action_prompt,
                max_new_tokens=self.action_gen_tokens,
                temperature=0.8,
                top_p=0.95,
                stop=["\n"],
                do_sample=True,
            )
            cand = self._normalize_action(cand)
            if not cand:
                continue
            if valid_actions:
                if cand not in valid_actions:
                    continue
            if cand in sampled:
                continue
            sampled.append(cand)

        if valid_actions:
            pool = [a for a in valid_actions if a not in sampled]
            random.shuffle(pool)
            while len(sampled) < self.d_actions and pool:
                sampled.append(pool.pop())

        for a in sampled:
            wm_prompt = build_world_model_prompt(s.goal, s.obs, s.inventory, a)
            pred_next_obs = self.wm_lm.generate_one(
                wm_prompt,
                max_new_tokens=self.wm_pred_tokens,
                temperature=0.8,
                top_p=0.95,
                stop=["\n\n", "\nAction:", "\nNext Action:", "\nAnswer:"],
                do_sample=True,
            )
            s_next = RAPState(goal=s.goal, obs=pred_next_obs, inventory=s.inventory)
            r_full, r_light = self._reward_full(s, a, pred_next_obs, action_prompt)
            node.edges[a] = EdgeStats(child_state=s_next, r_full=r_full, r_light=r_light, q=0.0, n=0)

    def _uct_score(self, parent_n: int, edge: EdgeStats) -> float:
        if edge.n == 0:
            return float("inf")
        return edge.q + self.uct_w * math.sqrt(max(0.0, math.log(max(1, parent_n))) / edge.n)

    def choose_action(self, root: RAPState, valid_actions: Optional[List[str]] = None) -> str:
        for _ in range(self.num_rollouts):
            path: List[Tuple[RAPState, str]] = []
            rewards: List[float] = []
            s = root
            depth = 0

            while depth < self.depth_limit:
                node = self._get_node(s)
                node.n += 1
                if not node.edges:
                    break
                unvisited = [(a, e) for a, e in node.edges.items() if e.n == 0]
                if unvisited:
                    a_sel, e_sel = max(unvisited, key=lambda x: x[1].r_light)
                else:
                    a_sel, e_sel = max(node.edges.items(), key=lambda x: self._uct_score(node.n, x[1]))
                path.append((s, a_sel))
                rewards.append(e_sel.r_full)
                s = e_sel.child_state if e_sel.child_state is not None else s
                depth += 1

            while depth < self.depth_limit:
                node = self._get_node(s)
                if not node.edges:
                    self._expand_state(s, valid_actions=valid_actions)
                if not node.edges:
                    break
                a_sim, e_sim = max(node.edges.items(), key=lambda x: x[1].r_light)
                path.append((s, a_sim))
                rewards.append(e_sim.r_full)
                s = e_sim.child_state if e_sim.child_state is not None else s
                depth += 1

            G = 0.0
            for i in reversed(range(len(path))):
                G = rewards[i] + self.gamma * G
                s_i, a_i = path[i]
                node_i = self._get_node(s_i)
                edge_i = node_i.edges[a_i]
                edge_i.n += 1
                edge_i.q += (G - edge_i.q) / max(1, edge_i.n)

        root_node = self._get_node(root)
        if not root_node.edges:
            self._expand_state(root, valid_actions=valid_actions)
        if not root_node.edges:
            if valid_actions:
                return random.choice(valid_actions)
            return "look"
        best_a, _ = max(root_node.edges.items(), key=lambda kv: (kv[1].n, kv[1].q))
        return best_a

