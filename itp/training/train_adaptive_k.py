from __future__ import annotations

import argparse
import json
import os
import re
import random
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

_OBS_RE = re.compile(r"Observation:\s*(.*?)(?:\nHistory:|\Z)", re.DOTALL)
_TASK_RE = re.compile(r"(?:Your task is to:|Task:|Mission:)\s*(.*)", re.IGNORECASE)

def extract_observation(state_text: str) -> str:
    m = _OBS_RE.search(state_text)
    if not m:
        s = state_text.strip()
        return re.sub(r"\s+", " ", s[-200:])
    obs = m.group(1).strip()
    obs = re.sub(r"\s+", " ", obs)
    return obs

def extract_task_from_text(text: str) -> str:
    m = _TASK_RE.search(text)
    if m:
        return re.sub(r"\s+", " ", m.group(1).strip())
    return ""

def build_state_text(mission: str, obs: str, history: str) -> str:
    mission = mission.strip()
    obs = obs.strip()
    history = history.strip()
    return (
        f"Mission: {mission}\n"
        f"Observation: {obs}\n"
        f"History: {history}"
    )

SPECIAL_TOKENS = ["<CTRL>", "<STATE>", "</STATE>", "<FORESIGHT>", "</FORESIGHT>"]

def build_prompt_for_action(state_text: str, k: int, obs_k: str) -> str:
    return (
        "You are an ALFWorld household agent.\n"
        "Given the current state, output ONLY the next action string.\n"
        "<STATE>\n"
        f"{state_text}\n"
        "</STATE>\n"
        "<CTRL>\n"
        "<FORESIGHT>\n"
        f"K={k}\n"
        f"Obs@K: {obs_k}\n"
        "</FORESIGHT>\n"
        "Action:"
    )

def build_prompt_for_controller(state_text: str) -> str:
    return (
        "You are an ALFWorld household agent.\n"
        "Decide how many steps to look ahead with a world model.\n"
        "<STATE>\n"
        f"{state_text}\n"
        "</STATE>\n"
        "<CTRL>"
    )

def _truncate_left_1d(
    ids: torch.Tensor,
    mask: torch.Tensor,
    labels: Optional[torch.Tensor],
    max_len: int,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    if ids.numel() <= max_len:
        return ids, mask, labels
    ids = ids[-max_len:]
    mask = mask[-max_len:]
    if labels is not None:
        labels = labels[-max_len:]
    return ids, mask, labels

def _pad_1d_batch(
    seqs: List[torch.Tensor],
    pad_value: int,
    padding_side: Literal["left", "right"],
    target_len: Optional[int] = None,
) -> torch.Tensor:
    assert len(seqs) > 0
    max_len = max(int(s.numel()) for s in seqs) if target_len is None else int(target_len)
    out = torch.full((len(seqs), max_len), pad_value, dtype=seqs[0].dtype)
    for i, s in enumerate(seqs):
        L = int(s.numel())
        if padding_side == "left":
            out[i, max_len - L:] = s
        else:
            out[i, :L] = s
    return out

def _ensure_pad(tokenizer: AutoTokenizer) -> None:
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has no pad_token_id and no eos_token_id; please set pad/eos.")
        tokenizer.pad_token = tokenizer.eos_token

def make_lm_batch(
    tokenizer: AutoTokenizer,
    prompts: List[str],
    actions: List[str],
    max_seq_len: int,
) -> Dict[str, torch.Tensor]:
    assert len(prompts) == len(actions)
    _ensure_pad(tokenizer)
    pad_id = tokenizer.pad_token_id
    padding_side: Literal["left", "right"] = getattr(tokenizer, "padding_side", "right")

    eos_str = tokenizer.eos_token if tokenizer.eos_token is not None else ""

    ids_list: List[torch.Tensor] = []
    mask_list: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []

    for p, a in zip(prompts, actions):
        p_ids = tokenizer(p, add_special_tokens=False).input_ids
        a_ids = tokenizer(a + eos_str, add_special_tokens=False).input_ids

        input_ids = torch.tensor(p_ids + a_ids, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        labels = torch.tensor([-100] * len(p_ids) + a_ids, dtype=torch.long)

        input_ids, attention_mask, labels = _truncate_left_1d(input_ids, attention_mask, labels, max_seq_len)

        ids_list.append(input_ids)
        mask_list.append(attention_mask)
        labels_list.append(labels)

    input_ids_b = _pad_1d_batch(ids_list, pad_value=pad_id, padding_side=padding_side)
    attention_b = _pad_1d_batch(mask_list, pad_value=0, padding_side=padding_side)
    labels_b = _pad_1d_batch(labels_list, pad_value=-100, padding_side=padding_side)

    return {"input_ids": input_ids_b, "attention_mask": attention_b, "labels": labels_b}

def make_controller_batch(
    tokenizer: AutoTokenizer,
    prompts: List[str],
    max_seq_len: int,
) -> Dict[str, torch.Tensor]:
    _ensure_pad(tokenizer)
    pad_id = tokenizer.pad_token_id
    padding_side: Literal["left", "right"] = getattr(tokenizer, "padding_side", "right")

    ids_list: List[torch.Tensor] = []
    mask_list: List[torch.Tensor] = []

    for p in prompts:
        p_ids = tokenizer(p, add_special_tokens=False).input_ids
        input_ids = torch.tensor(p_ids, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        input_ids, attention_mask, _ = _truncate_left_1d(input_ids, attention_mask, None, max_seq_len)

        ids_list.append(input_ids)
        mask_list.append(attention_mask)

    input_ids_b = _pad_1d_batch(ids_list, pad_value=pad_id, padding_side=padding_side)
    attention_b = _pad_1d_batch(mask_list, pad_value=0, padding_side=padding_side)

    return {"input_ids": input_ids_b, "attention_mask": attention_b}

def find_ctrl_positions(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    ctrl_token_id: int,
) -> torch.Tensor:
    B, L = input_ids.shape
    ctrl_pos = torch.full((B,), L - 1, dtype=torch.long, device=input_ids.device)
    for i in range(B):
        idx = (input_ids[i] == ctrl_token_id).nonzero(as_tuple=False)
        if idx.numel() > 0:
            ctrl_pos[i] = idx[-1].item()
        else:
            nonpad = (attention_mask[i] == 1).nonzero(as_tuple=False)
            if nonpad.numel() > 0:
                ctrl_pos[i] = nonpad[-1].item()
    return ctrl_pos

def add_special_tokens_and_resize(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    special_tokens: List[str],
) -> int:
    old_vocab = model.get_input_embeddings().weight.size(0)
    n_added = tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    _ensure_pad(tokenizer)
    if n_added > 0:
        model.resize_token_embeddings(len(tokenizer))

        with torch.no_grad():
            emb = model.get_input_embeddings().weight
            mean_vec = emb[:old_vocab].mean(dim=0, keepdim=True)
            emb[old_vocab:] = mean_vec
    else:

        if model.get_input_embeddings().weight.size(0) != len(tokenizer):
            model.resize_token_embeddings(len(tokenizer))
    return n_added

class PolicyWithHeads(nn.Module):
    def __init__(self, lm: AutoModelForCausalLM, hidden_size: int, kmax: int):
        super().__init__()
        self.lm = lm
        self.kmax = kmax
        self.k_head = nn.Linear(hidden_size, kmax + 1)
        self.v_head = nn.Linear(hidden_size, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        ctrl_pos: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if hasattr(self.lm, "model") and hasattr(self.lm, "lm_head"):
            base_out = self.lm.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            hidden = base_out.last_hidden_state
            logits = self.lm.lm_head(hidden)
        else:
            out = self.lm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            logits = out.logits
            hidden = out.hidden_states[-1]

        if ctrl_pos.device != hidden.device:
            ctrl_pos = ctrl_pos.to(hidden.device)
        h_ctrl = hidden[torch.arange(hidden.size(0), device=hidden.device), ctrl_pos]

        head_dtype = self.k_head.weight.dtype
        if h_ctrl.dtype != head_dtype:
            h_ctrl_for_heads = h_ctrl.to(dtype=head_dtype)
        else:
            h_ctrl_for_heads = h_ctrl

        k_logits = self.k_head(h_ctrl_for_heads)
        values = self.v_head(h_ctrl_for_heads).squeeze(-1)

        result = {"logits": logits, "k_logits": k_logits, "values": values}

        if labels is not None:
            if labels.device != logits.device:
                labels = labels.to(logits.device)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            lm_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="mean",
            )
            result["lm_loss"] = lm_loss

        return result

class WorldModelWrapper:
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        device: torch.device,
        max_new_tokens: int = 192,
        autocast_dtype: torch.dtype = torch.float16,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.autocast_dtype = autocast_dtype

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def _wm_prompt(self, state_text: str, action_text: str) -> str:
        return (
            "You are a world model for ALFWorld.\n"
            "Given the current state and an action, predict the next state in the same format.\n"
            "<STATE>\n"
            f"{state_text}\n"
            "</STATE>\n"
            "Action:\n"
            f"{action_text}\n"
            "Next state:\n"
        )

    @torch.inference_mode()
    def predict_next_states_batch(self, states: List[str], actions: List[str]) -> List[str]:
        prompts = [self._wm_prompt(s, a) for s, a in zip(states, actions)]
        tok = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1536,
            add_special_tokens=False,
        ).to(self.device)

        use_amp = (self.device.type == "cuda")
        with torch.cuda.amp.autocast(enabled=use_amp, dtype=self.autocast_dtype):
            gen = self.model.generate(
                **tok,
                do_sample=False,
                num_beams=1,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        input_len = tok["input_ids"].shape[1]

        out_texts: List[str] = []
        for i in range(gen.size(0)):
            new_tokens = gen[i, input_len:]
            txt = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            out_texts.append(txt)
        return out_texts

@torch.inference_mode()
def compute_action_logprobs_batch(
    model: PolicyWithHeads,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    actions: List[str],
    max_seq_len: int,
    ctrl_token_id: int,
    device: torch.device,
    normalize: Literal["sum", "mean"] = "sum",
) -> torch.Tensor:
    batch = make_lm_batch(tokenizer, prompts, actions, max_seq_len)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    ctrl_pos = find_ctrl_positions(input_ids, attention_mask, ctrl_token_id)

    use_amp = (device.type == "cuda")
    with torch.cuda.amp.autocast(enabled=use_amp):
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            ctrl_pos=ctrl_pos,
            labels=None,
        )
        logits = out["logits"]

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    token_nll = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(shift_labels.size(0), shift_labels.size(1))

    valid = (shift_labels != -100).float()
    sum_nll = (token_nll * valid).sum(dim=1)

    if normalize == "mean":
        denom = valid.sum(dim=1).clamp(min=1.0)
        logprob = -(sum_nll / denom)
    else:
        logprob = -sum_nll

    return logprob

def compute_action_logprob_single_grad(
    model: PolicyWithHeads,
    tokenizer: AutoTokenizer,
    prompt: str,
    action: str,
    max_seq_len: int,
    ctrl_token_id: int,
    device: torch.device,
    normalize: Literal["sum", "mean"] = "sum",
) -> torch.Tensor:
    batch = make_lm_batch(tokenizer, [prompt], [action], max_seq_len)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    ctrl_pos = find_ctrl_positions(input_ids, attention_mask, ctrl_token_id)

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        ctrl_pos=ctrl_pos,
        labels=None,
    )
    logits = out["logits"]

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    logp_all = F.log_softmax(shift_logits.float(), dim=-1)

    mask = (shift_labels != -100)
    safe_labels = shift_labels.clone()
    safe_labels[~mask] = 0
    chosen = logp_all.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    chosen = chosen * mask.float()

    if normalize == "mean":
        denom = mask.float().sum(dim=1).clamp(min=1.0)
        return chosen.sum(dim=1) / denom
    return chosen.sum(dim=1)

def load_expert_jsonl(path: str) -> Dict[str, List[Dict[str, Any]]]:
    episodes: Dict[str, List[Dict[str, Any]]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            eid = obj["episode_id"]
            episodes.setdefault(eid, []).append(obj)

    for eid in episodes:
        episodes[eid] = sorted(episodes[eid], key=lambda x: int(x["step"]))
    return episodes

class LabeledKDataset(Dataset):
    def __init__(self, jsonl_path: str):
        self.samples: List[Dict[str, Any]] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.samples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]

@dataclass
class CollateConfig:
    max_seq_len: int
    kmax: int

def collate_sft(batch: List[Dict[str, Any]], tokenizer: AutoTokenizer, cfg: CollateConfig) -> Dict[str, torch.Tensor]:
    actions = [b["action"] for b in batch]
    k_labels = torch.tensor([int(b["k_label"]) for b in batch], dtype=torch.long)

    prompts: List[str] = []
    for b in batch:
        k = int(b["k_label"])
        obs_k = b["obs_list"][k]
        prompts.append(build_prompt_for_action(b["state"], k, obs_k))

    lm_batch = make_lm_batch(tokenizer, prompts, actions, cfg.max_seq_len)
    return {
        "input_ids": lm_batch["input_ids"],
        "attention_mask": lm_batch["attention_mask"],
        "labels": lm_batch["labels"],
        "k_labels": k_labels,
    }

def save_policy_with_heads(
    out_dir: str,
    tokenizer: AutoTokenizer,
    base_lm: AutoModelForCausalLM,
    policy: PolicyWithHeads,
    kmax: int,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    tokenizer.save_pretrained(out_dir)
    base_lm.save_pretrained(out_dir)

    heads = {
        "kmax": int(kmax),
        "k_head": policy.k_head.state_dict(),
        "v_head": policy.v_head.state_dict(),
    }
    torch.save(heads, os.path.join(out_dir, "adaptive_heads.pt"))

def load_policy_with_heads(
    model_dir: str,
    device: torch.device,
    kmax: int,
    torch_dtype: torch.dtype = torch.float16,
    device_map: Optional[str] = None,
) -> Tuple[AutoTokenizer, AutoModelForCausalLM, PolicyWithHeads, int]:
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    lm_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
    }

    if device_map:
        lm_kwargs["device_map"] = device_map
        base_lm = AutoModelForCausalLM.from_pretrained(model_dir, **lm_kwargs)
    else:
        base_lm = AutoModelForCausalLM.from_pretrained(model_dir, **lm_kwargs).to(device)

    add_special_tokens_and_resize(tokenizer, base_lm, SPECIAL_TOKENS)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    hidden_size = getattr(base_lm.config, "hidden_size", None) or getattr(base_lm.config, "n_embd", None)
    if hidden_size is None:
        raise ValueError("Cannot infer hidden size from model config.")

    heads_path = os.path.join(model_dir, "adaptive_heads.pt")
    if not os.path.exists(heads_path):
        policy = PolicyWithHeads(base_lm, hidden_size=hidden_size, kmax=kmax)
        if device_map:

            head_dev = getattr(getattr(base_lm, "lm_head", None), "weight", None)
            head_device = head_dev.device if head_dev is not None else next(base_lm.parameters()).device
            policy.k_head.to(device=head_device, dtype=torch_dtype)
            policy.v_head.to(device=head_device, dtype=torch_dtype)
        else:
            policy = policy.to(device=device, dtype=torch_dtype)
        return tokenizer, base_lm, policy, kmax

    heads = torch.load(heads_path, map_location="cpu")
    saved_kmax = int(heads["kmax"])
    policy = PolicyWithHeads(base_lm, hidden_size=hidden_size, kmax=saved_kmax)
    policy.k_head.load_state_dict(heads["k_head"])
    policy.v_head.load_state_dict(heads["v_head"])
    if device_map:
        head_dev = getattr(getattr(base_lm, "lm_head", None), "weight", None)
        head_device = head_dev.device if head_dev is not None else next(base_lm.parameters()).device
        policy.k_head.to(device=head_device, dtype=torch_dtype)
        policy.v_head.to(device=head_device, dtype=torch_dtype)
    else:
        policy = policy.to(device=device, dtype=torch_dtype)
    return tokenizer, base_lm, policy, saved_kmax

def stage_label(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float16 if device.type == "cuda" else torch.float32

    episodes = load_expert_jsonl(args.expert_jsonl)

    tok_p = AutoTokenizer.from_pretrained(args.policy_model_path, trust_remote_code=True)
    lm_p = AutoModelForCausalLM.from_pretrained(
        args.policy_model_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    ).to(device)

    add_special_tokens_and_resize(tok_p, lm_p, SPECIAL_TOKENS)
    tok_p.padding_side = args.padding_side

    hidden_size = getattr(lm_p.config, "hidden_size", None) or getattr(lm_p.config, "n_embd", None)
    if hidden_size is None:
        raise ValueError("Cannot infer hidden size from policy model config.")
    policy = PolicyWithHeads(lm_p, hidden_size=hidden_size, kmax=args.kmax).to(device)
    policy.eval()

    ctrl_token_id = tok_p.convert_tokens_to_ids("<CTRL>")
    if ctrl_token_id is None or ctrl_token_id < 0:
        raise ValueError("Cannot find <CTRL> token id. Ensure SPECIAL_TOKENS are added correctly.")

    tok_wm = AutoTokenizer.from_pretrained(args.wm_model_path, trust_remote_code=True)
    if tok_wm.pad_token is None:
        tok_wm.pad_token = tok_wm.eos_token or tok_wm.unk_token
    tok_wm.padding_side = args.padding_side

    wm_lm = AutoModelForCausalLM.from_pretrained(
        args.wm_model_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    ).to(device)
    wm = WorldModelWrapper(wm_lm, tok_wm, device=device, max_new_tokens=args.wm_max_new_tokens)

    os.makedirs(os.path.dirname(args.out_labeled_jsonl) or ".", exist_ok=True)
    fout = open(args.out_labeled_jsonl, "w", encoding="utf-8")

    if args.k_candidates:
        k_candidates = [int(x) for x in args.k_candidates.split(",") if x.strip() != ""]
        k_candidates = [k for k in k_candidates if 0 <= k <= args.kmax]
        k_candidates = sorted(set(k_candidates))
        if 0 not in k_candidates:
            k_candidates = [0] + k_candidates
    else:
        k_candidates = list(range(args.kmax + 1))

    total_written = 0

    for eid, steps in tqdm(episodes.items(), desc="Label episodes"):
        T = len(steps)
        if T == 0:
            continue

        pred_next_states: List[str] = [""] * T
        for i in range(0, T, args.wm_batch_size):
            chunk = steps[i:i + args.wm_batch_size]
            states_chunk = [c["state"] for c in chunk]
            actions_chunk = [c["action"] for c in chunk]
            preds = wm.predict_next_states_batch(states_chunk, actions_chunk)
            for j, txt in enumerate(preds):
                pred_next_states[i + j] = txt

        obs_lists: List[List[str]] = []
        for t in range(T):
            obs0 = extract_observation(steps[t]["state"])
            obs_list = [obs0]
            for k in range(1, args.kmax + 1):
                idx = t + k - 1
                if idx < T:
                    obs_list.append(extract_observation(pred_next_states[idx]))
                else:
                    obs_list.append(obs_list[-1])
            obs_lists.append(obs_list)

        scores = torch.full((T, args.kmax + 1), -1e30, dtype=torch.float32)

        score_prompts: List[str] = []
        score_actions: List[str] = []
        meta: List[Tuple[int, int]] = []

        def flush():
            nonlocal score_prompts, score_actions, meta, scores
            if not score_prompts:
                return
            logp = compute_action_logprobs_batch(
                model=policy,
                tokenizer=tok_p,
                prompts=score_prompts,
                actions=score_actions,
                max_seq_len=args.max_seq_len,
                ctrl_token_id=ctrl_token_id,
                device=device,
                normalize=args.logprob_norm,
            ).detach().cpu()

            for i2, (t2, k2) in enumerate(meta):
                scores[t2, k2] = logp[i2].float()
            score_prompts, score_actions, meta = [], [], []

        for t in range(T):
            for k in k_candidates:
                obs_k = obs_lists[t][k]
                ptxt = build_prompt_for_action(steps[t]["state"], k, obs_k)
                score_prompts.append(ptxt)
                score_actions.append(steps[t]["action"])
                meta.append((t, k))
                if len(score_prompts) >= args.score_batch_size:
                    flush()
        flush()

        for t in range(T):
            best_k = 0
            best_val = -1e30
            for k in k_candidates:
                val = float(scores[t, k].item()) - args.lambda_k * float(k)
                if val > best_val:
                    best_val = val
                    best_k = k

            out_obj = {
                "episode_id": steps[t]["episode_id"],
                "step": int(steps[t]["step"]),
                "state": steps[t]["state"],
                "action": steps[t]["action"],
                "next_state": steps[t].get("next_state", ""),
                "k_label": int(best_k),
                "obs_list": obs_lists[t],
            }
            fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            total_written += 1

    fout.close()
    print(f"[label] wrote {total_written} samples -> {args.out_labeled_jsonl}")

def stage_sft(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    amp_enabled = (device.type == "cuda" and (args.fp16 or getattr(args, "bf16", False)))
    if device.type == "cuda" and getattr(args, "bf16", False):
        param_dtype = torch.bfloat16
        autocast_dtype = torch.bfloat16
    elif device.type == "cuda" and args.fp16:
        param_dtype = torch.float16
        autocast_dtype = torch.float16
    else:
        param_dtype = torch.float32
        autocast_dtype = torch.float16

    tokenizer, base_lm, policy, loaded_kmax = load_policy_with_heads(
        args.policy_model_path,
        device=device,
        kmax=args.kmax,
        torch_dtype=param_dtype,
        device_map=(args.device_map or None),
    )
    if loaded_kmax != args.kmax:
        print(f"[sft] Warning: loaded heads kmax={loaded_kmax}, args kmax={args.kmax} mismatch.")
        print("      Please keep kmax consistent across stages.")
        return

    tokenizer.padding_side = args.padding_side
    ctrl_token_id = tokenizer.convert_tokens_to_ids("<CTRL>")
    if ctrl_token_id is None or ctrl_token_id < 0:
        raise ValueError("Cannot find <CTRL> token id. Ensure SPECIAL_TOKENS are added correctly.")

    if args.gradient_checkpointing and hasattr(base_lm, "gradient_checkpointing_enable"):
        base_lm.gradient_checkpointing_enable()

    ds = LabeledKDataset(args.train_jsonl)
    cfg = CollateConfig(max_seq_len=args.max_seq_len, kmax=args.kmax)

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda b: collate_sft(b, tokenizer, cfg),
    )

    if args.train_only_heads:
        for p in base_lm.parameters():
            p.requires_grad = False
        train_params = list(policy.k_head.parameters()) + list(policy.v_head.parameters())
    else:
        for p in base_lm.parameters():
            p.requires_grad = True
        train_params = list(policy.parameters())

    opt = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=args.wd)

    scaler_enabled = False
    scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)

    policy.train()
    global_step = 0
    os.makedirs(args.out_dir, exist_ok=True)

    for epoch in range(args.epochs):
        pbar = tqdm(dl, desc=f"SFT epoch {epoch+1}/{args.epochs}")
        opt.zero_grad(set_to_none=True)
        accum_in_epoch = 0

        for batch in pbar:

            if args.device_map:
                input_device = base_lm.get_input_embeddings().weight.device
            else:
                input_device = device

            input_ids = batch["input_ids"].to(input_device)
            attention_mask = batch["attention_mask"].to(input_device)
            labels = batch["labels"].to(input_device)
            k_labels = batch["k_labels"].to(input_device)

            ctrl_pos = find_ctrl_positions(input_ids, attention_mask, ctrl_token_id)

            use_amp = amp_enabled
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=autocast_dtype):
                out = policy(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    ctrl_pos=ctrl_pos,
                    labels=labels,
                )
                lm_loss = out["lm_loss"]
                k_logits = out["k_logits"]
                if k_labels.device != k_logits.device:
                    k_labels = k_labels.to(k_logits.device)
                k_loss = F.cross_entropy(k_logits, k_labels, reduction="mean")
                loss = lm_loss + args.beta_k * k_loss

            if not torch.isfinite(loss):
                pbar.set_postfix({"loss": "nan/inf"})
                print("[sft] Non-finite loss detected, aborting this run to avoid corrupting weights.")
                return

            loss_val = float(loss.detach().cpu())
            pbar.set_postfix({
                "loss": f"{loss_val:.4f}",
                "lm": f"{float(lm_loss.detach().cpu()):.4f}",
                "k": f"{float(k_loss.detach().cpu()):.4f}",
            })

            scaled_loss = loss / args.grad_accum
            if scaler.is_enabled():
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            accum_in_epoch += 1
            global_step += 1

            if accum_in_epoch % args.grad_accum == 0:
                if scaler.is_enabled():
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)

            if args.save_every_steps > 0 and global_step % args.save_every_steps == 0:
                ckpt_dir = os.path.join(args.out_dir, f"ckpt_step_{global_step}")
                save_policy_with_heads(ckpt_dir, tokenizer, base_lm, policy, kmax=args.kmax)

        if accum_in_epoch % args.grad_accum != 0:
            if scaler.is_enabled():
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            opt.zero_grad(set_to_none=True)

        ckpt_dir = os.path.join(args.out_dir, f"ckpt_epoch_{epoch+1}")
        save_policy_with_heads(ckpt_dir, tokenizer, base_lm, policy, kmax=args.kmax)

    save_policy_with_heads(args.out_dir, tokenizer, base_lm, policy, kmax=args.kmax)
    print(f"[sft] saved trained policy -> {args.out_dir}")

def _clean_action_text(x: str) -> str:
    x = x.strip()

    x = re.sub(r"^\s*(Action\s*:)\s*", "", x, flags=re.IGNORECASE)
    x = x.split("\n")[0].strip()
    x = re.sub(r"\s+", " ", x)
    return x

def _project_to_admissible(action: str, admissible: Optional[List[str]]) -> str:
    if not admissible:
        return action

    flat: List[str] = []
    for c in admissible:
        if isinstance(c, str):
            flat.append(c)
        elif isinstance(c, (list, tuple)) and c:

            for x in c:
                if isinstance(x, str):
                    flat.append(x)
    if not flat:
        return action
    a = action.strip().lower()
    amap = {c.strip().lower(): c for c in flat}
    if a in amap:
        return amap[a]

    for c in flat:
        if c.strip().lower().startswith(a) or a.startswith(c.strip().lower()):
            return c

    for c in flat:
        if c.strip().lower() == "look":
            return c
    return flat[0]

@torch.no_grad()
def generate_action_text(
    base_lm: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> str:
    _ensure_pad(tokenizer)
    tok = tokenizer(
        [prompt],
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=1536,
        add_special_tokens=False,
    ).to(device)
    gen_kwargs: Dict[str, Any] = dict(
        **tok,
        do_sample=do_sample,
        num_beams=1,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    if do_sample:
        gen_kwargs["temperature"] = max(float(temperature), 1e-6)
        gen_kwargs["top_p"] = float(top_p)
    gen = base_lm.generate(**gen_kwargs)
    input_len = tok["input_ids"].shape[1]
    new_tokens = gen[0, input_len:]
    txt = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return _clean_action_text(txt)

@torch.no_grad()
def imagine_obs_k(
    policy_lm: AutoModelForCausalLM,
    tokenizer_policy: AutoTokenizer,
    wm: WorldModelWrapper,
    state_text: str,
    k: int,
    device: torch.device,
    imagine_action_max_new_tokens: int,
) -> str:
    if k <= 0:
        return extract_observation(state_text)

    sim_state = state_text
    for _ in range(k):
        obs0 = extract_observation(sim_state)
        prompt0 = build_prompt_for_action(sim_state, 0, obs0)

        a_hat = generate_action_text(
            base_lm=policy_lm,
            tokenizer=tokenizer_policy,
            prompt=prompt0,
            device=device,
            max_new_tokens=imagine_action_max_new_tokens,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
        )
        if not a_hat:
            break
        pred = wm.predict_next_states_batch([sim_state], [a_hat])[0]
        if not pred.strip():
            break
        sim_state = pred
    return extract_observation(sim_state)

def load_alfworld_env(
    alfworld_config_path: str,
    env_data_path: str,
    env_split: str,
    batch_size: int = 1,
):
    try:
        import yaml
        import alfworld
        import alfworld.agents.environment as environment
    except Exception as e:
        raise RuntimeError(
            "Failed to import alfworld/yaml. Please ensure `alfworld` and dependencies are installed. "
            f"Original error: {repr(e)}"
        )

    with open(alfworld_config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    base_root = env_data_path.rstrip("/")
    if os.path.basename(base_root) != "json_2.1.1":
        cand = os.path.join(base_root, "json_2.1.1")
        if os.path.exists(cand):
            base_root = cand

    os.environ["ALFWORLD_DATA"] = os.path.dirname(base_root)

    split_lower = env_split.lower()
    if split_lower in ["train"]:
        train_eval = "train"
        split_path = os.path.join(base_root, "train")
    elif split_lower in ["valid_seen", "dev", "seen"]:
        train_eval = "eval_in_distribution"
        split_path = os.path.join(base_root, "valid_seen")
    elif split_lower in ["valid_unseen", "test", "unseen"]:
        train_eval = "eval_out_of_distribution"
        split_path = os.path.join(base_root, "valid_unseen")
    else:
        raise ValueError(f"Unknown env_split={env_split}, expected train/dev(valid_seen)/test(valid_unseen).")

    if isinstance(config, dict) and "env" in config and isinstance(config["env"], dict):

        for k in ["data_path", "data_dir", "data_root"]:
            if k in config["env"]:
                config["env"][k] = env_data_path

        if "data_path" not in config["env"] and "data_dir" not in config["env"] and "data_root" not in config["env"]:
            config["env"]["data_path"] = env_data_path
        config["env"]["split"] = train_eval

        env_type = config["env"].get("type", None)
    else:
        raise ValueError("Invalid alfworld config yaml: expected dict with key `env`.")

    if env_type is None:
        raise ValueError("alfworld config must specify config['env']['type'] (e.g., 'AlfredTWEnv').")

    if hasattr(environment, "get_environment"):
        env_cls = environment.get_environment(env_type)
    else:
        if not hasattr(environment, env_type):
            raise ValueError(f"alfworld.agents.environment has no attribute `{env_type}`; check your config env.type.")
        env_cls = getattr(environment, env_type)

    if "dataset" not in config:
        config["dataset"] = {}

    if train_eval == "train":
        config["dataset"]["data_path"] = split_path
    else:
        config["dataset"]["data_path"] = os.path.join(base_root, "train")

    if train_eval == "eval_in_distribution":
        config["dataset"]["eval_id_data_path"] = split_path
    else:
        config["dataset"]["eval_id_data_path"] = os.path.join(base_root, "valid_seen")

    if train_eval == "eval_out_of_distribution":
        config["dataset"]["eval_ood_data_path"] = split_path
    else:
        config["dataset"]["eval_ood_data_path"] = os.path.join(base_root, "valid_unseen")

    env = env_cls(config, train_eval=train_eval)
    env = env.init_env(batch_size=batch_size)
    return env

def env_reset_1(env) -> Tuple[str, Dict[str, Any]]:
    out = env.reset()
    if isinstance(out, tuple) and len(out) == 2:
        obs, info = out
    else:
        raise RuntimeError(f"Unexpected env.reset() output: {type(out)}")

    if isinstance(obs, (list, tuple)):
        if len(obs) == 0:
            obs0 = ""
        elif len(obs) == 1:
            obs0 = obs[0]
        else:

            obs0 = "\n".join([str(x) for x in obs])
    else:
        obs0 = obs
    if isinstance(info, (list, tuple)):
        info0 = info[0] if len(info) > 0 else {}
    else:
        info0 = info
    obs_text = obs0 if isinstance(obs0, str) else str(obs0)
    info_dict = dict(info0) if isinstance(info0, dict) else {"info": info0}
    return obs_text, info_dict

def env_step_1(env, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
    try:
        out = env.step([action])
    except Exception:
        out = env.step(action)

    if not (isinstance(out, tuple) and len(out) == 4):
        raise RuntimeError(f"Unexpected env.step output: {type(out)}")

    obs, reward, done, info = out

    if isinstance(obs, (list, tuple)):
        if len(obs) == 0:
            obs0 = ""
        elif len(obs) == 1:
            obs0 = obs[0]
        else:
            obs0 = "\n".join([str(x) for x in obs])
    else:
        obs0 = obs
    if isinstance(reward, (list, tuple)):
        r0 = float(reward[0]) if len(reward) > 0 else 0.0
    else:
        r0 = float(reward)
    if isinstance(done, (list, tuple)):
        d0 = bool(done[0]) if len(done) > 0 else False
    else:
        d0 = bool(done)
    if isinstance(info, (list, tuple)):
        info0 = info[0] if len(info) > 0 else {}
    else:
        info0 = info

    obs_text = obs0 if isinstance(obs0, str) else str(obs0)
    info_dict = dict(info0) if isinstance(info0, dict) else {"info": info0}
    return obs_text, r0, d0, info_dict

def stage_rl_k(args: argparse.Namespace) -> None:

    def _pick_device(name: str) -> torch.device:
        dev_str = getattr(args, name, None) or ("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(dev_str)

    device_policy = _pick_device("policy_device")
    device_wm = _pick_device("wm_device")
    device = device_policy

    bf16 = bool(getattr(args, "bf16", False))
    fp16 = bool(getattr(args, "fp16", False))
    amp_enabled = (device_policy.type == "cuda" and (fp16 or bf16))
    if device_policy.type == "cuda" and bf16:
        policy_param_dtype = torch.bfloat16
        autocast_dtype = torch.bfloat16
    elif device_policy.type == "cuda" and fp16:
        policy_param_dtype = torch.float16
        autocast_dtype = torch.float16
    else:
        policy_param_dtype = torch.float32
        autocast_dtype = torch.float16

    tokenizer, base_lm, policy, loaded_kmax = load_policy_with_heads(
        args.policy_model_path,
        device=device_policy,
        kmax=args.kmax,
        torch_dtype=policy_param_dtype,
    )
    if loaded_kmax != args.kmax:

        print(f"[rl_k] kmax mismatch: loaded {loaded_kmax} vs args {args.kmax} -> override args.kmax={loaded_kmax}")
        args.kmax = int(loaded_kmax)

    tokenizer.padding_side = args.padding_side
    ctrl_token_id = tokenizer.convert_tokens_to_ids("<CTRL>")
    if ctrl_token_id is None or ctrl_token_id < 0:
        raise ValueError("Cannot find <CTRL> token id. Ensure SPECIAL_TOKENS are added correctly.")

    tok_wm = AutoTokenizer.from_pretrained(args.wm_model_path, trust_remote_code=True)
    if tok_wm.pad_token is None:
        tok_wm.pad_token = tok_wm.eos_token or tok_wm.unk_token
    tok_wm.padding_side = args.padding_side

    wm_lm = AutoModelForCausalLM.from_pretrained(
        args.wm_model_path,
        trust_remote_code=True,
        torch_dtype=(
            torch.bfloat16
            if (device_wm.type == "cuda" and bf16)
            else (torch.float16 if (device_wm.type == "cuda" and fp16) else torch.float32)
        ),
    ).to(device_wm)
    wm = WorldModelWrapper(
        wm_lm,
        tok_wm,
        device=device_wm,
        max_new_tokens=args.wm_max_new_tokens,
        autocast_dtype=(torch.bfloat16 if (device_wm.type == "cuda" and bf16) else torch.float16),
    )

    env = load_alfworld_env(
        alfworld_config_path=args.alfworld_config,
        env_data_path=args.env_data_path,
        env_split=args.env_split,
        batch_size=1,
    )

    if args.train_only_heads:
        for p in base_lm.parameters():
            p.requires_grad = False
        for p in policy.k_head.parameters():
            p.requires_grad = True
        for p in policy.v_head.parameters():
            p.requires_grad = True

        policy.k_head.to(dtype=torch.float32, device=device_policy)
        policy.v_head.to(dtype=torch.float32, device=device_policy)
        train_params = list(policy.k_head.parameters()) + list(policy.v_head.parameters())
    else:
        for p in base_lm.parameters():
            p.requires_grad = True
        for p in policy.k_head.parameters():
            p.requires_grad = True
        for p in policy.v_head.parameters():
            p.requires_grad = True
        train_params = list(policy.parameters())
    opt = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=args.wd)

    scaler_enabled = False
    scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)

    policy.train()
    global_step = 0
    os.makedirs(args.out_dir, exist_ok=True)

    def value_of_state(state_text: str) -> torch.Tensor:
        ctrl_prompt = build_prompt_for_controller(state_text)
        batch = make_controller_batch(tokenizer, [ctrl_prompt], args.max_seq_len)
        input_ids = batch["input_ids"].to(device_policy)
        attention_mask = batch["attention_mask"].to(device_policy)
        ctrl_pos = find_ctrl_positions(input_ids, attention_mask, ctrl_token_id)
        out = policy(
            input_ids=input_ids,
            attention_mask=attention_mask,
            ctrl_pos=ctrl_pos,
            labels=None,
        )
        return out["values"].view(-1)

    def sample_k_and_value(state_text: str) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        ctrl_prompt = build_prompt_for_controller(state_text)
        batch = make_controller_batch(tokenizer, [ctrl_prompt], args.max_seq_len)
        input_ids = batch["input_ids"].to(device_policy)
        attention_mask = batch["attention_mask"].to(device_policy)
        ctrl_pos = find_ctrl_positions(input_ids, attention_mask, ctrl_token_id)

        use_amp = amp_enabled
        with torch.cuda.amp.autocast(enabled=use_amp, dtype=autocast_dtype):
            out = policy(
                input_ids=input_ids,
                attention_mask=attention_mask,
                ctrl_pos=ctrl_pos,
                labels=None,
            )
            k_logits = out["k_logits"]
            v = out["values"].view(-1)
            if (not torch.isfinite(k_logits).all()) or (not torch.isfinite(v).all()):

                z = torch.zeros((1,), device=device_policy, dtype=torch.float32)
                return 0, z, z, z
            dist = torch.distributions.Categorical(logits=k_logits.float())
            k = int(dist.sample().item())
            logp_k = dist.log_prob(torch.tensor([k], device=device_policy))
            ent_k = dist.entropy()
        return k, v, logp_k.view(-1), ent_k.view(-1)

    for epoch in range(args.epochs):
        ep_returns: List[float] = []
        ep_success: List[int] = []

        pbar = tqdm(range(args.episodes_per_epoch), desc=f"Online A2C epoch {epoch+1}/{args.epochs}")

        opt.zero_grad(set_to_none=True)
        accum_steps = 0

        for _ in pbar:
            obs0, info0 = env_reset_1(env)
            mission = extract_task_from_text(obs0) or extract_task_from_text(str(info0)) or "unknown"
            history = ""
            state_text = build_state_text(mission, obs0, history)

            done = False
            ep_ret = 0.0
            ep_won = 0

            for t in range(args.max_steps):

                k, v_s, logp_k, ent_k = sample_k_and_value(state_text)

                obs_k = imagine_obs_k(
                    policy_lm=base_lm,
                    tokenizer_policy=tokenizer,
                    wm=wm,
                    state_text=state_text,
                    k=k,
                    device=device,
                imagine_action_max_new_tokens=args.imagine_action_max_new_tokens,
                )

                act_prompt = build_prompt_for_action(state_text, k, obs_k)
                a_raw = generate_action_text(
                    base_lm=base_lm,
                    tokenizer=tokenizer,
                    prompt=act_prompt,
                    device=device_policy,
                    max_new_tokens=args.action_max_new_tokens,
                    do_sample=bool(int(getattr(args, "action_do_sample", 1))),
                    temperature=args.action_temperature,
                    top_p=args.action_top_p,
                )

                admissible = None
                if isinstance(info0, dict):
                    admissible = info0.get("admissible_commands", None) or info0.get("admissible_actions", None)
                a_exec = _project_to_admissible(a_raw, admissible)

                try:
                    obs1, r_env, done, info1 = env_step_1(env, a_exec)
                    invalid = 0
                except Exception:

                    obs1, r_env, done, info1 = obs0, float(args.invalid_action_penalty), False, {}
                    invalid = 1

                won = 0
                if isinstance(info1, dict) and ("won" in info1):
                    won = int(bool(info1.get("won")))
                if done and won == 0 and r_env > 0:

                    won = 1

                r = float(r_env)
                if won:
                    r += float(args.success_bonus)
                r -= float(args.lambda_k) * float(k)
                r -= float(args.step_cost)

                ep_ret += r
                ep_won = max(ep_won, won)

                obs0, info0 = obs1, info1

                seg = f"Action: {a_exec}; Obs: {obs1}"
                if history.strip() == "":
                    history = seg
                else:
                    history = (history.strip() + " " + seg).strip()

                hist_keep_steps = int(getattr(args, "history_keep_steps", 6))
                if hist_keep_steps > 0:

                    parts = history.split(" Action: ")
                    if len(parts) > hist_keep_steps:
                        kept = parts[-hist_keep_steps:]

                        history = " Action: ".join(kept)
                        if not history.startswith("Action: "):
                            history = "Action: " + history
                next_state_text = build_state_text(mission, obs1, history)

                with torch.no_grad():
                    v_next = value_of_state(next_state_text)
                    if done:
                        v_next = torch.zeros_like(v_next)

                td = torch.tensor([r], device=device_policy, dtype=v_s.dtype) + args.gamma * v_next - v_s

                logp_a = compute_action_logprob_single_grad(
                    model=policy,
                    tokenizer=tokenizer,
                    prompt=act_prompt,
                    action=a_exec,
                    max_seq_len=args.max_seq_len,
                    ctrl_token_id=ctrl_token_id,
                    device=device_policy,
                    normalize=args.logprob_norm,
                )

                adv = td.detach()
                loss_action = -(adv * logp_a).mean()
                loss_k = -(adv * logp_k).mean()
                loss_value = 0.5 * (td ** 2).mean()
                loss_ent_k = -float(args.entropy_coef) * ent_k.mean()

                loss = loss_action + loss_k + float(args.value_coef) * loss_value + loss_ent_k

                scaled_loss = loss / max(int(args.update_every), 1)
                if scaler.is_enabled():
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                accum_steps += 1
                global_step += 1

                if accum_steps % max(int(args.update_every), 1) == 0:
                    if args.max_grad_norm > 0:
                        if scaler.is_enabled():
                            scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)

                    if scaler.is_enabled():
                        scaler.step(opt)
                        scaler.update()
                    else:
                        opt.step()
                    opt.zero_grad(set_to_none=True)

                obs0, info0 = obs1, info1
                state_text = next_state_text

                if done:
                    break

            ep_returns.append(ep_ret)
            ep_success.append(ep_won)

            pbar.set_postfix({
                "R(ep)": f"{ep_ret:.2f}",
                "SR": f"{(sum(ep_success)/max(1,len(ep_success))):.3f}",
                "k": f"{k}",
            })

        if accum_steps % max(int(args.update_every), 1) != 0:
            if args.max_grad_norm > 0:
                if scaler.is_enabled():
                    scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
            if scaler.is_enabled():
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            opt.zero_grad(set_to_none=True)

        ckpt_dir = os.path.join(args.out_dir, f"ckpt_epoch_{epoch+1}")
        save_policy_with_heads(ckpt_dir, tokenizer, base_lm, policy, kmax=args.kmax)

        mean_ret = sum(ep_returns) / max(1, len(ep_returns))
        sr = sum(ep_success) / max(1, len(ep_success))
        print(f"[rl_k][epoch {epoch+1}] mean_return={mean_ret:.3f}  success_rate={sr:.3f}")

    save_policy_with_heads(args.out_dir, tokenizer, base_lm, policy, kmax=args.kmax)
    print(f"[rl_k] saved ONLINE A2C tuned policy -> {args.out_dir}")

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    def add_common_padding_and_norm(p_: argparse.ArgumentParser) -> None:
        p_.add_argument("--padding_side", type=str, default="left", choices=["left", "right"])
        p_.add_argument(
            "--logprob_norm",
            type=str,
            default="sum",
            choices=["sum", "mean"],
            help="How to aggregate action-token logprob for labeling/RL reward.",
        )

    p = sub.add_parser("label")
    p.add_argument("--expert_jsonl", type=str, required=True)
    p.add_argument("--policy_model_path", type=str, required=True)
    p.add_argument("--wm_model_path", type=str, required=True)
    p.add_argument("--out_labeled_jsonl", type=str, required=True)

    p.add_argument("--kmax", type=int, default=5)
    p.add_argument("--lambda_k", type=float, default=0.2)

    p.add_argument("--wm_batch_size", type=int, default=4)
    p.add_argument("--wm_max_new_tokens", type=int, default=192)

    p.add_argument("--score_batch_size", type=int, default=48)
    p.add_argument("--max_seq_len", type=int, default=1536)

    p.add_argument("--k_candidates", type=str, default="")
    add_common_padding_and_norm(p)

    p = sub.add_parser("sft")
    p.add_argument("--train_jsonl", type=str, required=True)
    p.add_argument("--policy_model_path", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)

    p.add_argument("--kmax", type=int, default=5)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=8)

    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--wd", type=float, default=0.0)
    p.add_argument("--beta_k", type=float, default=1.0)

    p.add_argument("--max_seq_len", type=int, default=1536)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true", help="Use bf16 weights + bf16 autocast (recommended on A100).")
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument(
        "--device_map",
        type=str,
        default="",
        help="Optional HF device_map for model-parallel loading, e.g. 'auto' (use with CUDA_VISIBLE_DEVICES=0,1).",
    )

    p.add_argument("--train_only_heads", action="store_true")
    p.add_argument("--save_every_steps", type=int, default=0)
    p.add_argument("--padding_side", type=str, default="left", choices=["left", "right"])

    p = sub.add_parser("rl_k")
    p.add_argument("--policy_model_path", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)

    p.add_argument("--alfworld_config", type=str, required=True, help="Path to ALFWorld config yaml.")
    p.add_argument("--env_data_path", type=str, required=True, help="Path to ALFWorld data json_2.1.1 root.")
    p.add_argument("--env_split", type=str, default="valid_seen", help="e.g. train, valid_seen, valid_unseen.")

    p.add_argument("--wm_model_path", type=str, required=True)
    p.add_argument("--wm_max_new_tokens", type=int, default=192)

    p.add_argument("--kmax", type=int, default=5)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--episodes_per_epoch", type=int, default=50)
    p.add_argument("--max_steps", type=int, default=50)
    p.add_argument("--policy_device", type=str, default="cuda:0")
    p.add_argument("--wm_device", type=str, default="cuda:1")

    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--wd", type=float, default=0.0)
    p.add_argument("--train_only_heads", action="store_true")

    p.add_argument("--lambda_k", type=float, default=0.2)
    p.add_argument("--step_cost", type=float, default=0.0)
    p.add_argument("--success_bonus", type=float, default=0.0)
    p.add_argument("--invalid_action_penalty", type=float, default=-0.1)

    p.add_argument("--entropy_coef", type=float, default=0.01)
    p.add_argument("--value_coef", type=float, default=1.0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    p.add_argument("--action_max_new_tokens", type=int, default=16)
    p.add_argument("--action_temperature", type=float, default=0.8)
    p.add_argument("--action_top_p", type=float, default=0.9)
    p.add_argument(
        "--action_do_sample",
        type=int,
        default=1,
        choices=[0, 1],
        help="1=sample actions (more exploration), 0=greedy actions (more stable/less invalid).",
    )

    p.add_argument("--imagine_action_max_new_tokens", type=int, default=12)

    p.add_argument("--max_seq_len", type=int, default=1536)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true", help="Use bf16 weights + bf16 autocast (recommended on A100).")
    p.add_argument("--update_every", type=int, default=1, help="Optimizer step every N env steps (grad-accum).")
    p.add_argument(
        "--history_keep_steps",
        type=int,
        default=6,
        help="How many recent (action,obs) pairs to keep in History during rl_k. 0 disables truncation.",
    )

    add_common_padding_and_norm(p)

    args = parser.parse_args()
    set_seed(42)

    if args.cmd == "label":
        stage_label(args)
    elif args.cmd == "sft":
        stage_sft(args)
    elif args.cmd == "rl_k":
        stage_rl_k(args)
    else:
        raise ValueError(f"Unknown cmd: {args.cmd}")

if __name__ == "__main__":
    main()
