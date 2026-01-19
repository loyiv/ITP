#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True, help="Path to alfworld_sft.json")
    p.add_argument("--output", type=Path, required=True, help="Output transition JSONL")
    p.add_argument("--max_episodes", type=int, default=0, help="If >0, only convert first N episodes")
    p.add_argument("--max_steps", type=int, default=0, help="If >0, stop after writing N steps total")
    return p.parse_args()

def _get_episode_id(obj: Dict[str, Any], fallback_idx: int) -> str:
    for k in ("game_file", "episode_id", "dialog_id", "id"):
        v = obj.get(k)
        if v is not None and str(v).strip() != "":
            return str(v)
    return str(fallback_idx)

def _extract_action(action_text: str) -> Optional[str]:
    t = action_text.strip()
    if not t:
        return None
    if t.lower() in {"ok", "okay"}:
        return None

    if "Action:" in t:
        lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
        action_lines = [ln for ln in lines if ln.startswith("Action:")]
        if action_lines:
            act = action_lines[-1].split("Action:", 1)[-1].strip()
            return act or None
        act = t.rsplit("Action:", 1)[-1].strip()
        return act or None

    allowed_prefixes = (
        "go to ",
        "open ",
        "close ",
        "take ",
        "put ",
        "toggle ",
        "clean ",
        "heat ",
        "cool ",
        "task ",
        "examine ",
        "look",
        "inventory",
    )
    if t.lower().startswith(allowed_prefixes):
        return t
    return None

def iter_transitions(sample: Dict[str, Any], episode_id: str) -> Iterable[Tuple[int, str, str, str]]:
    conv: List[Dict[str, Any]] = sample.get("conversations", []) or []

    turns: List[Tuple[str, str]] = []
    for t in conv:
        frm = str(t.get("from", "")).strip()
        val = str(t.get("value", "")).strip()
        if frm and val:
            turns.append((frm, val))

    step = 0
    i = 0
    while i + 2 < len(turns):
        frm0, state = turns[i]
        frm1, action = turns[i + 1]
        frm2, next_state = turns[i + 2]
        if frm0 == "human" and frm1 == "gpt" and frm2 == "human":
            act = _extract_action(action)
            if act is not None:
                yield (step, state, act, next_state)
                step += 1
            i += 2
        else:
            i += 1

def main() -> None:
    args = parse_args()
    inp = args.input.expanduser().resolve()
    outp = args.output.expanduser().resolve()
    outp.parent.mkdir(parents=True, exist_ok=True)

    with inp.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected a JSON list at top-level.")

    written = 0
    with outp.open("w", encoding="utf-8") as fo:
        for epi_idx, sample in enumerate(data):
            if args.max_episodes and epi_idx >= args.max_episodes:
                break
            if not isinstance(sample, dict):
                continue
            episode_id = _get_episode_id(sample, epi_idx)
            for step, state, action, next_state in iter_transitions(sample, episode_id):
                obj = {
                    "episode_id": episode_id,
                    "step": step,
                    "state": state,
                    "action": action,
                    "next_state": next_state,
                }
                fo.write(json.dumps(obj, ensure_ascii=False) + "\n")
                written += 1
                if args.max_steps and written >= args.max_steps:
                    break
            if args.max_steps and written >= args.max_steps:
                break

    print(f"[convert] wrote {written} transitions -> {outp}")

if __name__ == "__main__":
    main()

