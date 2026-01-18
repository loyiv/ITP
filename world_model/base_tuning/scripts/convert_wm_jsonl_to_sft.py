#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import os
from typing import Dict, Any, Iterable

DEFAULT_INSTRUCTION = "You are a world model. Predict the NEXT STATE textually."
DEFAULT_USER_SUFFIX = "Please write the NEXT STATE (observation, inventory, brief outcome)."

def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                raise ValueError(f"Invalid JSON at {path}:{ln}: {e}") from e

def build_user_value(state: str, action: str, user_suffix: str) -> str:
    return (
        "STATE:\n" + str(state) + "\n\n"
        + "ACTION:\n" + str(action) + "\n\n"
        + str(user_suffix)
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="输入 jsonl（每行包含 state/action/next_state）")
    ap.add_argument("--output", required=True, help="输出 jsonl（SFT conversations 格式）")
    ap.add_argument("--instruction", default=DEFAULT_INSTRUCTION, help="写入到样本的 instruction 字段")
    ap.add_argument("--user_suffix", default=DEFAULT_USER_SUFFIX, help="human 消息末尾的提示词")
    ap.add_argument(
        "--dialog_id_mode",
        default="episode_step",
        choices=["episode_step", "index"],
        help="dialog_id 生成方式：episode_step=episode_id_step；index=递增编号",
    )
    args = ap.parse_args()

    inp = args.input
    out = args.output
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)

    n = 0
    with open(out, "w", encoding="utf-8") as wf:
        for item in iter_jsonl(inp):
            if not all(k in item for k in ("state", "action", "next_state")):
                raise ValueError(
                    f"Missing keys in sample (need state/action/next_state). got keys={list(item.keys())}"
                )

            episode_id = str(item.get("episode_id", ""))
            step = item.get("step", None)

            if args.dialog_id_mode == "episode_step" and episode_id and step is not None:
                dialog_id = f"{episode_id}_{step}"
            elif args.dialog_id_mode == "episode_step" and episode_id:
                dialog_id = episode_id
            else:
                dialog_id = str(n)

            user_val = build_user_value(item["state"], item["action"], args.user_suffix)
            gpt_val = "" if item["next_state"] is None else str(item["next_state"])

            out_item = {
                "dialog_id": dialog_id,
                "turn_id": int(step) if isinstance(step, int) else 0,
                "instruction": args.instruction,
                "conversations": [
                    {"from": "human", "value": user_val},
                    {"from": "gpt", "value": gpt_val},
                ],
                "metadata": {
                    "episode_id": episode_id,
                    "step": step,
                    "source": os.path.basename(inp),
                },
            }
            wf.write(json.dumps(out_item, ensure_ascii=False) + "\n")
            n += 1

    print(f"[OK] converted {n} samples -> {out}")

if __name__ == "__main__":
    main()

