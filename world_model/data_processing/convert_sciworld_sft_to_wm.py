import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple

def extract_mission(conversations: List[Dict[str, Any]]) -> str:
    for msg in conversations:
        if msg.get("from") == "human":
            value = msg.get("value", "")
            if "Your task is" in value or "task is" in value:
                return value.strip()

    for msg in conversations:
        if msg.get("from") == "human":
            return msg.get("value", "").strip()
    return ""

def extract_action(text: str) -> str:
    for line in text.splitlines():
        if line.strip().startswith("Action:"):
            return line.split("Action:", 1)[1].strip()
    return text.strip()

def extract_observation(text: str) -> str:
    if "Observation:" in text:
        return text.split("Observation:", 1)[1].strip()
    return text.strip()

def build_history(pairs: List[Tuple[str, str]]) -> str:
    parts = []
    for action, obs in pairs:
        obs_clean = obs.strip()
        if not obs_clean.endswith("."):
            obs_clean += "."
        parts.append(f"Action: {action}; Obs: {obs_clean}")
    return " ".join(parts)

def convert_episode(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    conversations = sample.get("conversations", [])
    mission = extract_mission(conversations)

    pairs: List[Tuple[str, str]] = []
    for i, msg in enumerate(conversations):
        if msg.get("from") == "gpt":
            action = extract_action(msg.get("value", ""))

            if i + 1 < len(conversations):
                obs = extract_observation(conversations[i + 1].get("value", ""))
                pairs.append((action, obs))

    if len(pairs) < 2:
        return []

    episode_id = str(sample.get("id", "episode"))
    outputs = []

    for step in range(1, len(pairs)):
        prev_obs = pairs[step - 1][1]
        history_before = build_history(pairs[:step])
        action = pairs[step][0]
        next_obs = pairs[step][1]
        history_after = build_history(pairs[: step + 1])

        human_text = (
            f"STATE:\n"
            f"Mission: {mission}\n"
            f"Observation: {prev_obs}\n"
            f"History: {history_before}\n\n"
            f"ACTION:\n{action}\n\n"
            f"Please write the NEXT STATE (observation, inventory, brief outcome)."
        )

        gpt_text = (
            f"Mission: {mission}\n"
            f"Observation: {next_obs}\n"
            f"History: {history_after}"
        )

        outputs.append(
            {
                "dialog_id": f"{episode_id}_{step}",
                "turn_id": step,
                "instruction": "You are a world model. Predict the NEXT STATE textually.",
                "conversations": [
                    {"from": "human", "value": human_text},
                    {"from": "gpt", "value": gpt_text},
                ],
                "metadata": {
                    "episode_id": episode_id,
                    "step": step,
                    "source": "sciworld_sft.json",
                },
            }
        )
    return outputs

def main():
    parser = argparse.ArgumentParser(
        description="将 sciworld_sft.json 转成 wm_train 格式（类似 wm_train_alf.jsonl）。"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="输入 SFT json 路径（建议使用相对路径或你自己的数据目录）。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="输出 jsonl 路径（wm_train_sciworld.jsonl）。",
    )
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8") as f:
        data = json.load(f)

    all_lines: List[Dict[str, Any]] = []
    for sample in data:
        all_lines.extend(convert_episode(sample))

    with args.output.open("w", encoding="utf-8") as f:
        for line in all_lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    print(f"Done. Wrote {len(all_lines)} lines to {args.output}")

if __name__ == "__main__":
    main()

