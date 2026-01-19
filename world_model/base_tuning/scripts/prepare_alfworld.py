#!/usr/bin/env python3
import argparse
import json
import pathlib
from typing import Any, Dict, List, Tuple

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize ALFWorld SFT data for FastChat fine-tuning.",
    )
    parser.add_argument(
        "--input",
        type=pathlib.Path,
        default=pathlib.Path("data/alfworld_sft.json"),
        help="Path to the raw ALFWorld SFT JSON file.",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("data/alfworld_sft_processed.json"),
        help="Where to store the processed JSON file.",
    )
    parser.add_argument(
        "--min-turns",
        type=int,
        default=2,
        help="Skip dialogues with fewer than this number of human/assistant turns.",
    )
    parser.add_argument(
        "--keep-system-turn",
        action="store_true",
        help="Keep the very first human instruction turn inside conversations.",
    )
    return parser.parse_args()

def extract_instruction_and_dialogue(
    conversations: List[Dict[str, Any]],
    keep_system_turn: bool,
) -> Tuple[str, List[Dict[str, str]]]:
    conv = [dict(turn) for turn in conversations]
    instruction = ""

    if conv and conv[0].get("from") == "human":
        instruction = conv[0].get("value", "").strip()
        if not keep_system_turn:
            conv = conv[1:]
    else:
        instruction = ""

    while conv and conv[0].get("from") != "human":
        conv.pop(0)
    while conv and conv[-1].get("from") != "gpt":
        conv.pop()

    for turn in conv:
        text = turn.get("value", "")
        turn["value"] = text.strip()

    return instruction, conv

def main():
    args = parse_args()
    raw_path = args.input.expanduser().resolve()
    out_path = args.output.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with raw_path.open("r", encoding="utf-8") as f:
        raw_data = json.load(f)

    processed = []
    skipped = 0

    for idx, sample in enumerate(raw_data):
        turns = sample.get("conversations", [])
        instruction, conv = extract_instruction_and_dialogue(
            turns,
            keep_system_turn=args.keep_system_turn,
        )

        if len(conv) < args.min_turns:
            skipped += 1
            continue

        dialog_id = str(sample.get("id", idx))
        record: Dict[str, Any] = {
            "dialog_id": dialog_id,
            "instruction": instruction,
            "conversations": conv,
        }

        game_file = sample.get("game_file")
        if game_file:
            record["metadata"] = {"game_file": game_file}

        processed.append(record)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)

    print(
        f"Processed {len(processed)} dialogues, skipped {skipped}, "
        f"saved to {out_path}"
    )

if __name__ == "__main__":
    main()

