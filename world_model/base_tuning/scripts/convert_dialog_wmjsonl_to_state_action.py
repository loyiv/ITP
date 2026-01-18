import argparse
import json
from pathlib import Path

def parse_state_action(human_text: str):
    if "\n\nACTION:\n" not in human_text:
        return None, None
    state_part, rest = human_text.split("\n\nACTION:\n", 1)

    action_part = rest.split("\n\n", 1)[0].strip()
    state_clean = state_part.strip()
    return state_clean, action_part

def convert(input_path: Path, output_path: Path):
    total = 0
    kept = 0
    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)
            except Exception:
                continue
            convs = obj.get("conversations", [])
            if len(convs) < 2:
                continue
            human_text = convs[0].get("value", "")
            assistant_text = convs[1].get("value", "")
            state, action = parse_state_action(human_text)
            if not state or not action:
                continue
            rec = {
                "state": state,
                "action": action,
                "next_state": assistant_text.strip(),
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1
    print(f"Done. parsed={total}, kept={kept}, output={output_path}")

def main():
    ap = argparse.ArgumentParser(description="将 worldmodel 对话格式 JSONL 转成 state/action/next_state 格式")
    ap.add_argument("--input", type=Path, required=True, help="输入 JSONL，形如 wm_train_sciworld.jsonl（含 conversations）")
    ap.add_argument("--output", type=Path, required=True, help="输出 JSONL（含 state/action/next_state）")
    args = ap.parse_args()
    convert(args.input, args.output)

if __name__ == "__main__":
    main()

