from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Any
from dataclasses import asdict

def to_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    if hasattr(obj, "__dataclass_fields__"):
        return to_serializable(asdict(obj))
    if hasattr(obj, "item"):
        return obj.item()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)

def save_split_outputs(
    split_results: Dict[str, Dict[str, Any]],
    args,
) -> Path:
    base_dir = Path(args.output_dir)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = base_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "timestamp": timestamp,
        "config": {
            "episodes": args.episodes,
            "max_steps": args.max_steps,
            "env_data_root": args.env_data_root,
            "splits": [s.strip() for s in args.splits.split(",") if s.strip()],
            "policy_model": args.policy_model,
            "wm_model": args.wm_model,
        },
        "splits": {},
    }

    for split, result in split_results.items():
        episodes = result["episodes"]
        successes = result["successes"]
        success_rate = result["success_rate"]

        split_dir = run_dir / split
        split_dir.mkdir(exist_ok=True)
        episode_files = []
        for idx, episode in enumerate(episodes):
            data = to_serializable(episode)
            ep_path = split_dir / f"episode_{idx:03d}_{episode.episode_id}.json"
            with ep_path.open("w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            episode_files.append(str(ep_path))

        summary["splits"][split] = {
            "episodes": len(episodes),
            "successes": successes,
            "success_rate": success_rate,
            "total_available": result.get("total_available", len(episodes)),
            "episode_files": episode_files,
        }

    summary_path = run_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[Output] Saved trajectories to {run_dir}")
    return summary_path

