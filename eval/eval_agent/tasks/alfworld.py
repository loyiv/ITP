import os
import json
import yaml
import logging
from typing import Iterable, Tuple, Optional, List, Any

import alfworld
import alfworld.agents.environment as envs

from eval_agent.tasks.base import Task

logger = logging.getLogger("agent_frame")

PREFIXES = {
    "pick_and_place": "put",
    "pick_clean_then_place": "clean",
    "pick_heat_then_place": "heat",
    "pick_cool_then_place": "cool",
    "look_at_obj": "examine",
    "pick_two_obj": "puttwo",
}

class AlfWorldTask(Task):

    task_name = "alfworld"

    def __init__(
        self,
        game_file: str,
        env: Any,
        obs: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.game_file = game_file
        self.observation = obs or ""

        self.env = env

    @classmethod
    def load_tasks(cls, split: str, part_num: int, part_idx: int = -1, batch_size: int = 1) -> Tuple[Iterable[Task], int]:
        os.environ["ALFWORLD_DATA"] = "eval_agent/data/alfworld"
        alfworld_data_path = os.environ.get("ALFWORLD_DATA")

        with open(os.path.join(alfworld_data_path, "base_config.yaml")) as f:
            config = yaml.safe_load(f)

        if split == 'train':
            split_name = "train"
        elif split == 'dev':
            split_name = "eval_in_distribution"
        elif split == 'test':
            split_name = "eval_out_of_distribution"
        else:
            raise ValueError(f"Unknown split: {split}. Expected one of: train/dev/test.")

        env_type = config["env"]["type"]
        env_cls = getattr(envs, env_type, None)
        if env_cls is None and hasattr(envs, "get_environment"):
            env_cls = envs.get_environment(env_type)
        if env_cls is None:
            raise AttributeError(f"alfworld.agents.environment has no env type: {env_type}")

        env = env_cls(config, train_eval=split_name)
        env = env.init_env(batch_size=batch_size)

        gamefiles: List[str] = list(getattr(env, "gamefiles", []))
        if not gamefiles:
            raise RuntimeError("Failed to load ALFWorld gamefiles from environment.")

        n_tasks_total = len(gamefiles)

        if part_num > 1:
            assert part_idx != -1
            per_part_num = n_tasks_total // part_num + 1
            start = per_part_num * part_idx
            end = min(start + per_part_num, n_tasks_total)
            split_indices = list(range(start, end))
        else:
            split_indices = list(range(n_tasks_total))

        def generator():
            for idx in split_indices:
                game_file = gamefiles[idx]
                yield cls(
                    task_id=idx,
                    game_file=game_file,
                    env=env,
                    obs="",
                )

        return generator(), len(split_indices)
