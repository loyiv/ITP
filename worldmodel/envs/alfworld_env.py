from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from alfworld.agents.environment import get_environment


class AlfworldEnvWrapper:
    def __init__(
        self,
        data_root: str,
        split: str = "valid_seen",
        max_steps: int = 50,
        seed: int = 42,
    ):
        self.data_root = Path(data_root)
        if not self.data_root.exists():
            raise FileNotFoundError(f"ALFWorld data root not found: {self.data_root}")

        self.split = split
        self.max_steps = max_steps
        self.seed = seed

        self._t = 0
        self._mission: str = ""
        self._history: List[Tuple[str, str]] = []
        self._episode_id: Optional[str] = None
        self._last_info: Dict[str, Any] = {}

        config, train_eval = self._build_config(split, max_steps, seed)
        env_cls = get_environment(config["env"]["type"])
        manager = env_cls(config, train_eval=train_eval)
        self._game_files = list(getattr(manager, "game_files", []))
        self._env = manager.init_env(batch_size=1)

    def num_episodes(self) -> int:
        return len(self._game_files)

    def reset(self, episode_id: Optional[str] = None) -> Tuple[Dict[str, str], Dict[str, Any]]:
        obs_text, info = self._reset_until_episode(episode_id)
        obs_dict = self._format_obs(obs_text)
        self._mission = obs_dict["mission"]
        self._history = []
        self._t = 0
        self._episode_id = info.get("scene_id")
        self._last_info = info
        return obs_dict, info

    def step(self, action: str) -> Tuple[Dict[str, str], float, bool, Dict[str, Any]]:
        obs_raw, rewards, dones, infos = self._env.step([action])
        obs_text = self._extract_text(obs_raw)
        reward = float(self._extract_scalar(rewards))
        done_flag = bool(self._extract_scalar(dones))
        info = self._normalize_info(infos)

        self._t += 1
        obs_dict = self._format_obs(obs_text)
        self._history.append((action, obs_dict["text"]))
        self._mission = obs_dict["mission"] or self._mission
        self._episode_id = info.get("scene_id", self._episode_id)
        self._last_info = info

        done = done_flag or (self._t >= self.max_steps)
        return obs_dict, reward, done, info

    def get_admissible_commands(self) -> List[str]:
        try:
            admissible = self._env.get_admissible_commands()
        except Exception:
            admissible = []
        if not admissible:
            return []
        if isinstance(admissible, (list, tuple)):
            return [str(x).strip() for x in admissible if str(x).strip()]
        return [str(admissible).strip()]

    def success(self, info: Dict[str, Any]) -> bool:
        if not info:
            return False
        if "won" in info:
            return bool(info.get("won"))
        if "won" in self._last_info:
            return bool(self._last_info.get("won"))
        return False

    def encode_state(self, obs: Dict[str, str]) -> Dict[str, str]:
        return dict(obs or {})

    def stringify_state(self, state: Dict[str, str]) -> str:
        mission = (state.get("mission") or "").strip()
        text = (state.get("text") or "").strip()
        inv = (state.get("inventory") or "").strip()
        parts = []
        if mission:
            parts.append(f"Mission: {mission}")
        if text:
            parts.append(f"Observation: {text}")
        if inv:
            parts.append(f"Inventory: {inv}")
        return "\n".join(parts).strip()

    def _build_config(self, split: str, max_steps: int, seed: int) -> Tuple[Dict[str, Any], str]:
        repo_root = None
        config_path = self.data_root / "base_config.yaml"
        if not config_path.exists():
            repo_root = self._resolve_repo_root()
            config_path = repo_root / "configs" / "base_config.yaml"
            if not config_path.exists():
                raise FileNotFoundError(f"ALFWorld config not found: {config_path}")
        with config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        split_mode, split_path = self._resolve_split(split)
        self._ensure_dir(split_path)

        base = (self.data_root / "json_2.1.1") if (self.data_root / "json_2.1.1").exists() else self.data_root
        config["dataset"]["data_path"] = str(base / "train")
        config["dataset"]["eval_id_data_path"] = str(base / "valid_seen")
        config["dataset"]["eval_ood_data_path"] = str(base / "valid_unseen")

        if split_mode == "train":
            config["dataset"]["data_path"] = str(split_path)
        elif split_mode == "eval_in_distribution":
            config["dataset"]["eval_id_data_path"] = str(split_path)
        else:
            config["dataset"]["eval_ood_data_path"] = str(split_path)

        logic_dir = self._resolve_logic_dir(repo_root) if repo_root is not None else self._resolve_logic_dir(self.data_root)
        config["logic"]["domain"] = str(logic_dir / "alfred.pddl")
        config["logic"]["grammar"] = str(logic_dir / "alfred.twl2")
        config["general"]["random_seed"] = seed
        config["rl"]["training"]["max_nb_steps_per_episode"] = max_steps
        config["dagger"]["training"]["max_nb_steps_per_episode"] = max_steps
        return config, split_mode

    def _resolve_repo_root(self) -> Path:
        candidates: List[Path] = []
        repo_env = os.environ.get("ALFWORLD_REPO_ROOT")
        if repo_env:
            candidates.append(Path(repo_env))
        candidates.extend(
            [
                Path(__file__).resolve().parents[2] / "alfworld" / "alfworld-master",
                Path(__file__).resolve().parents[2] / "alfworld-master",
                self.data_root.parents[1] / "alfworld-master",
                self.data_root.parents[2] / "alfworld-master",
            ]
        )
        for path in candidates:
            if path and path.exists():
                return path
        raise FileNotFoundError("Unable to locate a local ALFWorld repository. Set ALFWORLD_REPO_ROOT to the repo root.")

    def _resolve_logic_dir(self, repo_root: Path) -> Path:
        candidates = [
            self.data_root / "logic",
            self.data_root.parent / "logic",
            repo_root / "alfworld" / "data",
            repo_root / "data",
        ]
        for path in candidates:
            domain = path / "alfred.pddl"
            grammar = path / "alfred.twl2"
            if domain.exists() and grammar.exists():
                return path
        raise FileNotFoundError("alfred.pddl/alfred.twl2 not found in expected locations.")

    def _resolve_split(self, split: str) -> Tuple[str, Path]:
        s = (split or "").lower()
        base = (self.data_root / "json_2.1.1") if (self.data_root / "json_2.1.1").exists() else self.data_root
        if s == "train":
            return "train", base / "train"
        if s in {"valid_seen", "eval_seen", "seen", "eval_in_distribution"}:
            return "eval_in_distribution", base / "valid_seen"
        if s in {"valid_unseen", "eval_unseen", "unseen", "eval_out_of_distribution"}:
            return "eval_out_of_distribution", base / "valid_unseen"
        if s in {"valid_train", "train_valid"}:
            return "train", base / "valid_train"
        return "eval_in_distribution", base / "valid_seen"

    def _reset_until_episode(self, target_episode: Optional[str]) -> Tuple[str, Dict[str, Any]]:
        attempts = max(1, len(self._game_files) or 1)
        for _ in range(attempts):
            obs_raw, info_raw = self._env.reset()
            obs_text = self._extract_text(obs_raw)
            info = self._normalize_info(info_raw)
            scene_id = info.get("scene_id")
            if not target_episode or scene_id == target_episode:
                return obs_text, info
        raise ValueError(f"Episode '{target_episode}' not found in current split.")

    @staticmethod
    def _extract_text(value: Any) -> str:
        if isinstance(value, (list, tuple)):
            return AlfworldEnvWrapper._extract_text(value[0])
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="ignore")
        return str(value)

    @staticmethod
    def _extract_scalar(value: Any) -> Any:
        if isinstance(value, (list, tuple)):
            return AlfworldEnvWrapper._extract_scalar(value[0])
        return value

    def _normalize_info(self, infos: Any) -> Dict[str, Any]:
        if isinstance(infos, (list, tuple)):
            if not infos:
                return {}
            v = infos[0]
        else:
            v = infos
        if isinstance(v, dict):
            return dict(v)
        return {}

    def _format_obs(self, obs_text: str) -> Dict[str, str]:
        text = (obs_text or "").strip()
        mission = ""
        inventory = ""
        if "Your task is to:" in text:
            parts = text.split("Your task is to:", 1)
            mission = parts[1].strip().split("\n")[0].strip()
        if "Inventory:" in text:
            inv = text.split("Inventory:", 1)[1]
            inventory = inv.strip().split("\n")[0].strip()
        return {"mission": mission, "text": text, "inventory": inventory}

    @staticmethod
    def _ensure_dir(path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"ALFWorld split path not found: {path}")

