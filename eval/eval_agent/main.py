from __future__ import annotations

import os
import json
import logging
import pathlib
import argparse
from typing import List, Dict, Any, TYPE_CHECKING
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

try:
    from colorama import Fore
except Exception:
    class _Fore:
        BLACK = ""
        RED = ""
        GREEN = ""
        YELLOW = ""
        BLUE = ""
        MAGENTA = ""
        CYAN = ""
        WHITE = ""
        RESET = ""

    Fore = _Fore()
import csv

from eval_agent.utils.datatypes import State

if TYPE_CHECKING:
    import eval_agent.tasks as tasks
    import eval_agent.agents as agents
    import eval_agent.envs as envs

logger = logging.getLogger("agent_frame")

def interactive_loop(
    task: tasks.Task,
    agent: agents.LMAgent,
    env_config: Dict[str, Any],
) -> State:

    import eval_agent.envs as envs

    logger.info(f"Loading environment: {env_config['env_class']}")
    env: envs.BaseEnv = getattr(envs, env_config["env_class"])(task, **env_config)

    observation, state = env.reset()

    init_msg = observation

    logger.info(f"\n{Fore.YELLOW}{init_msg}{Fore.RESET}")

    cur_step = 1
    while not state.finished:
        logger.info(f"\n{Fore.RED}Step {cur_step}{Fore.RESET}\n")
        cur_step += 1

        try:
            llm_output: str = agent(state.history)

            logger.info(
                f"\n{Fore.GREEN}{llm_output}{Fore.RESET}\n"
            )
        except Exception as e:
            logger.info(f"Agent failed with error: {e}")
            state.success = False
            state.finished = True
            state.terminate_reason = "exceeding maximum input length"
            break

        observation, state = env.step(llm_output)

        if not state.finished:

            logger.info(
                f"\n{Fore.BLUE}{observation}{Fore.RESET}\n"
            )

        if state.finished:
            break

    if state.reward is not None:
        logger.info(
            f"Task finished in {state.steps} steps. Success: {state.success}. Reward: {state.reward}"
        )
    else:
        logger.info(
            f"Task finished in {state.steps} steps. Success: {state.success}"
        )

    return state

def main(args: argparse.Namespace):

    import eval_agent.tasks as tasks
    import eval_agent.agents as agents
    import eval_agent.envs as envs

    with open(os.path.join(args.exp_path, f"{args.exp_config}.json")) as f:
        exp_config: Dict[str, Any] = json.load(f)
    with open(os.path.join(args.agent_path, f"{args.agent_config}.json")) as f:
        agent_config: Dict[str, Any] = json.load(f)

    print(agent_config)

    if args.model_name is not None:
        agent_config['config']['model_name'] = args.model_name

    if args.output_path == "":
        output_path = os.path.join("outputs", agent_config['config']['model_name'].replace('/', '_'), args.exp_config+args.exp_name)
    else:
        output_path = args.output_path
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(os.path.join(output_path, "log.txt"), mode='w')
    logging.basicConfig(
        format="%(message)s",
        handlers=[logging.StreamHandler(), file_handler],
    )

    env_config = exp_config["env_config"]

    logger.info(f"Experiment config: \n{json.dumps(exp_config, indent=2)}")

    if env_config['env_class'] == 'WebShopEnv':
        from webshop.web_agent_site.envs import WebAgentTextEnv
        env_config['env'] = WebAgentTextEnv(observation_mode="text", human_goals=True)
    elif env_config['env_class'] == 'SciWorldEnv':
        from scienceworld import ScienceWorldEnv
        from eval_agent.utils.replace_sciworld_score import sciworld_monkey_patch
        sciworld_monkey_patch()
        env_config['env'] = ScienceWorldEnv("", serverPath=os.path.join(os.getcwd(), env_config['env_jar_path']), envStepLimit=200)

    task_config: Dict[str, Any] = exp_config["task"]
    task_class: tasks.Task = getattr(tasks, task_config["task_class"])
    all_tasks, n_tasks = task_class.load_tasks(args.split, args.part_num, args.part_idx)

    agent: agents.LMAgent = getattr(agents, agent_config["agent_class"])(
        agent_config["config"]
    )

    state_list = []

    done_task_id = []
    if os.path.exists(output_path) and not args.override:
        for file in os.listdir(output_path):
            if not file.endswith('json'):
                continue
            state = State.load_json(json.load(open(os.path.join(output_path, file))))
            state_list.append(state)
            done_task_id.append(file.split('.')[0])
        logger.info(f"Existing output file found. {len(done_task_id)} tasks done.")

    if len(done_task_id) == n_tasks:
        logger.info("All tasks done. Exiting.")

        reward_list = []
        success_list = []
        for state in state_list:
            if state.reward is not None:
                reward_list.append(state.reward)
            success_list.append(state.success)

        if len(reward_list) != 0:
            logger.warning(f"Average reward: {sum(reward_list)/len(success_list):.4f}")
        logger.warning(f"Success rate: {sum(success_list)/len(success_list):.4f}")

        if args.exp_config.lower() == "alfworld":
            try:
                summarize_alfworld(output_path, state_list, logger)
            except Exception as e:
                logger.warning(f"[Warn] Failed to summarize ALFWorld metrics: {e}")
        return

    logging.info(f"Running interactive loop for {n_tasks} tasks.")
    n_todo_tasks = n_tasks - len(done_task_id)

    with logging_redirect_tqdm():
        pbar = tqdm(total=n_todo_tasks)
        for i, task in enumerate(all_tasks):

            if args.debug and i == 5:
                break

            if task.task_id in done_task_id or str(task.task_id) in done_task_id:
                continue

            state = interactive_loop(
                task, agent, env_config
            )

            state_list.append(state)
            json.dump(state.to_dict(), open(os.path.join(output_path, f"{task.task_id}.json"), 'w'), indent=4)

            msg = (
                f"[TaskResult] id={task.task_id} success={state.success} steps={state.steps} "
                f"terminate_reason={state.terminate_reason} reward={state.reward}"
            )
            try:
                tqdm.write(msg)
            except Exception:
                print(msg, flush=True)

            pbar.update(1)
        pbar.close()

    logger.warning("All tasks done.")
    logger.warning(f"Output saved to {output_path}")

    reward_list = []
    success_list = []
    for state in state_list:
        if state.reward is not None:
            reward_list.append(state.reward)
        success_list.append(state.success)

    if len(reward_list) != 0:
        logger.warning(f"Average reward: {sum(reward_list)/len(success_list):.4f}")
    logger.warning(f"Success rate: {sum(success_list)/len(success_list):.4f}")

    if args.exp_config.lower() == "alfworld":
        try:
            summarize_alfworld(output_path, state_list, logger)
        except Exception as e:
            logger.warning(f"[Warn] Failed to summarize ALFWorld metrics: {e}")

def summarize_alfworld(output_path: str, state_list: List[State], logger: logging.Logger) -> Dict[str, Any]:
    def get_task_type_from_gamefile(game_file: str) -> str:
        if not game_file:
            return ""

        try:
            task_dir = os.path.basename(os.path.dirname(os.path.dirname(game_file)))
            if not task_dir:
                return ""
            return task_dir.split("-")[0].strip().lower()
        except Exception:
            return ""

    def map_task_type_to_bucket(task_type: str) -> str:
        t = (task_type or "").lower()

        if t.startswith("pick_two_obj"):
            return "PICK2"
        if "clean" in t:
            return "CLEAN"
        if "heat" in t:
            return "HEAT"
        if "cool" in t:
            return "COOL"
        if t.startswith("look_at") or t.startswith("look"):
            return "LOOK"
        return "PICK"

    buckets = ["PICK", "LOOK", "CLEAN", "HEAT", "COOL", "PICK2"]
    stats = {b: {"n": 0, "success": 0} for b in buckets}
    overall_n = 0
    overall_s = 0

    for st in state_list:
        overall_n += 1
        overall_s += 1 if st.success else 0
        task_type = get_task_type_from_gamefile(getattr(st, "error", "") or "")
        b = map_task_type_to_bucket(task_type)
        if b not in stats:
            continue
        stats[b]["n"] += 1
        stats[b]["success"] += 1 if st.success else 0

    rates = {}
    for b in buckets:
        n = stats[b]["n"]
        s = stats[b]["success"]
        rates[b] = (s / n) if n else 0.0
    overall_rate = (overall_s / overall_n) if overall_n else 0.0

    summary = {
        "buckets": buckets,
        "counts": {b: stats[b]["n"] for b in buckets},
        "success": {b: stats[b]["success"] for b in buckets},
        "rates": rates,
        "overall": {"n": overall_n, "success": overall_s, "rate": overall_rate},
    }

    try:
        with open(os.path.join(output_path, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
    except Exception:
        pass

    try:
        with open(os.path.join(output_path, "summary.csv"), "w", newline="") as f:
            writer = csv.writer(f)

            header = ["Pick", "Look", "Clean", "Heat", "Cool", "Pick2", "All"]
            writer.writerow(header)
            writer.writerow(
                [
                    f"{rates['PICK']:.4f}",
                    f"{rates['LOOK']:.4f}",
                    f"{rates['CLEAN']:.4f}",
                    f"{rates['HEAT']:.4f}",
                    f"{rates['COOL']:.4f}",
                    f"{rates['PICK2']:.4f}",
                    f"{overall_rate:.4f}",
                ]
            )
    except Exception:
        pass

    try:
        col_header = ["ALFWorld", "Pick", "Look", "Clean", "Heat", "Cool", "Pick2", "All"]
        col_vals = [
            "",
            f"{rates['PICK']:.4f}",
            f"{rates['LOOK']:.4f}",
            f"{rates['CLEAN']:.4f}",
            f"{rates['HEAT']:.4f}",
            f"{rates['COOL']:.4f}",
            f"{rates['PICK2']:.4f}",
            f"{overall_rate:.4f}",
        ]

        widths = [max(len(h), 7) for h in col_header]
        def fmt_row(items):
            return "  ".join(str(it).ljust(w) for it, w in zip(items, widths))
        logger.warning(fmt_row(col_header))
        logger.warning(fmt_row(col_vals))

        logger.warning(
            "ALFWorld counts: "
            f"Pick(n={stats['PICK']['n']}), "
            f"Look(n={stats['LOOK']['n']}), "
            f"Clean(n={stats['CLEAN']['n']}), "
            f"Heat(n={stats['HEAT']['n']}), "
            f"Cool(n={stats['COOL']['n']}), "
            f"Pick2(n={stats['PICK2']['n']}), "
            f"All(n={overall_n})"
        )
    except Exception:
        pass

    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run the interactive loop.")
    parser.add_argument(
        "--exp_name",
        type=str,
        default="",
        help="The name of the experiemnt.",
    )
    parser.add_argument(
        "--exp_path",
        type=str,
        default="./eval_agent/configs/task",
        help="Config path of experiment.",
    )
    parser.add_argument(
        "--exp_config",
        type=str,
        default="alfworld",
        help="Config of experiment.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Evaluation split.",
    )
    parser.add_argument(
        "--part_num",
        type=int,
        default=1,
        help="Evaluation part.",
    )
    parser.add_argument(
        "--part_idx",
        type=int,
        default=-1,
        help="Evaluation part.",
    )
    parser.add_argument(
        "--agent_path",
        type=str,
        default="./eval_agent/configs/model",
        help="Config path of model.",
    )
    parser.add_argument(
        "--agent_config",
        type=str,
        default="fastchat",
        help="Config of model.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=False,
        help="Model name. It will override the 'model_name' in agent_config"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Whether to run in debug mode (10 ex per task).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Whether to run in debug mode (10 ex per task).",
    )
    parser.add_argument(
        "--override",
        action="store_true",
        help="Whether to ignore done tasks.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Whether to run in interactive mode for demo purpose.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="",
    )
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.INFO)
    elif args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.WARNING)

    main(args)

