"""Compare saved reward JSON files without overwriting previous experiments.

Example:
python compare_runs.py --dqn runs/dqn_eval.json --random runs/random_eval.json
"""
import argparse
import json
from pathlib import Path


def load_run(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rewards = data.get("rewards")
    if not isinstance(rewards, list) or not rewards:
        raise ValueError(f"{path}: missing episode rewards")
    return data


def compare(dqn, random_run):
    if dqn.get("episodes") != random_run.get("episodes"):
        raise ValueError("Both runs must use the same number of episodes")
    if dqn.get("seed") != random_run.get("seed"):
        raise ValueError("Both runs must use the same evaluation seed")
    return {
        "evaluation_episodes_per_agent": dqn["episodes"],
        "seed": dqn["seed"],
        "dqn": {k: dqn[k] for k in ("mean_reward", "median_reward", "std_reward", "max_reward")},
        "random": {k: random_run[k] for k in ("mean_reward", "median_reward", "std_reward", "max_reward")},
        "mean_reward_difference": dqn["mean_reward"] - random_run["mean_reward"],
        "note": "Reward difference is not accuracy, win rate, or pipes passed.",
    }


def main():
    parser = argparse.ArgumentParser(description="Compare learned DQN and random policy reward")
    parser.add_argument("--dqn", required=True)
    parser.add_argument("--random", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    result = compare(load_run(args.dqn), load_run(args.random))
    report = json.dumps(result, indent=2)
    print(report)
    if args.output:
        Path(args.output).write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
