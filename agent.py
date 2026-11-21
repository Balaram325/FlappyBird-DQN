"""Train and evaluate a finite Deep Q-Network for FlappyBird-v0.

Training results depend on seed, environment, hardware and hyperparameters.
Run 'python agent.py --help' for commands. No performance claims are hardcoded.
"""
import argparse
import json
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

from dqn import DQN

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def epsilon_at(episode, start, minimum, decay):
    return max(minimum, start * decay ** episode)

def dqn_loss(policy, target, batch, gamma, loss_fn, double_dqn=False):
    """Compute TD loss. The final batch field marks states with no bootstrap."""
    states, actions, next_states, rewards, dones = zip(*batch)
    states = torch.stack(states)
    actions = torch.tensor(actions, dtype=torch.long, device=states.device)
    next_states = torch.stack(next_states)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=states.device)
    dones = torch.tensor(dones, dtype=torch.bool, device=states.device)
    q_values = policy(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        if double_dqn:
            # Choose the next action with the policy network; value it with target.
            next_actions = policy(next_states).argmax(dim=1, keepdim=True)
            next_values = target(next_states).gather(1, next_actions).squeeze(1)
        else:
            next_values = target(next_states).max(dim=1).values
        targets = rewards + gamma * next_values * (~dones).float()
    return loss_fn(q_values, targets)

def make_environment(render=False):
    # Lazy imports so unit tests can run without pygame or the game installed.
    import gymnasium as gym
    import flappy_bird_gymnasium  # noqa: F401 - registers FlappyBird-v0
    return gym.make("FlappyBird-v0", render_mode="human" if render else None)

def load_parameters(name, config_path):
    with open(config_path, encoding="utf-8") as handle:
        configs = yaml.safe_load(handle) or {}
    if name not in configs:
        raise ValueError(f"Unknown parameter set '{name}'. Available: {list(configs)}")
    params = configs[name]
    required = {"alpha", "gamma", "epsilon_init", "epsilon_min", "epsilon_decay",
                "replay_memory_size", "mini_batch_size", "network_sync_rate", "reward_threshold"}
    missing = required - set(params)
    if missing:
        raise ValueError(f"Missing parameters: {sorted(missing)}")
    if params["mini_batch_size"] < 1 or params["replay_memory_size"] < params["mini_batch_size"]:
        raise ValueError("Replay capacity must be at least the positive mini-batch size")
    if params["network_sync_rate"] < 1 or params["alpha"] <= 0:
        raise ValueError("Network sync rate and learning rate must be positive")
    if not 0 <= params["epsilon_min"] <= params["epsilon_init"] <= 1 or not 0 < params["epsilon_decay"] <= 1:
        raise ValueError("Invalid epsilon parameters")
    if not 0 <= params["gamma"] <= 1:
        raise ValueError("Gamma must be between 0 and 1")
    return params

class Agent:
    def __init__(self, param_set="FlappyBird-v0", config_path="parameters.yaml", runs_dir="runs", seed=42):
        self.param_set = param_set
        self.params = load_parameters(param_set, config_path)
        self.seed = seed
        self.runs_dir = Path(runs_dir)
        self.model_file = self.runs_dir / f"{param_set}.pt"
        self.device = torch.device("cuda" if torch.cuda.is_available() else
                                   "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")

    def run(self, is_training=True, render=False, episodes=300, max_steps=5000,
            random_baseline=False, checkpoint_window=20, double_dqn=False):
        if random_baseline and is_training:
            raise ValueError("Random baseline is an evaluation mode, not training")
        if checkpoint_window < 1:
            raise ValueError("Checkpoint window must be positive")
        if episodes < 1 or max_steps < 1:
            raise ValueError("Episodes and maximum steps must be positive")
        seed_everything(self.seed)
        env = make_environment(render=render)
        rewards_history = []
        try:
            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.n
            policy = DQN(state_dim, action_dim).to(self.device)
            if is_training:
                target = DQN(state_dim, action_dim).to(self.device)
                target.load_state_dict(policy.state_dict())
                target.eval()
                optimizer = torch.optim.Adam(policy.parameters(), lr=self.params["alpha"])
                memory = deque(maxlen=self.params["replay_memory_size"])
                updates = 0
                self.runs_dir.mkdir(parents=True, exist_ok=True)
            elif not random_baseline:
                if not self.model_file.is_file():
                    raise FileNotFoundError(f"No checkpoint: {self.model_file}. Train first with --train.")
                policy.load_state_dict(torch.load(self.model_file, map_location=self.device, weights_only=True))
                policy.eval()
            loss_fn = nn.SmoothL1Loss()
            # Use a rolling mean instead of a single unusually lucky episode.
            best_rolling_mean = float("-inf")
            for episode in range(episodes):
                state, _ = env.reset(seed=self.seed + episode)
                if hasattr(env.action_space, "seed"):
                    env.action_space.seed(self.seed + episode)
                episode_reward = 0.0
                epsilon = epsilon_at(episode, self.params["epsilon_init"],
                                     self.params["epsilon_min"], self.params["epsilon_decay"]) if is_training else 0.0
                for step in range(max_steps):
                    tensor_state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
                    if random_baseline or (is_training and random.random() < epsilon):
                        action = int(env.action_space.sample())
                    else:
                        with torch.no_grad():
                            action = int(policy(tensor_state.unsqueeze(0)).argmax(dim=1).item())
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    episode_reward += float(reward)
                    done = bool(terminated or truncated or step + 1 == max_steps)
                    if is_training:
                        memory.append((tensor_state, action,
                                       torch.as_tensor(next_state, dtype=torch.float32, device=self.device),
                                       float(reward), done))
                        if len(memory) >= self.params["mini_batch_size"]:
                            batch = random.sample(memory, self.params["mini_batch_size"])
                            loss = dqn_loss(policy, target, batch, self.params["gamma"], loss_fn, double_dqn=double_dqn)
                            optimizer.zero_grad(set_to_none=True)
                            loss.backward()
                            nn.utils.clip_grad_norm_(policy.parameters(), max_norm=10)
                            optimizer.step()
                            updates += 1
                            if updates % self.params["network_sync_rate"] == 0:
                                target.load_state_dict(policy.state_dict())
                    state = next_state
                    if done:
                        break
                rewards_history.append(episode_reward)
                if is_training and len(rewards_history) >= checkpoint_window:
                    rolling_mean = float(np.mean(rewards_history[-checkpoint_window:]))
                    if rolling_mean > best_rolling_mean:
                        best_rolling_mean = rolling_mean
                        torch.save(policy.state_dict(), self.model_file)
                if (episode + 1) % 10 == 0 or episode == 0 or episode + 1 == episodes:
                    last = rewards_history[-min(10, len(rewards_history)):]
                    print(f"Episode {episode+1}/{episodes} reward={episode_reward:.2f} mean_last_10={np.mean(last):.2f} epsilon={epsilon:.3f}")
            # Ensure short training runs still produce a checkpoint.
            if is_training and len(rewards_history) < checkpoint_window:
                torch.save(policy.state_dict(), self.model_file)
            result = {
                "mode": "train" if is_training else ("random" if random_baseline else "evaluate"),
                "episodes": episodes, "seed": self.seed,
                "max_steps": max_steps,
                "algorithm": "Double DQN" if double_dqn else "DQN",
                "checkpoint_selection": f"best trailing {checkpoint_window}-episode mean" if is_training and episodes >= checkpoint_window else ("final model (short run)" if is_training else "not applicable"),
                "mean_reward": float(np.mean(rewards_history)),
                "median_reward": float(np.median(rewards_history)),
                "std_reward": float(np.std(rewards_history)),
                "max_reward": float(max(rewards_history)),
                "rewards": rewards_history,
            }
            self.runs_dir.mkdir(parents=True, exist_ok=True)
            output = self.runs_dir / f"{self.param_set}_{result['mode']}_metrics.json"
            output.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(f"Metrics saved to {output}")
            return result
        finally:
            env.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Flappy Bird DQN train/evaluate")
    parser.add_argument("hyperparameters", nargs="?", default="FlappyBird-v0")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train", action="store_true")
    mode.add_argument("--eval", action="store_true", help="Evaluate saved checkpoint without exploration")
    mode.add_argument("--random", action="store_true", help="Evaluate a random-action baseline without a checkpoint")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", default="parameters.yaml")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--render", action="store_true", help="Display game window (requires desktop)")
    parser.add_argument("--double-dqn", action="store_true", help="Use Double DQN Bellman targets during training")
    parser.add_argument("--checkpoint-window", type=int, default=20,
                        help="Training checkpoint chosen by trailing mean reward")
    args = parser.parse_args(argv)
    runner = Agent(args.hyperparameters, args.config, args.runs_dir, args.seed)
    return runner.run(is_training=args.train, render=args.render,
                      episodes=args.episodes if args.episodes is not None else (300 if args.train else 30),
                      max_steps=args.max_steps, random_baseline=args.random,
                      checkpoint_window=args.checkpoint_window, double_dqn=args.double_dqn)


if __name__ == "__main__":
    main()
