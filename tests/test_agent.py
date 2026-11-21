import unittest
from unittest.mock import patch
from tempfile import TemporaryDirectory

import torch
import torch.nn as nn

from agent import Agent, dqn_loss, epsilon_at, load_parameters
from dqn import DQN
from experience_replay import ReplayMemory

class DummyActionSpace:
    n = 2
    def sample(self):
        return 0

class DummyEnv:
    observation_space = type("Space", (), {"shape": (12,)})()
    action_space = DummyActionSpace()
    def reset(self, seed=None):
        self.steps = 0
        return [0.0] * 12, {}
    def step(self, action):
        self.steps += 1
        return [0.0] * 12, 1.0, self.steps >= 2, False, {}
    def close(self):
        self.closed = True

class Tests(unittest.TestCase):
    def test_epsilon_schedule(self):
        self.assertEqual(epsilon_at(0, 1, .05, .9), 1)
        self.assertAlmostEqual(epsilon_at(1000, 1, .05, .9), .05)

    def test_replay_capacity(self):
        buf = ReplayMemory(2)
        for i in range(5):
            buf.append(i)
        self.assertEqual(len(buf), 2)
        self.assertEqual(set(buf.sample(2)), {3, 4})

    def test_model_shape(self):
        self.assertEqual(DQN(12, 2)(torch.zeros(3, 12)).shape, (3, 2))

    def test_terminal_target_has_no_bootstrap(self):
        policy = nn.Linear(1, 2, bias=False)
        target = nn.Linear(1, 2, bias=False)
        with torch.no_grad():
            policy.weight.zero_()
            target.weight.fill_(20)
        batch = [(torch.tensor([1.]), 0, torch.tensor([2.]), 1., True)]
        loss = dqn_loss(policy, target, batch, .99, nn.MSELoss())
        self.assertAlmostEqual(loss.item(), 1.0)

    def test_parameter_validation(self):
        with self.assertRaises(ValueError):
            load_parameters("missing", "parameters.yaml")

    @patch("agent.make_environment", side_effect=lambda render=False: DummyEnv())
    def test_random_baseline_needs_no_model(self, factory):
        with TemporaryDirectory() as path:
            runner = Agent(runs_dir=path)
            result = runner.run(is_training=False, random_baseline=True,
                                episodes=3, max_steps=3)
            self.assertEqual(result["mode"], "random")
            self.assertEqual(result["mean_reward"], 2.0)
            self.assertFalse(runner.model_file.exists())

    @patch("agent.make_environment", side_effect=lambda render=False: DummyEnv())
    def test_rolling_checkpoint_training(self, factory):
        with TemporaryDirectory() as path:
            runner = Agent(runs_dir=path)
            result = runner.run(is_training=True, episodes=3,
                                max_steps=3, checkpoint_window=2)
            self.assertEqual(result["episodes"], 3)
            self.assertTrue(runner.model_file.exists())
            self.assertIn("2-episode", result["checkpoint_selection"])

    def test_invalid_random_training_combination(self):
        with TemporaryDirectory() as path:
            runner = Agent(runs_dir=path)
            with self.assertRaises(ValueError):
                runner.run(is_training=True, random_baseline=True)

    @patch("agent.make_environment", side_effect=lambda render=False: DummyEnv())
    def test_finite_training_then_evaluation(self, factory):
        with TemporaryDirectory() as path:
            runner = Agent(runs_dir=path)
            train = runner.run(is_training=True, episodes=2, max_steps=3)
            self.assertEqual(train["episodes"], 2)
            self.assertEqual(len(train["rewards"]), 2)
            evaluation = runner.run(is_training=False, episodes=3, max_steps=3)
            self.assertEqual(len(evaluation["rewards"]), 3)
            self.assertEqual(evaluation["mean_reward"], 2.0)

if __name__ == "__main__":
    unittest.main()
