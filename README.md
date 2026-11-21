# Flappy Bird AI — Deep Q-Network (PyTorch)

An educational **reinforcement-learning project** that trains a Deep Q-Network (DQN) to choose between *flap* and *do nothing* in the [Flappy Bird Gymnasium environment](https://pypi.org/project/flappy-bird-gymnasium/).

The agent uses a small MLP Q-network, epsilon-greedy exploration, experience replay, a target Q-network and a Bellman TD loss. **This is reinforcement learning, not Agentic AI or a large-language-model agent.**

## Setup

Python 3.10+ recommended. Prefer Google Colab or an existing PyTorch environment if laptop storage is limited. Install only missing requirements:

```bash
python -m pip install -r requirements.txt
```

The Flappy Bird environment and first-run model may need additional downloads. A CUDA GPU is optional; CPU works, though training can be slow.

## Train, evaluate and play

From the repository root:

```bash
python agent.py --train --episodes 300 --seed 42
python agent.py --eval --episodes 30 --seed 123
python agent.py --eval --episodes 3 --render
```

For Windows, run these in PowerShell from the project directory. To inspect settings, edit `parameters.yaml`; the code also accepts `--config`, `--runs-dir` and `--max-steps`.

- `runs/FlappyBird-v0.pt` contains the best training-episode checkpoint (reward-based selection).
- `runs/FlappyBird-v0_train_metrics.json` and `runs/FlappyBird-v0_evaluate_metrics.json` hold measured reward statistics.
- Training and evaluation stop after a **finite** number of episodes; each episode has a maximum-step safety limit.
- Evaluation uses **epsilon = 0** and loads the saved checkpoint. Evaluation seed should differ from the training seed.
- For credible performance comparison, run multiple evaluation seeds and compare to a random-action baseline. Don't claim a specific trained score unless you ran the experiment and saved its result.

## Implementation details

| Component | Implementation |
| --- | --- |
| Q-network | State vector → Dense(256) + ReLU → 2 Q-values |
| Optimizer | Adam |
| Replay buffer | Bounded deque, uniform random mini-batches |
| Training loss | Smooth L1 temporal-difference loss |
| Target network | Periodic weight synchronization by optimizer update count |
| Action selection | Decaying epsilon-greedy (training); greedy (evaluation) |
| Episode ending | Handles `terminated`, `truncated`, and max steps |
| Reproducibility | Python, NumPy and PyTorch random seeds |

## Tests

```bash
python -m unittest discover -s tests -v
```

The unit tests exercise finite episodes, checkpoint loading, terminal Bellman targets, replay buffer behavior and epsilon schedule without downloading the game environment. They do not establish gameplay performance.

## Limitations

The agent is a simplified DQN baseline. It does not use prioritized replay, Double DQN, frame stacking, or reward normalization. The checkpoint is selected on a single highest training reward, which may overestimate generalization. Runs on different hardware or environment versions may produce different scores.

This project has **not** been independently benchmarked for a published high score. Report measured mean reward, median, standard deviation and evaluation episode count rather than an invented success rate.

## Portfolio description

Developed a Deep Q-Network Flappy Bird agent using PyTorch, epsilon-greedy exploration, experience replay, target-network synchronization, and reproducible training/evaluation with episode reward metrics.
