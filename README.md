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
python agent.py --eval --episodes 100 --seed 123
python agent.py --random --episodes 100 --seed 123
python agent.py --eval --episodes 3 --render
```

For Windows, run these in PowerShell from the project directory. To inspect settings, edit `parameters.yaml`; the code also accepts `--config`, `--runs-dir`, `--checkpoint-window` and `--max-steps`.

- `runs/FlappyBird-v0.pt` contains the training checkpoint chosen by the best trailing 20-episode mean reward (or final weights for runs shorter than 20 episodes).
- `runs/FlappyBird-v0_train_metrics.json`, `runs/FlappyBird-v0_evaluate_metrics.json`, and `runs/FlappyBird-v0_random_metrics.json` hold episode reward statistics after you run the respective commands.
- Training and evaluation stop after a **finite** number of episodes; each episode has a maximum-step safety limit.
- Model evaluation uses **epsilon = 0** and loads a checkpoint. Random baseline takes uniformly random actions and does not need a checkpoint. Use the **same evaluation seed and number of episodes** for both, distinct from the training seed.
- For credible performance comparison, run multiple evaluation seeds and compare to the random-action baseline. Don't claim a specific trained score unless you ran the experiment and saved its result.

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
| Reproducibility | Python, NumPy and PyTorch random seeds, plus action-space seeding when supported |
| Checkpoint selection | Best trailing 20-episode mean reward during training |
| Random baseline | `--random` compares reward distribution without a checkpoint |

## Tests

```bash
python -m unittest discover -s tests -v
```

The unit tests exercise finite episodes, checkpoint loading, terminal Bellman targets, replay buffer behavior and epsilon schedule without downloading the game environment. They do not establish gameplay performance.

## Limitations

The agent is a simplified DQN baseline with an optional Double DQN target. It does not use prioritized replay, frame stacking, or reward normalization. The checkpoint uses the training reward over a trailing 20-episode window; it can still overestimate evaluation performance and should be checked on independent episodes. Runs on different hardware or environment versions may produce different scores.

A single 100-episode benchmark was reported by the project owner (see below); it has not been independently reproduced or repeated across training seeds. Report measured mean reward, median, standard deviation and evaluation episode count rather than an invented success rate.

## Measured baseline experiment (Colab, reported by project owner)

One completed experiment used **300 training episodes** (seed 42) and **100 evaluation episodes** (seed 123) per policy:

| Metric (episode reward) | Trained DQN | Random-action policy |
| --- | ---: | ---: |
| Mean | -0.27 | -7.64 |
| Median | -0.30 | -7.50 |
| Standard deviation | 0.69 | 0.87 |
| Maximum | 1.50 | -4.50 |

The mean difference is **+7.37 reward units per episode**. These are the numbers shared from the Colab summaries, rounded to two decimal places; original per-episode JSON/checkpoint files have not been uploaded to this repository. No success rate, pipes-passed count, or accuracy is inferred. Results reflect **one seed and one training run**, and the environment emitted Gymnasium observation-space warnings.

### Stronger experiment: compare DQN and Double DQN

The original DQN code remains the default. Pass `--double-dqn` for **Double DQN TD targets** (next action selected by policy, valued by target). Use different run directories so checkpoints and metrics aren't overwritten:

```bash
python agent.py --train --episodes 500 --seed 42 --runs-dir runs/dqn500
python agent.py --eval --episodes 100 --seed 123 --runs-dir runs/dqn500
python agent.py --random --episodes 100 --seed 123 --runs-dir runs/dqn500

python agent.py --train --double-dqn --episodes 500 --seed 42 --runs-dir runs/double500
python agent.py --eval --episodes 100 --seed 123 --runs-dir runs/double500
```

Run the comparison helper after evaluation:

```bash
python compare_runs.py --dqn runs/dqn500/FlappyBird-v0_evaluate_metrics.json --random runs/dqn500/FlappyBird-v0_random_metrics.json
```

**Caution:** If you change the algorithm, training length or reward setup, do not present the resulting score as an improvement until it has actually been measured. At least three training seeds and additional evaluation seeds would make conclusions more credible. Save `runs/` outputs before Colab disconnects.

## Matched 500-episode experiment (reported from Google Colab)

The project owner reported the following comparison after training **standard DQN and Double DQN for 500 episodes each (seed 42)**, and evaluating each trained policy over **100 episodes (seed 123)**, using the same environment and settings:

| Metric (episode reward) | Standard DQN (500) | Double DQN (500) |
| --- | ---: | ---: |
| Mean | -0.048 | **0.777** |
| Median | -0.300 | **0.900** |
| Standard deviation | 0.941 | 1.050 |
| Maximum | 2.100 | **2.200** |

**Observed mean-reward difference: +0.825 reward units per episode**, favouring Double DQN in this run. This is **not** an accuracy or success percentage; no pipes-passed metrics were measured. These numbers were transcribed from the user's Colab summary, not independently reproduced, and the per-episode JSON and weights are not yet committed. The environment emitted Gymnasium observation-space warnings during experiments. Multiple independent training seeds are necessary before concluding the algorithm is consistently better.

## Portfolio description

Developed a Deep Q-Network Flappy Bird agent using PyTorch, epsilon-greedy exploration, experience replay, target-network synchronization, and reproducible training/evaluation with episode reward metrics.

## Hyperparameters used for the 300-episode training setup

| Parameter | Value | Purpose |
| --- | ---: | --- |
| Learning rate (`alpha`) | 0.001 | Adam optimizer step size |
| Discount factor (`gamma`) | 0.99 | Future reward importance |
| Initial / minimum epsilon | 1.0 / 0.05 | Random-action exploration range |
| Epsilon decay per episode | 0.985 | Multiplier; epsilon reaches 0.05 around episode 199 |
| Replay memory capacity | 100,000 | Most recent transitions retained |
| Mini-batch size | 32 | Transitions per DQN update |
| Target network synchronization | every 10 optimizer updates | Copies current network weights to target |
| Hidden layer width | 256 | Single fully connected ReLU layer |
| Default training episodes | 300 | Finite experiment horizon |
| Maximum steps per episode | 5,000 | Hard safety cutoff |
| Checkpoint window | 20 episodes | Best training rolling mean |

The older setting `epsilon_decay=0.9995` would have left epsilon around 0.86 at 300 episodes; this version changes the **training configuration**, so results from old checkpoints are not directly comparable. The configured `reward_threshold: 1000` is retained in YAML for compatibility but is **not used as an early stopping condition**: the environment handles actual episode termination.

### Record an honest benchmark

After training, compare the saved policy with the random baseline using the same evaluation conditions:

```bash
python agent.py --eval --episodes 100 --seed 123
python agent.py --random --episodes 100 --seed 123
python agent.py --eval --episodes 100 --seed 456
python agent.py --random --episodes 100 --seed 456
```

The JSON summaries for a given mode are overwritten on repeated runs. Save copies if you want to compare seeds. Record the **mean reward, median reward, standard deviation, and number of evaluation episodes**. These are reward metrics, **not** bird score/pipes passed, win rate, or model accuracy. To report pipes passed, add and verify a separate environment-specific score field first.

Unit tests are code checks, not gameplay benchmarks; run them locally before submitting the project.
