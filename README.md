# Foule

This project contains a simple evolutionary simulation where agents called
**Fouloïdes** learn to move and collect apples on a grid using a small neural
network. The repository has been reorganised as a Python package to ease
usage and future development.

## Installation

```bash
pip install -r requirements.txt
```

## Training

```bash
python -m foule.scripts.train
```

## Visualisation

After some training you can visualise the best agent:

```bash
python -m foule.scripts.visualize
```

## Neural network improvement ideas

- Increase the convolutional depth and add residual connections to capture more
  complex spatial patterns.
- Use `torch.nn.LSTM` to give the agent a short-term memory of previous states.
- Experiment with reinforcement learning algorithms (e.g. PPO) instead of the
  current evolutionary approach to optimise the network weights.
- Regularise the network with dropout or weight decay to improve generalisation.
