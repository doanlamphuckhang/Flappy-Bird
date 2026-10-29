"""Hyperparameter dataclasses (single source of truth, validated on creation)."""
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

NUM_ACTIONS = 2  # 0 = do nothing, 1 = flap

# Observation layout per mode: (shape, replay dtype).
OBSERVATIONS = {
    "pixel": ((4, 84, 84), np.uint8),
    "state": ((12,), np.float32),
}

# Mode-specific defaults; override any field with AgentConfig.for_mode(mode, **kw).
_AGENT_PRESETS = {
    "pixel": dict(lr=1e-4, batch_size=32, buffer_capacity=50_000, exploration_steps=250_000),
    "state": dict(lr=2.5e-4, batch_size=64, buffer_capacity=100_000, exploration_steps=100_000),
}
_TRAIN_PRESETS = {
    "pixel": dict(total_steps=2_000_000, warmup_steps=20_000, train_every=4, eval_every=50_000),
    "state": dict(total_steps=1_000_000, warmup_steps=5_000, train_every=2, eval_every=25_000),
}


def _positive_int(name, value):
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def _check_mode(mode):
    if mode not in OBSERVATIONS:
        raise ValueError(f"mode must be one of {tuple(OBSERVATIONS)}, got {mode!r}")


@dataclass(frozen=True)
class AgentConfig:
    mode: str = "pixel"
    gamma: float = 0.99
    n_step: int = 3               # multi-step returns spread the sparse pipe reward faster
    lr: float = 1e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 10.0
    batch_size: int = 32
    buffer_capacity: int = 50_000
    # Target network: Polyak averaging (tau) or a hard copy every target_update_freq updates.
    soft_update: bool = True
    tau: float = 0.005
    target_update_freq: int = 1_000
    # Epsilon decays linearly over exploration_steps environment steps.
    epsilon_start: float = 1.0
    epsilon_min: float = 0.01
    exploration_steps: int = 250_000
    # Probability that a random action is "flap". A fair coin keeps the bird glued to the
    # ceiling, so it never passes a pipe; ~8% keeps it near mid-height.
    explore_flap_prob: float = 0.08

    def __post_init__(self):
        _check_mode(self.mode)
        for name in ("n_step", "batch_size", "buffer_capacity", "exploration_steps",
                     "target_update_freq"):
            _positive_int(name, getattr(self, name))
        if not (0 <= self.gamma <= 1 and 0 <= self.tau <= 1):
            raise ValueError("gamma and tau must be in [0, 1]")
        if self.soft_update and self.tau == 0:
            raise ValueError("tau must be positive with soft_update, or the target never moves")
        if not (np.isfinite(self.lr) and self.lr > 0):
            raise ValueError("lr must be finite and positive")
        if not (np.isfinite(self.max_grad_norm) and self.max_grad_norm > 0):
            raise ValueError("max_grad_norm must be finite and positive")
        if not self.weight_decay >= 0:
            raise ValueError("weight_decay must be non-negative")
        if not 0 <= self.epsilon_min <= self.epsilon_start <= 1:
            raise ValueError("require 0 <= epsilon_min <= epsilon_start <= 1")
        if not 0 <= self.explore_flap_prob <= 1:
            raise ValueError("explore_flap_prob must be in [0, 1]")

    @classmethod
    def for_mode(cls, mode, **overrides):
        _check_mode(mode)
        return cls(mode=mode, **{**_AGENT_PRESETS[mode], **overrides})

    @property
    def state_shape(self):
        return OBSERVATIONS[self.mode][0]

    @property
    def state_dtype(self):
        return OBSERVATIONS[self.mode][1]

    def replace(self, **changes):
        return replace(self, **changes)


@dataclass(frozen=True)
class TrainConfig:
    """Training runs for a fixed budget of environment steps (not episodes).

    Everything a run produces goes to `out_dir`: best.pt, last.pt, history.json, progress.csv.
    """

    total_steps: int = 2_000_000
    warmup_steps: int = 20_000         # random transitions collected before learning
    train_every: int = 4               # env steps per gradient update
    max_episode_steps: int = 20_000    # long episodes are truncated (and bootstrapped)
    eval_every: int = 50_000           # env steps between greedy evaluations
    eval_episodes: int = 5
    eval_max_steps: int = 5_000
    log_every: int = 10_000
    seed: int = 42
    out_dir: str = "flappy_bird_output/run"

    def __post_init__(self):
        for name in ("total_steps", "train_every", "max_episode_steps", "eval_every",
                     "eval_episodes", "eval_max_steps", "log_every"):
            _positive_int(name, getattr(self, name))
        if not isinstance(self.warmup_steps, int) or self.warmup_steps < 0:
            raise ValueError("warmup_steps must be a non-negative integer")

    @classmethod
    def for_mode(cls, mode, **overrides):
        _check_mode(mode)
        return cls(**{**_TRAIN_PRESETS[mode], **overrides})

    @property
    def eval_seed(self):
        """Seeds used to pick the best checkpoint; the final holdout uses a different offset."""
        return self.seed + 10_000

    @property
    def best_path(self):
        return Path(self.out_dir) / "best.pt"

    @property
    def last_path(self):
        return Path(self.out_dir) / "last.pt"

    @property
    def history_path(self):
        return Path(self.out_dir) / "history.json"

    @property
    def progress_path(self):
        return Path(self.out_dir) / "progress.csv"

    def replace(self, **changes):
        return replace(self, **changes)
