"""Hyperparameter dataclasses (single source of truth, validated on creation)."""
from dataclasses import dataclass, replace

import numpy as np

# Observation layout per mode: (shape, replay dtype).
OBSERVATIONS = {
    "pixel": ((4, 84, 84), np.uint8),
    "state": ((12,), np.float32),
}

# Mode-specific defaults; override any field with AgentConfig.for_mode(mode, **kw).
_PRESETS = {
    "pixel": dict(batch_size=64, exploration_steps=50_000, buffer_capacity=50_000),
    "state": dict(batch_size=256, exploration_steps=15_000, buffer_capacity=100_000),
}


def _positive_int(name, value):
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


@dataclass(frozen=True)
class AgentConfig:
    mode: str = "pixel"
    gamma: float = 0.99
    lr: float = 1e-4
    weight_decay: float = 1e-5
    max_grad_norm: float = 10.0
    batch_size: int = 64
    buffer_capacity: int = 50_000
    # Target network: Polyak averaging (tau) or a hard copy every target_update_freq updates.
    soft_update: bool = True
    tau: float = 0.005
    target_update_freq: int = 1_000
    # Epsilon decays linearly over exploration_steps environment steps.
    epsilon_start: float = 1.0
    epsilon_min: float = 0.01
    exploration_steps: int = 50_000

    def __post_init__(self):
        if self.mode not in OBSERVATIONS:
            raise ValueError(f"mode must be one of {tuple(OBSERVATIONS)}, got {self.mode!r}")
        for name in ("batch_size", "buffer_capacity", "exploration_steps", "target_update_freq"):
            _positive_int(name, getattr(self, name))
        if not (0 <= self.gamma <= 1 and 0 <= self.tau <= 1):
            raise ValueError("gamma and tau must be in [0, 1]")
        if not (np.isfinite(self.lr) and self.lr > 0):
            raise ValueError("lr must be finite and positive")
        if not 0 <= self.epsilon_min <= self.epsilon_start <= 1:
            raise ValueError("require 0 <= epsilon_min <= epsilon_start <= 1")

    @classmethod
    def for_mode(cls, mode, **overrides):
        if mode not in _PRESETS:
            raise ValueError(f"mode must be one of {tuple(_PRESETS)}, got {mode!r}")
        return cls(mode=mode, **{**_PRESETS[mode], **overrides})

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
    num_episodes: int = 1500
    max_steps: int = 5000
    warmup_steps: int = 1000
    train_every: int = 4          # env steps per gradient update
    eval_every: int = 50
    eval_episodes: int = 5
    log_every: int = 10
    seed: int = 42
    best_path: str = "best.pt"
    last_path: str = "last.pt"

    def __post_init__(self):
        for name in ("num_episodes", "max_steps", "train_every", "eval_every",
                     "eval_episodes", "log_every"):
            _positive_int(name, getattr(self, name))
        if not isinstance(self.warmup_steps, int) or self.warmup_steps < 0:
            raise ValueError("warmup_steps must be a non-negative integer")

    @property
    def eval_seed(self):
        """Seeds used to pick the best checkpoint; the final holdout uses a different offset."""
        return self.seed + 10_000
