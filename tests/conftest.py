import numpy as np
import pytest
import torch

from flappy_rl import AgentConfig, DQNAgent

torch.set_num_threads(2)


class TinyEnv:
    """Two-step episodes with 12-d observations; mimics the gymnasium API."""

    def __init__(self, truncated=False, fail=False):
        self.truncated, self.fail = truncated, fail
        self.seeds, self.steps = [], 0

    def reset(self, seed=None):
        self.seeds.append(seed)
        self.steps = 0
        return np.zeros(12, np.float32), {}

    def step(self, action):
        if self.fail:
            raise RuntimeError("step failed")
        self.steps += 1
        end = self.steps == 2
        return (np.ones(12, np.float32), 0.1, end and not self.truncated,
                end and self.truncated, {"score": self.steps})

    def close(self):
        pass


@pytest.fixture
def make_agent():
    def factory(**overrides):
        cfg = AgentConfig.for_mode("state", batch_size=2, buffer_capacity=8,
                                   exploration_steps=8, **overrides)
        return DQNAgent(cfg)
    return factory
