"""Double Dueling DQN for Flappy Bird (pixel and state observations)."""
from .agent import DQNAgent
from .config import AgentConfig, TrainConfig

__all__ = ["DQNAgent", "AgentConfig", "TrainConfig"]
