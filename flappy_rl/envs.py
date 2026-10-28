"""Environment wrappers: reward shaping, pixel stacking and state vectors."""
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # headless pygame

from collections import deque

import cv2
import gymnasium as gym
import numpy as np

ENV_ID = "FlappyBird-v0"


def shape_reward(reward, terminated):
    """Death -1, pipe passed +1, alive +0.1 (dense signal that favours survival)."""
    if terminated:
        return -1.0
    return 1.0 if reward > 0.5 else 0.1


def _make_base(render_mode):
    import flappy_bird_gymnasium  # noqa: F401  (registers FlappyBird-v0)
    return gym.make(ENV_ID, render_mode=render_mode, use_lidar=False)


class FlappyBirdStateEnv(gym.Wrapper):
    """12-d state vector observation with shaped reward."""

    def __init__(self, render=False):
        super().__init__(_make_base("rgb_array" if render else None))

    def get_frame(self):
        return self.env.render()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs.astype(np.float32), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs.astype(np.float32), shape_reward(reward, terminated), terminated, truncated, info


class FlappyBirdPixelEnv(gym.Wrapper):
    """Grayscale, resized, frame-stacked observations of shape (stack, H, W), uint8."""

    def __init__(self, frame_size=(84, 84), stack_size=4):
        if stack_size <= 0 or len(frame_size) != 2 or min(frame_size) <= 0:
            raise ValueError("frame_size and stack_size must be positive")
        super().__init__(_make_base("rgb_array"))
        self.frame_size, self.stack_size = frame_size, stack_size
        self.observation_space = gym.spaces.Box(
            0, 255, shape=(stack_size, frame_size[1], frame_size[0]), dtype=np.uint8)
        self.frames = deque(maxlen=stack_size)
        self.last_frame = None  # full-colour frame, used for video

    def _process(self, frame):
        if frame is None:
            raise RuntimeError("Environment did not render an RGB frame")
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return cv2.resize(gray, self.frame_size, interpolation=cv2.INTER_AREA)

    def get_frame(self):
        return self.last_frame

    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)
        self.last_frame = self.env.render()
        first = self._process(self.last_frame)
        self.frames.clear()
        self.frames.extend([first] * self.stack_size)
        return np.stack(self.frames), info

    def step(self, action):
        _, reward, terminated, truncated, info = self.env.step(action)
        self.last_frame = self.env.render()
        self.frames.append(self._process(self.last_frame))
        return np.stack(self.frames), shape_reward(reward, terminated), terminated, truncated, info


def make_env(mode="pixel", render=False):
    """Create a "pixel" or "state" environment. Pixel mode always renders."""
    if mode == "pixel":
        return FlappyBirdPixelEnv()
    if mode == "state":
        return FlappyBirdStateEnv(render=render)
    raise ValueError(f"Unsupported mode: {mode}")
