"""Wrapper and viewer tests using a fake game (the real package is not required)."""
import json

import gymnasium as gym
import numpy as np
import pytest

from flappy_rl import DQNAgent, envs, export, viewer


class FakeGame(gym.Env):
    observation_space = gym.spaces.Box(-1, 1, (12,), np.float32)
    action_space = gym.spaces.Discrete(2)

    def __init__(self):
        self.t = 0

    def reset(self, seed=None, options=None):
        self.t = 0
        return np.zeros(12, np.float64), {}

    def step(self, action):
        self.t += 1
        return np.zeros(12, np.float64), 1.0 if self.t == 2 else 0.0, self.t == 3, False, {"score": 0}

    def render(self):
        """Plain sky, a 40x40 "bird" that moves 30 px down per step, ground at the bottom."""
        frame = np.full((512, 288, 3), envs.BACKGROUND_RGB, np.uint8)
        y = 50 + 30 * self.t
        frame[y:y + 40, 60:100] = (250, 180, 40)
        frame[405:] = (220, 210, 150)  # ground (cropped away)
        return frame


@pytest.fixture(autouse=True)
def fake_game(monkeypatch):
    calls = []

    def make_base(**kwargs):
        calls.append(kwargs)
        return FakeGame()

    monkeypatch.setattr(envs, "_make_base", make_base)
    return calls


def test_reward_shaping():
    assert envs.shape_reward(0.1, True) == -1.0
    assert envs.shape_reward(1.0, False) == 1.0
    assert envs.shape_reward(0.1, False) == 0.1
    assert envs.shape_reward(-0.5, False) == -0.5  # above the screen keeps its penalty


def test_preprocess_drops_sky_and_ground():
    frame = FakeGame().render()
    mask = envs.preprocess_frame(frame)
    assert mask.shape == (84, 84) and mask.dtype == np.uint8
    assert mask[-1].max() == 0 and mask[0].max() == 0  # ground row and empty sky are black
    assert mask.max() == 255  # the bird is white
    with pytest.raises(RuntimeError):
        envs.preprocess_frame(None)


def test_pixel_env_shape_dtype_and_frame_stack(fake_game):
    env = envs.make_env("pixel")
    assert fake_game[-1] == {"render_mode": "rgb_array", "background": None}
    obs, _ = env.reset()
    assert obs.shape == (4, 84, 84) and obs.dtype == np.uint8 and env.observation_space.contains(obs)
    assert (obs[0] == obs[-1]).all()  # the stack starts as copies of the first frame
    obs, reward, *_ = env.step(0)
    assert not (obs[0] == obs[-1]).all() and reward == 0.1
    assert env.get_frame().shape == (512, 288, 3)


def test_state_env_casts_to_float32():
    env = envs.make_env("state")
    obs, _ = env.reset()
    assert obs.dtype == np.float32 and env.observation_space.dtype == np.float32
    assert env.observation_space.contains(obs)
    assert env.step(0)[0].dtype == np.float32
    with pytest.raises(ValueError):
        envs.make_env("bogus")


def test_viewer_escapes_script_end_and_keeps_template(tmp_path):
    before = viewer.VIEWER_TEMPLATE.read_bytes()
    label = "</script><!--<script>bad()</script>"
    out = viewer.build_3d_viewer(replay={"label": label}, out_path=tmp_path / "v.html")
    page = out.read_text(encoding="utf-8")
    assert "window.FLAPPY_REPLAY=" in page and "\\u003c/script\\u003e" in page
    assert viewer.DATA_PLACEHOLDER not in page and "<!--<script>bad()" not in page
    embedded = page.split("window.FLAPPY_REPLAY=", 1)[1].split(";</script>", 1)[0]
    assert json.loads(embedded) == {"label": label}  # the escapes are still valid JSON
    assert viewer.VIEWER_TEMPLATE.read_bytes() == before


def test_viewer_rejects_template_without_placeholder(tmp_path):
    template = tmp_path / "t.html"
    template.write_text("<html></html>")
    with pytest.raises(ValueError):
        viewer.build_3d_viewer(template=template, out_path=tmp_path / "o.html")


class FakeRecordedGame(FakeGame):
    """FakeGame plus the private fields the replay exporter reads."""

    _pipe_gap = 100

    def reset(self, seed=None, options=None):
        self._upper_pipes = [{"x": 288.0 + 144 * i, "y": -200.0} for i in range(3)]
        self._lower_pipes = [{"x": 288.0 + 144 * i, "y": 220.0} for i in range(3)]
        self._player_y, self._player_vel_y, self._player_rot, self._score = 244, -9, 45, 0
        return super().reset(seed, options)

    def step(self, action):
        self._player_y += 1
        self._score = self.t // 2
        return super().step(action)


def test_record_replay_and_weights_export(make_agent, monkeypatch, tmp_path):
    monkeypatch.setattr(envs, "_make_base", lambda **kw: FakeRecordedGame())
    agent = make_agent()
    replay = export.record_replay(agent, num_episodes=2, max_steps=10, out_path=tmp_path / "r.json")
    assert json.loads((tmp_path / "r.json").read_text(encoding="utf-8")) == replay
    episode = replay["episodes"][0]
    assert replay["pipe_gap"] == 100 and len(episode["y"]) == len(episode["act"]) + 1 == 4
    assert episode["pipes"][0][:2] == [288.0, 120.0] and episode["terminated"]
    assert episode["score"] == episode["score_t"][-1]

    weights = export.export_mlp_weights(agent, out_path=tmp_path / "w.json")
    assert set(weights["tensors"]) == set(agent.policy_net.state_dict())
    pixel_agent = DQNAgent(agent.config.replace(mode="pixel", buffer_capacity=1))
    assert export.export_mlp_weights(pixel_agent) is None  # the CNN cannot run in the browser
