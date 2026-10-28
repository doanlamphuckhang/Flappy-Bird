"""Wrapper and viewer tests using a fake game (the real package is not required)."""
import gymnasium as gym
import numpy as np
import pytest

from flappy_rl import envs, viewer


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
        return np.full((512, 288, 3), 40 * self.t, np.uint8)


@pytest.fixture(autouse=True)
def fake_game(monkeypatch):
    monkeypatch.setattr(envs, "_make_base", lambda render_mode: FakeGame())


def test_reward_shaping():
    assert envs.shape_reward(0.0, True) == -1.0
    assert envs.shape_reward(1.0, False) == 1.0
    assert envs.shape_reward(0.0, False) == 0.1


def test_pixel_env_shape_dtype_and_frame_stack():
    env = envs.make_env("pixel")
    obs, _ = env.reset()
    assert obs.shape == (4, 84, 84) and obs.dtype == np.uint8 and env.observation_space.contains(obs)
    assert (obs[0] == obs[-1]).all()  # the stack starts as copies of the first frame
    obs, reward, *_ = env.step(0)
    assert not (obs[0] == obs[-1]).all() and reward == 0.1
    assert env.get_frame().shape == (512, 288, 3)


def test_state_env_casts_to_float32():
    env = envs.make_env("state")
    obs, _ = env.reset()
    assert obs.dtype == np.float32
    assert env.step(0)[0].dtype == np.float32
    with pytest.raises(ValueError):
        envs.make_env("bogus")


def test_viewer_escapes_script_end_and_keeps_template(tmp_path):
    before = viewer.VIEWER_TEMPLATE.read_bytes()
    out = viewer.build_3d_viewer(replay={"label": "</script><script>bad()</script>"},
                                 out_path=tmp_path / "v.html")
    page = out.read_text(encoding="utf-8")
    assert "window.FLAPPY_REPLAY=" in page and "<\\/script>" in page
    assert viewer.DATA_PLACEHOLDER not in page and "</script><script>bad()" not in page
    assert viewer.VIEWER_TEMPLATE.read_bytes() == before


def test_viewer_rejects_template_without_placeholder(tmp_path):
    template = tmp_path / "t.html"
    template.write_text("<html></html>")
    with pytest.raises(ValueError):
        viewer.build_3d_viewer(template=template, out_path=tmp_path / "o.html")
