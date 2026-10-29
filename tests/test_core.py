"""CPU-only unit tests; no game, GPU or real training needed."""
import numpy as np
import pytest
import torch

from conftest import TinyEnv
from flappy_rl import AgentConfig, DQNAgent, TrainConfig
from flappy_rl.agent import linear_schedule
from flappy_rl.evaluation import evaluate_policy
from flappy_rl.models import DuelingDQNCNN, DuelingDQNMLP
from flappy_rl.replay import NStepCollector, ReplayBuffer
from flappy_rl.runtime import new_run_dir, output_root
from flappy_rl.training import History, train


def test_replay_wraparound_and_validation():
    replay = ReplayBuffer(2, (1,), np.float32)
    for i in range(3):
        replay.push([i], i % 2, i, [i + 1], i == 2)
    assert len(replay) == 2 and set(replay.states[:, 0]) == {1, 2}
    states, _, _, next_states, _, nsteps = replay.sample(10)
    assert (nsteps == 1).all()
    np.testing.assert_array_equal(next_states, states + 1)
    with pytest.raises(ValueError):
        ReplayBuffer(0, (1,))
    with pytest.raises(ValueError):
        ReplayBuffer(1, (1,)).sample(1)


@pytest.mark.parametrize("bad", [{"exploration_steps": 0}, {"tau": 2}, {"batch_size": 0},
                                 {"lr": float("nan")}, {"epsilon_min": 2.0}, {"tau": 0.0},
                                 {"max_grad_norm": 0.0}, {"weight_decay": -1e-3},
                                 {"n_step": 0}, {"explore_flap_prob": 1.5}])
def test_invalid_agent_config(bad):
    with pytest.raises(ValueError):
        AgentConfig.for_mode("state", **bad)


def test_linear_schedule():
    assert linear_schedule(1.0, 0.0, 10, 0) == 1.0
    assert linear_schedule(1.0, 0.0, 10, 5) == pytest.approx(0.5)
    assert linear_schedule(1.0, 0.1, 10, 99) == pytest.approx(0.1)


def test_nstep_returns_termination_and_truncation():
    c = NStepCollector(3, 0.5)
    assert c.push("s0", 0, 1.0, "s1", False) == []
    assert c.push("s1", 1, 2.0, "s2", False) == []
    # Window full: s0 gets 1 + 0.5*2 + 0.25*4 and bootstraps from s3 after 3 steps.
    assert c.push("s2", 0, 4.0, "s3", False) == [("s0", 0, 3.0, "s3", False, 3)]
    # Termination flushes every pending start state with done=True.
    out = c.push("s3", 1, 8.0, "end", True)
    assert [(t[0], t[2], t[4], t[5]) for t in out] == [
        ("s1", 2 + 2 + 2.0, True, 3), ("s2", 4 + 4.0, True, 2), ("s3", 8.0, True, 1)]
    assert len(c) == 0
    # Truncation flushes too but keeps done=False so the target bootstraps.
    c.push("a", 0, 1.0, "b", False)
    assert c.push("b", 0, 1.0, "cut", False, truncated=True) == [
        ("a", 0, 1.5, "cut", False, 2), ("b", 0, 1.0, "cut", False, 1)]


def test_one_step_collector_is_plain_transitions():
    c = NStepCollector(1, 0.99)
    assert c.push("s", 1, 0.1, "t", False) == [("s", 1, 0.1, "t", False, 1)]


def test_exploration_is_biased_towards_not_flapping(make_agent):
    import random
    random.seed(0)
    agent = make_agent(explore_flap_prob=0.1)
    flaps = np.mean([agent.select_action(np.zeros(12, np.float32)) for _ in range(5_000)])
    assert 0.07 < flaps < 0.13  # epsilon is 1 here, so every action is random
    assert make_agent(explore_flap_prob=0.0).random_action() == 0


def test_network_shapes_and_state_dict_keys():
    with torch.no_grad():
        assert DuelingDQNCNN()(torch.zeros(1, 4, 84, 84, dtype=torch.uint8)).shape == (1, 2)
        mlp = DuelingDQNMLP()
        assert mlp(torch.zeros(3, 12)).shape == (3, 2)
    # The 3D viewer reads these exact layer names.
    assert {"features.0.weight", "features.2.weight", "value_stream.0.weight",
            "value_stream.2.weight", "advantage_stream.0.weight",
            "advantage_stream.2.weight"} <= set(mlp.state_dict())


def test_double_dqn_target_and_terminal_mask(make_agent):
    agent = make_agent(gamma=0.5)
    with torch.no_grad():
        for net in (agent.policy_net, agent.target_net):
            for p in net.parameters():
                p.zero_()
        agent.policy_net.advantage_stream[-1].bias.copy_(torch.tensor([2., 0.]))  # picks action 0
        agent.target_net.value_stream[-1].bias.fill_(5.)
        agent.target_net.advantage_stream[-1].bias.copy_(torch.tensor([0., 4.]))   # its own max is 1
    rewards = torch.ones(3, 1)
    dones = torch.tensor([[False], [True], [False]])
    nsteps = torch.tensor([[1], [1], [2]])
    target = agent._td_target(rewards, torch.zeros(3, 12), dones, nsteps)
    # Non-terminal: 1 + 0.5^k * Q_target(action 0) with Q = 5 + 0 - 2 = 3; terminal: reward only.
    assert target.flatten().tolist() == pytest.approx([2.5, 1.0, 1.75])


def test_update_changes_weights_and_checkpoint_roundtrip(make_agent, tmp_path):
    agent = make_agent(n_step=1)
    for i in range(2):
        agent.observe(np.zeros(12, np.float32), i, 1., np.ones(12, np.float32), False)
    before = [p.detach().clone() for p in agent.policy_net.parameters()]
    assert np.isfinite(agent.update())
    assert any(not torch.equal(a, b) for a, b in zip(before, agent.policy_net.parameters(), strict=True))

    path = tmp_path / "ckpt.pt"
    agent.save(path)
    assert not path.with_name("ckpt.pt.tmp").exists()

    resumed = DQNAgent.from_checkpoint(path, training=True)
    assert (resumed.env_steps, resumed.total_steps) == (2, 1)
    assert resumed.config == agent.config
    for a, b in zip(agent.target_net.parameters(), resumed.target_net.parameters(), strict=True):
        assert torch.equal(a, b)

    inference = DQNAgent.from_checkpoint(path)
    assert inference.memory.capacity == 1
    state = np.zeros(12, np.float32)
    assert inference.select_action(state, True) == agent.select_action(state, True)
    with pytest.raises(ValueError):  # resuming with a different config must fail
        make_agent(gamma=0.9).load(path, training=True)


def test_train_step_budget_outputs_and_truncation(make_agent, tmp_path):
    agent, env, eval_env = make_agent(), TinyEnv(truncated=True), TinyEnv()
    cfg = TrainConfig(total_steps=6, warmup_steps=2, train_every=2, eval_every=4, eval_episodes=1,
                      log_every=3, seed=7, out_dir=str(tmp_path / "run"))
    history = train(agent, env, eval_env, cfg)
    out = tmp_path / "run"
    for name in ("best.pt", "last.pt", "history.json", "progress.csv", "config.json"):
        assert (out / name).exists(), name
    assert (agent.env_steps, agent.total_steps) == (6, 3)
    assert not agent.memory.dones[:len(agent.memory)].any()  # truncation is not termination
    assert len(agent.memory) == 8 and len(history.episodes) == 3  # 2 warmup + 6 steps, all flushed
    assert history.evals == [[4, 2.0], [6, 2.0]]
    assert history.steps == [2, 4, 6] and history.scores == [2, 2, 2]
    assert env.seeds[:3] == [7, None, 8] and eval_env.seeds == [10007, 10007]
    rows = (out / "progress.csv").read_text(encoding="utf-8").splitlines()
    assert rows[0].startswith("step,") and [r.split(",")[0] for r in rows[1:]] == ["3", "6"]
    assert History.load(out / "history.json") == history


def test_train_resumes_from_last_checkpoint(make_agent, tmp_path):
    cfg = TrainConfig(total_steps=4, warmup_steps=2, eval_every=2, eval_episodes=1,
                      out_dir=str(tmp_path))
    first = train(make_agent(), TinyEnv(), TinyEnv(), cfg)
    agent = DQNAgent.from_checkpoint(cfg.last_path, training=True)
    assert agent.env_steps == 4
    history = train(agent, TinyEnv(), TinyEnv(), cfg.replace(total_steps=8),
                    history=History.load(cfg.history_path))
    assert agent.env_steps == 8 and len(history.episodes) == 4
    assert history.evals[:2] == first.evals and [e[0] for e in history.evals] == [2, 4, 6, 8]
    assert (tmp_path / "config_resume_4.json").exists()


def test_train_rejects_shared_env_and_saves_on_interrupt(make_agent, tmp_path):
    env = TinyEnv()
    with pytest.raises(ValueError):
        train(make_agent(), env, env, TrainConfig(out_dir=str(tmp_path)))

    class Interrupting(TinyEnv):
        def step(self, action):
            raise KeyboardInterrupt

    cfg = TrainConfig(warmup_steps=0, out_dir=str(tmp_path))
    with pytest.raises(KeyboardInterrupt):
        train(make_agent(), Interrupting(), TinyEnv(), cfg)
    assert cfg.last_path.exists() and cfg.history_path.exists()


def test_run_dirs_default_to_flappy_bird_output(monkeypatch, tmp_path):
    monkeypatch.delenv("FLAPPY_OUTPUT", raising=False)
    monkeypatch.setattr("flappy_rl.runtime.COLAB_DRIVE", tmp_path / "no-drive")
    assert output_root().as_posix() == "flappy_bird_output"
    drive = tmp_path / "MyDrive"
    drive.mkdir()
    monkeypatch.setattr("flappy_rl.runtime.COLAB_DRIVE", drive)
    assert output_root() == drive / "flappy_bird_output"
    monkeypatch.setenv("FLAPPY_OUTPUT", str(tmp_path / "custom"))
    first, second = new_run_dir("state"), new_run_dir("state")
    assert first.parent == tmp_path / "custom" and first.name.startswith("state_")
    assert first != second and first.is_dir() and second.is_dir()


def test_train_config_presets():
    assert TrainConfig.for_mode("state").total_steps < TrainConfig.for_mode("pixel").total_steps
    assert AgentConfig.for_mode("pixel", lr=1e-3).lr == 1e-3
    with pytest.raises(ValueError):
        TrainConfig.for_mode("bogus")
    with pytest.raises(ValueError):
        TrainConfig(total_steps=0)


def test_greedy_evaluation_is_repeatable_and_leaves_epsilon(make_agent):
    agent, env = make_agent(), TinyEnv()
    first = evaluate_policy(agent, env, 2, 2, seed=100)
    assert first == evaluate_policy(agent, env, 2, 2, seed=100)
    assert env.seeds == [100, 101, 100, 101]
    assert agent.env_steps == 0 and agent.epsilon == 1.0


def test_hard_target_update_allows_zero_tau(make_agent):
    agent = make_agent(soft_update=False, tau=0.0, target_update_freq=1, n_step=1)
    for i in range(2):
        agent.observe(np.zeros(12, np.float32), i, 1., np.ones(12, np.float32), False)
    agent.update()
    for a, b in zip(agent.policy_net.parameters(), agent.target_net.parameters(), strict=True):
        assert torch.equal(a, b)


def test_history_save_load_roundtrip(tmp_path):
    history = History([[10, 1.5, 3, 40], [20, -1.0, 0, 10]], [[50, 2.5]])
    history.save(tmp_path / "h.json")
    assert History.load(tmp_path / "h.json") == history
    assert history.rewards == [1.5, -1.0] and history.scores == [3, 0] and history.steps == [10, 20]
