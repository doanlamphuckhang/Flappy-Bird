"""CPU-only unit tests; no game, GPU or real training needed."""
import numpy as np
import pytest
import torch

from conftest import TinyEnv
from flappy_rl import AgentConfig, DQNAgent, TrainConfig
from flappy_rl.agent import linear_schedule
from flappy_rl.evaluation import evaluate_policy
from flappy_rl.models import DuelingDQNCNN, DuelingDQNMLP
from flappy_rl.replay import ReplayBuffer
from flappy_rl.training import train


def test_replay_wraparound_and_validation():
    replay = ReplayBuffer(2, (1,), np.float32)
    for i in range(3):
        replay.push([i], i % 2, i, [i + 1], i == 2)
    assert len(replay) == 2 and set(replay.states[:, 0]) == {1, 2}
    states, _, _, next_states, _ = replay.sample(10)
    np.testing.assert_array_equal(next_states, states + 1)
    with pytest.raises(ValueError):
        ReplayBuffer(0, (1,))
    with pytest.raises(ValueError):
        ReplayBuffer(1, (1,)).sample(1)


@pytest.mark.parametrize("bad", [{"exploration_steps": 0}, {"tau": 2}, {"batch_size": 0},
                                 {"lr": float("nan")}, {"epsilon_min": 2.0}])
def test_invalid_agent_config(bad):
    with pytest.raises(ValueError):
        AgentConfig.for_mode("state", **bad)


def test_linear_schedule():
    assert linear_schedule(1.0, 0.0, 10, 0) == 1.0
    assert linear_schedule(1.0, 0.0, 10, 5) == pytest.approx(0.5)
    assert linear_schedule(1.0, 0.1, 10, 99) == pytest.approx(0.1)


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
    rewards = torch.ones(2, 1)
    dones = torch.tensor([[False], [True]])
    target = agent._td_target(rewards, torch.zeros(2, 12), dones)
    # Non-terminal: 1 + 0.5 * Q_target(action 0) = 1 + 0.5 * (5 + 0 - 2); terminal: reward only.
    assert target.flatten().tolist() == pytest.approx([2.5, 1.0])


def test_update_changes_weights_and_checkpoint_roundtrip(make_agent, tmp_path):
    agent = make_agent()
    for i in range(2):
        agent.observe(np.zeros(12, np.float32), i, 1., np.ones(12, np.float32), False)
    before = [p.detach().clone() for p in agent.policy_net.parameters()]
    assert np.isfinite(agent.update())
    assert any(not torch.equal(a, b) for a, b in zip(before, agent.policy_net.parameters()))

    path = tmp_path / "ckpt.pt"
    agent.save(path)
    assert not path.with_name("ckpt.pt.tmp").exists()

    resumed = DQNAgent.from_checkpoint(path, training=True)
    assert (resumed.env_steps, resumed.total_steps) == (2, 1)
    assert resumed.config == agent.config
    for a, b in zip(agent.target_net.parameters(), resumed.target_net.parameters()):
        assert torch.equal(a, b)

    inference = DQNAgent.from_checkpoint(path)
    assert inference.memory.capacity == 1
    state = np.zeros(12, np.float32)
    assert inference.select_action(state, True) == agent.select_action(state, True)
    with pytest.raises(ValueError):  # resuming with a different config must fail
        make_agent(gamma=0.9).load(path, training=True)


def test_train_cadence_truncation_and_best_checkpoint(make_agent, tmp_path):
    agent, env, eval_env = make_agent(), TinyEnv(truncated=True), TinyEnv()
    cfg = TrainConfig(num_episodes=2, max_steps=2, warmup_steps=2, train_every=2, eval_every=50,
                      eval_episodes=1, seed=7, best_path=str(tmp_path / "b.pt"),
                      last_path=str(tmp_path / "l.pt"))
    history = train(agent, env, eval_env, cfg)
    assert (tmp_path / "b.pt").exists() and (tmp_path / "l.pt").exists()
    assert (agent.env_steps, agent.total_steps) == (4, 2)
    assert not agent.memory.dones[:len(agent.memory)].any()  # truncation is not termination
    assert history.evals == [(2, 2.0)] and len(history.rewards) == 2
    assert env.seeds[0] == 7 and eval_env.seeds == [10007]


def test_train_rejects_shared_env_and_saves_on_interrupt(make_agent, tmp_path):
    env = TinyEnv()
    with pytest.raises(ValueError):
        train(make_agent(), env, env, TrainConfig())

    class Interrupting(TinyEnv):
        def step(self, action):
            raise KeyboardInterrupt

    cfg = TrainConfig(warmup_steps=0, best_path=str(tmp_path / "b.pt"), last_path=str(tmp_path / "l.pt"))
    with pytest.raises(KeyboardInterrupt):
        train(make_agent(), Interrupting(), TinyEnv(), cfg)
    assert (tmp_path / "l.pt").exists()


def test_greedy_evaluation_is_repeatable_and_leaves_epsilon(make_agent):
    agent, env = make_agent(), TinyEnv()
    first = evaluate_policy(agent, env, 2, 2, seed=100)
    assert first == evaluate_policy(agent, env, 2, 2, seed=100)
    assert env.seeds == [100, 101, 100, 101]
    assert agent.env_steps == 0 and agent.epsilon == 1.0
