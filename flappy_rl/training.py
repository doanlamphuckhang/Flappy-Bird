"""Training loop with periodic greedy evaluation and best-checkpoint selection."""
import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .config import NUM_ACTIONS
from .evaluation import evaluate_policy


@dataclass
class History:
    rewards: list = field(default_factory=list)
    scores: list = field(default_factory=list)
    evals: list = field(default_factory=list)  # (episode, mean greedy score)

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self)), encoding="utf-8")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["rewards"], data["scores"], [tuple(e) for e in data["evals"]])


def _warmup(agent, env, steps, seed):
    """Fill the replay buffer with random-policy transitions."""
    state, _ = env.reset(seed=seed)
    for _ in range(steps):
        action = random.randrange(NUM_ACTIONS)
        next_state, reward, terminated, truncated, _ = env.step(action)
        # Bootstrap through truncation; only true termination zeroes the future value.
        agent.memory.push(state, action, reward, next_state, terminated)
        state = env.reset()[0] if terminated or truncated else next_state


def _run_episode(agent, env, cfg):
    state, _ = env.reset()
    total_reward, score, losses = 0.0, 0, []
    for _ in range(cfg.max_steps):
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        agent.observe(state, action, reward, next_state, terminated)
        if agent.env_steps % cfg.train_every == 0:
            loss = agent.update()
            if loss is not None:
                losses.append(loss)
        state = next_state
        total_reward += reward
        score = info.get("score", score)
        if terminated or truncated:
            break
    return total_reward, score, losses


def train(agent, env, eval_env, cfg):
    """Train `agent` and return a History. Closing the environments is up to the caller.

    The best checkpoint (by greedy evaluation score) goes to cfg.best_path, the final one to
    cfg.last_path. Ctrl+C saves cfg.last_path before re-raising.
    """
    if env is eval_env:
        raise ValueError("Training and evaluation need separate environments")
    if agent.memory.capacity < agent.config.batch_size:
        raise ValueError("buffer_capacity must be at least batch_size")

    history, best, start = History(), -np.inf, time.time()
    _warmup(agent, env, cfg.warmup_steps, cfg.seed)
    print(f"Warmup done: {len(agent.memory)} transitions")
    try:
        for episode in range(1, cfg.num_episodes + 1):
            reward, score, losses = _run_episode(agent, env, cfg)
            history.rewards.append(reward)
            history.scores.append(score)

            if episode % cfg.eval_every == 0 or episode == cfg.num_episodes:
                mean_score = evaluate_policy(agent, eval_env, cfg.eval_episodes,
                                             cfg.max_steps, cfg.eval_seed)
                history.evals.append((episode, mean_score))
                improved = mean_score > best
                if improved:
                    best = mean_score
                    agent.save(cfg.best_path)
                print(f"[Eval] ep {episode:4d} | greedy score {mean_score:6.2f}"
                      f"{'  <- best, saved' if improved else ''}")

            if episode % cfg.log_every == 0:
                print(f"ep {episode:4d} | score {score:3d} (max {max(history.scores):3d}, "
                      f"avg{cfg.log_every} {np.mean(history.scores[-cfg.log_every:]):5.1f}) | "
                      f"reward {reward:7.1f} | loss {np.mean(losses) if losses else 0.0:.4f} | "
                      f"eps {agent.epsilon:.3f} | {time.time() - start:6.0f}s")
    except KeyboardInterrupt:
        agent.save(cfg.last_path)
        print(f"Interrupted; saved {cfg.last_path}")
        raise
    agent.save(cfg.last_path)
    return history
