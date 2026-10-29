"""Step-budget training loop with periodic greedy evaluation and best-checkpoint selection."""
import csv
import json
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from pathlib import Path

import numpy as np

from .evaluation import evaluate_policy

PROGRESS_FIELDS = ["step", "episodes", "avg_score", "max_score", "avg_reward", "loss", "epsilon",
                   "steps_per_sec", "elapsed_sec", "eval_score"]
AVG_WINDOW = 20  # episodes in the running averages


@dataclass
class History:
    episodes: list = field(default_factory=list)  # [env_step, reward, score, length]
    evals: list = field(default_factory=list)     # [env_step, mean greedy score]

    @property
    def steps(self):
        return [e[0] for e in self.episodes]

    @property
    def rewards(self):
        return [e[1] for e in self.episodes]

    @property
    def scores(self):
        return [e[2] for e in self.episodes]

    def save(self, path):
        path = Path(path)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(asdict(self)), encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls([list(e) for e in data["episodes"]], [list(e) for e in data["evals"]])


class _ProgressLog:
    """Appends one CSV row per log interval (survives resumes into the same folder)."""

    def __init__(self, path):
        self.path = Path(path)
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(PROGRESS_FIELDS)

    def write(self, row):
        with self.path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, PROGRESS_FIELDS).writerow(row)


def _save_run_config(agent, cfg):
    config = {"agent": asdict(agent.config), "train": asdict(cfg), "resumed_at_step": agent.env_steps}
    name = "config.json" if agent.env_steps == 0 else f"config_resume_{agent.env_steps}.json"
    (Path(cfg.out_dir) / name).write_text(json.dumps(config, indent=2), encoding="utf-8")


def _warmup(agent, env, cfg):
    """Fill the replay buffer before learning starts.

    A fresh agent acts randomly here (epsilon = 1); a resumed one uses its current epsilon,
    so the refilled buffer matches its policy. Warmup steps do not move the epsilon schedule.
    """
    state, _ = env.reset(seed=cfg.seed)
    length = 0
    for _ in range(cfg.warmup_steps):
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        length += 1
        truncated = truncated or length >= cfg.max_episode_steps
        agent.observe(state, action, reward, next_state, terminated, truncated, count=False)
        state = next_state
        if terminated or truncated:
            state, _ = env.reset()
            length = 0
    agent.nstep.clear()  # the next reset starts a new episode


def train(agent, env, eval_env, cfg, history=None):
    """Train `agent` until it has taken cfg.total_steps environment steps; return a History.

    Pass the History loaded from a previous run (and an agent from its last.pt) to resume.
    Every eval saves last.pt and history.json, and best.pt when the greedy score improves.
    Ctrl+C saves both before re-raising. Closing the environments is up to the caller.
    """
    if env is eval_env:
        raise ValueError("Training and evaluation need separate environments")
    if agent.memory.capacity < agent.config.batch_size:
        raise ValueError("buffer_capacity must be at least batch_size")

    Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)
    history = history if history is not None else History()
    best = max((score for _, score in history.evals), default=-np.inf)
    _save_run_config(agent, cfg)
    log = _ProgressLog(cfg.progress_path)

    _warmup(agent, env, cfg)
    print(f"Warmup done: {len(agent.memory)} transitions | start at step {agent.env_steps:,} "
          f"of {cfg.total_steps:,} | output {cfg.out_dir}")

    start_time, start_step = time.time(), agent.env_steps
    state, _ = env.reset(seed=cfg.seed + 1 + agent.env_steps)
    ep_reward, ep_len, score = 0.0, 0, 0
    losses = deque(maxlen=1_000)
    try:
        while agent.env_steps < cfg.total_steps:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            ep_len += 1
            ep_reward += reward
            score = info.get("score", score)
            truncated = truncated or ep_len >= cfg.max_episode_steps
            agent.observe(state, action, reward, next_state, terminated, truncated)
            state = next_state

            step = agent.env_steps
            if step % cfg.train_every == 0:
                loss = agent.update()
                if loss is not None:
                    losses.append(loss)

            if terminated or truncated:
                history.episodes.append([step, round(float(ep_reward), 3), int(score), ep_len])
                state, _ = env.reset()
                ep_reward, ep_len, score = 0.0, 0, 0

            finished = step >= cfg.total_steps
            eval_score = ""
            if step % cfg.eval_every == 0 or finished:
                eval_score = evaluate_policy(agent, eval_env, cfg.eval_episodes,
                                             cfg.eval_max_steps, cfg.eval_seed)
                history.evals.append([step, eval_score])
                improved = eval_score > best
                if improved:
                    best = eval_score
                    agent.save(cfg.best_path)
                agent.save(cfg.last_path)
                history.save(cfg.history_path)
                print(f"[Eval] step {step:>10,} | greedy score {eval_score:7.2f} | best {best:7.2f}"
                      f"{'  <- saved best.pt' if improved else ''}")

            if step % cfg.log_every == 0 or finished:
                elapsed = time.time() - start_time
                recent = history.episodes[-AVG_WINDOW:]
                row = {
                    "step": step,
                    "episodes": len(history.episodes),
                    "avg_score": round(float(np.mean([e[2] for e in recent])), 3) if recent else 0.0,
                    "max_score": max(history.scores, default=0),
                    "avg_reward": round(float(np.mean([e[1] for e in recent])), 3) if recent else 0.0,
                    "loss": round(float(np.mean(losses)), 5) if losses else 0.0,
                    "epsilon": round(agent.epsilon, 4),
                    "steps_per_sec": round((step - start_step) / max(elapsed, 1e-9), 1),
                    "elapsed_sec": round(elapsed, 1),
                    "eval_score": eval_score,
                }
                log.write(row)
                print(f"step {step:>10,} | ep {row['episodes']:6d} | score avg{AVG_WINDOW} "
                      f"{row['avg_score']:6.2f} max {row['max_score']:4d} | reward avg "
                      f"{row['avg_reward']:7.2f} | loss {row['loss']:.4f} | eps {row['epsilon']:.3f} | "
                      f"{row['steps_per_sec']:5.0f} st/s | {timedelta(seconds=int(elapsed))}")
    except KeyboardInterrupt:
        agent.save(cfg.last_path)
        history.save(cfg.history_path)
        print(f"Interrupted at step {agent.env_steps:,}; saved {cfg.last_path}")
        raise
    agent.save(cfg.last_path)
    history.save(cfg.history_path)
    return history
