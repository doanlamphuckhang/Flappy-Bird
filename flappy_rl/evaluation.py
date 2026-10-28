"""Greedy evaluation and gameplay video."""
import numpy as np

from .envs import make_env


def run_episode(agent, env, max_steps, seed=None, callback=None):
    """Play one greedy episode and return its score.

    `callback(env)` runs after reset and after every step (e.g. to record frames).
    """
    state, _ = env.reset(seed=seed)
    if callback:
        callback(env)
    score = 0
    for _ in range(max_steps):
        state, _, terminated, truncated, info = env.step(agent.select_action(state, evaluate=True))
        score = info.get("score", score)
        if callback:
            callback(env)
        if terminated or truncated:
            break
    return score


def evaluate_policy(agent, env, num_episodes=5, max_steps=3000, seed=10042):
    """Mean greedy score over `num_episodes` episodes seeded `seed, seed+1, ...`."""
    if num_episodes <= 0 or max_steps <= 0:
        raise ValueError("num_episodes and max_steps must be positive")
    return float(np.mean([run_episode(agent, env, max_steps, seed + i) for i in range(num_episodes)]))


def record_video(agent, video_path="flappy_gameplay.mp4", num_episodes=3, max_steps=1500, fps=30):
    """Stream greedy gameplay to an MP4 file and return the episode scores."""
    import imageio

    env = make_env(agent.config.mode, render=True)
    try:
        with imageio.get_writer(video_path, fps=fps) as writer:
            scores = [run_episode(agent, env, max_steps,
                                  callback=lambda e: writer.append_data(e.get_frame()))
                      for _ in range(num_episodes)]
    finally:
        env.close()
    print(f"Video: {video_path} | scores {scores} | mean {np.mean(scores):.2f}")
    return scores
