"""Export greedy replays and MLP weights for the 3D browser viewer."""
import base64
import json
import os

import numpy as np

from .envs import make_env

REPLAY_FORMAT = "flappy-replay-v1"
WEIGHTS_FORMAT = "flappy-dueling-mlp-v1"


def _snapshot(env, rec, pipe_gap):
    """Append the current game state (288x512 coordinates) to a recording."""
    game = env.unwrapped
    rec["pipes"].append([v for up, low in zip(game._upper_pipes, game._lower_pipes)
                         for v in (round(float(up["x"]), 2), round(float(low["y"]) - pipe_gap, 2))])
    rec["y"].append(round(float(game._player_y), 2))
    rec["vel"].append(float(game._player_vel_y))
    rec["rot"].append(float(game._player_rot))
    rec["score_t"].append(int(game._score))


def _record_episode(agent, env, pipe_gap, max_steps):
    rec = {"y": [], "vel": [], "rot": [], "score_t": [], "pipes": [], "act": [], "q": [],
           "terminated": False}
    state, _ = env.reset()
    _snapshot(env, rec, pipe_gap)
    for _ in range(max_steps):
        q = agent.q_values(state)
        action = int(q.argmax())
        state, _, terminated, truncated, _ = env.step(action)
        rec["act"].append(action)
        rec["q"].append([round(float(v), 4) for v in q])
        _snapshot(env, rec, pipe_gap)
        if terminated or truncated:
            rec["terminated"] = bool(terminated)
            break
    rec["score"] = rec["score_t"][-1]
    return rec


def record_replay(agent, num_episodes=3, max_steps=3000, out_path="flappy_replay.json"):
    """Record greedy episodes (both modes) as JSON."""
    env = make_env(agent.config.mode)
    try:
        pipe_gap = float(getattr(env.unwrapped, "_pipe_gap", 100))
        episodes = [_record_episode(agent, env, pipe_gap, max_steps) for _ in range(num_episodes)]
    finally:
        env.close()
    replay = {"format": REPLAY_FORMAT, "mode": agent.config.mode, "pipe_gap": pipe_gap,
              "episodes": episodes}
    with open(out_path, "w") as f:
        json.dump(replay, f, separators=(",", ":"))
    print(f"Replay: {out_path} ({os.path.getsize(out_path) / 1024:.0f} KB), "
          f"scores {[e['score'] for e in episodes]}")
    return replay


def export_mlp_weights(agent, out_path="flappy_mlp_weights.json"):
    """Export MLP weights (base64 float32) so the browser can run the policy. State mode only."""
    if agent.config.mode != "state":
        print("Skipped weights export: only the state-mode MLP runs in the browser.")
        return None
    tensors = {
        name: {"shape": list(t.shape),
               "data": base64.b64encode(t.detach().cpu().numpy().astype("<f4").tobytes()).decode("ascii")}
        for name, t in agent.policy_net.state_dict().items()
    }
    weights = {"format": WEIGHTS_FORMAT, "tensors": tensors}
    with open(out_path, "w") as f:
        json.dump(weights, f)
    print(f"Weights: {out_path} ({os.path.getsize(out_path) / 1024:.0f} KB)")
    return weights
