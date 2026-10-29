"""Training curves."""
import numpy as np


def _moving_average(values, window):
    return np.convolve(values, np.ones(window) / window, mode="valid")


def _curve(ax, steps, values, label, color, window):
    ax.plot(steps, values, color=color, alpha=0.25, linewidth=0.8, label=label)
    if len(values) >= window:
        ax.plot(steps[window - 1:], _moving_average(values, window), color=color,
                label=f"Moving avg {window} episodes")


def plot_training(history, out_path="training_results.png", window=50, show=True):
    """Plot episode reward and score against env steps, with the greedy evaluation points."""
    import matplotlib.pyplot as plt

    steps = np.asarray(history.steps)
    fig, (ax_reward, ax_score) = plt.subplots(1, 2, figsize=(13, 5))
    _curve(ax_reward, steps, history.rewards, "Episode reward", "tab:blue", window)
    _curve(ax_score, steps, history.scores, "Episode score", "tab:green", window)
    if history.evals:
        eval_steps, eval_scores = zip(*history.evals, strict=True)
        ax_score.plot(eval_steps, eval_scores, "o-", color="tab:orange", label="Greedy eval (mean)")
    for ax, title, ylabel in ((ax_reward, "Reward per episode", "Reward"),
                              (ax_score, "Score per episode", "Score (pipes)")):
        ax.set(title=title, xlabel="Environment steps", ylabel=ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    if show:
        plt.show()
    else:
        plt.close(fig)
