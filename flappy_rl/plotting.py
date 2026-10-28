"""Training curves."""
import numpy as np


def _moving_average(values, window):
    return np.convolve(values, np.ones(window) / window, mode="valid")


def _curve(ax, values, label, color, window):
    ax.plot(values, color=color, alpha=0.4, label=label)
    if len(values) >= window:
        ax.plot(range(window - 1, len(values)), _moving_average(values, window),
                color=color, label=f"Moving avg {window}")


def plot_training(history, out_path="training_results.png", window=20, show=True):
    """Plot episode reward and score (with greedy evaluation points) from a History."""
    import matplotlib.pyplot as plt

    fig, (ax_reward, ax_score) = plt.subplots(1, 2, figsize=(12, 5))
    _curve(ax_reward, history.rewards, "Episode reward", "tab:blue", window)
    _curve(ax_score, history.scores, "Episode score", "tab:green", window)
    if history.evals:
        episodes, scores = zip(*history.evals)
        ax_score.plot(episodes, scores, "o-", color="tab:orange", label="Greedy eval (mean)")
    for ax, title, ylabel in ((ax_reward, "Reward per episode", "Reward"),
                              (ax_score, "Score per episode", "Score")):
        ax.set(title=title, xlabel="Episode", ylabel=ylabel)
        ax.grid(True)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    if show:
        plt.show()
    else:
        plt.close(fig)
