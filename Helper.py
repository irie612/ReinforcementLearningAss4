import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


def plot_colormap(file_name, values, alphas, gammas, cmap):
    fig, ax = plt.subplots()
    c = ax.matshow(values, cmap=cmap)
    ax.set_xticklabels([''] + gammas)
    ax.set_yticklabels([''] + alphas)
    fig.colorbar(c, ax=ax)
    for (i, j), values in np.ndenumerate(values):
        ax.text(j, i, format(values, ".4f"), ha='center', va='center')
    fig.savefig(f"{file_name}.png", dpi=300)


def plot_reward_graph(rewards, labels, file_name, legend_title, y_lim=None):
    fig, ax = plt.subplots()
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.legendlabel_format()
    for i, r in enumerate(rewards):
        ax.plot(r, label=labels[i])
    ax.legend(title=legend_title)
    if y_lim is not None:
        plt.ylim(y_lim)
    fig.savefig(f"{file_name}.png", dpi=300)


def smooth(y, window, poly=1):
    return savgol_filter(y, window, poly)
