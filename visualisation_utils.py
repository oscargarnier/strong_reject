import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime


def cumulative(scores, plot_title=None, save_path=None, quantiles=None):
    values, counts = np.unique(scores, return_counts=True)
    cumulative_counts = np.cumsum(counts)

    fig, ax = plt.subplots()
    ax.step(values, cumulative_counts, where="post")
    if quantiles is not None:
        for quantile in quantiles:
            ax.axvline(quantile, color="tab:red", linestyle="--", alpha=0.7,
                       label="Quantile boundary")
    ax.set_xlabel("Score")
    ax.set_ylabel("Cumulative count")
    plt.ylim(0, 100)           # fixed y-axis scale
    plt.title(plot_title if plot_title else 'Cumulative Frequency of Evaluation Levels',
              fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path,
                    format='pdf',
                    bbox_inches='tight',
                    pad_inches=0.1,
                    facecolor='white',
                    transparent=False)
        print(f"Chart saved to {save_path}")

    fig.show()
    return ax


def latest_evaluation_file(results_root, model, filename="evaluation_results.csv"):
    """Return the evaluation file from the newest timestamped model run."""
    model_directory = Path(results_root) / f"{model.lower()}_jbb"
    timestamp_format = "%Y%m%d_%H%M%S"
    run_directories = []

    for path in model_directory.iterdir():
        if not path.is_dir():
            continue
        try:
            timestamp = datetime.strptime(path.name, timestamp_format)
        except ValueError:
            continue
        run_directories.append((timestamp, path))

    if not run_directories:
        raise FileNotFoundError(f"No timestamped runs found in {model_directory}")

    latest_run = max(run_directories, key=lambda item: item[0])[1]
    evaluation_file = latest_run / filename
    if not evaluation_file.is_file():
        raise FileNotFoundError(f"{filename} not found in {latest_run}")
    return evaluation_file


def compute_quantiles(scores, frequencies):
    """Return score boundaries matching the cumulative category frequencies."""
    scores = np.asarray(scores)
    if scores.size == 0:
        raise ValueError("scores must not be empty")
    if not frequencies:
        raise ValueError("frequencies must not be empty")

    ordered_frequencies = [frequencies[key] for key in sorted(frequencies, key=int)]
    total = sum(ordered_frequencies)
    if total <= 0 or any(frequency < 0 for frequency in ordered_frequencies):
        raise ValueError("frequencies must be non-negative and have a positive total")

    cumulative_proportions = np.cumsum(ordered_frequencies)[:-1] / total
    return np.quantile(scores, cumulative_proportions)

