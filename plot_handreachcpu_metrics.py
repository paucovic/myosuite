#!/usr/bin/env python3
"""Plot Stable-Baselines3 JSONL training metrics.

Creates one separate figure/window for every available selected metric.
"""

import argparse
import json

import matplotlib.pyplot as plt


def load_jsonl(path):
    rows = []

    with open(path, "r") as file:
        for line in file:
            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows


def get_all_columns(rows):
    columns = set()

    for row in rows:
        columns.update(row.keys())

    return sorted(columns)


def get_x_value(row):
    """Support both the classic SB3 file and your MJX JSONL files."""

    if "time/total_timesteps" in row:
        return row["time/total_timesteps"]

    if "num_steps" in row:
        return row["num_steps"]

    return None


def get_series(rows, metric_name):
    x_values = []
    y_values = []

    for row in rows:
        x_value = get_x_value(row)

        if x_value is None:
            continue

        if metric_name not in row:
            continue

        x_values.append(x_value)
        y_values.append(row[metric_name])

    return x_values, y_values


def rolling_mean(values, window):
    if window <= 1:
        return values

    smoothed = []

    for index in range(len(values)):
        start = max(
            0,
            index - window + 1,
        )

        subset = values[start:index + 1]

        smoothed.append(
            sum(subset) / len(subset)
        )

    return smoothed


def clean_title(metric_name):
    return (
        metric_name
        .replace("/", " - ")
        .replace("_", " ")
        .strip()
    )


def plot_metric(
    rows,
    metric_name,
    smoothing,
    ylabel,
):
    x_values, y_values = get_series(
        rows,
        metric_name,
    )

    if len(x_values) == 0:
        return False

    y_values = rolling_mean(
        y_values,
        smoothing,
    )

    title = clean_title(metric_name)

    plt.figure(
        title,
        figsize=(10, 5),
    )

    plt.plot(
        x_values,
        y_values,
        label=metric_name,
    )

    plt.xlabel("Environment steps")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    return True


def main(jsonl_path, smoothing):
    rows = load_jsonl(jsonl_path)

    if len(rows) == 0:
        raise ValueError(
            f"No metric rows found in {jsonl_path}"
        )

    columns = get_all_columns(rows)

    print()
    print("Available columns:")

    for column in columns:
        print(" ", column)

    metrics_to_plot = [
        (
            "eval/mean_reward",
            "Evaluation episode reward",
        ),
        (
            "eval/mean_ep_length",
            "Evaluation episode length",
        ),
        (
            "rollout/ep_rew_mean",
            "Training episode reward",
        ),
        (
            "rollout/ep_len_mean",
            "Training episode length",
        ),
        (
            "train/loss",
            "Total PPO loss",
        ),
        (
            "train/policy_gradient_loss",
            "Policy loss",
        ),
        (
            "train/value_loss",
            "Value loss",
        ),
        (
            "train/entropy_loss",
            "Entropy loss",
        ),
        (
            "train/approx_kl",
            "Approximate KL divergence",
        ),
        (
            "train/clip_fraction",
            "Clip fraction",
        ),
        (
            "train/explained_variance",
            "Explained variance",
        ),
        (
            "train/learning_rate",
            "Learning rate",
        ),
    ]

    plotted_metrics = []

    for metric_name, ylabel in metrics_to_plot:
        if metric_name not in columns:
            continue

        plotted = plot_metric(
            rows=rows,
            metric_name=metric_name,
            smoothing=smoothing,
            ylabel=ylabel,
        )

        if plotted:
            plotted_metrics.append(metric_name)

    if len(plotted_metrics) == 0:
        print()
        print(
            "No selected evaluation, reward, or loss "
            "metrics were found."
        )
        return

    print()
    print("Created plots for:")

    for metric_name in plotted_metrics:
        print(" ", metric_name)

    print()
    print(
        "Close all plot windows to finish the script."
    )

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "jsonl_path",
        type=str,
        help="Path to the *_all_metrics.jsonl file.",
    )

    parser.add_argument(
        "--smoothing",
        type=int,
        default=1,
        help=(
            "Rolling-average window. "
            "Use 1 for no smoothing."
        ),
    )

    args = parser.parse_args()

    main(
        jsonl_path=args.jsonl_path,
        smoothing=args.smoothing,
    )