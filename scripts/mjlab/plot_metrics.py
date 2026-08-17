#!/usr/bin/env python3
"""
Shows each RL metrics

Run from the repo root:

    python scripts/mjlab/plot_metrics.py --metrics-path my_elbow_policy_all_metrics.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]


# =============================================================================
# USER CONFIG
# =============================================================================

DEFAULT_METRICS_PATH = "my_elbow_policy_metrics.jsonl"

# Possible x-axis names.
X_AXIS_CANDIDATES = (
    "env_steps",
    "total_steps",
    "num_steps",
    "steps",
    "timestep",
    "timesteps",
    "time/total_timesteps",
    "iteration",
    "update",
    "epoch",
    "eval_index",
)

# By default, only plot RL-relevant metrics.
PLOT_ONLY_RL_METRICS = True

RL_KEYWORDS = (
    "reward",
    "return",
    "loss",
    "actor",
    "critic",
    "value",
    "policy",
    "surrogate",
    "entropy",
    "kl",
    "clip",
    "grad",
    "explained_variance",
    "learning_rate",
    "lr",
    "fps",
    "episode_length",
    "ep_len",
    "ep_rew",
    "mean_reward",
)

SKIP_KEYS = {
    "eval_index",
    "iteration",
    "update",
    "epoch",
    "env_steps",
    "total_steps",
    "num_steps",
    "steps",
    "timestep",
    "timesteps",
    "time/total_timesteps",
    "elapsed_sec",
    "time",
    "wall_time",
    "seed",
}

DISPLAY_NAME_OVERRIDES = {
    "eval/episode_reward": "Evaluation episode return",
    "eval/mean_reward": "Evaluation mean return",
    "eval/reward_per_step": "Evaluation reward per step",

    "train/value": "Critic value loss",
    "train/value_loss": "Critic value loss",
    "train/critic_value": "Critic value loss",
    "train/critic_value_loss": "Critic value loss",

    "train/surrogate": "Policy surrogate loss",
    "train/surrogate_loss": "Policy surrogate loss",
    "train/policy_loss": "Policy surrogate loss",
    "train/policy_gradient_loss": "Policy gradient loss",

    "train/entropy": "Entropy loss",
    "train/entropy_loss": "Entropy loss",

    "train/approx_kl": "KL mean",
    "train/kl": "KL mean",
    "train/kl_mean": "KL mean",
    "train/mean_kl": "KL mean",
    "train/kl_divergence": "KL mean",

    "train/clip_fraction": "Clip fraction",
    "train/explained_variance": "Explained variance",
    "train/learning_rate": "Learning rate",
}


# =============================================================================
# FILE LOADING
# =============================================================================

def resolve_path(path_arg: str) -> Path:
    path = Path(path_arg)

    if not path.is_absolute():
        path = REPO_ROOT / path

    return path


def load_json_or_jsonl(metrics_path: Path) -> Any:
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    if metrics_path.suffix.lower() == ".jsonl":
        records = []

        with metrics_path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()

                if not line:
                    continue

                records.append(json.loads(line))

        return records

    with metrics_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def extract_records(data: Any) -> list[dict[str, Any]]:
    """
    Extract a list of metric dictionaries from common RL metric formats.
    """
    if isinstance(data, list):
        records = data

    elif isinstance(data, dict):
        if isinstance(data.get("rl_metrics"), list):
            records = data["rl_metrics"]

        elif isinstance(data.get("evals"), list):
            records = data["evals"]

        elif isinstance(data.get("metrics"), list):
            records = data["metrics"]

        elif isinstance(data.get("logs"), list):
            records = data["logs"]

        elif isinstance(data.get("history"), list):
            records = data["history"]

        else:
            raise ValueError(
                "Could not find a list of metric records. "
                "Expected one of: 'rl_metrics', 'evals', 'metrics', 'logs', "
                "or 'history'."
            )

    else:
        raise ValueError("Metrics file must contain either a list or a dictionary.")

    clean_records = []

    for record in records:
        if isinstance(record, dict):
            clean_records.append(record)

    if len(clean_records) == 0:
        raise ValueError("No valid metric dictionaries found.")

    return clean_records


# =============================================================================
# METRIC PROCESSING
# =============================================================================

def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def flatten_dict(
    data: dict[str, Any],
    prefix: str = "",
) -> dict[str, Any]:
    """
    Flatten nested dictionaries.

    Example:
        {"train": {"loss": 0.5}}
    becomes:
        {"train.loss": 0.5}
    """
    flat = {}

    for key, value in data.items():
        if prefix:
            new_key = f"{prefix}.{key}"
        else:
            new_key = str(key)

        if isinstance(value, dict):
            flat.update(flatten_dict(value, prefix=new_key))
        else:
            flat[new_key] = value

    return flat


def flatten_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [flatten_dict(record) for record in records]


def choose_x_axis(
    records: list[dict[str, Any]],
    requested_x_axis: str | None,
) -> str:
    if requested_x_axis is not None:
        return requested_x_axis

    all_keys = set()

    for record in records:
        all_keys.update(record.keys())

    for candidate in X_AXIS_CANDIDATES:
        if candidate in all_keys:
            return candidate

    raise ValueError(
        "Could not automatically choose an x-axis. "
        "Use --x-axis with one of the numeric keys in your metrics file."
    )


def is_rl_metric_key(key: str) -> bool:
    key_lower = key.lower()

    return any(keyword in key_lower for keyword in RL_KEYWORDS)


def get_metric_keys(
    records: list[dict[str, Any]],
    x_axis_key: str,
    plot_all: bool,
) -> list[str]:
    keys = set()

    for record in records:
        for key, value in record.items():
            if key == x_axis_key:
                continue

            if key in SKIP_KEYS:
                continue

            if not is_number(value):
                continue

            if not plot_all and PLOT_ONLY_RL_METRICS and not is_rl_metric_key(key):
                continue

            keys.add(key)

    return sorted(keys)


def clean_title(metric_key: str) -> str:
    if metric_key in DISPLAY_NAME_OVERRIDES:
        return DISPLAY_NAME_OVERRIDES[metric_key]

    title = metric_key

    for prefix in ("eval_", "train_", "training_", "rollout_"):
        if title.startswith(prefix):
            title = title[len(prefix):]

    title = title.replace("/", " / ")
    title = title.replace(".", " / ")
    title = title.replace("_", " ")

    return title


def clean_axis_label(axis_key: str) -> str:
    label = axis_key
    label = label.replace("/", " / ")
    label = label.replace(".", " / ")
    label = label.replace("_", " ")

    return label


# =============================================================================
# PLOTTING
# =============================================================================

def plot_metric(
    records: list[dict[str, Any]],
    metric_key: str,
    x_axis_key: str,
) -> None:
    x_values = []
    y_values = []

    for index, record in enumerate(records):
        if metric_key not in record:
            continue

        y_value = record[metric_key]

        if not is_number(y_value):
            continue

        if x_axis_key in record and is_number(record[x_axis_key]):
            x_value = record[x_axis_key]
        else:
            x_value = index

        x_values.append(x_value)
        y_values.append(y_value)

    if len(x_values) == 0:
        print(f"Skipping {clean_title(metric_key)}: no valid values")
        return

    display_name = clean_title(metric_key)

    plt.figure()
    plt.plot(x_values, y_values, marker="o")

    plt.xlabel(clean_axis_label(x_axis_key))
    plt.ylabel(display_name)
    plt.title(display_name)

    plt.grid(True)
    plt.tight_layout()

    print(f"Showing: {display_name}")
    plt.show()


def plot_all_metrics(
    metrics_path: Path,
    requested_x_axis: str | None,
    plot_all: bool,
) -> None:
    data = load_json_or_jsonl(metrics_path)
    records = extract_records(data)
    records = flatten_records(records)

    x_axis_key = choose_x_axis(
        records=records,
        requested_x_axis=requested_x_axis,
    )

    metric_keys = get_metric_keys(
        records=records,
        x_axis_key=x_axis_key,
        plot_all=plot_all,
    )

    if len(metric_keys) == 0:
        raise ValueError(
            "No numeric RL metric keys found to plot. "
            "Try running again with --all."
        )

    print("=" * 80)
    print("SHOWING RL METRICS")
    print("=" * 80)
    print(f"metrics file: {metrics_path}")
    print(f"x axis:       {x_axis_key}")
    print(f"num records:  {len(records)}")
    print(f"num metrics:  {len(metric_keys)}")
    print("=" * 80)

    print("Metrics to plot:")
    for key in metric_keys:
        print(f"  - {clean_title(key)} ({key})")

    print("=" * 80)

    for metric_key in metric_keys:
        plot_metric(
            records=records,
            metric_key=metric_key,
            x_axis_key=x_axis_key,
        )

    print("=" * 80)
    print("DONE")
    print("=" * 80)


# =============================================================================
# COMMAND LINE
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--metrics-path",
        type=str,
        default=DEFAULT_METRICS_PATH,
        help="Path to the metrics JSON or JSONL file.",
    )

    parser.add_argument(
        "--x-axis",
        type=str,
        default=None,
        help=(
            "Metric key to use as x-axis. "
            "If not provided, the script chooses automatically."
        ),
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Plot every numeric metric, not only RL-looking metrics.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    metrics_path = resolve_path(args.metrics_path)

    plot_all_metrics(
        metrics_path=metrics_path,
        requested_x_axis=args.x_axis,
        plot_all=args.all,
    )


if __name__ == "__main__":
    main()