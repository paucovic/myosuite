import argparse
import json
import matplotlib.pyplot as plt


def load_jsonl(path):
    rows = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    return rows


def get_all_columns(rows):
    cols = set()
    for row in rows:
        cols.update(row.keys())
    return sorted(cols)


def get_column(rows, col):
    xs = []
    ys = []

    for row in rows:
        if "num_steps" in row and col in row:
            xs.append(row["num_steps"])
            ys.append(row[col])

    return xs, ys


def rolling_mean(values, window):
    if window <= 1:
        return values

    smoothed = []

    for i in range(len(values)):
        start = max(0, i - window + 1)
        subset = values[start:i + 1]
        smoothed.append(sum(subset) / len(subset))

    return smoothed


def clean_title(col):
    return col.replace("/", " - ").replace("_", " ").strip()


def plot_single_metric(rows, col, smoothing, ylabel):
    x, y = get_column(rows, col)

    if len(x) == 0:
        return False

    y = rolling_mean(y, smoothing)
    title = clean_title(col)

    # One figure/window per metric.
    plt.figure(title, figsize=(10, 5))
    plt.plot(x, y, label=col)

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
        raise ValueError(f"No rows found in {jsonl_path}")

    columns = get_all_columns(rows)

    print("\nAvailable columns:")
    for col in columns:
        print(" ", col)

    if "num_steps" not in columns:
        raise ValueError("Could not find 'num_steps' in the metrics file.")

    # ============================================================
    # METRICS TO PLOT
    # ============================================================

    episode_reward_columns = [
        "eval/episode_reward",
        "eval/episode_return",
        "training/episode_reward",
        "training/episode_return",
        "training/reward",
        "training/episode_metrics/reward",
    ]

    evaluation_columns = [
        "eval/episode_reach_reward",
        "eval/episode_bonus_reward",
        "eval/episode_penalty_reward",
        "eval/episode_pose_reward",
        "eval/episode_act_reg_reward",
        "eval/episode_solved_frac",
        "eval/avg_episode_length",
    ]

    loss_columns = [
        "training/total_loss",
        "training/policy_loss",
        "training/v_loss",
        "training/value_loss",
        "training/entropy_loss",
        "training/kl_mean",
    ]

    plotted_columns = []

    # ============================================================
    # PLOT EPISODE REWARDS
    # ============================================================

    for col in episode_reward_columns:
        if col in columns:
            made_plot = plot_single_metric(
                rows=rows,
                col=col,
                smoothing=smoothing,
                ylabel="Episode reward / return",
            )
            if made_plot:
                plotted_columns.append(col)

    # ============================================================
    # PLOT EVALUATION METRICS
    # ============================================================

    for col in evaluation_columns:
        if col in columns:
            made_plot = plot_single_metric(
                rows=rows,
                col=col,
                smoothing=smoothing,
                ylabel="Evaluation metric value",
            )
            if made_plot:
                plotted_columns.append(col)

    # ============================================================
    # PLOT LOSSES
    # ============================================================

    for col in loss_columns:
        if col in columns:
            made_plot = plot_single_metric(
                rows=rows,
                col=col,
                smoothing=smoothing,
                ylabel="Loss / training metric value",
            )
            if made_plot:
                plotted_columns.append(col)

    if len(plotted_columns) == 0:
        print("\nNo selected reward, evaluation, or loss columns were found.")
        print("The available columns printed above are the only ones present.")
        return

    print("\nCreated one separate figure/window for each selected metric:")
    for col in plotted_columns:
        print(" ", col)

    print("\nDisplaying plot window(s). Close the window(s) to finish the script.")
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
        help="Rolling average window. Use 1 for no smoothing.",
    )

    args = parser.parse_args()
    main(args.jsonl_path, args.smoothing)