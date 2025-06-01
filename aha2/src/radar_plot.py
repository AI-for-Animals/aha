import argparse
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from pathlib import Path
from inspect_ai.log import read_eval_log


def create_radar_chart(models_scores, min_scale=0, max_scale=1, output_path=None):
    first_model = list(models_scores.keys())[0]
    unordered_categories = sorted(
        list(models_scores[first_model].keys()), key=lambda x: len(x)
    )
    categories = []
    i = 0
    j = len(unordered_categories) - 1

    while i < j:
        categories.append(unordered_categories[i])
        categories.append(unordered_categories[j])
        i += 1
        j -= 1

    if i == j:
        categories.append(unordered_categories[i])

    formatted_categories = [cat.replace(
        "_", " ").title() for cat in categories]

    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))

    # Set up the axes
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], formatted_categories, size=9)
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.5", "0.75"],
               color="grey", size=8)
    plt.ylim(min_scale, max_scale)
    colors = plt.cm.tab10.colors

    # Plot each model
    for i, (model_name, scores) in enumerate(models_scores.items()):
        values = [scores[cat] for cat in categories]
        values += values[:1]

        color = colors[i % len(colors)]
        ax.plot(
            angles,
            values,
            linewidth=2,
            linestyle="solid",
            label=model_name,
            color=color,
        )
        ax.fill(angles, values, alpha=0.25, color=color)

    plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1))

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path)
    else:
        plt.show()


def extract_dimension_scores(log_file):
    eval_log = read_eval_log(log_file)
    metrics = eval_log.results.scores[0].metrics
    scores_by_dimension = {
        key: value.value
        for key, value in metrics.items()
        if key not in ["question_normalized_avg", "dimension_normalized_avg"]
    }
    return scores_by_dimension


def get_latest_log_files(lookback: int = 1) -> list[str]:
    """Find the most recent .eval file in the logs/ directory"""
    logs_dir = Path("logs")
    if not logs_dir.exists() or not logs_dir.is_dir():
        return None
    log_files = list(logs_dir.glob("*.eval"))
    if not log_files:
        return None
    items = sorted(log_files, key=lambda x: x.stat().st_mtime,
                   reverse=True)[0:lookback]
    return [str(i) for i in items]


def chart_log(log_path: str | None = None, lookback: int = 1):
    log_paths = [log_path] if log_path else get_latest_log_files(lookback)
    models_scores = {}

    for log_path in log_paths:
        eval_log = read_eval_log(log_path)
        model_name = eval_log.eval.model
        models_scores[model_name] = extract_dimension_scores(log_path)
        metrics = eval_log.results.scores[0].metrics
        print("Model: ", model_name)
        print(
            "- Question-normalized average score: ",
            int(100 * metrics["question_normalized_avg"].value),
            "%",
        )
        print(
            "- Dimension-normalized average score: ",
            int(100 * metrics["dimension_normalized_avg"].value),
            "%",
        )
        print("\n")

    create_radar_chart(models_scores)


def main():
    parser = argparse.ArgumentParser(
        description="Generate radar chart from evaluation log"
    )
    parser.add_argument(
        "log_file", nargs="?", help="Path to the evaluation log file (optional)"
    )
    parser.add_argument(
        "--output", "-o", help="Path to save the chart (optional)")
    parser.add_argument(
        "--lookback",
        "-l",
        type=int,
        default=1,
        help="Number of log files to look back (default: 1)",
    )
    args = parser.parse_args()

    log_paths = (
        [args.log_file]
        if args.log_file
        else get_latest_log_files(lookback=args.lookback)
    )

    for log_path in log_paths:
        log_path = Path(log_path)
        if not log_path.exists():
            print(f"Error: Log file {log_path} not found")
            sys.exit(1)

    models_scores = {}

    for log_path in log_paths:
        eval_log = read_eval_log(log_path)
        model_name = eval_log.eval.model
        models_scores[model_name] = extract_dimension_scores(log_path)
        metrics = eval_log.results.scores[0].metrics
        print("Model: ", model_name)
        print(
            "- Question-normalized average score: ",
            int(100 * metrics["question_normalized_avg"].value),
            "%",
        )
        print(
            "- Dimension-normalized average score: ",
            int(100 * metrics["dimension_normalized_avg"].value),
            "%",
        )
        print("\n")

    output_path = args.output
    if not output_path:
        results_dir = Path("results")
        os.makedirs(results_dir, exist_ok=True)
        output_path = results_dir / f"{Path(log_paths[-1]).stem}_radar.png"
        print(f"Output will be saved to: {output_path}")

    create_radar_chart(models_scores, output_path=output_path)


if __name__ == "__main__":
    main()
