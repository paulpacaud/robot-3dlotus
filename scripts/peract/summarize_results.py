import os
import numpy as np
import jsonlines
import collections
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class TaskMetrics:
    success_rate: float
    fps: float
    std: float = 0.0


def process_results(
    result_file: str, ckpt_step: int
) -> Tuple[Dict[str, TaskMetrics], TaskMetrics]:
    task_results = collections.defaultdict(lambda: {"sr": [], "fps": []})
    seen = set()

    with jsonlines.open(result_file, "r") as f:
        for item in f:
            if isinstance(item.get("checkpoint"), int):
                res_ckpt = item["checkpoint"]
            else:
                try:
                    res_ckpt = int(
                        os.path.basename(item["checkpoint"])
                        .split("_")[-1]
                        .split(".")[0]
                    )
                except (AttributeError, IndexError, ValueError):
                    continue

            if res_ckpt != ckpt_step:
                continue

            key = (item["task"], item["variation"])
            if key in seen:
                continue
            seen.add(key)

            task_str = item["task"]
            success_rate = item["sr"] * 100
            fps = item.get("fps", 0.0)

            task_results[task_str]["sr"].append(success_rate)
            task_results[task_str]["fps"].append(fps)

    task_averages = {}
    all_success_rates = []
    all_fps = []
    all_stds = []

    for task_str, metrics in task_results.items():
        if not metrics["sr"]:
            continue

        avg_success_rate = np.mean(metrics["sr"])
        avg_fps = np.mean(metrics["fps"])
        std = np.std(metrics["sr"]) if len(metrics["sr"]) > 1 else 0.0

        task_averages[task_str] = TaskMetrics(
            success_rate=round(avg_success_rate, 1),
            fps=round(avg_fps, 1),
            std=round(std, 1),
        )

        all_success_rates.append(avg_success_rate)
        all_fps.append(avg_fps)
        all_stds.append(std)

    overall_metrics = TaskMetrics(
        success_rate=round(np.mean(all_success_rates), 1) if all_success_rates else 0.0,
        fps=round(np.mean(all_fps), 1) if all_fps else 0.0,
        std=round(np.mean(all_stds), 1) if all_stds else 0.0,
    )

    return task_averages, overall_metrics


def display_results(
    task_averages: Dict[str, TaskMetrics], overall_metrics: TaskMetrics, ckpt_step: int
):
    # Skip if no results
    if len(task_averages) == 0:
        return

    # ANSI color codes
    RED = "\033[91m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"

    def color_success_rate(rate: float) -> str:
        if rate < 33:
            return f"{RED}{rate:>11.1f}%{RESET}"
        elif rate < 70:
            return f"{YELLOW}{rate:>11.1f}%{RESET}"
        return f"{rate:>11.1f}%"

    print(f"\nResults for checkpoint step {ckpt_step}:")
    print("\nPer-Task Metrics:")
    print("-" * 75)
    print(f"{'Task':<30} {'Success Rate':>12} {'Std Dev':>12} {'FPS':>12}")
    print("-" * 75)

    for task_str, metrics in sorted(task_averages.items()):
        print(
            f"{task_str:<30} {color_success_rate(metrics.success_rate)} {metrics.std:>11.1f}% {metrics.fps:>11.1f}"
        )

    print("\n" + "-" * 75)
    print(f"Overall Averages:")
    print(
        f"Success Rate: {color_success_rate(overall_metrics.success_rate)} (±{overall_metrics.std:.1f}%)"
    )
    print(f"FPS: {overall_metrics.fps:.1f}")
    print("-" * 75)


def display_summary(summary):
    print("\nSummary:")
    print("-" * 75)
    for ckpt_step, metrics in summary.items():
        print(
            f"{ckpt_step}: {metrics.success_rate:.1f}% (±{metrics.std:.1f}%), FPS: {metrics.fps:.1f}"
        )

    print("-" * 75)
    print("Best Checkpoint:")
    best_ckpt = max(summary, key=lambda x: summary[x].success_rate)
    print(
        f"{best_ckpt}: {summary[best_ckpt].success_rate:.1f}% (±{summary[best_ckpt].std:.1f}%), FPS: {summary[best_ckpt].fps:.1f}"
    )


def main(result_file, ckpt_steps):
    summary = {}
    for ckpt_step in ckpt_steps:
        task_averages, overall_metrics = process_results(result_file, ckpt_step)
        display_results(task_averages, overall_metrics, ckpt_step)
        if overall_metrics.success_rate > 0:
            summary[ckpt_step] = overall_metrics
    display_summary(summary)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("result_file", help="Path to the results file")
    parser.add_argument(
        "--ckpt_steps",
        nargs="+",
        type=int,
        help="Checkpoint steps to filter results (space-separated list)",
        default=[
            180000,
            170000,
            160000,
            150000,
            140000,
            130000,
            120000,
            110000,
            100000,
            90000,
            80000,
            70000,
            60000,
        ],
    )
    args = parser.parse_args()
    main(args.result_file, args.ckpt_steps)
