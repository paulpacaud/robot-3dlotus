import os
import numpy as np
import jsonlines
import collections
from typing import Dict, List, Tuple, NamedTuple
from dataclasses import dataclass


@dataclass
class TaskMetrics:
    success_rate: float
    fps: float
    std: float = 0.0
    total_episodes: int = 0


class VariationMetrics(NamedTuple):
    variation: str
    success_rate: float
    fps: float
    num_episodes: int
    avg_inference_time: float


@dataclass
class DetailedTaskMetrics:
    # Overall metrics for the task
    aggregate: TaskMetrics
    # List of metrics for each variation
    variations: List[VariationMetrics]


def process_results(
    result_file: str, ckpt_step: int
) -> Tuple[Dict[str, DetailedTaskMetrics], TaskMetrics]:
    # Store both aggregate results and individual variation results
    task_results = collections.defaultdict(
        lambda: {
            "sr": [],
            "fps": [],
            "episodes": 0,
            "variations": collections.defaultdict(
                lambda: {"sr": None, "fps": None, "episodes": 0, "inference_time": None}
            ),
        }
    )
    seen = set()

    # Process the results file
    with jsonlines.open(result_file, "r") as f:
        for item in f:
            # Extract checkpoint step
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

            # Skip if not matching desired checkpoint
            if res_ckpt != ckpt_step:
                continue

            key = (item["task"], item["variation"])
            if key in seen:
                continue
            seen.add(key)

            task_str = item["task"]
            variation = item["variation"]
            success_rate = item["sr"] * 100
            fps = item.get("fps", 0.0)
            num_episodes = item.get("num_demos", 0)  # Get number of episodes
            avg_inference_time = item.get("avg_inference_time", 0.0)

            # Store both aggregate and variation-specific results
            task_results[task_str]["sr"].append(success_rate)
            task_results[task_str]["fps"].append(fps)
            task_results[task_str]["episodes"] += num_episodes
            task_results[task_str]["variations"][variation] = {
                "sr": success_rate,
                "fps": fps,
                "episodes": num_episodes,
                "inference_time": avg_inference_time,
            }

    # Process results into final format
    task_averages = {}
    all_success_rates = []
    all_fps = []
    all_stds = []
    total_episodes = 0

    for task_str, metrics in task_results.items():
        if not metrics["sr"]:
            continue

        # Calculate aggregate metrics
        avg_success_rate = np.mean(metrics["sr"])
        avg_fps = np.mean(metrics["fps"])
        std = np.std(metrics["sr"]) if len(metrics["sr"]) > 1 else 0.0
        task_episodes = metrics["episodes"]

        # Create list of variation metrics
        variation_metrics = [
            VariationMetrics(
                variation=var_name,
                success_rate=round(var_data["sr"], 1),
                fps=round(var_data["fps"], 1),
                num_episodes=var_data["episodes"],
                avg_inference_time=var_data["inference_time"],
            )
            for var_name, var_data in metrics["variations"].items()
        ]

        # Sort variations by success rate
        variation_metrics.sort(key=lambda x: x.success_rate, reverse=True)

        # Create detailed metrics object
        task_averages[task_str] = DetailedTaskMetrics(
            aggregate=TaskMetrics(
                success_rate=round(avg_success_rate, 1),
                fps=round(avg_fps, 1),
                std=round(std, 1),
                total_episodes=task_episodes,
            ),
            variations=variation_metrics,
        )

        all_success_rates.append(avg_success_rate)
        all_fps.append(avg_fps)
        all_stds.append(std)
        total_episodes += task_episodes

    # Calculate overall metrics
    overall_metrics = TaskMetrics(
        success_rate=round(np.mean(all_success_rates), 1) if all_success_rates else 0.0,
        fps=round(np.mean(all_fps), 1) if all_fps else 0.0,
        std=round(np.mean(all_stds), 1) if all_stds else 0.0,
        total_episodes=total_episodes,
    )

    return task_averages, overall_metrics


def display_results(
    task_averages: Dict[str, DetailedTaskMetrics],
    overall_metrics: TaskMetrics,
    ckpt_step: int,
):
    if len(task_averages) == 0:
        return

    # ANSI color codes
    RED = "\033[91m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    RESET = "\033[0m"

    def color_success_rate(rate: float) -> str:
        if rate < 33:
            return f"{RED}{rate:>11.1f}%{RESET}"
        elif rate < 70:
            return f"{YELLOW}{rate:>11.1f}%{RESET}"
        return f"{GREEN}{rate:>11.1f}%{RESET}"

    print(f"\nResults for checkpoint step {ckpt_step}:")

    # Display per-task results with variations
    for task_str, metrics in sorted(task_averages.items()):
        print("\n" + "=" * 120)
        print(f"Task: {task_str}")
        print("-" * 120)

        # Display aggregate metrics
        print(
            f"Aggregate Metrics (Total Episodes: {metrics.aggregate.total_episodes}):"
        )
        print(
            f"Average Success Rate: {color_success_rate(metrics.aggregate.success_rate)} (±{metrics.aggregate.std:>5.1f}%)"
        )
        print(f"Average FPS: {metrics.aggregate.fps:>11.1f}")

        # Display variation breakdown
        print("\nVariation Breakdown:")
        print(
            f"{'Variation':<50} {'Success Rate':>12} {'Episodes':>10} {'FPS':>8} {'Inf Time':>10}"
        )
        print("-" * 120)

        for var_metrics in metrics.variations:
            print(
                f"{var_metrics.variation:<50} "
                f"{color_success_rate(var_metrics.success_rate)} "
                f"{var_metrics.num_episodes:>10} "
                f"{var_metrics.fps:>8.1f} "
                f"{var_metrics.avg_inference_time:>10.3f}s"
            )

    # Display overall metrics
    print("\n" + "=" * 120)
    print(f"Overall Metrics (Total Episodes: {overall_metrics.total_episodes}):")
    print(
        f"Success Rate: {color_success_rate(overall_metrics.success_rate)} (±{overall_metrics.std:.1f}%)"
    )
    print(f"FPS: {overall_metrics.fps:.1f}")
    print("=" * 120)


def display_summary(summary):
    print("\nCheckpoint Summary:")
    print("=" * 85)
    for ckpt_step, metrics in summary.items():
        print(
            f"Step {ckpt_step}: {metrics.success_rate:.1f}% (±{metrics.std:.1f}%), "
            f"FPS: {metrics.fps:.1f}, Episodes: {metrics.total_episodes}"
        )

    print("\nBest Checkpoint:")
    best_ckpt = max(summary, key=lambda x: summary[x].success_rate)
    print(
        f"Step {best_ckpt}: {summary[best_ckpt].success_rate:.1f}% "
        f"(±{summary[best_ckpt].std:.1f}%), FPS: {summary[best_ckpt].fps:.1f}, "
        f"Episodes: {summary[best_ckpt].total_episodes}"
    )
    print("=" * 85)


def main(result_file, ckpt_step):
    summary = {}
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
        "--ckpt_step",
        type=int,
        help="Checkpoint steps to filter results (space-separated list)",
        default=120000,
    )
    args = parser.parse_args()
    main(args.result_file, args.ckpt_step)
