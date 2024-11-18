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


def process_results(result_file: str, ckpt_step: int) -> Tuple[Dict[str, TaskMetrics], TaskMetrics]:
    # Read results and group by task
    task_results = collections.defaultdict(lambda: {'sr': [], 'fps': []})
    seen = set()

    with jsonlines.open(result_file, 'r') as f:
        for item in f:
            # Filter by checkpoint
            if isinstance(item.get('checkpoint'), int):
                res_ckpt = item['checkpoint']
            else:
                try:
                    res_ckpt = int(os.path.basename(item['checkpoint']).split('_')[-1].split('.')[0])
                except (AttributeError, IndexError, ValueError):
                    continue

            if res_ckpt != ckpt_step:
                continue

            # Skip duplicates
            key = (item['task'], item['variation'])
            if key in seen:
                continue
            seen.add(key)

            # Extract task name, success rate, and fps
            task_str = item['task']
            success_rate = item['sr'] * 100  # Convert to percentage
            fps = item.get('fps', 0.0)  # Default to 0 if fps not found

            task_results[task_str]['sr'].append(success_rate)
            task_results[task_str]['fps'].append(fps)

    # Calculate per-task averages
    task_averages = {}
    all_success_rates = []
    all_fps = []

    for task_str, metrics in task_results.items():
        if not metrics['sr']:  # Skip if no results after filtering
            continue

        avg_success_rate = np.mean(metrics['sr'])
        avg_fps = np.mean(metrics['fps'])

        task_averages[task_str] = TaskMetrics(
            success_rate=avg_success_rate,
            fps=avg_fps
        )

        all_success_rates.append(avg_success_rate)
        all_fps.append(avg_fps)

    # Calculate overall averages
    overall_metrics = TaskMetrics(
        success_rate=np.mean(all_success_rates) if all_success_rates else 0.0,
        fps=np.mean(all_fps) if all_fps else 0.0
    )

    return task_averages, overall_metrics


def display_results(task_averages: Dict[str, TaskMetrics], overall_metrics: TaskMetrics, ckpt_step: int):
    # Print header with checkpoint information
    print(f"\nResults for checkpoint step {ckpt_step}:")
    print("\nPer-Task Metrics:")
    print("-" * 60)
    print(f"{'Task':<30} {'Success Rate':>12} {'FPS':>12}")
    print("-" * 60)

    # Display per-task metrics
    for task_str, metrics in sorted(task_averages.items()):
        print(f"{task_str:<30} {metrics.success_rate:>11.2f}% {metrics.fps:>11.2f}")

    # Print overall averages
    print("\n" + "-" * 60)
    print(f"Overall Averages:")
    print(f"Success Rate: {overall_metrics.success_rate:.2f}%")
    print(f"FPS: {overall_metrics.fps:.2f}")
    print("-" * 60)


def main(result_file, ckpt_step):
    if ckpt_step is None:
        for ckpt_step in [150000, 140000, 130000, 120000, 110000, 100000]:
            task_averages, overall_metrics = process_results(result_file, ckpt_step)
            display_results(task_averages, overall_metrics, ckpt_step)
    else:
        task_averages, overall_metrics = process_results(result_file, ckpt_step)
        display_results(task_averages, overall_metrics, ckpt_step)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('result_file', help='Path to the results file')
    parser.add_argument('--ckpt_step', type=int, help='Checkpoint step to filter results', default=None)
    args = parser.parse_args()
    main(args.result_file, args.ckpt_step)