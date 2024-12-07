import os
import numpy as np
import json
import jsonlines
import collections
import argparse
from typing import Dict, List

SPLIT_NAMES = ['taskvars_train', 'taskvars_test_l2', 'taskvars_test_l3', 'taskvars_test_l4']
TABLE_WIDTH = 60


class ResultsAnalyzer:
    def __init__(self, result_dir: str, ckpt_step: int, seeds: List[int]):
        self.result_dir = result_dir
        self.ckpt_step = ckpt_step
        self.seeds = seeds
        self.results = collections.defaultdict(lambda: {'sr': [], 'fps': []})

    def load_results(self):
        """Load results from all seed directories"""
        for seed in self.seeds:
            print(f'Loading seed {seed}')
            result_file = os.path.join(self.result_dir, f'seed{seed}', 'results.jsonl')

            if not os.path.exists(result_file):
                print(f'{result_file} missing')
                continue

            with jsonlines.open(result_file, 'r') as f:
                for item in f:
                    if isinstance(item['checkpoint'], int):
                        res_ckpt = item['checkpoint']
                    else:
                        res_ckpt = int(os.path.basename(item['checkpoint']).split('_')[-1].split('.')[0])

                    if res_ckpt == self.ckpt_step:
                        taskvar = f"{item['task']}+{item['variation']}"
                        self.results[taskvar]['sr'].append(item['sr'])
                        self.results[taskvar]['fps'].append(item.get('fps', 0.0))

    def print_header(self, text: str):
        """Print formatted header"""
        print('-' * TABLE_WIDTH)
        print(f'Split type: {text}')
        print('-' * TABLE_WIDTH)
        print('-' * TABLE_WIDTH)
        print(f'{"Task":<30} {"Success Rate":>15} {"FPS":>10}')
        print('-' * TABLE_WIDTH)

    def print_footer(self, avg_sr: float, avg_fps: float):
        """Print formatted footer with averages"""
        print('-' * TABLE_WIDTH)
        print('Overall Averages:')
        print(f'Success Rate: {avg_sr:.2f}%')
        print(f'FPS: {avg_fps:.2f}')
        print('-' * TABLE_WIDTH)
        print()

    def analyze_split(self, split_name: str):
        """Analyze and display results for a specific split"""
        self.print_header(split_name)

        # Load task variations for this split
        taskvars = json.load(open(os.path.join('assets', f'{split_name}.json')))
        taskvars.sort()

        # Calculate metrics for each task
        sr_values = []
        fps_values = []

        for taskvar in taskvars:
            if taskvar in self.results:
                sr = np.mean(self.results[taskvar]['sr']) * 100
                fps = np.mean(self.results[taskvar]['fps'])

                sr_values.append(sr)
                fps_values.append(fps)

                # Extract just the task name from taskvar (remove variation)
                task_name = taskvar.split('+')[0]
                print(f'{task_name:<30} {sr:>14.2f}% {fps:>9.2f}')

        # Calculate overall averages
        avg_sr = np.mean(sr_values) if sr_values else 0
        avg_fps = np.mean(fps_values) if fps_values else 0

        self.print_footer(avg_sr, avg_fps)

    def analyze_seed_performance(self, split_name: str):
        """Analyze performance across seeds for a split"""
        taskvars = json.load(open(os.path.join('assets', f'{split_name}.json')))
        num_seeds = min([len(self.results[taskvar]['sr']) for taskvar in taskvars])

        seed_results = [
            100 * np.mean([self.results[taskvar]['sr'][i] for taskvar in taskvars])
            for i in range(min(len(self.seeds), num_seeds))
        ]

        print(f'Performance over {num_seeds} seeds:')
        print(f'Mean: {round(np.mean(seed_results),1)}%')
        print(f'Std: {round(np.std(seed_results),1)}%')
        print()


def main(args):
    analyzer = ResultsAnalyzer(args.result_dir, args.ckpt_step, args.seeds)
    analyzer.load_results()

    for split_name in SPLIT_NAMES:
        print(f'\nAnalyzing {split_name}')
        analyzer.analyze_split(split_name)
        analyzer.analyze_seed_performance(split_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('result_dir')
    parser.add_argument('ckpt_step', type=int)
    parser.add_argument('--seeds', type=int, nargs='+', default=[200, 300, 400, 500, 600])
    args = parser.parse_args()

    main(args)