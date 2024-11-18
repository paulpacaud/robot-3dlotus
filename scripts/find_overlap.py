import json
import os
from typing import Dict, List, Set


def find_labeled_taskvars(taskvar_file: str, gt_label_file: str, microsteps_file: str) -> List[str]:
    """
    Find taskvars that are both in the taskvar file and have labeled episodes in gt_label_file.

    Args:
        taskvar_file: Path to the file containing the list of taskvars
        gt_label_file: Path to the file containing episode labels

    Returns:
        List of taskvars that have labeled episodes
    """
    # Load the task variations
    with open(taskvar_file, 'r') as f:
        taskvars = json.load(f)

    # Load the ground truth labels
    with open(gt_label_file, 'r') as f:
        gt_labels = json.load(f)

    labeled_taskvars = []

    for taskvar in taskvars:
        print(f"\nChecking {taskvar}")
        task_str, variation = taskvar.split('+')
        episodes_dir = os.path.join(microsteps_file, task_str, f"variation{variation}", "episodes")
        if not os.path.exists(episodes_dir):
            print(f"Episodes directory {episodes_dir} does not exist")
            continue
        episode_ids = os.listdir(episodes_dir)
        episode_ids.sort(key=lambda ep: int(ep[7:]))
        # keep the episodes whose folder is not empty
        present_episodes = []
        for ep in episode_ids:
            print(f"listing episodes in {episodes_dir}/{ep}: {os.listdir(os.path.join(episodes_dir, ep))}")
            if len(os.listdir(os.path.join(episodes_dir, ep))) > 0:
                present_episodes.append(ep)
        print(f"Checking {taskvar} with {len(present_episodes)} episodes")

        for present_episode in present_episodes:
            # checking if [taskvar][present_episode] exists in gt_labels
            if taskvar in gt_labels and present_episode in gt_labels[taskvar]:
                print(f"Found labeled episode {taskvar}_{present_episode}")
                labeled_taskvars.append(taskvar)
            else:
                print(f"Episode {taskvar}_{present_episode} is not labeled")

    return labeled_taskvars


def main():
    set, seed = 'test', 200
    taskvar_file = f'../assets/taskvars_{set}_peract.json'
    gt_label_file = '../assets/taskvars_target_label_zrange_peract.json'
    microsteps_file = f'../data/peract/{set}_dataset/microsteps/seed{seed}'

    labeled_taskvars = find_labeled_taskvars(taskvar_file, gt_label_file, microsteps_file)

    print(f"Found {len(labeled_taskvars)} labeled taskvars out of {len(json.load(open(taskvar_file)))} total taskvars")
    print("\nLabeled taskvars:")
    for taskvar in labeled_taskvars:
        print(taskvar)

    print("\nUnlabeled taskvars:")
    for taskvar in json.load(open(taskvar_file)):
        if taskvar not in labeled_taskvars:
            print(taskvar)


if __name__ == "__main__":
    main()