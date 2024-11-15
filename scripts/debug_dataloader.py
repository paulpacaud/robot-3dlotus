import os
import json
import numpy as np
import torch
from functools import partial
from genrobo3d.train.datasets.motion_planner_dataset_peract_augment import MotionPlannerDatasetPerActAugment


def inspect_dataset():
    action_embed_file = '../data/peract/train_dataset/motion_keysteps_bbox_pcd/action_embeds_clip.npy'

    dataset = MotionPlannerDatasetPerActAugment(
        data_dir='../data/peract/train_dataset/motion_keysteps_bbox_pcd/seed0/voxel1cm',
        action_embed_file=action_embed_file,
        gt_act_obj_label_file='../assets/taskvars_target_label_zrange_peract.json',
        taskvar_file='../assets/taskvars_peract.json',
        num_points=4096,
        xyz_shift='center',
        xyz_norm=False,
        use_height=True,
        max_traj_len=5,
        pc_label_type='fine',
        pc_label_augment=0.0,
        pc_midstep_augment=True,
        rot_type='euler_disc',
        instr_embed_type='all',
        all_step_in_batch=True,
        rm_robot='box_keep_gripper',
        include_last_step=False,
        augment_pc=True,
        pos_type='disc',
        pos_bins=15,
        pos_bin_size=0.01,
        pos_heatmap_type='dist',
        pos_heatmap_no_robot=True,
        use_color=False,
        sampling_ratio=0.5,
        noise_range=0.15,
        safe_distance=0.04,
    )

    # Try loading samples one by one
    print("\nTrying to load individual samples:")
    for i in range(min(5, len(dataset))):
        sample = dataset[i]
        print(f"Successfully loaded sample {i}")

if __name__ == "__main__":
    inspect_dataset()