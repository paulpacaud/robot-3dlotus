import os
import numpy as np
import json
from tqdm import tqdm
import argparse
import copy
import lmdb
import msgpack
import msgpack_numpy

msgpack_numpy.patch()

import open3d as o3d

from genrobo3d.configs.rlbench.constants import get_robot_workspace
from genrobo3d.utils.point_cloud import voxelize_pcd

code_dir = os.path.join(os.environ.get('HOME'), 'Projects/robot-3dlotus')

input_dir = os.path.join(code_dir, 'data/gembench/train_dataset/keysteps_bbox/seed0')
output_dir = os.path.join(code_dir, 'data/gembench/train_dataset/keysteps_bbox_pcd/seed0/voxel0.5cm')
taskvar_file = os.path.join(code_dir, 'assets', 'taskvars_train.json')


def zoom_around_point(xyz, workspace, point_of_interest, zoom_factor):
    """
    Refines the point cloud to focus around the `point_of_interest`.

    Parameters:
    - xyz: numpy array, the point cloud coordinates.
    - workspace: dict, bounding box limits for the workspace.
    - point_of_interest: numpy array, central point for refinement.
    - zoom_factor: float, proportion of the workspace bounds to keep

    Returns:
    - mask: boolean numpy array, indicating points within the refined area.
    """
    workspace_width_x = workspace['X_BBOX'][1] - workspace['X_BBOX'][0]
    workspace_width_y = workspace['Y_BBOX'][1] - workspace['Y_BBOX'][0]
    workspace_width_z = workspace['Z_BBOX'][1] - workspace['Z_BBOX'][0]
    x_min, x_max = point_of_interest[0] - workspace_width_x * zoom_factor / 2, \
                   point_of_interest[0] + workspace_width_x * zoom_factor / 2
    y_min, y_max = point_of_interest[1] - workspace_width_y * zoom_factor / 2, \
                   point_of_interest[1] + workspace_width_y * zoom_factor / 2
    z_min, z_max = point_of_interest[2] - workspace_width_z * zoom_factor / 2, \
                   point_of_interest[2] + workspace_width_z * zoom_factor / 2

    mask = (xyz[:, 0] > x_min) & (xyz[:, 0] < x_max) & \
           (xyz[:, 1] > y_min) & (xyz[:, 1] < y_max) & \
           (xyz[:, 2] > z_min) & (xyz[:, 2] < z_max)

    return mask

def noise_position(position, noise_range):
    """
    Apply a random noise to the ground truth action (point of interest).

    Parameters:
    - position: numpy array or list with shape (3,), representing the x, y, z coordinates.
    - noise_range: float, maximum amount of noise to apply to each axis, in meters.

    Returns:
    - Noisy position as a numpy array with shape (3,).
    """
    noise = np.random.uniform(-noise_range, noise_range, size=3)

    noisy_position = position + noise
    return noisy_position

def main():
    print("starting to generate simple policy data")
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default=input_dir)
    parser.add_argument('--output_dir', default=output_dir)
    parser.add_argument('--taskvar_file', default=taskvar_file)
    parser.add_argument('--voxel_size', type=float, default=0.005, help='meters')
    parser.add_argument('--real_robot', default=False, action='store_true')
    parser.add_argument('--num_cameras', default=None, type=int, help='use all by default')
    parser.add_argument('--noise_range', default=0.02, type=float)
    parser.add_argument('--zoom_factor', default=0.4, type=float)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.taskvar_file is not None:
        taskvars = json.load(open(args.taskvar_file))
    else:
        taskvars = os.listdir(args.input_dir)

    workspace = get_robot_workspace(real_robot=args.real_robot)
    taskvar_idx = 0
    for taskvar in tqdm(taskvars):
        taskvar_idx += 1
        print(f"task name: {taskvar}")
        print(f"taskvar {taskvar_idx}/{len(taskvars)}")
        input_lmdb_dir = os.path.join(args.input_dir, taskvar)
        if not os.path.exists(input_lmdb_dir):
            print(taskvar, 'not exists')
            continue
        if os.path.exists(os.path.join(args.output_dir, taskvar)):
            continue

        out_lmdb_env = lmdb.open(os.path.join(args.output_dir, taskvar), map_size=int(1024 ** 3))
        with lmdb.open(input_lmdb_dir, readonly=True) as lmdb_env:
            with lmdb_env.begin() as txn:
                key_idx = 0
                for key, value in txn.cursor():
                    value = msgpack.unpackb(value)

                    key_idx += 1

                    rgb = value['rgb'][:, :args.num_cameras]  # (T, N_cam, H, W, 3)
                    pc = value['pc'][:, :args.num_cameras]  # (T, N_cam, H, W, 3)
                    if 'mask' in value:
                        sem = value['mask'][:, :args.num_cameras]  # (T, N_cam, H, W)
                    elif 'gt_masks' in value:
                        sem = value['gt_masks'][:, :args.num_cameras]  # (T, N_cam, H, W)
                    else:
                        sem = None

                    outs = {
                        'xyz': [], 'rgb': [], 'sem': [], 'point_of_interest': []
                    }
                    for value_key in ['bbox_info', 'pose_info', 'key_frameids', 'action']:
                        if value_key in value:
                            outs[value_key] = value[value_key]

                    num_steps = rgb.shape[0]
                    for t in range(num_steps):
                        print(f"step_id {t}/{num_steps} for key {key} - key_id: {key_idx}/100 for taskvar {taskvar} - taskvar_id: {taskvar_idx}/{len(taskvars)}")

                        t_pc = pc[t].reshape(-1, 3)

                        in_mask = (t_pc[:, 0] > workspace['X_BBOX'][0]) & (t_pc[:, 0] < workspace['X_BBOX'][1]) & \
                                  (t_pc[:, 1] > workspace['Y_BBOX'][0]) & (t_pc[:, 1] < workspace['Y_BBOX'][1]) & \
                                  (t_pc[:, 2] > workspace['Z_BBOX'][0]) & (t_pc[:, 2] < workspace['Z_BBOX'][1])

                        if t < num_steps - 1:
                            gt_action = copy.deepcopy(outs['action'][t + 1])
                        else:
                            gt_action = copy.deepcopy(outs['action'][-1])

                        point_of_interest = noise_position(gt_action[:3], args.noise_range)
                        mask_area_of_interest = zoom_around_point(t_pc, workspace, point_of_interest, args.zoom_factor)
                        in_mask = in_mask & mask_area_of_interest

                        if args.real_robot:
                            in_mask = in_mask & (t_pc[:, 2] > workspace['TABLE_HEIGHT'])

                        t_pc = t_pc[in_mask]
                        t_rgb = rgb[t].reshape(-1, 3)[in_mask]
                        if sem is not None:
                            t_sem = sem[t].reshape(-1)[in_mask]

                        t_pc, mask = voxelize_pcd(t_pc, voxel_size=args.voxel_size)
                        t_rgb = t_rgb[mask]
                        if sem is not None:
                            t_sem = t_sem[mask]

                        if args.real_robot:
                            # remove point cloud outliers for noisy real point clouds
                            pcd = o3d.geometry.PointCloud()
                            pcd.points = o3d.utility.Vector3dVector(t_pc)
                            pcd.colors = o3d.utility.Vector3dVector(t_rgb)
                            pcd, outlier_masks = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=0.2)
                            t_pc = t_pc[outlier_masks]
                            t_rgb = t_rgb[outlier_masks]
                            if sem is not None:
                                t_sem = t_sem[outlier_masks]

                        outs['xyz'].append(t_pc)
                        outs['rgb'].append(t_rgb)
                        outs['point_of_interest'].append(point_of_interest)
                        if sem is not None:
                            outs['sem'].append(t_sem)

                    if len(outs['sem']) == 0:
                        del outs['sem']

                    out_txn = out_lmdb_env.begin(write=True)
                    out_txn.put(key, msgpack.packb(outs))
                    out_txn.commit()

        out_lmdb_env.close()
        lmdb_env.close()


if __name__ == '__main__':
    main()