import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import numpy as np
import json
from tqdm import tqdm
import argparse

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
taskvar_file = os.path.join(code_dir, 'assets', 'taskvars_debug.json')

def zoom_around_point(xyz, workspace, point_of_interest, scale_factor):
    """
    Refines the point cloud to focus around the `point_of_interest`.

    Parameters:
    - xyz: numpy array, the point cloud coordinates.
    - workspace: dict, bounding box limits for the workspace.
    - point_of_interest: numpy array, central point for refinement.
    - scale_factor: float, proportion of the workspace bounds to keep (e.g., 0.25 to keep 1/4).

    Returns:
    - mask: boolean numpy array, indicating points within the refined area.
    """
    workspace_width_x = workspace['X_BBOX'][1] - workspace['X_BBOX'][0]
    workspace_width_y = workspace['Y_BBOX'][1] - workspace['Y_BBOX'][0]
    workspace_width_z = workspace['Z_BBOX'][1] - workspace['Z_BBOX'][0]
    x_min, x_max = point_of_interest[0] - workspace_width_x * scale_factor / 2, \
                   point_of_interest[0] + workspace_width_x * scale_factor / 2
    y_min, y_max = point_of_interest[1] - workspace_width_y * scale_factor / 2, \
                   point_of_interest[1] + workspace_width_y * scale_factor / 2
    z_min, z_max = point_of_interest[2] - workspace_width_z * scale_factor / 2, \
                   point_of_interest[2] + workspace_width_z * scale_factor / 2

    mask = (xyz[:, 0] > x_min) & (xyz[:, 0] < x_max) & \
           (xyz[:, 1] > y_min) & (xyz[:, 1] < y_max) & \
           (xyz[:, 2] > z_min) & (xyz[:, 2] < z_max)

    return mask

def visualize_point_cloud(points, title, rgb=None, point_of_interest=None, box_bounds=None):
    """
       Visualize point cloud with optional RGB colors and bounding box
       """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot points
    if rgb is not None:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=rgb / 255.0, s=1)
    else:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='blue', alpha=0.5, s=1)

    # Plot point of interest if provided
    if point_of_interest is not None:
        ax.scatter(point_of_interest[0], point_of_interest[1], point_of_interest[2],
                   color='red', s=500, marker='*', label='Point of Interest')

    # Plot bounding box if provided
    if box_bounds is not None:
        x_min, x_max = box_bounds['X_BBOX']
        y_min, y_max = box_bounds['Y_BBOX']
        z_min, z_max = box_bounds['Z_BBOX']

        # Create bounding box edges
        for z in [z_min, z_max]:
            ax.plot([x_min, x_max], [y_min, y_min], [z, z], 'r--', alpha=0.5)
            ax.plot([x_min, x_max], [y_max, y_max], [z, z], 'r--', alpha=0.5)
            ax.plot([x_min, x_min], [y_min, y_max], [z, z], 'r--', alpha=0.5)
            ax.plot([x_max, x_max], [y_min, y_max], [z, z], 'r--', alpha=0.5)

        # Connect z levels
        for x, y in [(x_min, y_min), (x_max, y_min), (x_min, y_max), (x_max, y_max)]:
            ax.plot([x, x], [y, y], [z_min, z_max], 'r--', alpha=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    # Add point count to title
    plt.figtext(0.02, 0.02, f'Number of points: {len(points)}')


def visualize_open3d_point_cloud(points, rgb=None, point_of_interest=None, box_bounds=None):
    """
    Visualize point cloud using Open3D with optional RGB colors, point of interest, and bounding box.

    Parameters:
    - points: numpy array of shape (N, 3), the point cloud coordinates
    - rgb: numpy array of shape (N, 3), RGB colors for each point (values in [0, 255])
    - point_of_interest: numpy array of shape (3,), coordinates to highlight
    - box_bounds: dict with keys 'X_BBOX', 'Y_BBOX', 'Z_BBOX', each containing [min, max]
    """
    # Create point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if rgb is not None:
        # Convert RGB to float [0, 1]
        colors = rgb.astype(float) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)

    # Create visualization list
    vis_objects = [pcd]

    # Add point of interest if provided
    if point_of_interest is not None:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
        sphere.translate(point_of_interest)
        sphere.paint_uniform_color([1, 0, 0])  # Red color
        vis_objects.append(sphere)

    # Add bounding box if provided
    if box_bounds is not None:
        min_bound = np.array([
            box_bounds['X_BBOX'][0],
            box_bounds['Y_BBOX'][0],
            box_bounds['Z_BBOX'][0]
        ])
        max_bound = np.array([
            box_bounds['X_BBOX'][1],
            box_bounds['Y_BBOX'][1],
            box_bounds['Z_BBOX'][1]
        ])

        # Create line set for bounding box
        box_points = [
            min_bound,
            [max_bound[0], min_bound[1], min_bound[2]],
            [max_bound[0], max_bound[1], min_bound[2]],
            [min_bound[0], max_bound[1], min_bound[2]],
            [min_bound[0], min_bound[1], max_bound[2]],
            [max_bound[0], min_bound[1], max_bound[2]],
            max_bound,
            [min_bound[0], max_bound[1], max_bound[2]]
        ]

        lines = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
            [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
            [0, 4], [1, 5], [2, 6], [3, 7]  # Connecting edges
        ]

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(box_points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in range(len(lines))])  # Red lines
        vis_objects.append(line_set)

    # Create visualizer
    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window()

    # Add all objects to visualizer
    for obj in vis_objects:
        visualizer.add_geometry(obj)

    # Set default camera view
    view_control = visualizer.get_view_control()
    view_control.set_zoom(0.8)
    view_control.set_front([0, 0, -1])
    view_control.set_up([0, -1, 0])

    # Add coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    visualizer.add_geometry(coord_frame)

    # Run visualizer
    visualizer.run()
    visualizer.destroy_window()

def main():
       print("starting to generate simple policy data")
       parser = argparse.ArgumentParser()
       parser.add_argument('--input_dir', default=input_dir)
       parser.add_argument('--output_dir', default=output_dir)
       parser.add_argument('--taskvar_file', default=taskvar_file)
       parser.add_argument('--voxel_size', type=float, default=0.005, help='meters')
       parser.add_argument('--real_robot', default=False, action='store_true')
       parser.add_argument('--num_cameras', default=None, type=int, help='use all by default')
       args = parser.parse_args()

       os.makedirs(args.output_dir, exist_ok=True)

       if args.taskvar_file is not None:
              taskvars = json.load(open(args.taskvar_file))
       else:
              taskvars = os.listdir(args.input_dir)

       workspace = get_robot_workspace(real_robot=args.real_robot)

       taskvar_idx = 0

       workspace = get_robot_workspace(real_robot=args.real_robot)

       for taskvar in tqdm(taskvars):
        input_lmdb_dir = os.path.join(args.input_dir, taskvar)
        if not os.path.exists(input_lmdb_dir):
            continue

        with lmdb.open(input_lmdb_dir, readonly=True) as lmdb_env:
            with lmdb_env.begin() as txn:
                key_idx = 1
                for key, value in txn.cursor():
                    if key != b'episode0':
                        print(f"Skipping key {key}")
                        continue
                    key_idx += 1
                    print(f"key {key}")
                    value = msgpack.unpackb(value)

                    rgb = value['rgb'][:, :args.num_cameras]
                    pc = value['pc'][:, :args.num_cameras]
                    if 'mask' in value:
                        sem = value['mask'][:, :args.num_cameras]  # (T, N_cam, H, W)
                    elif 'gt_masks' in value:
                        sem = value['gt_masks'][:, :args.num_cameras]  # (T, N_cam, H, W)
                    else:
                        sem = None

                    num_steps = rgb.shape[0]
                    # For visualization, let's look at the first timestep
                    for t in range(num_steps):
                        if t != 2:
                            print(f"Skipping step {t}")
                            continue
                        print(f"step {t}/{num_steps} for key {key_idx}/100 taskvar {taskvar_idx}/{len(taskvars)}")
                        if t < num_steps - 1:
                            point_of_interest = value['action'][t + 1][:3]
                        else:
                            point_of_interest = None

                        # Original point cloud
                        t_pc = pc[t].reshape(-1, 3)
                        t_rgb = rgb[t].reshape(-1, 3)
                        # visualize_point_cloud(t_pc, 'Original Point Cloud',
                        #                       rgb=t_rgb,
                        #                       point_of_interest=point_of_interest,
                        #                       box_bounds=workspace)

                        # After workspace mask
                        in_mask = (t_pc[:, 0] > workspace['X_BBOX'][0]) & (t_pc[:, 0] < workspace['X_BBOX'][1]) & \
                                  (t_pc[:, 1] > workspace['Y_BBOX'][0]) & (t_pc[:, 1] < workspace['Y_BBOX'][1]) & \
                                  (t_pc[:, 2] > workspace['Z_BBOX'][0]) & (t_pc[:, 2] < workspace['Z_BBOX'][1])

                        t_pc_workspace = t_pc[in_mask]
                        t_rgb_workspace = t_rgb[in_mask]
                        visualize_point_cloud(t_pc_workspace, f'After Workspace Mask {taskvar}',
                                              rgb=t_rgb_workspace,
                                              point_of_interest=point_of_interest,
                                              box_bounds=workspace)

                        # After area of interest mask (if applicable)
                        if point_of_interest is not None:
                            mask_area_of_interest = zoom_around_point(t_pc, workspace, point_of_interest, scale_factor=1)
                            t_pc_zoomed = t_pc[mask_area_of_interest]
                            t_rgb_zoomed = t_rgb[mask_area_of_interest]
                            visualize_point_cloud(t_pc_zoomed, f'After Zoom Around Point of Interest 1 {taskvar}',
                                                  rgb=t_rgb_zoomed,
                                                  point_of_interest=point_of_interest,
                                                  box_bounds=workspace)
                            # visualize_open3d_point_cloud(t_pc_zoomed,
                            #                       rgb=t_rgb_zoomed,
                            #                       point_of_interest=point_of_interest,
                            #                       box_bounds=workspace)

                            mask_area_of_interest = zoom_around_point(t_pc, workspace, point_of_interest,
                                                                      scale_factor=0.5)
                            t_pc_zoomed = t_pc[mask_area_of_interest]
                            t_rgb_zoomed = t_rgb[mask_area_of_interest]
                            visualize_point_cloud(t_pc_zoomed, f'After Zoom Around Point of Interest 0.5 {taskvar}',
                                                  rgb=t_rgb_zoomed,
                                                  point_of_interest=point_of_interest,
                                                  box_bounds=workspace)
                            # visualize_open3d_point_cloud(t_pc_zoomed,
                            #                              rgb=t_rgb_zoomed,
                            #                              point_of_interest=point_of_interest,
                            #                              box_bounds=workspace)

                            mask_area_of_interest = zoom_around_point(t_pc, workspace, point_of_interest,
                                                                      scale_factor=0.4)
                            t_pc_zoomed = t_pc[mask_area_of_interest]
                            t_rgb_zoomed = t_rgb[mask_area_of_interest]
                            visualize_point_cloud(t_pc_zoomed, f'After Zoom Around Point of Interest 0.4 {taskvar}',
                                                  rgb=t_rgb_zoomed,
                                                  point_of_interest=point_of_interest,
                                                  box_bounds=workspace)
                            # visualize_open3d_point_cloud(t_pc_zoomed,
                            #                              rgb=t_rgb_zoomed,
                            #                              point_of_interest=point_of_interest,
                            #                              box_bounds=workspace)


                            in_mask = in_mask & mask_area_of_interest

                        t_pc = t_pc[in_mask]
                        t_rgb = rgb[t].reshape(-1, 3)[in_mask]
                        if sem is not None:
                            t_sem = sem[t].reshape(-1)[in_mask]
                        plt.show()
                        t_pc_voxel, mask = voxelize_pcd(t_pc, voxel_size=args.voxel_size)
                        t_rgb_voxel = t_rgb[mask]

                        # After voxelization
                        # visualize_point_cloud(t_pc_voxel, 'After Voxelization',
                        #                       rgb=t_rgb_voxel,
                        #                       point_of_interest=point_of_interest,
                        #                       box_bounds=workspace)

                        # Example usage in your main function:
                        # visualize_open3d_point_cloud(
                        #     t_pc,
                        #     rgb=t_rgb,
                        #     point_of_interest=point_of_interest,
                        #     box_bounds=workspace
                        # )

                        # Break after first sample for visualization


if __name__ == '__main__':
    main()
    plt.show()