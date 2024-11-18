import numpy as np
import random
import torch
import copy

def pad_tensors(tensors, lens=None, pad=0, max_len=None):
    """B x [T, ...] torch tensors"""
    if lens is None:
        lens = [t.size(0) for t in tensors]
    max_len = max(lens) if max_len is None else max_len
    bs = len(tensors)
    hid = list(tensors[0].size()[1:])
    size = [bs, max_len] + hid

    dtype = tensors[0].dtype
    output = torch.zeros(*size, dtype=dtype)
    if pad:
        output.data.fill_(pad)
    for i, (t, l) in enumerate(zip(tensors, lens)):
        output.data[i, :l, ...] = t.data
    return output

def gen_seq_masks(seq_lens, max_len=None):
    """
    Args:
        seq_lens: list or nparray int, shape=(N, )
    Returns:
        masks: nparray, shape=(N, L), padded=0
    """
    seq_lens = np.array(seq_lens)
    if max_len is None:
        max_len = max(seq_lens)
    if max_len == 0:
        return np.zeros((len(seq_lens), 0), dtype=bool)
    batch_size = len(seq_lens)
    masks = np.arange(max_len).reshape(-1, max_len).repeat(batch_size, 0)
    masks = masks < seq_lens.reshape(-1, 1)
    return masks


def normalize_pc(pc, centroid=None, return_params=False):
    # Normalize the point cloud to [-1, 1]
    if centroid is None:
        centroid = np.mean(pc, axis=0)
    else:
        centroid = copy.deepcopy(centroid)
    
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    if m < 1e-6:
        pc = np.zeros_like(pc)
    else:
        pc = pc / m
    if return_params:
        return pc, (centroid, m)
    return pc

def random_scale_pc(pc, scale_low=0.8, scale_high=1.25):
    # Randomly scale the point cloud.
    scale = np.random.uniform(scale_low, scale_high)
    pc = pc * scale
    return pc

def shift_pc(pc, shift_range=0.1):
    # Randomly shift point cloud.
    shift = np.random.uniform(-shift_range, shift_range, size=[3])
    pc = pc + shift
    return pc

def rotate_perturbation_pc(pc, angle_sigma=0.06, angle_clip=0.18):
    # Randomly perturb the point cloud by small rotations (unit: radius)
    angles = np.clip(angle_sigma * np.random.randn(3), -angle_clip, angle_clip)
    cosval, sinval = np.cos(angles), np.sin(angles)
    Rx = np.array([[1, 0, 0], [0, cosval[0], -sinval[0]], [0, sinval[0], cosval[0]]])
    Ry = np.array([[cosval[1], 0, sinval[1]], [0, 1, 0], [-sinval[1], 0, cosval[1]]])
    Rz = np.array([[cosval[2], -sinval[2], 0], [sinval[2], cosval[2], 0], [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))
    pc = np.dot(pc, np.transpose(R))
    return pc

def random_rotate_z(pc, angle=None):
    # Randomly rotate around z-axis
    if angle is None:
        angle = np.random.uniform() * 2 * np.pi
    cosval, sinval = np.cos(angle), np.sin(angle)
    R = np.array([[cosval, -sinval, 0], [sinval, cosval, 0], [0, 0, 1]])
    return np.dot(pc, np.transpose(R))

def random_rotate_xyz(pc):
    # Randomly rotate around x, y, z axis
    angles = np.random.uniform(size=[3]) * 2 * np.pi
    cosval, sinval = np.cos(angles), np.sin(angles)
    Rx = np.array([[1, 0, 0], [0, cosval[0], -sinval[0]], [0, sinval[0], cosval[0]]])
    Ry = np.array([[cosval[1], 0, sinval[1]], [0, 1, 0], [-sinval[1], 0, cosval[1]]])
    Rz = np.array([[cosval[2], -sinval[2], 0], [sinval[2], cosval[2], 0], [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))
    pc = np.dot(pc, np.transpose(R))
    return pc

def random_translate(xyz, ee_pose):
    gripper_pos = ee_pose[:3]

def augment_pc(pc):
    pc = random_scale_pc(pc)
    pc = shift_pc(pc)
    # pc = rotate_perturbation_pc(pc)
    pc = random_rotate_z(pc)
    return pc


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


class GripperAugmenter:
    def __init__(
            self,
            sampling_ratio=0.5,
            noise_range=0.15,  # 15cm
            safe_distance=0.04,  # 4cm
            restart_pose=False,
            workspace_dims=None
    ):
        self.sampling_ratio = sampling_ratio
        self.noise_range = noise_range
        self.safe_distance = safe_distance
        self.restart_pose = restart_pose
        self.workspace_dims = workspace_dims

    def is_point_far_from_objects(self, candidate_point, workspace_points):
        """Check if a candidate point is far enough from workspace objects."""
        if len(workspace_points) == 0:
            return True
        distances = np.linalg.norm(workspace_points - candidate_point, axis=1)
        return np.all(distances > self.safe_distance)

    def is_point_in_workspace(self, candidate_point):
        """Check if a candidate point is within the workspace."""
        x_min, x_max = self.workspace_dims['X_BBOX']
        y_min, y_max = self.workspace_dims['Y_BBOX']
        z_min, z_max = self.workspace_dims['Z_BBOX']
        x, y, z = candidate_point
        return x_min <= x <= x_max and y_min <= y <= y_max and z_min <= z <= z_max

    def find_new_noisy_point(self, original_pos, workspace_points, max_attempts=100):
        """Find a new noisy point within noise range of original position."""
        for _ in range(max_attempts):
            noise = np.random.uniform(-self.noise_range, self.noise_range, size=3)
            candidate_point = original_pos + noise
            if not self.is_point_in_workspace(candidate_point):
                continue
            if self.is_point_far_from_objects(candidate_point, workspace_points):
                return candidate_point

        return original_pos

    def __call__(self, xyz, pc_label, ee_pose, gt_trajs):
        """
        Augment gripper position if it is far from workspace points, and if sampling probability is met.

        Args:
            xyz: Point cloud coordinates (N, 3)
            pc_label: Point cloud labels (N,) with 1=robot, 2=object, 3=target
            ee_pose: Current end effector pose

        Returns:
            Updated xyz, pc_label, ee_pose, gt_trajs
        """
        # Only augment with probability defined by sampling ratio
        if random.random() > self.sampling_ratio:
            return xyz, pc_label, ee_pose, gt_trajs

        robot_mask = pc_label == 1
        workspace_mask = (pc_label == 2) | (pc_label == 3)
        robot_points = xyz[robot_mask]
        workspace_points = xyz[workspace_mask]

        if len(robot_points) == 0:
            return xyz, pc_label, ee_pose, gt_trajs

        current_gripper_pos = ee_pose[:3]

        if not self.is_point_far_from_objects(current_gripper_pos, workspace_points):
            # Gripper is close to workspace points -> probably manipulating, we skip augmentation
            return xyz, pc_label, ee_pose, gt_trajs
        if ee_pose[7] < 0.5:
            # Gripper is closed -> probably manipulating, we skip augmentation
            return xyz, pc_label, ee_pose, gt_trajs

        restart_action = copy.deepcopy(ee_pose)
        new_pos = self.find_new_noisy_point(current_gripper_pos, workspace_points)

        translation = new_pos - current_gripper_pos

        # Update robot points and end effector pose
        xyz[robot_mask] = robot_points + translation
        ee_pose[:3] = ee_pose[:3] + translation

        if self.restart_pose:
            # Add restart action at beginning of trajectory and remove last action to keep same trajectory length
            gt_trajs = np.vstack([restart_action[None], gt_trajs])
            gt_trajs = gt_trajs[:-1]

        return xyz, pc_label, ee_pose, gt_trajs