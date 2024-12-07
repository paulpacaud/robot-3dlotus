from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any, List
import numpy as np
import torch
import open3d as o3d
from sklearn.neighbors import LocalOutlierFactor
from scipy.spatial.transform import Rotation as R
from scipy.special import softmax
import os
import json
import copy

from genrobo3d.configs.default import get_config
from genrobo3d.train.train_simple_policy_coarse import MODEL_FACTORY
from genrobo3d.configs.rlbench.constants import get_robot_workspace, get_rlbench_labels
from genrobo3d.utils.robot_box import RobotBox
from genrobo3d.train.datasets.common import gen_seq_masks


@dataclass
class WorkspaceConfig:
    """Robot workspace configuration."""

    X_BBOX: Tuple[float, float]
    Y_BBOX: Tuple[float, float]
    Z_BBOX: Tuple[float, float]
    TABLE_HEIGHT: float


class PointCloudProcessor:
    """Handles point cloud and robot mask processing operations."""

    def __init__(self, config: Any, workspace: WorkspaceConfig, real_robot: bool):
        """Initialize processor with config and workspace parameters."""
        self.config = config
        self.data_cfg = config.TRAIN_DATASET
        self.workspace = workspace
        self.real_robot = real_robot

    def get_mask_with_label_ids(
        self, sem: np.ndarray, label_ids: List[int]
    ) -> np.ndarray:
        """Generate mask based on semantic labels."""
        mask = sem == label_ids[0]
        for label_id in label_ids[1:]:
            mask = mask | (sem == label_id)
        return mask

    def get_mask_with_robot_box(
        self, xyz: np.ndarray, arm_links_info: Dict, rm_robot_type: str
    ) -> np.ndarray:
        """Generate mask for robot points using bounding box."""
        keep_gripper = rm_robot_type == "box_keep_gripper"
        robot_box = RobotBox(
            arm_links_info,
            keep_gripper=keep_gripper,
            env_name="real" if self.real_robot else "rlbench",
        )
        _, robot_point_ids = robot_box.get_pc_overlap_ratio(
            xyz=xyz, return_indices=True
        )
        robot_point_ids = np.array(list(robot_point_ids))

        mask = np.ones((xyz.shape[0],), dtype=bool)
        if len(robot_point_ids) > 0:
            mask[robot_point_ids] = False
        return mask

    def remove_outliers(
        self, xyz: np.ndarray, rgb: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Remove outlier points."""
        if self.real_robot:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            pcd.colors = o3d.utility.Vector3dVector(rgb)
            pcd, outlier_masks = pcd.remove_statistical_outlier(
                nb_neighbors=50, std_ratio=0.2
            )
            xyz = xyz[outlier_masks]
            if rgb is not None:
                rgb = rgb[outlier_masks]
        elif self.data_cfg.rm_pc_outliers:
            clf = LocalOutlierFactor(n_neighbors=self.data_cfg.rm_pc_outliers_neighbors)
            preds = clf.fit_predict(xyz)
            idxs = preds == 1
            xyz = xyz[idxs]
            if rgb is not None:
                rgb = rgb[idxs]

        return xyz, rgb

    def process_point_clouds(
        self,
        xyz: np.ndarray,
        rgb: np.ndarray,
        gt_sem: Optional[np.ndarray],
        ee_pose: np.ndarray,
        arm_links_info: Dict,
        taskvar: str,
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
        """Process point clouds for model input."""
        # Filter workspace points
        xyz, rgb, gt_sem = self._filter_workspace_points(xyz, rgb, gt_sem)

        # Downsample
        xyz, rgb, gt_sem = self._downsample_points(xyz, rgb, gt_sem)

        # Remove outliers
        xyz, rgb = self.remove_outliers(xyz, rgb)

        # Remove non-object points
        if not self.real_robot:
            xyz, rgb = self._remove_non_object_points(xyz, rgb, gt_sem, taskvar)

        if self.data_cfg.rm_robot.startswith("box"):
            mask = self.get_mask_with_robot_box(
                xyz, arm_links_info, self.data_cfg.rm_robot
            )
            xyz = xyz[mask]
            rgb = rgb[mask]

        # Sample points
        xyz, rgb = self._sample_points(xyz, rgb, ee_pose)

        # Calculate features
        height = xyz[:, -1] - self.workspace.TABLE_HEIGHT
        pc_ft, centroid, radius = self._normalize_points(xyz, rgb, height, ee_pose)

        return pc_ft, centroid, radius, ee_pose

    def _filter_workspace_points(
        self, xyz: np.ndarray, rgb: np.ndarray, gt_sem: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, ...]:
        """Filter points to keep only those within workspace."""
        xyz = xyz.reshape(-1, 3)
        in_mask = (
            (xyz[:, 0] > self.workspace.X_BBOX[0])
            & (xyz[:, 0] < self.workspace.X_BBOX[1])
            & (xyz[:, 1] > self.workspace.Y_BBOX[0])
            & (xyz[:, 1] < self.workspace.Y_BBOX[1])
            & (xyz[:, 2] > self.workspace.Z_BBOX[0])
            & (xyz[:, 2] < self.workspace.Z_BBOX[1])
        )

        if self.data_cfg.rm_table:
            in_mask &= xyz[:, 2] > self.workspace.TABLE_HEIGHT

        xyz = xyz[in_mask]
        rgb = rgb.reshape(-1, 3)[in_mask]
        if gt_sem is not None:
            gt_sem = gt_sem.reshape(-1)[in_mask]

        return xyz, rgb, gt_sem

    def _downsample_points(
        self, xyz: np.ndarray, rgb: np.ndarray, gt_sem: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, ...]:
        """Downsample points using voxel grid."""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd, _, trace = pcd.voxel_down_sample_and_trace(
            self.config.MODEL.action_config.voxel_size, np.min(xyz, 0), np.max(xyz, 0)
        )

        xyz = np.asarray(pcd.points)
        trace = np.array([v[0] for v in trace])
        rgb = rgb[trace]
        if gt_sem is not None:
            gt_sem = gt_sem[trace]

        return xyz, rgb, gt_sem

    def _remove_non_object_points(
        self, xyz: np.ndarray, rgb: np.ndarray, gt_sem: np.ndarray, taskvar: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Remove points not belonging to objects."""
        rm_label_ids = get_rlbench_labels(
            taskvar.split("+")[0],
            table=self.data_cfg.rm_table,
            robot=(self.data_cfg.rm_robot == "gt"),
            wall=False,
            floor=False,
        )
        if len(rm_label_ids) > 0:
            rm_mask = self.get_mask_with_label_ids(gt_sem, rm_label_ids)
            xyz = xyz[~rm_mask]
            rgb = rgb[~rm_mask]
        return xyz, rgb

    def _sample_points(
        self, xyz: np.ndarray, rgb: np.ndarray, ee_pose: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample points based on configuration."""
        if len(xyz) > self.data_cfg.num_points:
            if self.data_cfg.sample_points_by_distance:
                dists = np.sqrt(np.sum((xyz - ee_pose[:3]) ** 2, 1))
                probs = 1 / np.maximum(dists, 0.1)
                probs = np.maximum(softmax(probs), 1e-30)
                probs = probs / sum(probs)
                point_idxs = np.random.choice(
                    len(xyz), self.data_cfg.num_points, replace=False, p=probs
                )
            else:
                point_idxs = np.random.choice(
                    len(xyz), self.data_cfg.num_points, replace=False
                )
        else:
            if self.data_cfg.same_npoints_per_example:
                point_idxs = np.random.choice(
                    xyz.shape[0], self.data_cfg.num_points, replace=True
                )
            else:
                point_idxs = np.arange(xyz.shape[0])

        return xyz[point_idxs], rgb[point_idxs]

    def _normalize_points(
        self, xyz: np.ndarray, rgb: np.ndarray, height: np.ndarray, ee_pose: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Normalize point cloud features."""
        # Calculate centroid
        if self.data_cfg.xyz_shift == "none":
            centroid = np.zeros((3,))
        elif self.data_cfg.xyz_shift == "center":
            centroid = np.mean(xyz, 0)
        elif self.data_cfg.xyz_shift == "gripper":
            centroid = copy.deepcopy(ee_pose[:3])

        # Calculate radius
        if self.data_cfg.xyz_norm:
            radius = np.max(np.sqrt(np.sum((xyz - centroid) ** 2, axis=1)))
        else:
            radius = 1

        # Normalize
        xyz = (xyz - centroid) / radius
        height = height / radius
        ee_pose[:3] = (ee_pose[:3] - centroid) / radius
        rgb = (rgb / 255.0) * 2 - 1

        # Create feature vector
        pc_ft = np.concatenate([xyz, rgb], 1)
        if self.data_cfg.get("use_height", False):
            pc_ft = np.concatenate([pc_ft, height[:, None]], 1)

        return pc_ft, centroid, radius


class Actioner:
    """Main class for processing observations and generating actions."""

    def __init__(self, args: Any):
        self.args = args
        self.device = torch.device(args.device)

        # Create save directory if needed
        if self.args.save_obs_outs_dir is not None:
            os.makedirs(self.args.save_obs_outs_dir, exist_ok=True)

        # Initialize workspace
        ws_config = get_robot_workspace(real_robot=args.real_robot)
        self.workspace = WorkspaceConfig(
            X_BBOX=ws_config["X_BBOX"],
            Y_BBOX=ws_config["Y_BBOX"],
            Z_BBOX=ws_config["Z_BBOX"],
            TABLE_HEIGHT=ws_config["TABLE_HEIGHT"],
        )

        # Initialize model and config
        self.config = self._initialize_config()
        self.model = self._initialize_model()

        # Initialize processors
        self.pc_processor = PointCloudProcessor(
            self.config, self.workspace, args.real_robot
        )

        # Load instruction data
        self._load_instruction_data()

    def _initialize_config(self) -> Any:
        """Initialize configuration."""
        config = get_config(self.args.exp_config, self.args.remained_args)
        config.defrost()

        # Set dataset config defaults
        config.TRAIN_DATASET.sample_points_by_distance = config.TRAIN_DATASET.get(
            "sample_points_by_distance", False
        )
        config.TRAIN_DATASET.rm_pc_outliers = config.TRAIN_DATASET.get(
            "rm_pc_outliers", False
        )
        config.TRAIN_DATASET.rm_pc_outliers_neighbors = config.TRAIN_DATASET.get(
            "rm_pc_outliers_neighbors", 10
        )
        config.TRAIN_DATASET.same_npoints_per_example = config.TRAIN_DATASET.get(
            "same_npoints_per_example", False
        )

        # Set model config
        config.MODEL.action_config.best_disc_pos = self.args.best_disc_pos

        if self.args.checkpoint is not None:
            config.checkpoint = self.args.checkpoint

        if not self.args.enable_flashattn:
            config.MODEL.ptv3_config.enable_flash = self.args.enable_flashattn

        config.freeze()
        return config

    def _initialize_model(self) -> torch.nn.Module:
        """Initialize and load model."""
        model_class = MODEL_FACTORY[self.config.MODEL.model_class]
        model = model_class(self.config.MODEL)

        if self.config.checkpoint:
            checkpoint = torch.load(
                self.config.checkpoint, map_location=lambda storage, loc: storage
            )
            model.load_state_dict(checkpoint, strict=True)

        model.to(self.device)
        model.eval()
        return model

    def _load_instruction_data(self) -> None:
        """Load instruction embeddings and task variations."""
        self.instr_embeds = np.load(
            self.config.TRAIN_DATASET.instr_embed_file, allow_pickle=True
        ).item()

        if self.config.TRAIN_DATASET.instr_embed_type == "last":
            self.instr_embeds = {
                instr: embeds[-1:] for instr, embeds in self.instr_embeds.items()
            }

        self.taskvar_instrs = json.load(open(self.args.taskvars_instructions_file))

    def predict(
        self,
        task_str: str,
        variation: int,
        step_id: int,
        obs_state_dict: Dict,
        episode_id: Optional[int] = None,
        instructions: Optional[str] = None,
    ) -> Dict:
        """Generate action prediction from observation."""
        taskvar = f"{task_str}+{variation}"
        batch = self._preprocess_obs(taskvar, step_id, obs_state_dict)

        with torch.no_grad():
            actions = self._get_ensemble_predictions(batch)
            action = self._aggregate_predictions(actions)

        action = self._postprocess_action(action, batch)

        if self.args.save_obs_outs_dir is not None:
            self._save_outputs(episode_id, step_id, batch, obs_state_dict, action)

        return {"action": action}

    def _preprocess_obs(
        self, taskvar: str, step_id: int, obs_state_dict: Dict
    ) -> Dict[str, torch.Tensor]:
        """Preprocess observation into model input batch."""
        # Stack input arrays
        rgb = np.stack(obs_state_dict["rgb"], 0)  # (N, H, W, C)
        xyz = np.stack(obs_state_dict["pc"], 0)  # (N, H, W, C)
        gt_sem = (
            np.stack(obs_state_dict["gt_mask"], 0)  # (N, H, W)
            if "gt_mask" in obs_state_dict
            else None
        )

        # Get instruction embedding
        instr = self.taskvar_instrs[taskvar][0]
        instr_embed = self.instr_embeds[instr]

        # Process point cloud
        pc_ft, pc_centroid, pc_radius, ee_pose = self.pc_processor.process_point_clouds(
            xyz,
            rgb,
            gt_sem=gt_sem,
            ee_pose=copy.deepcopy(obs_state_dict["gripper"]),
            arm_links_info=obs_state_dict["arm_links_info"],
            taskvar=taskvar,
        )

        # Create batch dictionary
        batch = {
            "pc_fts": torch.from_numpy(pc_ft).float(),
            "pc_centroids": pc_centroid,
            "pc_radius": pc_radius,
            "ee_poses": torch.from_numpy(ee_pose).float().unsqueeze(0),
            "step_ids": torch.LongTensor([step_id]),
            "txt_embeds": torch.from_numpy(instr_embed).float(),
            "txt_lens": [instr_embed.shape[0]],
            "npoints_in_batch": [pc_ft.shape[0]],
            "offset": torch.LongTensor([pc_ft.shape[0]]),
        }

        # Add additional fields for PCT model
        if self.config.MODEL.model_class == "SimplePolicyPCT":
            batch["pc_fts"] = batch["pc_fts"].unsqueeze(0)
            batch["txt_masks"] = torch.from_numpy(
                gen_seq_masks(batch["txt_lens"])
            ).bool()
            batch["txt_embeds"] = batch["txt_embeds"].unsqueeze(0)

        return batch

    def _get_ensemble_predictions(
        self, batch: Dict[str, torch.Tensor]
    ) -> List[torch.Tensor]:
        """Get predictions from ensemble of models."""
        actions = []
        for _ in range(self.args.num_ensembles):
            action = self.model(batch)[0].data.cpu()
            actions.append(action)
        return actions

    def _aggregate_predictions(self, actions: List[torch.Tensor]) -> torch.Tensor:
        """Aggregate predictions from ensemble models."""
        if len(actions) > 1:
            # Average position and gripper state
            avg_action = torch.stack(actions, 0).mean(0)

            # Average rotation in euler space then convert back to quaternion
            euler_angles = np.mean(
                [R.from_quat(x[3:-1]).as_euler("xyz") for x in actions], 0
            )
            pred_rot = torch.from_numpy(R.from_euler("xyz", euler_angles).as_quat())

            action = torch.cat([avg_action[:3], pred_rot, avg_action[-1:]], 0)
        else:
            action = actions[0]

        # Convert gripper state to binary
        action[-1] = torch.sigmoid(action[-1]) > 0.5
        return action

    def _postprocess_action(
        self, action: torch.Tensor, batch: Dict[str, torch.Tensor]
    ) -> np.ndarray:
        """Post-process model output into final action."""
        action = action.numpy()

        # Denormalize position
        action[:3] = action[:3] * batch["pc_radius"] + batch["pc_centroids"]

        # Ensure minimum height above table
        action[2] = max(action[2], self.workspace.TABLE_HEIGHT + 0.005)

        return action

    def _save_outputs(
        self,
        episode_id: int,
        step_id: int,
        batch: Dict,
        obs_state_dict: Dict,
        action: np.ndarray,
    ) -> None:
        """Save outputs to file."""
        if episode_id is None:
            return

        obs_dir = os.path.join(self.args.save_obs_outs_dir, f"ep_{episode_id}", "steps")
        os.makedirs(obs_dir, exist_ok=True)

        save_path = os.path.join(obs_dir, f"step_{step_id}.npy")

        # Convert tensors to numpy arrays
        processed_batch = {
            k: v.data.cpu().numpy() if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        results = {
            "batch": processed_batch,
            "obs": obs_state_dict,
            "action": action,
        }

        np.save(
            save_path,
            results,
        )
