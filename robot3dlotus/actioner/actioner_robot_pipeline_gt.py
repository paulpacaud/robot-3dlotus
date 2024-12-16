"""
Actioner Robot Pipeline Ground Truth
"""

import os
import random
import json
import re
import numpy as np
from easydict import EasyDict
import copy
from typing import Dict, List
import torch
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import json
import copy
import numpy as np
import torch

from genrobo3d.configs.rlbench.constants import get_robot_workspace
from genrobo3d.utils.point_cloud import voxelize_pcd, get_pc_foreground_mask
from genrobo3d.utils.robot_box import RobotBox

from genrobo3d.vlm_models.clip_encoder import ClipEncoder
from genrobo3d.models.motion_planner_ptv3 import (
    MotionPlannerPTV3AdaNorm,
    MotionPlannerPTV3CA,
)
from genrobo3d.configs.default import get_config as get_model_config
from genrobo3d.evaluation.common import load_checkpoint, parse_code
from genrobo3d.train.utils.logger import LOGGER


class GroundtruthTaskPlanner(object):
    def __init__(self, gt_plan_file):
        self.taskvar_plans = {}
        with open(gt_plan_file) as f:
            for x in f:
                x = x.strip()
                if len(x) == 0:
                    continue
                if x.startswith("# taskvar: "):
                    taskvar = x.split("# taskvar: ")[-1]
                    self.taskvar_plans[taskvar] = []
                elif not x.startswith("#"):
                    self.taskvar_plans[taskvar].append(x)

    def __call__(self, taskvar):
        plans = self.taskvar_plans[taskvar]
        return plans

    def estimate_height_range(self, target_name, obj_height):
        if "middle bottom" in target_name:
            zrange = [obj_height / 4 * 1, obj_height / 4 * 2]
        elif "middle top" in target_name:
            zrange = [obj_height / 4 * 2, obj_height / 4 * 3]
        elif "bottom" in target_name:
            zrange = [0, obj_height / 3]
        elif "middle" in target_name:
            zrange = [obj_height / 3, obj_height / 3 * 2]
        elif "top" in target_name:
            zrange = [obj_height / 3 * 2, obj_height]
        else:
            zrange = [0, obj_height]
        return np.array(zrange)


class GroundtruthVision(object):
    def __init__(
        self,
        gt_label_file,
        num_points=4096,
        voxel_size=0.01,
        same_npoints_per_example=False,
        rm_robot="box_keep_gripper",
        xyz_shift="center",
        xyz_norm=False,
        use_height=True,
        pc_label_type="coarse",
        use_color=False,
    ):
        self.taskvar_gt_target_labels = json.load(open(gt_label_file))
        self.workspace = get_robot_workspace(real_robot=False)
        self.TABLE_HEIGHT = self.workspace["TABLE_HEIGHT"]

        self.num_points = num_points
        self.voxel_size = voxel_size
        self.pc_label_type = pc_label_type
        self.same_npoints_per_example = same_npoints_per_example
        self.rm_robot = rm_robot
        self.xyz_shift = xyz_shift
        self.xyz_norm = xyz_norm
        self.use_height = use_height
        self.use_color = use_color

    def get_target_labels(self, taskvar, step_id, episode):
        """
        Flexibly access target labels handling both direct step_id indexing and episode[step_id] cases.

        Args:
            taskvar: Task variable key
            step_id: Step ID to access
            episode: Optional episode number

        Returns:
            Target labels dictionary for the specified step
        """
        try:
            # First try direct step_id indexing
            return self.taskvar_gt_target_labels[taskvar][step_id]
        except KeyError:
            return self.taskvar_gt_target_labels[taskvar][episode][step_id]

    def __call__(
        self,
        taskvar,
        step_id,
        pcd_images,
        sem_images,
        gripper_pose,
        arm_links_info,
        rgb_images=None,
        episode_id=None,
    ):
        episode = f"episode{episode_id}"
        task, variation = taskvar.split("+")
        pcd_xyz = pcd_images.reshape(-1, 3)
        pcd_sem = sem_images.reshape(-1)
        if self.use_color:
            assert rgb_images is not None
            pcd_rgb = rgb_images.reshape(-1, 3)

        # remove background and table points
        fg_mask = get_pc_foreground_mask(pcd_xyz, self.workspace)
        pcd_xyz = pcd_xyz[fg_mask]
        pcd_sem = pcd_sem[fg_mask]
        if self.use_color:
            pcd_rgb = pcd_rgb[fg_mask]

        pcd_xyz, idxs = voxelize_pcd(pcd_xyz, voxel_size=self.voxel_size)
        pcd_sem = pcd_sem[idxs]
        if self.use_color:
            pcd_rgb = pcd_rgb[idxs]

        if self.rm_robot != "none":
            if self.rm_robot == "box":
                robot_box = RobotBox(arm_links_info, keep_gripper=False)
            elif self.rm_robot == "box_keep_gripper":
                robot_box = RobotBox(arm_links_info, keep_gripper=True)
            robot_point_idxs = robot_box.get_pc_overlap_ratio(
                xyz=pcd_xyz, return_indices=True
            )[1]
            robot_point_idxs = np.array(list(robot_point_idxs))
            if len(robot_point_idxs) > 0:
                mask = np.ones((pcd_xyz.shape[0],), dtype=bool)
                mask[robot_point_idxs] = False
                pcd_xyz = pcd_xyz[mask]
                pcd_sem = pcd_sem[mask]
                if self.use_color:
                    pcd_rgb = pcd_rgb[mask]

        # sample points
        if len(pcd_xyz) > self.num_points:
            point_idxs = np.random.permutation(len(pcd_xyz))[: self.num_points]
        else:
            if self.same_npoints_per_example:
                point_idxs = np.random.choice(
                    pcd_xyz.shape[0], self.num_points, replace=True
                )
            else:
                point_idxs = np.arange(pcd_xyz.shape[0])
        pcd_xyz = pcd_xyz[point_idxs]
        pcd_sem = pcd_sem[point_idxs]
        height = pcd_xyz[..., 2] - self.TABLE_HEIGHT
        if self.use_color:
            pcd_rgb = pcd_rgb[point_idxs]

        # robot pcd_label
        pcd_label = np.zeros_like(pcd_sem)
        robot_box = RobotBox(arm_links_info, keep_gripper=False)
        robot_point_idxs = robot_box.get_pc_overlap_ratio(
            xyz=pcd_xyz, return_indices=True
        )[1]
        robot_point_idxs = np.array(list(robot_point_idxs))
        if len(robot_point_idxs) > 0:
            pcd_label[robot_point_idxs] = 1
        for query_key, query_label_id in zip(["object", "target"], [2, 3]):
            target_labels = self.get_target_labels(taskvar, step_id, episode)
            if target_labels is None or query_key not in target_labels:
                continue

            gt_target_labels = target_labels[query_key]

            if self.pc_label_type != "mix":
                pc_label_type = self.pc_label_type
            else:
                pc_label_type = random.choice(["coarse", "fine"])

            labels = (
                gt_target_labels[pc_label_type]
                if pc_label_type in gt_target_labels
                else gt_target_labels["fine"]
            )
            gt_query_mask = [pcd_sem == x for x in labels]
            gt_query_mask = np.sum(gt_query_mask, 0) > 0
            if "zrange" in gt_target_labels:
                gt_query_mask = (
                    gt_query_mask
                    & (pcd_xyz[..., 2] > gt_target_labels["zrange"][0])
                    & (pcd_xyz[..., 2] < gt_target_labels["zrange"][1])
                )
            if "xy_bbox" in gt_target_labels and self.pc_label_type != "coarse":
                bbox_offset = gt_target_labels["xy_bbox"]["bbox_offset"]
                bbox_size = gt_target_labels["xy_bbox"]["bbox_size"]

                obj_pose = gt_target_labels["xy_bbox"]["obj_pose"]
                bbox_pos = obj_pose[:3]
                gripper_quat = obj_pose[3:]

                bbox_rot = R.from_quat(gripper_quat).as_matrix()

                # Rotate the offset by the rotation matrix
                rotated_offset = bbox_rot @ bbox_offset

                obbox = o3d.geometry.OrientedBoundingBox(bbox_pos, bbox_rot, bbox_size)
                obbox.translate(rotated_offset, relative=True)

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pcd_xyz)
                box_idx = obbox.get_point_indices_within_bounding_box(pcd.points)

                bbox_mask = np.zeros((pcd_xyz.shape[0],), dtype=bool)
                if box_idx:  # If not empty
                    indices = np.array(box_idx, dtype=np.int64)
                    bbox_mask[indices] = True

                gt_query_mask = gt_query_mask & bbox_mask
            pcd_label[gt_query_mask] = query_label_id

        # normalize point cloud
        if self.xyz_shift == "none":
            pc_centroid = np.zeros((3,))
        elif self.xyz_shift == "center":
            pc_centroid = np.mean(pcd_xyz, 0)
        elif self.xyz_shift == "gripper":
            pc_centroid = copy.deepcopy(gripper_pose[:3])
        if self.xyz_norm:
            pc_radius = np.max(np.sqrt(np.sum((pcd_xyz - pc_centroid) ** 2, axis=1)))
        else:
            pc_radius = 1
        pcd_xyz = (pcd_xyz - pc_centroid) / pc_radius
        gripper_pose[:3] = (gripper_pose[:3] - pc_centroid) / pc_radius

        pcd_ft = pcd_xyz
        if self.use_height:
            pcd_ft = np.concatenate([pcd_ft, height[:, None]], -1)
        if self.use_color:
            pcd_rgb = (pcd_rgb / 255.0) * 2 - 1
            pcd_ft = np.concatenate([pcd_ft, pcd_rgb], -1)

        outs = {
            "pc_fts": torch.from_numpy(pcd_ft).float(),
            "pc_labels": torch.from_numpy(pcd_label).long(),
            "offset": torch.LongTensor([pcd_xyz.shape[0]]),
            "npoints_in_batch": [pcd_xyz.shape[0]],
            "pc_centroids": pc_centroid,
            "pc_radius": pc_radius,
            "ee_poses": torch.from_numpy(gripper_pose).float().unsqueeze(0),
        }

        return outs


class GroundtruthActioner(object):
    def __init__(self, config) -> None:
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # build LLM high-level planner
        llm_config = config.llm_planner
        self.llm_planner = GroundtruthTaskPlanner(llm_config.gt_plan_file)

        mp_expr_dir = config.motion_planner.expr_dir
        mp_config_file = config.motion_planner.config_file
        mp_config = get_model_config(mp_config_file)
        data_cfg = mp_config.TRAIN_DATASET
        self.instr_include_objects = data_cfg.get("instr_include_objects", False)
        self.vlm_pipeline = GroundtruthVision(
            self.config.object_grounding.gt_label_file,
            num_points=data_cfg.num_points,
            voxel_size=mp_config.MODEL.action_config.voxel_size,
            same_npoints_per_example=data_cfg.same_npoints_per_example,
            rm_robot=data_cfg.rm_robot,
            xyz_shift=data_cfg.xyz_shift,
            xyz_norm=data_cfg.xyz_norm,
            use_height=data_cfg.use_height,
            pc_label_type=(
                data_cfg.pc_label_type
                if config.motion_planner.pc_label_type is None
                else config.motion_planner.pc_label_type
            ),
            use_color=data_cfg.get("use_color", False),
        )

        # build motion planner
        # self.clip_model = OpenClipEncoder(device=self.device) # to encode action/object texts
        self.clip_model = ClipEncoder(config.clip_path, device=self.device)
        self.motion_planner = self.build_motion_planner(
            config.motion_planner, device=self.device
        )

        # caches
        self.set_system_caches()

    def set_system_caches(self):
        self.action_embeds, self.query_embeds = {}, {}

    def build_motion_planner(self, mp_config, device):
        mp_model_config = get_model_config(mp_config.config_file)
        mp_model_config.defrost()
        mp_model_config.MODEL.ptv3_config.enable_flash = mp_config.enable_flashattn
        mp_model_config.freeze()
        if mp_model_config.MODEL.model_class == "MotionPlannerPTV3CA":
            motion_planner = MotionPlannerPTV3CA(mp_model_config.MODEL).to(self.device)
        else:
            motion_planner = MotionPlannerPTV3AdaNorm(mp_model_config.MODEL).to(
                self.device
            )
        motion_planner.eval()
        load_checkpoint(motion_planner, mp_config.checkpoint)
        return motion_planner

    @torch.no_grad()
    def predict(
        self,
        task_str,
        variation,
        step_id,
        obs_state_dict,
        episode_id,
        instructions,
        cache=None,
    ):
        """
        returns an action [position_x, position_y, position_z, quat_w, quat_x, quat_y, quat_z, gripper_state]

        #action[0:3] represents (x, y, z) coordinates for the end-effector position
        These are rescaled from normalized values: pred_actions[:, :3] = (pred_actions[:, :3] * batch["pc_radius"]) + batch["pc_centroids"]
        The z coordinate is constrained to be above the table: pred_actions[:, 2] = np.maximum(pred_actions[:, 2], self.vlm_pipeline.TABLE_HEIGHT + 0.005)

        #action[3:7] represents rotation as a quaternion

        #Gripper State (1 value):
        action[7] represents the gripper state. It's a binary value (after sigmoid activation) where:
        1 = Open gripper
        0 = Closed gripper
        """
        taskvar = f"{task_str}+{variation}"

        if step_id == 0:
            cache = EasyDict(valid_actions=[], object_vars={})
            if self.config.motion_planner.save_obs_outs:
                cache.episode_outdir = os.path.join(
                    self.config.motion_planner.pred_dir,
                    "obs_outs",
                    f"{task_str}+{variation}",
                    f"{episode_id}",
                )
                os.makedirs(cache.episode_outdir, exist_ok=True)
            else:
                cache.episode_outdir = None

        if len(cache.valid_actions) > 0:
            cur_action = cache.valid_actions[0][:8]
            cache.valid_actions = cache.valid_actions[1:]
            out = {"action": cur_action, "cache": cache}
            if cache.episode_outdir:
                np.save(
                    os.path.join(cache.episode_outdir, f"{step_id}.npy"),
                    {"obs": obs_state_dict, "action": cur_action},
                )
            return out
        pcd_images = obs_state_dict["pc"]
        rgb_images = obs_state_dict["rgb"]
        sem_images = obs_state_dict["gt_mask"]
        arm_links_info = obs_state_dict["arm_links_info"]
        gripper_pose = copy.deepcopy(obs_state_dict["gripper"])

        # initialize: task planning
        if step_id == 0:
            LOGGER.info("task planning step 0")
            if self.config.llm_planner.use_groundtruth:
                highlevel_plans = self.llm_planner(taskvar)

            cache.highlevel_plans = [parse_code(x) for x in highlevel_plans]
            cache.highlevel_step_id = 0
            cache.highlevel_step_id_norelease = 0

            if self.config.save_obs_outs_dir is not None:
                obs_dir = os.path.join(
                    self.config.save_obs_outs_dir, f"ep_{episode_id}"
                )
                os.makedirs(obs_dir, exist_ok=True)

                save_path = os.path.join(obs_dir, f"highlevel_plans.json")
                json.dump(
                    {
                        "instructions": instructions,
                        "instruction": instructions[0],
                        "plans": highlevel_plans,
                        "parsed_plans": cache.highlevel_plans,
                    },
                    open(save_path, "w"),
                )

        if cache.highlevel_step_id >= len(cache.highlevel_plans):
            if self.config.pipeline.restart:
                cache.highlevel_step_id = 0
                cache.highlevel_step_id_norelease = 0
            else:
                if self.config.save_obs_outs_dir is not None:
                    LOGGER.info(f"Saving step obs outputs...")
                    self._save_outputs(
                        episode_id,
                        step_id,
                        None,
                        obs_state_dict,
                        [np.zeros((8,))],
                        None,
                        None,
                        None,
                        cache,
                    )
                return {"action": np.zeros((8,)), "cache": cache}

        plan = cache.highlevel_plans[cache.highlevel_step_id]
        if plan is None:
            if self.config.save_obs_outs_dir is not None:
                LOGGER.info(f"Saving step obs outputs...")
                self._save_outputs(
                    episode_id,
                    step_id,
                    None,
                    obs_state_dict,
                    [np.zeros((8,))],
                    None,
                    None,
                    None,
                    cache,
                )
            return {"action": np.zeros((8,))}

        if plan["action"] == "release":
            action = gripper_pose
            action[7] = 1
            cache.highlevel_step_id += 1

            if self.config.save_obs_outs_dir is not None:
                LOGGER.info(f"Saving step obs outputs...")
                batch = self.vlm_pipeline(
                    taskvar,
                    cache.highlevel_step_id_norelease,
                    pcd_images,
                    sem_images,
                    gripper_pose,
                    arm_links_info,
                    rgb_images=rgb_images,
                    episode_id=episode_id,
                )
                self._save_outputs(
                    episode_id,
                    step_id,
                    batch,
                    obs_state_dict,
                    [action],
                    None,
                    plan,
                    "release",
                    cache,
                )
            return {"action": action, "cache": cache}

        batch = self.vlm_pipeline(
            taskvar,
            cache.highlevel_step_id_norelease,
            pcd_images,
            sem_images,
            gripper_pose,
            arm_links_info,
            rgb_images=rgb_images,
            episode_id=episode_id,
        )

        # motion planning
        action_name = plan["action"]
        if self.instr_include_objects:
            if "object" in plan and plan["object"] is not None:
                object_name = "".join([x for x in plan["object"] if not x.isdigit()])
                object_name = object_name.replace("_", " ").strip()
                action_name = f"{action_name} {object_name}"
            if (
                "target" in plan
                and plan["target"] is not None
                and plan["target"] not in ["up", "down", "out", "in"]
            ):
                # TODO: should keep the target name is target is a variable
                target_name = "".join([x for x in plan["target"] if not x.isdigit()])
                target_name = target_name.replace("_", " ").strip()
                action_name = f"{action_name} to {target_name}"

        action_embeds = self.clip_model(
            "text", action_name, use_prompt=False, output_hidden_states=True
        )[
            0
        ]  # shape=(txt_len, hidden_size)
        batch.update(
            {
                "txt_embeds": action_embeds,
                "txt_lens": [action_embeds.size(0)],
            }
        )

        pred_actions = self.motion_planner(batch, compute_loss=False)[
            0
        ]  # action_length =8, 3 pos, 4 quaternions, 1 gripper state, 1 stop prob
        pred_actions[:, 7:] = torch.sigmoid(pred_actions[:, 7:])
        pred_actions = pred_actions.data.cpu().numpy()

        # rescale the predicted position
        pred_actions[:, :3] = (pred_actions[:, :3] * batch["pc_radius"]) + batch[
            "pc_centroids"
        ]
        pred_actions[:, 2] = np.maximum(
            pred_actions[:, 2], self.vlm_pipeline.TABLE_HEIGHT + 0.005
        )

        valid_actions = []
        for t, pred_action in enumerate(pred_actions):
            valid_actions.append(pred_action)
            if t + 1 >= self.config.motion_planner.run_action_step:
                break
            if pred_action[-1] > 0.5:
                break

        if pred_action[-1] > 0.5:
            cache.highlevel_step_id += 1
            cache.highlevel_step_id_norelease += 1

        cache.valid_actions = valid_actions[1:]
        out = {"action": valid_actions[0][:8], "cache": cache}

        if self.config.save_obs_outs_dir is not None:
            LOGGER.info(f"Saving step obs outputs...")
            self._save_outputs(
                episode_id,
                step_id,
                batch,
                obs_state_dict,
                valid_actions,
                None,
                plan,
                action_name,
                cache,
            )

        return out

    def _save_outputs(
        self,
        episode_id: int,
        step_id: int,
        batch: Dict,
        obs_state_dict: Dict,
        valid_actions: List,
        extra_outs: Dict,
        plan: Dict,
        action_name: str,
        cache: Dict,
    ) -> None:
        """Save outputs to file."""
        obs_dir = os.path.join(
            self.config.save_obs_outs_dir, f"ep_{episode_id}", "steps"
        )
        os.makedirs(obs_dir, exist_ok=True)

        save_path = os.path.join(obs_dir, f"step_{step_id}.npy")
        if batch is not None:
            del batch["txt_embeds"], batch["txt_lens"]

            # Convert tensors to numpy arrays
            processed_batch = {
                k: v.data.cpu().numpy() if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
        else:
            processed_batch = None

        results = {
            "batch": processed_batch,
            "obs": obs_state_dict,
            "valid_actions": valid_actions,
            "extra_outs": extra_outs,
            "plan": plan,
            "action_name": action_name,
            "cache": cache,
        }

        np.save(
            save_path,
            results,
        )
