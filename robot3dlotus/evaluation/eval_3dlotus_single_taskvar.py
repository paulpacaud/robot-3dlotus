from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import time
import os
import tap
from termcolor import colored

from genrobo3d.rlbench.environments import RLBenchEnv, Mover
from rlbench.backend.utils import task_file_to_task_class
from pyrep.errors import IKError, ConfigurationPathError
from rlbench.backend.exceptions import InvalidActionError

from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor
from genrobo3d.rlbench.recorder import (
    TaskRecorder,
    StaticCameraMotion,
    CircleCameraMotion,
    AttachedCameraMotion,
)

from genrobo3d.train.utils.misc import set_random_seed
from genrobo3d.evaluation.common import write_to_file
from robot3dlotus.actioner.actioner_3dlotus import Actioner
from genrobo3d.train.utils.logger import LOGGER


class EvaluationArguments(tap.Tap):
    """Arguments for evaluation."""

    expr_dir: str
    ckpt_step: int
    taskvar: str
    device: str = "cuda"
    image_size: List[int] = [256, 256]
    max_tries: int = 10
    max_steps: int = 25
    microstep_data_dir: str = ""
    seed: int = 100
    num_episodes: int = 20
    num_ensembles: int = 1
    best_disc_pos: str = "max"
    save_obs_outs_dir: str = None
    record_video: bool = False
    video_dir: str = None
    not_include_robot_cameras: bool = False
    video_rotate_cam: bool = False
    video_resolution: int = 480
    real_robot: bool = False
    enable_flashattn: bool = False


@dataclass
class EvaluationMetrics:
    """Stores evaluation metrics for a task."""

    success_rate: float = 0.0
    total_inference_time: float = 0.0
    total_inference_steps: int = 0

    @property
    def avg_inference_time(self) -> float:
        """Calculate average inference time."""
        if self.total_inference_steps == 0:
            return 0
        return self.total_inference_time / self.total_inference_steps

    @property
    def inference_speed_fps(self) -> float:
        """Calculate inference speed in FPS."""
        avg_time = self.avg_inference_time
        return 1.0 / avg_time if avg_time > 0 else 0


@dataclass
class EpisodeState:
    """Stores the current state of an episode."""

    obs_state_dict: Dict
    instructions: str
    reward: Optional[float] = None
    terminate: bool = False
    step_id: int = 0


@dataclass
class EpisodeResult:
    """Stores the results of an episode evaluation."""

    reward: Optional[float]
    step_id: int
    inference_time: float


class VideoRecorder:
    """Handles video recording setup and management."""

    def __init__(self, args: EvaluationArguments, task):
        self.args = args
        self.task = task
        self.recorder = None
        self.cameras = {}

    def setup(self) -> Optional[TaskRecorder]:
        """Sets up video recording if enabled."""
        if not self.args.record_video:
            return None

        # Add global camera
        cam_placeholder = Dummy("cam_cinematic_placeholder")
        cam_resolution = [self.args.video_resolution, self.args.video_resolution]
        cam = VisionSensor.create(cam_resolution)
        cam.set_pose(cam_placeholder.get_pose())
        cam.set_parent(cam_placeholder)

        if self.args.video_rotate_cam:
            self.cameras["global"] = CircleCameraMotion(
                cam, Dummy("cam_cinematic_base"), 0.005
            )
        else:
            self.cameras["global"] = StaticCameraMotion(cam)

        if not self.args.not_include_robot_cameras:
            self._setup_robot_cameras(cam_resolution)

        self.recorder = TaskRecorder(self.cameras, fps=30)
        self.task._scene.register_step_callback(self.recorder.take_snap)
        return self.recorder

    def _setup_robot_cameras(self, resolution: List[int]) -> None:
        """Sets up robot-mounted cameras."""
        cam_left = VisionSensor.create(resolution)
        cam_right = VisionSensor.create(resolution)
        cam_wrist = VisionSensor.create(resolution)

        self.cameras["left"] = AttachedCameraMotion(
            cam_left, self.task._scene._cam_over_shoulder_left
        )
        self.cameras["right"] = AttachedCameraMotion(
            cam_right, self.task._scene._cam_over_shoulder_right
        )
        self.cameras["wrist"] = AttachedCameraMotion(
            cam_wrist, self.task._scene._cam_wrist
        )


class EpisodeEvaluator:
    """Handles evaluation of individual episodes."""

    def __init__(
        self,
        args: EvaluationArguments,
        metrics: EvaluationMetrics,
        actioner: Actioner,
        env: RLBenchEnv,
    ):
        self.args = args
        self.metrics = metrics
        self.actioner = actioner
        self.env = env

    @dataclass
    class StepResult:
        """Results from executing a single step."""

        inference_time: float
        terminate: bool = False
        reward: Optional[float] = None
        obs_state_dict: Optional[Dict] = None

    def evaluate_episode(
        self,
        episode_id: int,
        task: Any,
        move: Mover,
        demos: Optional[List] = None,
        recorder: Optional[TaskRecorder] = None,
    ) -> None:
        """Main entry point for episode evaluation."""
        # Initialize episode
        state = self._initialize_episode(episode_id, task, move, demos)

        # Run episode steps
        result = self._run_episode_steps(episode_id, state, move)

        # Update metrics
        self._update_metrics(result)

        # Record video if enabled
        self._record_video(episode_id, result, recorder)

        # Log results
        self._log_episode_results(episode_id, result)

    def _initialize_episode(
        self, episode_id: int, task: Any, move: Mover, demos: Optional[List]
    ) -> EpisodeState:
        """Initialize episode state."""
        if demos is not None:
            instructions, obs = task.reset_to_demo(demos[episode_id])
        else:
            instructions, obs = task.reset()

        obs_state_dict = self.env.get_observation(obs)
        move.reset(obs_state_dict["gripper"])

        return EpisodeState(obs_state_dict=obs_state_dict, instructions=instructions)

    def _run_episode_steps(
        self, episode_id: int, state: EpisodeState, move: Mover
    ) -> EpisodeResult:
        """Execute steps for a single episode."""
        total_inference_time = 0.0

        while state.step_id < self.args.max_steps and not state.terminate:
            LOGGER.info(f"Episode {episode_id} Step {state.step_id + 1}")
            step_result = self._execute_step(episode_id, state, move)

            if step_result.terminate:
                state.terminate = True
            if step_result.reward is not None:
                state.reward = step_result.reward
            if step_result.obs_state_dict is not None:
                state.obs_state_dict = step_result.obs_state_dict

            total_inference_time += step_result.inference_time
            state.step_id += 1

            if state.reward == 1:  # Success
                break

        return EpisodeResult(
            reward=state.reward,
            step_id=state.step_id,
            inference_time=total_inference_time,
        )

    def _execute_step(
        self, episode_id: int, state: EpisodeState, move: Mover
    ) -> StepResult:
        """Execute a single step in the episode."""
        # Prepare batch for model
        batch = self._prepare_batch(episode_id, state)

        # Get model prediction
        start_time = time.time()
        output = self.actioner.predict(**batch)
        inference_time = time.time() - start_time

        # Update metrics
        self.metrics.total_inference_time += inference_time
        self.metrics.total_inference_steps += 1

        action = output["action"]
        if action is None:
            return self.StepResult(inference_time, terminate=True)

        # Execute action
        try:
            obs, reward, terminate, _ = move(action, verbose=False)
            return self.StepResult(
                inference_time=inference_time,
                terminate=terminate,
                reward=reward,
                obs_state_dict=self.env.get_observation(obs),
            )

        except (IKError, ConfigurationPathError, InvalidActionError) as e:
            LOGGER.info(f"Error in episode {episode_id} step {state.step_id}: {e}")
            return self.StepResult(
                inference_time=inference_time, terminate=True, reward=0
            )

    def _prepare_batch(self, episode_id: int, state: EpisodeState) -> Dict:
        """Prepare batch for model prediction."""
        return {
            "task_str": self.args.task_str,
            "variation": self.args.variation,
            "step_id": state.step_id,
            "obs_state_dict": state.obs_state_dict,
            "episode_id": episode_id,
            "instructions": state.instructions,
        }

    def _update_metrics(self, result: EpisodeResult) -> None:
        """Update evaluation metrics based on episode results."""
        if result.reward == 1:
            self.metrics.success_rate += 1 / self.args.num_episodes

    def _record_video(
        self, episode_id: int, result: EpisodeResult, recorder: Optional[TaskRecorder]
    ) -> None:
        """Record video if enabled."""
        if not (recorder and self.args.record_video):
            return

        save_path = os.path.join(self.args.video_dir, f"ep_{episode_id}")
        os.makedirs(save_path, exist_ok=True)
        recorder.save(os.path.join(save_path, f"video-SR{result.reward}"))

    def _log_episode_results(self, episode_id: int, result: EpisodeResult) -> None:
        """Log episode evaluation results."""
        LOGGER.info(
            f"Episode {episode_id} Step {result.step_id + 1} Reward {result.reward} "
            f"Accumulated SR: {(self.metrics.success_rate * 100):.2f} "
            f"Estimated SR: {(self.metrics.success_rate * self.args.num_episodes / (episode_id + 1) * 100):.2f}"
        )


class TaskEvaluator:
    """Handles evaluation of a single task."""

    def __init__(self, args: EvaluationArguments):
        self.args = args
        self.actioner = Actioner(args)
        self.metrics = EvaluationMetrics()

    def setup_environment(self) -> Tuple[RLBenchEnv, Any, Mover]:
        """Initialize the RLBench environment and task."""
        env = RLBenchEnv(
            data_path=self.args.microstep_data_dir,
            apply_rgb=True,
            apply_pc=True,
            apply_mask=True,
            headless=True,
            image_size=self.args.image_size,
            cam_rand_factor=0,
        )

        env.env.launch()
        task_type = task_file_to_task_class(self.args.task_str)
        task = env.env.get_task(task_type)
        task.set_variation(self.args.variation)

        move = Mover(task, max_tries=self.args.max_tries)
        return env, task, move

    def load_demos(self, env: RLBenchEnv) -> Optional[List]:
        """Load demonstration episodes if available."""
        if not self.args.microstep_data_dir:
            return None

        episodes_dir = os.path.join(
            self.args.microstep_data_dir,
            self.args.task_str,
            f"variation{self.args.variation}",
            "episodes",
        )

        if not os.path.exists(episodes_dir):
            LOGGER.info(
                f"No demos available for {self.args.task_str}+{self.args.variation}"
            )
            return None

        demos = []
        episode_ids = sorted(os.listdir(episodes_dir), key=lambda ep: int(ep[7:]))

        for idx, ep in enumerate(episode_ids):
            try:
                demo = env.get_demo(
                    self.args.task_str, self.args.variation, idx, load_images=False
                )
                demos.append(demo)
            except Exception as e:
                LOGGER.info(f"Problem loading demo_id: {idx} {ep}")
                LOGGER.info(e)

        return demos if demos else None

    def evaluate(self) -> None:
        """Run evaluation for the task."""
        # Setup environment
        env, task, move = self.setup_environment()

        # Load demos if available
        demos = self.load_demos(env)

        # Setup video recording if enabled
        video_recorder = VideoRecorder(self.args, task)
        recorder = video_recorder.setup()

        # Create episode evaluator
        episode_evaluator = EpisodeEvaluator(
            self.args, self.metrics, self.actioner, env
        )

        # Run evaluation episodes
        for episode_id in range(self.args.num_episodes):
            LOGGER.info(f"Starting eval of episode {episode_id}")
            episode_evaluator.evaluate_episode(episode_id, task, move, demos, recorder)

        # Write results
        self._save_results()

        # Cleanup
        env.env.shutdown()
        LOGGER.info(
            colored(
                f"Task: {self.args.task_str}+{self.args.variation} "
                f"SR: {self.metrics.success_rate:.2f}",
                "black",
                "on_yellow",
            )
        )

    def _save_results(self) -> None:
        """Save evaluation results to file."""
        pred_dir = os.path.join(self.args.expr_dir, "preds", f"seed{self.args.seed}")
        os.makedirs(pred_dir, exist_ok=True)
        pred_file = os.path.join(pred_dir, "results.jsonl")

        results = {
            "checkpoint": os.path.join(
                self.args.expr_dir, "ckpts", f"model_step_{self.args.ckpt_step}.pt"
            ),
            "task": self.args.task_str,
            "variation": self.args.variation,
            "num_episodes": self.args.num_episodes,
            "sr": round(self.metrics.success_rate, 2),
            "fps": round(self.metrics.inference_speed_fps, 2),
            "avg_inference_time": self.metrics.avg_inference_time,
        }
        write_to_file(pred_file, results)


def main():
    """Main evaluation function."""
    args = EvaluationArguments().parse_args(known_only=True)
    args.remained_args = args.extra_args
    args.exp_config = os.path.join(args.expr_dir, "logs", "training_config.yaml")
    args.checkpoint = os.path.join(
        args.expr_dir, "ckpts", f"model_step_{args.ckpt_step}.pt"
    )
    task_str, variation = args.taskvar.split("+")
    args.task_str = task_str
    args.variation = int(variation)

    # Set random seed
    set_random_seed(args.seed)

    # Run evaluation
    evaluator = TaskEvaluator(args)
    evaluator.evaluate()


if __name__ == "__main__":
    main()
