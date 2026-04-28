from collections import defaultdict
import json
import os
import signal
import sys
import time
import numpy as np
from typing import Annotated, Optional

import torch
import tree
from mani_skill.utils import common
from mani_skill.utils.geometry import rotation_conversions
from mani_skill.utils import visualization
from mani_skill.utils.visualization.misc import images_to_video
signal.signal(signal.SIGINT, signal.SIG_DFL) # allow ctrl+c
from simpler_env.utils.env.observation_utils import get_image_from_maniskill3_obs_dict

import gymnasium as gym
import numpy as np
from mani_skill.envs.tasks.digital_twins.bridge_dataset_eval import *
from mani_skill.envs.sapien_env import BaseEnv
import tyro
from dataclasses import dataclass
from pathlib import Path

from simpler_env.policies.praxis.bridge_state import normalize_bridge_gripper_qpos


_BRIDGE_DATASET_TOOL_ROTATION = torch.tensor(
    [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
    dtype=torch.float32,
)


def _make_env(*, env_id: str, num_envs: int, shader: str) -> BaseEnv:
    sensor_configs = {"shader_pack": shader}
    return gym.make(
        env_id,
        obs_mode="rgb+segmentation",
        num_envs=num_envs,
        sensor_configs=sensor_configs,
    )


def _get_env_device(env) -> torch.device:
    if hasattr(env, "device"):
        return env.device

    unwrapped = getattr(env, "unwrapped", None)
    if unwrapped is not None and hasattr(unwrapped, "device"):
        return unwrapped.device

    raise AttributeError(
        "Environment does not expose a .device attribute on either the wrapper "
        "or env.unwrapped."
    )


def get_bridge_state_from_maniskill3_env(env) -> torch.Tensor:
    """Build Bridge LeRobot state: x, y, z, roll, pitch, yaw, gripper."""
    unwrapped = getattr(env, "unwrapped", env)
    agent = unwrapped.agent
    controller = getattr(agent, "controller", None)
    arm_controller = getattr(controller, "controllers", {}).get("arm")
    if arm_controller is None or not hasattr(arm_controller, "ee_pose_at_base"):
        raise AttributeError(
            "Praxis Bridge observations require an arm controller with "
            "ee_pose_at_base."
        )

    ee_pose_at_base = arm_controller.ee_pose_at_base
    rotation = ee_pose_at_base.to_transformation_matrix()[..., :3, :3]
    bridge_tool_rotation = _BRIDGE_DATASET_TOOL_ROTATION.to(
        device=rotation.device,
        dtype=rotation.dtype,
    )
    state_rotation = torch.matmul(rotation, bridge_tool_rotation.T)
    euler_xyz = rotation_conversions.matrix_to_euler_angles(state_rotation, "XYZ")

    qpos = common.to_tensor(agent.robot.get_qpos(), device=ee_pose_at_base.p.device)
    if qpos.ndim == 1:
        qpos = qpos[None, :]
    if qpos.shape[-1] < 2:
        raise ValueError(
            f"Expected robot qpos to include two gripper joints, got {qpos.shape}"
        )
    qpos = qpos.to(dtype=ee_pose_at_base.p.dtype)
    qlimits = common.to_tensor(agent.robot.get_qlimits(), device=qpos.device).to(
        dtype=qpos.dtype,
    )
    gripper = normalize_bridge_gripper_qpos(qpos, qlimits)

    return torch.cat([ee_pose_at_base.p, euler_xyz, gripper], dim=-1).to(torch.float32)


def _mean_metric(eval_metrics, key: str) -> float:
    values = eval_metrics.get(key)
    if not values:
        return 0.0
    return float(np.mean(values))


def _emit_progress_line(
    *,
    env_id: str,
    done_count: int,
    total_episodes: int,
    eval_metrics,
    loop_start_time: float,
) -> None:
    elapsed_s = max(0.0, float(time.time() - loop_start_time))
    parts = [
        "SIMPLER_EVAL",
        f"env_id={env_id}",
        f"done={int(done_count)}/{int(total_episodes)}",
        f"succ_rate={100.0 * _mean_metric(eval_metrics, 'success'):.1f}%",
        f"elapsed_s={elapsed_s:.1f}",
    ]
    if done_count > 0 and elapsed_s > 0:
        parts.append(f"{elapsed_s / float(done_count):.2f}s/ep")
    print(" ".join(parts), flush=True)


def _video_indices_for_batch(
    *,
    batch_num_envs: int,
    max_videos: int,
    videos_saved: int,
) -> list[int]:
    if max_videos <= 0:
        return list(range(int(batch_num_envs)))
    remaining = max(0, int(max_videos) - int(videos_saved))
    return list(range(min(int(batch_num_envs), remaining)))


def _video_frame(image_batch, env_index: int) -> np.ndarray:
    frame = image_batch[int(env_index)]
    if torch.is_tensor(frame):
        return frame.detach().cpu().numpy()
    return np.asarray(frame)


@dataclass
class Args:
    """
    This is a script to evaluate policies on real2sim environments. Example command to run: 

    XLA_PYTHON_CLIENT_PREALLOCATE=false python real2sim_eval_maniskill3.py \
        --model="octo-small" -e "PutEggplantInBasketScene-v1" -s 0 --num-episodes 192 --num-envs 64
    """


    env_id: Annotated[str, tyro.conf.arg(aliases=["-e"])] = "PutCarrotOnPlateInScene-v1"
    """The environment ID of the task you want to simulate. Can be one of
    PutCarrotOnPlateInScene-v1, PutSpoonOnTableClothInScene-v1, StackGreenCubeOnYellowCubeBakedTexInScene-v1, PutEggplantInBasketScene-v1"""

    shader: str = "default"

    num_envs: int = 1
    """Number of environments to run. With more than 1 environment the environment will use the GPU backend 
    which runs faster enabling faster large-scale evaluations. Note that the overall behavior of the simulation
    will be slightly different between CPU and GPU backends."""

    num_episodes: int = 100
    """Number of episodes to run and record evaluation metrics over"""

    record_dir: str = "videos"
    """The directory to save videos and results"""

    metrics_output_path: Optional[str] = None
    """Optional explicit JSON output path for aggregated metrics."""

    model: Optional[str] = None
    """The model to evaluate on the given environment. Can be one of octo-base, octo-small, rt-1x, praxis-remote. If not given, random actions are sampled."""

    ckpt_path: str = ""
    """Checkpoint path for models. Only used for RT models"""

    praxis_host: str = "127.0.0.1"
    """Host for the Praxis gRPC policy server when --model=praxis-remote."""

    praxis_port: int = 50051
    """Port for the Praxis gRPC policy server when --model=praxis-remote."""

    praxis_policy_setup: str = "widowx_bridge"
    """SIMPLER policy setup string to pass to the Praxis remote wrapper."""

    praxis_action_scale: float = 1.0
    """Action scale to apply inside the Praxis remote wrapper."""

    praxis_primary_image_key: str = "observation.images.image"
    """Primary Praxis image observation key for --model=praxis-remote."""

    praxis_state_key: str = "observation.state"
    """Praxis state observation key for --model=praxis-remote."""

    praxis_policy_kwargs_json: str = ""
    """Optional JSON object of kwargs to forward through Praxis remote predict_action."""

    seed: Annotated[int, tyro.conf.arg(aliases=["-s"])] = 0
    """Seed the model and environment. Default seed is 0"""

    reset_by_episode_id: bool = True
    """Whether to reset by fixed episode ids instead of random sampling initial states."""

    info_on_video: bool = False
    """Whether to write info text onto the video"""

    save_video: bool = True
    """Whether to save videos"""

    max_videos: int = 0
    """Maximum number of episode videos to save (0 = save all when save_video is true)."""

    debug: bool = False

def main():
    args = tyro.cli(Args)
    if args.seed is not None:
        np.random.seed(args.seed)

    # Setup up the policy inference model
    model = None
    try:
        policy_setup = "widowx_bridge"
        if args.model is None:
            pass
        else:
            if args.model == "octo-base" or args.model == "octo-small":
                from simpler_env.policies.octo.octo_model import OctoInference

                model = OctoInference(model_type=args.model, policy_setup=policy_setup, init_rng=args.seed, action_scale=1)
            elif args.model == "rt-1x":
                from simpler_env.policies.rt1.rt1_model import RT1Inference

                ckpt_path=args.ckpt_path
                model = RT1Inference(
                    saved_model_path=ckpt_path,
                    policy_setup=policy_setup,
                    action_scale=1,
                )
            elif args.model == "praxis-remote":
                from simpler_env.policies.praxis import PraxisRemoteInference

                policy_kwargs = (
                    json.loads(args.praxis_policy_kwargs_json)
                    if args.praxis_policy_kwargs_json
                    else None
                )
                if policy_kwargs is not None and not isinstance(policy_kwargs, dict):
                    raise ValueError(
                        "--praxis-policy-kwargs-json must decode to a JSON object."
                    )
                model = PraxisRemoteInference(
                    host=args.praxis_host,
                    port=args.praxis_port,
                    policy_setup=args.praxis_policy_setup,
                    action_scale=args.praxis_action_scale,
                    primary_image_key=args.praxis_primary_image_key,
                    state_key=args.praxis_state_key,
                    policy_kwargs=policy_kwargs,
                )
            elif args.model is not None:
                raise ValueError(f"Model {args.model} does not exist / is not supported.")
    except:
        if args.model is not None:
            raise Exception("SIMPLER Env Policy Inference is not installed")

    model_name = args.model if args.model is not None else "random"
    if model_name == "random":
        print("Using random actions.")
    exp_dir = os.path.join(args.record_dir, f"real2sim_eval/{model_name}_{args.env_id}")
    Path(exp_dir).mkdir(parents=True, exist_ok=True)

    eval_metrics = defaultdict(list)
    eps_count = 0
    episode_sum_rewards: list[float] = []
    episode_max_rewards: list[float] = []
    episode_lengths: list[int] = []
    task_descriptions_seen: list[str] = []
    sim_backend = None
    videos_saved = 0
    env: BaseEnv | None = None
    current_num_envs = 0

    print(f"Running Real2Sim Evaluation of model {args.model} on environment {args.env_id}")
    if model is not None and hasattr(model, "health_check"):
        ready, info = model.health_check()
        print(f"model health_check ready={ready} info={info}")
        if not ready:
            raise RuntimeError(f"Policy model {args.model} is not ready: {info}")

    timers = {"env.step+inference": 0, "env.step": 0, "inference": 0, "total": 0}
    total_start_time = time.time()
    _emit_progress_line(
        env_id=args.env_id,
        done_count=0,
        total_episodes=int(args.num_episodes),
        eval_metrics=eval_metrics,
        loop_start_time=total_start_time,
    )
    
    while eps_count < args.num_episodes:
        remaining_episodes = int(args.num_episodes - eps_count)
        batch_num_envs = min(int(args.num_envs), remaining_episodes)
        if env is None or current_num_envs != batch_num_envs:
            if env is not None:
                env.close()
            env = _make_env(
                env_id=args.env_id,
                num_envs=batch_num_envs,
                shader=args.shader,
            )
            current_num_envs = batch_num_envs
            env_device = _get_env_device(env)
            sim_backend = "gpu" if env_device.type == "cuda" else "cpu"
            print(
                f"Using {batch_num_envs} environments on the {sim_backend} simulation backend"
            )

        seed = args.seed + eps_count
        obs, _ = env.reset(
            seed=seed,
            options={"episode_id": torch.tensor([seed + i for i in range(batch_num_envs)])},
        )
        instruction = env.unwrapped.get_language_instruction()
        print("instruction:", instruction[0])
        for value in instruction:
            text = str(value)
            if text not in task_descriptions_seen:
                task_descriptions_seen.append(text)
        if model is not None:
            model.reset(instruction)
        predicted_terminated, truncated = False, False
        current_image = get_image_from_maniskill3_obs_dict(env, obs)
        video_indices = (
            _video_indices_for_batch(
                batch_num_envs=batch_num_envs,
                max_videos=int(args.max_videos),
                videos_saved=videos_saved,
            )
            if args.save_video
            else []
        )
        video_frames = {
            index: [_video_frame(current_image, index)] for index in video_indices
        }
        elapsed_steps = 0
        sum_rewards_this_episode = np.zeros(batch_num_envs, dtype=np.float64)
        max_rewards_this_episode = np.full(batch_num_envs, -np.inf, dtype=np.float64)
        while not (predicted_terminated or truncated):
            if model is not None:
                start_time = time.time()
                if args.model == "praxis-remote":
                    state = get_bridge_state_from_maniskill3_env(env)
                    raw_action, action = model.step(
                        current_image,
                        instruction,
                        state=state,
                    )
                else:
                    raw_action, action = model.step(current_image, instruction)
                if args.model == "praxis-remote":
                    action = torch.cat(
                        [
                            action["world_vector"],
                            action["rotation_delta"],
                            action["gripper"],
                        ],
                        dim=1,
                    )
                else:
                    action = torch.cat(
                        [
                            action["world_vector"],
                            action["rot_axangle"],
                            action["gripper"],
                        ],
                        dim=1,
                    )
                timers["inference"] += time.time() - start_time
            else:
                action = env.action_space.sample()
            
            start_time = time.time()
            obs, reward, terminated, truncated, info = env.step(action)
            timers["env.step"] += time.time() - start_time
            elapsed_steps += 1
            reward_np = common.to_numpy(reward).reshape(-1)
            sum_rewards_this_episode += reward_np
            max_rewards_this_episode = np.maximum(max_rewards_this_episode, reward_np)
            info = common.to_numpy(info)
            
            truncated = bool(truncated.any()) # note that all envs truncate and terminate at the same time.
            current_image = get_image_from_maniskill3_obs_dict(env, obs)
            if video_frames:
                for i, frames in video_frames.items():
                    frame = _video_frame(current_image, i)
                    if args.info_on_video:
                        frame = visualization.put_info_on_image(
                            frame,
                            tree.map_structure(lambda x: x[i], info),
                        )
                    frames.append(frame)

        for k, v in info.items():
            eval_metrics[k].append(v.flatten())
        episode_sum_rewards.extend(float(value) for value in sum_rewards_this_episode)
        episode_max_rewards.extend(float(value) for value in max_rewards_this_episode)
        episode_lengths.extend([elapsed_steps] * batch_num_envs)
        for i, frames in video_frames.items():
            if args.max_videos > 0 and videos_saved >= args.max_videos:
                break
            images_to_video(
                frames,
                exp_dir,
                f"{sim_backend}_eval_{seed + i}_success={info['success'][i].item()}",
                fps=10,
                verbose=True,
            )
            videos_saved += 1
        eps_count += batch_num_envs
        _emit_progress_line(
            env_id=args.env_id,
            done_count=eps_count,
            total_episodes=int(args.num_episodes),
            eval_metrics=eval_metrics,
            loop_start_time=total_start_time,
        )
        if batch_num_envs == 1:
            print(f"Evaluated episode {eps_count}. Seed {seed}. Results after {eps_count} episodes:")
        else:
            print(f"Evaluated {batch_num_envs} episodes, seeds {seed} to {eps_count}. Results after {eps_count} episodes:")
        for k, v in eval_metrics.items():
            print(f"{k}: {np.mean(v)}")
    # Print timing information
    timers["total"] = time.time() - total_start_time
    timers["env.step+inference"] = timers["env.step"] + timers["inference"]
    mean_metrics = {k: np.mean(v) for k, v in eval_metrics.items()}
    mean_metrics["n_episodes"] = int(eps_count)
    mean_metrics["total_episodes"] = int(eps_count)
    mean_metrics["success_rate"] = float(np.mean(eval_metrics["success"])) if "success" in eval_metrics else 0.0
    mean_metrics["avg_episode_length"] = float(np.mean(episode_lengths)) if episode_lengths else 0.0
    mean_metrics["avg_reward"] = float(np.mean(episode_sum_rewards)) if episode_sum_rewards else 0.0
    mean_metrics["avg_sum_reward"] = mean_metrics["avg_reward"]
    mean_metrics["avg_max_reward"] = float(np.mean(episode_max_rewards)) if episode_max_rewards else 0.0
    if task_descriptions_seen:
        mean_metrics["task_description"] = task_descriptions_seen[0]
        mean_metrics["task_descriptions"] = task_descriptions_seen
    mean_metrics["eval_s"] = float(timers["total"])
    mean_metrics["eval_ep_s"] = float(timers["total"]) / max(1, eps_count)
    mean_metrics["time/episodes_per_second"] = eps_count / timers["total"]
    print("Timing Info:")
    for key, value in timers.items():
        mean_metrics[f"time/{key}"] = value
        print(f"{key}: {value:.2f} seconds")
    metrics_path = args.metrics_output_path
    if metrics_path is None:
        metrics_path = os.path.join(exp_dir, f"{sim_backend}_eval_metrics.json")
        if sim_backend == "gpu":
            metrics_path = metrics_path.replace("gpu", f"gpu_{args.num_envs}_envs")
    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(mean_metrics, f, indent=4)
    print(f"Evaluation complete. Results saved to {exp_dir}. Metrics saved to {metrics_path}")
    if model is not None and hasattr(model, "close"):
        model.close()
    if env is not None:
        env.close()

if __name__ == "__main__":
    main()
    # Some cluster SAPIEN/ManiSkill stacks finish evaluation successfully but
    # crash during interpreter teardown; hard-exit once results are flushed.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
