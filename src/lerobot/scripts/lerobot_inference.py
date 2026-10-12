#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run a policy in simulation one episode at a time: single GPU, batch size 1, strictly sequential.

A stripped-down variant of `lerobot_eval.py` meant for inspecting and debugging inference:
- one process on one device, one environment per task (a `SyncVectorEnv` with a single sub-env)
- tasks run one after another, and each task's episodes run one after another
- no async envs, no parallel tasks, no dataset recording / Hub push, no episode data for online training

Usage:
```
python -m lerobot.scripts.lerobot_inference \
    --policy.path=./vla_models/pi05_libero_finetuned \
    --policy.device=cuda:0 \
    --env.type=libero \
    --env.task=libero_object \
    --eval.n_episodes=1 \
    --output_dir=./eval_logs/inference_pi05
```

CLI options are the same as `lerobot-eval` (`EvalPipelineConfig` in lerobot/configs/eval.py), with
`--eval.n_episodes` meaning episodes *per task*. `--eval.batch_size`, `--eval.use_async_envs`,
`--eval.recording` and `--env.max_parallel_tasks` are ignored.
"""

import json
import logging
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from pprint import pformat
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from termcolor import colored

from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.envs import (
    check_env_attributes_and_types,
    close_envs,
    make_env,
    make_env_pre_post_processors,
    preprocess_observation,
)
from lerobot.envs.utils import NEW_ROLLOUT_OPTION
from lerobot.lerobot_types import PolicyAction
from lerobot.policies import PreTrainedPolicy, make_policy, make_pre_post_processors
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.constants import ACTION
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.eval_stats import success_summary
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.io_utils import write_video
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import init_logging

logger = logging.getLogger(__name__)

# Same cap as lerobot-eval: only the first episodes of each task are saved as videos.
MAX_EPISODES_RENDERED_PER_TASK = 10


def _task_description(env: gym.vector.VectorEnv) -> str:
    """Natural-language task of the single sub-env (empty if the env doesn't expose one)."""
    for attr in ("task_description", "task"):
        try:
            return list(env.call(attr))[0]
        except (AttributeError, NotImplementedError):
            continue
    return ""


def _is_success(info: dict) -> bool:
    """Read the single sub-env's success flag from a vector-env step `info`.

    VectorEnv stores it in `info["final_info"]["is_success"]` once the episode ends (Gymnasium >= 1.0), in
    a per-env sequence of dicts on older Gymnasium, or directly in `info["is_success"]`.
    """
    if "final_info" in info:
        final_info = info["final_info"]
        if isinstance(final_info, dict):
            return bool(np.asarray(final_info.get("is_success", False)).reshape(-1)[0])
        item = final_info[0]
        return isinstance(item, dict) and bool(item.get("is_success", False))
    if "is_success" in info:
        return bool(np.asarray(info["is_success"]).reshape(-1)[0])
    return False


def run_episode(
    env: gym.vector.VectorEnv,
    policy: PreTrainedPolicy,
    env_preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    env_postprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    seed: int | None = None,
    render: bool = False,
) -> dict:
    """Run one episode in a single-sub-env vector env until it terminates, truncates or hits max steps.

    Returns:
        A dict with `success`, `sum_reward`, `max_reward`, `n_steps`, `episode_s` (wall time),
        `policy_s` (time spent in `policy.select_action`) and, if `render`, the RGB `frames`.
    """
    if env.num_envs != 1:
        raise ValueError(f"run_episode expects a vector env with exactly 1 sub-env, got {env.num_envs}.")

    start = time.perf_counter()
    # Reset the policy (action queue), its processors (per-episode state) and the env.
    policy.reset()
    preprocessor.reset()
    postprocessor.reset()
    observation, _ = env.reset(seed=[seed] if seed is not None else None, options={NEW_ROLLOUT_OPTION: True})
    check_env_attributes_and_types(env)

    frames: list[np.ndarray] = []
    if render:
        frames.append(env.envs[0].render())

    task = _task_description(env)
    max_steps = env.call("_max_episode_steps")[0]
    rewards: list[float] = []
    success = False
    policy_s = 0.0

    for _ in range(max_steps):
        # Numpy -> torch and LeRobot key names, then env-specific and policy preprocessing.
        batch = preprocess_observation(observation)
        batch["task"] = [task]
        batch = env_preprocessor(batch)
        batch = preprocessor(batch)

        t0 = time.perf_counter()
        with torch.inference_mode():
            action = policy.select_action(batch)
        if action.is_cuda:
            torch.cuda.synchronize(action.device)
        policy_s += time.perf_counter() - t0

        action = postprocessor(action)
        action = env_postprocessor({ACTION: action})[ACTION]
        action_numpy: np.ndarray = action.to("cpu").numpy()
        assert action_numpy.shape[0] == 1, "Action dimensions should be (1, action_dim)"

        observation, reward, terminated, truncated, info = env.step(action_numpy)
        if render:
            frames.append(env.envs[0].render())

        rewards.append(float(reward[0]))
        success = success or _is_success(info)
        if terminated[0] or truncated[0]:
            break

    result = {
        "success": success,
        "sum_reward": float(np.sum(rewards)),
        "max_reward": float(np.max(rewards)) if rewards else float("nan"),
        "n_steps": len(rewards),
        "episode_s": time.perf_counter() - start,
        "policy_s": policy_s,
    }
    if render:
        result["frames"] = frames
    return result


def _summary(episodes: list[dict]) -> dict:
    """Success count, rate and 95% Wilson interval plus average rewards for a list of episode results."""
    return {
        **success_summary([ep["success"] for ep in episodes]),
        "avg_sum_reward": float(np.nanmean([ep["sum_reward"] for ep in episodes])),
        "avg_max_reward": float(np.nanmean([ep["max_reward"] for ep in episodes])),
    }


@parser.wrap()
def inference_main(cfg: EvalPipelineConfig):
    # Single batch, sequential: force one synchronous sub-env regardless of the eval config.
    if cfg.eval.batch_size != 1 or cfg.eval.use_async_envs:
        logging.info(
            f"Ignoring eval.batch_size={cfg.eval.batch_size} / eval.use_async_envs={cfg.eval.use_async_envs}: "
            "lerobot_inference always runs a single synchronous environment."
        )
    cfg.eval.batch_size = 1
    cfg.eval.use_async_envs = False

    logging.info(pformat(asdict(cfg)))
    device = get_safe_torch_device(cfg.policy.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)

    output_dir = Path(cfg.output_dir)
    videos_dir = output_dir / "videos"
    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {output_dir}")

    logging.info("Making environment (1 synchronous sub-env per task).")
    envs = make_env(cfg.env, n_envs=1, use_async_envs=False, trust_remote_code=cfg.trust_remote_code)

    logging.info("Making policy.")
    policy = make_policy(cfg=cfg.policy, env_cfg=cfg.env, rename_map=cfg.rename_map)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        preprocessor_overrides={
            "device_processor": {"device": str(policy.config.device)},
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        },
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=cfg.env, policy_cfg=cfg.policy)

    tasks = [(tg, tid, env) for tg, group in envs.items() for tid, env in group.items()]
    n_episodes = cfg.eval.n_episodes
    logging.info(f"Running {len(tasks)} task(s) x {n_episodes} episode(s), one at a time.")

    per_episode: list[dict] = []
    start = time.perf_counter()
    amp = torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext()
    try:
        with torch.no_grad(), amp:
            for task_group, task_id, env in tasks:
                fps = env.unwrapped.metadata["render_fps"]
                task_results: list[dict] = []
                for episode_ix in range(n_episodes):
                    # Same seeding as lerobot-eval with batch_size=1: seed + episode index, per task.
                    seed = None if cfg.seed is None else cfg.seed + episode_ix
                    render = episode_ix < MAX_EPISODES_RENDERED_PER_TASK
                    result = run_episode(
                        env,
                        policy,
                        env_preprocessor,
                        env_postprocessor,
                        preprocessor,
                        postprocessor,
                        seed=seed,
                        render=render,
                    )

                    video_path = None
                    if render:
                        video_path = videos_dir / f"{task_group}_{task_id}" / f"eval_episode_{episode_ix}.mp4"
                        video_path.parent.mkdir(parents=True, exist_ok=True)
                        write_video(str(video_path), result.pop("frames"), fps)

                    record = {
                        "task_group": task_group,
                        "task_id": task_id,
                        "episode_ix": episode_ix,
                        "seed": seed,
                        **result,
                        "video_path": str(video_path) if video_path else None,
                    }
                    per_episode.append(record)
                    task_results.append(record)
                    logging.info(
                        f"[{task_group} {task_id} ep {episode_ix}] success={record['success']} "
                        f"steps={record['n_steps']} episode={record['episode_s']:.1f}s "
                        f"policy={record['policy_s']:.1f}s"
                    )

                task_summary = success_summary([r["success"] for r in task_results])
                logging.info(
                    f"[{task_group} {task_id}] {task_summary['n_success']}/{task_summary['n_episodes']} success"
                )
                env.close()
    finally:
        close_envs(envs)

    eval_s = time.perf_counter() - start
    groups = sorted({ep["task_group"] for ep in per_episode})
    task_keys = sorted({(ep["task_group"], ep["task_id"]) for ep in per_episode})
    info = {
        "per_episode": per_episode,
        "per_task": [
            {
                "task_group": tg,
                "task_id": tid,
                **_summary([ep for ep in per_episode if (ep["task_group"], ep["task_id"]) == (tg, tid)]),
            }
            for tg, tid in task_keys
        ],
        "per_group": {g: _summary([ep for ep in per_episode if ep["task_group"] == g]) for g in groups},
        "overall": {
            **_summary(per_episode),
            "eval_s": eval_s,
            "eval_ep_s": eval_s / max(1, len(per_episode)),
        },
    }

    overall = info["overall"]
    ci_low, ci_high = overall["pc_success_ci95"]
    logging.info(
        "Success rate %.1f%% (%d/%d episodes, 95%% Wilson interval %.1f%% to %.1f%%)",
        overall["pc_success"],
        overall["n_success"],
        overall["n_episodes"],
        ci_low,
        ci_high,
    )
    for group, group_info in info["per_group"].items():
        logging.info(
            f"  {group}: {group_info['pc_success']:.1f}% ({group_info['n_success']}/{group_info['n_episodes']})"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "eval_info.json", "w") as f:
        json.dump(info, f, indent=2)
    logging.info("End of inference")


def main():
    init_logging()
    register_third_party_plugins()
    inference_main()


if __name__ == "__main__":
    main()
