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
"""Measure the batch-1 wall-clock latency (and Hz) of a VLA policy's vision encoder, LLM backbone and
action expert.

One real observation is taken from the first environment's reset, preprocessed once, and then
`policy.predict_action_chunk` is run `warmup + iters` times on it. No `env.step` happens inside the
measured loop, so nothing waits on the simulator or a robot.

Every stage is timed in wall mode: `torch.cuda.synchronize()` + `time.perf_counter()` at both of its
boundaries, so a stage's time includes its Python code, kernel launches and GPU execution, and stages
never overlap. `torch.compile` is disabled because it fuses across stage boundaries.

Usage:
```
python -m lerobot.scripts.lerobot_profiling \
    --policy.path=./vla_models/pi05_libero_finetuned \
    --policy.device=cuda:0 \
    --env.type=libero \
    --env.task=libero_object \
    --warmup=5 \
    --iters=50 \
    --output_dir=./eval_logs/profile_pi05
```

GR00T N1.7 LeRobot checkpoints store the trainer's local base-model path and expect the wrist camera as
`wrist_image`, so point `base_model_path` at the Hub repo and remap the LIBERO camera:
```
python -m lerobot.scripts.lerobot_profiling \
    --policy.path=./vla_models/gr00t17_libero_spatial \
    --policy.base_model_path=nvidia/GR00T-N1.7-3B \
    --policy.device=cuda:0 \
    --env.type=libero \
    --env.task=libero_spatial \
    --env.camera_name_mapping='{"agentview_image": "image", "robot0_eye_in_hand_image": "wrist_image"}' \
    --output_dir=./eval_logs/profile_groot
```

CLI options are those of `lerobot-eval` (`EvalPipelineConfig`) plus `--warmup` and `--iters`.
Stages are defined per policy type in `STAGES`; other policy types report the total only.
"""

import json
import logging
import time
from collections import defaultdict
from collections.abc import Callable
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat
from typing import Any

import numpy as np
import torch
from torch import nn

from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.envs import close_envs, make_env, make_env_pre_post_processors, preprocess_observation
from lerobot.envs.utils import NEW_ROLLOUT_OPTION
from lerobot.policies import PreTrainedPolicy, make_policy, make_pre_post_processors
from lerobot.utils.device_utils import get_safe_torch_device
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import init_logging

logger = logging.getLogger(__name__)


@dataclass
class ProfilingPipelineConfig(EvalPipelineConfig):
    # Inferences run before measuring (cuDNN autotuning, allocator warm-up); excluded from the stats.
    warmup: int = 5
    # Measured inferences.
    iters: int = 50


# ---------------------------------------------------------------------------------------------------
# Stage definitions
# ---------------------------------------------------------------------------------------------------


@dataclass
class Stage:
    """One timed part of a policy's inference.

    Exactly one of `modules` / `method` is set:
    - `modules`: time every `forward` of these modules (via forward hooks). Use it when the model calls
      the submodule through `nn.Module.__call__`.
    - `method`: time calls of this bound method (replaced by a wrapper on the instance). Use it when the
      part is not a module call, e.g. a custom layer loop. `when` filters which calls count.
    `steps` optionally names a module whose calls are only counted (e.g. the denoiser inside a loop).
    """

    name: str
    modules: Callable[[PreTrainedPolicy], list[nn.Module]] | None = None
    method: Callable[[PreTrainedPolicy], tuple[Any, str]] | None = None
    when: Callable[[tuple, dict], bool] | None = None
    steps: Callable[[PreTrainedPolicy], nn.Module] | None = None


def _inputs_embeds(args: tuple, kwargs: dict) -> list:
    return kwargs["inputs_embeds"] if "inputs_embeds" in kwargs else args[3]


def _is_prefix_call(args: tuple, kwargs: dict) -> bool:
    """`[prefix, None]` = VLM-only prefix pass (fills the KV cache); `[None, suffix]` = action expert."""
    embeds = _inputs_embeds(args, kwargs)
    return embeds[0] is not None and embeds[1] is None


def _pi05_paligemma(p):
    return p.model.paligemma_with_expert.paligemma.model


def _smolvla_vlm(p):
    return p.model.vlm_with_expert.get_vlm_model()


STAGES: dict[str, list[Stage]] = {
    "pi05": [
        # SigLIP + projector, once per camera (incl. the padded empty camera).
        Stage(
            "vision",
            modules=lambda p: [_pi05_paligemma(p).vision_tower, _pi05_paligemma(p).multi_modal_projector],
        ),
        # Gemma 2B over the prefix. PaliGemmaWithExpertModel.forward calls `language_model.forward(...)`
        # directly, which bypasses module hooks: time the `[prefix, None]` call of the joint forward.
        Stage("llm", method=lambda p: (p.model.paligemma_with_expert, "forward"), when=_is_prefix_call),
        # One flow-matching step of the Gemma 300M action expert (x num_inference_steps).
        Stage("action", method=lambda p: (p.model, "denoise_step")),
    ],
    "smolvla": [
        Stage("vision", modules=lambda p: [_smolvla_vlm(p).vision_model, _smolvla_vlm(p).connector]),
        # SmolVLMWithExpertModel.forward runs its own layer loop, so text_model.forward never fires:
        # time the prefix call of the joint forward instead.
        Stage("llm", method=lambda p: (p.model.vlm_with_expert, "forward"), when=_is_prefix_call),
        Stage("action", method=lambda p: (p.model, "denoise_step")),
    ],
    "groot": [
        Stage("vision", modules=lambda p: [p._groot_model.backbone.visual]),
        Stage("llm", modules=lambda p: [p._groot_model.backbone.language_model]),
        # The whole DiT flow-matching loop (action encoder + DiT + decoder per step); DiT calls = steps.
        Stage(
            "action",
            method=lambda p: (p._groot_model.action_head, "get_action_with_features"),
            steps=lambda p: p._groot_model.action_head.model,
        ),
    ],
}


# ---------------------------------------------------------------------------------------------------
# Wall-mode stage profiler
# ---------------------------------------------------------------------------------------------------


class WallStageProfiler:
    """Attach wall-clock timers (sync + perf_counter at both boundaries) to a policy's stages."""

    def __init__(self, policy: PreTrainedPolicy, stages: list[Stage], device: torch.device):
        self._sync = (lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else (lambda: None)
        self._handles: list[Any] = []
        self._patched: list[tuple[Any, str]] = []
        self._start: dict[str, float] = {}
        self._depth: dict[str, int] = defaultdict(int)
        self.time: dict[str, float] = defaultdict(float)
        self.calls: dict[str, int] = defaultdict(int)
        self.steps: dict[str, int] = defaultdict(int)
        self.names = [s.name for s in stages]
        for stage in stages:
            self._attach(policy, stage)

    def _begin(self, name: str) -> None:
        self._depth[name] += 1
        if self._depth[name] == 1:  # only the outermost call of a stage is timed
            self._sync()
            self._start[name] = time.perf_counter()

    def _end(self, name: str) -> None:
        self._depth[name] -= 1
        if self._depth[name] == 0:
            self._sync()
            self.time[name] += time.perf_counter() - self._start[name]
            self.calls[name] += 1

    def _attach(self, policy: PreTrainedPolicy, stage: Stage) -> None:
        name = stage.name
        if stage.modules is not None:
            modules = stage.modules(policy)
            for module in modules:
                self._handles.append(module.register_forward_pre_hook(lambda m, a, n=name: self._begin(n)))
                self._handles.append(module.register_forward_hook(lambda m, a, o, n=name: self._end(n)))
            if stage.steps is None:
                # Steps = calls of the first module (e.g. images through the vision tower), not of every
                # module in the stage.
                self._count_steps(modules[0], name)
        if stage.method is not None:
            obj, attr = stage.method(policy)
            original = getattr(obj, attr)

            def wrapper(*args, _orig=original, _when=stage.when, _name=name, **kwargs):
                if _when is not None and not _when(args, kwargs):
                    return _orig(*args, **kwargs)
                self._begin(_name)
                try:
                    return _orig(*args, **kwargs)
                finally:
                    self._end(_name)

            setattr(obj, attr, wrapper)  # instance attribute shadows the class method
            self._patched.append((obj, attr))
        if stage.steps is not None:
            self._count_steps(stage.steps(policy), name)

    def _count_steps(self, module: nn.Module, name: str) -> None:
        self._handles.append(
            module.register_forward_hook(lambda m, a, o, n=name: self.steps.__setitem__(n, self.steps[n] + 1))
        )

    def reset(self) -> None:
        self.time.clear()
        self.calls.clear()
        self.steps.clear()

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        for obj, attr in self._patched:
            delattr(obj, attr)
        self._handles.clear()
        self._patched.clear()


# ---------------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------------


def _stats_ms(values_s: list[float]) -> dict[str, float]:
    arr = np.asarray(values_s) * 1000
    return {
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
    }


def _hz(ms: float) -> float:
    return 1000.0 / ms if ms > 0 else float("inf")


@parser.wrap()
def profiling_main(cfg: ProfilingPipelineConfig):
    if cfg.policy is None:
        raise ValueError("--policy.path (or --policy.type) is required for profiling.")
    if getattr(cfg.policy, "compile_model", False):
        logging.info(
            "Disabling policy.compile_model: torch.compile fuses across the profiled stage boundaries."
        )
        cfg.policy.compile_model = False
    # Only one observation is needed: don't build a vec env for every task of the suite.
    if hasattr(cfg.env, "task_ids") and cfg.env.task_ids is None:
        cfg.env.task_ids = [0]

    logging.info(pformat(asdict(cfg)))
    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)

    # ---- one real observation --------------------------------------------------------------------
    envs = make_env(cfg.env, n_envs=1, use_async_envs=False, trust_remote_code=cfg.trust_remote_code)
    task_group, task_id, env = next((tg, tid, e) for tg, group in envs.items() for tid, e in group.items())
    observation, _ = env.reset(
        seed=[cfg.seed] if cfg.seed is not None else None, options={NEW_ROLLOUT_OPTION: True}
    )
    raw_batch = preprocess_observation(observation)
    try:
        raw_batch["task"] = list(env.call("task_description"))
    except (AttributeError, NotImplementedError):
        raw_batch["task"] = [""]
    close_envs(envs)
    logging.info(f"Profiling on {task_group} task {task_id}: {raw_batch['task'][0]!r}")

    # ---- policy and processors -------------------------------------------------------------------
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
    env_preprocessor, _ = make_env_pre_post_processors(env_cfg=cfg.env, policy_cfg=cfg.policy)

    stages = STAGES.get(cfg.policy.type, [])
    if not stages:
        logging.warning(f"No stage definition for policy type '{cfg.policy.type}': reporting the total only.")
    profiler = WallStageProfiler(policy, stages, device)
    sync = (lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else (lambda: None)

    records: list[dict[str, Any]] = []
    amp = torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext()
    try:
        with torch.inference_mode(), amp:
            for i in range(cfg.warmup + cfg.iters):
                if i == cfg.warmup and device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                policy.reset()
                preprocessor.reset()
                postprocessor.reset()
                profiler.reset()

                sync()
                t0 = time.perf_counter()
                batch = preprocessor(env_preprocessor(deepcopy(raw_batch)))
                sync()
                t1 = time.perf_counter()
                actions = policy.predict_action_chunk(batch)
                sync()
                t2 = time.perf_counter()
                # Real inference postprocesses one action per env step; time the first one.
                postprocessor(actions[:, 0]).to("cpu")
                t3 = time.perf_counter()

                if i >= cfg.warmup:
                    records.append(
                        {
                            "preprocess": t1 - t0,
                            "inference": t2 - t1,
                            "postprocess": t3 - t2,
                            "stages": dict(profiler.time),
                            "calls": dict(profiler.calls),
                            "steps": dict(profiler.steps),
                        }
                    )
    finally:
        profiler.detach()

    # ---- aggregate -------------------------------------------------------------------------------
    inference = _stats_ms([r["inference"] for r in records])
    result: dict[str, Any] = {
        "policy_type": cfg.policy.type,
        "policy_path": str(cfg.policy.pretrained_path),
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "dtype": str(getattr(cfg.policy, "dtype", None)),
        "compile_model": False,
        "batch_size": 1,
        "task": {"group": task_group, "id": task_id},
        "warmup": cfg.warmup,
        "iters": cfg.iters,
        "stages": {},
    }
    staged_total = np.zeros(len(records))
    for name in profiler.names:
        per_inference = [r["stages"].get(name, 0.0) for r in records]
        staged_total += np.asarray(per_inference)
        calls = int(np.median([r["calls"].get(name, 0) for r in records]))
        steps = int(np.median([r["steps"].get(name, 0) for r in records])) or calls
        stats = _stats_ms(per_inference)
        per_step_ms = stats["mean_ms"] / steps if steps else float("nan")
        result["stages"][name] = {
            **stats,
            "calls_per_inference": calls,
            "steps_per_inference": steps,
            "per_step_mean_ms": per_step_ms,
            "hz_per_inference": _hz(stats["mean_ms"]),
            "hz_per_step": _hz(per_step_ms),
            "share_pct": 100 * stats["mean_ms"] / inference["mean_ms"],
        }
    other = _stats_ms(list(np.asarray([r["inference"] for r in records]) - staged_total))
    result["stages"]["other"] = {**other, "share_pct": 100 * other["mean_ms"] / inference["mean_ms"]}
    result["inference"] = {**inference, "hz": _hz(inference["mean_ms"])}
    result["preprocess"] = _stats_ms([r["preprocess"] for r in records])
    result["postprocess"] = _stats_ms([r["postprocess"] for r in records])
    n_action_steps = getattr(cfg.policy, "n_action_steps", None)
    if n_action_steps:
        result["n_action_steps"] = n_action_steps
        result["actions_per_s"] = n_action_steps * result["inference"]["hz"]
    if device.type == "cuda":
        result["max_memory_allocated_gb"] = torch.cuda.max_memory_allocated(device) / 1024**3

    # ---- report ----------------------------------------------------------------------------------
    lines = [
        f"Latency (batch 1, wall clock, eager) - {result['policy_type']} on {result['device']}, "
        f"{cfg.iters} iters after {cfg.warmup} warmup",
        f"{'stage':<10} {'mean ms':>9} {'p50':>8} {'p90':>8} {'steps':>6} {'ms/step':>8} {'Hz':>7} {'Hz/step':>8} {'share':>6}",
    ]
    for name, s in result["stages"].items():
        if name == "other":
            lines.append(
                f"{name:<10} {s['mean_ms']:9.1f} {s['p50_ms']:8.1f} {s['p90_ms']:8.1f} {'':>6} {'':>8} {'':>7} {'':>8} {s['share_pct']:5.1f}%"
            )
            continue
        lines.append(
            f"{name:<10} {s['mean_ms']:9.1f} {s['p50_ms']:8.1f} {s['p90_ms']:8.1f} {s['steps_per_inference']:6d} "
            f"{s['per_step_mean_ms']:8.1f} {s['hz_per_inference']:7.1f} {s['hz_per_step']:8.1f} {s['share_pct']:5.1f}%"
        )
    inf = result["inference"]
    lines.append(
        f"{'inference':<10} {inf['mean_ms']:9.1f} {inf['p50_ms']:8.1f} {inf['p90_ms']:8.1f} {'':>6} {'':>8} {inf['hz']:7.1f}"
    )
    lines.append(
        f"preprocess {result['preprocess']['mean_ms']:.1f} ms | postprocess {result['postprocess']['mean_ms']:.1f} ms"
        + (
            f" | {result['actions_per_s']:.1f} actions/s (n_action_steps={n_action_steps})"
            if n_action_steps
            else ""
        )
        + (
            f" | peak GPU mem {result['max_memory_allocated_gb']:.2f} GB"
            if "max_memory_allocated_gb" in result
            else ""
        )
    )
    logging.info("\n" + "\n".join(lines))

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "profile.json", "w") as f:
        json.dump(result, f, indent=2)
    logging.info(f"Saved {output_dir / 'profile.json'}")


def main():
    init_logging()
    register_third_party_plugins()
    profiling_main()


if __name__ == "__main__":
    main()
