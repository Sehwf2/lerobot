#!/usr/bin/env bash
# Evaluate a local pi05 VLA checkpoint on the LIBERO benchmark.
# Optionally shards the 4 standard LIBERO suites across multiple GPUs.
#
# Prerequisites:
#   - `uv sync --extra libero` (or `CMAKE_POLICY_VERSION_MINIMUM=3.5 uv sync --extra libero`
#     if the `egl-probe` build fails against a CMake >= 4.0 toolchain)
#   - Weights at $POLICY_PATH (default: ./vla_models/pi05_base)
#
# $POLICY_PATH can point at either kind of checkpoint. Both lerobot/pi05_base and
# lerobot/pi05_libero_finetuned ship a config.json, so its mere presence isn't a reliable
# signal (see "Loading a checkpoint" in docs/source/pi05.mdx) -- what matters is whether the
# checkpoint's *stored input features already match LIBERO's camera keys*
# (observation.images.image / image2, as produced by --env.type=libero):
#   - Already LIBERO-shaped (e.g. lerobot/pi05_libero_finetuned) -> loaded with --policy.path
#     (weights AND config; --policy.type is omitted, read from config.json).
#   - Not LIBERO-shaped, e.g. lerobot/pi05_base (generic base_0_rgb/left_wrist_0_rgb/
#     right_wrist_0_rgb slots) -> loaded with --policy.type=pi05 --policy.pretrained_path
#     (weights only; feature names come from the env instead, stored settings like
#     n_action_steps reset to defaults). Loading pi05_base with --policy.path would lock in
#     the wrong camera keys and break against the LIBERO env.
# This script auto-detects which case applies by grepping config.json for the LIBERO
# "observation.images.image" key; override with FORCE_POLICY_PATH_MODE=path|pretrained if a
# checkpoint doesn't fit this heuristic.
#
# IMPORTANT: pi05_base is Physical Intelligence's *general* pretrained checkpoint — it has
# never seen LIBERO's action space / camera layout. Evaluating it directly will produce a
# near-random success rate; that's fine as an end-to-end sanity check of the eval pipeline,
# but for a meaningful benchmark point $POLICY_PATH at a LIBERO-finetuned checkpoint instead,
# e.g. lerobot/pi05_libero_finetuned (docs/source/libero.mdx "Reproducing published results").
#
# `lerobot-eval` itself is single-GPU/single-process (see AGENTS.md: no torchrun/accelerate
# for eval). Multi-GPU here means: shard the LIBERO suites round-robin across GPU_IDS,
# run one independent `lerobot-eval` process per GPU in the background, then merge each
# shard's eval_info.json into one combined result.
#
# Usage:
#   ./examples/libero/eval_pi05_base.sh                                       # pi05_base, single GPU (cuda:0)
#   POLICY_PATH=./vla_models/pi05_libero_finetuned ./examples/libero/eval_pi05_base.sh
#   GPU_IDS=0,1,2,3 ./examples/libero/eval_pi05_base.sh                        # shard the 4 suites across 4 GPUs
#   POLICY_PATH=./vla_models/pi05_base TASK=libero_object N_EPISODES=3 ./examples/libero/eval_pi05_base.sh
#   DEBUG=1 ./examples/libero/eval_pi05_base.sh                                # single GPU (first of GPU_IDS), breakpoint() enabled
#
# Without DEBUG=1, PYTHONBREAKPOINT=0 is exported so a stray breakpoint() is a no-op instead of
# hanging a background shard waiting on stdin.

# example:
# env -u DEVICE DEBUG=1 GPU_IDS=0 POLICY_PATH=./vla_models/pi05_libero_finetuned TASK=libero_object N_EPISODES=1 BATCH_SIZE=1 OUTPUT_DIR=./eval_logs/debug_pi05 ./examples/libero/eval_pi05_base.sh

# env -u DEVICE DEBUG=1 GPU_IDS=0 \
#     POLICY_PATH=./vla_models/pi05_libero_finetuned \
#     TASK=libero_object N_EPISODES=1 BATCH_SIZE=1 \
#     OUTPUT_DIR=./eval_logs/debug_pi05 \
#     ./examples/libero/eval_pi05_base.sh

# POLICY_PATH=./vla_models/pi05_libero_finetuned GPU_IDS=0,1,2,3 \
#   N_EPISODES=50 BATCH_SIZE=10 \
#   OUTPUT_DIR=./eval_logs/pi05_libero_finetuned_n50 \
#   ./examples/libero/eval_pi05_base.sh

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

# Headless MuJoCo rendering (unset/override if you have a display + GPU EGL context).
export MUJOCO_GL="${MUJOCO_GL:-egl}"

POLICY_PATH="${POLICY_PATH:-./vla_models/pi05_base}"
TASK="${TASK:-libero_spatial,libero_object,libero_goal,libero_10}"
N_EPISODES="${N_EPISODES:-10}"
BATCH_SIZE="${BATCH_SIZE:-1}"
N_ACTION_STEPS="${N_ACTION_STEPS:-10}" # matches OpenPI's reference LIBERO setting
OUTPUT_DIR="${OUTPUT_DIR:-./eval_logs/$(basename "${POLICY_PATH}")_libero}"

# Comma-separated CUDA device indices, e.g. GPU_IDS=0,1,2,3. Defaults to a single GPU
# (cuda:0), which reproduces the original single-process behavior exactly.
GPU_IDS="${GPU_IDS:-0}"

# See the header comment: config.json alone doesn't tell us --policy.path vs
# --policy.pretrained_path is correct -- check whether it's already shaped for LIBERO's cameras.
POLICY_PATH_MODE="${FORCE_POLICY_PATH_MODE:-}"
if [ -z "${POLICY_PATH_MODE}" ]; then
  if [ -f "${POLICY_PATH}/config.json" ] && grep -qF '"observation.images.image"' "${POLICY_PATH}/config.json"; then
    POLICY_PATH_MODE="path"
  else
    POLICY_PATH_MODE="pretrained"
  fi
fi

if [ "${POLICY_PATH_MODE}" = "path" ]; then
  echo "${POLICY_PATH}: config.json already LIBERO-shaped -> loading as a full checkpoint (--policy.path)."
  POLICY_ARGS=(--policy.path="${POLICY_PATH}")
else
  echo "${POLICY_PATH}: not LIBERO-shaped (or no config.json) -> loading weights-only (--policy.pretrained_path)."
  POLICY_ARGS=(--policy.type=pi05 --policy.pretrained_path="${POLICY_PATH}")
fi

IFS=',' read -ra GPUS <<< "${GPU_IDS}"
IFS=',' read -ra TASKS <<< "${TASK}"

# DEBUG=1: run a single foreground process on the first GPU of GPU_IDS so breakpoint()/pdb
# gets the terminal. Otherwise disable breakpoint() entirely.
DEBUG="${DEBUG:-0}"
if [ "${DEBUG}" != "0" ]; then
  if [ "${#GPUS[@]}" -gt 1 ]; then
    echo "DEBUG=${DEBUG}: forcing single GPU ${GPUS[0]} (ignoring the rest of GPU_IDS=${GPU_IDS})."
  fi
  GPUS=("${GPUS[0]:-0}")
else
  export PYTHONBREAKPOINT=0
fi

# This container may export a restrictive CUDA_VISIBLE_DEVICES (e.g. "0") at the shell
# level even when /dev/nvidia0..N are all present -- `nvidia-smi -L` bypasses that and
# always shows every physical GPU, which is misleading. `--policy.device=cuda:${gpu}`
# addresses an index into the *visible* set, not the physical GPU id, so it breaks under
# that restriction. Instead, override CUDA_VISIBLE_DEVICES to exactly one physical GPU per
# `uv run` invocation and always address it as cuda:0 inside that process -- this works
# regardless of what the outer shell's CUDA_VISIBLE_DEVICES was set to.
#
# MuJoCo's EGL renderer (via robosuite) always enumerates exactly one EGL device on this
# container's driver stack, regardless of which/how many CUDA devices CUDA_VISIBLE_DEVICES
# exposes (confirmed: even CUDA_VISIBLE_DEVICES=0,1,2,3 -> torch sees 4 devices, but EGL
# still enumerates only 1). libero's env wrapper calls robosuite with render_gpu_device_id=-1
# (its hardcoded default, never overridden), which makes egl_context.py's
# create_initialized_egl_device_display() read CUDA_VISIBLE_DEVICES's *first* comma-separated
# entry as the EGL device index -- so leaving MUJOCO_EGL_DEVICE_ID unset and just listing the
# physical GPU first (e.g. "2,0") picks index 2, which doesn't exist among the single
# enumerated EGL device and fails with "must be between 0 and 0, got 2".
# Setting MUJOCO_EGL_DEVICE_ID=0 explicitly bypasses that: it's read as a plain digit instead
# (device_idx = int("0")), skipping the device_id=-1/list-order logic entirely and always
# selecting the one EGL device that actually exists. But robosuite's binding_utils.py has its
# own independent, cruder sanity check that only runs when MUJOCO_EGL_DEVICE_ID is set: it
# asserts MUJOCO_EGL_DEVICE_ID is a literal substring of CUDA_VISIBLE_DEVICES. So both parts
# are required together: CUDA_VISIBLE_DEVICES="<gpu>,0" (target GPU first, so cuda:0 still
# maps to it; "0" present to satisfy that assert) AND MUJOCO_EGL_DEVICE_ID=0 (so EGL selection
# ignores list order and position of "0" within it).
cvd_for_gpu() {
  local gpu="$1"
  if [ "${gpu}" = "0" ]; then
    echo "0"
  else
    echo "${gpu},0"
  fi
}

if [ -n "${DEVICE:-}" ]; then
  # Explicit override (e.g. DEVICE=cpu, DEVICE=mps): use it verbatim, no GPU pinning.
  RUN_ENV=()
  POLICY_DEVICE="${DEVICE}"
else
  RUN_ENV=(env "CUDA_VISIBLE_DEVICES=$(cvd_for_gpu "${GPUS[0]:-0}")" "MUJOCO_EGL_DEVICE_ID=0")
  POLICY_DEVICE="cuda:0"
fi

if [ "${#GPUS[@]}" -le 1 ]; then
  exec "${RUN_ENV[@]}" uv run lerobot-eval \
    "${POLICY_ARGS[@]}" \
    --policy.n_action_steps="${N_ACTION_STEPS}" \
    --policy.device="${POLICY_DEVICE}" \
    --env.type=libero \
    --env.task="${TASK}" \
    --env.max_parallel_tasks=1 \
    --eval.batch_size="${BATCH_SIZE}" \
    --eval.n_episodes="${N_EPISODES}" \
    --output_dir="${OUTPUT_DIR}"
fi

# --- Multi-GPU: shard suites round-robin across GPU_IDS, one lerobot-eval process per GPU ---

echo "Sharding ${#TASKS[@]} suite(s) across ${#GPUS[@]} GPUs: ${GPU_IDS}"

declare -A GPU_TASKS
for i in "${!TASKS[@]}"; do
  gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
  GPU_TASKS["${gpu}"]+="${TASKS[$i]},"
done

pids=()
shard_dirs=()
for gpu in "${!GPU_TASKS[@]}"; do
  gpu_tasks="${GPU_TASKS[$gpu]%,}" # strip trailing comma
  shard_dir="${OUTPUT_DIR}/shard_gpu${gpu}"
  shard_dirs+=("${shard_dir}")
  mkdir -p "${shard_dir}"
  echo "  physical GPU ${gpu} <- ${gpu_tasks}  (logs: ${shard_dir}/run.log)"
  env CUDA_VISIBLE_DEVICES="$(cvd_for_gpu "${gpu}")" MUJOCO_EGL_DEVICE_ID=0 uv run lerobot-eval \
    "${POLICY_ARGS[@]}" \
    --policy.n_action_steps="${N_ACTION_STEPS}" \
    --policy.device=cuda:0 \
    --env.type=libero \
    --env.task="${gpu_tasks}" \
    --env.max_parallel_tasks=1 \
    --eval.batch_size="${BATCH_SIZE}" \
    --eval.n_episodes="${N_EPISODES}" \
    --output_dir="${shard_dir}" \
    > "${shard_dir}/run.log" 2>&1 &
  pids+=($!)
done

failed=0
for pid in "${pids[@]}"; do
  wait "${pid}" || failed=1
done
if [ "${failed}" -ne 0 ]; then
  echo "One or more shards failed. Check shard_gpu*/run.log under ${OUTPUT_DIR}." >&2
  exit 1
fi

echo "All shards done. Merging results into ${OUTPUT_DIR}/eval_info.json"

uv run python - "${OUTPUT_DIR}/eval_info.json" "${shard_dirs[@]}" <<'PYEOF'
import json
import sys
from lerobot.utils.eval_stats import success_summary

out_path, *shard_dirs = sys.argv[1:]

per_task = []
per_group: dict[str, dict] = {}
for shard_dir in shard_dirs:
    with open(f"{shard_dir}/eval_info.json") as f:
        shard = json.load(f)
    per_task.extend(shard["per_task"])
    per_group.update(shard["per_group"])  # suites don't overlap across shards

all_successes = [s for t in per_task for s in t["metrics"]["successes"]]
overall = success_summary(all_successes)

merged = {"per_task": per_task, "per_group": per_group, "overall": overall}
with open(out_path, "w") as f:
    json.dump(merged, f, indent=2)

ci_low, ci_high = overall["pc_success_ci95"]
print(
    f"Overall success rate: {overall['pc_success']:.1f}% "
    f"({overall['n_success']}/{overall['n_episodes']} episodes, "
    f"95% CI {ci_low:.1f}% to {ci_high:.1f}%)"
)
for group, info in per_group.items():
    print(f"  {group}: {info['pc_success']:.1f}% ({info['n_success']}/{info['n_episodes']})")
PYEOF
