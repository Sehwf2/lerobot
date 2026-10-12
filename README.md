# LeRobot VLA 평가 · 추론 · 프로파일링 (LIBERO)

이 저장소(LeRobot fork)에서 **pi0.5, GR00T N1.7, SmolVLA**를 LIBERO 시뮬레이션으로
평가(`lerobot-eval`), 순차 추론(`lerobot-inference`), 단계별 latency 측정(`lerobot_profiling`)하는 방법을 정리한 문서입니다.

모든 명령은 저장소 루트(`~/lerobot`)에서 실행합니다.

---

## 1. 디렉터리 구성

| 경로                                            | 내용                                                              |
| ----------------------------------------------- | ----------------------------------------------------------------- |
| `vla_models/` → `/nvme1/VLA/VLA_MODELS`         | 모델 체크포인트 (심볼릭 링크)                                     |
| `vla_benchmarks/` → `/nvme1/VLA/VLA_BENCHMARKS` | LIBERO 에셋·데이터셋 (심볼릭 링크)                                |
| `vla_models/download_libero_models.sh`          | LIBERO 파인튜닝 모델 다운로드                                     |
| `vla_benchmarks/download_libero.sh`             | LIBERO 에셋·설정·데이터셋 다운로드                                |
| `examples/libero/eval_pi05_base.sh`             | pi0.5 LIBERO 평가 (단일/멀티 GPU, DEBUG 모드)                     |
| `src/lerobot/scripts/lerobot_inference.py`      | 단일 GPU, 배치 1, 에피소드 하나씩 순차 추론 (`lerobot-inference`) |
| `src/lerobot/scripts/lerobot_profiling.py`      | vision / LLM / action 단계별 latency와 Hz 측정                    |
| `eval_logs/`                                    | 평가·추론·프로파일링 결과 (기본 출력 위치)                        |

---

## 2. 설치

### 2.1 Python 패키지

```bash
cd ~/lerobot
CMAKE_POLICY_VERSION_MINIMUM=3.5 uv sync --locked --inexact \
  --extra libero --extra pi --extra smolvla --extra groot
```

| 옵션                               | 이유                                                                   |
| ---------------------------------- | ---------------------------------------------------------------------- |
| `--extra libero`                   | `hf-libero`(LIBERO 시뮬레이터), `transformers`, `scipy`, 데이터셋 도구 |
| `--extra pi`                       | pi0.5 (의존성은 `libero`에 이미 포함)                                  |
| `--extra smolvla`                  | `num2words`, `accelerate` (SmolVLM 프로세서)                           |
| `--extra groot`                    | `diffusers`, `peft`, `timm`, `dm-tree`, `decord` (GR00T N1.7)          |
| `--inexact`                        | **필수.** 없으면 `uv sync`가 목록에 없는 extra의 패키지를 삭제합니다   |
| `CMAKE_POLICY_VERSION_MINIMUM=3.5` | CMake 4.x에서 `egl-probe` 빌드가 실패하는 문제 회피                    |

`pyproject.toml`의 `[project.scripts]`에 `lerobot-inference`가 등록되어 있어서, 위 명령을 한 번 실행하면
`.venv/bin/lerobot-inference`가 생성됩니다.

### 2.2 Hugging Face 로그인 (GR00T용)

GR00T N1.7은 `nvidia/Cosmos-Reason2-2B`(라이선스 동의 필요, 자동 승인)에서 토크나이저를 받습니다.

1. https://huggingface.co/nvidia/Cosmos-Reason2-2B 에서 라이선스 동의
2. 로그인

```bash
uv run hf auth login
```

### 2.3 모델 다운로드

```bash
./vla_models/download_libero_models.sh          # pi0.5 + GR00T N1.7 (4개 suite)
./vla_models/download_libero_models.sh pi05     # pi0.5만
./vla_models/download_libero_models.sh groot    # GR00T N1.7만
```

| 저장 위치                           | HF 저장소                                             | 크기   |
| ----------------------------------- | ----------------------------------------------------- | ------ |
| `vla_models/pi05_libero_finetuned`  | `lerobot/pi05_libero_finetuned_v044` (4개 suite 공용) | 7.0 GB |
| `vla_models/gr00t17_libero_spatial` | `nvidia/gr00t17-lerobot-libero_spatial-640`           | 12 GB  |
| `vla_models/gr00t17_libero_object`  | `nvidia/gr00t17-lerobot-libero_object-640`            | 12 GB  |
| `vla_models/gr00t17_libero_goal`    | `nvidia/gr00t17-lerobot-libero_goal-640`              | 12 GB  |
| `vla_models/gr00t17_libero_10`      | `nvidia/gr00t17-lerobot-libero_10-640`                | 12 GB  |

GR00T를 처음 실행하면 기반 모델 `nvidia/GR00T-N1.7-3B`(약 6.5 GB)가 `~/.cache/huggingface/hub`에 추가로 받아집니다.

### 2.4 LIBERO 에셋·설정

```bash
./vla_benchmarks/download_libero.sh sim          # 평가에 필요한 것만 (에셋 + 설정)
./vla_benchmarks/download_libero.sh dataset      # 학습용 lerobot/libero (MP4, ~1.9 GB)
./vla_benchmarks/download_libero.sh dataset-png  # HuggingFaceVLA/libero (PNG, ~70 GB)
./vla_benchmarks/download_libero.sh              # sim + dataset
```

`sim`이 하는 일:

1. `lerobot/libero-assets`(약 410 MB)를 `vla_benchmarks/libero/assets`에 다운로드
2. `~/.cache/libero/assets` → `vla_benchmarks/libero/assets` 심볼릭 링크 생성
   (`hf-libero` 0.1.4는 이 경로만 봅니다)
3. `~/.libero/config.yaml` 작성 (없으면 libero import 시 `input()`에서 멈춥니다)
4. 5개 suite의 태스크 수·초기 상태 파일 확인

bddl 태스크 정의와 초기 상태 파일은 `hf-libero` 패키지 안에 들어 있어서 따로 받지 않습니다.
패키지(2.1)를 먼저 설치해야 합니다.

### 2.5 확인

```bash
uv run python -c "import libero, transformers; print('ok')"
ls -l ~/.cache/libero/assets     # -> .../vla_benchmarks/libero/assets
cat ~/.libero/config.yaml
uv run lerobot-inference --help | head -3
```

---

## 3. GPU 지정 규칙 (중요)

이 서버는 셸에 `CUDA_VISIBLE_DEVICES="0"`과 `DEVICE="cuda"`가 기본으로 설정되어 있고(`/etc/environment`),
MuJoCo EGL 렌더러는 EGL 장치를 하나만 인식합니다. 그래서 직접 `lerobot-*`를 실행할 때는 아래 형식으로 GPU를 지정합니다.

| 사용할 GPU    | 환경변수                                          |
| ------------- | ------------------------------------------------- |
| GPU 0         | `CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0`   |
| GPU N (N ≠ 0) | `CUDA_VISIBLE_DEVICES=N,0 MUJOCO_EGL_DEVICE_ID=0` |

그리고 `--policy.device=cuda:0`을 줍니다. (보이는 장치 중 첫 번째 = 원하는 GPU)
공통으로 `MUJOCO_GL=egl`(헤드리스 렌더링)이 필요합니다.

측정·평가 전에 `nvidia-smi`로 비어 있는 GPU를 확인하세요.

---

## 4. 평가: `lerobot-eval` (`examples/libero/eval_pi05_base.sh`)

### 4.1 pi0.5 — 4 GPU 평가

GPU마다 suite 하나씩(spatial / object / goal / 10) 나눠서 `lerobot-eval`을 병렬로 실행하고,
끝나면 결과를 하나의 `eval_info.json`으로 합칩니다.

```bash
# 기본 평가: 태스크당 10 에피소드
POLICY_PATH=./vla_models/pi05_libero_finetuned GPU_IDS=0,1,2,3 \
  ./examples/libero/eval_pi05_base.sh

# 표준 평가: 태스크당 50 에피소드 (LIBERO 초기 상태는 태스크당 50개가 최대)
POLICY_PATH=./vla_models/pi05_libero_finetuned GPU_IDS=0,1,2,3 \
  N_EPISODES=50 BATCH_SIZE=10 \
  OUTPUT_DIR=./eval_logs/pi05_libero_finetuned_n50 \
  ./examples/libero/eval_pi05_base.sh

# 백그라운드 실행
nohup env POLICY_PATH=./vla_models/pi05_libero_finetuned GPU_IDS=0,1,2,3 \
  N_EPISODES=50 BATCH_SIZE=10 OUTPUT_DIR=./eval_logs/pi05_libero_finetuned_n50 \
  ./examples/libero/eval_pi05_base.sh > ./eval_logs/pi05_n50.log 2>&1 &
tail -f ./eval_logs/pi05_libero_finetuned_n50/shard_gpu*/run.log
```

### 4.2 pi0.5 — DEBUG 모드 (단일 GPU, `breakpoint()` 사용)

```bash
env -u DEVICE DEBUG=1 GPU_IDS=0 \
  POLICY_PATH=./vla_models/pi05_libero_finetuned \
  TASK=libero_object N_EPISODES=1 BATCH_SIZE=1 \
  OUTPUT_DIR=./eval_logs/debug_pi05 \
  ./examples/libero/eval_pi05_base.sh
```

- `DEBUG=1`: `GPU_IDS`의 첫 번째 GPU 하나로 포그라운드 실행 → `breakpoint()`에서 pdb가 뜹니다.
- `DEBUG` 없음: `PYTHONBREAKPOINT=0`이 설정되어 남아 있는 `breakpoint()`는 무시됩니다.
- `env -u DEVICE`: 단일 GPU 경로는 `DEVICE`가 있으면 GPU 지정을 건너뛰므로(항상 GPU 0), 이를 제거합니다.
  4 GPU 경로에는 필요 없습니다.

### 4.3 환경변수

| 변수             | 기본값                        | 설명                                                         |
| ---------------- | ----------------------------- | ------------------------------------------------------------ |
| `POLICY_PATH`    | `./vla_models/pi05_base`      | 모델 경로                                                    |
| `GPU_IDS`        | `0`                           | 쉼표로 구분한 GPU 번호                                       |
| `TASK`           | 4개 suite                     | `libero_spatial,libero_object,libero_goal,libero_10` 중 선택 |
| `N_EPISODES`     | `10`                          | 태스크당 에피소드 수 (의미 있는 최대값 50)                   |
| `BATCH_SIZE`     | `1`                           | GPU당 동시 환경 수. 50의 약수(5, 10, 25) 권장                |
| `N_ACTION_STEPS` | `10`                          | 추론 한 번에 실행할 행동 수 (openpi LIBERO 설정)             |
| `OUTPUT_DIR`     | `./eval_logs/<모델명>_libero` | 결과 위치                                                    |
| `DEBUG`          | `0`                           | `1`이면 단일 GPU + `breakpoint()` 사용                       |

결과: `OUTPUT_DIR/eval_info.json` (4 GPU면 `shard_gpu*/`에 GPU별 로그·영상이 함께 저장)

### 4.4 GR00T N1.7 — `lerobot-eval` 직접 실행 (미검증)

GR00T 체크포인트는 두 가지를 덮어써야 합니다 (4.5 참고). 이 조합으로 성공률 평가를 끝까지 돌려 보지는 않았습니다.

```bash
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 uv run lerobot-eval \
  --policy.path=./vla_models/gr00t17_libero_spatial \
  --policy.base_model_path=nvidia/GR00T-N1.7-3B \
  --policy.device=cuda:0 \
  --env.type=libero --env.task=libero_spatial \
  --env.camera_name_mapping='{"agentview_image": "image", "robot0_eye_in_hand_image": "wrist_image"}' \
  --env.max_parallel_tasks=1 \
  --eval.batch_size=1 --eval.n_episodes=10 \
  --output_dir=./eval_logs/gr00t17_libero_spatial
```

suite별 체크포인트는 `--env.task`와 짝을 맞춥니다:
`gr00t17_libero_spatial`↔`libero_spatial`, `gr00t17_libero_object`↔`libero_object`,
`gr00t17_libero_goal`↔`libero_goal`, `gr00t17_libero_10`↔`libero_10`.

### 4.5 GR00T N1.7 체크포인트 주의사항

| 문제                                                                                                             | 해결                                                                                                  |
| ---------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `config.json`의 `base_model_path`가 학습 서버 경로(`/cache/huggingface/...`)라서 `HFValidationError`             | `--policy.base_model_path=nvidia/GR00T-N1.7-3B`                                                       |
| 체크포인트는 손목 카메라를 `observation.images.wrist_image`로 기대 (LIBERO 기본은 `image2`) → `Feature mismatch` | `--env.camera_name_mapping='{"agentview_image": "image", "robot0_eye_in_hand_image": "wrist_image"}'` |

---

## 5. 순차 추론: `lerobot-inference`

`lerobot-eval`에서 병렬 태스크, 비동기 환경, 녹화·Hub 업로드, 학습용 데이터 수집을 뺀 버전입니다.
**단일 GPU, 배치 1, 태스크와 에피소드를 하나씩 순서대로** 실행합니다. 추론 흐름을 디버깅할 때 쓰기 좋습니다.

```bash
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 uv run lerobot-inference \
  --policy.path=./vla_models/pi05_libero_finetuned \
  --policy.n_action_steps=10 --policy.device=cuda:0 \
  --env.type=libero --env.task=libero_object \
  --eval.n_episodes=2 \
  --output_dir=./eval_logs/inference_pi05

# 태스크 일부만:  --env.task_ids='[0,1]'
# 컴파일 끄기 (첫 추론 전 ~7분 컴파일 생략, 모델 내부 중단점 사용 가능):  --policy.compile_model=false
```

- CLI 인자는 `lerobot-eval`과 같습니다. `--eval.n_episodes`는 **태스크당** 에피소드 수입니다.
- `--eval.batch_size`, `--eval.use_async_envs`, `--env.max_parallel_tasks`는 무시되고 1 / 동기 환경으로 고정됩니다.
- `python -m lerobot.scripts.lerobot_inference ...`로도 실행할 수 있습니다.

결과 `OUTPUT_DIR/eval_info.json`:

- `per_episode`: 태스크·에피소드별 `success`, `n_steps`, `seed`, `episode_s`(에피소드 시간), `policy_s`(`select_action` 시간), `video_path`
- `per_task`, `per_group`, `overall`: 성공률과 95% Wilson 신뢰구간
- `videos/<suite>_<task>/eval_episode_<n>.mp4`: 태스크당 처음 10개 에피소드

---

## 6. 단계별 latency 프로파일링: `lerobot_profiling`

실제 관측 하나를 받아서 전처리한 뒤, `predict_action_chunk`만 `warmup + iters`번 반복하며
**vision encoder / LLM backbone / action expert** 단계별 wall-clock 시간과 Hz를 잽니다.
측정 루프 안에 `env.step`이 없어서 시뮬레이터나 로봇을 기다리는 시간은 포함되지 않습니다.

### 6.1 실행

```bash
# pi0.5
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=1,0 MUJOCO_EGL_DEVICE_ID=0 \
uv run python -m lerobot.scripts.lerobot_profiling \
  --policy.path=./vla_models/pi05_libero_finetuned \
  --policy.device=cuda:0 --policy.n_action_steps=10 \
  --env.type=libero --env.task=libero_object \
  --warmup=5 --iters=50 --output_dir=./eval_logs/profile_pi05

# GR00T N1.7
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=1,0 MUJOCO_EGL_DEVICE_ID=0 \
uv run python -m lerobot.scripts.lerobot_profiling \
  --policy.path=./vla_models/gr00t17_libero_spatial \
  --policy.base_model_path=nvidia/GR00T-N1.7-3B \
  --policy.device=cuda:0 \
  --env.type=libero --env.task=libero_spatial \
  --env.camera_name_mapping='{"agentview_image": "image", "robot0_eye_in_hand_image": "wrist_image"}' \
  --warmup=5 --iters=50 --output_dir=./eval_logs/profile_groot

# SmolVLA (체크포인트가 있으면 --policy.path=<경로>; 아래는 무작위 가중치로 구조만 측정)
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=1,0 MUJOCO_EGL_DEVICE_ID=0 \
uv run python -m lerobot.scripts.lerobot_profiling \
  --policy.type=smolvla --policy.device=cuda:0 \
  --env.type=libero --env.task=libero_object \
  --warmup=5 --iters=50 --output_dir=./eval_logs/profile_smolvla
```

### 6.2 측정 방식

- **wall 모드**: 단계 경계마다 `torch.cuda.synchronize()` + `time.perf_counter()`.
  파이썬 코드, 커널 실행 요청, GPU 실행을 모두 포함하고, 단계끼리 겹치지 않습니다.
- **eager 기준**: `torch.compile`은 단계 경계를 합쳐 버려서 자동으로 끕니다.
- **환경**: `--env.task_ids`를 주지 않으면 태스크 0번 환경만 만들어 관측 하나를 받습니다.
- **warmup**: 처음 `--warmup`번은 통계에서 제외합니다.

| 단계   | pi0.5                                                    | SmolVLA                                 | GR00T N1.7                                       |
| ------ | -------------------------------------------------------- | --------------------------------------- | ------------------------------------------------ |
| vision | `vision_tower` + `multi_modal_projector` (카메라마다)    | `vision_model` + `connector`            | Qwen3-VL `visual`                                |
| llm    | `paligemma_with_expert.forward`의 prefix 호출 (Gemma 2B) | `vlm_with_expert.forward`의 prefix 호출 | Qwen3-VL `language_model`                        |
| action | `denoise_step` × 10 (Gemma 300M expert)                  | `denoise_step` × 10                     | `action_head.get_action_with_features` (DiT × 4) |

다른 정책은 `lerobot_profiling.py`의 `STAGES`에 추가하면 됩니다. 등록되지 않은 정책은 전체 시간만 측정합니다.

### 6.3 출력

```
stage        mean ms      p50      p90  steps  ms/step      Hz  Hz/step  share
vision          44.6     44.6     44.8      3     14.9    22.4     67.3  14.8%
llm             78.2     78.3     78.4      1     78.2    12.8     12.8  25.9%
action         174.4    172.4    182.7     10     17.4     5.7     57.3  57.7%
other            4.9      4.8      5.4                                    1.6%
inference      302.1    299.9    311.1                     3.3
preprocess 2.6 ms | postprocess 0.1 ms | 33.1 actions/s (n_action_steps=10) | peak GPU mem 8.87 GB
```

| 열                        | 의미                                                                       |
| ------------------------- | -------------------------------------------------------------------------- |
| `mean ms` / `p50` / `p90` | 추론 한 번에 그 단계가 쓴 시간                                             |
| `steps`                   | 추론 한 번에 반복된 단위 수 (vision: 이미지 수, action: denoising 스텝 수) |
| `ms/step`, `Hz/step`      | 단위 하나 기준 시간과 Hz                                                   |
| `Hz`                      | 추론 한 번 기준 Hz = 1000 / mean ms                                        |
| `other`                   | 전체 − 단계 합 (임베딩, 마스크, Euler 업데이트 등)                         |
| `inference`               | `predict_action_chunk` 전체 = 모델 추론 Hz                                 |

모든 값은 `OUTPUT_DIR/profile.json`에 저장됩니다.

### 6.4 측정 예 (RTX 3090, 배치 1, eager)

|                 | pi0.5                | SmolVLA (무작위 가중치) | GR00T N1.7          |
| --------------- | -------------------- | ----------------------- | ------------------- |
| vision          | 44.6 ms              | 32.1 ms                 | 19.8 ms             |
| llm             | 78.2 ms              | 18.2 ms                 | 17.3 ms             |
| action          | 174.4 ms (10 × 17.4) | 170.4 ms (10 × 17.0)    | 89.5 ms (4 × 22.4)  |
| **추론 전체**   | **302 ms (3.3 Hz)**  | **226 ms (4.4 Hz)**     | **141 ms (7.1 Hz)** |
| 전처리          | 2.6 ms               | 2.0 ms                  | 14.1 ms             |
| 최대 GPU 메모리 | 8.9 GB               | 1.2 GB                  | 12.4 GB             |

---

## 7. VS Code 디버깅

`.vscode/launch.json`(워크스페이스가 `~/lerobot`일 때) 또는 `~/.vscode/launch.json`(홈 폴더를 열었을 때)에
설정을 두고 Run and Debug(`Ctrl+Shift+D`)에서 실행합니다. VS Code는 **열어 둔 폴더의 `.vscode/launch.json`만** 읽습니다.

```json
{
  "name": "LIBERO inference pi05 (no compile)",
  "type": "debugpy",
  "request": "launch",
  "module": "lerobot.scripts.lerobot_inference",
  "python": "${workspaceFolder}/.venv/bin/python",
  "cwd": "${workspaceFolder}",
  "console": "integratedTerminal",
  "justMyCode": false,
  "env": {
    "MUJOCO_GL": "egl",
    "CUDA_VISIBLE_DEVICES": "0",
    "MUJOCO_EGL_DEVICE_ID": "0"
  },
  "args": [
    "--policy.path=./vla_models/pi05_libero_finetuned",
    "--policy.n_action_steps=10",
    "--policy.device=cuda:0",
    "--policy.compile_model=false",
    "--env.type=libero",
    "--env.task=libero_object",
    "--eval.n_episodes=1",
    "--output_dir=./eval_logs/inference_debug_pi05"
  ]
}
```

- 홈 폴더를 워크스페이스로 열었다면 `python`과 `cwd`의 `${workspaceFolder}`를 `${workspaceFolder}/lerobot`로 바꿉니다.
- `lerobot-eval`로 디버깅하려면 `module`을 `lerobot.scripts.lerobot_eval`로 바꿉니다.
- 모델 내부(`modeling_pi05.py`의 `sample_actions`, `denoise_step` 등)에 중단점을 걸려면 `--policy.compile_model=false`가 필요합니다.

---

## 8. 자주 보는 현상

| 현상                                                                          | 원인 / 대응                                                                                                                            |
| ----------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| 모델 로딩(`make_policy`)이 약 100초                                           | 가중치를 불러오기 전에 모델 전체를 CPU에서 무작위 초기화합니다 (`pi_gemma.py`의 래퍼가 백본을 여러 번 생성). 정상 동작                 |
| 첫 추론 전 약 7분 멈춤                                                        | pi0.5 체크포인트는 `compile_model: true`, `compile_mode: max-autotune`. 첫 호출에서 컴파일. `--policy.compile_model=false`로 생략 가능 |
| `OutOfMemoryError: out of resource: triton_mm ... Hardware limit:101376` 로그 | `max-autotune`이 RTX 3090 공유 메모리 한도를 넘는 커널 후보를 건너뛰는 로그. 결과에 영향 없음                                          |
| `libero` import 시 입력 대기                                                  | `~/.libero/config.yaml`이 없음 → `./vla_benchmarks/download_libero.sh sim`                                                             |
| `No module named 'libero'` / `transformers`                                   | 2.1의 `uv sync` 실행                                                                                                                   |
| `DEBUG=1`인데 항상 GPU 0에서 실행                                             | `/etc/environment`의 `DEVICE="cuda"` 때문 → `env -u DEVICE`                                                                            |
| LIBERO 이미지가 180° 회전되어 있음                                            | 시뮬레이터 원본이 뒤집혀 나오고, 학습 데이터(HuggingFaceVLA/libero)는 똑바로 저장됨. `LiberoProcessorStep`이 자동으로 회전             |
| 4 GPU 평가에서 `BATCH_SIZE`를 키워도 느림                                     | 환경 하나당 CPU 코어 약 1개 사용 (서버 36코어). 4 × `BATCH_SIZE`가 코어 수를 넘으면 CPU가 병목                                         |
