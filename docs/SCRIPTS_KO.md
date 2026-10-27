# 실행 스크립트·도구 매뉴얼

repo 루트에서 실행한다고 가정한다. 모든 서버 스크립트는 자기 위치(`scripts/`)를 기준으로 bundle 경로를 잡으므로, 어느 디렉터리에서 호출해도 된다.

| 분류 | 파일 | 용도 | GPU |
|---|---|---|---|
| 서버 | [`scripts/run_server_async.sh`](#run_server_asyncsh) | 추론 서버 (비동기, **실차 권장**) | 2장 |
| | [`scripts/run_server.sh`](#run_serversh) | 추론 서버 (직렬) | 1장 |
| | [`scripts/run_ui.sh`](#run_uish) | 브라우저 검증 UI | – |
| weight | [`tools/verify_weights.py`](#verify_weightspy) | 받은 weight의 SHA256 검증 | – |
| | [`tools/validate_release.py`](#validate_releasepy) | 모델 파일·S2/S1 호환성 점검 | – |
| | [`tools/upload_weights_gdrive.sh`](#upload_weights_gdrivesh) | weight를 Google Drive에 업로드 | – |
| 클라이언트 | [`tools/client_example.py`](#client_examplepy) | 이미지 1장 요청 | – |
| | [`tools/vehicle_client_example.py`](#vehicle_client_examplepy) | 2 Hz 차량 클라이언트 예제 | – |
| | [`ros2/dualvln_ros`](ROS2_KO.md) | ROS 2 노드 (별도 문서) | – |
| 평가 | [`tools/shadow_replay.py`](#shadow_replaypy) | 녹화 시퀀스를 서버에 재생하는 shadow test | 서버 |
| | [`tools/render_sequence.py`](#render_sequencepy) | 프레임별 시각화·GIF | 1장 |
| | [`reports/make_figures.py`](#make_figurespy) | 리포트 데이터셋·타임라인 그림 | – |
| 프로파일 | [`tools/profile_latency.py`](#profile_latencypy) | 단계별 latency (단일 프레임) | 1장 |
| | [`tools/profile_detailed.py`](#profile_detailedpy) | 단계별 latency (조건별 연속 프레임) | 1장 |
| 배포 | [`tools/build_portable_bundle.py`](#build_portable_bundlepy) | 학습 서버에서 휴대용 bundle 생성 | – |

공통 사항:

- Python은 `requirements-inference.txt`를 설치한 환경을 쓴다. 다른 인터프리터로 서버를 돌리려면 `PYTHON_BIN=/path/to/python`을 지정한다.
- 평가·프로파일 도구의 기본 데이터 경로(`/home/work/.datasets/VLA-E2E/...`, `.../outputs/s2_train_data_reviewed_landmark/conversations_val.json`)는 **학습 서버 기준**이다. 다른 장비에서는 `--manifest` 같은 인자로 경로를 바꿔야 하고, 인자가 없는 도구(`profile_detailed.py`)는 코드 상단 상수를 고쳐야 한다.
- GPU를 쓰는 도구는 서버와 같은 GPU에서 동시에 돌리지 않는다. `CUDA_VISIBLE_DEVICES`로 나눈다.

---

## 서버

### run_server_async.sh

S2를 별도 프로세스(GPU0)에서, S1을 요청 경로(GPU1)에서 돌리는 비동기 서버. gunicorn worker 1개, thread 1개로 실행한다.

```bash
./scripts/run_server_async.sh
HOST=0.0.0.0 PORT=5801 S2_DEVICE=cuda:0 S1_DEVICE=cuda:1 ./scripts/run_server_async.sh   # 외부 접속 허용
```

| 환경변수 | 기본값 | 설명 |
|---|---|---|
| `PYTHON_BIN` | `python3` | 실행 인터프리터 |
| `HOST` | `127.0.0.1` | bind 주소. 차량 PC에서 접속하려면 `0.0.0.0` |
| `PORT` | `5801` | |
| `S2_DEVICE` | `cuda:0` | S2 (Qwen2.5-VL) GPU |
| `S1_DEVICE` | `cuda:1` | S1 (InternVLA-N1) GPU |
| `MAX_GOAL_AGE_S` | `2.25` | S2 goal이 이보다 오래되면 `stale_s2_goal` STOP |
| `S1_GPU_PREPROCESS` | `1` | S1 이미지 전처리를 GPU에서 수행. `0`(CPU)이면 S1이 500 ms를 넘긴다 |

- 로딩 약 50 s. `curl http://127.0.0.1:5801/ready`가 200이면 준비 완료(로딩 중에는 503 또는 응답 대기).
- GPU 메모리: GPU0 약 16.6 GB, GPU1 약 16.7 GB.
- 엔드포인트: `GET /health`, `GET /ready`, `POST /reset`, `POST /infer` (API는 [REALTIME_VEHICLE_KO.md](REALTIME_VEHICLE_KO.md) §5).
- 종료: `Ctrl-C` 또는 gunicorn master PID에 `kill`. S2 하위 프로세스도 같이 종료된다.
- 첫 S1 호출은 1 s 이상 걸린다. 주행 전에 warm-up 요청을 보낸다(클라이언트 예제와 ROS 노드는 자동으로 보냄).

### run_server.sh

S2와 S1을 한 GPU에서 순서대로 돌리는 직렬 서버. 응답 p50 약 1.42 s로 2 Hz를 못 맞춘다. 디버깅, 오프라인 평가, GPU가 1장일 때 쓴다.

```bash
DEVICE=cuda:0 ./scripts/run_server.sh
```

| 환경변수 | 기본값 | 설명 |
|---|---|---|
| `PYTHON_BIN` | `python3` | |
| `HOST` / `PORT` | `127.0.0.1` / `5801` | |
| `DEVICE` | `cuda:0` | S2·S1 모두 이 GPU (약 33 GB) |

엔드포인트와 응답 형식은 비동기 서버와 같다. 다만 `goal_age_ms`, `s1_latency_ms` 같은 비동기 전용 필드가 없다.

### run_ui.sh

검증 데이터 샘플을 고르거나 이미지를 올려서 S2 goal과 S1 궤적 overlay를 보는 브라우저 UI. **추론 서버를 먼저 띄워야 한다.**

```bash
./scripts/run_server_async.sh &                 # 또는 run_server.sh
INFERENCE_URL=http://127.0.0.1:5801/infer ./scripts/run_ui.sh
# 브라우저: http://127.0.0.1:5803
```

| 환경변수 | 기본값 | 설명 |
|---|---|---|
| `PYTHON_BIN` | `python3` | |
| `UI_HOST` / `UI_PORT` | `127.0.0.1` / `5803` | |
| `INFERENCE_URL` | `http://127.0.0.1:5801/infer` | 추론 서버 `/infer` 주소 |

원격 서버의 UI를 볼 때는 SSH 포트 포워딩을 쓴다: `ssh -L 5803:127.0.0.1:5803 <server>`.

---

## Weight

### verify_weights.py

`SHA256SUMS`의 30개 파일(`models/system1`, `models/system2`, `checkpoints/`)을 해시로 검증한다. Drive에서 받은 직후 실행한다. 약 32 GB를 읽으므로 수 분 걸린다.

```bash
python3 tools/verify_weights.py
# [1/30] ok       checkpoints/depth_anything_v2_metric_hypersim_vits.pth ... all 30 files verified
```

누락이나 불일치가 있으면 목록을 출력하고 exit 1로 끝난다.

### validate_release.py

서버를 띄우기 전에 다음을 점검하고 결과 JSON을 출력한다(실패 시 예외).

- 각 모델의 필수 파일과 shard가 있는지
- DepthAnything checkpoint가 있는지
- S2/S1의 `vocab_size`, `image_token_id`, `eos_token_id`가 같은지
- 런타임 계약의 궤적 좌표축이 맞는지

```bash
python3 tools/validate_release.py              # 점검만
python3 tools/validate_release.py --write-lock # config/release_lock.json 갱신 (릴리스 담당자용)
```

### upload_weights_gdrive.sh

rclone으로 `models/`, `checkpoints/`, `SHA256SUMS`를 Drive에 올리고 `rclone check`로 검증한다. 릴리스 담당자용이다. rclone 설정은 [WEIGHTS_KO.md](WEIGHTS_KO.md)를 본다.

```bash
tools/upload_weights_gdrive.sh [remote=gdrive] [folder=dualvln-vla-e2e-20261021-rc3]
RCLONE=/path/to/rclone tools/upload_weights_gdrive.sh
```

---

## 클라이언트

### client_example.py

이미지 1장으로 `/infer`를 한 번 호출해 응답 JSON을 출력한다. 서버 동작을 확인할 때 쓴다.

```bash
python3 tools/client_example.py --image front.jpg --episode-id test-1 --reset \
    --command follow --instruction "도로를 따라 주행하라"
```

| 인자 | 필수 | 설명 |
|---|---|---|
| `--url` | | 기본 `http://127.0.0.1:5801/infer` |
| `--image` | ✓ | 전방 1920×1080 JPEG |
| `--lookdown` | | 별도 lookdown 이미지 (생략 권장, 서버가 하단 크롭으로 생성) |
| `--episode-id` | ✓ | |
| `--instruction` | ✓ | 학습 문장 그대로 ([REALTIME_VEHICLE_KO.md](REALTIME_VEHICLE_KO.md) §4) |
| `--command` | ✓ | `follow` `turn_left` `turn_right` `safe_stop` `stop` |
| `--reset` | | 새 episode 시작 |

비동기 서버에서는 첫 요청에 goal이 아직 없어서 `awaiting_s2_goal`이 나온다. 같은 episode로 1–2 s 뒤에 다시 보내면 궤적이 나온다.

### vehicle_client_example.py

차량 쪽 2 Hz 루프의 참조 구현이다. 카메라 → `/infer` → 제어기 순서로 동작하고, watchdog과 warm-up을 포함한다. `FrameSource`와 `Controller` 클래스를 실제 카메라 드라이버와 경로 추종기로 바꿔서 쓴다. ROS 2 차량에서는 이 대신 `ros2/dualvln_ros`를 쓴다.

```bash
python3 tools/vehicle_client_example.py --server http://<gpu-server>:5801 \
    --replay-dir /home/work/.datasets/VLA-E2E/20260515/undistort_orin_pc_data11/camera/rgb_front \
    --command follow
```

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--server` | `http://127.0.0.1:5801` | |
| `--command` | `follow` | |
| `--instruction` | command별 학습 문장 | `safe_stop`은 필수 |
| `--episode-id` | `drive-<시각>` | |
| `--rate-hz` | `2.0` | |
| `--timeout-s` | `0.8` | HTTP timeout. 넘으면 직전 경로를 유지하고 watchdog이 판단 |
| `--watchdog-s` | `1.0` | 유효 궤적이 이 시간 동안 없으면 STOP |
| `--replay-dir` | | 카메라 대신 JPEG 폴더 재생 (bench test) |
| `--warmup` | `6` | 주행 전 warm-up 요청 수 |

출력: 프레임마다 `http=… status=… rtt=… server=… goal_age=…` 한 줄과 `-> follow path …` 또는 `-> STOP (…)`.

---

## 평가

### shadow_replay.py

검증 episode를 프레임 순서대로 `/infer`에 보내는 shadow test. 서버가 history와 safe_stop prior를 실제 차량과 똑같이 쌓는다. 제어 출력은 기록만 한다.

```bash
# 서버 먼저 실행
python3 tools/shadow_replay.py --out outputs/shadow_$(date +%Y%m%d) --rate-hz 2       # 실시간 2 Hz
python3 tools/shadow_replay.py --out /tmp/s --episodes undistort_orin_pc_data901 --mode reset_each
```

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--url` | `http://127.0.0.1:5801` | |
| `--manifest` | 학습 서버의 `conversations_val.json` | episode·GT goal 출처 |
| `--episodes` | 전체 val episode | 일부만 돌릴 때 |
| `--mode` | `closed_loop` | `closed_loop`: episode당 reset 1번 / `reset_each`: 매 프레임 reset (history 없는 비교군) |
| `--max-frames` | 전체 | |
| `--rate-hz` | 없음 (응답 오면 바로 다음) | 실시간 스케줄로 보냄. **비동기 서버는 반드시 지정** |
| `--timeout` | `120` | 요청 timeout (s) |
| `--out` | (필수) | 결과 폴더 |

출력: `<mode>_<episode>.jsonl`(프레임별 응답), `summary_<mode>.json`(goal 오차 px, goal jump, 상태 분포, latency).

### render_sequence.py

서버 없이 런타임을 직접 올려서 episode의 각 프레임을 추론하고 그림으로 남긴다. 그림에는 전방 영상 위의 S2 goal(주황), GT goal(노랑), S1 후보(연빨강), 선택 궤적(자홍)과 top-down 패널이 들어간다.

```bash
CUDA_VISIBLE_DEVICES=0 python3 tools/render_sequence.py \
    --episodes undistort_orin_pc_data11 undistort_orin_pc_data901 --out /tmp/seq --gif
```

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--episodes` | (필수) | episode 이름 여러 개 |
| `--manifest` | shadow_replay와 같음 | |
| `--device` | `cuda:0` | S2·S1 모두 (약 33 GB) |
| `--max-frames` | 전체 | |
| `--out` | (필수) | `<episode>/0000.jpg…`, `<episode>.jsonl` |
| `--gif` | | `<episode>.gif` 추가 생성 (2프레임마다 1장, 2/3 크기) |

직렬 런타임으로 돌리므로, 결과는 비동기 서버의 지연(goal age) 효과가 없는 상태의 값이다.

### make_figures.py

`reports/figures/`의 `dataset_trajectories.png`, `dataset_commands.png`, `async_timeline.png`를 학습 서버 데이터로 다시 만든다.

```bash
python3 reports/make_figures.py --project /home/work/vln_project/vla_ws_restore/vla_project
```

---

## 프로파일

### profile_latency.py

연속 프레임으로 history를 채운 뒤, 마지막 프레임의 `infer`를 단계별로 잰다(전처리, S2 ViT, S2 generate, S1 latent, S1 diffusion). CUDA synchronize 타이머를 쓰고, 추론 결과는 바뀌지 않는다.

```bash
CUDA_VISIBLE_DEVICES=0 python3 tools/profile_latency.py \
    --frames /path/rgb_front/000{30..38}.jpg --instruction 좌회전하라 --command turn_left --repeats 3
```

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--bundle` | 이 repo | |
| `--frames` | (필수) | 연속 front 이미지들 |
| `--instruction` / `--command` | `좌회전하라` / `turn_left` | |
| `--repeats` | `3` | 측정 반복 (warm-up 1회 별도) |
| `--out` | | JSON 저장 경로 |

### profile_detailed.py

세 조건(`hist0`: history 없음, `hist8`: history 8장, `safe_prior`: safe_stop prior)에서 연속 프레임으로 단계별 latency의 mean, p50, p95를 잰다. 런타임 버전을 서로 비교할 때 쓴다.

```bash
CUDA_VISIBLE_DEVICES=0 python3 tools/profile_detailed.py \
    --runtime-dir runtime --label rc3 --measure 12 --out /tmp/profile_rc3.json
```

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--bundle` | 이 repo | 모델 위치 |
| `--runtime-dir` | (필수) | 측정할 `dual_model_runtime.py`가 있는 폴더 |
| `--label` | (필수) | 결과 라벨 |
| `--measure` | `12` | 조건별 측정 프레임 수 |
| `--out` | (필수) | |

입력 프레임 경로(`DRIVE`, `PARK`)는 코드 상단에 고정돼 있다(data11, data901).

---

## 배포

### build_portable_bundle.py

**학습 서버 전용.** 프로젝트 checkout(`outputs/`의 모델, `InternNav/`, `scripts/_fix_nan_params.py`, `checkpoints/`)에서 독립 실행 가능한 bundle 폴더를 만든다. 지금의 rc3 repo와 Drive weight가 이 결과물이다.

```bash
python3 tools/build_portable_bundle.py /path/to/empty_dir
```

대상 폴더는 없거나 비어 있어야 한다. 복사한 파일 목록은 `config/copied_files.json`에 남는다.
이 도구는 `SHA256SUMS`를 만들지 않으므로, 만든 뒤 bundle 폴더에서 직접 갱신하고 lock을 다시 쓴다.

```bash
cd /path/to/bundle
sha256sum models/system1/* models/system2/* checkpoints/*.pth > SHA256SUMS
python3 tools/verify_weights.py && python3 tools/validate_release.py --write-lock
```

---

## 처음 설치부터 실차 연결까지 순서 요약

```bash
git clone git@github.com:Furive-zo/DualVLN-FR09.git dualvln && cd dualvln
pip install -r requirements-inference.txt          # NVIDIA PyTorch 먼저
# Drive에서 models/, checkpoints/ 받기 (docs/WEIGHTS_KO.md)
python3 tools/verify_weights.py
python3 tools/validate_release.py
HOST=0.0.0.0 ./scripts/run_server_async.sh         # /ready 200 확인
python3 tools/vehicle_client_example.py --replay-dir <녹화 폴더>   # bench 확인
# 차량: ros2/dualvln_ros 빌드 후 ros2 launch (docs/ROS2_KO.md)
```
