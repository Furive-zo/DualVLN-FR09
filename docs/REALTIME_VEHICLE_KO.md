# 실시간 실차 추론 가이드

대상: `dualvln-vla-e2e-20261021-rc3`, 비동기 런타임(S2 GPU 0 / S1 GPU 1)

> **현재 운용 등급: shadow mode.** 차량은 기존 제어기 또는 안전 운전자가 몰고, 모델 출력은 기록·비교만 한다.
> 실제 조향/가감속 연결은 폐쇄구역 저속 검증과 safe_stop gate 통과 뒤에 한다.
> 특히 `safe_stop`은 gate 미통과(주차면 목표 median 오차 203–1,177 px)이므로 실차 제어에 쓰지 않는다.

## 1. 구성

```
 차량 PC                                       추론 서버 (GPU 2장)
 ┌───────────────────────────┐   HTTP POST   ┌───────────────────────────────┐
 │ 전방 카메라 → 왜곡 보정    │ ─ /infer ───▶ │ gunicorn (1 worker)           │
 │ 명령/지시문 (운전자·상위)  │   2 Hz        │  ├ S1 요청 경로 (GPU 1) ~360ms │
 │ 2 Hz 클라이언트 + watchdog │ ◀─ JSON ───── │  └ S2 서브프로세스 (GPU 0) ~0.9s│
 │ 경로 추종 제어기           │               │     최신 프레임으로 goal 갱신   │
 └───────────────────────────┘               └───────────────────────────────┘
```

- 요청 한 번 = 프레임 한 장. 서버는 그 프레임으로 **S1 궤적을 즉시** 만들고(약 360 ms), 같은 프레임을 S2에 넘겨
  다음 goal을 비동기로 갱신한다. S1은 가장 최근 S2 goal(보통 1–2 s 전 프레임)을 쓴다.
- 서버는 **차량 한 대의 episode 상태**(전방 history, safe_stop 이전 goal)를 보관한다. 서버 하나에 차량 하나.

## 2. 서버 켜기

### 2.1 준비 (최초 1회)

```bash
git clone git@github.com:Furive-zo/DualVLN-FR09.git dualvln && cd dualvln
pip install -r requirements-inference.txt          # NVIDIA PyTorch 먼저 (검증: torch 2.10.0a0 nv25.11)
# weight: docs/WEIGHTS_KO.md 대로 models/, checkpoints/ 받기
python3 tools/verify_weights.py && python3 tools/validate_release.py
```

### 2.2 실행

```bash
PYTHON_BIN=/path/to/python HOST=0.0.0.0 PORT=5801 \
S2_DEVICE=cuda:0 S1_DEVICE=cuda:1 ./scripts/run_server_async.sh
```

| 환경 변수 | 기본값 | 설명 |
|---|---|---|
| `HOST` / `PORT` | `127.0.0.1` / `5801` | 차량 PC가 직접 붙으면 `0.0.0.0`. 외부망 노출 금지(인증 없음) |
| `S2_DEVICE` / `S1_DEVICE` | `cuda:0` / `cuda:1` | 두 모델을 다른 GPU에 둔다. 같은 GPU면 S1이 S2와 경합해 느려짐 |
| `MAX_GOAL_AGE_S` | `2.25` | 이보다 오래된 S2 goal이면 STOP(`stale_s2_goal`) |
| `S1_GPU_PREPROCESS` | `1` | S1 이미지 전처리를 GPU에서. `0`이면 S1이 500 ms를 넘김 |
| `PYTHON_BIN` | `python3` | 가상환경 python |

- 모델 로드: H200 ×2, 로컬 SSD 기준 **약 50 s**(디스크 캐시가 비어 있으면 더 걸린다). 다음이 200이 될 때까지 기다린다.
  ```bash
  curl -s http://<server>:5801/ready     # {"ready": true, ...}
  ```
- GPU 메모리(실측): GPU 0(S2) 약 16.6 GB, GPU 1(S1) 약 16.7 GB. 다른 작업과 GPU를 공유하면 latency가 크게 늘어난다.
- **CPU 여유도 필요하다.** S2 전처리와 HTTP 이미지 처리는 CPU를 쓴다. 서버 CPU가 다른 작업으로 붐비면 S2 goal이 늦어져 STOP이 늘어난다(8절).
- **Warm-up**: 서버를 켠 직후 첫 S1 호출은 CUDA 초기화로 1 s 이상 걸린다. 주행 전에 아무 프레임으로 5–6회
  (0.5 s 간격, 긴 timeout) 호출해 둔다. 예제 클라이언트는 `--warmup 6`(기본값)으로 이것을 자동으로 한다.
  실제 주행은 그다음 새 `episode_id` + `reset=true`로 시작한다.
- 종료: `Ctrl+C` 또는 gunicorn PID에 SIGTERM.

## 3. 카메라 입력 조건

모델은 아래 카메라로만 학습됐다. 다르면 성능을 보장하지 않는다.

| 항목 | 학습 데이터 |
|---|---|
| 카메라 | 전방 1대 `CAM_FRONT`, 1920×1080 RGB |
| 왜곡 보정 | OpenCV plumb_bob, 보정 후 K = [[983.5, 0, 1006.9], [0, 984.8, 490.5], [0, 0, 1]], alpha 0 |
| 장착 | 차량 기준 전방 1.03 m, 높이 0.92 m, pitch −1.25° |
| 주기 | **2 Hz (0.5 s)** — 서버는 받은 프레임을 최대 64장(32 s) 보관하고 그중 8장을 고르게 뽑아 S2에 넣는다 |
| 포맷 | JPEG(quality 90 이상 권장) |

- 보정되지 않은 원본, 다른 해상도, 다른 화각을 보내지 않는다. 해상도가 다르면 1920×1080로 맞춘 뒤 보낸다.
- `lookdown` 이미지는 보내지 않아도 된다(서버가 전방 이미지 하단 60%를 잘라 만든다. 학습도 이 방식).
- 2 Hz보다 빠르게 보내면 history가 짧은 시간만 덮게 되고, 느리게 보내면 goal 나이가 커져 STOP이 늘어난다.
  **2 Hz 고정**을 권장한다. 차량이 정지해 있어도 프레임은 계속 보낸다.

## 4. 지시문 넣기

### 4.1 원칙

- 모델이 실제로 읽는 것은 **`instruction` 문장**이다. `command`는 서버 동작 분기용이다
  (`stop`이면 모델을 거치지 않고 정지, `safe_stop`이면 이전 goal 추적 사용).
- 학습 데이터의 지시문은 아래 **고정 문장**뿐이다(`config/runtime_contract.json`의 `trained_instructions`에도 있다). 다른 표현(예: "앞으로 가", "turn left")은 학습 분포 밖이라
  결과를 보장하지 않는다. **아래 문장을 글자 그대로** 쓴다.
- `command`와 `instruction`은 항상 짝을 맞춘다. 어긋나면 모델은 문장을 따른다.

| command | instruction (그대로 사용) | 서버 동작 |
|---|---|---|
| `follow` | `도로를 따라 주행하라` | S2 goal → S1 궤적 |
| `turn_left` | `좌회전하라` | S2 goal → S1 궤적 |
| `turn_right` | `우회전하라` | S2 goal → S1 궤적 |
| `safe_stop` | 아래 4.2 | S2 goal(이전 goal 추적) → S1 주차 진입 궤적 |
| `stop` | `정지하라` | 모델 우회, 즉시 `discrete_action=0` |

### 4.2 safe_stop(주차면 지정) 문장

학습에 쓰인 형식: **"<기준>에서 <N> 번째 빈 주차면에 안전하게 진입하여 정차하라"**

```
왼쪽에서 첫 번째 빈 주차면에 안전하게 진입하여 정차하라
왼쪽에서 두 번째 빈 주차면에 안전하게 진입하여 정차하라
오른쪽에서 두 번째 빈 주차면에 안전하게 진입하여 정차하라
오른쪽에서 세 번째 빈 주차면에 안전하게 진입하여 정차하라
오른쪽에서 네 번째 빈 주차면에 안전하게 진입하여 정차하라
왼쪽에 주차된 차량 기준 2칸 왼쪽의 빈 주차면에 안전하게 진입하여 정차하라
```

- 같은 주차 시도 동안에는 **문장을 바꾸지 않는다.** 문장이 바뀌면 서버는 이전 goal 추적을 버리고 새로 찾는다.
- 순서 표현(첫/두/세/네 번째)은 진입 시점 카메라 기준이다. 학습 장소(사내 주차장) 밖에서는 검증되지 않았다.

### 4.3 명령 전환

- 같은 주행 중 명령이 바뀌면(예: `follow` → `turn_left`) **같은 `episode_id`, `reset=false`** 로 새 명령·문장을 보낸다.
  전방 history는 유지된다.
- 비동기 서버는 새 명령의 S2 goal이 나올 때까지 약 1 s(2 프레임) 동안 `awaiting_s2_goal` STOP을 돌려준다.
  교차로 진입 전에 **미리** 명령을 바꿔 보내야 한다.
- 명령을 프레임마다 바꾸지 않는다(매번 1 s 대기가 생긴다).

### 4.4 Episode와 reset

| 상황 | episode_id | reset |
|---|---|---|
| 주행(또는 주차 시도) 시작 | 새 ID (예: `drive-20261021-1530`) | `true` |
| 주행 중 매 프레임 | 같은 ID | `false` |
| 명령/지시문 변경 | 같은 ID | `false` |
| 다른 장소로 이동, 차량 재시동, 서버 재시작 후 | 새 ID | `true` |

- reset 없이 다른 `episode_id`를 보내면 **409 + STOP**이 온다(이전 주행 history와 섞이는 것을 막는 계약).

## 5. API

### 요청

`POST /infer`, `multipart/form-data`

| 필드 | 타입 | 필수 | 내용 |
|---|---|---|---|
| `front` | file (JPEG) | ✓ | 왜곡 보정된 전방 1920×1080 |
| `json` | text | ✓ | `{"episode_id", "instruction", "command", "reset"}` |
| `lookdown` | file | | 생략 권장 |

```bash
curl -s http://<server>:5801/infer \
  -F front=@front.jpg \
  -F 'json={"episode_id":"drive-001","instruction":"도로를 따라 주행하라","command":"follow","reset":true}'
```

### 응답 (200)

```json
{
  "control_decision": "trajectory",
  "status": "coordinate",
  "s2_text": "951 287",
  "pixel_goal_lookdown": [951, 287],
  "trajectory": {
    "selected": [[x, y, θ], [x, y, θ], [x, y, θ], [x, y, θ]],
    "selected_index": 2,
    "candidates": [[[x, y, θ], ...], ...],
    "valid_candidate_mask": [true, true, true, false],
    "mean": [[x, y, θ], ...]
  },
  "latency_ms": 352.1, "s1_latency_ms": 341.0, "s2_latency_ms": 930.4,
  "goal_age_ms": 1512.3, "goal_frame_seq": 41, "frame_seq": 44,
  "dropped_s2_frames": 12, "mode": "async",
  "episode_id": "drive-001", "release": "dualvln-vla-e2e-20261021-rc3"
}
```

### 궤적 좌표계

- `trajectory.selected`: 차량 기준 4개 누적 pose `[x, y, θ]`. **요청 프레임을 찍은 순간의 차량 위치·방향**이 원점.
- **전진 = −y, 오른쪽 = +x** (단위 m). 예: `[0.3, -4.5, …]` = 4.5 m 앞, 0.3 m 오른쪽.
- θ: 각 점에서의 진행 방향(rad), **+x(오른쪽) 축에서 반시계 방향으로 잰 각도**. 직진 = −π/2(−90°).
  좌회전이면 −π/2보다 작아지고(−100°, −110° …), 우회전이면 커진다(−80°, −70° …).
  차량 yaw 변화(좌회전 +)로 바꾸려면 `yaw = -(θ + π/2)`.
- 4점 간격은 약 1.2–1.3 m(마지막 0.8–1.0 m), 끝점은 약 4.3–4.7 m 앞(data11 재생 실측). **시간 정보가 없는 경로**이므로 속도는 제어기가 정한다.
- 응답이 도착할 때 차량은 이미 움직였다(약 0.4 s). 오도메트리로 `captured_at` 이후 이동을 빼서 경로를 현재 차량 기준으로 옮긴다.
- 서버는 후진 성분이 있거나 끝점이 차량 뒤인 후보를 제외하고(`valid_candidate_mask`), 남은 후보 중 평균에 가장 가까운 것을 `selected`로 고른다.

### 상태값

| `status` | `control_decision` | 의미 | 클라이언트 동작 |
|---|---|---|---|
| `coordinate` | `trajectory` | 정상 | 경로 추종 |
| `stop` | `stop` | `command=stop` 또는 S2가 STOP 판단 | 정지 |
| `awaiting_s2_goal` | `stop` | 주행 시작·명령 변경 직후 goal 없음(약 1 s) | 정지 유지 |
| `stale_s2_goal` | `stop` | goal이 `MAX_GOAL_AGE_S`보다 오래됨 (S2 지연) | 정지, 반복되면 서버 점검 |
| `unsafe_trajectory` | `stop` | 유효 후보 없음 | 정지 |
| `invalid` | `stop` | S2 출력이 좌표가 아님 | 정지 |
| `s2_process_dead` | `stop` | S2 프로세스 종료 | 정지, 서버 재시작 |
| `runtime_error` (500) | `stop` | 예외 | 정지, 서버 로그 확인 |
| `not_ready` (503) | `stop` | 모델 로딩 중 | `/ready` 대기 |
| HTTP 409 | `stop` | reset 없이 episode 변경 | 새 주행이면 `reset=true`로 재전송 |
| HTTP 400 | `stop` | 필드 누락·명령 오타 | 요청 수정 |
| (timeout) | – | 응답이 0.8 s 안에 안 옴 | 직전 경로 유지, watchdog 1.0 s 넘으면 정지 |

## 6. 차량 측 안전 규칙

1. **`control_decision == "trajectory"`인 200 응답만** 경로로 쓴다. 그 외 모든 경우는 정지.
2. **Watchdog**: 마지막 유효 궤적 이후 1.0 s 안에 새 유효 궤적이 없으면 정지(네트워크 끊김·서버 정지 대비).
3. HTTP timeout은 약 0.8 s로 두고, 응답을 기다리는 동안은 직전 유효 경로를 계속 추종하되 watchdog(2번)이 넘으면 정지한다.
   응답에는 그 요청 프레임 기준 경로가 오므로, 늦게 온 응답일수록 오도메트리 보정이 커진다.
4. 속도 상한과 경로 길이 상한(끝점 6 m 이내 등)을 제어기에서 따로 건다. 모델 출력만 믿지 않는다.
5. 수동 개입(브레이크·E-stop)이 항상 우선한다. 안전 운전자 탑승.
6. 로그: 요청 시각, 응답 전체 JSON, 실제 차량 pose를 같이 저장한다(사후 분석·재현용).

## 7. 예제 클라이언트

`tools/vehicle_client_example.py` — 2 Hz 루프, watchdog, 응답 처리까지 들어 있다.
카메라(`FrameSource`)와 제어기(`Controller`)만 차량 코드로 바꾸면 된다.

```bash
# 녹화 폴더로 서버와 벤치 테스트
python3 tools/vehicle_client_example.py --server http://<server>:5801 \
  --replay-dir /path/to/camera/rgb_front --command turn_left
# safe_stop은 문장을 직접 지정
python3 tools/vehicle_client_example.py --server http://<server>:5801 --replay-dir ... \
  --command safe_stop --instruction "오른쪽에서 두 번째 빈 주차면에 안전하게 진입하여 정차하라"
```

녹화 episode 전체를 실시간 속도로 재생해 서버를 검증하려면:

```bash
python3 tools/shadow_replay.py --url http://<server>:5801 --rate-hz 2 --out /tmp/shadow
```

출력 가시화(모델을 직접 돌려 프레임별 그림·GIF 생성, GPU 1장 사용):

```bash
python3 tools/render_sequence.py --episodes undistort_orin_pc_data11 --out /tmp/seq --gif
```

## 8. 성능 기준 (H200 ×2, 2 Hz 재생)

| 항목 | 값 |
|---|---|
| 응답 시간 p50 / p95 | 363 / 413 ms |
| S2 goal 주기 | 약 0.94 s |
| goal 나이 p50 / p95 | 1.51 / 2.00 s |
| 명령 유지 구간 trajectory 비율 | 95.8% |
| 명령 변경 직후 STOP | 약 1 s |

예제 클라이언트 벤치 테스트(2026-10-21, 같은 서버, 다른 사용자 작업으로 호스트 CPU load average 약 57/40코어):

| 항목 | 값 |
|---|---|
| 서버 처리 시간 p50 | 347 ms |
| 클라이언트 왕복(JPEG 인코딩·전송 포함) p50 / p90 / max | 377 / 533 / 620 ms |
| goal 나이 p50 / p90 | 1.96 / 2.48 s |
| 40프레임 중 | 궤적 32, goal 대기 2(시작), stale STOP 6 |

CPU가 붐비면 S2가 늦어져 goal 나이가 2.25 s 상한을 넘는 STOP(`stale_s2_goal`)이 늘어난다.
**차량용 추론 서버는 다른 작업과 공유하지 않는다.** 배포 서버에서 `tools/shadow_replay.py --rate-hz 2`로 먼저 재측정한다.
