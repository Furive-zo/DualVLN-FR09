# 모델 입출력 (S2 / S1)

rc3 런타임(`runtime/dual_model_runtime.py`, `runtime/async_runtime.py`)이 각 모델에 실제로 넣는 입력과 받는 출력.
서버 API(HTTP)는 [REALTIME_VEHICLE_KO.md](REALTIME_VEHICLE_KO.md) §5, ROS 2 연동은 [ROS2_KO.md](ROS2_KO.md) 참고.

## 0. 전체 흐름

```
전방 카메라 1920×1080 (왜곡 보정, 2 Hz)
   │
   ├─ history (최근 64장 보관) ──┐
   ├─ pseudo-lookdown (하단 60% 크롭 → 1920×1080 리사이즈)
   │                             ▼
   │   S2  Qwen2.5-VL-7B  ── 지시문 + history 8장 + 현재 front + lookdown
   │        출력 텍스트 "u v" (lookdown 픽셀 좌표) 또는 "STOP"
   │                             ▼
   └──► S1  InternVLA-N1 (Qwen2.5-VL-7B + NextDiT) ── 지시문 + front + lookdown + "u v"
            출력 4후보 × 4점 [x, y, θ]  → 안전 필터 → selected 1개
```

| | S2 (System 2) | S1 (System 1) |
|---|---|---|
| 체크포인트 | `models/system2` = `s2_e2e_v6_bootstrap_robust` | `models/system1` = `s1_e2e_v9_endpoint_focus` |
| 구조 | Qwen2.5-VL-7B-Instruct (full fine-tune) | Qwen2.5-VL-7B 백본 + latent query 4개 + NextDiT(`nextdit_async`) 궤적 헤드 + DepthAnything-V2 ViT-S memory encoder |
| 역할 | 지시문·시간 이력 → **어디로 갈지** (픽셀 goal) | 픽셀 goal·현재 영상 → **어떻게 갈지** (궤적) |
| dtype | bfloat16 | bfloat16 |
| GPU 메모리 | 약 16.6 GB | 약 16.7 GB |
| 지연 (H200) | 약 0.9–1.0 s (generate) | 약 0.34 s (latent + diffusion 10 step) |
| 추가 파일 | – | `checkpoints/depth_anything_v2_metric_hypersim_vits.pth` |

## 1. 공통 이미지 전처리

| 항목 | 값 |
|---|---|
| 입력 해상도 | 1920×1080 RGB, plumb_bob 왜곡 보정 후 (K = [[983.5, 0, 1006.9], [0, 984.8, 490.5], [0, 0, 1]]) |
| 카메라 장착 | 차량 기준 전방 1.03 m, 높이 0.92 m, pitch −1.25° |
| 프레임 간격 | 2 Hz (학습 데이터 기준) |
| pseudo-lookdown | `pseudo_lookdown_crop`: 원본의 v ≥ 432 (하단 60%) 를 잘라 다시 1920×1080으로 늘린 영상. 실제 하향 카메라가 없어 근거리 지면 뷰를 흉내 낸 것 |
| VLM 리사이즈 | Qwen2-VL `smart_resize` (min 3,136 / max 313,600 px, patch 14, merge 2) → **728×392**, 이미지당 **364 visual token** |
| 정규화 | CLIP mean (0.481, 0.458, 0.408) / std (0.269, 0.261, 0.276) |

좌표 변환 (lookdown ↔ 원본):

```
v_original = v_lookdown * 0.6 + 432      u_original = u_lookdown
v_lookdown = (v_original - 432) / 0.6
```

## 2. S2 입력

### 2.1 이미지

| 순서 | 이미지 | 비고 |
|---|---|---|
| 1…N | history front | 이번 프레임 이전의 front 최대 64장 중 **균등 샘플 최대 8장** (`np.linspace`) — safe_stop prior 사용 시 7장 |
| (N+1) | 이전 lookdown | `command=safe_stop`이고 직전 goal이 있을 때만 |
| 다음 | 현재 front | |
| 마지막 | 현재 lookdown | 2번째 user turn |

이미지 최대 10장 → visual token 약 3,640개.

### 2.2 프롬프트 (chat template, 영어 틀 + 한국어 지시문)

prior 없음 (follow / turn_left / turn_right, safe_stop 첫 프레임):

```
[user]  You are an autonomous navigation assistant. Your task is to {instruction}.
        The following images are historical front observations in chronological order;
        the first image is the initial observation and the last image is the most recent
        front observation: <image>\n<image>\n… . There is no valid previous pixel-goal.
        Where should you go next to stay on track? Please output the next waypoint's
        coordinates in the image. Please output STOP when you have successfully completed
        the task. The current front observation is you can see <image>.
[assistant] ↓
[user]  you can see <image>.            ← 현재 lookdown
[assistant]                             ← 여기서 generate
```

safe_stop prior 있음:

```
… {history} The previous look-down observation is <image>, and the tracked goal in that
previous look-down frame was {u_prev} {v_prev}. Keep tracking the same physical target
region. The current front observation is <image>. Where should you go next? Please output
the current waypoint's coordinates and output STOP when the task is complete.
```

- `{instruction}`은 요청의 `instruction` 문자열 그대로. **학습 문장과 글자까지 같아야 한다** (`config/runtime_contract.json` `trained_instructions`).
- history가 0장(주행 첫 프레임)이면 history 문장 자체가 빠진다.

### 2.3 디코딩과 출력

| 항목 | 값 |
|---|---|
| 방식 | greedy (`do_sample=False`), `max_new_tokens=48` |
| 정상 출력 | `"951 287"` — **현재 lookdown 이미지(1920×1080) 기준** 픽셀 (u 오른쪽+, v 아래+) |
| 정지 | 텍스트에 `STOP` 포함 → `status=stop` |
| 파싱 | 정수 2개 추출, `0 ≤ u < 1920`, `0 ≤ v < 1080` 아니면 `status=invalid` |
| 의미 | 차량이 향해야 할 지면 위 지점. 실측상 S1 궤적 끝점(약 4.5 m)보다 멀리 찍히는 경우가 많음 |

## 3. S1 입력

| 입력 | 내용 |
|---|---|
| VLM 이미지 | 현재 front, 현재 lookdown (각 728×392) — **history 없음** |
| VLM 텍스트 | 아래 프롬프트. 마지막 assistant turn에 S2 출력 `"u v"`를 넣어 teacher-forcing |
| memory encoder 이미지 | 현재 lookdown을 224×224로 줄여 [0,1] 정규화, 같은 이미지 2장을 쌓은 텐서 `(1, 2, 224, 224, 3)` |
| 토크나이저 | S2 텍스트를 **S1 토크나이저로 다시 토큰화** (S2 토큰 ID를 그대로 넘기지 않음) |

```
[user]  You are an autonomous navigation assistant. Your task is to {instruction}.
        Where should you go next to stay on track? Please output the next waypoint's
        coordinates in the image. Please output STOP when you have successfully completed
        the task. ahead of you is <image>.     ← front
[assistant] ↓
[user]  ahead of you is <image>.               ← lookdown
[assistant] {u} {v}                            ← S2 goal
```

처리 단계:

1. `generate_latents`: VLM 순전파, latent query 4개(`n_query=4`)의 hidden state (4 × 3584) 추출.
2. `generate_traj`: NextDiT flow-matching, **10 step**, 후보 **4개** 샘플링. 각 후보는 4 step의 (dx, dy, dθ) — 정규화 공간.

## 4. S1 출력과 후처리 (`physical_trajectories`)

1. 역정규화: `dx, dy /= 0.4` (`S1_TRAJ_XY_SCALE_VLA_E2E`), `dθ /= 1.0`.
2. 축 복원: 사전학습(R2R, 전진 = +x) 정렬을 되돌려 VLA-E2E 좌표(전진 = −y)로 (`unrotate_*_for_pretrain_alignment`).
3. 누적합 → 4개 누적 pose `[x, y, θ]`.
4. 안전 필터: 0.05 m 넘게 움직이는 step 중 heading과 이동 방향이 90° 넘게 어긋나면(후진) 제외, 끝점이 차량 뒤(전진 < −0.05 m)면 제외 → `valid_candidate_mask`.
5. 선택: 유효 후보 중 4후보 평균 끝점에 가장 가까운 것 → `selected`. 없으면 `status=unsafe_trajectory`, STOP.

출력 형식:

| 필드 | shape | 내용 |
|---|---|---|
| `candidates` | 4 × 4 × 3 | 후보별 누적 pose |
| `mean` | 4 × 3 | 후보 평균 |
| `valid_candidate_mask` | 4 | 안전 필터 통과 여부 |
| `selected_index` | int / null | 선택 후보 |
| `selected` | 4 × 3 / null | 제어기에 넘길 경로 |

좌표계 (요청 프레임 촬영 시점의 차량 기준):

| 성분 | 단위 | 방향 |
|---|---|---|
| x | m | 오른쪽 + |
| y | m | **전진 = −y** |
| θ | rad | +x축에서 반시계. 직진 = −π/2, 좌회전 → 더 작아짐, 우회전 → 0쪽. 차량 yaw(좌 +) = −(θ + π/2) |

- 학습 GT는 주행 경로를 **1.0 m 간격으로 리샘플한 4점**. 추론 실측 간격 1.2–1.3 m(마지막 0.8–1.0 m), 끝점 4.3–4.7 m.
- 시간 정보 없음 → 속도는 제어기 몫.
- 차량 base_link 기준(전진 +x, 왼쪽 +y, ROS REP-103)으로 바꾸려면:
  `x_fwd = -y`, `y_left = -x`, `yaw_left = -(θ + π/2)` (카메라 1.03 m 전방 오프셋은 학습 데이터가 차량 pose 기준이므로 더하지 않음).

## 5. 상태(state)와 command

| 상태 | 보관 위치 | 초기화 |
|---|---|---|
| front history (최대 64장) | S2 | `reset=true` 또는 새 episode |
| safe_stop 이전 lookdown·goal | S2 | reset, 또는 safe_stop 지시문 변경, safe_stop 아닌 command |
| 최신 goal (async) | 서버 | reset. key = (generation, instruction, command) — 바뀌면 새 goal 올 때까지 `awaiting_s2_goal` |

`command`는 모델 입력이 아니다. 런타임 분기에만 쓰인다.

| command | 동작 |
|---|---|
| `stop` | 모델 호출 없이 STOP (history에는 프레임 추가) |
| `safe_stop` | S2에 이전 goal prior 사용 |
| 그 외 | 일반 경로 |

## 6. 비동기 모드에서 달라지는 점

- S2는 별도 프로세스(GPU0)에서 **가장 최근 프레임만** 처리하고, 밀린 프레임은 history에만 추가 (`dropped_s2_frames`).
- 각 `/infer`는 그 시점 최신 S2 goal로 S1(GPU1)만 실행. goal이 2.25 s(`MAX_GOAL_AGE_S`)보다 오래되면 `stale_s2_goal` STOP.
- 즉 S1은 **현재 프레임 + 과거 프레임의 goal 픽셀**을 받는다. goal 나이는 평소 1.5–2.0 s.
- S1 이미지 전처리를 GPU에서 하므로(`S1_GPU_PREPROCESS=1`) 동기 모드와 비트 단위로 같지는 않다.
