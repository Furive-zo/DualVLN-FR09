# ROS 2 연동 (`ros2/dualvln_ros`)

> **shadow mode 전용.** 출력은 기록·모니터링만 하고 차량 제어에 연결하지 않는다. `safe_stop`은 shadow gate 미통과.

카메라 토픽을 받아 GPU 추론 서버(HTTP)에 보내고, 결과를 `nav_msgs/Path`와 정지 플래그로 내보내는 ament_python 패키지.
모델은 노드 안에서 돌지 않는다. 차량 PC에는 GPU·PyTorch가 필요 없다.

```
 camera driver ── image_rect (1920×1080, 2 Hz) ──►  dualvln_inference  ── HTTP ──►  GPU 서버
                                                     (ros2 node)        ◄─ JSON ──  scripts/run_server_async.sh
 mission / 운전자 ── ~/command (String) ─────────►      │
                                                         ├─► ~/path        nav_msgs/Path
                                                         ├─► ~/stop        std_msgs/Bool (10 Hz)
                                                         ├─► ~/status      std_msgs/String (JSON)
                                                         └─► ~/candidates  visualization_msgs/MarkerArray
```

모델 입출력의 상세 내용은 [MODEL_IO_KO.md](MODEL_IO_KO.md), HTTP API와 서버 운용은 [REALTIME_VEHICLE_KO.md](REALTIME_VEHICLE_KO.md)를 본다.

## 1. 빌드

ROS 2 Humble 기준(Python 3.10 이상). 의존 패키지는 표준 메시지와 `python3-requests`, `python3-pil`, `python3-numpy`만 쓴다. `cv_bridge`는 필요 없다.

```bash
mkdir -p ~/dualvln_ws/src
cp -r <repo>/ros2/dualvln_ros ~/dualvln_ws/src/     # 또는 ln -s
cd ~/dualvln_ws
rosdep install --from-paths src -y --ignore-src
colcon build --packages-select dualvln_ros
source install/setup.bash
```

> 최신 setuptools에서는 `--symlink-install`이 `option --editable not recognized`로 실패할 수 있다. 그 경우 일반 `colcon build`를 쓴다.

## 2. 실행 순서

```bash
# (1) GPU 서버: 모델 로딩 약 50 s, /ready가 200이 될 때까지 대기
./scripts/run_server_async.sh             # HOST=0.0.0.0 으로 열면 차량 PC에서 접속 가능

# (2) 차량 PC: 노드 실행 (시작 시 command = stop)
ros2 launch dualvln_ros dualvln.launch.py \
    server_url:=http://<gpu-server>:5801 \
    image_topic:=/camera/front/image_rect/compressed

# (3) 주행 시작: 지시문 넣기
ros2 topic pub --once /dualvln_inference/command std_msgs/String "{data: follow}"
```

노드는 시작하면 `/ready`를 기다린 뒤, 첫 카메라 프레임으로 warm-up 요청을 6번 보낸다(첫 S1 호출은 1 s 이상 걸림).
warm-up이 끝나면 새 episode를 만들고 로그에 `ready, episode drive-…`를 출력한다. 그 전까지 `~/stop`은 계속 True다.

### 녹화 데이터로 시험

카메라 대신 `replay_publisher`로 녹화한 JPEG를 2 Hz `CompressedImage`로 내보낸다.

```bash
ros2 run dualvln_ros replay_publisher --ros-args \
    -p dir:=/home/work/.datasets/VLA-E2E/20260515/undistort_orin_pc_data11/camera/rgb_front -p loop:=true
ros2 run dualvln_ros inference_node --ros-args \
    --params-file install/dualvln_ros/share/dualvln_ros/config/dualvln.yaml
ros2 topic pub --once /dualvln_inference/command std_msgs/String "{data: follow}"
ros2 topic echo /dualvln_inference/status
```

## 3. 지시문 넣기 (`~/command`, `std_msgs/String`)

| 보내는 값 | 서버에 가는 instruction |
|---|---|
| `follow` | 도로를 따라 주행하라 |
| `turn_left` | 좌회전하라 |
| `turn_right` | 우회전하라 |
| `stop` | 정지하라 (모델 호출 없이 STOP) |
| `{"command": "safe_stop", "instruction": "오른쪽에서 두 번째 빈 주차면에 안전하게 진입하여 정차하라"}` | 그대로 |

```bash
ros2 topic pub --once /dualvln_inference/command std_msgs/String "{data: turn_left}"
ros2 topic pub --once /dualvln_inference/command std_msgs/String \
  "{data: '{\"command\": \"safe_stop\", \"instruction\": \"오른쪽에서 두 번째 빈 주차면에 안전하게 진입하여 정차하라\"}'}"
```

- 모델은 instruction 문장만 읽는다. **학습 문장과 글자까지 같아야 한다.** safe_stop용 6개 문장은 `dualvln_ros/core.py`의 `SAFE_STOP_INSTRUCTIONS`와 `config/runtime_contract.json`에 있다.
- JSON에서 `instruction`을 생략하면 command별 기본 문장을 쓴다. `safe_stop`은 생략할 수 없다.
- 형식이 틀리거나 모르는 command는 거부하고(에러 로그) 직전 command를 유지한다.
- command를 바꿔도 episode는 그대로 유지된다(S2 history 보존). 새 goal이 나올 때까지 약 1 s 동안 `awaiting_s2_goal` STOP이 나오므로, 교차로 진입 전에 미리 보낸다.

## 4. 새 주행 (`~/reset`, `std_srvs/Trigger`)

```bash
ros2 service call /dualvln_inference/reset std_srvs/srv/Trigger
```

새 `episode_id`를 만들고, 다음 요청에 `reset=true`를 실어 보내 서버의 history를 비운다.
주행 시작이나 장소 이동처럼 이전 영상과 이어지지 않을 때만 호출한다. 노드를 시작하면 한 번 자동으로 실행된다.

## 5. 출력

### `~/path` (`nav_msgs/Path`)

- `header.stamp` = **카메라 이미지의 stamp**(촬영 시각). `frame_id` = `base_link`(파라미터로 변경 가능).
- pose 5개: 원점(0, 0, 0) + 모델 궤적 4점. **REP-103 기준**(x 전진, y 왼쪽, yaw 반시계(좌) +)으로 변환했다.
  모델 좌표(x 오른쪽, 전진 = −y, θ)에서 `x_fwd = -y`, `y_left = -x`, `yaw = -(θ + π/2)`.
- 유효한 궤적이 나왔을 때만 publish한다. STOP 응답에는 path를 내지 않는다.
- 시간 정보가 없는 기하 경로다(4점 간격 약 1.2–1.4 m, 끝점 약 4.3–5.6 m, data11 실측). 속도는 제어기가 정한다.
- **촬영 시점의 base_link 기준**이다. 응답은 약 0.4 s 늦게 오므로, 제어기는 `tf2`로 stamp 시점의 `odom→base_link`를 조회해 odom 좌표로 바꿔서 추종한다.

  ```python
  # 예: 제어기 쪽
  path_odom = tf_buffer.transform(pose_stamped, 'odom', timeout=Duration(seconds=0.1))  # tf2_geometry_msgs
  ```

### `~/stop` (`std_msgs/Bool`, 10 Hz)

True면 반드시 정지한다. 다음 경우 True가 된다.

- 노드 준비 전(서버 대기, warm-up 중)
- 서버가 STOP 응답(`control_decision != trajectory`): `command=stop`, S2 STOP, `awaiting_s2_goal`, `stale_s2_goal`, `unsafe_trajectory`, HTTP 4xx/5xx
- **watchdog**: 유효한 궤적이 `watchdog_s`(1.0 s) 동안 안 옴. 카메라 끊김과 서버 무응답도 여기에 포함된다.

요청 timeout(0.8 s) 한 번으로는 정지하지 않고 직전 상태를 유지한다(`action = hold`). watchdog이 최종 판단한다.
**제어기는 `~/stop`을 구독해야 하고, 이 토픽이 0.3 s 넘게 안 오면 스스로 정지해야 한다**(노드가 죽은 경우).

### `~/status` (`std_msgs/String`, 응답마다)

```json
{"status": "coordinate", "control_decision": "trajectory", "http_status": 200,
 "rtt_ms": 377.2, "latency_ms": 352.1, "s1_latency_ms": 341.0, "s2_latency_ms": 930.4,
 "goal_age_ms": 1499.0, "dropped_s2_frames": 12, "pixel_goal_lookdown": [951, 287],
 "episode_id": "drive-20261021-173430", "action": "follow", "command": "follow"}
```

`action`: `follow`(path 발행), `stop`, `hold`(timeout, 직전 상태 유지). 각 `status` 값의 의미는 REALTIME_VEHICLE_KO.md의 상태값 표를 본다.

### `~/candidates` (`visualization_msgs/MarkerArray`)

RViz용. 후보 4개(연한 빨강), 선택 궤적(자홍, 굵게), 상태 텍스트를 표시한다. Fixed Frame은 `base_link`.

## 6. 파라미터 (`config/dualvln.yaml`)

| 이름 | 기본값 | 설명 |
|---|---|---|
| `server_url` | `http://127.0.0.1:5801` | GPU 추론 서버 |
| `image_topic` | `/camera/front/image_rect/compressed` | 왜곡 보정된 전방 1920×1080 |
| `compressed` | `true` | `false`면 `sensor_msgs/Image`(`rgb8`/`bgr8`/`rgba8`/`bgra8`) |
| `rate_hz` | `2.0` | 최대 전송률. 학습 데이터가 2 Hz라 올리지 않는다 |
| `timeout_s` | `0.8` | HTTP timeout |
| `watchdog_s` | `1.0` | 이 시간 동안 유효 궤적이 없으면 stop |
| `frame_id` | `base_link` | `~/path` frame |
| `initial_command` | `stop` | 시작 command (`follow` 등으로 바꾸면 바로 주행 추론) |
| `warmup` | `6` | warm-up 요청 수 (0이면 생략) |
| `jpeg_quality` | `95` | raw `Image`를 JPEG로 바꿀 때 품질 (JPEG `CompressedImage`는 재인코딩 없이 그대로 보냄) |

## 7. 동작 방식

- **이미지 도착이 추론을 구동한다.** 새 프레임이 오고, 진행 중인 요청이 없고, 직전 전송 후 `0.75 / rate_hz` s가 지났으면 바로 보낸다. 요청 중에 온 프레임은 건너뛴다(서버 history에도 들어가지 않는다).
- HTTP 요청은 별도 스레드에서 돌아서 ROS executor를 막지 않는다. 10 Hz 타이머는 `~/stop` heartbeat와 watchdog만 처리한다.
- 입력 해상도가 1920×1080이 아니면 경고를 낸다. 서버는 다른 해상도도 받지만, 학습 분포와 달라진다.
- 카메라 조건(왜곡 보정, 장착 위치 1.03 m 전방·0.92 m 높이)은 REALTIME_VEHICLE_KO.md §3을 따라야 한다. 다른 카메라나 장착 위치에서는 성능을 보장할 수 없다.

## 8. 시험 결과 (2026-10-21, ROS 2 Humble/RoboStack, 같은 호스트에서 H200 ×2 async 서버)

data11 녹화 영상을 `replay_publisher`로 2 Hz 재생했다. 처음 30 s는 follow, 다음 10 s는 safe_stop, 마지막 4 s는 stop을 보냈다. 호스트 load average는 약 50이었다.

| 항목 | 값 |
|---|---|
| 노드 시작 → ready (warm-up 포함, 서버는 이미 로딩됨) | 7 s |
| 응답 수 | 82 |
| follow | 정상 궤적 53, `awaiting_s2_goal` 2 (명령 직후) |
| safe_stop | 정상 궤적 17, `awaiting_s2_goal` 2 |
| stop | STOP 8 |
| `~/path` 발행 | 70 |
| RTT p50 / max | 377 / 549 ms |
| S2 goal age p50 | 1.50 s |
| 노드 에러 | 0 |

첫 path 예: `(0,0) → (1.27, 0.10) → (2.79, 0.18) → (4.18, 0.20) → (5.55, 0.18)` (base_link, m). 거의 직진이고 왼쪽으로 약간 치우쳤다.

## 9. 파일

| 파일 | 내용 |
|---|---|
| `dualvln_ros/inference_node.py` | ROS 2 노드 |
| `dualvln_ros/core.py` | ROS 의존 없는 로직: 지시문 표, command 파싱, HTTP client, 좌표 변환, stop 판단 |
| `dualvln_ros/replay_publisher.py` | 녹화 JPEG → `CompressedImage` 시험용 노드 |
| `launch/dualvln.launch.py`, `config/dualvln.yaml` | launch와 파라미터 |
| `test/test_core.py` | `pytest` (ROS 없이 실행 가능) |
