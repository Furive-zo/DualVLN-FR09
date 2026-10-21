# DualVLN VLA-E2E inference runtime

S2(지시문 + 시간축 영상 → look-down pixel-goal/STOP)와 S1(goal + 현재 영상 → ego-local 궤적 후보 4개)의
2모델 추론 서버. 상세 사용법·API는 [docs/README_KO.md](docs/README_KO.md), 인수인계 이력은 [docs/HANDOFF_KO.md](docs/HANDOFF_KO.md),
데이터셋·실험·평가는 [reports/EXPERIMENTS_KO.md](reports/EXPERIMENTS_KO.md).
**실차 실시간 추론(서버 켜기·지시문·API·안전 규칙): [docs/REALTIME_VEHICLE_KO.md](docs/REALTIME_VEHICLE_KO.md)**
· 모델 입출력: [docs/MODEL_IO_KO.md](docs/MODEL_IO_KO.md) · ROS 2 노드: [docs/ROS2_KO.md](docs/ROS2_KO.md)

> **운용 상태: shadow mode 전용.** safe_stop은 recorded-sequence shadow gate를 통과하지 못했다
> (goal median 오차 203~1177 px). 폐쇄구역 검증 전 실제 차량 제어에 연결하지 않는다.

## 설치

```bash
git clone <this repo> dualvln && cd dualvln
# NVIDIA PyTorch(검증: torch 2.10.0a0 nv25.11)를 먼저 설치한 뒤
pip install -r requirements-inference.txt
# weight 받기(아래 링크 / docs/WEIGHTS_KO.md) 후 검증
python3 tools/verify_weights.py && python3 tools/validate_release.py
```

**Weights (Google Drive):**
[dualvln-vla-e2e-20261021-rc3](https://drive.google.com/drive/folders/1sZl4lgkVkJfAL7DB9BCApyN_PDl99_m_)
(`models/`, `checkpoints/`, `SHA256SUMS`). 받는 방법은 [docs/WEIGHTS_KO.md](docs/WEIGHTS_KO.md).

## 실행

| 모드 | 명령 | GPU | 제어 응답 (H200, 2 Hz shadow) |
|---|---|---|---|
| 직렬 | `./scripts/run_server.sh` | 1장 | p50 1.42 s |
| 비동기 (권장) | `S2_DEVICE=cuda:0 S1_DEVICE=cuda:1 ./scripts/run_server_async.sh` | 2장 | p50 363 ms / p95 413 ms |
| UI | `./scripts/run_ui.sh` (backend 실행 후) | – | – |

기본 bind는 `127.0.0.1:5801`(UI 5803). 외부 접근은 `HOST=0.0.0.0`.

## 폴더 구성

| 경로 | 내용 |
|---|---|
| `runtime/` | 추론 서버(S2/S1 런타임, 비동기 런타임, UI, InternNav 코드) |
| `scripts/` | 서버·UI 실행 스크립트 |
| `config/` | 릴리스 manifest, 런타임 계약, 평가 요약, weight lock |
| `tools/` | 검증·재생·profiling·배포 도구 |
| `ros2/dualvln_ros/` | ROS 2 패키지: 카메라 토픽 → 추론 서버 → `nav_msgs/Path` + stop 플래그 |
| `docs/` | 실차 추론(`REALTIME_VEHICLE_KO.md`), 모델 입출력(`MODEL_IO_KO.md`), ROS 2(`ROS2_KO.md`), 사용법(`README_KO.md`), 인수인계(`HANDOFF_KO.md`), weight 배포(`WEIGHTS_KO.md`) |
| `reports/` | 데이터셋·실험·정량/정성 평가 보고서와 그림 |
| `models/`, `checkpoints/` | weight (git 제외, Google Drive) |

## 검증 도구

- `tools/vehicle_client_example.py` — 2 Hz 차량 클라이언트 예제(watchdog, warm-up)
- `tools/render_sequence.py` — 녹화 episode에 모델을 돌려 프레임별 시각화·GIF 생성
- `tools/shadow_replay.py` — 녹화 시퀀스를 HTTP로 재생(`--rate-hz 2`로 실시간)
- `tools/profile_detailed.py` — S1/S2/전처리 단계별 latency
- `tools/verify_weights.py`, `tools/validate_release.py` — weight 무결성·호환성
