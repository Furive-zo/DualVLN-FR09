# DualVLN VLA-E2E inference runtime (rc3)

S2(지시문 + 시간축 영상 → look-down pixel-goal/STOP)와 S1(goal + 현재 영상 → ego-local 궤적 후보 4개)의
2모델 추론 서버. 상세 사용법·API는 `README_KO.md`, 인수인계 이력은 `HANDOFF_KO.md`.

> **운용 상태: shadow mode 전용.** safe_stop은 recorded-sequence shadow gate를 통과하지 못했다
> (goal median 오차 203~1177 px). 폐쇄구역 검증 전 실제 차량 제어에 연결하지 않는다.

## 설치

```bash
git clone <this repo> dualvln && cd dualvln
# NVIDIA PyTorch(검증: torch 2.10.0a0 nv25.11)를 먼저 설치한 뒤
pip install -r requirements-inference.txt
# weight 받기 + 검증: WEIGHTS_KO.md
# (Drive, 비공개: https://drive.google.com/drive/folders/1sZl4lgkVkJfAL7DB9BCApyN_PDl99_m_)
python3 tools/verify_weights.py && python3 tools/validate_release.py
```

## 실행

| 모드 | 명령 | GPU | 제어 응답 (H200, 2 Hz shadow) |
|---|---|---|---|
| 직렬 | `./run_server.sh` | 1장 | p50 1.42 s |
| 비동기 (권장) | `S2_DEVICE=cuda:0 S1_DEVICE=cuda:1 ./run_server_async.sh` | 2장 | p50 363 ms / p95 413 ms |
| UI | `./run_ui.sh` (backend 실행 후) | – | – |

기본 bind는 `127.0.0.1:5801`(UI 5803). 외부 접근은 `HOST=0.0.0.0`.

## 검증 도구

- `tools/shadow_replay.py` — 녹화 시퀀스를 HTTP로 재생(`--rate-hz 2`로 실시간)
- `tools/profile_detailed.py` — S1/S2/전처리 단계별 latency
- `tools/verify_weights.py`, `tools/validate_release.py` — weight 무결성·호환성
