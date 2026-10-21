# DualVLN VLA-E2E 세션 종료 인수인계

작성일: 2026-10-10 UTC

## 현재 상태

- 릴리스: `dualvln-vla-e2e-20261009-rc2`
- 상태: release candidate / shadow deployment 준비 완료
- S2: `s2_e2e_v6_bootstrap_robust`
- S1: `s1_e2e_v9_endpoint_focus`
- 세션 종료 시 추론 backend와 UI를 정상 종료함. 포트 5801/5803은 비어 있음.
- `keep_alive.py --fill-vram` GPU 버닝 프로세스도 종료했으며 GPU compute process는 없음.
- 최종 휴대용 artifact: `/home/work/vln_project/vla_ws_restore/vla_project/artifacts/dualvln-vla-e2e-20261009-rc2`
- 크기: 약 32 GiB

## 최종 평가

S1 v9 전체 validation 2,446개, cached free-generated S2 v6 goal, diffusion 후보 4개, 10 steps 기준:

- mean-trajectory ADE/FDE: `0.405 / 0.739 m`
- sample ADE/FDE: `0.526 / 0.943 m`
- bearing MAE: `6.74°`
- bearing 10° 이내: `90.9%`
- heading-relative reverse candidate: `3.11%`
- v8 대비 mean ADE `40.9%`, FDE `44.5%`, bearing MAE `49.7%` 감소

명령별 mean ADE/FDE:

- follow: `0.370 / 0.651 m`
- turn_left: `0.448 / 0.840 m`
- turn_right: `0.431 / 0.817 m`
- safe_stop: `0.594 / 1.211 m`

`safe_stop` bearing 10° 이내 비율은 `42.6%`로 실제 차량 연결 전 폐쇄 주차장 저속 검증이 필요하다.

## 런타임 계약

- S2와 S1은 별도 모델로 로드한다.
- S2 좌표 문자열을 decode한 뒤 반드시 S1 tokenizer로 다시 tokenize한다. 두 모델의 `vocab.json`이 달라 이 단계를 생략할 수 없다.
- STOP은 S1을 우회하고 `discrete_action=0`을 반환한다.
- trajectory는 `[x_m, y_m, theta_rad]` 누적 pose 4점이다. 전진은 `-y`, 우측은 `+x`다.
- 후보는 heading-relative reverse와 endpoint-behind 검사 후 선택한다. 유효 후보가 없거나 예외가 발생하면 fail-safe STOP을 반환한다.
- backend는 단일 차량 episode 상태를 유지한다. episode가 바뀌면 `reset=true`가 필요하다.
- safe_stop temporal prior는 같은 episode와 같은 목표 지시문에서만 유지한다.

## 재시작

터미널 1 — S2/S1 backend:

```bash
cd /home/work/vln_project/vla_ws_restore/vla_project/artifacts/dualvln-vla-e2e-20261009-rc2
PYTHON_BIN=/home/work/vln_project/venv/bin/python HOST=0.0.0.0 PORT=5801 ./run_server.sh
```

터미널 2 — 추론 UI:

```bash
cd /home/work/vln_project/vla_ws_restore/vla_project/artifacts/dualvln-vla-e2e-20261009-rc2
PYTHON_BIN=/home/work/vln_project/venv/bin/python UI_HOST=0.0.0.0 UI_PORT=5803 \
  INFERENCE_URL=http://127.0.0.1:5801/infer ./run_ui.sh
```

원격 PC에서는:

```bash
ssh -N -L 5803:127.0.0.1:5803 <user>@<server>
```

브라우저 주소는 `http://127.0.0.1:5803`이다. SSH가 Docker 바깥 호스트에 연결되면 tunnel target을 `172.17.0.10:5803`으로 사용한다.

## 추론 UI

- validation sample 2,571개와 command filter 제공
- front 및 별도 look-down 이미지 업로드 지원
- 사용자가 지시문과 5종 command를 입력
- S2 pred/GT pixel-goal 표시
- S1 후보 4개, valid mask, 제어 선택 후보 표시
- pred-goal에 가장 가까운 후보를 별도 표시
- raw JSON 및 latency 표시
- 원격 포트 프록시를 위해 모든 browser API 경로는 상대경로로 수정됨

동일 이미지에서 지시문만 바꿀 때는 temporal reset을 켠다. 연속 safe_stop 프레임 시험에서만 같은 episode ID로 reset을 해제한다.

## 완료된 배포 검증

- source 및 portable Python compile 통과
- shell syntax 통과
- model shard와 special-token 계약 검증 통과
- S2/S1 vocab 차이에 따른 retokenization 확인
- DepthAnything V2 checkpoint를 portable bundle에 포함하고 경로 고정
- portable 내부 InternNav import 확인
- Gunicorn 단일 GPU worker startup 및 `/ready` 확인
- 실제 HTTP S2→S1 추론 200 확인
- UI→backend 실제 추론과 궤적 투영 확인
- UI JavaScript syntax 확인
- 모델 29개 파일 + DepthAnything checkpoint SHA-256 기록

최근 UI smoke test: 좌회전 샘플, S2 `794 325`, 후보 4개, controller selected `#3`, goal-closest `#1`, latency `4.276 s`. 별도 portable HTTP smoke는 약 `4.45 s`였다.

## 남은 작업

1. recorded sequence shadow test로 episode history와 safe_stop previous-goal 연속성을 확인한다.
2. H200 기준 약 4.3~4.5초인 E2E latency를 제어 주기에 맞게 줄인다.
3. 저속 폐쇄구역에서 camera frame, trajectory frame, watchdog STOP을 검증한다.
4. safe_stop을 주차장 시나리오에서 별도 acceptance gate로 평가한다.
5. 위 검증 후 실제 차량 controller와 연결한다.

세부 파일:

- `README_KO.md`: 실행 및 API 사용법
- `config/release_manifest.json`: 선택 모델과 런타임 설정
- `config/runtime_contract.json`: 좌표계와 temporal 계약
- `config/evaluation_summary.json`: 정량 평가
- `config/deployment_validation.json`: 배포 및 UI smoke 결과
- `config/release_lock.json`: 모델/checkpoint 해시
- `SHA256SUMS`: portable 모델/checkpoint 무결성

## 2026-10-20 후속 (shadow test / latency)

상세: `outputs/shadow_test_rc2_20261020/SHADOW_TEST_REPORT_KO.md`

- 남은 작업 1 (recorded sequence shadow test) 수행: 813 요청 오류 0, 409/STOP 계약 통과,
  주행 pixel-goal median 44~106 px로 오프라인과 동일 수준.
- **safe_stop은 gate 미통과**: closed-loop median 203/245/1177 px. prior는 튐만 줄이고 정확도는 그대로.
  S2 safe_stop grounding 개선(및 data904 라벨 불일치 확인) 전에는 실차 safe_stop 사용 금지.
- 남은 작업 2 (latency): H200 단독 기준 rc2 실제 E2E는 1.44 s(4.3 s 아님). 배포 소스 런타임에
  bit-exact 전처리 캐시를 넣어 ~1.17 s. 아직 2 Hz(0.5 s) 미달 — 다음은 ViT embedding 캐시.
  수정본은 rc2 번들에 반영하지 않았으며 새 rc로 묶을 때 `validate_release.py`와 hash 갱신 필요.
- 도구: `tools/shadow_replay.py`, `tools/profile_latency.py`
- 비동기 S2/S1 검토: 현재 런타임은 직렬. 비동기 시 제어 루프(S1)는 약 365 ms로 2 Hz 충족.
  goal 지연 오프라인 평가(`outputs/async_goal_lag_s1v9/REPORT_KO.md`) 결과 1.0~1.5 s 지연에서 ADE +3~6 mm로
  재학습 없이 분리 가능. 단 S1은 goal 의존도가 낮음(goal 좌우 반전 시 ADE +0.05 m, safe_stop은 거의 무변화).
- **비동기 런타임 구현** (`scripts/run_server_async.sh`, `runtime/async_runtime.py`, 상세 `outputs/shadow_test_async_20261020/REPORT_KO.md`):
  S2 서브프로세스(GPU 0) + S1 요청 경로(GPU 1, GPU 전처리). 2 Hz 실시간 shadow test에서 제어 응답
  p50 363 / p95 413 ms(직렬 1420 ms), 명령 유지 구간 trajectory 출력 95.8%. goal 나이 상한 기본 2.25 s.
  명령 변경 직후 약 1 s는 goal 대기 STOP. 직렬 `run_server.sh`는 rc2와 bit-identical로 유지.
