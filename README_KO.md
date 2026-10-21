# DualVLN VLA-E2E 배포 후보 rc2

이 릴리스는 **S2 v6 + S1 v9의 2모델 파이프라인**이다. S2가 지시문과 시간축 영상을 보고 look-down pixel-goal 또는 STOP을 만들고, 좌표 문자열을 S1 토크나이저로 다시 인코딩한 뒤 S1이 4개 ego-local 궤적 후보를 생성한다. S2와 S1의 `vocab.json`이 다르므로 재토큰화 단계는 제거하면 안 된다.

## 선택 모델

- S2: `s2_e2e_v6_bootstrap_robust`
- S1: `s1_e2e_v9_endpoint_focus`
- 전체 validation 2,446개: mean-trajectory ADE 0.405 m, FDE 0.739 m
- v8 대비 ADE 40.9%, FDE 44.5%, bearing MAE 49.7% 감소
- 명령별 ADE/FDE: follow 0.370/0.651, left 0.448/0.840, right 0.431/0.817, safe_stop 0.594/1.211 m

`safe_stop`의 bearing 10도 이내 비율은 42.6%로 다른 명령보다 낮다. 실제 차량 투입 전 주차장 폐쇄루프 검증이 필요하다. 서버는 S2 파싱 실패, 예외, 유효 궤적 부재 시 `control_decision=stop`, `discrete_action=0`을 반환한다.

## 로컬 실행

프로젝트 체크포인트를 직접 참조할 때:

```bash
cd /home/work/vln_project/vla_ws_restore/vla_project
PYTHON_BIN=/home/work/vln_project/venv/bin/python \
  deployments/dualvln_vla_e2e_20261009/run_server.sh
```

기본 주소는 `127.0.0.1:5801`이다. 외부 인터페이스가 필요하면 `HOST=0.0.0.0`을 명시한다. 모델 로드는 H200에서 약 40초였고 첫 E2E 추론은 약 3.6초였다. 서버는 단일 프로세스·단일 차량 에피소드용이다. 새 에피소드 첫 요청에는 `reset=true`를 보내야 한다.

```bash
curl http://127.0.0.1:5801/ready
python3 deployments/dualvln_vla_e2e_20261009/tools/client_example.py \
  --image /path/to/front.jpg --episode-id drive-001 \
  --command turn_left --instruction '좌회전하라' --reset
```

`front`는 1920x1080 RGB를 권장한다. `lookdown`을 생략하면 front 하단 60%로 pseudo look-down을 만든다. 궤적 `selected`는 `[x_m, y_m, theta_rad]` 4점의 누적 pose이며 **전진 축은 `-y`, 우측은 `+x`**다. STOP은 S1을 호출하지 않는다.


## 브라우저 추론 UI

먼저 S2/S1 backend를 실행한 뒤 별도 터미널에서 UI를 실행한다.

```bash
PYTHON_BIN=/home/work/vln_project/venv/bin/python HOST=0.0.0.0 PORT=5801 ./run_server.sh
PYTHON_BIN=/home/work/vln_project/venv/bin/python UI_HOST=0.0.0.0 UI_PORT=5803 ./run_ui.sh
```

브라우저에서 `http://<server-ip>:5803`을 연다. validation 샘플 2,571개를 명령별로 탐색하거나 직접 front/look-down 이미지를 업로드할 수 있다. 지시문과 행동 타입을 입력하면 다음을 함께 표시한다.

- S2 원문과 look-down pixel-goal, annotation GT goal
- S1 후보 4개와 물리 검사를 통과한 후보 mask
- 실제 제어 선택 궤적과 S2 pred-goal에 가장 가까운 후보 궤적
- 누적 ego-local pose, 처리 시간, 전체 JSON 응답

독립적인 지시문 비교는 `temporal history 초기화`를 켠 채 실행한다. 동일 episode의 연속 safe_stop 프레임을 시험할 때만 체크를 해제한다.

## 휴대용 번들

```bash
python3 deployments/dualvln_vla_e2e_20261009/tools/build_portable_bundle.py \
  artifacts/dualvln-vla-e2e-20261009-rc2
cd artifacts/dualvln-vla-e2e-20261009-rc2
python3 tools/validate_release.py
PYTHON_BIN=/path/to/python ./run_server.sh
```

번들은 checkpoint 중간 저장본을 제외하고 최종 모델 shard만 복사한다. 예상 모델 weight는 약 31.1 GiB다.

## API 계약

`POST /infer`는 multipart 요청이다. 이미지 필드 `front`는 필수, `lookdown`은 선택이며 `json` 필드는 다음 값을 가진다.

```json
{
  "episode_id": "drive-001",
  "instruction": "오른쪽에서 세 번째 빈 주차면에 안전하게 진입하여 정차하라",
  "command": "safe_stop",
  "reset": true
}
```

명령은 `follow`, `turn_left`, `turn_right`, `safe_stop`, `stop` 중 하나다. `safe_stop`에는 반드시 식별 가능한 랜드마크와 주차면 위치를 포함한 지시문을 사용한다. 에피소드 ID가 바뀌었는데 reset이 없으면 HTTP 409와 정지 응답을 반환한다.

응답에서 제어기가 사용할 필드는 `control_decision`, `discrete_action`, `trajectory.selected`다. 후보 전체와 `valid_candidate_mask`, S2 원문, pixel-goal, 지연시간도 함께 기록한다.

## 확인 절차

1. `python3 tools/validate_release.py`로 파일과 special token 계약을 검사한다.
2. `/ready`가 200이 될 때까지 기다린다.
3. 정지·직진·좌회전·우회전·safe_stop 순으로 recorded-frame shadow test를 수행한다.
4. 실제 차량에서는 먼저 제어 출력을 사용하지 않는 shadow mode로 지연시간과 STOP/invalid 비율을 수집한다.
5. 저속 폐쇄구역에서 trajectory frame 변환과 watchdog 정지를 검증한 뒤 제어기에 연결한다.

상세 축·좌표·temporal 상태 규칙은 `config/runtime_contract.json`, 고정 파일 정보는 `config/release_lock.json`, 평가 수치는 `config/evaluation_summary.json`에 있다.

## 비동기 실행 (S2/S1 분리, GPU 2장)

```bash
PYTHON_BIN=/home/work/vln_project/venv/bin/python HOST=0.0.0.0 PORT=5801 \
  S2_DEVICE=cuda:0 S1_DEVICE=cuda:1 ./run_server_async.sh
```

API는 동일하다. 응답에 `goal_age_ms`, `goal_frame_seq`, `s1_latency_ms`, `s2_latency_ms`가 추가되며,
goal이 없거나(`awaiting_s2_goal`) `MAX_GOAL_AGE_S`(기본 2.25 s)보다 오래되면(`stale_s2_goal`) STOP을 반환한다.
episode 시작과 명령 변경 직후 첫 goal까지 약 1 s는 STOP이다.
