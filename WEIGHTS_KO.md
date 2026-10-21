# 모델 weight 배포 (Google Drive)

weight(약 31 GiB)는 git에 포함하지 않는다. 저장소 루트 기준 아래 구조 그대로 Google Drive 폴더에 올리고 받는다.

```
models/system2/   # S2: s2_e2e_v6_bootstrap_robust (config, tokenizer, safetensors 4개)
models/system1/   # S1: s1_e2e_v9_endpoint_focus   (config, tokenizer, safetensors 4개)
checkpoints/depth_anything_v2_metric_hypersim_vits.pth
```

## 링크 (rc3, 비공개 — 공유받은 계정만 접근)

| 항목 | Google Drive |
|---|---|
| 릴리스 폴더 `dualvln-vla-e2e-20261021-rc3/` | https://drive.google.com/drive/folders/1sZl4lgkVkJfAL7DB9BCApyN_PDl99_m_ |
| `models/` | https://drive.google.com/drive/folders/1r2adB3OpJ7qujIPGQOpGE34OAuesYM5Z |
| `checkpoints/` | https://drive.google.com/drive/folders/1P-O8oXz4FDMeww2IDEDPFrd69JL8HyrO |
| `SHA256SUMS` | https://drive.google.com/file/d/1CpYH_2Zp8Y0StJVME1B13llTa2_D-59G/view |

## 올리기 (릴리스 담당)

release 번들의 `models/`, `checkpoints/` 폴더를 Drive의 `dualvln-vla-e2e-20261021-rc3/` 폴더에 그대로 업로드한다.
공유는 필요한 사람/그룹에만 준다(링크 공개 금지: 사내 차량 데이터로 학습된 weight).

```bash
tools/upload_weights_gdrive.sh [remote=gdrive] [folder=dualvln-vla-e2e-20261021-rc3]
```

업로드 후 `rclone check`(크기·MD5)로 원본과 대조하고, `SHA256SUMS`도 같은 폴더에 올린다.
rc3는 2026-10-21 업로드·검증 완료(models 29개 + checkpoints 1개, 차이 0).

업로드용 rclone remote는 `scope=drive.file`(rclone이 만든 파일만 접근)로 만든다. 서버에 브라우저가 없으면
`rclone config`에서 자동 인증을 `n`으로 두고, 출력되는 `rclone authorize "drive" "..."`를 브라우저 있는 PC에서
실행해 받은 `eyJ...` 문자열을 `config_token>`에 붙여 넣는다. rclone 공용 client_id는 2026년 중 폐지 예정이므로
Google Cloud에서 데스크톱 앱 OAuth client를 만들어 `client_id`/`client_secret`에 넣는 것을 권장한다.

## 받기

방법 1 — rclone (권장, 비공개 폴더 가능). 공유받은 폴더는 `drive.file` 범위로는 보이지 않으므로 받는 쪽 remote는
기본 `scope=drive`로 만들고, 공유 폴더를 "내 드라이브에 바로가기 추가"하거나 `--drive-shared-with-me`를 붙인다:

```bash
rclone copy "gdrive:dualvln-vla-e2e-20261021-rc3/models" ./models -P
rclone copy "gdrive:dualvln-vla-e2e-20261021-rc3/checkpoints" ./checkpoints -P

# 바로가기 없이 폴더 ID로 직접 받기
rclone copy "gdrive,root_folder_id=1sZl4lgkVkJfAL7DB9BCApyN_PDl99_m_:models" ./models -P
rclone copy "gdrive,root_folder_id=1sZl4lgkVkJfAL7DB9BCApyN_PDl99_m_:checkpoints" ./checkpoints -P
```

방법 2 — Drive 웹/데스크톱에서 폴더를 내려받아 저장소 루트에 위 구조로 둔다.

## 검증 (필수)

```bash
python3 tools/verify_weights.py      # SHA256SUMS 30개 파일 대조
python3 tools/validate_release.py    # shard·special token·S2/S1 호환성 검사
```

하나라도 MISSING/MISMATCH면 서버를 띄우지 않는다.
