# 모델 weight 배포 (Google Drive)

weight(약 31 GiB)는 git에 포함하지 않는다. 저장소 루트 기준 아래 구조 그대로 Google Drive 폴더에 올리고 받는다.

```
models/system2/   # S2: s2_e2e_v6_bootstrap_robust (config, tokenizer, safetensors 4개)
models/system1/   # S1: s1_e2e_v9_endpoint_focus   (config, tokenizer, safetensors 4개)
checkpoints/depth_anything_v2_metric_hypersim_vits.pth
```

## 올리기 (릴리스 담당)

release 번들의 `models/`, `checkpoints/` 폴더를 Drive의 `dualvln-vla-e2e-20261021-rc3/` 폴더에 그대로 업로드한다.
공유는 필요한 사람/그룹에만 준다(링크 공개 금지: 사내 차량 데이터로 학습된 weight).

## 받기

방법 1 — rclone (권장, 비공개 폴더 가능):

```bash
rclone copy "gdrive:dualvln-vla-e2e-20261021-rc3/models" ./models -P
rclone copy "gdrive:dualvln-vla-e2e-20261021-rc3/checkpoints" ./checkpoints -P
```

방법 2 — Drive 웹/데스크톱에서 폴더를 내려받아 저장소 루트에 위 구조로 둔다.

## 검증 (필수)

```bash
python3 tools/verify_weights.py      # SHA256SUMS 30개 파일 대조
python3 tools/validate_release.py    # shard·special token·S2/S1 호환성 검사
```

하나라도 MISSING/MISMATCH면 서버를 띄우지 않는다.
