#!/usr/bin/env bash
# 멀티 GPU 서버에서 실행. 사용법: bash run_bench.sh <모델 경로>
# 결과는 result.jsonl 에 한 줄씩 쌓인다. 4가지 측정에 약 20~30분.
set -u
M=${1:?모델 경로(S2 체크포인트 폴더 또는 Qwen/Qwen2.5-VL-7B-Instruct)를 주세요}
OUT=result.jsonl
nvidia-smi --query-gpu=index,name,memory.total --format=csv | tee gpu_info.txt
nvidia-smi topo -m | tee gpu_topo.txt          # 카드 간 연결(PCIe/P2P) 확인용

echo "== 1) 1장, batch 4 x accum 4 (학습 기본 설정)"
CUDA_VISIBLE_DEVICES=0 python bench_s2_train.py --model "$M" --batch-size 4 --grad-accum 4 --steps 3 --warmup 1 --out $OUT
echo "== 2) 1장, batch 1 x accum 4 (메모리 최소 설정)"
CUDA_VISIBLE_DEVICES=0 python bench_s2_train.py --model "$M" --batch-size 1 --grad-accum 4 --steps 4 --warmup 1 --out $OUT
echo "== 3) N장 DDP, batch 1 x N장 x accum 4"
torchrun --nproc_per_node ${NGPU:-4} bench_s2_train.py --model "$M" --batch-size 1 --grad-accum 4 --steps 4 --warmup 1 --out $OUT
echo "== 4) N장 ZeRO-3, batch 1 x N장 x accum 4"
torchrun --nproc_per_node ${NGPU:-4} bench_s2_train.py --model "$M" --batch-size 1 --grad-accum 4 --steps 4 --warmup 1 --deepspeed ds_zero3.json --out $OUT
echo "완료: $OUT, gpu_info.txt, gpu_topo.txt 에 결과가 저장됨"
