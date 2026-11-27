# S2 학습 메모리·속도 벤치마크

S2 학습(Qwen2.5-VL-7B 기반, 8.4B)과 같은 설정을 합성 데이터로 재현해 GPU별 최대 메모리와
step 시간을 잰다. 학습 데이터 없이 모델 가중치만 있으면 실행할 수 있다.

재현하는 학습 설정:
- bf16, sdpa attention, gradient checkpointing
- LLM LoRA rank 64(q/k/v/o/gate/up/down) + `visual.merger` 학습, 비전 인코더 동결 (학습 파라미터 206.1M)
- 샘플당 이미지 9장(560×560, 이미지당 400토큰) + 텍스트 240토큰 ≈ 3,858토큰
- 기본 batch 4 × gradient accumulation 4

## 환경
```bash
pip install torch            # GPU에 맞는 CUDA 빌드 (예: RTX 50 시리즈는 CUDA 12.8 이상 빌드, torch 2.7 이상)
pip install transformers==4.57.1 peft==0.18.0 accelerate
pip install deepspeed        # ZeRO-3 측정에만 필요
```

## 실행
```bash
cd benchmarks/train_memory
bash run_bench.sh <S2 체크포인트 폴더 | Qwen/Qwen2.5-VL-7B-Instruct>   # GPU 수는 NGPU=4 (기본)
```
1. GPU 1장, batch 4 × accum 4
2. GPU 1장, batch 1 × accum 4
3. GPU N장 DDP (GPU마다 모델 전체)
4. GPU N장 DeepSpeed ZeRO-3 (모델을 N장에 분산)

결과는 `result.jsonl`(설정별 한 줄), GPU 정보는 `gpu_info.txt`, 카드 간 연결은 `gpu_topo.txt`에 저장된다.
개별 실행은 `python bench_s2_train.py --help` 참고.

## 측정 결과

| GPU | 설정 | 샘플당 시간 | GPU당 최대 메모리 |
|---|---|---|---|
| H200 (141 GB) × 1 | batch 4 × accum 4 | 0.853 s | 48.7 GiB |
| H200 (141 GB) × 1 | batch 1 × accum 4 | 0.909 s | 25.8 GiB |
| RTX 5090 (32 GB) × 1 | batch 4 × accum 4 | OOM | 28.9 GiB 시점에 메모리 부족 |
| RTX 5090 (32 GB) × 1 | batch 1 × accum 4 | 2.618 s | 27.0 GiB |
| RTX 5090 (32 GB) × 2, DDP | batch 1 × accum 4 | 1.337 s | 27.7 GiB |
| RTX 5090 (32 GB) × 4, DDP | batch 1 × accum 4 | 0.689 s | 27.7 GiB |

샘플당 시간은 전체 GPU 기준(처리량의 역수)이다. H200 2장을 각각 독립 학습에 쓰면 샘플당 약 0.427 s로,
RTX 5090 2장 DDP(1.337 s) 대비 약 3.13배 빠르다. RTX 5090은 batch 1에서도 GPU당 27~28 GiB(가용 31.4 GiB)를
사용해 H200 기준 설정(batch 4)은 실행할 수 없다. RTX 5090 4장은 PCIe로만 연결되어 있다(`gpu_topo_rtx5090.txt`).
DeepSpeed ZeRO-3 측정은 RTX 5090 환경(transformers 5.x / peft 0.19)의 dtype 오류로 수행하지 못했다.

원본 수치: `results_h200.jsonl`, `results_rtx5090.jsonl` (RTX 5090: torch 2.8.0+cu128). GPU 정보: `gpu_info_rtx5090.txt`. 실제 S2 학습 로그(데이터 로딩·증강 포함)는 샘플당 약 1.06 s였다.
