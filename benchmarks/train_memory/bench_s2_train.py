"""S2 학습 조건 재현 벤치마크 (메모리 / step 시간 측정).

실제 S2 학습(scripts/train_s2_e2e_v2.py)과 같은 설정을 합성 데이터로 재현한다.
  - Qwen2.5-VL-7B 구조(8.4B), bf16, sdpa attention, gradient checkpointing
  - LLM LoRA rank 64 (q,k,v,o,gate,up,down) + visual.merger 학습, 비전 인코더 동결
  - 샘플당 이미지 9장(560x560 → 이미지당 400토큰) + 텍스트 약 240토큰 = 약 3,850토큰
  - batch 4 x gradient accumulation 4 (H200 기준 설정)
학습 데이터는 필요 없다(난수 이미지·토큰). 모델 가중치만 있으면 된다.

실행 예 (자세한 내용은 README.md):
  단일 GPU      : python bench_s2_train.py --model <경로> --batch-size 1 --grad-accum 16
  DDP 4장       : torchrun --nproc_per_node 4 bench_s2_train.py --model <경로> --batch-size 1 --grad-accum 4
  ZeRO-3 4장    : torchrun --nproc_per_node 4 bench_s2_train.py --model <경로> --batch-size 4 --grad-accum 1 --deepspeed ds_zero3.json
"""
import argparse, json, os, platform, socket, time

import torch
from torch.utils.data import Dataset
from transformers import (Qwen2_5_VLForConditionalGeneration, Trainer, TrainingArguments,
                          TrainerCallback)
from peft import LoraConfig, TaskType, get_peft_model

IMAGE_PAD, VISION_START, VISION_END = 151655, 151652, 151653
PATCH_DIM = 3 * 2 * 14 * 14          # channel x temporal_patch x patch x patch
GRID = (1, 40, 40)                    # 560x560 이미지 → 1600 patch → merge 2x2 후 400 토큰


class SyntheticS2(Dataset):
    def __init__(self, n, images, text_tokens, label_tokens):
        self.n, self.images, self.text, self.lab = n, images, text_tokens, label_tokens

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        g = torch.Generator().manual_seed(i)
        per_img = GRID[0] * GRID[1] * GRID[2] // 4
        ids = []
        for _ in range(self.images):
            ids += [VISION_START] + [IMAGE_PAD] * per_img + [VISION_END]
        ids += torch.randint(1000, 30000, (self.text,), generator=g).tolist()
        ids = torch.tensor(ids)
        labels = torch.full_like(ids, -100)
        labels[-self.lab:] = ids[-self.lab:]
        patches = self.images * GRID[0] * GRID[1] * GRID[2]
        return {'input_ids': ids, 'labels': labels,
                'pixel_values': torch.randn(patches, PATCH_DIM, generator=g),
                'image_grid_thw': torch.tensor([GRID] * self.images)}


def collate(batch):
    return {'input_ids': torch.stack([b['input_ids'] for b in batch]),
            'labels': torch.stack([b['labels'] for b in batch]),
            'attention_mask': torch.ones(len(batch), batch[0]['input_ids'].numel(), dtype=torch.long),
            'pixel_values': torch.cat([b['pixel_values'] for b in batch]).to(torch.bfloat16),
            'image_grid_thw': torch.cat([b['image_grid_thw'] for b in batch])}


class Timer(TrainerCallback):
    def __init__(self, warmup):
        self.warmup, self.times, self.t0 = warmup, [], None

    def on_step_begin(self, args, state, control, **kw):
        torch.cuda.synchronize()
        self.t0 = time.perf_counter()

    def on_step_end(self, args, state, control, **kw):
        torch.cuda.synchronize()
        if state.global_step > self.warmup:
            self.times.append(time.perf_counter() - self.t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, help='S2 체크포인트 또는 Qwen/Qwen2.5-VL-7B-Instruct')
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--grad-accum', type=int, default=4)
    ap.add_argument('--steps', type=int, default=8, help='측정할 optimizer step 수(warmup 제외)')
    ap.add_argument('--warmup', type=int, default=2)
    ap.add_argument('--images', type=int, default=9)
    ap.add_argument('--text-tokens', type=int, default=240)
    ap.add_argument('--label-tokens', type=int, default=12)
    ap.add_argument('--deepspeed', default=None)
    ap.add_argument('--out', default='bench_result.jsonl')
    a = ap.parse_args()

    rank = int(os.environ.get('RANK', 0))
    world = int(os.environ.get('WORLD_SIZE', 1))
    total_steps = a.steps + a.warmup

    # ZeRO-3는 TrainingArguments를 먼저 만들어야 모델을 GPU마다 나눠서 로드한다.
    targs = TrainingArguments(
        output_dir='/tmp/bench_s2_out', max_steps=total_steps, per_device_train_batch_size=a.batch_size,
        gradient_accumulation_steps=a.grad_accum, learning_rate=1e-5, bf16=True,
        gradient_checkpointing=True, save_strategy='no', eval_strategy='no', logging_steps=1000,
        report_to=[], remove_unused_columns=False, dataloader_num_workers=2,
        deepspeed=a.deepspeed, ddp_find_unused_parameters=False)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        a.model, dtype=torch.bfloat16, attn_implementation='sdpa')
    model.config.use_cache = False
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=64, lora_alpha=128, lora_dropout=0.05, bias='none',
        target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"))
    for n, p in model.named_parameters():
        p.requires_grad_('lora_' in n or 'visual.merger' in n)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    n_samples = a.batch_size * a.grad_accum * world * (total_steps + 1)
    ds = SyntheticS2(n_samples, a.images, a.text_tokens, a.label_tokens)
    timer = Timer(a.warmup)
    torch.cuda.reset_peak_memory_stats()
    status = 'ok'
    try:
        Trainer(model=model, args=targs, train_dataset=ds, data_collator=collate,
                callbacks=[timer]).train()
    except torch.cuda.OutOfMemoryError:
        status = 'OOM'

    peak = torch.cuda.max_memory_allocated() / 2**30
    reserved = torch.cuda.max_memory_reserved() / 2**30
    peaks = [torch.zeros(1, device='cuda') + peak]
    if world > 1 and torch.distributed.is_initialized():
        allp = [torch.zeros(1, device='cuda') for _ in range(world)]
        torch.distributed.all_gather(allp, peaks[0])
        peaks = allp
    if rank == 0:
        step_s = sum(timer.times) / len(timer.times) if timer.times else None
        samples_per_step = a.batch_size * a.grad_accum * world
        res = {
            'host': socket.gethostname(), 'gpu': torch.cuda.get_device_name(0),
            'gpu_mem_total_gib': round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1),
            'num_gpus': world, 'mode': 'zero3' if a.deepspeed else ('ddp' if world > 1 else 'single'),
            'batch_size': a.batch_size, 'grad_accum': a.grad_accum, 'samples_per_step': samples_per_step,
            'tokens_per_sample': a.images * 402 + a.text_tokens, 'trainable_M': round(trainable / 1e6, 1),
            'status': status, 'measured_steps': len(timer.times),
            'step_s': round(step_s, 2) if step_s else None,
            'sec_per_sample': round(step_s / samples_per_step, 3) if step_s else None,
            'peak_alloc_gib_per_gpu': [round(float(x), 1) for x in peaks],
            'peak_reserved_gib_rank0': round(reserved, 1),
            'torch': torch.__version__, 'python': platform.python_version(),
        }
        print('\n[RESULT]', json.dumps(res, ensure_ascii=False))
        with open(a.out, 'a') as f:
            f.write(json.dumps(res, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
