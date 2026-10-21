#!/usr/bin/env python3
"""Detailed per-stage latency of DualVLNE2ERuntime.infer on recorded sequences.

Every stage is wrapped with torch.cuda.synchronize() timers (adds a little sync
overhead, inference outputs are unchanged). Conditions stream real consecutive
frames so history / safe_stop prior are the same as in the shadow test:

  hist0      reset=true on every frame (no history, no prior)
  hist8      one reset, measured after >= 9 frames (8-frame sampled history)
  safe_prior safe_stop episode, measured after >= 9 frames (7 history + prior)

`--runtime-dir` selects which dual_model_runtime.py to profile (rc2 bundle or
the deployment source with the preprocessing cache).
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

BUNDLE = Path(__file__).resolve().parents[1]
DRIVE = '/home/work/.datasets/VLA-E2E/20260515/undistort_orin_pc_data11/camera/rgb_front'
PARK = '/home/work/.datasets/VLA-E2E/20260519_park_forward/undistort_orin_pc_data901/camera/rgb_front'
PARK_INSTRUCTION = '오른쪽에서 세 번째 빈 주차면에 안전하게 진입하여 정차하라'


class Profiler:
    def __init__(self):
        self.on = False
        self.frame = defaultdict(float)
        self.calls = defaultdict(int)

    def wrap(self, name, fn):
        def wrapper(*a, **kw):
            if not self.on:
                return fn(*a, **kw)
            torch.cuda.synchronize()
            t = time.perf_counter()
            out = fn(*a, **kw)
            torch.cuda.synchronize()
            self.frame[name] += (time.perf_counter() - t) * 1000
            self.calls[name] += 1
            return out
        return wrapper

    def patch(self, obj, attr, name):
        setattr(obj, attr, self.wrap(name, getattr(obj, attr)))


def instrument(m, rt, prof):
    s2, s1 = rt.s2_model, rt.s1_model
    # Preprocessing: attribute sub-stages to whichever model's inputs are being built.
    ctx = {'tag': 'pre'}

    def ctx_wrap(attr_name, fn):
        def wrapper(*a, **kw):
            return prof.wrap(f"{ctx['tag']}.{attr_name}", fn)(*a, **kw)
        return wrapper
    for proc in (rt.s2_processor, rt.s1_processor):
        proc.apply_chat_template = ctx_wrap('pre.chat_template', proc.apply_chat_template)
        proc.image_processor.preprocess = ctx_wrap('pre.image_processor', proc.image_processor.preprocess)
    orig_inputs = rt._processor_inputs

    def processor_inputs(processor, *a, **kw):
        ctx['tag'] = 's2' if processor is rt.s2_processor else 's1'
        try:
            return prof.wrap(f"{ctx['tag']}.pre.total", orig_inputs)(processor, *a, **kw)
        finally:
            ctx['tag'] = 'pre'
    rt._processor_inputs = processor_inputs
    # S2 model
    prof.patch(s2, 'generate', 's2.generate')
    prof.patch(s2.model.visual, 'forward', 's2.vit')
    lm = s2.model.language_model
    orig_lm = lm.forward

    def lm_forward(*a, **kw):
        name = 's2.llm.prefill' if prof.calls['s2.llm.prefill'] == 0 else 's2.llm.decode'
        return prof.wrap(name, orig_lm)(*a, **kw)
    lm.forward = lm_forward
    prof.patch(rt.s2_processor.tokenizer, 'decode', 's2.post.detokenize')
    prof.patch(m, 'parse_s2_output', 's2.post.parse')
    prof.patch(m, 'pseudo_lookdown_crop', 'pre.pseudo_lookdown_crop')
    # S1 model
    prof.patch(s1, 'generate_latents', 's1.latents')
    prof.patch(s1.model.visual, 'forward', 's1.vit')
    prof.patch(s1.model.language_model, 'forward', 's1.llm.forward')
    prof.patch(s1, 'generate_traj', 's1.traj')
    g = s1.get_model()
    prof.patch(g.cond_projector, 'forward', 's1.traj.cond_projector')
    prof.patch(g.rgb_model, 'get_intermediate_layers', 's1.traj.depthanything_vits')
    prof.patch(g.memory_encoder, 'forward', 's1.traj.memory_encoder')
    prof.patch(g.rgb_resampler, 'forward', 's1.traj.rgb_resampler')
    prof.patch(g.traj_dit, 'forward', 's1.traj.dit_step')
    prof.patch(m, 'physical_trajectories', 's1.post.physical_check')


def run_condition(rt, prof, name, frames, instruction, command, reset_each, warm, measure):
    rows = []
    rt.reset()
    for i, path in enumerate(frames[:warm + measure]):
        t = time.perf_counter()
        img = Image.open(path).convert('RGB')
        decode_ms = (time.perf_counter() - t) * 1000
        prof.on = i >= warm
        prof.frame, prof.calls = defaultdict(float), defaultdict(int)
        torch.cuda.synchronize()
        t = time.perf_counter()
        res = rt.infer(img, instruction, command, reset=reset_each or i == 0)
        torch.cuda.synchronize()
        total = (time.perf_counter() - t) * 1000
        if prof.on:
            rows.append({'total': total, 'jpeg_decode': decode_ms, 'history_len': len(rt.history) - 1,
                         'status': res['status'], 's2_tokens': prof.calls['s2.llm.decode'] + 1,
                         'dit_steps': prof.calls['s1.traj.dit_step'],
                         'n_images_processed': prof.calls['s2.pre.image_processor'] + prof.calls['s1.pre.image_processor'], **prof.frame})
    prof.on = False
    return rows


def stats(rows):
    keys = sorted({k for r in rows for k in r if isinstance(r[k], (int, float))})
    out = {}
    for k in keys:
        v = np.array([r.get(k, 0.0) for r in rows], dtype=float)
        out[k] = {'mean': round(float(v.mean()), 1), 'p50': round(float(np.median(v)), 1),
                  'p95': round(float(np.percentile(v, 95)), 1), 'std': round(float(v.std()), 1)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle', default=str(BUNDLE))
    ap.add_argument('--runtime-dir', required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--measure', type=int, default=12)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    bundle = Path(args.bundle)
    sys.path.insert(0, str(bundle / 'runtime'))
    sys.path.insert(0, str(Path(args.runtime_dir)))
    import dual_model_runtime as m
    t = time.perf_counter()
    rt = m.DualVLNE2ERuntime(bundle / 'models/system2', bundle / 'models/system1', 'cuda:0')
    load_s = time.perf_counter() - t
    prof = Profiler()
    instrument(m, rt, prof)

    drive = [f'{DRIVE}/{i:05d}.jpg' for i in range(30, 60)]
    park = [f'{PARK}/{i:05d}.jpg' for i in range(0, 25)]
    # Warm-up (CUDA kernels, allocator) on unrelated frames
    run_condition(rt, prof, 'warmup', drive, '좌회전하라', 'turn_left', False, 4, 0)
    conditions = {
        'hist0_turn_left': (drive, '좌회전하라', 'turn_left', True, 2),
        'hist8_turn_left': (drive, '좌회전하라', 'turn_left', False, 9),
        'hist8_follow': (drive, '전방 경로를 따라 주행하라', 'follow', False, 9),
        'safe_prior_safe_stop': (park, PARK_INSTRUCTION, 'safe_stop', False, 9),
    }
    report = {'label': args.label, 'runtime': str(Path(args.runtime_dir) / 'dual_model_runtime.py'),
              'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
              'model_load_s': round(load_s, 1), 'conditions': {}}
    for name, (frames, ins, cmd, reset_each, warm) in conditions.items():
        rows = run_condition(rt, prof, name, frames, ins, cmd, reset_each, warm, args.measure)
        report['conditions'][name] = {'n': len(rows), 'stats_ms': stats(rows),
                                      'status': sorted({r['status'] for r in rows})}
        print(name, 'total p50', report['conditions'][name]['stats_ms']['total']['p50'], flush=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
