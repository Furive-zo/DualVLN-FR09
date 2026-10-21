#!/usr/bin/env python3
"""Stage-level latency breakdown of DualVLNE2ERuntime.infer on one recorded frame.

Wraps the runtime's model calls with CUDA-synchronized timers; it does not change
inference behaviour. Earlier frames only fill the front history; the last frame is
timed. Run on an idle GPU (CUDA_VISIBLE_DEVICES) for clean numbers.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bundle', default=str(BUNDLE))
    ap.add_argument('--frames', nargs='+', required=True, help='consecutive front images')
    ap.add_argument('--instruction', default='좌회전하라')
    ap.add_argument('--command', default='turn_left')
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--out')
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.bundle) / 'runtime'))
    import server
    rt = server.initialize_runtime('cuda:0')
    state = {'on': False, 'times': defaultdict(float)}

    def timed(name, fn):
        def wrapper(*a, **kw):
            if not state['on']:
                return fn(*a, **kw)
            torch.cuda.synchronize()
            t = time.perf_counter()
            out = fn(*a, **kw)
            torch.cuda.synchronize()
            state['times'][name] += (time.perf_counter() - t) * 1000
            return out
        return wrapper

    visual = getattr(rt.s2_model.model, 'visual', None) or rt.s2_model.visual
    visual.forward = timed('s2_vit', visual.forward)
    rt.s2_model.generate = timed('s2_generate', rt.s2_model.generate)
    rt.s1_model.generate_latents = timed('s1_latents', rt.s1_model.generate_latents)
    rt.s1_model.generate_traj = timed('s1_diffusion', rt.s1_model.generate_traj)
    rt._processor_inputs = timed('preprocess', rt._processor_inputs)

    frames = [Image.open(p).convert('RGB') for p in args.frames]
    runs = []
    for rep in range(args.repeats + 1):  # rep 0 is warm-up
        rt.reset()
        for frame in frames[:-1]:
            rt.infer(frame, args.instruction, args.command)
        state['on'], state['times'] = True, defaultdict(float)
        torch.cuda.synchronize()
        t = time.perf_counter()
        res = rt.infer(frames[-1], args.instruction, args.command)
        torch.cuda.synchronize()
        state['on'] = False
        if rep:
            runs.append({'total': (time.perf_counter() - t) * 1000, **state['times']})
    report = {'history_len': len(frames) - 1, 'command': args.command,
              'gpu': torch.cuda.get_device_name(), 's2_text': res.get('s2_text'),
              'median_ms': {k: round(float(np.median([r.get(k, 0.0) for r in runs])), 1)
                            for k in runs[0]}}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
