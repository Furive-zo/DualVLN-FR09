#!/usr/bin/env python3
"""Run the S2->S1 runtime over recorded validation episodes and render each frame.

Every frame shows the front image with the S2 pixel-goal (orange), the annotated
GT goal (yellow), the S1 candidates (light red) and the controller-selected
trajectory (magenta) projected with the training camera calibration, plus a
top-down panel of the same candidates. Writes per-frame JSONL, JPG frames and an
optional GIF.

  python3 tools/render_sequence.py --episodes undistort_orin_pc_data11 --out /tmp/seq --gif
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'runtime'), str(ROOT / 'tools')]

from shadow_replay import DEFAULT_MANIFEST, build_sequences  # noqa: E402

W, H = 960, 540          # front panel size (half of 1920x1080)
PANEL = 300              # top-down panel width
BEV_RANGE = (-3.0, 3.0, -0.5, 6.0)  # right_min, right_max, fwd_min, fwd_max [m]
COLORS = {'goal': (255, 150, 0), 'gt': (255, 230, 0), 'cand': (255, 140, 140),
          'sel': (230, 0, 230), 'text': (255, 255, 255)}


def font(size):
    for name in ('NanumBarunGothic.ttf', 'NanumGothic.ttf', 'NotoSansCJK-Regular.ttc', 'DejaVuSans.ttf'):
        for base in ('/usr/share/fonts/truetype/nanum/', '/usr/share/fonts/opentype/noto/',
                     '/usr/share/fonts/truetype/dejavu/', ''):
            try:
                return ImageFont.truetype(base + name, size)
            except OSError:
                continue
    return ImageFont.load_default()


def lookdown_to_front(goal):
    # runtime_contract.json: v_lookdown = (v_original - 432) / 0.6
    return goal[0], goal[1] * 0.6 + 432


def bev_xy(right, forward):
    r0, r1, f0, f1 = BEV_RANGE
    return W + (right - r0) / (r1 - r0) * PANEL, H - (forward - f0) / (f1 - f0) * H


def draw_frame(front, step, result, project, fonts):
    img = front.resize((W, H))
    canvas = Image.new('RGB', (W + PANEL, H), (24, 24, 28))
    canvas.paste(img, (0, 0))
    d = ImageDraw.Draw(canvas)
    sx, sy = W / 1920, H / 1080
    traj = result.get('trajectory') or {}
    cands = traj.get('candidates') or []
    sel = traj.get('selected_index')
    for i, cand in enumerate(cands):
        color, width = (COLORS['sel'], 5) if i == sel else (COLORS['cand'], 2)
        pts = [(0.0, 0.0)] + [(p[0], p[1]) for p in cand]
        uv = [project(x, y) for x, y in pts]
        line = [(u * sx, v * sy) for u, v in (p for p in uv if p) if 0 <= u < 1920 and 0 <= v < 1080]
        if len(line) > 1:
            d.line(line, fill=color, width=width)
        # top-down: right = +x, forward = -y
        d.line([bev_xy(x, -y) for x, y in pts], fill=color, width=width if i == sel else 2)
    for key, goal, r in (('gt', step.get('gt_goal'), 9), ('goal', result.get('pixel_goal_lookdown'), 11)):
        if goal:
            u, v = lookdown_to_front(goal)
            u, v = u * sx, v * sy
            if key == 'goal':
                d.line([(u - r, v), (u + r, v)], fill=COLORS[key], width=4)
                d.line([(u, v - r), (u, v + r)], fill=COLORS[key], width=4)
            else:
                d.ellipse([u - r, v - r, u + r, v + r], outline=COLORS[key], width=3)
    # top-down grid and ego
    r0, r1, f0, f1 = BEV_RANGE
    for f in range(0, int(f1) + 1):
        y = bev_xy(0, f)[1]
        d.line([(W, y), (W + PANEL, y)], fill=(55, 55, 62))
        d.text((W + 4, y - 14), f'{f} m', fill=(140, 140, 150), font=fonts[2])
    d.line([bev_xy(0, f0), bev_xy(0, f1)], fill=(80, 80, 90))
    ex, ey = bev_xy(0, 0)
    d.polygon([(ex, ey - 10), (ex - 7, ey + 6), (ex + 7, ey + 6)], fill=(120, 200, 255))
    decision = result.get('control_decision', '?')
    banner = (40, 140, 60) if decision == 'trajectory' else (170, 40, 40)
    d.rectangle([0, 0, W, 64], fill=(0, 0, 0))
    d.text((10, 6), f"{step['episode'].replace('undistort_orin_pc_', '')}  frame {step['frame_idx']}"
           f"   command={step['command']}", fill=COLORS['text'], font=fonts[0])
    d.text((10, 36), step['instruction'], fill=(220, 220, 220), font=fonts[1])
    d.rectangle([W - 250, 8, W - 10, 56], fill=banner)
    d.text((W - 240, 12), decision.upper(), fill=COLORS['text'], font=fonts[0])
    d.text((W - 240, 36), f"{result.get('status')}  {result.get('latency_ms', 0):.0f} ms",
           fill=COLORS['text'], font=fonts[2])
    legend = [('S2 goal', COLORS['goal']), ('GT goal', COLORS['gt']),
              ('S1 candidates', COLORS['cand']), ('selected', COLORS['sel'])]
    for i, (name, color) in enumerate(legend):
        d.rectangle([W + 10, 10 + i * 20, W + 24, 24 + i * 20], fill=color)
        d.text((W + 30, 8 + i * 20), name, fill=COLORS['text'], font=fonts[2])
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', nargs='+', required=True)
    ap.add_argument('--manifest', default=DEFAULT_MANIFEST)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--max-frames', type=int)
    ap.add_argument('--out', required=True)
    ap.add_argument('--gif', action='store_true', help='also write <episode>.gif (every 2nd frame)')
    args = ap.parse_args()

    import torch  # noqa: F401  (loaded after argparse so --help stays fast)
    from dual_model_runtime import DualVLNE2ERuntime
    from internnav.dataset.internvla_n1_lerobot_dataset import project_relative_xy_vla_e2e
    from server import model_path, release

    out = Path(args.out)
    runtime = DualVLNE2ERuntime(model_path(release['models']['system2']),
                                model_path(release['models']['system1']), args.device,
                                release['runtime']['num_trajectory_candidates'],
                                release['runtime']['num_inference_steps'])
    fonts = (font(22), font(18), font(14))
    sequences = build_sequences(json.loads(Path(args.manifest).read_text()), set(args.episodes))
    for episode in args.episodes:
        steps = sequences[episode][:args.max_frames]
        frame_dir = out / episode
        frame_dir.mkdir(parents=True, exist_ok=True)
        frames = []
        with (out / f'{episode}.jsonl').open('w') as log:
            for i, step in enumerate(steps):
                front = Image.open(step['front']).convert('RGB')
                t0 = time.perf_counter()
                result = runtime.infer(front, step['instruction'], step['command'], reset=(i == 0))
                result.setdefault('latency_ms', (time.perf_counter() - t0) * 1000)
                step = {**step, 'episode': episode}
                log.write(json.dumps({'step': i, **{k: step[k] for k in ('frame_idx', 'command', 'instruction',
                                                                          'gt_goal')}, **result},
                                     ensure_ascii=False) + '\n')
                canvas = draw_frame(front, step, result, project_relative_xy_vla_e2e, fonts)
                canvas.save(frame_dir / f'{i:04d}.jpg', quality=85)
                frames.append(canvas)
                print(f"{episode} {i + 1}/{len(steps)} {step['command']:<10} {result['status']}", flush=True)
        if args.gif and frames:
            small = [f.resize((f.width * 2 // 3, f.height * 2 // 3)).quantize(colors=128) for f in frames[::2]]
            small[0].save(out / f'{episode}.gif', save_all=True, append_images=small[1:],
                          duration=500, loop=0, optimize=True)


if __name__ == '__main__':
    main()
