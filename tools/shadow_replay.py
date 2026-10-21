#!/usr/bin/env python3
"""Recorded-sequence shadow test for the S2->S1 backend.

Streams validation episodes frame by frame through POST /infer with one
episode_id, so the backend keeps its own front history and safe_stop
previous-goal state exactly as it would on a vehicle. Control outputs are only
logged. `--mode reset_each` sends reset=true on every frame as a no-temporal
baseline for the same frames.
"""
import argparse
import collections
import gzip
import json
import re
import time
from pathlib import Path

import numpy as np
import requests

DEFAULT_MANIFEST = ('/home/work/vln_project/vla_ws_restore/vla_project/outputs/'
                    's2_train_data_reviewed_landmark/conversations_val.json')
_FRAME_RE = re.compile(r'(\d+)\.jpg$')


def frame_index(path):
    return int(_FRAME_RE.search(path).group(1))


def build_sequences(rows, episodes=None):
    by_episode = collections.defaultdict(list)
    for row in rows:
        if episodes and row['episode'] not in episodes:
            continue
        by_episode[row['episode']].append(row)
    sequences = {}
    for episode, items in by_episode.items():
        frames = collections.defaultdict(dict)
        stops = []
        for row in items:
            if row['category'] == 'stop':
                stops.append(row)
            else:
                frames[frame_index(row['source_image'])][row['category']] = row
        steps = []
        for idx in sorted(frames):
            options = frames[idx]
            # The recorded motion label is the command the driver actually executed.
            label = next(iter(options.values())).get('source_motion_label')
            category = ('safe_stop' if 'safe_stop' in options else
                        label if label in options else
                        'follow' if 'follow' in options else sorted(options)[0])
            row = options[category]
            steps.append({'frame_idx': idx, 'command': category, 'instruction': row['instruction'],
                          'front': row['source_image'],
                          'gt_goal': row.get('goal_lookdown')})
        if stops and steps:
            # STOP rows carry the instruction only inside the prompt text.
            prompt = stops[0]['conversations'][0]['value']
            match = re.search(r'Your task is to (.+?)\. ', prompt)
            steps.append({'frame_idx': steps[-1]['frame_idx'], 'command': 'stop',
                          'instruction': match.group(1) if match else 'STOP',
                          'front': steps[-1]['front'],
                          'gt_goal': None})
        sequences[episode] = steps
    return sequences


def frame_period_s(front_path):
    anno = Path(front_path).parents[2] / 'anno'
    stamps = []
    for path in sorted(anno.glob('*.json.gz'))[:40]:
        with gzip.open(path, 'rt') as f:
            stamps.append(json.load(f)['timestamp_ns'])
    if len(stamps) < 2:
        return None
    return float(np.median(np.diff(stamps)) / 1e9)


def post(url, step, episode_id, reset, timeout):
    meta = {'episode_id': episode_id, 'instruction': step['instruction'],
            'command': step['command'], 'reset': reset}
    with open(step['front'], 'rb') as f:
        started = time.perf_counter()
        resp = requests.post(url, data={'json': json.dumps(meta, ensure_ascii=False)},
                             files={'front': ('front.jpg', f, 'image/jpeg')}, timeout=timeout)
    return resp.status_code, resp.json(), (time.perf_counter() - started) * 1000


def summarize(records, period):
    out = {'frames': len(records), 'frame_period_s': period}
    out['http_status'] = dict(collections.Counter(str(r['http']) for r in records))
    out['status'] = dict(collections.Counter(r['status'] for r in records))
    out['control_decision'] = dict(collections.Counter(r['control_decision'] for r in records))
    moving = [r for r in records if r['command'] != 'stop']
    out['unexpected_stop_rate'] = (
        sum(r['control_decision'] != 'trajectory' for r in moving) / len(moving) if moving else None)
    lat = np.array([r['latency_ms'] for r in moving if r['latency_ms'] is not None])
    if lat.size:
        out['latency_ms'] = {'p50': float(np.percentile(lat, 50)), 'p95': float(np.percentile(lat, 95)),
                             'max': float(lat.max())}
    for key in ('rtt_ms', 'goal_age_ms', 's1_latency_ms', 's2_latency_ms'):
        v = np.array([r[key] for r in moving if r.get(key) is not None])
        if v.size:
            out[key] = {'p50': float(np.percentile(v, 50)), 'p95': float(np.percentile(v, 95)),
                        'max': float(v.max())}
    if records and records[-1].get('dropped_s2_frames') is not None:
        out['dropped_s2_frames_total'] = records[-1]['dropped_s2_frames']
    errs = np.array([r['goal_err_px'] for r in records if r['goal_err_px'] is not None])
    if errs.size:
        out['goal_err_px'] = {'mean': float(errs.mean()), 'median': float(np.median(errs)),
                              'p90': float(np.percentile(errs, 90)),
                              'within_50px': float((errs <= 50).mean())}
    by_cmd = collections.defaultdict(list)
    for r in records:
        if r['goal_err_px'] is not None:
            by_cmd[r['command']].append(r['goal_err_px'])
    out['goal_err_px_by_command'] = {k: float(np.median(v)) for k, v in by_cmd.items()}
    # History window: the runtime keeps only the last 64 fronts, so later frames
    # lose the episode's initial observation that training always provided.
    for name, sel in (('early_lt64', lambda r: r['step'] < 64), ('late_ge64', lambda r: r['step'] >= 64)):
        v = [r['goal_err_px'] for r in records if sel(r) and r['goal_err_px'] is not None]
        if v:
            out[f'goal_err_px_median_{name}'] = float(np.median(v))
            out[f'n_{name}'] = len(v)
    jumps, gt_jumps = [], []
    for a, b in zip(records, records[1:]):
        if a['pred_goal'] and b['pred_goal'] and a['command'] == b['command'] != 'stop':
            jumps.append(float(np.hypot(b['pred_goal'][0] - a['pred_goal'][0],
                                        b['pred_goal'][1] - a['pred_goal'][1])))
            if a['gt_goal'] and b['gt_goal']:
                gt_jumps.append(float(np.hypot(b['gt_goal'][0] - a['gt_goal'][0],
                                               b['gt_goal'][1] - a['gt_goal'][1])))
    if jumps:
        out['pred_goal_jump_px'] = {'median': float(np.median(jumps)), 'p95': float(np.percentile(jumps, 95)),
                                    'max': float(max(jumps)), 'over_150px': int(sum(j > 150 for j in jumps))}
    if gt_jumps:
        out['gt_goal_jump_px'] = {'median': float(np.median(gt_jumps)),
                                  'p95': float(np.percentile(gt_jumps, 95)), 'max': float(max(gt_jumps))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', default='http://127.0.0.1:5801')
    ap.add_argument('--manifest', default=DEFAULT_MANIFEST)
    ap.add_argument('--episodes', nargs='*')
    ap.add_argument('--mode', choices=('closed_loop', 'reset_each'), default='closed_loop')
    ap.add_argument('--max-frames', type=int)
    ap.add_argument('--rate-hz', type=float,
                    help='send frames on a real-time schedule (async backend test)')
    ap.add_argument('--timeout', type=float, default=120)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    infer_url = args.url.rstrip('/') + '/infer'
    sequences = build_sequences(json.loads(Path(args.manifest).read_text()), args.episodes)
    summary = {'mode': args.mode, 'episodes': {}}
    contract_checks = []

    for episode, steps in sequences.items():
        if args.max_frames:
            steps = steps[:args.max_frames]
        episode_id = f'shadow-{args.mode}-{episode}'
        # Contract check: a new episode without reset must be rejected with 409 + STOP.
        # A freshly started server has no active episode and accepts the first request.
        code, body, _ = post(infer_url, steps[0], episode_id, False, args.timeout)
        rejected = code == 409 and body.get('discrete_action') == 0
        contract_checks.append({'episode': episode, 'check': 'episode_change_without_reset',
                                'http': code, 'control_decision': body.get('control_decision'),
                                'pass': rejected if contract_checks or code != 200 else None})
        records = []
        log_path = out_dir / f'{args.mode}_{episode}.jsonl'
        with log_path.open('w') as log:
            t0 = time.perf_counter()
            for i, step in enumerate(steps):
                if args.rate_hz:
                    time.sleep(max(0.0, t0 + i / args.rate_hz - time.perf_counter()))
                reset = i == 0 or args.mode == 'reset_each'
                code, body, rtt = post(infer_url, step, episode_id, reset, args.timeout)
                goal = body.get('pixel_goal_lookdown')
                gt = step['gt_goal']
                traj = body.get('trajectory') or {}
                rec = {'step': i, 'frame_idx': step['frame_idx'], 'command': step['command'],
                       'http': code, 'status': body.get('status'),
                       'control_decision': body.get('control_decision'),
                       's2_text': body.get('s2_text'), 'pred_goal': goal, 'gt_goal': gt,
                       'goal_err_px': (float(np.hypot(goal[0] - gt[0], goal[1] - gt[1]))
                                       if goal and gt else None),
                       'valid_mask': traj.get('valid_candidate_mask'),
                       'selected_index': traj.get('selected_index'),
                       'selected_endpoint': (traj['selected'][-1] if traj.get('selected') else None),
                       'latency_ms': body.get('latency_ms'), 'rtt_ms': round(rtt, 1),
                       'error': body.get('error'),
                       **{k: body.get(k) for k in ('goal_age_ms', 's1_latency_ms', 's2_latency_ms',
                                                   'frame_seq', 'goal_frame_seq', 'dropped_s2_frames')
                          if k in body}}
                records.append(rec)
                log.write(json.dumps(rec, ensure_ascii=False) + '\n')
                log.flush()
                print(f"{episode} {i+1}/{len(steps)} {step['command']:<10} {rec['status']:<18} "
                      f"err={rec['goal_err_px'] if rec['goal_err_px'] is None else round(rec['goal_err_px'])} "
                      f"lat={rec['latency_ms']}", flush=True)
        summary['episodes'][episode] = summarize(records, frame_period_s(steps[0]['front']))
        (out_dir / f'summary_{args.mode}.json').write_text(
            json.dumps({**summary, 'contract_checks': contract_checks}, ensure_ascii=False, indent=2))
    print(json.dumps({**summary, 'contract_checks': contract_checks}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
