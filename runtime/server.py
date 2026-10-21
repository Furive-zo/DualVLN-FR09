#!/usr/bin/env python3
import argparse
import io
import json
import os
import traceback
from pathlib import Path

from flask import Flask, jsonify, request
from PIL import Image

from dual_model_runtime import BUNDLE_DIR, PROJECT_DIR, DualVLNE2ERuntime

app = Flask(__name__)
release = json.loads((BUNDLE_DIR / 'config/release_manifest.json').read_text())
runtime = None
active_episode_id = None


def model_path(spec):
    portable = BUNDLE_DIR / spec['portable_path']
    return portable if portable.is_dir() else PROJECT_DIR / spec['source_path']


@app.get('/health')
def health():
    return jsonify({'ok': True, 'release': release['release']})


@app.get('/ready')
def ready():
    return jsonify({'ready': runtime is not None, 'release': release['release']}), (200 if runtime else 503)


@app.post('/reset')
def reset():
    global active_episode_id
    if runtime is None:
        return jsonify({'status': 'not_ready', 'control_decision': 'stop', 'discrete_action': 0}), 503
    meta = request.get_json(silent=True) or {}
    runtime.reset()
    active_episode_id = meta.get('episode_id')
    return jsonify({'ok': True, 'episode_id': active_episode_id})


@app.post('/infer')
def infer():
    global active_episode_id
    if runtime is None:
        return jsonify({'status': 'not_ready', 'control_decision': 'stop', 'discrete_action': 0}), 503
    if 'front' not in request.files:
        return jsonify({'error': 'multipart field front is required', 'control_decision': 'stop',
                        'discrete_action': 0}), 400
    try:
        meta = json.loads(request.form.get('json', '{}'))
    except json.JSONDecodeError:
        return jsonify({'error': 'multipart json field is invalid', 'control_decision': 'stop',
                        'discrete_action': 0}), 400
    instruction = str(meta.get('instruction', '')).strip()
    command = str(meta.get('command', '')).strip()
    episode_id = str(meta.get('episode_id', '')).strip()
    reset_requested = bool(meta.get('reset', False))
    if not instruction or command not in {'follow', 'turn_left', 'turn_right', 'safe_stop', 'stop'}:
        return jsonify({'error': 'instruction and valid command are required',
                        'control_decision': 'stop', 'discrete_action': 0}), 400
    if not episode_id:
        return jsonify({'error': 'episode_id is required', 'control_decision': 'stop',
                        'discrete_action': 0}), 400
    if active_episode_id not in (None, episode_id) and not reset_requested:
        return jsonify({'error': 'episode changed; resend with reset=true',
                        'active_episode_id': active_episode_id,
                        'control_decision': 'stop', 'discrete_action': 0}), 409
    try:
        front = Image.open(io.BytesIO(request.files['front'].read())).convert('RGB')
        lookdown = None
        if 'lookdown' in request.files:
            lookdown = Image.open(io.BytesIO(request.files['lookdown'].read())).convert('RGB')
        result = runtime.infer(front, instruction, command, lookdown, reset_requested)
        active_episode_id = episode_id
        result.update(episode_id=episode_id, release=release['release'])
        return jsonify(result)
    except Exception as exc:
        app.logger.error('inference failed: %s\n%s', exc, traceback.format_exc())
        return jsonify({'status': 'runtime_error', 'error': str(exc),
                        'control_decision': 'stop', 'discrete_action': 0,
                        'episode_id': episode_id, 'release': release['release']}), 500


def initialize_runtime(device=None):
    global runtime
    if runtime is None:
        device = device or release['runtime']['device']
        s2 = model_path(release['models']['system2'])
        s1 = model_path(release['models']['system1'])
        if os.environ.get('ASYNC_S2', '0') == '1':
            from async_runtime import AsyncDualVLNRuntime
            runtime = AsyncDualVLNRuntime(
                s2, s1, os.environ.get('S2_DEVICE', 'cuda:0'), os.environ.get('S1_DEVICE', 'cuda:1'),
                release['runtime']['num_trajectory_candidates'],
                release['runtime']['num_inference_steps'],
                max_goal_age_s=float(os.environ.get('MAX_GOAL_AGE_S', '2.25')),
                s1_gpu_preprocess=os.environ.get('S1_GPU_PREPROCESS', '1') == '1')
        else:
            runtime = DualVLNE2ERuntime(s2, s1, device,
                                        release['runtime']['num_trajectory_candidates'],
                                        release['runtime']['num_inference_steps'])
    return runtime


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=5801)
    ap.add_argument('--device', default=release['runtime']['device'])
    args = ap.parse_args()
    initialize_runtime(args.device)
    app.run(host=args.host, port=args.port, threaded=False)


if __name__ == '__main__':
    main()
