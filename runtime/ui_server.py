#!/usr/bin/env python3
import io
import json
import math
import os
from pathlib import Path

import requests
from flask import Flask, jsonify, render_template, request, send_file
from PIL import Image

RUNTIME_DIR = Path(__file__).resolve().parent
BUNDLE_DIR = RUNTIME_DIR.parent
PROJECT_DIR = BUNDLE_DIR.parents[1]
INFERENCE_URL = os.environ.get('INFERENCE_URL', 'http://127.0.0.1:5801/infer')
READY_URL = INFERENCE_URL.rsplit('/', 1)[0] + '/ready'
MANIFEST = Path(os.environ.get(
    'SAMPLE_MANIFEST',
    PROJECT_DIR / 'outputs/s2_train_data_reviewed_landmark/conversations_val.json',
))
COMMANDS = {'follow', 'turn_left', 'turn_right', 'safe_stop', 'stop'}
CAM_K = ((983.4798, 0.0, 1006.9), (0.0, 984.759, 490.492), (0.0, 0.0, 1.0))

app = Flask(__name__, template_folder=str(RUNTIME_DIR / 'templates'))


def load_catalog():
    if not MANIFEST.is_file():
        return []
    rows = json.loads(MANIFEST.read_text())
    catalog = []
    for row in rows:
        path = Path(row.get('source_image') or row.get('image', [''])[0])
        if not path.is_file():
            continue
        goal = row.get('goal_lookdown')
        if goal is None and len(row.get('conversations', [])) >= 4:
            try:
                goal = [int(x) for x in row['conversations'][-1]['value'].split()[:2]]
            except (ValueError, TypeError):
                goal = None
        catalog.append({
            'id': len(catalog),
            'category': row.get('category', 'unknown'),
            'instruction': row.get('instruction', ''),
            'source_id': row.get('id', path.name),
            'filename': path.name,
            'image_path': str(path),
            'gt_goal_lookdown': goal,
            'target_description': row.get('target_description'),
        })
    return catalog


CATALOG = load_catalog()


def public_sample(row):
    return {k: row.get(k) for k in (
        'id', 'category', 'instruction', 'source_id', 'filename',
        'gt_goal_lookdown', 'target_description')}


def project_xy(x, y, z=0.0):
    # _VLA_E2E_T_EGO2CAM followed by OpenCV camera convention.
    p0 = -0.9998 * y + 0.0218 * z - 1.0498
    p1 = x
    p2 = 0.0218 * y + 0.9998 * z - 0.8973
    if p0 <= 1e-3:
        return None
    u = CAM_K[0][0] * (p1 / p0) + CAM_K[0][2]
    v = CAM_K[1][1] * (-p2 / p0) + CAM_K[1][2]
    return [u, v]


def trajectory_projection(result):
    trajectory = result.get('trajectory') or {}
    candidates = trajectory.get('candidates') or []
    projected = []
    for candidate in candidates:
        projected.append([
            project_xy(float(p[0]), float(p[1])) if len(p) >= 2 else None
            for p in candidate
        ])
    goal = result.get('pixel_goal_lookdown')
    goal_front = [float(goal[0]), 432.0 + 0.6 * float(goal[1])] if goal else None
    valid = trajectory.get('valid_candidate_mask') or [True] * len(projected)
    closest_index, closest_distance = None, None
    if goal_front:
        for index, points in enumerate(projected):
            if index >= len(valid) or not valid[index]:
                continue
            endpoint = next((point for point in reversed(points) if point is not None), None)
            if endpoint is None:
                continue
            distance = math.hypot(endpoint[0] - goal_front[0], endpoint[1] - goal_front[1])
            if closest_distance is None or distance < closest_distance:
                closest_index, closest_distance = index, distance
    return {
        'candidates_front_px': projected,
        'goal_front_px': goal_front,
        'closest_to_goal_index': closest_index,
        'closest_to_goal_distance_px': closest_distance,
    }


@app.get('/')
def index():
    return render_template('inference_ui.html')


@app.get('/api/catalog')
def catalog():
    category = request.args.get('category', '').strip()
    items = CATALOG if not category or category == 'all' else [r for r in CATALOG if r['category'] == category]
    return jsonify({'manifest': str(MANIFEST), 'count': len(items),
                    'items': [public_sample(r) for r in items]})


@app.get('/api/image/<int:sample_id>')
def sample_image(sample_id):
    if sample_id < 0 or sample_id >= len(CATALOG):
        return jsonify({'error': 'sample not found'}), 404
    return send_file(CATALOG[sample_id]['image_path'], conditional=True)


@app.get('/api/health')
def health():
    try:
        response = requests.get(READY_URL, timeout=2)
        body = response.json()
        return jsonify({'ui': True, 'backend': body, 'backend_url': INFERENCE_URL,
                        'sample_count': len(CATALOG)}), (200 if response.ok else 503)
    except Exception as exc:
        return jsonify({'ui': True, 'backend': {'ready': False, 'error': str(exc)},
                        'backend_url': INFERENCE_URL, 'sample_count': len(CATALOG)}), 503


@app.post('/api/infer')
def infer():
    instruction = request.form.get('instruction', '').strip()
    command = request.form.get('command', '').strip()
    episode_id = request.form.get('episode_id', '').strip()
    reset = request.form.get('reset', 'true').lower() == 'true'
    if not instruction or command not in COMMANDS or not episode_id:
        return jsonify({'error': 'instruction, valid command, and episode_id are required'}), 400

    sample = None
    upload = request.files.get('front')
    if upload and upload.filename:
        image = Image.open(upload.stream).convert('RGB')
    else:
        try:
            sample_id = int(request.form.get('sample_id', '-1'))
            if sample_id < 0 or sample_id >= len(CATALOG):
                raise IndexError(sample_id)
            sample = CATALOG[sample_id]
        except (ValueError, IndexError):
            return jsonify({'error': 'select a sample or upload a front image'}), 400
        image = Image.open(sample['image_path']).convert('RGB')

    front_bytes = io.BytesIO()
    image.save(front_bytes, format='JPEG', quality=95)
    front_bytes.seek(0)
    files = {'front': ('front.jpg', front_bytes, 'image/jpeg')}
    lookdown_upload = request.files.get('lookdown')
    lookdown_bytes = None
    if lookdown_upload and lookdown_upload.filename:
        lookdown = Image.open(lookdown_upload.stream).convert('RGB')
        lookdown_bytes = io.BytesIO()
        lookdown.save(lookdown_bytes, format='JPEG', quality=95)
        lookdown_bytes.seek(0)
        files['lookdown'] = ('lookdown.jpg', lookdown_bytes, 'image/jpeg')
    meta = {'instruction': instruction, 'command': command,
            'episode_id': episode_id, 'reset': reset}
    try:
        response = requests.post(INFERENCE_URL, data={'json': json.dumps(meta, ensure_ascii=False)},
                                 files=files, timeout=300)
        try:
            result = response.json()
        except ValueError:
            result = {'error': response.text[:1000]}
    except requests.RequestException as exc:
        return jsonify({'error': f'inference backend unavailable: {exc}'}), 503

    payload = {
        'result': result,
        'projection': trajectory_projection(result),
        'sample': public_sample(sample) if sample else None,
        'image_size': list(image.size),
        'backend_status': response.status_code,
    }
    return jsonify(payload), response.status_code
