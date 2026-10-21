#!/usr/bin/env python3
"""Reference 2 Hz vehicle-side client for the DualVLN inference server.

The loop captures a front frame, POSTs it to /infer with the current command and
instruction, and hands the selected trajectory to the controller. Anything other
than a fresh `control_decision == "trajectory"` response is a STOP, and a
watchdog stops the vehicle when no valid trajectory arrived within --watchdog-s.

Camera and controller are stubs: replace FrameSource / Controller with the
vehicle's camera driver (undistorted 1920x1080 front image) and path follower.
For bench tests, --replay-dir streams JPEGs from a recorded folder instead.

  python3 tools/vehicle_client_example.py --server http://<server>:5801 \
      --replay-dir /path/to/camera/rgb_front --command follow
"""
import argparse
import io
import json
import time
from pathlib import Path

import requests
from PIL import Image

# Instruction text the models were trained with (see docs/REALTIME_VEHICLE_KO.md).
INSTRUCTIONS = {
    'follow': '도로를 따라 주행하라',
    'turn_left': '좌회전하라',
    'turn_right': '우회전하라',
    'stop': '정지하라',
    # safe_stop needs a target description, e.g. '오른쪽에서 두 번째 빈 주차면에 안전하게 진입하여 정차하라'
}


class FrameSource:
    """Replace with the camera driver. Must return an undistorted 1920x1080 RGB PIL image."""

    def __init__(self, replay_dir=None):
        self.files = sorted(Path(replay_dir).glob('*.jpg')) if replay_dir else []
        self.index = 0

    def read(self):
        if not self.files:
            raise NotImplementedError('connect the vehicle camera here')
        if self.index >= len(self.files):
            return None
        image = Image.open(self.files[self.index]).convert('RGB')
        self.index += 1
        return image


class Controller:
    """Replace with the vehicle path follower."""

    def follow(self, path, captured_at):
        # path: 4 poses [x_right_m, y_m (forward is -y), theta_rad] relative to the vehicle
        # pose at captured_at. Compensate for motion since captured_at with odometry.
        print(f'  -> follow path end=({path[-1][0]:+.2f} m right, {-path[-1][1]:.2f} m fwd)')

    def stop(self, reason):
        print(f'  -> STOP ({reason})')


def infer(session, url, image, meta, timeout):
    buf = io.BytesIO()
    image.save(buf, format='JPEG', quality=95)
    resp = session.post(url, data={'json': json.dumps(meta, ensure_ascii=False)},
                        files={'front': ('front.jpg', buf.getvalue(), 'image/jpeg')}, timeout=timeout)
    return resp.status_code, resp.json()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--server', default='http://127.0.0.1:5801')
    ap.add_argument('--command', default='follow',
                    choices=['follow', 'turn_left', 'turn_right', 'safe_stop', 'stop'])
    ap.add_argument('--instruction', help='defaults to the trained phrase for --command')
    ap.add_argument('--episode-id', default=time.strftime('drive-%Y%m%d-%H%M%S'))
    ap.add_argument('--rate-hz', type=float, default=2.0)
    ap.add_argument('--timeout-s', type=float, default=0.8,
                    help='HTTP timeout; a late response is dropped and the watchdog decides')
    ap.add_argument('--watchdog-s', type=float, default=1.0, help='STOP if no valid trajectory for this long')
    ap.add_argument('--replay-dir', help='bench test: stream JPEGs from this folder')
    ap.add_argument('--warmup', type=int, default=6,
                    help='requests sent with one frame before driving (first S1 call takes ~1.4 s)')
    args = ap.parse_args()
    instruction = args.instruction or INSTRUCTIONS.get(args.command)
    if not instruction:
        ap.error('--instruction is required for safe_stop')

    session = requests.Session()
    ready = session.get(args.server + '/ready', timeout=5)
    if ready.status_code != 200:
        raise SystemExit(f'server not ready: {ready.text}')
    camera, controller = FrameSource(args.replay_dir), Controller()
    if args.warmup:
        # Warm-up episode: lets S2 produce a goal and S1 run its first (slow) CUDA calls.
        frame = camera.read()
        for i in range(args.warmup):
            meta = {'episode_id': 'warmup', 'instruction': instruction, 'command': args.command,
                    'reset': i == 0}
            infer(session, args.server + '/infer', frame, meta, timeout=30)
            time.sleep(0.5)
    last_valid = time.monotonic()
    reset = True  # first request of a drive starts a new episode on the server
    period = 1.0 / args.rate_hz
    next_tick = time.monotonic()
    while True:
        next_tick += period
        image = camera.read()
        if image is None:
            break
        captured_at = time.monotonic()
        meta = {'episode_id': args.episode_id, 'instruction': instruction,
                'command': args.command, 'reset': reset}
        try:
            code, body = infer(session, args.server + '/infer', image, meta, args.timeout_s)
            body['rtt_ms'] = round((time.monotonic() - captured_at) * 1000, 1)
            reset = False
        except (requests.RequestException, ValueError) as exc:
            code, body = None, {'status': f'request_failed: {exc.__class__.__name__}'}
        if code == 200 and body.get('control_decision') == 'trajectory':
            controller.follow(body['trajectory']['selected'], captured_at)
            last_valid = time.monotonic()
        elif time.monotonic() - last_valid > args.watchdog_s or body.get('discrete_action') == 0:
            controller.stop(body.get('status'))
        print(f"[{args.episode_id}] http={code} status={body.get('status')} "
              f"rtt={body.get('rtt_ms')} ms server={body.get('latency_ms')} ms goal_age={body.get('goal_age_ms')} ms", flush=True)
        time.sleep(max(0.0, next_tick - time.monotonic()))
    controller.stop('end of stream')


if __name__ == '__main__':
    main()
