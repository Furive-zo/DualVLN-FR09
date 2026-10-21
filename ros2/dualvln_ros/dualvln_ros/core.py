"""ROS-independent client logic for the DualVLN inference server.

Kept free of rclpy so it can be tested on a machine without ROS 2
(see test/test_core.py and tools/ros2_core_bench.py).
"""
import io
import json
import math
import time

import requests

# Instruction text the models were trained with (config/runtime_contract.json trained_instructions).
TRAINED_INSTRUCTIONS = {
    'follow': '도로를 따라 주행하라',
    'turn_left': '좌회전하라',
    'turn_right': '우회전하라',
    'stop': '정지하라',
}
SAFE_STOP_INSTRUCTIONS = (
    '왼쪽에서 첫 번째 빈 주차면에 안전하게 진입하여 정차하라',
    '왼쪽에서 두 번째 빈 주차면에 안전하게 진입하여 정차하라',
    '오른쪽에서 두 번째 빈 주차면에 안전하게 진입하여 정차하라',
    '오른쪽에서 세 번째 빈 주차면에 안전하게 진입하여 정차하라',
    '오른쪽에서 네 번째 빈 주차면에 안전하게 진입하여 정차하라',
    '왼쪽에 주차된 차량 기준 2칸 왼쪽의 빈 주차면에 안전하게 진입하여 정차하라',
)
COMMANDS = ('follow', 'turn_left', 'turn_right', 'safe_stop', 'stop')


def parse_command(text):
    """Accept 'follow' or '{"command": "safe_stop", "instruction": "..."}'.

    Returns (command, instruction). Raises ValueError on anything the server would reject.
    """
    text = text.strip()
    if text.startswith('{'):
        data = json.loads(text)
        command = str(data.get('command', '')).strip()
        instruction = str(data.get('instruction') or TRAINED_INSTRUCTIONS.get(command, '')).strip()
    else:
        command, instruction = text, TRAINED_INSTRUCTIONS.get(text, '')
    if command not in COMMANDS:
        raise ValueError(f'unknown command {command!r}; expected one of {COMMANDS}')
    if not instruction:
        raise ValueError(f'command {command!r} needs an instruction (e.g. {SAFE_STOP_INSTRUCTIONS[2]!r})')
    return command, instruction


def model_to_base_link(pose):
    """Model pose [x_right, y (forward = -y), theta] -> REP-103 (x_forward, y_left, yaw_left)."""
    x, y, theta = pose
    yaw = -(theta + math.pi / 2)
    return -y, -x, math.atan2(math.sin(yaw), math.cos(yaw))


class DualVLNClient:
    """One drive = one episode. Thread-unsafe; call from a single worker."""

    def __init__(self, server, timeout_s=0.8):
        self.server = server.rstrip('/')
        self.timeout_s = timeout_s
        self.session = requests.Session()
        self.episode_id = None
        self.reset_pending = True

    def ready(self):
        try:
            return self.session.get(self.server + '/ready', timeout=2).status_code == 200
        except requests.RequestException:
            return False

    def new_episode(self, episode_id=None):
        self.episode_id = episode_id or time.strftime('drive-%Y%m%d-%H%M%S')
        self.reset_pending = True
        return self.episode_id

    def infer(self, jpeg, command, instruction, timeout_s=None):
        """POST one frame. Always returns a dict with control_decision; never raises."""
        if self.episode_id is None:
            self.new_episode()
        meta = {'episode_id': self.episode_id, 'instruction': instruction,
                'command': command, 'reset': self.reset_pending}
        started = time.monotonic()
        try:
            resp = self.session.post(
                self.server + '/infer', data={'json': json.dumps(meta, ensure_ascii=False)},
                files={'front': ('front.jpg', jpeg, 'image/jpeg')},
                timeout=timeout_s or self.timeout_s)
            body = resp.json()
            body['http_status'] = resp.status_code
            if resp.status_code == 200:
                self.reset_pending = False
        except (requests.RequestException, ValueError) as exc:
            # No answer is not a STOP answer: the caller keeps the last path until its watchdog fires.
            body = {'status': f'request_failed:{exc.__class__.__name__}', 'http_status': None,
                    'control_decision': None}
        body['rtt_ms'] = round((time.monotonic() - started) * 1000, 1)
        return body


def usable_trajectory(body):
    return (body.get('http_status') == 200 and body.get('control_decision') == 'trajectory'
            and (body.get('trajectory') or {}).get('selected') is not None)


def decide(body, since_valid_s, watchdog_s):
    """'follow' a fresh trajectory, 'stop' on any server STOP or watchdog, else 'hold' the last path."""
    if usable_trajectory(body):
        return 'follow'
    if body.get('control_decision') == 'stop' or body.get('http_status') not in (None, 200):
        return 'stop'
    return 'stop' if since_valid_s > watchdog_s else 'hold'


def encode_jpeg(image, quality=95):
    buf = io.BytesIO()
    image.save(buf, format='JPEG', quality=quality)
    return buf.getvalue()
