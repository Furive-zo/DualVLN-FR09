"""ROS 2 node: front camera -> DualVLN inference server (HTTP) -> nav_msgs/Path + stop flag.

The models run on the GPU inference server (scripts/run_server_async.sh); this node
sends each new rectified front image (at most rate_hz) with the current
command/instruction and publishes the result. See docs/ROS2_KO.md.

Subscribes
  image_topic (sensor_msgs/Image or CompressedImage)  rectified 1920x1080 front camera
  ~/command   (std_msgs/String)  'follow' | 'turn_left' | 'turn_right' | 'stop'
                                 or JSON {"command": "safe_stop", "instruction": "..."}
Publishes
  ~/path        (nav_msgs/Path)               selected trajectory, 4 poses, REP-103
  ~/stop        (std_msgs/Bool)               True = vehicle must stop (10 Hz heartbeat)
  ~/status      (std_msgs/String)             JSON: status, latencies, goal age, episode
  ~/candidates  (visualization_msgs/MarkerArray)  all candidates + S2 goal text, for RViz
Services
  ~/reset       (std_srvs/Trigger)            start a new drive (new episode_id, history cleared)
"""
import io
import json
import math
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from PIL import Image as PILImage
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray

from dualvln_ros.core import DualVLNClient, decide, encode_jpeg, model_to_base_link, parse_command

EXPECTED_SIZE = (1920, 1080)


def yaw_quaternion(yaw):
    return 0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)


class DualVLNInferenceNode(Node):
    def __init__(self):
        super().__init__('dualvln_inference')
        p = self.declare_parameter
        self.server = p('server_url', 'http://127.0.0.1:5801').value
        image_topic = p('image_topic', '/camera/front/image_rect/compressed').value
        compressed = p('compressed', True).value
        self.rate_hz = p('rate_hz', 2.0).value
        self.watchdog_s = p('watchdog_s', 1.0).value
        self.frame_id = p('frame_id', 'base_link').value
        self.warmup = p('warmup', 6).value
        self.jpeg_quality = p('jpeg_quality', 95).value
        command, instruction = parse_command(p('initial_command', 'stop').value)
        self.client = DualVLNClient(self.server, p('timeout_s', 0.8).value)

        self.lock = threading.Lock()
        self.command, self.instruction = command, instruction
        self.latest = None          # last image msg (used for warm-up)
        self.busy = False
        self.last_sent = 0.0
        self.ready = False
        self.last_valid = time.monotonic()
        self.stopped = True

        sensor_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(CompressedImage if compressed else Image, image_topic,
                                 self.on_image, sensor_qos)
        self.create_subscription(String, '~/command', self.on_command, 10)
        self.path_pub = self.create_publisher(Path, '~/path', 10)
        self.stop_pub = self.create_publisher(Bool, '~/stop', 10)
        self.status_pub = self.create_publisher(String, '~/status', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '~/candidates', 10)
        self.create_service(Trigger, '~/reset', self.on_reset)
        # Inference is driven by image arrival; this timer is only the stop-flag heartbeat/watchdog.
        self.create_timer(0.1, self.on_heartbeat)
        threading.Thread(target=self.startup, daemon=True).start()
        self.get_logger().info(f'server={self.server} image={image_topic} command={command} '
                               f'instruction={instruction}')

    # ---------------------------------------------------------------- inputs
    def on_image(self, msg):
        now = time.monotonic()
        with self.lock:
            self.latest = msg
            # Send each new frame at most rate_hz; a frame arriving while a request is in
            # flight is skipped (the server only sees the frames it is sent).
            if not self.ready or self.busy or now - self.last_sent < 0.75 / self.rate_hz:
                return
            self.busy, self.last_sent = True, now
            command, instruction = self.command, self.instruction
        threading.Thread(target=self.run_once, args=(msg, command, instruction), daemon=True).start()

    def on_command(self, msg):
        try:
            command, instruction = parse_command(msg.data)
        except ValueError as exc:
            self.get_logger().error(f'rejected command: {exc}')
            return
        with self.lock:
            changed = (command, instruction) != (self.command, self.instruction)
            self.command, self.instruction = command, instruction
        if changed:
            # Same episode: S2 keeps its history; expect ~1 s of awaiting_s2_goal STOP.
            self.get_logger().info(f'command -> {command}: {instruction}')

    def on_reset(self, _request, response):
        with self.lock:
            episode = self.client.new_episode()
        response.success, response.message = True, episode
        self.get_logger().info(f'new episode {episode}')
        return response

    def to_jpeg(self, msg):
        if isinstance(msg, CompressedImage) and 'jpeg' in msg.format.lower():
            data = bytes(msg.data)
            size = PILImage.open(io.BytesIO(data)).size  # header only
            return data, size
        if isinstance(msg, CompressedImage):
            image = PILImage.open(io.BytesIO(bytes(msg.data))).convert('RGB')
        else:
            channels = {'rgb8': 3, 'bgr8': 3, 'rgba8': 4, 'bgra8': 4}.get(msg.encoding)
            if channels is None:
                raise ValueError(f'unsupported encoding {msg.encoding}')
            arr = np.frombuffer(bytes(msg.data), np.uint8).reshape(msg.height, msg.step)
            arr = arr[:, :msg.width * channels].reshape(msg.height, msg.width, channels)[..., :3]
            if msg.encoding.startswith('bgr'):
                arr = arr[..., ::-1]
            image = PILImage.fromarray(np.ascontiguousarray(arr))
        return encode_jpeg(image, self.jpeg_quality), image.size

    # ---------------------------------------------------------------- loop
    def startup(self):
        while rclpy.ok() and not self.client.ready():
            self.get_logger().info('waiting for inference server /ready ...', throttle_duration_sec=5.0)
            time.sleep(1.0)
        while rclpy.ok() and self.warmup and self.latest is None:
            time.sleep(0.2)
        if self.warmup:
            # First S1 CUDA calls take >1 s; spend them on a throw-away episode.
            jpeg, _ = self.to_jpeg(self.latest)
            self.client.new_episode('warmup')
            command = self.command if self.command != 'stop' else 'follow'
            instruction = self.instruction if self.command != 'stop' else parse_command('follow')[1]
            for _ in range(self.warmup):
                self.client.infer(jpeg, command, instruction, timeout_s=30)
                time.sleep(0.5)
        with self.lock:
            self.client.new_episode()
            self.ready = True
        self.get_logger().info(f'ready, episode {self.client.episode_id}')

    def on_heartbeat(self):
        with self.lock:
            if self.ready and not self.stopped and time.monotonic() - self.last_valid > self.watchdog_s:
                self.stopped = True  # no valid trajectory (or no image) for watchdog_s
                self.get_logger().warn('watchdog: no valid trajectory, STOP')
            stop = self.stopped or not self.ready
        self.publish_stop(stop)

    def run_once(self, msg, command, instruction):
        stamp = msg.header.stamp
        try:
            jpeg, size = self.to_jpeg(msg)
            if size != EXPECTED_SIZE:
                self.get_logger().warn(f'image {size} != {EXPECTED_SIZE}; model trained on '
                                       'rectified 1920x1080', throttle_duration_sec=10.0)
            body = self.client.infer(jpeg, command, instruction)
            body['command'] = command
            self.apply_result(body, stamp)
        except Exception as exc:  # never let the worker die silently
            self.get_logger().error(f'inference failed: {exc}')
            with self.lock:
                self.stopped = True
        finally:
            with self.lock:
                self.busy = False

    def apply_result(self, body, stamp):
        now = time.monotonic()
        with self.lock:
            action = decide(body, now - self.last_valid, self.watchdog_s)
            if action == 'follow':
                self.last_valid = now
                self.stopped = False
            elif action == 'stop':
                self.stopped = True
            stopped = self.stopped
        if action == 'follow':
            self.publish_path(body['trajectory']['selected'], stamp)
        self.publish_stop(stopped)
        self.publish_markers(body, stamp)
        keys = ('status', 'control_decision', 'http_status', 'rtt_ms', 'latency_ms', 's1_latency_ms',
                's2_latency_ms', 'goal_age_ms', 'dropped_s2_frames', 'pixel_goal_lookdown', 'episode_id')
        status = {k: body.get(k) for k in keys}
        status.update(action=action, command=body['command'])
        self.status_pub.publish(String(data=json.dumps(status, ensure_ascii=False)))

    # ---------------------------------------------------------------- outputs
    def poses(self, trajectory, stamp):
        out = [(0.0, 0.0, 0.0)] + [model_to_base_link(p) for p in trajectory]
        msgs = []
        for x, y, yaw in out:
            pose = PoseStamped()
            pose.header.stamp, pose.header.frame_id = stamp, self.frame_id
            pose.pose.position.x, pose.pose.position.y = float(x), float(y)
            q = yaw_quaternion(yaw)
            (pose.pose.orientation.x, pose.pose.orientation.y,
             pose.pose.orientation.z, pose.pose.orientation.w) = q
            msgs.append(pose)
        return msgs

    def publish_path(self, trajectory, stamp):
        # stamp = image capture time: the path is relative to base_link at that instant.
        # A controller compensates the ~0.4 s round trip with tf (odom) at this stamp.
        path = Path()
        path.header.stamp, path.header.frame_id = stamp, self.frame_id
        path.poses = self.poses(trajectory, stamp)
        self.path_pub.publish(path)

    def publish_stop(self, stop):
        try:
            self.stop_pub.publish(Bool(data=bool(stop)))
        except Exception:  # context already shut down (Ctrl-C)
            pass

    def publish_markers(self, body, stamp):
        markers = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)
        traj = body.get('trajectory') or {}
        for i, cand in enumerate(traj.get('candidates') or []):
            m = Marker()
            m.header.stamp, m.header.frame_id = stamp, self.frame_id
            m.ns, m.id, m.type, m.action = 'candidates', i, Marker.LINE_STRIP, Marker.ADD
            selected = i == traj.get('selected_index')
            m.scale.x = 0.15 if selected else 0.05
            m.color.r, m.color.g, m.color.b, m.color.a = (0.9, 0.0, 0.9, 1.0) if selected else (1.0, 0.55, 0.55, 0.7)
            m.pose.orientation.w = 1.0
            m.points = [p.pose.position for p in self.poses(cand, stamp)]
            markers.markers.append(m)
        text = Marker()
        text.header.stamp, text.header.frame_id = stamp, self.frame_id
        text.ns, text.id, text.type, text.action = 'status', 0, Marker.TEXT_VIEW_FACING, Marker.ADD
        text.pose.position.z, text.pose.orientation.w, text.scale.z = 2.0, 1.0, 0.4
        text.color.r = text.color.g = text.color.b = text.color.a = 1.0
        text.text = f"{body['command']}: {body.get('status')} goal={body.get('pixel_goal_lookdown')}"
        markers.markers.append(text)
        self.marker_pub.publish(markers)


def main():
    rclpy.init()
    node = DualVLNInferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_stop(True)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
