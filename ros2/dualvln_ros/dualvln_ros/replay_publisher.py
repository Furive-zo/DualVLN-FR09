"""Bench test: publish recorded rectified JPEGs as sensor_msgs/CompressedImage at 2 Hz.

  ros2 run dualvln_ros replay_publisher --ros-args -p dir:=/path/to/camera/rgb_front \
      -p topic:=/camera/front/image_rect/compressed
"""
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage


class ReplayPublisher(Node):
    def __init__(self):
        super().__init__('dualvln_replay')
        folder = self.declare_parameter('dir', '').value
        topic = self.declare_parameter('topic', '/camera/front/image_rect/compressed').value
        rate = self.declare_parameter('rate_hz', 2.0).value
        self.loop = self.declare_parameter('loop', False).value
        self.files = sorted(Path(folder).glob('*.jpg'))
        if not self.files:
            raise SystemExit(f'no *.jpg in {folder!r}')
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        self.pub = self.create_publisher(CompressedImage, topic, qos)
        self.index = 0
        self.create_timer(1.0 / rate, self.tick)
        self.get_logger().info(f'{len(self.files)} frames from {folder} -> {topic}')

    def tick(self):
        if self.index >= len(self.files):
            if not self.loop:
                self.get_logger().info('end of replay', once=True)
                return
            self.index = 0
        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_front'
        msg.format = 'jpeg'
        msg.data = self.files[self.index].read_bytes()
        self.pub.publish(msg)
        self.index += 1


def main():
    rclpy.init()
    node = ReplayPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
