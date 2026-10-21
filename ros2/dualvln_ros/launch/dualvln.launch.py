import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    params = os.path.join(get_package_share_directory('dualvln_ros'), 'config', 'dualvln.yaml')
    return LaunchDescription([
        DeclareLaunchArgument('server_url', default_value='http://127.0.0.1:5801'),
        DeclareLaunchArgument('image_topic', default_value='/camera/front/image_rect/compressed'),
        Node(
            package='dualvln_ros', executable='inference_node', name='dualvln_inference',
            output='screen',
            parameters=[params, {'server_url': LaunchConfiguration('server_url'),
                                 'image_topic': LaunchConfiguration('image_topic')}],
        ),
    ])
