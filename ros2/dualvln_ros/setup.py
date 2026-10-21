from glob import glob

from setuptools import setup

package_name = 'dualvln_ros'

setup(
    name=package_name,
    version='0.3.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools', 'requests', 'numpy', 'pillow'],
    zip_safe=True,
    description='ROS 2 client node for the DualVLN inference server',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'inference_node = dualvln_ros.inference_node:main',
            'replay_publisher = dualvln_ros.replay_publisher:main',
        ],
    },
)
