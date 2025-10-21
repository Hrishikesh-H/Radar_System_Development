from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='v4l2_camera',
            executable='v4l2_camera_node',
            name='v4l2_camera',
            parameters=[{
                'video_device': '/dev/video0',  # Adjust if using v4l2loopback: e.g. /dev/video10
                'image_size': [1280, 720],
                'pixel_format': 'UYVY',
                'frame_rate': 60.0,
                'camera_frame_id': 'v4l2_cam'
            }],
            output='screen'
        ),

        Node(
            package='usb_cam_tuner',
            executable='tuner_node',
            name='tuner_node',
            output='screen'
        )
    ])
