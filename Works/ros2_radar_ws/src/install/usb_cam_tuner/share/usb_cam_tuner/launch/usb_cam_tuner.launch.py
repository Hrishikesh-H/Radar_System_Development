from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='usb_cam',
            executable='usb_cam_node_exe',
            name='usb_cam',
            parameters=[{
                'video_device': '/dev/video0',
                'pixel_format': 'uyvy',
                'image_width': 1280,
                'image_height': 720,
                'framerate': 30.0,
                'io_method': "mmap",
                'camera_frame_id': "usb_cam"
            }],
            output='screen'
        ),
        Node(
            package='usb_cam_tuner',
            executable='tuner_node',
            name='tuner',
            output='screen'
        )
    ])
