from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='v4l2_camera',
            executable='v4l2_camera_node',
            name='v4l2_camera',
            parameters=[{
                'video_device': '/dev/video0',
                'pixel_format': 'YUYV',
                'image_size': [1280, 720],
                'camera_frame_id': 'v4l2_frame',
                # 'output_encoding': 'yuv422_yuy2',
                'output_encoding': 'yuv422_yuy2',
                'fps': 30.0,
                # 'time_per_frame' : [1,30],
                # 'io_method': "mmap"  # Enables MMAP
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


# from launch import LaunchDescription
# from launch_ros.actions import Node

# def generate_launch_description():
#     return LaunchDescription([
#         Node(
#             package='usb_cam',
#             executable='usb_cam_node_exe',
#             name='usb_cam',
#             parameters=[{
#                 'video_device': '/dev/video0',
#                 'pixel_format': 'raw_mjpeg',             # Must be lowercase for usb_cam
#                 'image_width': 1280,
#                 'image_height': 720,
#                 'framerate': 60.0,
#                 'camera_frame_id': 'usb_cam_frame',
#                 'io_method': 'mmap',                # Optional, can use 'userptr' or 'read'
#             }],
#             output='screen'
#         ),
#         Node(
#             package='usb_cam_tuner',
#             executable='tuner_node',
#             name='tuner',
#             output='screen'
#         )
#     ])
