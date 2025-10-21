import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/airl-radar/ros2_radar_ws/src/install/mmwave_radar_driver'
