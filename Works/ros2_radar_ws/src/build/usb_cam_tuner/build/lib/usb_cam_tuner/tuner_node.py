import subprocess
import re
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class CamTunerNode(Node):
    def __init__(self, device='/dev/video0'):
        super().__init__('cam_tuner')
        self.device = device
        self.bridge = CvBridge()
        self.latest_image = None

        # Subscription to image topic
        self.create_subscription(Image, '/image_raw', self.image_callback, qos_profile_sensor_data)

    def image_callback(self, msg):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().warn(f"Image conversion failed: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = CamTunerNode()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
            if node.latest_image is not None:
                frame = cv2.resize(node.latest_image, (1280, 720))
                cv2.imshow("ROS2 Camera Feed", frame)
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
