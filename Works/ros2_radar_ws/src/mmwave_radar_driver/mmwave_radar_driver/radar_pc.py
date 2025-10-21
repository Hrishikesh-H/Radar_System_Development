import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from mmwave_radar_interfaces.msg import DetectedObjects
import struct
import numpy as np
from std_msgs.msg import Header
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster  # Changed from TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud  # Correct import
from builtin_interfaces.msg import Time 

class RadarPointCloudNode(Node):
    """
    Converts detected radar objects to PointCloud2 format with TF2 transformations
    """
    def __init__(self):
        super().__init__('radar_pointcloud_converter')
        # Declare parameters
        self.declare_parameter('source_frame', 'radar_link')
        self.declare_parameter('target_frame', 'base_link')  # More common default
        self.declare_parameter('publish_tf', True)
        self.declare_parameter('static_tf', True)
        
        # Get parameters
        self.source_frame = self.get_parameter('source_frame').value
        self.target_frame = self.get_parameter('target_frame').value
        self.publish_tf = self.get_parameter('publish_tf').value
        self.static_tf = self.get_parameter('static_tf').value
        
        # TF2 setup
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Static TF broadcaster setup
        if self.publish_tf and self.static_tf:
            self.static_broadcaster = StaticTransformBroadcaster(self)
            self.publish_static_tf()
        
        # Subscribe to detected objects
        self.subscription = self.create_subscription(
            DetectedObjects,
            'mmWave_radar/det_obj',
            self.detection_callback,
            10
        )
        
        # Publisher for PointCloud2
        self.publisher = self.create_publisher(
            PointCloud2,
            'mmWave_radar/point_cloud',
            10
        )
        
        self.get_logger().info('RadarPointCloudNode initialized')
        self.get_logger().info(f'Transforming from {self.source_frame} to {self.target_frame}')

    def publish_static_tf(self):
        t = TransformStamped()
        t.header.stamp = Time(sec=0, nanosec=0)  # Use zero time
        t.header.frame_id = self.target_frame
        t.child_frame_id = self.source_frame
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0

        self.static_broadcaster.sendTransform(t)
        self.get_logger().info(f'Publishing static transform from {t.header.frame_id} to {t.child_frame_id}')

    def detection_callback(self, msg):
        """
        Convert DetectedObjects to PointCloud2 and transform to target frame
        """
        # Create point cloud in source frame
        pc2_source = self._create_pointcloud(msg)
        
        try:
            # Wait for transform to be available
            if not self.tf_buffer.can_transform(
                self.target_frame,
                self.source_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1)
            ):
                self.get_logger().warn('Transform not available, using source frame')
                pc2_source.header.stamp = self.get_clock().now().to_msg()
                self.publisher.publish(pc2_source)
                return
                
            # Lookup transform to target frame
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,
                self.source_frame,
                rclpy.time.Time()
            )
            
            # Transform point cloud using correct function
            pc2_transformed = do_transform_cloud(pc2_source, transform)
            pc2_transformed.header.stamp = self.get_clock().now().to_msg()
            self.publisher.publish(pc2_transformed)
            
        except TransformException as ex:
            self.get_logger().error(f'TF transformation failed: {ex}')
            # Fallback: publish in source frame
            pc2_source.header.stamp = self.get_clock().now().to_msg()
            self.publisher.publish(pc2_source)

    def _create_pointcloud(self, msg):
        """Create PointCloud2 message in source frame"""
        pc2_msg = PointCloud2()
        pc2_msg.header = Header()
        pc2_msg.header.frame_id = self.source_frame
        
        # Define point fields (x, y, z, velocity)
        pc2_msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='velocity', offset=12, datatype=PointField.FLOAT32, count=1)
        ]
        
        # Point structure parameters
        pc2_msg.height = 1  # Unorganized cloud
        pc2_msg.width = msg.num_obj
        pc2_msg.point_step = 16  # 4 fields * 4 bytes each
        pc2_msg.row_step = pc2_msg.point_step * pc2_msg.width
        pc2_msg.is_dense = True  # No invalid points
        
        # Pack data into binary blob
        points = bytearray()
        for i in range(msg.num_obj):
            points.extend(struct.pack(
                '4f', 
                msg.x[i], 
                msg.y[i], 
                msg.z[i], 
                msg.velocity[i]
            ))
        
        pc2_msg.data = bytes(points)
        return pc2_msg

def main(args=None):
    rclpy.init(args=args)
    node = RadarPointCloudNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
