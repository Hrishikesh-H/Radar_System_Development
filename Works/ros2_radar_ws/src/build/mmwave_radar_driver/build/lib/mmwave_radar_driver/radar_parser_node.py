"""
ROS2 Node for mmWave Radar Parsing
Publishes header, detected objects, SNR/noise, raw packets (optional), and heartbeat.
Handles serial reconnections, data hangs, and error resilience without exiting.
"""
import os
import struct
import serial
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.clock import Clock
from rclpy.rate import Rate
from serial.serialutil import SerialException
from mmwave_radar_driver.msg import (
    Header,
    DetectedObjects,
    SNRNoise,
    RawPacket,
    Heartbeat,
)

# Constants
MAGIC_WORD = b"\x02\x01\x04\x03\x06\x05\x08\x07"  # Fixed magic word bytes
HEADER_LEN = 40  # mmWave SDK v3 header size

class RadarParserNode(Node):
    """
    Wraps serial communication to parse radar frames and publish ROS2 messages.
    """
    def __init__(self):
        super().__init__('radar_parser')
        self._declare_parameters()
        self._initialize_state()
        self._setup_publishers()
        self._connect_serials()
        self._send_config()
        self._setup_timers()
        self.get_logger().info('RadarParserNode initialized successfully')

    def _declare_parameters(self):
        # Comm parameters
        self.declare_parameter('cli_port', '/dev/ttyUSB0')
        self.declare_parameter('data_port', '/dev/ttyUSB1')
        self.declare_parameter('config_file', '/config/best_res_4cm.cfg')
        self.declare_parameter('cli_baud', 115200)
        self.declare_parameter('data_baud', 921600)
        # Publish toggles
        self.declare_parameter('publish_header', False)
        self.declare_parameter('publish_tlv1', True)
        self.declare_parameter('publish_tlv7', True)
        self.declare_parameter('publish_raw', False)
        self.declare_parameter('publish_heartbeat', True)
        # Timing
        self.declare_parameter('heartbeat_freq', 1.0)
        self.declare_parameter('reconnect_interval', 5.0)
        self.declare_parameter('no_data_timeout', 2.0)

        p = self.get_parameters([
            'cli_port','data_port','config_file','cli_baud','data_baud',
            'publish_header','publish_tlv1','publish_tlv7','publish_raw','publish_heartbeat',
            'heartbeat_freq','reconnect_interval','no_data_timeout'
        ])
        vals = [param.value for param in p]
        (self.cli_port, self.data_port, self.config_file,
         self.cli_baud, self.data_baud,
         self.publish_header, self.publish_tlv1, self.publish_tlv7,
         self.publish_raw, self.publish_heartbeat,
         self.hb_freq, self.reconnect_interval,
         self.no_data_timeout) = vals

    def _initialize_state(self):
        self.cli_serial = None
        self.data_serial = None
        self.buffer = bytearray()
        self.frame_count = 0
        self.clock = Clock()
        self.last_data_time = self.clock.now()

    def _setup_publishers(self):
        self.header_pub = self.create_publisher(Header, 'mmWave_radar/header', 10)
        self.det_pub = self.create_publisher(DetectedObjects, 'mmWave_radar/det_obj', 10)
        self.snr_pub = self.create_publisher(SNRNoise, 'mmWave_radar/snr_noise', 10)
        if self.publish_heartbeat:
            self.hb_pub = self.create_publisher(Heartbeat, 'heartbeat', 10)
        if self.publish_raw:
            self.raw_pub = self.create_publisher(RawPacket, 'mmWave_radar/raw_packet', 10)

    def _setup_timers(self):
        self.create_timer(0.001, self._read_serial)
        if self.publish_heartbeat:
            self.create_timer(1.0 / self.hb_freq, self._publish_heartbeat)
        self.create_timer(self.reconnect_interval, self._check_and_reconnect)

    # Serial Connect & Config
    def _connect_serials(self):
        self._open_cli_serial()
        self._open_data_serial()

    def _open_cli_serial(self):
        if self.cli_serial and self.cli_serial.is_open:
            return
        try:
            self.cli_serial = serial.Serial(self.cli_port, self.cli_baud, timeout=0.1)
            self.get_logger().info(f"CLI port opened: {self.cli_port}")
        except Exception as e:
            self.get_logger().error(f"CLI open failed: {e}")
            self.cli_serial = None

    def _open_data_serial(self):
        if self.data_serial and self.data_serial.is_open:
            return
        try:
            self.data_serial = serial.Serial(self.data_port, self.data_baud, timeout=0.1)
            self.get_logger().info(f"Data port opened: {self.data_port}")
        except Exception as e:
            self.get_logger().error(f"Data open failed: {e}")
            self.data_serial = None

    def _send_config(self):
        if not (self.cli_serial and self.cli_serial.is_open):
            self.get_logger().warning('CLI not open; skipping config')
            return
        path = os.path.normpath(self.config_file)
        if not os.path.isfile(path):
            alt = os.path.join(os.path.dirname(__file__), os.path.basename(path))
            path = alt if os.path.isfile(alt) else None
        if not path:
            self.get_logger().error(f"Config not found: {self.config_file}")
            return
        rate = Rate(self, 100)  # Fixed Rate initialization
        with open(path, 'r') as cfg:
            for line in cfg:
                cmd = line.strip()
                if not cmd or cmd.startswith('%'):
                    continue
                try:
                    self.cli_serial.write((cmd + '\n').encode('ascii'))
                    self.cli_serial.flush()
                    self.get_logger().debug(f"Sent config cmd: {cmd}")
                except Exception as e:
                    self.get_logger().error(f"Config cmd error: {e}")
                rate.sleep()
        self.get_logger().info('Config transmission complete')

    # Data Read & Processing
    def _read_serial(self):
        if not (self.data_serial and self.data_serial.is_open):
            return
        try:
            data = self.data_serial.read(4096)
            now = self.clock.now()
            if data:
                self.last_data_time = now
                if self.publish_raw:
                    self.raw_pub.publish(RawPacket(data=list(data)))
                self.buffer.extend(data)
                self._process_buffer()
            elif (now - self.last_data_time).nanoseconds * 1e-9 > self.no_data_timeout:
                self.get_logger().warning('No data timeout; clearing buffer')
                self.buffer.clear()
        except SerialException as e:
            self.get_logger().error(f"Serial exception: {e}")
            if self.data_serial: 
                self.data_serial.close()
            self.data_serial = None
        except Exception as e:
            self.get_logger().fatal(f"Read error: {e}")

    def _process_buffer(self):
        """Extract packets from buffer and publish."""
        while True:
            idx = self.buffer.find(MAGIC_WORD)
            if idx < 0:
                break
                
            # Remove any data before magic word
            if idx > 0:
                del self.buffer[:idx]
                continue
                
            # Check if we have enough data for header
            if len(self.buffer) < HEADER_LEN:
                break
                
            # Extract packet length
            length = struct.unpack_from('<I', self.buffer, 12)[0]
            
            # Validate packet length
            if length < HEADER_LEN or length > 100000:  # Increased max length
                del self.buffer[:len(MAGIC_WORD)]
                continue
                
            # Check if full packet is available
            if len(self.buffer) < length:
                break
                
            # Process complete packet
            pkt = bytes(self.buffer[:length])
            del self.buffer[:length]
            self._publish_packet(pkt)

    # Publish Parsed Data
    def _publish_packet(self, packet: bytes):
        self.frame_count += 1
        hdr = self._parse_header(packet)
        det, (snr, noise) = self._parse_tlvs(packet, hdr)
        if self.publish_header: 
            self.header_pub.publish(hdr)
        if self.publish_tlv1 and det: 
            self.det_pub.publish(det)
        if self.publish_tlv7 and snr is not None: 
            self.snr_pub.publish(SNRNoise(snr=snr, noise=noise))

    # Parsing Helpers
    def _parse_header(self, packet: bytes) -> Header:
        h = Header()
        # Magic word (8 bytes)
        h.magic_word = list(packet[:8])
        
        # Unpack remaining header (8 integers = 32 bytes)
        vals = struct.unpack_from('<8I', packet, 8)
        (h.version, h.total_packet_len, h.platform, h.frame_number,
         h.time_cpu_cycles, h.num_detected_obj, h.num_tlvs,
         h.subframe_number) = vals
        return h

    def _parse_tlvs(self, packet: bytes, header: Header):
        offset = HEADER_LEN
        det = None
        snr_arr = None
        noise_arr = None
        
        for _ in range(header.num_tlvs):
            t, l = struct.unpack_from('<II', packet, offset)
            offset += 8
            
            if t == 1:  # Detected objects
                det, offset = self._parse_tlv1(packet, offset, header.num_detected_obj)
            elif t == 7:  # SNR/Noise
                (snr_arr, noise_arr), offset = self._parse_tlv7(packet, offset, header.num_detected_obj)
            else:
                offset += l  # Skip unknown TLV
                
        return det, (snr_arr, noise_arr)

    def _parse_tlv1(self, packet, offset: int, num: int):
        # Each object: 4 floats (x, y, z, velocity) = 16 bytes
        det = DetectedObjects()
        det.num_obj = num
        
        # Pre-allocate lists
        det.x = [0.0] * num
        det.y = [0.0] * num
        det.z = [0.0] * num
        det.velocity = [0.0] * num
        
        # Unpack directly into lists
        for i in range(num):
            det.x[i], det.y[i], det.z[i], det.velocity[i] = \
                struct.unpack_from('<4f', packet, offset + i*16)
                
        return det, offset + num*16

    def _parse_tlv7(self, packet, offset: int, num: int):
        # Each entry: 2 uint16 (snr, noise) = 4 bytes
        snr = [0.0] * num
        noise = [0.0] * num
        
        for i in range(num):
            snr_val, noise_val = struct.unpack_from('<HH', packet, offset + i*4)
            snr[i] = snr_val * 0.1  # Convert to dB
            noise[i] = noise_val * 0.1  # Convert to dB
            
        return (snr, noise), offset + num*4

    # Heartbeat & Reconnect
    def _publish_heartbeat(self):
        msg = Heartbeat()
        msg.builtin_time = self.get_clock().now().to_msg()
        msg.frame_count = self.frame_count
        self.hb_pub.publish(msg)

    def _check_and_reconnect(self):
        """Reconnect ports on error and resend config if CLI revived."""
        cfg_needed = False
        if not (self.cli_serial and self.cli_serial.is_open):
            self.get_logger().warning('CLI disconnected, attempting reconnect')
            self._open_cli_serial()
            cfg_needed = True
        if not (self.data_serial and self.data_serial.is_open):
            self.get_logger().warning('Data port disconnected, attempting reconnect')
            self._open_data_serial()
        if cfg_needed and self.cli_serial: 
            self._send_config()

    def destroy_node(self):
        if self.cli_serial and self.cli_serial.is_open: 
            self.cli_serial.close()
        if self.data_serial and self.data_serial.is_open: 
            self.data_serial.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = RadarParserNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()