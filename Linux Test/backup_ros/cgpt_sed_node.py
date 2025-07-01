#!/usr/bin/env python3
import os
import sys
import time
import threading
import datetime
import traceback
import struct
import csv

import numpy as np
import serial
from serial.tools import list_ports

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs_py.point_cloud2 as pc2


# === your original RadarParser class, unchanged ===
MAGIC_WORD = b'\x02\x01\x04\x03\x06\x05\x08\x07'
MAGIC_WORD_LEN = 8
HEADER_LEN = 40  # mmWave SDK v3 header size

class RadarParser:
    def __init__(self, cli_port, data_port, config_file,
                 cli_baud=115200, data_baud=921600,
                 debug=False, enable_logging=False, log_prefix="radar_log"):
        # Configuration
        self.cli_port = cli_port
        self.data_port = data_port
        self.config_file = config_file
        self.cli_baud = cli_baud
        self.data_baud = data_baud
        self.debug = debug

        # Serial handles and buffer
        self.cli_serial = None
        self.data_serial = None
        self.buffer = bytearray()
        self.frame_count = 0

        # Logging setup
        self.logging_enabled = enable_logging
        self.log_file = None
        self.csv_writer = None
        if self.logging_enabled:
            self._init_logger(log_prefix)

    def _timestamp(self):
        return datetime.datetime.now().isoformat()

    def _init_logger(self, prefix):
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{ts}.csv"
        try:
            self.log_file = open(filename, mode='w', newline='')
            self.csv_writer = csv.writer(self.log_file)
            self.csv_writer.writerow(['timestamp', 'level', 'message'])
            self.info_print(f"Logging started: {filename}")
        except Exception:
            # self.error_print(f"Failed to initialize logger with file {filename}")
            

    def info_print(self, msg):
        print(f"[{self._timestamp()}] [INFO] {msg}")
        if self.logging_enabled:
            self._log_row(['INFO', msg])

    def warn_print(self, msg):
        print(f"[{self._timestamp()}] [WARN] {msg}")
        if self.logging_enabled:
            self._log_row(['WARN', msg])

    def error_print(self, msg):
        print(f"[{self._timestamp()}] [ERROR] {msg}")
        print(traceback.format_exc())
        if self.logging_enabled:
            self._log_row(['ERROR', msg])

    def debug_print(self, msg):
        if self.debug:
            print(f"[{self._timestamp()}] [DEBUG] {msg}")
            if self.logging_enabled:
                self._log_row(['DEBUG', msg])

    def debug_print_raw(self, data):
        if self.debug:
            print(f"[{self._timestamp()}] [DEBUG_RAW] {data.hex()}")
            if self.logging_enabled:
                self._log_row(['DEBUG_RAW', data.hex()])

    def _log_row(self, row_fields):
        if self.csv_writer:
            ts = self._timestamp()
            try:
                self.csv_writer.writerow([ts] + row_fields)
            except Exception:
                print(f"[{ts}] [ERROR] Failed to write log row: {row_fields}")

    def initialize_ports(self):
        # CLI
        try:
            print(f"[{self._timestamp()}] [DEBUG] Attempting to open CLI port '{self.cli_port}' at {self.cli_baud} baud.")
            self.cli_serial = serial.Serial(self.cli_port, self.cli_baud, timeout=0.5,
                                            xonxoff=False, rtscts=False, dsrdtr=False)
            print(f"[{self._timestamp()}] [INFO] Opened CLI port: {self.cli_port}")
        except Exception as e:
            self.error_print(f"Failed to open CLI port: {self.cli_port} | Exception: {e}")
            sys.exit(1)

        # Data
        try:
            print(f"[{self._timestamp()}] [DEBUG] Attempting to open DATA port '{self.data_port}' at {self.data_baud} baud.")
            self.data_serial = serial.Serial(self.data_port, self.data_baud, timeout=0.5,
                                             xonxoff=False, rtscts=False, dsrdtr=False)
            print(f"[{self._timestamp()}] [INFO] Opened Data port: {self.data_port}")
        except Exception as e:
            self.error_print(f"Failed to open Data port: {self.data_port} | Exception: {e}")
            sys.exit(1)

        # Confirm
        try:
            settings = f"CLI port settings: {self.cli_serial.portstr} -> {self.cli_serial.baudrate}, " \
                       f"timeout={self.cli_serial.timeout}, xonxoff={self.cli_serial.xonxoff}, " \
                       \
                       f"Data port settings: {self.data_serial.ports f"rtscts={self.cli_serial.rtscts}\n"tr} -> {self.data_serial.baudrate}, " \
                       f"timeout={self.data_serial.timeout}, xonxoff={self.data_serial.xonxoff}, " \
                       f"rtscts={self.data_serial.rtscts}"
            self.debug_print(settings)
        except Exception:
            pass

    def send_config(self):
        self.info_print(f"Sending config from {self.config_file} to CLI port...")

        cfg_path = os.path.normpath(self.config_file)
        if not os.path.isfile(cfg_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            alt_path = os.path.join(script_dir, os.path.basename(cfg_path))
            if os.path.isfile(alt_path):
                cfg_path = alt_path
                self.debug_print(f"Config found in script directory: {cfg_path}")
            else:
                self.error_print(
                    f"Config file '{self.config_file}' not found in '{os.getcwd()}' or '{script_dir}'"
                )
                sys.exit(1)

        try:
            with open(cfg_path, 'r', encoding='ascii', errors='ignore') as cfg:
                for line in cfg:
                    try:
                        content = line.strip()
                        if not content or content.startswith('%'):
                            continue
                        self.debug_print(f"Sending line: {content}")
                        self.cli_serial.write((content + '\r\n').encode('ascii', errors='ignore'))
                        time.sleep(0.02)
                    except Exception:
                        self.error_print(f"Failed to send config line: {line.rstrip()}")
                self.info_print("Config sent successfully.")
        except Exception:
            self.error_print("Unexpected error during config send")
            sys.exit(1)

    def find_magic_word(self, buffer):
        try:
            idx = buffer.find(MAGIC_WORD)
            if self.debug:
                print(f"[{self._timestamp()}] [DEBUG] find_magic_word returned index {idx}")
            return idx
        except Exception:
            self.error_print("Error finding magic word in buffer")
            return -1

    def parse_header(self, packet):
        if len(packet) < HEADER_LEN:
            return None
        try:
            header = {
                'magic_word': packet[0:8],
                'version': struct.unpack_from('<I', packet, 8)[0],
                'total_packet_len': struct.unpack_from('<I', packet, 12)[0],
                'platform': struct.unpack_from('<I', packet, 16)[0],
                'frame_number': struct.unpack_from('<I', packet, 20)[0],
                'time_cpu_cycles': struct.unpack_from('<I', packet, 24)[0],
                'num_detected_obj': struct.unpack_from('<I', packet, 28)[0],
                'num_tlvs': struct.unpack_from('<I', packet, 32)[0],
                'subframe_number': struct.unpack_from('<I', packet, 36)[0]
            }
            return header
        except Exception:
            self.warn_print("Error parsing header")
            return None

    def parse_tlv_1(self, packet, offset, num_objs):
        obj_size = 16
        if offset + (num_objs * obj_size) > len(packet):
            self.warn_print("Insufficient data for detected points TLV")
            return None, offset

        x = np.zeros(num_objs, dtype=np.float32)
        y = np.zeros(num_objs, dtype=np.float32)
        z = np.zeros(num_objs, dtype=np.float32)
        velocity = np.zeros(num_objs, dtype=np.float32)

        for i in range(num_objs):
            x[i], y[i], z[i], velocity[i] = struct.unpack_from('<ffff', packet, offset)
            offset += obj_size

        self.info_print(f"Detected {num_objs} objects.")
        self.info_print(f"X: {x}")
        self.info_print(f"Y: {y}")
        self.info_print(f"Z: {z}")
        self.info_print(f"Velocity: {velocity}")

        return {"numObj": num_objs, "x": x, "y": y, "z": z, "velocity": velocity}, offset

    def parse_tlv_7(self, packet, offset, num_objs):
        obj_size = 4
        if offset + (num_objs * obj_size) > len(packet):
            self.warn_print("Insufficient data for side info TLV")
            return None, None, offset

        snr = np.zeros(num_objs, dtype=np.float32)
        noise = np.zeros(num_objs, dtype=np.float32)

        for i in range(num_objs):
            snr_raw, noise_raw = struct.unpack_from('<HH', packet, offset)
            snr[i] = snr_raw * 0.1
            noise[i] = noise_raw * 0.1
            offset += obj_size

        return snr, noise, offset

    def parse_packet(self, packet):
        header = self.parse_header(packet)
        if header is None:
            self.warn_print("Invalid packet header")
            return None, None, None

        self.debug_print(f"Parsing Frame {header['frame_number']}, Total Length {header['total_packet_len']}")

        offset = HEADER_LEN
        det_obj, snr, noise = None, None, None

        for i in range(header['num_tlvs']):
            if offset + 8 > len(packet):
                self.warn_print(f"Incomplete TLV header at TLV {i}")
                break

            tlv_type, tlv_length = struct.unpack_from('<II', packet, offset)
            self.info_print(f"TLV {i+1}: Type={tlv_type}, Length={tlv_length}")
            offset += 8

            if offset + tlv_length > len(packet):
                self.warn_print(f"Incomplete TLV payload for type {tlv_type}")
                break

            if tlv_type == 1:
                det_obj, new_offset = self.parse_tlv_1(packet, offset, header['num_detected_obj'])
                offset = new_offset if det_obj else offset + tlv_length
            elif tlv_type == 7:
                snr, noise, new_offset = self.parse_tlv_7(packet, offset, header['num_detected_obj'])
                offset = new_offset if snr is not None else offset + tlv_length
            else:
                self.info_print(f"Skipping unknown TLV type {tlv_type}")
                offset += tlv_length

        return header, det_obj, (snr, noise)

    def read_frame(self):
        try:
            if not self.data_serial or not self.data_serial.is_open:
                raise serial.SerialException("Serial port is not open or was disconnected.")

            try:
                data = self.data_serial.read(2048)
            except (serial.SerialException, PermissionError, OSError) as e:
                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                port_info = getattr(self.data_serial, 'port', 'Unknown')
                self.error_print(f"[{current_time}] [RadarParser: {port_info}] Radar read failed: {e}")
                if self.data_serial:
                    try:
                        self.data_serial.close()
                    except Exception:
                        pass
                self.data_serial = None
                raise serial.SerialException("Radar disconnected.") from e

            if data:
                print(f"[{self._timestamp()}] [DEBUG_RAW] Read {len(data)} bytes: {data[:32].hex()}" +
                      ((" ..." + data[-4:].hex()) if len(data) > 36 else ""))
                self.buffer.extend(data)
            else:
                if self.debug:
                    print(f"[{self._timestamp()}] [DEBUG] No data read (data == b''), sleeping 1ms")
                time.sleep(0.001)

        except Exception as e:
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            port_info = getattr(self.data_serial, 'port', 'Unknown') if self.data_serial else 'Unknown'
            self.error_print(f"[{current_time}] [RadarParser: {port_info}] Unexpected error: {e}")
            raise

        if self.buffer:
            while True:
                idx = self.find_magic_word(self.buffer)
                if idx == -1:
                    if len(self.buffer) > 8192:
                        print(f"[{self._timestamp()}] [DEBUG] Buffer >8192 bytes without magic word, trimming")
                        self.buffer = self.buffer[-1024:]
                    break

                if idx > 0:
                    if self.debug:
                        print(f"[{self._timestamp()}] [DEBUG] Discarding {idx} bytes before magic word")
                    self.buffer = self.buffer[idx:]
                    idx = 0

                if len(self.buffer) < HEADER_LEN:
                    if self.debug:
                        print(f"[{self._timestamp()}] [DEBUG] Buffer length {len(self.buffer)} < HEADER_LEN {HEADER_LEN}, need more data")
                    break

                try:
                    total_len = struct.unpack_from("<I", self.buffer, 12)[0]
                except Exception:
                    self.warn_print("Error reading packet length, discarding buffer")
                    self.buffer = bytearray()
                    break

                if total_len < HEADER_LEN or total_len > 65536:
                    self.warn_print(f"Invalid packet length {total_len}, skipping magic word and continuing")
                    self.buffer = self.buffer[MAGIC_WORD_LEN:]
                    continue

                if len(self.buffer) < total_len:
                    if self.debug:
                        print(f"[{self._timestamp()}] [DEBUG] Have {len(self.buffer)} bytes, need {total_len}, waiting for more")
                    break

                packet = self.buffer[:total_len]
                header, det_obj, (snr, noise) = self.parse_packet(packet)
                self.buffer = self.buffer[total_len:]
                self.frame_count += 1
                return header, det_obj, snr, noise

        return None, None, None, None

    def close(self):
        try:
            if self.cli_serial and self.cli_serial.is_open:
                self.cli_serial.close()
            if self.data_serial and self.data_serial.is_open:
                self.data_serial.close()
            self.info_print("Serial ports closed.")
        except Exception:
            self.error_print("Error closing serial ports")


# === ROS2 node wrapping it all ===
class RadarNode(Node):
    def __init__(self):
        super().__init__('radar_node')

        # parameters
        self.declare_parameter('radar_keyword', 'ti-ic-');
        self.declare_parameter('config_file', "/home/airl-radar/ros2_radar_ws/src/radar_system_hub/radar_system_hub/best_res_4cm.cfg");
        self.radar_keyword = self.get_parameter('radar_keyword').get_parameter_value().string_value
        self.config_file = self.get_parameter('config_file').get_parameter_value().string_value

        # publishers
        self.hb_pub = self.create_publisher(String, 'radar_heartbeat', 10)
        self.ports_pub = self.create_publisher(String, 'radar_ports', 10)
        self.pc_pub = self.create_publisher(PointCloud2, 'radar_pointcloud', 10)

        # find ports
        ports = list_ports.comports()
        matches = []
        for p in ports:
            desc = (p.description or "").lower()
            hwid = (p.hwid or "").lower()
            manuf = (p.manufacturer or "").lower()
            if self.radar_keyword in desc or self.radar_keyword in hwid or self.radar_keyword in manuf:
                matches.append(p.device)
        if len(matches) < 2 and sys.platform.startswith('linux'):
            usb_ports = [p.device for p in ports if ('ttyusb' in p.device.lower() or 'ttyacm' in p.device.lower())]
            for port in usb_ports:
                if port not in matches:
                    matches.append(port)
                if len(matches) >= 2:
                    break

        if len(matches) < 2:
            self.get_logger().error('Could not find two radar ports!')
            sys.exit(1)

        cli_port, data_port = matches[0], matches[1]
        cli_baud = 115200
        data_baud = 921600

        # publish ports once
        port_msg = String()
        port_msg.data = f"CLI={cli_port}@{cli_baud}, DATA={data_port}@{data_baud}"
        self.ports_pub.publish(port_msg)
        self.get_logger().info('Published radar_ports: ' + port_msg.data)

        # initialize parser
        self.parser = RadarParser(cli_port, data_port, self.config_file,
                                  cli_baud=cli_baud, data_baud=data_baud, debug=False)
        self.parser.initialize_ports()
        self.parser.send_config()

        # heartbeat timer
        self.create_timer(1.0, self.publish_heartbeat)

        # start parsing thread
        threading.Thread(target=self.read_loop, daemon=True).start()

    def publish_heartbeat(self):
        msg = String()
        msg.data = f"Radar connection alive @ {datetime.datetime.now().isoformat()}"
        self.hb_pub.publish(msg)

    def read_loop(self):
        while rclpy.ok():
            try:
                header, det_obj, snr, noise = self.parser.read_frame()
                if det_obj is not None:
                    # build PointCloud2
                    points = list(zip(det_obj['x'], det_obj['y'], det_obj['z']))
                    header_ros = self.get_clock().now().to_msg()
                    pc2_msg = pc2.create_cloud_xyz32(
                        header=rclpy.msg.Header(stamp=header_ros, frame_id='radar_frame'),
                        fields=[
                            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                        ],
                        points=points
                    )
                    self.pc_pub.publish(pc2_msg)
            except Exception as e:
                self.get_logger().error('Error in read_loop: ' + str(e))
                break

    def destroy_node(self):
        self.parser.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = RadarNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

