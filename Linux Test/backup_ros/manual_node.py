#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

#for serial connections
import serial
from serial.tools import list_ports

import sys
import os

#math tools
import numpy as np
import struct

import datetime
from radar_msgs.msg import RadarData

#=====================DO NOT TOUCH=========================
#Internal Parameters for usage
CLI_BAUD = 115200
DATA_BAUD = 921600 

#Parsing internal parameters
MAGIC_WORD = b'\x02\x01\x04\x03\x06\x05\x08\x07'
MAGIC_WORD_LEN = 8
HEADER_LEN = 40  # mmWave SDK v3 header size
#===========================================================

class RadarNode(Node):
    def __init__(self) -> None:
        super().__init__('radar_node')

        #Parameters
        self.declare_parameter('config_file', "/home/airl-radar/ros2_radar_ws/src/radar_system_hub/radar_system_hub/best_res_4cm.cfg");
        self.declare_parameter('cli_baud', CLI_BAUD)
        self.declare_parameter('data_baud', DATA_BAUD)
        

        self.rate_20ms = self.create_rate(50, self.get_clock())  # 50 Hz = 0.02 seconds
        self.rate_1ms = self.create_rate(1000, self.get_clock())  # 1000 Hz = 0.001 seconds


        self.cli_port = None
        self.data_port = None
        self.config_file = self.get_parameter('config_file').get_parameter_value().string_value;
        self.cli_baud = self.get_parameter('cli_baud').get_parameter_value().integer_value
        self.data_baud = self.get_parameter('data_baud').get_parameter_value().integer_value

        # Serial handles and buffer
        self.cli_serial = None
        self.data_serial = None
        self.buffer = bytearray()
        self.frame_count = 0



       
        #Publisher Initializations
        self.hb_pub = self.create_publisher(String, 'radar_heartbeat', 10)
    
        #finding the port with the radar
        ports = list_ports.comports()
        matches = []
        
        #find port loop
        #find only if linux
        if sys.platform.startswith('linux'):
            usb_ports = [p.device for p in ports if ('ttyusb' in p.device.lower() or 'ttyacm' in p.device.lower())]
            for port in usb_ports:
                if port not in matches:
                    matches.append(port)
                if len(matches) >= 2:
                    break
        else:
            self.get_logger().warn('No radar ports found! | Try again later |')


        if len(matches) < 2:
            self.get_logger().error('Could not find two radar ports!')
            rclpy.shutdown()
        
        # declare the ports for further use
        self.cli_port, self.data_port = matches[0], matches[1]
        #initialise port message
        port_msg = String()
        port_msg.data = f"CLI={self.cli_port}@{self.cli_baud}, DATA={self.data_port}@{self.data_baud}"
        self.ports_pub.publish(port_msg)
        self.get_logger().info('Published radar_ports: ' + port_msg.data)

        self.initialize_ports()
        self.send_config()
        self.get_logger().info('[INIT] Radar Connected .. .. .. | :)')

        # heartbeat timer after connection is verified
        self.create_timer(1.0, self.heartbeat_callback)
        self.create_timer(1.0, self.init_connet)
    
    ##Callbacks
    def heartbeat_callback(self):
        msg = String()
        msg.data = f"Radar connection alive @ {datetime.datetime.now().isoformat()}"
        self.hb_pub.publish(msg) 

    #-----------------------------------------


    def initialize_ports(self):
        # CLI
        try:
            self.get_logger.debug(f"[RADAR_INIT] Attempting to open CLI port '{self.cli_port}' at {self.cli_baud} baud.")
            self.cli_serial = serial.Serial(self.cli_port, self.cli_baud, timeout=0.5,
                                            xonxoff=False, rtscts=False, dsrdtr=False)
            self.get_logger.info(f"[RADAR_INIT] Opened CLI port: {self.cli_port}")
        except Exception as e:
            self.get_logger.error(f"[RADAR_INIT] Failed to open CLI port: {self.cli_port} | Exception: {e}")
            rclpy.shutdown()

        # Data
        try:
            self.get_logger.debug(f"[RADAR_INIT] Attempting to open DATA port '{self.data_port}' at {self.data_baud} baud.")
            self.data_serial = serial.Serial(self.data_port, self.data_baud, timeout=0.5,
                                             xonxoff=False, rtscts=False, dsrdtr=False)
            self.get_logger.info(f"[RADAR_INIT] Opened Data port: {self.data_port}")
        except Exception as e:
            self.get_logger().error(f"Failed to open Data port: {self.data_port} | Exception: {e}")
            sys.exit(1)

        # Confirm
        try:
            settings = f"CLI port settings: {self.cli_serial.portstr} -> {self.cli_serial.baudrate}, " \
                       f"timeout={self.cli_serial.timeout}, xonxoff={self.cli_serial.xonxoff}, " \
                       \
                       f"Data port settings: {self.data_serial.ports f"rtscts={self.cli_serial.rtscts}\n"tr} -> {self.data_serial.baudrate}, " \
                       f"timeout={self.data_serial.timeout}, xonxoff={self.data_serial.xonxoff}, " \
                       f"rtscts={self.data_serial.rtscts}"
            self.get_logger().debug(settings)
        except Exception:
            pass
    


    def send_config(self):
        self.get_logger().info(f"[CONFIG_FILE] Sending config from {self.config_file} to CLI port...")

        cfg_path = os.path.normpath(self.config_file)
        if not os.path.isfile(cfg_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            alt_path = os.path.join(script_dir, os.path.basename(cfg_path))
            if os.path.isfile(alt_path):
                cfg_path = alt_path
                self.get_logger().debug(f"[CONFIG_FILE] Config found in script directory: {cfg_path}")
            else:
                self.get_logger().error(
                    f"[CONFIG_FILE] Config file '{self.config_file}' not found in '{os.getcwd()}' or '{script_dir}'"
                )
                rclpy.shutdown()
                return


        with open(cfg_path, 'r', encoding='ascii', errors='ignore') as cfg:
            for line in cfg:
                try:
                    content = line.strip()
                    if not content or content.startswith('%'):
                        continue
                    self.get_logger().debug(f"[CONFIG_FILE] Sending line: {content}")
                    self.cli_serial.write((content + '\r\n').encode('ascii', errors='ignore'))
                    self.rate_20ms.sleep()  # replaces time.sleep(0.02)
                except Exception:
                    self.get_logger().error(f"[CONFIG_FILE] Failed to send config line: {line.rstrip()}")
            
            self.get_logger().info("[CONFIG_FILE] Config sent successfully.")

#-------------------------------------------------------------------

    def find_magic_word(self, buffer):
        try:
            idx = buffer.find(MAGIC_WORD)
            if self.debug:
                self.get_logger.debug(f"[MAGIC_WORD] find_magic_word returned index {idx}")
            return idx
        except Exception:
            self.get_logger().error("[MAGIC_WORD] Error finding magic word in buffer")
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
            self.get_logger().warn("[PARSER] Error parsing header | Returning Value: None")
            return None

#---------------------------------------------------------------------------------
    def parse_tlv_1(self, packet, offset, num_objs):
        obj_size = 16
        if offset + (num_objs * obj_size) > len(packet):
            self.get_logger().warn("[PARSE_TLV1] Insufficient data for detected points TLV")
            return None, offset

        x = np.zeros(num_objs, dtype=np.float32)
        y = np.zeros(num_objs, dtype=np.float32)
        z = np.zeros(num_objs, dtype=np.float32)
        velocity = np.zeros(num_objs, dtype=np.float32)

        for i in range(num_objs):
            x[i], y[i], z[i], velocity[i] = struct.unpack_from('<ffff', packet, offset)
            offset += obj_size

        self.get_logger().info(
            f"[PARSE_TLV1] Detected {num_objs} objects\n"
            f"X: {x}\n"
            f"Y: {y}\n"
            f"Z: {z}\n"
            f"Velocity: {velocity}"
        )

        return {"numObj": num_objs, "x": x, "y": y, "z": z, "velocity": velocity}, offset

#-----------------------------------------------------------------------------------------------------
    def parse_tlv_7(self, packet, offset, num_objs):
        obj_size = 4
        if offset + (num_objs * obj_size) > len(packet):
            self.get_logger().warn("[PARSE_TLV7] Insufficient data for side info TLV | Returning None")
            return None, None, offset

        snr = np.zeros(num_objs, dtype=np.float32)
        noise = np.zeros(num_objs, dtype=np.float32)

        for i in range(num_objs):
            snr_raw, noise_raw = struct.unpack_from('<HH', packet, offset)
            snr[i] = snr_raw * 0.1
            noise[i] = noise_raw * 0.1
            offset += obj_size
        
        self.get_logger().info(
            f"[PARSE_TLV7] Sensor Params Found: \n"
            f"SNR: {snr}\n"
            f"Noise: {noise}\n"
            f"Offset: {offset}"
        )

        return snr, noise, offset

#-------------------------------------------------------------------------------------------------------------

    def parse_packet(self, packet):
        header = self.parse_header(packet)
        if header is None:
            self.get_logger().warn("[PARSE_PACKET] Invalid packet header | Returning None")
            return None, None, None

        self.get_logger().debug(f"[PARSE_PACKET] Parsing Frame {header['frame_number']}, Total Length {header['total_packet_len']}")

        offset = HEADER_LEN
        det_obj, snr, noise = None, None, None

        for i in range(header['num_tlvs']):
            if offset + 8 > len(packet):
                self.get_logger().warn(f"[PARSE_PACKET] Incomplete TLV header at TLV {i}")
                break

            tlv_type, tlv_length = struct.unpack_from('<II', packet, offset)
            self.get_logger().info(f"[PARSE_PACKET] TLV {i+1}: Type={tlv_type}, Length={tlv_length}")
            offset += 8

            if offset + tlv_length > len(packet):
                self.get_logger().warn(f"[Parse_PACKET] Incomplete TLV payload for type {tlv_type}")
                break

            if tlv_type == 1:
                det_obj, new_offset = self.parse_tlv_1(packet, offset, header['num_detected_obj'])
                offset = new_offset if det_obj else offset + tlv_length
            elif tlv_type == 7:
                snr, noise, new_offset = self.parse_tlv_7(packet, offset, header['num_detected_obj'])
                offset = new_offset if snr is not None else offset + tlv_length
            else:
                self.get_logger().info(f"[PARSE_PACKET] Skipping unknown TLV type {tlv_type}")
                offset += tlv_length

        return header, det_obj, (snr, noise)
#-----------------------------------------------------------------------------------------------------------------

    def read_frame(self):
        try:
            if not self.data_serial or not self.data_serial.is_open:
                raise serial.SerialException("[READ_FRAME] Serial port is not open or was disconnected.")

            try:
                data = self.data_serial.read(2048)
            except (serial.SerialException, PermissionError, OSError) as e:
                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                port_info = getattr(self.data_serial, 'port', 'Unknown')
                self.get_logger().error(f"[READ_FRAME] [{current_time}] [RadarParser: {port_info}] Radar read failed: {e}")
                if self.data_serial:
                    try:
                        self.data_serial.close()
                    except Exception:
                        pass
                self.data_serial = None
                raise serial.SerialException("[READ_FRAME] Radar disconnected.") from e

            # In your data reading loop:
            if data:
                self.get_logger().debug(f" [READ_RAW] Read {len(data)} bytes: {data[:32].hex()}" +
                    ((" ..." + data[-4:].hex()) if len(data) > 36 else ""))
                self.buffer.extend(data)
            else:
                if self.debug:
                    self.get_logger().debug(f"[READ_FRAME] No data read (data == b''), sleeping 1ms")
                self.rate_1ms.sleep()  # replaces time.sleep(0.001)

        except Exception as e:
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            port_info = getattr(self.data_serial, 'port', 'Unknown') if self.data_serial else 'Unknown'
            self.get_logger().error(f"[READ_FRAME] [{current_time}] [RadarParser: {port_info}] Unexpected error: {e}")
            raise

        if self.buffer:
            while True:
                idx = self.find_magic_word(self.buffer)
                if idx == -1:
                    if len(self.buffer) > 8192:
                        self.get_logger().warn(f"[READ_FRAME] Buffer >8192 bytes without magic word, trimming")
                        self.buffer = self.buffer[-1024:]
                    break

                if idx > 0:
                    if self.debug:
                        self.get_logger().warn(f"[READ_FRAME] Discarding {idx} bytes before magic word")
                    self.buffer = self.buffer[idx:]
                    idx = 0

                if len(self.buffer) < HEADER_LEN:
                    if self.debug:
                        self.get_logger().warn(f"[READ_FRAME] Buffer length {len(self.buffer)} < HEADER_LEN {HEADER_LEN}, need more data")
                    break

                try:
                    total_len = struct.unpack_from("<I", self.buffer, 12)[0]
                except Exception:
                    self.get_logger().warn("[READ_FRAME] Error reading packet length, discarding buffer")
                    self.buffer = bytearray()
                    break

                if total_len < HEADER_LEN or total_len > 65536:
                    self.get_logger().warn(f"[READ_FRAME] Invalid packet length {total_len}, skipping magic word and continuing")
                    self.buffer = self.buffer[MAGIC_WORD_LEN:]
                    continue

                if len(self.buffer) < total_len:
                    if self.debug:
                        self.get_logger().warn(f"[READ_FRAME] Have {len(self.buffer)} bytes, need {total_len}, waiting for more")
                    break

                packet = self.buffer[:total_len]
                header, det_obj, (snr, noise) = self.parse_packet(packet)
                self.buffer = self.buffer[total_len:]
                self.frame_count += 1
                return header, det_obj, snr, noise

        return None, None, None, None
#------------------------------------------------------------

    def init_connet(self):
        while rclpy.ok():
            try:
                header, det_obj, snr, noise = self.read_frame()
                    
                if det_obj is None or not getattr(self, 'data_serial', None) or not self.data_serial.is_open:

                    self.get_logger().info("[RADAR_CONNECTION] Failed to retrieve data ! Trying again")
                    continue

            except Exception as e:
                self.get_logger().error('Error in read_loop: ' + str(e))
                break
    

    def close(self):
        try:
            if self.cli_serial and self.cli_serial.is_open:
                self.cli_serial.close()
            if self.data_serial and self.data_serial.is_open:
                self.data_serial.close()
            self.get_logger().info("[READ_FRAME] Serial ports closed.")
        except Exception:
            self.get_logger().fatal("[READ_FRAME] Error closing serial ports")
    
    
    


def main(args=Node):
    rclpy.init(agrs=args)
    node = RadarNode()


    rclpy.shutdown()