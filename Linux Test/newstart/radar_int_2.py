import serial
import struct
import time
import sys
import traceback
import numpy as np
import os
from serial.serialutil import SerialException
from system_logger import get_logger

MAGIC_WORD = b'\x02\x01\x04\x03\x06\x05\x08\x07'
MAGIC_WORD_LEN = 8
HEADER_LEN = 40
BUFFER_READ_SIZE = 8192  # Increased buffer size


class RadarParser:
    def __init__(self, cli_port, data_port, config_file,
                 cli_baud=115200, data_baud=921600):  # Default to higher baud
        # Configuration
        self.cli_port = cli_port
        self.data_port = data_port
        self.config_file = config_file
        self.cli_baud = cli_baud
        self.data_baud = data_baud

        # Serial handles and buffer
        self.cli_serial = None
        self.data_serial = None
        self.buffer = bytearray()
        self.frame_count = 0

        # Performance metrics
        self.total_bytes = 0
        self.start_time = time.time()
        self.last_frame_time = self.start_time
        self.frequency = 0.0
        self.pending_bytes = 0

        # Configure logging
        self.logger = get_logger('RadarParser')
        self.logger.info(f"Initializing RadarParser with CLI: {cli_port}, DATA: {data_port}")

    def initialize_ports(self):
        """Open both CLI and Data serial ports with optimized settings"""
        # CLI port
        try:
            self.logger.debug(f"Opening CLI port '{self.cli_port}' at {self.cli_baud} baud")
            self.cli_serial = serial.Serial(
                self.cli_port, self.cli_baud, timeout=0.5,
                xonxoff=False, rtscts=False, dsrdtr=False
            )
            self.logger.info(f"Opened CLI port: {self.cli_port}")
        except Exception as e:
            self.logger.error(f"Failed to open CLI port: {self.cli_port} | Exception: {e}")
            sys.exit(1)

        # Data port with reduced timeout and higher baud
        try:
            self.logger.debug(f"Opening DATA port '{self.data_port}' at {self.data_baud} baud")
            self.data_serial = serial.Serial(
                self.data_port, self.data_baud, timeout=0.01,  # Reduced timeout
                xonxoff=False, rtscts=False, dsrdtr=False
            )
            self.data_serial.reset_input_buffer()  # Clear any stale data
            self.logger.info(f"Opened Data port: {self.data_port}")
        except Exception as e:
            self.logger.error(f"Failed to open Data port: {self.data_port} | Exception: {e}")
            sys.exit(1)

    def send_config(self):
        """Send configuration with optimized flushing"""
        self.logger.info(f"Sending config from {self.config_file} to CLI port...")
        cfg_path = os.path.normpath(self.config_file)

        # Config file discovery
        if not os.path.isfile(cfg_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            alt_path = os.path.join(script_dir, os.path.basename(cfg_path))
            if os.path.isfile(alt_path):
                cfg_path = alt_path
                self.logger.debug(f"Using config from script directory: {alt_path}")
            else:
                self.logger.error(f"Config file not found at {self.config_file} or {alt_path}")
                sys.exit(1)

        # Batch command sending
        try:
            commands = []
            with open(cfg_path, 'r', encoding='ascii', errors='ignore') as cfg:
                for line in cfg:
                    content = line.strip()
                    if content and not content.startswith('%'):
                        commands.append(content + '\r\n')

            # Send all commands in a single batch
            self.cli_serial.write(b''.join([cmd.encode('ascii') for cmd in commands]))
            self.cli_serial.flush()
            self.logger.info(f"Sent {len(commands)} configuration commands")

        except Exception as e:
            self.logger.error(f"Config send failed: {e}")
            sys.exit(1)

    def find_magic_word(self, buffer: bytearray) -> int:
        """Efficient magic word search"""
        return buffer.find(MAGIC_WORD)

    def parse_header(self, packet: memoryview) -> dict:
        """Header parsing optimized with direct struct unpacking"""
        if len(packet) < HEADER_LEN:
            return None

        # Single struct unpack for performance
        header_data = struct.unpack('<8sIIIIIIII', packet[:HEADER_LEN])
        return {
            'magic_word': header_data[0],
            'version': header_data[1],
            'total_packet_len': header_data[2],
            'platform': header_data[3],
            'frame_number': header_data[4],
            'time_cpu_cycles': header_data[5],
            'num_detected_obj': header_data[6],
            'num_tlvs': header_data[7],
            'subframe_number': header_data[8]
        }

    def parse_tlv_1(self, packet: memoryview, offset: int, num_objs: int):
        """Optimized TLV parsing with direct memory mapping"""
        if num_objs == 0:
            return {'numObj': 0, 'x': np.array([]), 'y': np.array([]),
                    'z': np.array([]), 'velocity': np.array([])}, offset

        # Direct memory mapping without intermediate copy
        obj_size = 16
        required_bytes = num_objs * obj_size
        if offset + required_bytes > len(packet):
            self.logger.warning("TLV1 data out of bounds")
            return None, offset

        arr = np.frombuffer(packet[offset:offset + required_bytes], dtype=np.float32)
        arr = arr.reshape(-1, 4)
        det = {
            'numObj': num_objs,
            'x': arr[:, 0],
            'y': arr[:, 1],
            'z': arr[:, 2],
            'velocity': arr[:, 3]
        }
        return det, offset + required_bytes

    def parse_tlv_7(self, packet: memoryview, offset: int, num_objs: int):
        """Optimized TLV parsing with pre-allocation"""
        if num_objs == 0:
            return np.array([]), np.array([]), offset

        required_bytes = num_objs * 4
        if offset + required_bytes > len(packet):
            self.logger.warning("TLV7 data out of bounds")
            return None, None, offset

        # Efficient conversion using numpy
        arr = np.frombuffer(packet[offset:offset + required_bytes], dtype=np.uint16)
        arr = arr.reshape(-1, 2)
        snr = arr[:, 0].astype(np.float32) * 0.1
        noise = arr[:, 1].astype(np.float32) * 0.1
        return snr, noise, offset + required_bytes

    def parse_packet(self, packet: bytearray):
        """Packet parsing with reduced branching and logging"""
        mv = memoryview(packet)
        header = self.parse_header(mv)
        if header is None:
            return None, None, None

        offset = HEADER_LEN
        det_obj = None
        snr = None
        noise = None

        # Process TLVs without intermediate logging
        for _ in range(header['num_tlvs']):
            if offset + 8 > len(mv):
                break

            tlv_type, tlv_length = struct.unpack_from('<II', mv, offset)
            offset += 8

            if tlv_type == 1:
                det_obj, offset = self.parse_tlv_1(mv, offset, header['num_detected_obj'])
            elif tlv_type == 7:
                snr, noise, offset = self.parse_tlv_7(mv, offset, header['num_detected_obj'])
            else:
                offset += tlv_length

        return header, det_obj, (snr, noise)

    def read_frame(self):
        """Optimized frame reading with bulk data processing"""
        try:
            # Bulk read with larger buffer size
            data = self.data_serial.read(BUFFER_READ_SIZE)
            if not data:
                return None, None, None, None

            self.total_bytes += len(data)
            self.buffer.extend(data)
            self.pending_bytes = len(self.buffer)

        except Exception as e:
            self.logger.error(f"Radar read failed: {e}")
            if self.data_serial:
                self.data_serial.close()
            self.data_serial = None
            return None, None, None, None

        # Process all complete packets in buffer
        while True:
            idx = self.find_magic_word(self.buffer)
            if idx < 0 or len(self.buffer) < HEADER_LEN:
                break

            # Remove any leading garbage before magic word
            if idx > 0:
                del self.buffer[:idx]

            # Validate packet length
            if len(self.buffer) < HEADER_LEN:
                break

            total_len = struct.unpack_from('<I', self.buffer, 12)[0]
            if total_len < HEADER_LEN or total_len > 65536:
                del self.buffer[:MAGIC_WORD_LEN]
                continue

            # Check for complete packet
            if len(self.buffer) < total_len:
                break

            # Process packet
            packet = self.buffer[:total_len]
            del self.buffer[:total_len]
            self.frame_count += 1

            # Update frame rate metrics
            now = time.time()
            self.frequency = 1.0 / (now - self.last_frame_time) if self.last_frame_time != now else 0.0
            self.last_frame_time = now

            # Return parsed frame data
            header, det_obj, aux_data = self.parse_packet(packet)
            if header:
                return header, det_obj, aux_data, None

        return None, None, None, None

    def close(self):
        """Close serial ports efficiently"""
        try:
            for port in [self.cli_serial, self.data_serial]:
                if port and port.is_open:
                    port.close()
            self.logger.info("Serial ports closed")
        except Exception as e:
            self.logger.error(f"Error closing ports: {e}")