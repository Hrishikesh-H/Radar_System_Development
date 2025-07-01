import serial
import struct
import time
import sys
import traceback
import numpy as np
import os
from serial.serialutil import SerialException
from system_logger import get_logger  # Import centralized logger

MAGIC_WORD = b'\x02\x01\x04\x03\x06\x05\x08\x07'
MAGIC_WORD_LEN = 8
HEADER_LEN = 40  # mmWave SDK v3 header size
MAX_READ_SIZE = 2048 # Max bytes to read in one operation


class RadarParser:
    def __init__(self, cli_port, data_port, config_file,
                 cli_baud=115200, data_baud=921600):
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

        # Data port (non-blocking)
        try:
            self.logger.debug(f"Opening DATA port '{self.data_port}' at {self.data_baud} baud")
            self.data_serial = serial.Serial(
                self.data_port, self.data_baud, timeout=0,  # Non-blocking mode
                xonxoff=False, rtscts=False, dsrdtr=False
            )
            self.logger.info(f"Opened Data port: {self.data_port}")
        except Exception as e:
            self.logger.error(f"Failed to open Data port: {self.data_port} | Exception: {e}")
            sys.exit(1)

    def send_config(self):
        """Optimized configuration sending with reduced sleeps"""
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
        commands = []
        try:
            with open(cfg_path, 'r', encoding='ascii', errors='ignore') as cfg:
                for line in cfg:
                    content = line.strip()
                    if content and not content.startswith('%'):
                        commands.append(content)
        except Exception as e:
            self.logger.error(f"Config read failed: {e}")
            sys.exit(1)

        # Send all commands in batch
        try:
            for cmd in commands:
                self.cli_serial.write((cmd + '\r\n').encode('ascii'))
            self.cli_serial.flush()
            self.logger.info(f"Sent {len(commands)} configuration commands")
        except Exception as e:
            self.logger.error(f"Config send failed: {e}")
            sys.exit(1)

    def find_magic_word(self, buffer: bytearray) -> int:
        """Efficient magic word location with minimal logging"""
        return buffer.find(MAGIC_WORD)

    def parse_header(self, packet: memoryview) -> dict:
        """Header parsing remains unchanged but critical for structure"""
        if len(packet) < HEADER_LEN:
            return None

        return {
            'magic_word': packet[0:8].tobytes(),
            'version': struct.unpack_from('<I', packet, 8)[0],
            'total_packet_len': struct.unpack_from('<I', packet, 12)[0],
            'platform': struct.unpack_from('<I', packet, 16)[0],
            'frame_number': struct.unpack_from('<I', packet, 20)[0],
            'time_cpu_cycles': struct.unpack_from('<I', packet, 24)[0],
            'num_detected_obj': struct.unpack_from('<I', packet, 28)[0],
            'num_tlvs': struct.unpack_from('<I', packet, 32)[0],
            'subframe_number': struct.unpack_from('<I', packet, 36)[0]
        }

    def parse_tlv_1(self, packet: memoryview, offset: int, num_objs: int):
        """Optimized object parsing with numpy views"""
        obj_size = 16
        x = np.frombuffer(packet[offset:offset + num_objs * obj_size], dtype='<f4')
        x = x.reshape(-1, 4)
        det = {
            'numObj': num_objs,
            'x': x[:, 0], 'y': x[:, 1],
            'z': x[:, 2], 'velocity': x[:, 3]
        }
        return det, offset + num_objs * obj_size

    def parse_tlv_7(self, packet: memoryview, offset: int, num_objs: int):
        """Efficient SNR/noise parsing"""
        arr = np.frombuffer(packet[offset:offset + num_objs * 4], dtype='<u2')
        arr = arr.reshape(-1, 2)
        snr = arr[:, 0].astype(np.float32) * 0.1
        noise = arr[:, 1].astype(np.float32) * 0.1
        return snr, noise, offset + num_objs * 4

    def parse_packet(self, packet: bytearray):
        """Packet parsing remains structurally identical"""
        mv = memoryview(packet)
        header = self.parse_header(mv)
        if header is None:
            return None, None, None

        offset = HEADER_LEN
        det_obj = None
        snr = None
        noise = None

        for _ in range(header['num_tlvs']):
            tlv_type, tlv_length = struct.unpack_from('<II', mv, offset)
            offset += 8
            if tlv_type == 1:
                det_obj, offset = self.parse_tlv_1(mv, offset, header['num_detected_obj'])
            elif tlv_type == 7:
                snr, noise, offset = self.parse_tlv_7(mv, offset, header['num_detected_obj'])
            else:
                offset += tlv_length

        return header, det_obj, (snr, noise)

    def _extract_frame(self):
        """Efficient frame extraction from buffer"""
        while len(self.buffer) >= MAGIC_WORD_LEN:
            idx = self.find_magic_word(self.buffer)
            if idx < 0:
                break

            # Discard malformed data before magic word
            if idx > 0:
                del self.buffer[:idx]
                continue

            # Validate header presence
            if len(self.buffer) < HEADER_LEN:
                break

            # Get packet length from header
            total_len = struct.unpack_from('<I', self.buffer, 12)[0]
            if total_len < HEADER_LEN or total_len > 65536:
                del self.buffer[:MAGIC_WORD_LEN]
                continue

            # Check for complete packet
            if len(self.buffer) < total_len:
                break

            # Extract complete packet
            packet = bytes(self.buffer[:total_len])
            del self.buffer[:total_len]

            # Update frame metrics
            self.frame_count += 1
            now = time.time()
            interval = now - self.last_frame_time
            self.frequency = 1.0 / interval if interval > 0 else 0.0
            self.last_frame_time = now

            self.logger.info(f"Frame {self.frame_count} processed at {self.frequency:.1f} Hz")
            return self.parse_packet(packet) + (None,)

        return None

    def _read_data(self):
        """Bulk data reading with error handling"""
        try:
            if not self.data_serial or not self.data_serial.is_open:
                raise SerialException("Data port not open")

            # Read all available data
            to_read = self.data_serial.in_waiting
            if not to_read:
                return 0

            data = self.data_serial.read(min(to_read, MAX_READ_SIZE))
            if not data:
                return 0

            self.total_bytes += len(data)
            self.buffer.extend(data)
            self.pending_bytes = len(self.buffer)
            return len(data)

        except Exception as e:
            self.logger.error(f"Data read error: {e}")
            if self.data_serial:
                self.data_serial.close()
            self.data_serial = None
            raise

    def read_frame(self):
        """Optimized frame reading pipeline"""
        # Try to extract existing frame from buffer
        frame = self._extract_frame()
        if frame:
            return frame

        # Read new data if no frame available
        try:
            if self._read_data() > 0:
                # Try extraction again after new data
                return self._extract_frame()
        except Exception:
            pass

        return None, None, None, None

    def close(self):
        """Close resources with error handling"""
        try:
            if self.cli_serial and self.cli_serial.is_open:
                self.cli_serial.close()
            if self.data_serial and self.data_serial.is_open:
                self.data_serial.close()
            self.logger.info("Ports closed")
        except Exception as e:
            self.logger.error(f"Close error: {e}")