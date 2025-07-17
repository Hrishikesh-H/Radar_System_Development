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
        """
        Open both CLI and Data serial ports and report any errors.
        """
        # 1) CLI port
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

        # 2) Data port
        try:
            self.logger.debug(f"Opening DATA port '{self.data_port}' at {self.data_baud} baud")
            self.data_serial = serial.Serial(
                self.data_port, self.data_baud, timeout=0.5,
                xonxoff=False, rtscts=False, dsrdtr=False
            )
            self.logger.info(f"Opened Data port: {self.data_port}")
        except Exception as e:
            self.logger.error(f"Failed to open Data port: {self.data_port} | Exception: {e}")
            sys.exit(1)

        # Log actual port settings
        try:
            settings = f"CLI: {self.cli_serial.portstr} @ {self.cli_serial.baudrate}b, timeout={self.cli_serial.timeout}s\n" \
                       f"DATA: {self.data_serial.portstr} @ {self.data_serial.baudrate}b, timeout={self.data_serial.timeout}s"
            self.logger.debug(settings)
        except Exception:
            pass

    def send_config(self):
        """
        Send the radar configuration commands over the CLI serial port.
        """
        self.logger.info(f"Sending config from {self.config_file} to CLI port...")

        # Normalize and validate config file path
        cfg_path = os.path.normpath(self.config_file)
        if not os.path.isfile(cfg_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            alt_path = os.path.join(script_dir, os.path.basename(cfg_path))
            if os.path.isfile(alt_path):
                cfg_path = alt_path
                self.logger.debug(f"Using config from script directory: {alt_path}")
            else:
                self.logger.error(f"Config file not found at {self.config_file} or {alt_path}")
                sys.exit(1)

        # Read and send each non-comment line
        try:
            with open(cfg_path, 'r', encoding='ascii', errors='ignore') as cfg:
                for line in cfg:
                    content = line.strip()
                    if not content or content.startswith('%'):
                        continue
                    self.logger.debug(f"Sending command: {content}")
                    self.cli_serial.write((content + '\r\n').encode('ascii', errors='ignore'))
                    self.cli_serial.flush()
                    time.sleep(0.02)
                self.logger.info("Configuration sent successfully")
        except Exception as e:
            self.logger.error(f"Config send failed: {e}")
            sys.exit(1)

    def find_magic_word(self, buffer: bytearray) -> int:
        """
        Search for the MAGIC_WORD in the buffer and return its index.
        """
        idx = buffer.find(MAGIC_WORD)
        self.logger.debug(f"Magic word search index: {idx}")
        return idx

    def parse_header(self, packet: memoryview) -> dict:
        """
        Extract and return the header dictionary from a raw packet.
        """
        if len(packet) < HEADER_LEN:
            self.logger.warning("Packet too short for header")
            return None

        header = {
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
        return header

    def parse_tlv_1(self, packet: memoryview, offset: int, num_objs: int):
        """
        Parse TLV type 1 (detected objects). Returns (det_obj_dict, new_offset).
        """
        obj_size = 16
        x = np.frombuffer(packet[offset:offset + num_objs*obj_size], dtype='<f4')
        x = x.reshape(-1, 4)
        det = { 'numObj': num_objs,
                'x': x[:,0], 'y': x[:,1],
                'z': x[:,2], 'velocity': x[:,3] }
        new_offset = offset + num_objs * obj_size
        self.logger.info(f"Detected {num_objs} objects")
        # self.logger.debug(f"[TLV_1] X: {det['x']}, Y: {det['y']}, Z: {det['z']}, velocity: {det['velocity']}")
        return det, new_offset

    def parse_tlv_7(self, packet: memoryview, offset: int, num_objs: int):
        """
        Parse TLV type 7 (side info: SNR & noise). Returns (snr_array, noise_array, new_offset).
        """
        # raw halfword data: 2 bytes each -> 4 bytes total for both
        arr = np.frombuffer(packet[offset:offset + num_objs*4], dtype='<u2')
        arr = arr.reshape(-1, 2)
        snr = arr[:,0].astype(np.float32) * 0.1
        noise = arr[:,1].astype(np.float32) * 0.1
        new_offset = offset + num_objs*4
        # self.logger.debug(f"[TLV_7] SNR: {snr}, noise: {noise}")
        return snr, noise, new_offset

    def parse_packet(self, packet: bytearray):
        """
        Given a raw packet, extract header, then parse each TLV (1 and 7).
        Returns (header_dict, det_obj_dict, (snr_array, noise_array)).
        """
        mv = memoryview(packet)
        header = self.parse_header(mv)
        if header is None:
            self.logger.warning("Skipping packet due to invalid header")
            return None, None, None

        offset = HEADER_LEN
        det_obj = None
        snr = None
        noise = None

        # Get actual packet length for bounds checking
        packet_len = len(packet)

        for i in range(header['num_tlvs']):
            # Check if we have enough bytes for TLV header
            if offset + 8 > packet_len:
                self.logger.warning("Insufficient bytes for TLV header")
                break

            tlv_type, tlv_length = struct.unpack_from('<II', mv, offset)
            self.logger.info(f"TLV {i + 1}: Type={tlv_type}, Length={tlv_length}")
            offset += 8

            # Check if TLV data fits in packet
            if offset + tlv_length > packet_len:
                self.logger.warning("TLV data exceeds packet length")
                break

            if tlv_type == 1:
                det_obj, new_offset = self.parse_tlv_1(mv, offset, header['num_detected_obj'])
                offset = new_offset
            elif tlv_type == 7:
                snr, noise, new_offset = self.parse_tlv_7(mv, offset, header['num_detected_obj'])
                offset = new_offset
            else:
                self.logger.info(f"Skipping TLV type {tlv_type}")
                offset += tlv_length  # Skip unknown TLV types

        return header, det_obj, (snr, noise)

    def read_frame(self):
        """
        Attempt to read one complete radar frame.
        Returns (header, det_obj, snr, noise) or (None, None, None, None) if incomplete.
        """
        try:
            if not self.data_serial or not self.data_serial.is_open:
                raise SerialException("Data port is not open")

            data = self.data_serial.read(2048)
            self.total_bytes += len(data)
            if data:
                self.buffer.extend(data)
                self.pending_bytes = len(self.buffer)
                self.logger.debug(f"Received {len(data)} bytes, total received: {self.total_bytes}, pending: {self.pending_bytes}")
            else:
                time.sleep(0.001)
                return None, None, None, None

        except Exception as e:
            self.logger.error(f"Radar read failed: {e}")
            if self.data_serial:
                self.data_serial.close()
            self.data_serial = None
            raise

        # Process buffer for complete packet
        while True:
            idx = self.find_magic_word(self.buffer)
            if idx < 0 or len(self.buffer) < HEADER_LEN:
                break

            # Discard before magic
            if idx > 0:
                del self.buffer[:idx]

            total_len = struct.unpack_from('<I', self.buffer, 12)[0]
            if total_len < HEADER_LEN or total_len > 65536:
                self.logger.warning(f"Invalid packet length {total_len}, discarding magic word")
                del self.buffer[:MAGIC_WORD_LEN]
                continue

            if len(self.buffer) < total_len:
                break

            # Got full packet
            packet = self.buffer[:total_len]
            del self.buffer[:total_len]
            self.frame_count += 1

            # Update frequency
            now = time.time()
            interval = now - self.last_frame_time
            self.frequency = 1.0 / interval if interval > 0 else 0.0
            self.last_frame_time = now

            self.logger.info(f"Frame {self.frame_count} processed at {self.frequency:.2f} Hz, buffer pending {len(self.buffer)} bytes")
            return self.parse_packet(packet) + (None,)

        return None, None, None, None

    def close(self):
        """Close serial ports and log shutdown"""
        try:
            if self.cli_serial and self.cli_serial.is_open:
                self.cli_serial.close()
            if self.data_serial and self.data_serial.is_open:
                self.data_serial.close()
            self.logger.info("Serial ports closed")
        except Exception as e:
            self.logger.error(f"Error closing ports: {e}")
