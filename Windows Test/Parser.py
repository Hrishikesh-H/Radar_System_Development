import serial
import struct
import time
import datetime
import sys
import traceback
import numpy as np
import csv
import os

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
        """
        Internal: open CSV file for logging using prefix and timestamp.
        """
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{ts}.csv"
        try:
            self.log_file = open(filename, mode='w', newline='')
            self.csv_writer = csv.writer(self.log_file)
            self.csv_writer.writerow(['timestamp', 'level', 'message'])
            self.info_print(f"Logging started: {filename}")
        except Exception:
            self.error_print(f"Failed to initialize logger with file {filename}")

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
        """
        Print raw sensor data in debug mode.
        Call inside read_frame() when data is received.
        """
        if self.debug:
            print(f"[{self._timestamp()}] [DEBUG_RAW] {data}")
            if self.logging_enabled:
                self._log_row(['DEBUG_RAW', data.hex()])

    def _log_row(self, row_fields):
        """Internal: write a row to the CSV log."""
        if self.csv_writer:
            ts = self._timestamp()
            try:
                self.csv_writer.writerow([ts] + row_fields)
            except Exception:
                print(f"[{ts}] [ERROR] Failed to write log row: {row_fields}")

    def initialize_ports(self):
        try:
            self.cli_serial = serial.Serial(self.cli_port, self.cli_baud, timeout=0.5)
            self.data_serial = serial.Serial(self.data_port, self.data_baud, timeout=0.5)
            self.info_print(f"Opened CLI port: {self.cli_port} and Data port: {self.data_port}")
        except Exception:
            self.error_print(f"Failed to open serial ports: {self.cli_port}, {self.data_port}")
            sys.exit(1)

    def send_config(self):
        """
        Send the radar configuration commands over the CLI serial port.
        Works on both Windows and Linux by normalizing paths and falling back
        to the script directory if needed.
        """

        self.info_print(f"Sending config from {self.config_file} to CLI port...")

        # Normalize the provided path
        cfg_path = os.path.normpath(self.config_file)

        # If the file isn't in cwd, try alongside this script
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

        # Read and send each non-comment line, with per-line error handling
        try:
            with open(cfg_path, 'r') as cfg:
                for line in cfg:
                    try:
                        content = line.strip()
                        if not content or content.startswith('%'):
                            continue
                        self.debug_print(f"Sending line: {content}")
                        self.cli_serial.write((content + '\n').encode())
                        time.sleep(0.01)
                    except Exception:
                        self.error_print(f"Failed to send config line: {line.rstrip()}")
                self.info_print("Config sent successfully.")
        except Exception:
            self.error_print("Unexpected error during config send")
            sys.exit(1)


    def find_magic_word(self, buffer):
        try:
            idx = buffer.find(MAGIC_WORD)
            self.debug_print(f"Magic word index: {idx}")
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
            data = self.data_serial.read(2048)
            self.debug_print_raw(data)
        except Exception:
            self.error_print("Error reading from data port")
            return None, None, None, None

        if data:
            self.buffer.extend(data)
            while True:
                idx = self.find_magic_word(self.buffer)
                if idx == -1:
                    if len(self.buffer) > 8192:
                        self.buffer = self.buffer[-1024:]
                    break

                if idx > 0:
                    self.buffer = self.buffer[idx:]
                    idx = 0

                if len(self.buffer) < HEADER_LEN:
                    break

                try:
                    total_len = struct.unpack_from("<I", self.buffer, 12)[0]
                except Exception:
                    self.warn_print("Error reading packet length, discarding buffer")
                    self.buffer = bytearray()
                    break

                if total_len < HEADER_LEN or total_len > 65536:
                    self.warn_print(f"Invalid packet length {total_len}, skipping")
                    self.buffer = self.buffer[MAGIC_WORD_LEN:]
                    continue

                if len(self.buffer) < total_len:
                    break

                packet = self.buffer[:total_len]
                header, det_obj, (snr, noise) = self.parse_packet(packet)
                self.buffer = self.buffer[total_len:]
                self.frame_count += 1
                return header, det_obj, snr, noise
        else:
            time.sleep(0.001)
        return None, None, None, None

    def close(self):
        try:
            if self.cli_serial:
                self.cli_serial.close()
            if self.data_serial:
                self.data_serial.close()
            self.info_print("Serial ports closed.")
        except Exception:
            self.error_print("Error closing serial ports")


