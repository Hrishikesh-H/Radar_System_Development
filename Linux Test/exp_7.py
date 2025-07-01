# optimized_uav_system.py

import os
import sys
import time
import datetime
import logging
import traceback
import threading
import queue
from statistics import median
from collections import deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np
from pymavlink import mavutil
import serial

# External modules
from Parser import RadarParser
from Filter import RadarDespiker
from PortFinder import DevicePortFinder
from PlaneLand import LandingZoneAssessor
from IMUCompensator import AttitudeCompensator

# ------------------ Logging Setup ------------------
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%dT%H:%M:%S'
)
logger = logging.getLogger('UAV')

# ------------------ Global Locks ------------------
mav_lock = threading.Lock()  # Protects all MAVLink operations
mode_lock = threading.Lock()  # Protects mode updates

# ------------------ IMU Data Buffer ------------------
class IMUDataBuffer:
    """Thread-safe buffer for latest IMU data"""
    def __init__(self):
        self.data = None
        self.lock = threading.Lock()
        self.last_update = 0
        
    def update(self, data):
        with self.lock:
            self.data = data
            self.last_update = time.time()
            
    def get(self):
        with self.lock:
            return self.data, self.last_update

# ------------------ Distance Sensor Sender ------------------
class DistanceSensorSender:
    """Send MAVLink DISTANCE_SENSOR messages with thread safety."""
    def __init__(self, mav, min_range=0.05, max_range=5.0, rate_hz=20):
        self.mav = mav
        self.min_cm = max(0, min(int(min_range * 100), 65535))
        self.max_cm = max(0, min(int(max_range * 100), 65535))
        self.interval = 1.0 / rate_hz
        self._last_time = 0
        self._sequence = 0
        self._last_sent_distance = None

    def send(self, distance_m):
        now = time.monotonic()
        if now - self._last_time < self.interval and distance_m == self._last_sent_distance:
            return
            
        if distance_m is None:
            dist_cm = 0
        else:
            try:
                raw_cm = int(round(float(distance_m) * 100))
                dist_cm = max(self.min_cm, min(raw_cm, self.max_cm))
            except (TypeError, ValueError):
                dist_cm = 0

        with mav_lock:  # Protect MAVLink access
            if hasattr(self.mav, 'mav') and hasattr(self.mav.mav, 'distance_sensor_send'):
                try:
                    self.mav.mav.distance_sensor_send(
                        0, self.min_cm, self.max_cm, dist_cm,
                        mavutil.mavlink.MAV_DISTANCE_SENSOR_RADAR,
                        self._sequence % 256,
                        mavutil.mavlink.MAV_SENSOR_ROTATION_PITCH_270, 0
                    )
                    self._sequence += 1
                    self._last_time = now
                    self._last_sent_distance = distance_m
                    logger.debug(f"Sent DISTANCE_SENSOR: {dist_cm}cm")
                except Exception as e:
                    logger.error(f"Distance send failed: {str(e)}")

# ------------------ Mode Watcher ------------------
def mode_watcher(mav, stop_event, mode_holder):
    """Continuously read HEARTBEAT and update mode with locking."""
    while not stop_event.is_set():
        try:
            with mav_lock:  # Protect MAVLink access
                msg = mav.recv_match(type='HEARTBEAT', blocking=False)
                
            if msg:
                try:
                    new_mode = msg.custom_mode.decode('utf-8')
                except Exception:
                    base = msg.base_mode
                    new_mode = 'ARMED' if (base & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED) else 'DISARMED'
                
                with mode_lock:  # Protect mode update
                    mode_holder['mode'] = new_mode
            else:
                time.sleep(0.1)
        except Exception:
            logger.exception("Error in mode_watcher")
            time.sleep(1)

# ------------------ Landing Monitor ------------------
class LandingMonitor:
    """Evaluate landing safety with thread-safe operations."""
    def __init__(self, mav, **cfg):
        self.mav = mav
        self.buf_slope = deque(maxlen=cfg.get('buffer_size', 20))
        self.buf_inlier = deque(maxlen=cfg.get('buffer_size', 20))
        self.th_slope = cfg.get('slope_threshold_deg', 5.0)
        self.th_inlier = cfg.get('inlier_threshold', 0.6)
        self.warn_time = cfg.get('warning_duration', 3.0)
        self.min_warn = cfg.get('min_consecutive_to_warn', 5)
        self.min_clear = cfg.get('min_consecutive_to_clear', 5)
        self._reset()

    def _reset(self):
        self.buf_slope.clear()
        self.buf_inlier.clear()
        self.unsafe_start = None
        self.warned = False
        self.aborted = False
        self.count_unsafe = 0
        self.count_safe = 0

    def update(self, smoothed, assess, mode):
        if not mode or ('LAND' not in mode.upper() and 'RTL' not in mode.upper()):
            self._reset()
            return

        safe_frame = False
        if smoothed.get('numObj', 0) >= 3:
            s = assess.get('slope_deg', 0.0)
            i = assess.get('inlier_ratio', 0.0)
            self.buf_slope.append(s)
            self.buf_inlier.append(i)
            if len(self.buf_slope) == self.buf_slope.maxlen:
                if median(self.buf_slope) < self.th_slope and median(self.buf_inlier) > self.th_inlier:
                    safe_frame = True

        now = time.time()
        if safe_frame:
            self.count_safe += 1
            self.count_unsafe = 0
        else:
            self.count_unsafe += 1
            self.count_safe = 0

        if self.count_unsafe >= self.min_warn and not self.warned:
            if not self.unsafe_start:
                self.unsafe_start = now
            elif now - self.unsafe_start >= self.warn_time:
                self._abort()

        if (self.warned or self.aborted) and self.count_safe >= self.min_clear:
            self._reset()

    def _issue_warning(self, text):
        with mav_lock:  # Protect MAVLink access
            try:
                self.mav.mav.statustext_send(
                    mavutil.mavlink.MAV_SEVERITY_WARNING,
                    text.encode('utf-8'))
            except Exception:
                logger.exception("Failed to send warning")

    def _abort(self):
        with mav_lock:  # Protect MAVLink access
            try:
                self.mav.set_mode_loiter()
            except Exception:
                try:
                    self.mav.mav.command_long_send(
                        self.mav.target_system, self.mav.target_component,
                        mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
                        mavutil.mavlink.MAV_MODE_FLAG_AUTO_ENABLED, 2, 0, 0, 0, 0, 0
                    )
                except Exception:
                    logger.exception("Mode change command failed")
        self._issue_warning("Landing unsafe: aborting to LOITER")
        self.warned = True
        self.aborted = True

# ------------------ Radar Loop ------------------
def radar_loop(finder, despiker, frame_queue, stop_event, cli, data, cfg_path):
    """Dedicated radar data acquisition thread."""
    radar = None
    while not stop_event.is_set():
        try:
            if radar is None or not radar.data_serial.is_open:
                if radar:
                    radar.close()
                radar = RadarParser(cli, data, cfg_path, debug=False, enable_logging=False)
                radar.initialize_ports()
                radar.send_config()
                logger.info("Radar connected")

            header, det_obj, snr, noise = radar.read_frame()
            frame_queue.put((header, det_obj, snr, noise), block=True, timeout=1)

        except queue.Full:
            logger.warning("Frame queue full, dropping frame")
        except Exception:
            logger.exception("Radar loop error")
            radar = None
            time.sleep(1)

# ------------------ IMU Data Reader ------------------
def imu_reader(mav, imu_buffer, stop_event):
    """Dedicated thread for reading IMU data."""
    while not stop_event.is_set():
        try:
            with mav_lock:  # Protect MAVLink access
                msg = mav.recv_match(type='ATTITUDE', blocking=False)
            if msg:
                imu_buffer.update({
                    'roll': msg.roll,
                    'pitch': msg.pitch,
                    'yaw': msg.yaw,
                    'rollspeed': msg.rollspeed,
                    'pitchspeed': msg.pitchspeed,
                    'yawspeed': msg.yawspeed,
                    'time_boot_ms': msg.time_boot_ms
                })
            time.sleep(0.01)
        except Exception:
            logger.exception("IMU read error")
            time.sleep(0.1)

# ------------------ Autopilot Manager ------------------
class AutopilotManager:
    """Centralized autopilot connection and resource management."""
    def __init__(self, finder):
        self.finder = finder
        self.master = None
        self.mode_holder = {'mode': None}
        self.stop_mode_event = threading.Event()
        self.stop_imu_event = threading.Event()
        self.imu_buffer = IMUDataBuffer()
        self.components = {}
        
    def connect(self):
        while not self.stop_mode_event.is_set():
            if self.master is None:
                try:
                    master, _ = self.finder.find_autopilot_connection(timeout=2.0)
                    if master:
                        logger.info("Autopilot connected")
                        self._initialize_components(master)
                        return True
                    time.sleep(2)
                except Exception:
                    logger.exception("Autopilot connection failed")
                    time.sleep(2)
        return False

    def _initialize_components(self, master):
        self.master = master
        
        # Create components
        self.components['compensator'] = AttitudeCompensator(master)
        self.components['distance_sender'] = DistanceSensorSender(master)
        self.components['landing_monitor'] = LandingMonitor(master,
            buffer_size=20, slope_threshold_deg=5.0, inlier_threshold=0.6,
            warning_duration=3.0, min_consecutive_to_warn=5, min_consecutive_to_clear=5)
        
        # Start monitoring threads
        threading.Thread(target=mode_watcher, args=(master, self.stop_mode_event, self.mode_holder), daemon=True).start()
        threading.Thread(target=imu_reader, args=(master, self.imu_buffer, self.stop_imu_event), daemon=True).start()
        
        # Initial calibration
        self.components['compensator'].internal_calibrate_offsets(num_samples=100, delay=0.01)

    def close(self):
        self.stop_mode_event.set()
        self.stop_imu_event.set()
        if self.master:
            try:
                self.master.close()
            except Exception:
                pass
        self.master = None
        self.components = {}

    def get_mode(self):
        with mode_lock:
            return self.mode_holder.get('mode')

    def send_heartbeat(self):
        if self.master:
            with mav_lock:
                try:
                    self.master.mav.heartbeat_send(
                        mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
                        mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0
                    )
                except Exception:
                    logger.warning("Heartbeat send failed")

# ------------------ Main Processing Loop ------------------
def main_processing_loop(frame_queue, ap_manager, despiker, assessor, stop_event):
    """Core processing pipeline with shared memory model."""
    while not stop_event.is_set():
        try:
            # Get radar frame
            header, det_obj, snr, noise = frame_queue.get(timeout=2.0)
            
            # Process frame in parallel
            with ThreadPoolExecutor(max_workers=2) as executor:
                despike_future = executor.submit(despiker.process, det_obj, snr, noise)
                assess_future = executor.submit(assessor.assess, det_obj)
                despike_res = despike_future.result()
                safe, assess_res = assess_future.result()

            # Get closest distance
            closest_distance = min(despike_res['z']) if despike_res.get('numObj', 0) > 0 else None

            # Send distance if available
            if closest_distance is not None and 'distance_sender' in ap_manager.components:
                ap_manager.components['distance_sender'].send(closest_distance)
                ap_manager.send_heartbeat()

            # Transform pointcloud if compensator available
            smoothed = {'x': [], 'y': [], 'z': [], 'numObj': 0}
            if 'compensator' in ap_manager.components and despike_res.get('numObj', 0) >= 3:
                pts_body = np.vstack((despike_res['x'], despike_res['y'], despike_res['z'])).T
                try:
                    imu_data, _ = ap_manager.imu_buffer.get()
                    pts_enu = ap_manager.components['compensator'].transform_pointcloud(pts_body, imu_data)
                    smoothed = {
                        'x': pts_enu[:,0], 'y': pts_enu[:,1], 'z': pts_enu[:,2],
                        'numObj': despike_res['numObj']
                    }
                except Exception:
                    logger.exception("Compensation error")

            # Landing safety assessment
            mode = ap_manager.get_mode()
            if 'landing_monitor' in ap_manager.components and mode:
                ap_manager.components['landing_monitor'].update(smoothed, assess_res, mode)

            # Log assessment results
            coeffs = assess_res.get('plane') or []
            if len(coeffs) >= 4:
                status = "SAFE" if safe else "UNSAFE"
                logger.info(f"Landing zone {status}: slope={assess_res.get('slope_deg',0):.1f}deg, "
                           f"inliers={assess_res.get('inlier_ratio',0)*100:.0f}%")

        except queue.Empty:
            pass  # Normal timeout for frame queue
        except KeyboardInterrupt:
            stop_event.set()
        except Exception:
            logger.exception("Processing error")

# ------------------ Main ------------------
if __name__ == "__main__":
    # Initialize components
    finder = DevicePortFinder()
    despiker = RadarDespiker()
    assessor = LandingZoneAssessor(0.05, 500, 10, 0.7)
    ap_manager = AutopilotManager(finder)
    
    # Shared resources
    frame_queue = queue.Queue(maxsize=100)
    stop_main = threading.Event()
    stop_radar = threading.Event()

    # Start radar thread
    radar_thread = threading.Thread(
        target=radar_loop,
        args=(finder, despiker, frame_queue, stop_radar,
              '/dev/ttyUSB0', '/dev/ttyUSB1', 
              os.path.join(os.getcwd(), 'best_res_4cm.cfg')),
        daemon=True
    )
    radar_thread.start()

    # Main connection and processing loop
    try:
        while not stop_main.is_set():
            if ap_manager.connect():
                main_processing_loop(
                    frame_queue, ap_manager, despiker, assessor, stop_main
                )
            else:
                time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception:
        logger.exception("Unexpected error")
    finally:
        stop_main.set()
        stop_radar.set()
        ap_manager.close()
        radar_thread.join(timeout=1.0)
        sys.exit(0)