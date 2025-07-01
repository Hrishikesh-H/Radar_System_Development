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

# ------------------ Sensor Sender ------------------
class DistanceSensorSender:
    """Send MAVLink DISTANCE_SENSOR messages with improved reliability."""
    def __init__(self, mav, min_range=0.05, max_range=5.0, rate_hz=20):
        self.mav = mav
        self.min_cm = max(0, min(int(min_range * 100), 65535))
        self.max_cm = max(0, min(int(max_range * 100), 65535))
        self.interval = 1.0 / rate_hz
        self._last_time = 0  # Initialize to 0 to ensure first message is sent
        self._sequence = 0  # Add sequence counter
        self._last_sent_distance = None

    def send(self, distance_m):
        now = time.monotonic()
        
        # Skip if rate limit not exceeded and distance hasn't changed
        if (now - self._last_time < self.interval and 
            distance_m == self._last_sent_distance):
            return
            
        # Calculate distance in cm
        if distance_m is None:
            dist_cm = 0  # Use 0 instead of max_cm for invalid measurements
        else:
            try:
                raw_cm = int(round(float(distance_m) * 100))
                dist_cm = max(self.min_cm, min(raw_cm, self.max_cm))
            except (TypeError, ValueError):
                dist_cm = 0

        # Only send if we have a valid MAVLink connection
        if not hasattr(self.mav, 'mav') or not hasattr(self.mav.mav, 'distance_sensor_send'):
            logger.warning("MAVLink connection not ready for distance sensor")
            return

        try:
            # Use distance_sensor_send which is more reliable than encode+send
            self.mav.mav.distance_sensor_send(
                0,  # time_boot_ms (not used by most flight controllers)
                self.min_cm,
                self.max_cm,
                dist_cm,
                mavutil.mavlink.MAV_DISTANCE_SENSOR_RADAR,
                self._sequence % 256,  # Sensor ID (using sequence counter)
                mavutil.mavlink.MAV_SENSOR_ROTATION_PITCH_270,
                0  # Covariance (0 for unknown)
            )
            
            self._sequence += 1
            self._last_time = now
            self._last_sent_distance = distance_m
            logger.debug(f"Sent DISTANCE_SENSOR: {dist_cm}cm (seq: {self._sequence})")
            
        except (AttributeError, mavutil.mavlink.MAVError) as e:
            logger.error(f"Distance send failed: {str(e)}")
        except Exception as e:
            logger.exception("Unexpected error in distance send")

# ------------------ Mode Watcher ------------------
def mode_watcher(mav, mode_holder, stop_event):
    """Continuously read HEARTBEAT and update mode_holder['mode']."""
    while not stop_event.is_set():
        try:
            # Flush buffer periodically
            if hasattr(mav.mav, 'buffer'):
                mav.mav.buffer.flush()
                
            msg = mav.recv_match(type='HEARTBEAT', blocking=True, timeout=1)
            if msg:
                try:
                    mode_holder['mode'] = msg.custom_mode.decode('utf-8')
                except Exception:
                    base = msg.base_mode
                    mode_holder['mode'] = (
                        'ARMED' if (base & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                        else 'DISARMED'
                    )
        except (serial.SerialException, mavutil.mavutil.MAVError) as e:
            logger.warning("Heartbeat read error: %s", e)
            time.sleep(0.1)
        except Exception:
            logger.exception("Unexpected error in mode_watcher")
            time.sleep(0.1)

# ------------------ Landing Monitor ------------------
class LandingMonitor:
    """Evaluate landing safety; issue warnings and abort if necessary."""
    def __init__(self, mav, **cfg):
        self.mav = mav
        self.buf_slope  = deque(maxlen=cfg.get('buffer_size', 20))
        self.buf_inlier = deque(maxlen=cfg.get('buffer_size', 20))
        self.th_slope   = cfg.get('slope_threshold_deg', 5.0)
        self.th_inlier  = cfg.get('inlier_threshold', 0.6)
        self.warn_time  = cfg.get('warning_duration', 3.0)
        self.min_warn   = cfg.get('min_consecutive_to_warn', 5)
        self.min_clear  = cfg.get('min_consecutive_to_clear', 5)
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
        try:
            self.mav.mav.statustext_send(
                mavutil.mavlink.MAV_SEVERITY_WARNING,
                text.encode('utf-8')
            )
        except Exception:
            logger.exception("Failed to send warning")

    def _abort(self):
        try:
            self.mav.set_mode_loiter()
        except Exception:
            self.mav.mav.command_long_send(
                self.mav.target_system,
                self.mav.target_component,
                mavutil.mavlink.MAV_CMD_DO_SET_MODE,
                0,
                mavutil.mavlink.MAV_MODE_FLAG_AUTO_ENABLED,
                2, 0, 0, 0, 0, 0
            )
        self._issue_warning("Landing unsafe: aborting and switching to LOITER")
        self.warned = True
        self.aborted = True

# ------------------ Radar Loop ------------------
def radar_loop(finder, despiker, assessor, frame_queue, stop_event, cli, data, cfg_path, shared):
    """Continuously read radar frames and put them into frame_queue."""
    while not stop_event.is_set():
        try:
            # Connect or reconnect
            radar = shared.get('radar')
            if radar is None or not getattr(radar, 'data_serial', None) or not radar.data_serial.is_open:
                if radar:
                    radar.close()
                    logger.info("Radar connection closed")
                radar = RadarParser(cli, data, cfg_path, debug=False, enable_logging=False)
                radar.initialize_ports()
                radar.send_config()
                shared['radar'] = radar
                logger.info("Radar connected")

            header, det_obj, snr, noise = radar.read_frame()
            frame_queue.put((header, det_obj, snr, noise), block=True, timeout=1)
            shared['last_radar_time'] = time.time()

        except queue.Full:
            logger.warning("Frame queue full, dropping frame")
        except Exception as e:
            logger.exception("Radar loop error")
            shared['radar'] = None
            time.sleep(1.0)

# ------------------ Autopilot Loop ------------------
def autopilot_loop(finder, shared, stop_event):
    """Persistent autopilot connection, calibration, and mode watching."""
    while not stop_event.is_set():
        if shared.get('master') is None:
            try:
                master, _ = finder.find_autopilot_connection(
                    timeout=2.0,
                    exclude_ports=[shared['cli'], shared['data']]
                )
                if master:
                    logger.info("Autopilot connected %s/%s", master.target_system, master.target_component)
                    shared['master'] = master

                    # Inside autopilot_loop, after master connection is established:
                    shared['last_mavlink_activity'] = time.time()
                    shared['mavlink_healthy'] = True

                    # Add periodic heartbeat when master is connected
                    if shared.get('master'):
                        try:
                            shared['master'].mav.heartbeat_send(
                                mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
                                mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                                0, 0, 0
                            )
                            shared['last_mavlink_activity'] = time.time()
                        except Exception:
                            logger.warning("Heartbeat send failed, resetting connection")
                            shared['master'] = None

                    comp = AttitudeCompensator(master)
                    comp.internal_calibrate_offsets(num_samples=100, delay=0.01)
                    shared['compensator'] = comp

                    shared['distance_sender'] = DistanceSensorSender(master)

                    shared['landing_monitor'] = LandingMonitor(
                        master,
                        buffer_size=20,
                        slope_threshold_deg=5.0,
                        inlier_threshold=0.6,
                        warning_duration=3.0,
                        min_consecutive_to_warn=5,
                        min_consecutive_to_clear=5
                    )

                    shared['mode_holder'] = {'mode': None}
                    shared['mode_watcher_started'] = True
                    stop_mode_event = threading.Event()
                    shared['stop_mode_event'] = stop_mode_event
                    shared['io_executor'].submit(
                        mode_watcher,
                        master,
                        shared['mode_holder'],
                        stop_mode_event
                    )
                else:
                    time.sleep(2.0)
            except serial.SerialException as e:
                logger.warning("Autopilot connection error: %s", e)
                time.sleep(2.0)
            except Exception:
                logger.exception("Unexpected in autopilot_loop")
                time.sleep(2.0)
        else:
            time.sleep(1.0)

# ------------------ Main ------------------
if __name__ == "__main__":
    finder   = DevicePortFinder()
    despiker = RadarDespiker()
    assessor = LandingZoneAssessor(0.05, 500, 10, 0.7)

    frame_queue = queue.Queue(maxsize=100)

    shared = {
        'cli': '/dev/ttyUSB0',
        'data': '/dev/ttyUSB1',
        'master': None,
        'radar': None,
        'compensator': None,
        'distance_sender': None,
        'landing_monitor': None,
        'mode_holder': None,
        'mode_watcher_started': False,
        'io_executor': None,
        'last_radar_time': time.time()
    }

    io_executor  = ThreadPoolExecutor(max_workers=2)
    cpu_executor = ProcessPoolExecutor(max_workers=2)
    shared['io_executor'] = io_executor

    stop_radar = threading.Event()
    stop_auto  = threading.Event()

    io_executor.submit(
        radar_loop,
        finder, despiker, assessor,
        frame_queue, stop_radar,
        shared['cli'], shared['data'],
        os.path.join(os.getcwd(), 'best_res_4cm.cfg'),
        shared
    )

    io_executor.submit(autopilot_loop, finder, shared, stop_auto)

    try:
        while True:
            header, det_obj, snr, noise = frame_queue.get(timeout=2.0)

            despike_f = cpu_executor.submit(despiker.process, det_obj, snr, noise)
            assess_f  = cpu_executor.submit(assessor.assess, det_obj)
            despike_res = despike_f.result()
            safe, assess_res = assess_f.result()

            # Calculate closest distance
            closest_distance = None
            if despike_res.get('numObj', 0) > 0:
                closest_distance = min(despike_res['z'])
            
            # Send distance to autopilot
            # In your main loop where you process radar data:
            if closest_distance is not None:
                ds = shared.get('distance_sender')
                if ds and shared.get('master') is not None:
                    try:
                        # Add small delay to ensure MAVLink connection is ready
                        if time.time() - shared.get('last_mavlink_activity', 0) > 1.0:
                            shared['master'].mav.heartbeat_send(
                                mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
                                mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                                0, 0, 0
                            )
                            shared['last_mavlink_activity'] = time.time()
                        
                        ds.send(closest_distance)
                    except Exception as e:
                        logger.error(f"Failed to send distance: {e}")
                        shared['master'] = None  # Force reconnection

            smoothed = {'x': [], 'y': [], 'z': [], 'numObj': 0}
            comp = shared.get('compensator')
            if despike_res.get('numObj', 0) >= 3 and comp:
                pts_body = np.vstack((despike_res['x'], despike_res['y'], despike_res['z'])).T
                try:
                    pts_enu = comp.transform_pointcloud(pts_body)
                    logger.info("Compensated first point: %s", pts_enu[0])
                except Exception:
                    logger.exception("Compensation error")
                    comp.close()
                    shared['compensator'] = None
                    pts_enu = pts_body
                smoothed = {
                    'x': pts_enu[:,0],
                    'y': pts_enu[:,1],
                    'z': pts_enu[:,2],
                    'numObj': despike_res['numObj']
                }

            radar = shared.get('radar')
            coeffs = assess_res.get('plane') or []
            if len(coeffs) < 4:
                if radar: radar.warn_print("Landing zone UNSAFE (Insufficient data)")
            else:
                if safe:
                    if radar: radar.info_print(
                        f"Landing zone SAFE slope={assess_res['slope_deg']:.1f}deg, "
                        f"inliers={assess_res['inlier_ratio']*100:.0f}%, "
                        f"res={assess_res['mean_residual']*100:.1f}cm"
                    )
                else:
                    if radar: radar.warn_print(
                        f"Landing zone UNSAFE ({assess_res.get('reason','')}) "
                        f"slope={assess_res.get('slope_deg',0):.1f}deg, "
                        f"inliers={assess_res.get('inlier_ratio',0)*100:.0f}%, "
                        f"res={assess_res.get('mean_residual',0)*100:.1f}cm"
                    )

            lm = shared.get('landing_monitor')
            mode = (shared.get('mode_holder') or {}).get('mode')
            if lm:
                lm.update(smoothed, assess_res, mode)

    except KeyboardInterrupt:
        logger.info("Interrupted by user, shutting down...")

    except Exception:
        logger.exception("Unexpected error in main loop")

    finally:
        stop_radar.set()
        stop_auto.set()
        io_executor.shutdown(wait=False, cancel_futures=True)
        cpu_executor.shutdown(wait=False, cancel_futures=True)
        sys.exit(0)