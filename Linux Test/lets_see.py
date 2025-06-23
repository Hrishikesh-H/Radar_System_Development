import os
import sys
import time
import serial
import struct
import threading
import traceback
import datetime
import numpy as np
from collections import deque
from pymavlink import mavutil
from serial.serialutil import SerialException
from statistics import median

#--------------------------------------------------------
from Parser import RadarParser  # We will overwrite this class below
from Filter import RadarDespiker
# from Plotter import RadarPlotter
# from GUI import DroneLandingStatus
from PortFinder import DevicePortFinder
from PlaneLand import LandingZoneAssessor
from IMUCompensator import AttitudeCompensator

# ============================ DistanceSensorSender ============================
class DistanceSensorSender:
    def __init__(self, mav, min_range=0.05, max_range=5.0, send_rate_hz=10):
        self.mav = mav
        self.min_distance_cm = max(0, min(int(min_range * 100), 65535))
        self.max_distance_cm = max(0, min(int(max_range * 100), 65535))
        
        # Fixed parameters
        self.SENSOR_TYPE = mavutil.mavlink.MAV_DISTANCE_SENSOR_LASER
        self.SENSOR_ID = 1  # Changed from 0 to avoid conflicts
        self.ROTATION_DOWNWARD = mavutil.mavlink.MAV_SENSOR_ROTATION_PITCH_270
        self.COVARIANCE_UNKNOWN = 0  # Changed from 255 for compatibility
        
        # MAVLink 2.0 parameters
        self.horizontal_fov = 0.0
        self.vertical_fov = 0.0
        self.quaternion = [0.0, 0.0, 0.0, 0.0]
        self.signal_quality = 0
        
        self.send_interval = 1.0 / send_rate_hz
        self._last_send_time = 0.0

    def send(self, distance_m):
        now = time.time()
        if (now - self._last_send_time) < self.send_interval:
            return
            
        self._last_send_time = now
        ts = datetime.datetime.now().isoformat()
        
        try:
            # Improved distance validation
            if distance_m is None or not isinstance(distance_m, (int, float)) or distance_m <= 0:
                current_cm = self.max_distance_cm
            else:
                raw_cm = int(round(distance_m * 100))
                current_cm = max(self.min_distance_cm, min(raw_cm, self.max_distance_cm))
            
            # Improved time_boot_ms calculation with fallback
            try:
                time_boot_ms = int(self.mav.time_since('SYSTEM_BOOT') * 1000)
                if time_boot_ms <= 0:
                    time_boot_ms = int((time.time() % (2**32)) * 1000)
            except:
                time_boot_ms = int((time.time() % (2**32)) * 1000)
            
            # Try MAVLink 2.0 first, fallback to 1.0
            try:
                self.mav.mav.distance_sensor_send(
                    time_boot_ms, self.min_distance_cm, self.max_distance_cm,
                    current_cm, self.SENSOR_TYPE, self.SENSOR_ID,
                    self.ROTATION_DOWNWARD, self.COVARIANCE_UNKNOWN,
                    self.horizontal_fov, self.vertical_fov,
                    self.quaternion, self.signal_quality
                )
            except Exception:
                # Fallback to MAVLink 1.0 format
                self.mav.mav.distance_sensor_send(
                    time_boot_ms, self.min_distance_cm, self.max_distance_cm,
                    current_cm, self.SENSOR_TYPE, self.SENSOR_ID,
                    self.ROTATION_DOWNWARD, self.COVARIANCE_UNKNOWN
                )
            
            # Small delay to ensure transmission
            time.sleep(0.001)
            
        except Exception as e:
            print(f"[{ts}] [ERROR] Distance sensor send failed: {e}")
            traceback.print_exc()



# ============================ ModeWatcher ============================

class ModeWatcher(threading.Thread):
    """
    Watches MAVLink HEARTBEAT messages to track current flight mode (e.g., LAND, RTL, LOITER).
    Updates self.current_mode as a string.
    """
    def __init__(self, mav):
        super().__init__(daemon=True)
        self.mav = mav
        self.current_mode = None
        self._stop_event = threading.Event()

    def run(self):
        while not self._stop_event.is_set():
            msg = self.mav.recv_match(type='HEARTBEAT', blocking=True, timeout=1)
            if msg:
                try:
                    self.current_mode = msg.custom_mode.decode('utf-8', errors='ignore')
                except:
                    base = msg.base_mode
                    self.current_mode = "ARMED" if (base & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED) else "DISARMED"
                print(f"[ModeWatcher] Current mode: {self.current_mode}")
            time.sleep(0.1)

    def stop(self):
        self._stop_event.set()


# ============================ LandingMonitor ============================

class LandingMonitor:
    """
    Evaluates landing surface flatness over recent frames and issues warnings/aborts if unsafe.
    """
    def __init__(self, mav, distance_sender,
                 buffer_size=20,
                 slope_threshold_deg=5.0,
                 inlier_threshold=0.6,
                 warning_duration=3.0,
                 min_consecutive_to_warn=5,
                 min_consecutive_to_clear=5):
        self.mav = mav
        self.distance_sender = distance_sender
        self.slope_buffer = deque(maxlen=buffer_size)
        self.inlier_buffer = deque(maxlen=buffer_size)
        self.safety_history = deque(maxlen=buffer_size)

        self.threshold_slope = slope_threshold_deg
        self.threshold_inlier = inlier_threshold
        self.warning_duration = warning_duration
        self.min_consecutive_to_warn = min_consecutive_to_warn
        self.min_consecutive_to_clear = min_consecutive_to_clear

        self.unsafe_start_time = None
        self.abort_issued = False
        self.currently_warned = False
        self.consec_unsafe_count = 0
        self.consec_safe_count = 0

    def update(self, smoothed_det_obj, assessment_result, mode):
        now = time.time()
        distance_m = None
        if smoothed_det_obj.get('numObj', 0) > 0:
            try:
                distance_m = float(min(smoothed_det_obj['z']))
            except Exception:
                distance_m = None

        self.distance_sender.send(distance_m)

        try:
            msg_fb = self.mav.recv_match(type='DISTANCE_SENSOR', blocking=False)
            if msg_fb and msg_fb.current_distance >= self.distance_sender.max_distance_cm:
                err_text = "FC distance sensor error: no data."
                print(f"[LandingMonitor][ERROR] {err_text}")
                try:
                    severity = mavutil.mavlink.MAV_SEVERITY_ERROR
                    self.mav.mav.statustext_send(severity, err_text.encode('utf-8'))
                except Exception:
                    pass
        except Exception as e:
            print(f"[LandingMonitor][ERROR] Failed to read back DISTANCE_SENSOR: {e}")

        if not mode:
            self._reset_state()
            return

        mode_upper = mode.upper()
        if "LAND" in mode_upper or "RTL" in mode_upper:
            num_points = smoothed_det_obj.get('numObj', 0)
            frame_is_safe = False

            if num_points < 3:
                print(f"[LandingMonitor] Frame has only {num_points} point(s) -> UNSAFE")
            else:
                slope_raw = assessment_result.get('slope_deg')
                inlier_raw = assessment_result.get('inlier_ratio')
                if slope_raw is None or inlier_raw is None:
                    print("[LandingMonitor] Missing slope/inlier -> UNSAFE")
                else:
                    self.slope_buffer.append(slope_raw)
                    self.inlier_buffer.append(inlier_raw)
                    if len(self.slope_buffer) == self.slope_buffer.maxlen:
                        median_slope = median(list(self.slope_buffer))
                        median_inlier = median(list(self.inlier_buffer))
                        frame_is_safe = (median_slope < self.threshold_slope) and (median_inlier > self.threshold_inlier)
                        print(f"[LandingMonitor] Median slope={median_slope:.2f}deg, "
                              f"Median inlier={median_inlier:.2f} -> {'SAFE' if frame_is_safe else 'UNSAFE'}")
                    else:
                        print(f"[LandingMonitor] Buffers not full (size={len(self.slope_buffer)}) -> UNSAFE")

            if frame_is_safe:
                self.consec_safe_count += 1
                self.consec_unsafe_count = 0
            else:
                self.consec_unsafe_count += 1
                self.consec_safe_count = 0

            self.safety_history.append(frame_is_safe)

            if self.consec_unsafe_count >= self.min_consecutive_to_warn and not self.currently_warned:
                if not self.unsafe_start_time:
                    self.unsafe_start_time = now
                    print(f"[LandingMonitor] {self.consec_unsafe_count} UNSAFE -> starting warning timer at t={now:.2f}")
                elif now - self.unsafe_start_time >= self.warning_duration and not self.abort_issued:
                    self._issue_abort()

            if (self.currently_warned or self.abort_issued) and self.consec_safe_count >= self.min_consecutive_to_clear:
                print(f"[LandingMonitor] {self.consec_safe_count} SAFE -> clearing warning/abort")
                self._reset_state()
        else:
            self._reset_state()

    def _issue_warning(self, text):
        severity = mavutil.mavlink.MAV_SEVERITY_WARNING
        try:
            self.mav.mav.statustext_send(severity, text.encode('utf-8'))
        except Exception:
            pass
        print(f"[LandingMonitor][WARNING] {text}")

    def _issue_abort(self):
        reason = "Landing unsafe: aborting and switching to LOITER. Please relocate to a flat area."
        print("[LandingMonitor] Aborting landing now (LOITER).")
        try:
            self.mav.set_mode_loiter()
            self._issue_warning(reason)
        except Exception:
            try:
                DO_SET_MODE = mavutil.mavlink.MAV_CMD_DO_SET_MODE
                MAV_MODE_FLAG_AUTO_ENABLED = mavutil.mavlink.MAV_MODE_FLAG_AUTO_ENABLED
                custom_mode_loiter = 2
                self.mav.mav.command_long_send(
                    self.mav.target_system,
                    self.mav.target_component,
                    DO_SET_MODE,
                    0,
                    MAV_MODE_FLAG_AUTO_ENABLED,
                    custom_mode_loiter,
                    0, 0, 0, 0, 0
                )
                self._issue_warning(reason)
            except Exception:
                print("[LandingMonitor][ERROR] Failed to switch to LOITER")
        self.abort_issued = True
        self.currently_warned = True

    def send_warning(self, text):
        self._issue_warning(text)

    def _reset_state(self):
        self.slope_buffer.clear()
        self.inlier_buffer.clear()
        self.safety_history.clear()
        self.unsafe_start_time = None
        self.abort_issued = False
        self.currently_warned = False
        self.consec_safe_count = 0
        self.consec_unsafe_count = 0


# ============================ RadarManager (Background) ============================

class RadarManager(threading.Thread):
    """
    Background thread that handles radar connection and reconnection.
    Maintains a shared global 'radar' reference, 'cli', 'data', and 'last_radar_time'.
    For Linux, uses explicit device paths /dev/ttyUSB0 and /dev/ttyUSB1.
    """
    def __init__(self, finder, despiker, assessor):
        super().__init__(daemon=True)
        self.finder = finder
        self.despiker = despiker
        self.assessor = assessor
        self._stop_event = threading.Event()

    def run(self):
        global radar, cli, data, last_radar_time
        reconnect_interval = 3.0

        while not self._stop_event.is_set():
            now = time.time()
            if (not radar or not getattr(radar, 'data_serial', None) or not radar.data_serial.is_open or
               (now - last_radar_time > 5.0)):
                if radar:
                    try:
                        radar.close()
                        print(f"[{datetime.datetime.now().isoformat()}] [Radar] Closed existing radar instance.")
                    except Exception as e:
                        print(f"[{datetime.datetime.now().isoformat()}] [Radar] Error closing radar: {e}")
                    radar = None
                    print(f"[{datetime.datetime.now().isoformat()}] [Radar] Restarting connection.")

                print(f"[{datetime.datetime.now().isoformat()}] [Radar] Attempting to reconnect...")
                try:
                    cli = "/dev/ttyUSB0"
                    data = "/dev/ttyUSB1"
                    print(f"[{datetime.datetime.now().isoformat()}] [Radar] Using CLI={cli}, DATA={data}")
                    config_path = os.path.join(os.getcwd(), "best_res_4cm.cfg")
                    radar = RadarParser(cli, data, config_path, debug=False, enable_logging=False, log_prefix="radar_log")
                    radar.initialize_ports()
                    radar.send_config()
                    time.sleep(2)
                    radar.info_print("Connected to radar successfully (Linux).")
                    last_radar_time = time.time()
                except Exception as e:
                    print(f"[Radar][Connection Error] {e.__class__.__name__}: {e} | Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
                    time.sleep(reconnect_interval)
                    continue
            else:
                time.sleep(0.1)

    def stop(self):
        self._stop_event.set()


# ============================ main_application.py ============================

compensator = None
last_att_time = 0
_autopilot_stop = False
calli_req = True
radar = None
last_radar_time = 0
cli = None
data = None
mode_watcher = None
distance_sender = None
landing_monitor = None


def _autopilot_reconnect_loop(finder):
    global compensator, last_att_time, _autopilot_stop, cli, data
    global mode_watcher, distance_sender, landing_monitor, calli_req

    reconnect_interval = 5.0

    while not _autopilot_stop:
        if compensator is None:
            try:
                master, _ = None, None
                try:
                    master, _ = finder.find_autopilot_connection(timeout=10.0, exclude_ports=[])
                except RuntimeError:
                    master = None

                if master is None and cli and data:
                    master, _ = finder.find_autopilot_connection(timeout=10.0, exclude_ports=[cli, data])

                if master:
                    compensator = AttitudeCompensator(master)
                    print("[Autopilot] MAVLink connection established.")
                    last_att_time = time.time()

                    if mode_watcher is None:
                        mode_watcher = ModeWatcher(master)
                        mode_watcher.start()

                        distance_sender = DistanceSensorSender(master)

                        landing_monitor = LandingMonitor(
                            master,
                            distance_sender,
                            buffer_size=20,
                            slope_threshold_deg=5.0,
                            inlier_threshold=0.6,
                            warning_duration=3.0,
                            min_consecutive_to_warn=5,
                            min_consecutive_to_clear=5
                        )
                time.sleep(reconnect_interval)
            except Exception:
                print(f"[Autopilot] Reconnect failed. Retrying in {reconnect_interval:.1f}s")
                traceback.print_exc()
                time.sleep(reconnect_interval)
        else:
            time.sleep(reconnect_interval)


if __name__ == "__main__":
    finder = DevicePortFinder()
    despiker = RadarDespiker()
    assessor = LandingZoneAssessor(0.05, 500, 10, 0.7)

    print(f"[{datetime.datetime.now().isoformat()}] [MAIN] Starting up, attempting to connect to Radar.")

    radar_manager = RadarManager(finder, despiker, assessor)
    radar_manager.start()

    _autopilot_stop = False
    _thread = threading.Thread(target=_autopilot_reconnect_loop, args=(finder,), daemon=True)
    _thread.start()

    try:
        while True:
            loop_start = time.time()
            if radar:
                try:
                    start_read = loop_start
                    header, det_obj, snr, noise = radar.read_frame()
                    elapsed_read = time.time() - start_read

                    if elapsed_read > 10.0:
                        raise RuntimeError(f"read_frame took {elapsed_read:.2f}s")

                    if header is None:
                        continue

                    last_radar_time = time.time()
                except (serial.SerialException, SerialException, RuntimeError) as e:
                    print(f"[{datetime.datetime.now().isoformat()}] [Radar][Error] {e}")
                    try:
                        radar.close()
                        print(f"[{datetime.datetime.now().isoformat()}] [Radar] Closed radar.")
                    except Exception as ce:
                        print(f"[{datetime.datetime.now().isoformat()}] [Radar] Close error: {ce}")
                    radar = None
                    continue
                except Exception as e:
                    print(f"[{datetime.datetime.now().isoformat()}] [Radar][Frame Error] {e}")
                    try:
                        radar.close()
                        print(f"[{datetime.datetime.now().isoformat()}] [Radar] Closed radar.")
                    except Exception as ce:
                        print(f"[{datetime.datetime.now().isoformat()}] [Radar] Close error: {ce}")
                    radar = None
                    continue

                smoothed_det_obj = {'x': [], 'y': [], 'z': [], 'numObj': 0}
                if det_obj and det_obj.get('numObj', 0) >= 3:
                    spr = despiker.process(det_obj, snr, noise)
                    pts_body = np.vstack((spr['x'], spr['y'], spr['z'])).T

                    if compensator:
                        try:
                            if calli_req:
                                compensator.internal_calibrate_offsets(num_samples=100, delay=0.01)
                                calli_req = False

                            pts_enu = compensator.transform_pointcloud(pts_body)
                            last_att_time = time.time()
                        except Exception as e:
                            print(f"[Autopilot][Comp Error] {e}. Using raw points.")
                            traceback.print_exc()
                            try:
                                compensator.close()
                            except Exception:
                                pass
                            compensator = None
                            pts_enu = pts_body
                    else:
                        pts_enu = pts_body

                    smoothed_det_obj = {
                        'x': pts_enu[:, 0],
                        'y': pts_enu[:, 1],
                        'z': pts_enu[:, 2],
                        'numObj': det_obj['numObj']
                    }

                safe, m = assessor.assess(det_obj)
                coeffs = m.get('plane')
                if not coeffs or len(coeffs) < 4:
                    radar.warn_print("Landing zone UNSAFE (Insufficient data)")
                else:
                    if safe:
                        radar.info_print(
                            f"Landing zone SAFE slope={m['slope_deg']:.1f}deg, "
                            f"inliers={m['inlier_ratio']*100:.0f}%, res={m['mean_residual']*100:.1f}cm"
                        )
                    else:
                        radar.warn_print(
                            f"Landing zone UNSAFE ({m.get('reason','')}) slope={m.get('slope_deg',0):.1f}deg, "
                            f"inliers={m.get('inlier_ratio',0)*100:.0f}%, res={m.get('mean_residual',0)*100:.1f}cm"
                        )

                if mode_watcher and landing_monitor:
                    landing_monitor.update(smoothed_det_obj, m, mode_watcher.current_mode)
            else:
                time.sleep(0.1)

    except KeyboardInterrupt:
        print(f"[{datetime.datetime.now().isoformat()}] [System] Interrupted by user. Exiting...")
    except Exception as e:
        print(f"[{datetime.datetime.now().isoformat()}] [System] Unexpected: {e}")
        traceback.print_exc()
        time.sleep(1)
    finally:
        _autopilot_stop = True
        if mode_watcher:
            mode_watcher.stop()
        _thread.join(timeout=1.0)

        radar_manager.stop()
        radar_manager.join(timeout=1.0)

        if radar:
            try:
                radar.close()
            except Exception as e:
                print(f"[{datetime.datetime.now().isoformat()}] [Cleanup] Failed to close radar: {e}")
        if compensator:
            try:
                compensator.close()
            except Exception as e:
                print(f"[{datetime.datetime.now().isoformat()}] [Cleanup] Failed to close compensator: {e}")
