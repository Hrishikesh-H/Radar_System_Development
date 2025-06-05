import os
import sys
import time
import serial
import threading
import traceback
import datetime
import numpy as np
from collections import deque
from PyQt6.QtWidgets import QApplication
from pymavlink import mavutil

#--------------------------------------------------------
from Parser import RadarParser
from Filter import RadarDespiker
# from Plotter import RadarPlotter
# from GUI import DroneLandingStatus
from PortFinder import DevicePortFinder
from PlaneLand import LandingZoneAssessor
from IMUCompensator import AttitudeCompensator
from serial.serialutil import SerialException

# ============================ NEW CLASSES ============================

class DistanceSensorSender:
    """
    Handles sending MAVLink DISTANCE_SENSOR messages continuously, with console logging.
    """
    def __init__(self, mav, min_range=0.05, max_range=5.0):
        """
        mav: an active pymavlink connection (master)
        min_range, max_range: sensor bounds in meters
        """
        self.mav = mav
        self.min_range_m = min_range
        self.max_range_m = max_range
        self.min_distance_cm = max(0, min(int(self.min_range_m * 100), 65535))
        self.max_distance_cm = max(0, min(int(self.max_range_m * 100), 65535))
        self.SENSOR_TYPE = mavutil.mavlink.MAV_DISTANCE_SENSOR_LASER
        self.SENSOR_ID = 0
        self.ROTATION_DOWNWARD = mavutil.mavlink.MAV_SENSOR_ROTATION_PITCH_270
        self.COVARIANCE_UNKNOWN = 255

    def send(self, distance_m):
        """
        Send a single DISTANCE_SENSOR message based on distance in meters.
        If distance_m is None or invalid, sends max_distance to indicate no reading.
        Logs the raw and clamped distances to console.
        """
        try:
            if distance_m is not None and isinstance(distance_m, (int, float)):
                dist_cm = int(round(distance_m * 100))
                # clamp
                clamped_cm = max(self.min_distance_cm, min(dist_cm, self.max_distance_cm))
                print(f"[DistanceSensor] Raw distance = {distance_m:.3f} m → {dist_cm} cm, Clamped = {clamped_cm} cm")
                dist_cm = clamped_cm
            else:
                # No valid reading → send max range
                dist_cm = self.max_distance_cm
                print(f"[DistanceSensor] No valid radar reading. Sending max range = {dist_cm} cm")

            # Timestamp since system boot in ms
            time_boot_ms = int(self.mav.time_since('SYSTEM_BOOT') * 1000)

            # Send the MAVLink distance_sensor message
            self.mav.mav.distance_sensor_send(
                time_boot_ms,
                self.min_distance_cm,
                self.max_distance_cm,
                dist_cm,
                self.SENSOR_TYPE,
                self.SENSOR_ID,
                self.ROTATION_DOWNWARD,
                self.COVARIANCE_UNKNOWN
            )
            print(f"[DistanceSensor] Sent DISTANCE_SENSOR message: {dist_cm} cm")

            # Optional: capture any DISTANCE_SENSOR reply for confirmation
            msg = self.mav.recv_match(type='DISTANCE_SENSOR', blocking=False)
            if msg:
                print(f"[DistanceSensor] FC reply: DISTANCE_SENSOR = {msg.current_distance} cm")
        except Exception as e:
            ts = datetime.datetime.now().isoformat()
            print(f"[DistanceSensor][ERROR][{ts}] send failed: {e}\n{traceback.format_exc()}")


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
                    if base & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED:
                        self.current_mode = "ARMED"
                    else:
                        self.current_mode = "DISARMED"
                print(f"[ModeWatcher] Current mode: {self.current_mode}")
            time.sleep(0.1)

    def stop(self):
        self._stop_event.set()


class LandingMonitor:
    """
    Evaluates landing surface flatness over recent frames and issues warnings/aborts if unsafe.
    - Buffers recent slope and inlier_ratio values.
    - Determines 'flat' if median slope < slope_threshold AND median inlier_ratio > inlier_threshold.
    - When in LAND or RTL mode, continuously sends distance sensor and checks surface:
        * If unsafe for more than warning_duration seconds, abort landing (switch to LOITER).
        * Otherwise, if returns to safe before timeout, cancel abort.
    Additionally prints debug information and reads back the FC’s reported distance after sending.
    """
    def __init__(self, mav, distance_sender,
                 buffer_size=10,
                 slope_threshold_deg=5.0,
                 inlier_threshold=0.6,
                 warning_duration=3.0):
        """
        mav: pymavlink connection
        distance_sender: DistanceSensorSender instance
        buffer_size: how many past frames to use
        slope_threshold_deg: max median slope (degrees) to consider flat
        inlier_threshold: min median inlier_ratio to consider flat
        warning_duration: seconds to wait before aborting landing if unsafe
        """
        self.mav = mav
        self.distance_sender = distance_sender
        self.slope_buffer = deque(maxlen=buffer_size)
        self.inlier_buffer = deque(maxlen=buffer_size)
        self.threshold_slope = slope_threshold_deg
        self.threshold_inlier = inlier_threshold
        self.warning_duration = warning_duration
        self.unsafe_start_time = None
        self.abort_issued = False

    def update(self, smoothed_det_obj, assessment_result, mode):
        """
        Called each cycle with:
        - smoothed_det_obj: dict with keys 'x','y','z','numObj'
        - assessment_result: dict returned by LandingZoneAssessor.assess()
        - mode: current flight mode string
        Behavior:
         1. Compute distance from smoothed_det_obj (min z).
         2. Send distance to FC via distance_sender and log.
         3. After sending, read back any DISTANCE_SENSOR from FC and print it.
         4. If mode in ['LAND','RTL'], evaluate surface flatness.
             - If unsafe detected, record start time, send warning via MAVLink STATUSTEXT.
             - If unsafe persists beyond warning_duration, switch to LOITER.
             - If returns safe before warning_duration, clear timers.
         5. Print debug info: raw slope & inlier each frame, median when available.
        """
        # 1. Compute distance (in meters) as minimum z value if available
        distance_m = None
        if smoothed_det_obj.get('numObj', 0) > 0:
            try:
                distance_m = float(np.min(smoothed_det_obj['z']))
            except Exception:
                distance_m = None

        # 2. Send distance sensor MAVLink every call
        self.distance_sender.send(distance_m)

        # 3. Read back any DISTANCE_SENSOR from FC and print it
        try:
            msg_fb = self.mav.recv_match(type='DISTANCE_SENSOR', blocking=False)
            if msg_fb:
                print(f"[LandingMonitor] FC reports distance: {msg_fb.current_distance} cm")
        except Exception as e:
            print(f"[LandingMonitor][ERROR] Failed to read back DISTANCE_SENSOR: {e}")

        # 4. Landing/RTL specific logic
        if mode is None:
            return

        mode_upper = mode.upper()
        # Debug: print raw slope and inlier if available
        slope_raw = assessment_result.get('slope_deg', None)
        inlier_raw = assessment_result.get('inlier_ratio', None)
        if slope_raw is not None and inlier_raw is not None:
            print(f"[LandingMonitor] Debug raw - Slope: {slope_raw:.2f}°, Inlier: {inlier_raw:.2f}")

        if 'LAND' in mode_upper or 'RTL' in mode_upper:
            if slope_raw is not None and inlier_raw is not None:
                self.slope_buffer.append(slope_raw)
                self.inlier_buffer.append(inlier_raw)

            if len(self.slope_buffer) >= self.slope_buffer.maxlen:
                median_slope = np.median(np.array(self.slope_buffer))
                median_inlier = np.median(np.array(self.inlier_buffer))
                flat = (median_slope < self.threshold_slope) and (median_inlier > self.threshold_inlier)
                print(f"[LandingMonitor] Median slope: {median_slope:.2f}°, "
                      f"Median inlier: {median_inlier:.2f}, Flat={flat}")
            else:
                flat = True
                print(f"[LandingMonitor] Not enough data for median; assuming flat.")

            if not flat:
                now = time.time()
                if self.unsafe_start_time is None:
                    self.unsafe_start_time = now
                    self.send_warning("Landing surface unsafe: non-flat detected. Please relocate.")
                elif not self.abort_issued and (now - self.unsafe_start_time) > self.warning_duration:
                    self.send_abort_and_loiter()
                    self.abort_issued = True
            else:
                self.unsafe_start_time = None
                self.abort_issued = False
        else:
            self.unsafe_start_time = None
            self.abort_issued = False

    def send_warning(self, text):
        """
        Send a MAVLink STATUSTEXT with severity WARNING (3).
        Also print to console.
        """
        severity = mavutil.mavlink.MAV_SEVERITY_WARNING
        try:
            self.mav.mav.statustext_send(severity, text.encode('utf-8'))
        except Exception as e:
            print(f"[LandingMonitor][ERROR] Unable to send STATUSTEXT: {e}")
        print(f"[LandingMonitor][WARNING] {text}")

    def send_abort_and_loiter(self):
        """
        Abort landing by commanding flight mode change to LOITER.
        Sends a MAV_CMD_DO_SET_MODE command_long.
        """
        try:
            self.mav.set_mode_loiter()
            self.send_warning("Landing aborted: switching to LOITER. Please relocate and try again.")
        except Exception:
            try:
                DO_SET_MODE = mavutil.mavlink.MAV_CMD_DO_SET_MODE
                MAV_MODE_FLAG_AUTO_ENABLED = mavutil.mavlink.MAV_MODE_FLAG_AUTO_ENABLED
                custom_mode_loiter = 2  # ArduPilot LOITER mode
                self.mav.mav.command_long_send(
                    self.mav.target_system,
                    self.mav.target_component,
                    DO_SET_MODE,
                    0,
                    MAV_MODE_FLAG_AUTO_ENABLED,
                    custom_mode_loiter,
                    0, 0, 0, 0, 0
                )
                self.send_warning("Landing aborted: commanded LOITER via COMMAND_LONG. Please relocate and try again.")
            except Exception as e2:
                print(f"[LandingMonitor][ERROR] Failed to switch to LOITER: {e2}")

# ========================= END NEW CLASSES ==========================

# main_application.py

# Shared state for autopilot
compensator = None
last_att_time = 0
_autopilot_stop = False
calli_req = True  # Set True once here for a single calibration call

# Radar port names, to avoid scanning them
cli = None
data = None

# Instances for new functionality (initialized after MAVLink connection)
mode_watcher = None
distance_sender = None
landing_monitor = None

def _autopilot_reconnect_loop(finder):
    """
    Background thread: periodically tries to establish MAVLink connection.
    When successful, assigns to module-level 'compensator' and updates 'last_att_time'.
    Also starts ModeWatcher, DistanceSensorSender, and LandingMonitor when first connected.
    """
    global compensator, last_att_time, _autopilot_stop, cli, data
    global mode_watcher, distance_sender, landing_monitor, calli_req

    reconnect_interval = 5.0  # attempt every 5 seconds

    while not _autopilot_stop:
        if compensator is None:
            try:
                # First try UDP-only (no serial scan)
                master, ap_name = None, None
                try:
                    master, ap_name = finder.find_autopilot_connection(timeout=2.0, exclude_ports=[])  # skip serial fallback
                except RuntimeError:
                    master = None

                # If UDP failed and we know radar ports, try serial excluding them
                if master is None and cli and data:
                    master, ap_name = finder.find_autopilot_connection(timeout=2.0, exclude_ports=[cli, data])

                if master is not None:
                    compensator = AttitudeCompensator(master)
                    print("[Autopilot] MAVLink connection established for compensation (background).")
                    last_att_time = time.time()

                    # Initialize new functionality once
                    if mode_watcher is None:
                        # ModeWatcher
                        mode_watcher = ModeWatcher(master)
                        mode_watcher.start()

                        # DistanceSensorSender
                        distance_sender = DistanceSensorSender(master)

                        # LandingMonitor
                        landing_monitor = LandingMonitor(
                            master,
                            distance_sender,
                            buffer_size=10,
                            slope_threshold_deg=5.0,
                            inlier_threshold=0.6,
                            warning_duration=3.0
                        )
                time.sleep(reconnect_interval)
            except Exception as e:
                print(f"[Autopilot] Background reconnect failed: {e}. Retrying in {reconnect_interval:.1f}s…")
                traceback.print_exc()
                time.sleep(reconnect_interval)
        else:
            time.sleep(reconnect_interval)


# Add this constant at the top of the file
ENABLE_TRACE = False

def maybe_traceback():
    if ENABLE_TRACE:
        traceback.print_exc()

if __name__ == "__main__":
    finder = DevicePortFinder()
    radar = None
    # cli and data will be set after radar connects once

    # plotter = RadarPlotter()
    despiker = RadarDespiker()
    assessor = LandingZoneAssessor(0.05, 500, 10, 0.7)

    # app = QApplication(sys.argv)
    # gui = DroneLandingStatus()
    # gui.show()

    last_radar_time = 0
    RECONNECT_INTERVAL = 3.0

    # Start background thread for autopilot reconnection
    _autopilot_stop = False
    _thread = threading.Thread(target=_autopilot_reconnect_loop, args=(finder,), daemon=True)
    _thread.start()

    try:
        while True:
            now = time.time()

            # --- RADAR RECONNECTION LOGIC ---
            if radar is None or not radar.data_serial or not radar.data_serial.is_open or (now - last_radar_time > 5.0):
                if radar:
                    try:
                        radar.close()
                        print("[Radar] Closed existing radar instance.")
                    except Exception as e:
                        print(f"[Radar] Error during radar close: {e}")
                    radar = None
                    print("[Radar] Previous connection closed due to timeout or error.")

                print("[Radar] Connection required — attempting to reconnect...")
                try:
                    cli, data = finder.find_radar_ports_by_description()
                    config_path = os.path.join(os.getcwd(), "best_res_4cm.cfg")
                    radar = RadarParser(cli, data, config_path, debug=False, enable_logging=False, log_prefix="radar_log")
                    radar.initialize_ports()
                    radar.send_config()
                    time.sleep(2)
                    radar.info_print("Connected to radar successfully.")
                    last_radar_time = time.time()
                except RuntimeError as e:
                    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
                    print(f"[Radar][Port Detection Error] {e.__class__.__name__}: {e} | Time: {current_time} | Ports Found: {getattr(e, 'matches', 'unknown')} | Type: RuntimeError. Retrying in {RECONNECT_INTERVAL:.1f}s...")
                    maybe_traceback()
                    time.sleep(RECONNECT_INTERVAL)
                    continue
                except Exception as e:
                    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
                    print(f"[Radar][Unknown Connection Error] {e.__class__.__name__}: {e} | Time: {current_time}. Retrying in {RECONNECT_INTERVAL:.1f}s...")
                    maybe_traceback()
                    time.sleep(RECONNECT_INTERVAL)
                    continue  # skip downstream until radar returns

            # --- RADAR DATA ACQUISITION ---
            try:
                header, det_obj, snr, noise = radar.read_frame()
                last_radar_time = time.time()
            except (serial.SerialException, SerialException) as e:
                print(f"[Radar][Serial Error] {e.__class__.__name__}: {e}. Closing radar and restarting reconnection...")
                maybe_traceback()
                try:
                    radar.close()
                    print("[Radar] Closed radar after SerialException.")
                except Exception as ce:
                    print(f"[Radar] Exception during close after SerialException: {ce}")
                radar = None
                continue  # go back to radar reconnect
            except Exception as e:
                print(f"[Radar][Frame Read Error] {e.__class__.__name__}: {e}. Closing radar instance and restarting reconnection...")
                maybe_traceback()
                try:
                    radar.close()
                    print("[Radar] Closed radar after general read failure.")
                except Exception as ce:
                    print(f"[Radar] Exception during close after general failure: {ce}")
                radar = None
                continue  # go back to radar reconnect

            # --- POINT PROCESSING AND COMPENSATION ---
            smoothed_det_obj = {'x': [], 'y': [], 'z': [], 'numObj': 0}
            if det_obj and det_obj.get('numObj', 0) >= 3:
                spr = despiker.process(det_obj, snr, noise)
                pts_body = np.vstack((spr['x'], spr['y'], spr['z'])).T

                if compensator:
                    try:
                        if calli_req:
                            # Call calibration once
                            compensator.internal_calibrate_offsets(num_samples=100, delay=0.01)
                            calli_req = False  # Reset after one call

                        pts_enu = compensator.transform_pointcloud(pts_body)
                        last_att_time = time.time()
                    except Exception as e:
                        print(f"[Autopilot][Compensation Error] {e.__class__.__name__}: {e}. Switching to raw points.")
                        maybe_traceback()
                        try:
                            compensator.close()
                        except Exception as ce:
                            print(f"[Autopilot] Exception during compensator close: {ce}")
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

            # --- PLOTTING AND ASSESSMENT ---
            # plotter.update(det_obj, smoothed_det_obj)

            safe, m = assessor.assess(det_obj)
            coeffs = m.get('plane', None)
            if coeffs is None or not hasattr(coeffs, '__len__') or len(coeffs) < 4:
                radar.warn_print("Landing zone UNSAFE (Insufficient data for slope estimation)")
            else:
                if safe:
                    radar.info_print(
                        f"Landing zone SAFE  slope={m['slope_deg']:.1f}°, "
                        f"inliers={m['inlier_ratio']*100:.0f}%, res={m['mean_residual']*100:.1f}cm"
                    )
                else:
                    radar.warn_print(
                        f"Landing zone UNSAFE ({m.get('reason','')})  "
                        f"slope={m.get('slope_deg',0):.1f}°, "
                        f"inliers={m.get('inlier_ratio',0)*100:.0f}%, res={m.get('mean_residual',0)*100:.1f}cm"
                    )

            # --- NEW: DISTANCE SENDING & LANDING MONITORING ---
            if mode_watcher and landing_monitor:
                current_mode = mode_watcher.current_mode
                # This call happens every iteration, so distance gets sent continuously:
                landing_monitor.update(smoothed_det_obj, m, current_mode)

            # gui.update_status(safe, m)
            # app.processEvents()

    except KeyboardInterrupt:
        print("[System] Interrupted by user. Exiting cleanly...")
    except Exception as e:
        print(f"[System] Unexpected error: {e.__class__.__name__}: {e}")
        maybe_traceback()
        time.sleep(2)
    finally:
        _autopilot_stop = True
        if mode_watcher:
            mode_watcher.stop()
        _thread.join(timeout=1.0)

        # --- CLEANUP ---
        if radar:
            try:
                radar.close()
            except Exception as e:
                print(f"[Cleanup] Failed to close radar: {e}")
        if compensator:
            try:
                compensator.close()
            except Exception as e:
                print(f"[Cleanup] Failed to close compensator: {e}")
