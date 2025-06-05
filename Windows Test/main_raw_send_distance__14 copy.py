import os
import sys
import time
import serial
import threading
import traceback
import datetime
import numpy as np
from collections import deque
from pymavlink import mavutil
from serial.serialutil import SerialException

from Parser import RadarParser
from Filter import RadarDespiker
# from Plotter import RadarPlotter
# from GUI import DroneLandingStatus
from PortFinder import DevicePortFinder
from PlaneLand import LandingZoneAssessor
from IMUCompensator import AttitudeCompensator


# ============================ DistanceSensorSender ============================

class DistanceSensorSender:
    """
    Handles sending MAVLink DISTANCE_SENSOR messages at a fixed rate (10 Hz),
    using MAVLink 1.0 framing, with console logging, error reporting, and traceback on failure.
    """
    def __init__(self, mav, min_range=0.05, max_range=5.0, send_rate_hz=10):
        """
        mav: an active pymavlink connection (master)
        min_range, max_range: sensor bounds in meters
        send_rate_hz: frequency at which to send DISTANCE_SENSOR messages
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

        self.send_interval = 1.0 / send_rate_hz
        self._last_send_time = 0.0

    def send(self, distance_m):
        """
        Send a DISTANCE_SENSOR message at up to the configured rate.
        Uses the helper .distance_sensor_send() to guarantee correct MAVLink 1.0 formatting.
        Prints traceback if send fails, and dumps raw bytes before sending.
        """
        now = time.time()
        if (now - self._last_send_time) < self.send_interval:
            return  # skip: enforcing 10 Hz

        self._last_send_time = now

        try:
            # 1) Compute raw and clamped centimeters
            if distance_m is not None and isinstance(distance_m, (int, float)):
                raw_cm = int(round(distance_m * 100))
                clamped_cm = max(self.min_distance_cm, min(raw_cm, self.max_distance_cm))
                current_cm = clamped_cm
            else:
                current_cm = self.max_distance_cm

            # 2) Build timestamp for debug messages
            ts = datetime.datetime.now().isoformat()

            # Safe formatting for distance_m (None or float)
            if distance_m is None:
                distance_str = "None"
            else:
                distance_str = f"{distance_m:.3f}"
            print(f"[{ts}] [DEBUG] Raw distance = {distance_str} m -> {current_cm} cm")
            print(f"[{ts}] [DEBUG] Clamping distance between {self.min_distance_cm} and {self.max_distance_cm} cm")

            # 3) Build time_boot_ms
            time_boot_ms = int(self.mav.time_since('SYSTEM_BOOT') * 1000)
            if time_boot_ms <= 0:
                print(f"[{ts}] [WARNING] time_boot_ms = {time_boot_ms} (check FC time_since('SYSTEM_BOOT')).")

            # 4) Confirm target_system and target_component
            tsys = getattr(self.mav, 'target_system', None)
            tcomp = getattr(self.mav, 'target_component', None)
            if (tsys is None) or (tcomp is None) or (tsys == 0) or (tcomp == 0):
                print(f"[{ts}] [ERROR] INVALID target_system/component: {tsys}/{tcomp}")
                # If tcomp == 0, wait up to 0.5 seconds for a heartbeat
                if tcomp == 0:
                    start_hb_wait = time.time()
                    while (self.mav.target_component == 0) and (time.time() - start_hb_wait < 0.5):
                        hb = self.mav.recv_match(type='HEARTBEAT', blocking=True, timeout=0.1)
                        if hb:
                            continue
                    tcomp = self.mav.target_component
                    if tcomp == 0:
                        print(f"[{ts}] [ERROR] Still no valid target_component after waiting. Aborting send().")
                        return

            # 5) Encode the message so we can show raw bytes
            msg = self.mav.mav.distance_sensor_encode(
                time_boot_ms,
                self.min_distance_cm,
                self.max_distance_cm,
                current_cm,
                self.SENSOR_TYPE,
                self.SENSOR_ID,
                self.ROTATION_DOWNWARD,
                self.COVARIANCE_UNKNOWN
            )
            raw = msg.pack(self.mav.mav)    # Byte-string to send
            hex_dump = raw.hex(' ')
            print(f"[{ts}] [DEBUG] -> RAW MAVLINK DISTANCE_SENSOR: {hex_dump}")

            # Now actually write it
            try:
                self.mav.write(raw)
                print(f"[{ts}] [DEBUG] distance_sensor packet written to MAVLink channel")
            except Exception as write_exc:
                print(f"[{ts}] [ERROR] MAVLink write(raw) raised an exception:")
                traceback.print_exc()
                return

            # 6) Optional feedback: use non-blocking recv_match to avoid SerialException
            try:
                msg_fb = self.mav.recv_match(type='DISTANCE_SENSOR', blocking=False)
                if msg_fb:
                    print(f"[{ts}] [DEBUG] FC reply: DISTANCE_SENSOR = {msg_fb.current_distance} cm")
                else:
                    # On many firmwares, DISTANCE_SENSOR is not echoed back
                    pass
            except SerialException as ser_e:
                print(f"[{ts}] [ERROR] SerialException in non-blocking recv_match: {ser_e}")
            except Exception as e:
                print(f"[{ts}] [ERROR] Unexpected exception in recv_match: {e}")
                traceback.print_exc()

        except Exception as e:
            ts = datetime.datetime.now().isoformat()
            print(f"[{ts}] [ERROR] send() failed: {e}\n{traceback.format_exc()}")


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
                    if base & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED:
                        self.current_mode = "ARMED"
                    else:
                        self.current_mode = "DISARMED"
                print(f"[ModeWatcher] Current mode: {self.current_mode}")
            time.sleep(0.1)

    def stop(self):
        self._stop_event.set()


# ============================ LandingMonitor ============================

class LandingMonitor:
    """
    Evaluates landing surface flatness over recent frames and issues warnings/aborts if unsafe.
    - Buffers recent slope and inlier_ratio values (for median calculation).
    - Also buffers a history of frame-by-frame safety decisions (booleans) for hysteresis.
    - If fewer than 3 points are detected in a frame, that frame is automatically classified as UNSAFE.
    - When in LAND or RTL mode, uses median slope/inlier over a buffer to decide SAFE/UNSAFE.
    - Requires a certain number of consecutive UNSAFE frames before starting a warning timer.
    - If UNSAFE persists beyond warning_duration, aborts landing (switch to LOITER).
    - Requires a certain number of consecutive SAFE frames to clear an active warning/abort.
    """
    def __init__(self, mav, distance_sender,
                 buffer_size=20,
                 slope_threshold_deg=5.0,
                 inlier_threshold=0.6,
                 warning_duration=3.0,
                 min_consecutive_to_warn=5,
                 min_consecutive_to_clear=5):
        """
        mav: pymavlink connection
        distance_sender: DistanceSensorSender instance
        buffer_size: maxlen for slope and inlier buffering for median
        slope_threshold_deg: maximum median slope (degrees) to treat frame as SAFE
        inlier_threshold: minimum median inlier_ratio to treat frame as SAFE
        warning_duration: seconds UNSAFE must persist before aborting landing
        min_consecutive_to_warn: number of consecutive UNSAFE frames before starting warning timer
        min_consecutive_to_clear: number of consecutive SAFE frames to clear an active warning
        """
        self.mav = mav
        self.distance_sender = distance_sender

        # Buffers for raw slopes and inliers (for median calculation)
        self.slope_buffer = deque(maxlen=buffer_size)
        self.inlier_buffer = deque(maxlen=buffer_size)

        # History of frame safety decisions: True = SAFE, False = UNSAFE
        self.safety_history = deque(maxlen=buffer_size)

        self.threshold_slope = slope_threshold_deg
        self.threshold_inlier = inlier_threshold
        self.warning_duration = warning_duration

        # Hysteresis: how many consecutive UNSAFE to begin warning timer,
        # and how many consecutive SAFE to clear an active warning/abort.
        self.min_consecutive_to_warn = min_consecutive_to_warn
        self.min_consecutive_to_clear = min_consecutive_to_clear

        # State variables
        self.unsafe_start_time = None     # timestamp when continuous UNSAFE streak began
        self.abort_issued = False         # whether abort (LOITER) has been sent
        self.currently_warned = False     # whether currently in warning/abort state
        self.consec_unsafe_count = 0      # how many frames in a row have been UNSAFE
        self.consec_safe_count = 0        # how many frames in a row have been SAFE

    def update(self, smoothed_det_obj, assessment_result, mode):
        """
        Called each cycle with:
        - smoothed_det_obj: dict with keys 'x', 'y', 'z', 'numObj'
        - assessment_result: dict returned by LandingZoneAssessor.assess()
        - mode: current flight mode string
        Behavior:
         1. Compute distance from smoothed_det_obj (min z).
         2. Send distance to FC via distance_sender.
         3. Read back any DISTANCE_SENSOR from FC; if equals max_range, send error STATUSTEXT.
         4. If mode in LAND or RTL:
             a. If <3 points: frame is UNSAFE.
             b. Else: buffer slope and inlier. Once buffers are full, compute median.
                If median_slope < threshold and median_inlier > threshold -> SAFE; else UNSAFE.
             c. Update counters: consec_unsafe_count or consec_safe_count.
             d. If consec_unsafe_count >= min_consecutive_to_warn, start warning timer.
                If warning_duration elapsed, abort landing (LOITER).
             e. If currently warned or aborted, require consec_safe_count >= min_consecutive_to_clear to clear state.
         5. If mode not LAND/RTL: reset all timers/counters.
        """
        # 1. Compute distance (in meters) as minimum z value if available
        distance_m = None
        if smoothed_det_obj.get('numObj', 0) > 0:
            try:
                distance_m = float(np.min(smoothed_det_obj['z']))
            except Exception:
                distance_m = None

        # 2. Send distance sensor MAVLink every call (rate-limited internally)
        self.distance_sender.send(distance_m)

        # 3. Read back any DISTANCE_SENSOR from FC and print it; if it equals max_range, report error
        try:
            msg_fb = self.mav.recv_match(type='DISTANCE_SENSOR', blocking=False)
            if msg_fb:
                reported_cm = msg_fb.current_distance
                print(f"[LandingMonitor] FC reports distance: {reported_cm} cm")
                if reported_cm >= self.distance_sender.max_distance_cm:
                    err_text = "FC distance sensor error: no data."
                    print(f"[LandingMonitor][ERROR] {err_text}")
                    try:
                        severity = mavutil.mavlink.MAV_SEVERITY_ERROR
                        self.mav.mav.statustext_send(severity, err_text.encode('utf-8'))
                    except Exception as e:
                        print(f"[LandingMonitor][ERROR] Failed to send STATUSTEXT: {e}")
        except Exception as e:
            print(f"[LandingMonitor][ERROR] Failed to read back DISTANCE_SENSOR: {e}")

        # 4. Landing/RTL specific logic
        if mode is None:
            # If mode is unknown, reset everything
            self._reset_state()
            return

        mode_upper = mode.upper()

        # Only run landing checks if in LAND or RTL
        if ("LAND" in mode_upper) or ("RTL" in mode_upper):
            # STEP A: Classify current frame as SAFE or UNSAFE
            num_points = smoothed_det_obj.get('numObj', 0)
            frame_is_safe = False

            # If fewer than 3 points, classify as UNSAFE
            if num_points < 3:
                frame_is_safe = False
                print(f"[LandingMonitor] Frame has only {num_points} point(s) -> classifying as UNSAFE")
            else:
                slope_raw = assessment_result.get('slope_deg', None)
                inlier_raw = assessment_result.get('inlier_ratio', None)
                if (slope_raw is None) or (inlier_raw is None):
                    # Missing data -> UNSAFE
                    frame_is_safe = False
                    print("[LandingMonitor] Missing slope/inlier -> classifying as UNSAFE")
                else:
                    self.slope_buffer.append(slope_raw)
                    self.inlier_buffer.append(inlier_raw)

                    # If buffers are full, compute median
                    if (len(self.slope_buffer) == self.slope_buffer.maxlen) and \
                       (len(self.inlier_buffer) == self.inlier_buffer.maxlen):
                        median_slope = float(np.median(np.array(self.slope_buffer)))
                        median_inlier = float(np.median(np.array(self.inlier_buffer)))
                        frame_is_safe = (median_slope < self.threshold_slope) and (median_inlier > self.threshold_inlier)
                        print(f"[LandingMonitor] Median slope = {median_slope:.2f}°, "
                              f"Median inlier = {median_inlier:.2f} -> "
                              f"{'SAFE' if frame_is_safe else 'UNSAFE'}")
                    else:
                        # Not enough data yet -> UNSAFE
                        frame_is_safe = False
                        print(f"[LandingMonitor] Buffers not full (size={len(self.slope_buffer)}) -> classifying as UNSAFE")

            # STEP B: Update consecutive unsafe/safe counters
            if frame_is_safe:
                self.consec_safe_count += 1
                self.consec_unsafe_count = 0
            else:
                self.consec_unsafe_count += 1
                self.consec_safe_count = 0

            # Append to safety_history for debugging if needed
            self.safety_history.append(frame_is_safe)

            # STEP C: If enough consecutive UNSAFE frames, possibly start warning timer
            if (self.consec_unsafe_count >= self.min_consecutive_to_warn) and (not self.currently_warned):
                if self.unsafe_start_time is None:
                    self.unsafe_start_time = time.time()
                    print(f"[LandingMonitor] {self.consec_unsafe_count} consecutive UNSAFE frames; starting warning timer at t={self.unsafe_start_time:.2f}")
                else:
                    elapsed = time.time() - self.unsafe_start_time
                    if elapsed >= self.warning_duration:
                        if not self.abort_issued:
                            self._issue_abort()
                            self.abort_issued = True
            # else: either not enough consec UNSAFE or already warned/aborted

            # STEP D: If currently warned or aborted, check for consecutive SAFE frames to clear
            if self.currently_warned or self.abort_issued:
                if self.consec_safe_count >= self.min_consecutive_to_clear:
                    print(f"[LandingMonitor] {self.consec_safe_count} consecutive SAFE frames; clearing warning/abort")
                    self._reset_state()
        else:
            # Not in LAND or RTL: reset everything
            self._reset_state()

    def _issue_warning(self, text):
        """
        Send MAVLink STATUSTEXT with severity WARNING (3), and print to console.
        """
        severity = mavutil.mavlink.MAV_SEVERITY_WARNING
        try:
            self.mav.mav.statustext_send(severity, text.encode('utf-8'))
        except Exception as e:
            print(f"[LandingMonitor][ERROR] Unable to send STATUSTEXT: {e}")
        print(f"[LandingMonitor][WARNING] {text}")

    def _issue_abort(self):
        """
        Abort landing by changing mode to LOITER, and send a STATUSTEXT explaining the reason.
        """
        reason = "Landing unsafe: aborting and switching to LOITER. Please relocate to a flat area."
        print(f"[LandingMonitor] Aborting landing now (switching to LOITER).")
        # First try high-level set_mode_loiter()
        try:
            self.mav.set_mode_loiter()
            self._issue_warning(reason)
        except Exception:
            # Fallback to COMMAND_LONG if set_mode_loiter isn't available
            try:
                DO_SET_MODE = mavutil.mavlink.MAV_CMD_DO_SET_MODE
                MAV_MODE_FLAG_AUTO_ENABLED = mavutil.mavlink.MAV_MODE_FLAG_AUTO_ENABLED
                custom_mode_loiter = 2  # ArduPilot LOITER mode index
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
            except Exception as e2:
                print(f"[LandingMonitor][ERROR] Failed to switch to LOITER: {e2}")
        self.abort_issued = True
        self.currently_warned = True

    def send_warning(self, text):
        """
        External method to send STATUSTEXT WARNING immediately.
        """
        self._issue_warning(text)

    def _reset_state(self):
        """
        Reset all hysteresis and timer states.
        """
        self.slope_buffer.clear()
        self.inlier_buffer.clear()
        self.safety_history.clear()
        self.unsafe_start_time = None
        self.abort_issued = False
        self.currently_warned = False
        self.consec_safe_count = 0
        self.consec_unsafe_count = 0


# ========================= END New Classes ==========================


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

                        # DistanceSensorSender (10 Hz)
                        distance_sender = DistanceSensorSender(master, min_range=0.05, max_range=5.0, send_rate_hz=10)

                        # LandingMonitor with robust logic
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

    despiker = RadarDespiker()
    assessor = LandingZoneAssessor(0.05, 500, 10, 0.7)

    print(f"[{datetime.datetime.now().isoformat()}] [MAIN] Starting up, attempting to connect to Radar.")

    last_radar_time = 0
    RECONNECT_INTERVAL = 3.0
    READ_TIMEOUT_THRESHOLD = 7.0  # if read_frame takes > 3 s, abort

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
                        print(f"[{datetime.datetime.now().isoformat()}] [Radar] Closed existing radar instance.")
                    except Exception as e:
                        print(f"[{datetime.datetime.now().isoformat()}] [Radar] Error during close: {e}")
                    radar = None
                    print(f"[{datetime.datetime.now().isoformat()}] [Radar] Restarting radar connection.")

                print(f"[{datetime.datetime.now().isoformat()}] [Radar] Attempting to reconnect...")
                try:
                    cli, data = finder.find_radar_ports_by_description()
                    print(f"[{datetime.datetime.now().isoformat()}] [Radar] Ports found: CLI={cli}, DATA={data}")
                    config_path = os.path.join(os.getcwd(), "best_res_4cm.cfg")
                    radar = RadarParser(cli, data, config_path, debug=True, enable_logging=False, log_prefix="radar_log")
                    radar.initialize_ports()
                    radar.send_config()
                    print(f"[{datetime.datetime.now().isoformat()}] [Radar] Sleeping 3 seconds to allow radar to initialize and stream data.")
                    time.sleep(3)
                    radar.info_print("Connected to radar successfully (Linux).")
                    last_radar_time = time.time()
                except RuntimeError as e:
                    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
                    print(f"[Radar][Port Detection Error] {e.__class__.__name__}: {e} | Time: {current_time}")
                    time.sleep(RECONNECT_INTERVAL)
                    continue
                except Exception as e:
                    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
                    print(f"[Radar][Unknown Connection Error] {e.__class__.__name__}: {e} | Time: {current_time}")
                    time.sleep(RECONNECT_INTERVAL)
                    continue

            # --- RADAR DATA ACQUISITION ---
            try:
                start_read = time.time()
                header, det_obj, snr, noise = radar.read_frame()
                elapsed = time.time() - start_read

                if elapsed > READ_TIMEOUT_THRESHOLD:
                    raise RuntimeError(f"read_frame took {elapsed:.2f}s, exceeding threshold")

                if header is None:
                    # No complete frame yet; loop back
                    continue

                last_radar_time = time.time()
            except (serial.SerialException, SerialException, RuntimeError) as e:
                print(f"[{datetime.datetime.now().isoformat()}] [Radar][Serial/Timeout Error] {e.__class__.__name__}: {e}")
                try:
                    radar.close()
                    print(f"[{datetime.datetime.now().isoformat()}] [Radar] Closed radar after error.")
                except Exception as ce:
                    print(f"[{datetime.datetime.now().isoformat()}] [Radar] Close error: {ce}")
                radar = None
                continue
            except Exception as e:
                print(f"[{datetime.datetime.now().isoformat()}] [Radar][Frame Read Error] {e.__class__.__name__}: {e}")
                try:
                    radar.close()
                    print(f"[{datetime.datetime.now().isoformat()}] [Radar] Closed radar after general failure.")
                except Exception as ce:
                    print(f"[{datetime.datetime.now().isoformat()}] [Radar] Close error: {ce}")
                radar = None
                continue

            # At this point, we have a valid header + det_obj + snr + noise
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

            # --- NEW: DISTANCE SENDING and LANDING MONITORING ---
            if mode_watcher and landing_monitor:
                current_mode = mode_watcher.current_mode
                # This call is invoked each iteration, but send() enforces 10 Hz
                landing_monitor.update(smoothed_det_obj, m, current_mode)

            # gui.update_status(safe, m)
            # app.processEvents()

    except KeyboardInterrupt:
        print(f"[{datetime.datetime.now().isoformat()}] [System] Interrupted by user. Exiting...")
    except Exception as e:
        print(f"[{datetime.datetime.now().isoformat()}] [System] Unexpected: {e.__class__.__name__}: {e}")
        traceback.print_exc()
        time.sleep(1)
    finally:
        _autopilot_stop = True
        if mode_watcher:
            mode_watcher.stop()
        _thread.join(timeout=1.0)

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
