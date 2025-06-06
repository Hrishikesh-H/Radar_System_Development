import os
import sys
import time
import threading
import traceback
import datetime
import numpy as np
from collections import deque
from queue import Queue, Empty

import serial
from serial.serialutil import SerialException
from pymavlink import mavutil

# --------------------------------------------------------
# The following imports must remain exactly as they were in your original code.
# Do not alter these lines. They pull in your existing radar-parser, despiker, etc.
from Parser import RadarParser
from Filter import RadarDespiker
# from Plotter import RadarPlotter      # (still not used here, but kept in imports)
# from GUI import DroneLandingStatus   # (still not used here, but kept in imports)
from PortFinder import DevicePortFinder
from PlaneLand import LandingZoneAssessor
from IMUCompensator import AttitudeCompensator

# ============================ DistanceSensorSender ============================

class DistanceSensorSender:
    """
    Sends MAVLink DISTANCE_SENSOR messages at up to 10 Hz.

    Attributes:
        mav              : a pymavlink connection object (master).
        min_range_m      : minimum sensor distance (meters).
        max_range_m      : maximum sensor distance (meters).
        min_distance_cm  : same as min_range_m but converted to cm and clamped to [0,65535].
        max_distance_cm  : same as max_range_m but converted to cm and clamped to [0,65535].
        SENSOR_TYPE      : constant (MAV_DISTANCE_SENSOR_LASER).
        SENSOR_ID        : 0 (arbitrary sensor ID).
        ROTATION_DOWNWARD: indicates sensor facing downward (used in encoding).
        COVARIANCE_UNKNOWN: 255 = unknown measurement covariance.
        send_interval    : minimal time (s) between actual MAVLink packets (0.1 s for 10 Hz).
        _last_send_time  : timestamp of the last send, to enforce rate-limiting.
    """

    def __init__(self, mav, min_range=0.05, max_range=5.0, send_rate_hz=10):
        """
        Initialize the DistanceSensorSender.

        Arguments:
            mav          : a live pymavlink connection (master).
            min_range    : minimum detect distance in meters (float).
            max_range    : maximum detect distance in meters (float).
            send_rate_hz : how many times per second to send distance (int).
        """
        self.mav = mav
        self.min_range_m = min_range
        self.max_range_m = max_range

        # Convert to integer centimeters and clamp to [0..65535]
        self.min_distance_cm = max(0, min(int(self.min_range_m * 100), 65535))
        self.max_distance_cm = max(0, min(int(self.max_range_m * 100), 65535))

        # Constants for encoding the MAVLink message
        self.SENSOR_TYPE = mavutil.mavlink.MAV_DISTANCE_SENSOR_LASER
        self.SENSOR_ID = 0
        self.ROTATION_DOWNWARD = mavutil.mavlink.MAV_SENSOR_ROTATION_PITCH_270
        self.COVARIANCE_UNKNOWN = 255

        # Determine how often we can actually send (enforced in send())
        self.send_interval = 1.0 / send_rate_hz
        self._last_send_time = 0.0

    def send(self, distance_m):
        """
        Attempt to send one DISTANCE_SENSOR MAVLink message, if enough time has passed.
        If distance_m is None or invalid, the message will carry max_distance_cm.

        Steps:
            1) Convert distance_m (float) -> raw_cm (int) -> clamp to [min_distance_cm, max_distance_cm].
            2) Build time_boot_ms from flight controller's SYSTEM_BOOT.
            3) Verify that target_system/target_component are valid (nonzero).
               If target_component is 0, wait up to 0.5 s for a heartbeat to assign it.
            4) Encode a distance_sensor message using mav.distance_sensor_encode(...).
            5) Dump raw bytes (hex) to console for debugging.
            6) Write the raw packet via mav.write(raw).
            7) Attempt a nonblocking recv_match(type='DISTANCE_SENSOR') to see if FC echoes it.
        """
        now = time.time()
        if (now - self._last_send_time) < self.send_interval:
            # Not enough time has elapsed since last send -> skip.
            return

        self._last_send_time = now

        try:
            # ---- 1) Convert to centimeters and clamp ----
            if isinstance(distance_m, (int, float)):
                raw_cm = int(round(distance_m * 100))
                current_cm = max(self.min_distance_cm, min(raw_cm, self.max_distance_cm))
            else:
                # If distance_m is None or not numeric, send out max_distance.
                current_cm = self.max_distance_cm

            ts = datetime.datetime.now().isoformat()
            if distance_m is None:
                dist_str = "None"
            else:
                dist_str = f"{distance_m:.3f}"
            print(f"[{ts}] [DEBUG] Raw distance = {dist_str} m -> {current_cm} cm")
            print(f"[{ts}] [DEBUG] Clamping between {self.min_distance_cm} and {self.max_distance_cm} cm")

            # ---- 2) time_boot_ms ----
            time_boot_ms = int(self.mav.time_since('SYSTEM_BOOT') * 1000)
            if time_boot_ms <= 0:
                print(f"[{ts}] [WARNING] time_boot_ms = {time_boot_ms} (FC time_since('SYSTEM_BOOT') may be off).")

            # ---- 3) Validate target_system/target_component ----
            tsys = getattr(self.mav, 'target_system', None)
            tcomp = getattr(self.mav, 'target_component', None)
            if (tsys is None) or (tcomp is None) or (tsys == 0) or (tcomp == 0):
                print(f"[{ts}] [ERROR] INVALID target_system/component: {tsys}/{tcomp}")
                # If component is still 0, wait up to 0.5 s for heartbeat to set it.
                if tcomp == 0:
                    start_wait = time.time()
                    while (self.mav.target_component == 0) and ((time.time() - start_wait) < 0.5):
                        hb = self.mav.recv_match(type='HEARTBEAT', blocking=True, timeout=0.1)
                        if hb:
                            continue
                    tcomp = self.mav.target_component
                    if tcomp == 0:
                        print(f"[{ts}] [ERROR] Still no valid target_component after waiting. Aborting send().")
                        return

            # ---- 4) Encode the message ----
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
            raw = msg.pack(self.mav.mav)
            hex_dump = raw.hex(' ')
            print(f"[{ts}] [DEBUG] -> RAW MAVLINK DISTANCE_SENSOR: {hex_dump}")

            # ---- 5) Write the packet out ----
            try:
                self.mav.write(raw)
                print(f"[{ts}] [DEBUG] distance_sensor packet written to MAVLink channel")
            except Exception:
                print(f"[{ts}] [ERROR] MAVLink write(raw) raised exception:")
                traceback.print_exc()
                return

            # ---- 6) Try nonblocking echo read ----
            try:
                msg_fb = self.mav.recv_match(type='DISTANCE_SENSOR', blocking=False)
                if msg_fb:
                    print(f"[{ts}] [DEBUG] FC reply: DISTANCE_SENSOR = {msg_fb.current_distance} cm")
                # Otherwise, many firmwares do not echo DISTANCE_SENSOR, so we ignore None.
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
    Polls for HEARTBEAT messages to determine the current flight mode string.
    It sets self.current_mode to either:
        - msg.custom_mode.decode('utf-8'), or
        - "ARMED" / "DISARMED" fallback based on base_mode bitflags.

    This thread runs as a daemon (stops automatically on process exit).
    """

    def __init__(self, mav):
        super().__init__(daemon=True)
        self.mav = mav
        self.current_mode = None
        self._stop_event = threading.Event()

    def run(self):
        """
        Loop:
            1) recv_match(type='HEARTBEAT', blocking=True, timeout=1)
            2) If msg is found, attempt to decode custom_mode as UTF-8; on failure, use base_mode bit.
            3) Print the current_mode for debugging.
            4) Sleep 0.1 s before next iteration to avoid busy-looping.
        """
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
        """ Signal the thread to exit its loop and terminate cleanly. """
        self._stop_event.set()

# ============================ RobustLandingMonitor ============================

class RobustLandingMonitor:
    """
    Evaluates landing-zone flatness over the last buffer_size frames, applies outlier rejection,
    and issues warnings/aborts if the landing zone remains unsafe for too long.

    Key Parameters:
        buffer_size            : how many frames to store in buffers (int).
        slope_threshold_deg    : maximum allowed median slope (degrees) for a frame to be SAFE (float).
        inlier_threshold       : minimum allowed median inlier ratio (float).
        warning_duration       : seconds of continuous UNSAFE needed to abort (float).
        min_consecutive_to_warn: frames of continuous UNSAFE before starting warning timer (int).
        min_consecutive_to_clear: frames of continuous SAFE to clear an active warning/abort (int).

    Internal State:
        slope_buffer        : deque(maxlen=buffer_size) of recent raw slope_deg values.
        inlier_buffer       : deque(maxlen=buffer_size) of recent raw inlier_ratio values.
        safety_history      : deque(maxlen=buffer_size) of recent frame-level bools (True=SAFE, False=UNSAFE).
        consec_unsafe_count : how many UNSAFE frames in a row.
        consec_safe_count   : how many SAFE frames in a row.
        unsafe_start_time   : timestamp when consecutive UNSAFE reached min_consecutive_to_warn.
        abort_issued        : boolean flag indicating abort has been issued.
        currently_warned    : boolean flag indicating we are in warning/abort state (until cleared).
    """

    def __init__(self, mav, distance_sender,
                 buffer_size=20,
                 slope_threshold_deg=5.0,
                 inlier_threshold=0.6,
                 warning_duration=3.0,
                 min_consecutive_to_warn=5,
                 min_consecutive_to_clear=5):
        """
        Initialize the monitor.

        Arguments:
            mav                   : pymavlink connection.
            distance_sender       : DistanceSensorSender instance (used for continuous distance streaming).
            buffer_size           : number of frames to store in buffers (int).
            slope_threshold_deg   : maximum allowed slope for SAFE (float).
            inlier_threshold      : minimum allowed inlier ratio for SAFE (float).
            warning_duration      : seconds to wait (after min_consec_unsafe) before issuing abort (float).
            min_consecutive_to_warn : frames of continuous UNSAFE before starting warning timer (int).
            min_consecutive_to_clear: frames of continuous SAFE to clear any warning/abort (int).
        """
        self.mav = mav
        self.distance_sender = distance_sender

        # Buffers to hold the last buffer_size raw slope/inlier values
        self.slope_buffer = deque(maxlen=buffer_size)
        self.inlier_buffer = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size

        # History of SAFE/UNSAFE decisions (up to buffer_size)
        self.safety_history = deque(maxlen=buffer_size)

        # Thresholds
        self.threshold_slope = slope_threshold_deg
        self.threshold_inlier = inlier_threshold
        self.warning_duration = warning_duration
        self.min_consecutive_to_warn = min_consecutive_to_warn
        self.min_consecutive_to_clear = min_consecutive_to_clear

        # Hysteresis / state variables
        self.unsafe_start_time = None
        self.abort_issued = False
        self.currently_warned = False
        self.consec_unsafe_count = 0
        self.consec_safe_count = 0

    def _filtered_median(self, data_list):
        """
        Compute a robust median of data_list (length = buffer_size).
        1) raw_med = median(data_list)
        2) For each x in data_list, compute abs(x - raw_med) -> MAD items
        3) mad = median(MAD items)
        4) threshold = 2 * mad
        5) filtered = [x for x in data_list if abs(x - raw_med) <= threshold]
        6) If filtered is empty, return raw_med; else return median(filtered).

        Using MAD outlier rejection reduces the effect of occasional outliers.
        """
        arr = np.array(data_list, dtype=float)
        raw_med = float(np.median(arr))
        abs_dev = np.abs(arr - raw_med)
        mad = float(np.median(abs_dev))

        # If MAD is zero (all values identical), just return the raw median.
        if mad == 0:
            return raw_med

        threshold = 2.0 * mad
        filtered = arr[abs_dev <= threshold]
        if len(filtered) == 0:
            return raw_med
        return float(np.median(filtered))

    def update(self, smoothed_det_obj, assessment_result, mode):
        """
        Called every cycle (ideally >=10 Hz). Performs:
          1) Compute distance_m from smoothed_det_obj.
          2) Send distance via distance_sender.send(distance_m).
          3) Read any echo of DISTANCE_SENSOR from FC. If FC reports max_range, send STATUSTEXT ERROR.
          4) If mode contains LAND or RTL, run SAFE vs UNSAFE logic:
             a) If numObj < 3 -> UNSAFE immediately.
             b) Else:
                - Append raw slope_deg & inlier_ratio to buffers.
                - If buffers are full (length == buffer_size):
                    * Compute robust median slope/inlier via _filtered_median(...).
                    * frame_is_safe = (median_slope < threshold_slope) AND (median_inlier > threshold_inlier).
                  Else: frame_is_safe = False (still filling buffers).
             c) Update consec_unsafe_count or consec_safe_count accordingly.
             d) If consec_unsafe_count >= min_consecutive_to_warn and not currently_warned:
                - If unsafe_start_time is None, set it = now.
                - Else if now - unsafe_start_time >= warning_duration, call _issue_abort().
             e) If currently_warned or abort_issued, and consec_safe_count >= min_consecutive_to_clear,
                call _reset_state().
          5) If mode does NOT contain LAND/RTL, call _reset_state() immediately.
        """
        # ---- 1) Compute distance_m ----
        distance_m = None
        if smoothed_det_obj.get('numObj', 0) > 0:
            try:
                distance_m = float(np.min(smoothed_det_obj['z']))
            except Exception:
                distance_m = None

        # ---- 2) Continuous distance streaming (rate-limited inside send()) ----
        self.distance_sender.send(distance_m)

        # ---- 3) Try to read FC echo of DISTANCE_SENSOR ----
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
                    except Exception as e2:
                        print(f"[LandingMonitor][ERROR] Failed to send STATUSTEXT: {e2}")
        except Exception as e:
            print(f"[LandingMonitor][ERROR] Failed to read back DISTANCE_SENSOR: {e}")

        # ---- 4) LAND/RTL logic ----
        if mode is None:
            # If mode is unknown, reset everything
            self._reset_state()
            return

        mode_upper = mode.upper()
        if ("LAND" in mode_upper) or ("RTL" in mode_upper):
            num_points = smoothed_det_obj.get('numObj', 0)
            frame_is_safe = False

            # ---- A) If fewer than 3 points -> UNSAFE ----
            if num_points < 3:
                frame_is_safe = False
                print(f"[LandingMonitor] Frame has only {num_points} point(s) -> UNSAFE")
            else:
                slope_raw = assessment_result.get('slope_deg', None)
                inlier_raw = assessment_result.get('inlier_ratio', None)

                # If either is missing, treat as UNSAFE
                if (slope_raw is None) or (inlier_raw is None):
                    frame_is_safe = False
                    print("[LandingMonitor] Missing slope/inlier -> UNSAFE")
                else:
                    # Append the raw values to buffers
                    self.slope_buffer.append(slope_raw)
                    self.inlier_buffer.append(inlier_raw)

                    # If buffers are full (length == buffer_size), compute robust medians
                    if (len(self.slope_buffer) == self.buffer_size) and (len(self.inlier_buffer) == self.buffer_size):
                        median_slope = self._filtered_median(self.slope_buffer)
                        median_inlier = self._filtered_median(self.inlier_buffer)
                        frame_is_safe = (median_slope < self.threshold_slope) and (median_inlier > self.threshold_inlier)
                        print(f"[LandingMonitor] Median(slope)={median_slope:.2f} deg  Median(inlier)={median_inlier:.2f} -> {'SAFE' if frame_is_safe else 'UNSAFE'}")
                    else:
                        # Not enough frames yet to fill buffers -> UNSAFE
                        frame_is_safe = False
                        print(f"[LandingMonitor] Buffers not full ({len(self.slope_buffer)}/{self.buffer_size}) -> UNSAFE")

            # ---- B) Update consecutive Safe/Unsafe counters ----
            if frame_is_safe:
                self.consec_safe_count += 1
                self.consec_unsafe_count = 0
            else:
                self.consec_unsafe_count += 1
                self.consec_safe_count = 0

            self.safety_history.append(frame_is_safe)

            # ---- C) If enough consecutive UNSAFE -> start or check warning_timer ----
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

            # ---- D) If currently warned/aborted AND enough consecutive SAFE -> clear state ----
            if self.currently_warned or self.abort_issued:
                if self.consec_safe_count >= self.min_consecutive_to_clear:
                    print(f"[LandingMonitor] {self.consec_safe_count} consecutive SAFE frames; clearing warning/abort")
                    self._reset_state()
        else:
            # Not in LAND or RTL -> reset everything
            self._reset_state()

    def _issue_warning(self, text):
        """
        Send STATUSTEXT (severity=WARNING) and print to console.

        Arguments:
            text (str): the warning message text (ASCII only).
        """
        severity = mavutil.mavlink.MAV_SEVERITY_WARNING
        try:
            self.mav.mav.statustext_send(severity, text.encode('utf-8'))
        except Exception as e:
            print(f"[LandingMonitor][ERROR] Unable to send STATUSTEXT: {e}")
        print(f"[LandingMonitor][WARNING] {text}")

    def _issue_abort(self):
        """
        Abort landing by switching flight mode to LOITER. Then send a STATUSTEXT warning.

        1) Try high-level mav.set_mode_loiter().
        2) If that fails, use COMMAND_LONG with MAV_CMD_DO_SET_MODE -> LOITER.
        3) Send a warning STATUSTEXT explaining the abort reason.
        """
        reason = "Landing unsafe: aborting and switching to LOITER. Please relocate to a flat area."
        print("[LandingMonitor] Aborting landing now (switching to LOITER).")
        try:
            # Preferred: high-level set_mode_loiter()
            self.mav.set_mode_loiter()
            self._issue_warning(reason)
        except Exception:
            # Fallback: low-level command_long
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

    def _reset_state(self):
        """
        Clear all buffers, timers, and counters to return to 'no-warning' state.
        """
        self.slope_buffer.clear()
        self.inlier_buffer.clear()
        self.safety_history.clear()
        self.unsafe_start_time = None
        self.abort_issued = False
        self.currently_warned = False
        self.consec_safe_count = 0
        self.consec_unsafe_count = 0

# ============================ RadarHandler ============================

class RadarHandler(threading.Thread):
    """
    Background daemon that:
      1) Attempts to connect to the radar continuously (Linux: /dev/ttyUSB0 & /dev/ttyUSB1).
      2) Once connected, reads frames one by one as soon as available.
      3) Despikes and (optionally) compensates each point cloud.
      4) Runs the static LandingZoneAssessor.assess(...) to get (safe, metrics).
      5) Places (smoothed_det_obj, metrics) into a single-entry queue for the main thread.

    This ensures that the main thread never blocks on radar I/O, and always has at most one 'latest' result.
    """

    def __init__(self, finder, despiker, assessor, queue_out):
        """
        Arguments:
            finder    : DevicePortFinder instance (used to locate radar ports).
            despiker  : RadarDespiker instance.
            assessor  : LandingZoneAssessor instance.
            queue_out : Queue(maxsize=1) where (smoothed_det_obj, metrics) is posted.
        """
        super().__init__(daemon=True)
        self.finder = finder
        self.despiker = despiker
        self.assessor = assessor
        self.queue_out = queue_out
        self._stop_event = threading.Event()

    def run(self):
        """
        Loop forever (until stop() is called):
          1) If radar is missing or stale, attempt to reconnect:
             - For Linux, fix cli="/dev/ttyUSB0", data="/dev/ttyUSB1".
             - Instantiate RadarParser(cli, data, config_path, debug=True, enable_logging=False).
             - Call initialize_ports() + send_config(), then sleep 2 s to let streaming begin.
          2) If radar is connected, call radar.read_frame() once:
             - If it times out or errors, close radar and go back to step 1.
             - If header is None, continue (no full frame yet).
             - Otherwise, we have det_obj, snr, noise -> perform despiking + compensation.
               * Build smoothed_det_obj = {'x':..., 'y':..., 'z':..., 'numObj': int}.
          3) Run safe, m = assessor.assess(det_obj).
          4) Remove any existing item from queue_out (if present), then put the new (smoothed_det_obj, m).
        """
        global radar, cli, data, last_radar_time, compensator, calli_req
        RECONNECT_INTERVAL = 3.0  # seconds between reconnect attempts

        while not self._stop_event.is_set():
            now = time.time()
            # ---- Step 1: Check if radar is missing or stale ----
            if (radar is None) or (not getattr(radar, 'data_serial', None)) or (not radar.data_serial.is_open) or ((now - last_radar_time) > 5.0):
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
                    # For Linux, use fixed device names known to work:
                    cli = "/dev/ttyUSB0"
                    data = "/dev/ttyUSB1"
                    print(f"[{datetime.datetime.now().isoformat()}] [Radar] Using CLI={cli}, DATA={data}")
                    config_path = os.path.join(os.getcwd(), "best_res_4cm.cfg")

                    # Instantiate RadarParser exactly as before
                    radar = RadarParser(cli, data, config_path, debug=True, enable_logging=False, log_prefix="radar_log")
                    radar.initialize_ports()
                    radar.send_config()

                    # Sleep a moment so the radar has time to startup and stream at its own cadence
                    time.sleep(2.0)
                    radar.info_print("Connected to radar successfully (Linux).")
                    last_radar_time = time.time()
                except Exception as e:
                    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
                    print(f"[Radar][Connection Error] {e.__class__.__name__}: {e} | Time: {current_time}")
                    time.sleep(RECONNECT_INTERVAL)
                    continue
            else:
                # ---- Step 2: radar is connected and fresh; read exactly one frame ----
                try:
                    start = time.time()
                    header, det_obj, snr, noise = radar.read_frame()
                    elapsed = time.time() - start

                    # If read_frame spent too long, treat as error
                    if elapsed > 10.0:
                        raise RuntimeError(f"read_frame took {elapsed:.2f}s, exceeding threshold")

                    if header is None:
                        # Incomplete data; loop again
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

                # ---- Step 3: We have a valid header+det_obj+snr+noise, so process it ----
                smoothed_det_obj = {'x': [], 'y': [], 'z': [], 'numObj': 0}

                # If there are >= 3 points, despike and optionally compensate
                if (det_obj is not None) and (det_obj.get('numObj', 0) >= 3):
                    spr = self.despiker.process(det_obj, snr, noise)
                    pts_body = np.vstack((spr['x'], spr['y'], spr['z'])).T

                    if compensator:
                        try:
                            # Only calibrate once (calli_req flips to False afterward)
                            if calli_req:
                                compensator.internal_calibrate_offsets(num_samples=100, delay=0.01)
                                calli_req = False
                            pts_enu = compensator.transform_pointcloud(pts_body)
                            last_radar_time = time.time()
                        except Exception as e:
                            print(f"[Autopilot][Compensation Error] {e.__class__.__name__}: {e}. Using raw points.")
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
                        'x': pts_enu[:, 0].tolist(),
                        'y': pts_enu[:, 1].tolist(),
                        'z': pts_enu[:, 2].tolist(),
                        'numObj': det_obj['numObj']
                    }

                # ---- Step 4: Run landing-zone assessor on raw det_obj ----
                # Note: if det_obj is None or numObj < 3, assessor.assess(...) returns (False, {'reason':'...'}).
                safe, m = self.assessor.assess(det_obj)

                # ---- Step 5: Publish the latest smoothed_det_obj + metrics to the queue ----
                try:
                    # If queue already has an item, remove it so we only keep the LATEST result.
                    _ = self.queue_out.get_nowait()
                except Empty:
                    pass
                self.queue_out.put((smoothed_det_obj, m))

        # End of while loop

    def stop(self):
        """Signal the RadarHandler thread to exit its loop and terminate cleanly."""
        self._stop_event.set()

# ============================ AutopilotHandler ============================

class AutopilotHandler(threading.Thread):
    """
    Background daemon that:
      1) Attempts to establish a MAVLink connection continuously.
      2) Once connected, spawns:
         - ModeWatcher (daemon thread)
         - DistanceSensorSender
         - RobustLandingMonitor
      3) After that, it simply sleeps/retries if the connection drops.
    """

    def __init__(self, finder):
        """
        finder: DevicePortFinder instance (used to locate autopilot connection).
        """
        super().__init__(daemon=True)
        self.finder = finder
        self._stop_event = threading.Event()

    def run(self):
        """
        Loop until stop() is called:
          1) If compensator is None, attempt to connect:
             a) Try UDP-only via finder.find_autopilot_connection(timeout=2.0, exclude_ports=[]).
             b) If UDP fails and radar ports (cli, data) are known, try excluding them.
             c) If master is found, instantiate AttitudeCompensator(master) -> compensator.
          2) As soon as compensator is set:
             a) Spawn ModeWatcher(master).
             b) Create DistanceSensorSender(master, ...).
             c) Create RobustLandingMonitor(master, distance_sender, ...).
          3) Sleep 5 s between connection attempts or checks.
        """
        global compensator, last_att_time, mode_watcher, distance_sender, landing_monitor, cli, data

        reconnect_interval = 5.0
        while not self._stop_event.is_set():
            if compensator is None:
                try:
                    # Attempt UDP-only connection first
                    master, ap_name = None, None
                    try:
                        master, ap_name = self.finder.find_autopilot_connection(
                            timeout=2.0,
                            exclude_ports=[]
                        )
                    except RuntimeError:
                        master = None

                    # If UDP fails, exclude known radar ports from serial search
                    if (master is None) and cli and data:
                        master, ap_name = self.finder.find_autopilot_connection(
                            timeout=2.0,
                            exclude_ports=[cli, data]
                        )

                    if master is not None:
                        compensator = AttitudeCompensator(master)
                        print("[Autopilot] MAVLink connection established for compensation (background).")
                        last_att_time = time.time()

                        # Spawn ModeWatcher lazily (only once)
                        if mode_watcher is None:
                            mode_watcher = ModeWatcher(master)
                            mode_watcher.start()

                            # Instantiate DistanceSensorSender (rate-limited at 10 Hz)
                            distance_sender = DistanceSensorSender(
                                master,
                                min_range=0.05,
                                max_range=5.0,
                                send_rate_hz=10
                            )

                            # Instantiate RobustLandingMonitor (exact thresholds)
                            landing_monitor = RobustLandingMonitor(
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
                    print(f"[Autopilot] Background reconnect failed: {e}. Retrying in {reconnect_interval:.1f}s")
                    traceback.print_exc()
                    time.sleep(reconnect_interval)
            else:
                # Already connected -> just sleep and periodically check if still alive
                time.sleep(reconnect_interval)

    def stop(self):
        """Signal the AutopilotHandler thread to exit its loop and terminate cleanly."""
        self._stop_event.set()

# ============================ Global State & Entry Point ============================

# Shared global references (managed by AutopilotHandler / RadarHandler threads)
compensator = None
last_att_time = 0
calli_req = True  # calibrate AttitudeCompensator only once

radar = None
last_radar_time = 0
cli = None
data = None

mode_watcher = None
distance_sender = None
landing_monitor = None

ENABLE_TRACE = False  # If True, print full exception tracebacks in certain places

def maybe_traceback():
    """If ENABLE_TRACE is True, print the current exception traceback."""
    if ENABLE_TRACE:
        traceback.print_exc()

def main():
    """
    1) Create a Queue(maxsize=1) for RadarHandler -> main thread communication.
    2) Start RadarHandler (daemon) with that queue.
    3) Start AutopilotHandler (daemon).
    4) Enter a 50 Hz loop; each iteration:
       a) Pull the latest (smoothed_det_obj, metrics) from the queue if available; else use empty frame.
       b) If landing_monitor is ready, call landing_monitor.update(smoothed_det_obj, metrics, current_mode).
       c) Sleep as needed to maintain ~50 Hz.
    5) On KeyboardInterrupt or error, signal all threads to stop, join them, and close resources.
    """
    global radar, last_radar_time, cli, data

    # A size-1 queue so RadarHandler -> main thread can post the latest (smoothed_det_obj, metrics).
    # If RadarHandler tries to put twice before main thread consumes, the old item is discarded.
    radar_queue = Queue(maxsize=1)

    finder = DevicePortFinder()
    despiker = RadarDespiker()
    assessor = LandingZoneAssessor(0.05, 500, 10, 0.7)

    print(f"[{datetime.datetime.now().isoformat()}] [MAIN] Starting up. Launching RadarHandler + AutopilotHandler.")

    # 1) Launch RadarHandler as a daemon thread
    radar_handler = RadarHandler(finder, despiker, assessor, radar_queue)
    radar_handler.start()

    # 2) Launch AutopilotHandler as a daemon thread
    autopilot_handler = AutopilotHandler(finder)
    autopilot_handler.start()

    try:
        last_update = time.time()
        while True:
            now = time.time()

            # Run this loop at ~50 Hz (period = 0.02 s)
            if (now - last_update) < 0.02:
                time.sleep(0.005)
                continue
            last_update = now

            # 3) Pull the latest radar result if available; otherwise, empty frame
            try:
                smoothed_det_obj, m = radar_queue.get_nowait()
            except Empty:
                smoothed_det_obj = {'x': [], 'y': [], 'z': [], 'numObj': 0}
                m = {'reason': 'no data'}

            # 4) If landing_monitor is ready, evaluate/update it
            if landing_monitor:
                current_mode = mode_watcher.current_mode if mode_watcher else None
                landing_monitor.update(smoothed_det_obj, m, current_mode)
            # If landing_monitor is not yet instantiated (autopilot not connected), we simply skip.

    except KeyboardInterrupt:
        print(f"[{datetime.datetime.now().isoformat()}] [System] Interrupted by user. Exiting...")
    except Exception as e:
        print(f"[{datetime.datetime.now().isoformat()}] [System] Unexpected: {e.__class__.__name__}: {e}")
        traceback.print_exc()
        time.sleep(1)
    finally:
        # 5) Signal threads to stop and join them
        autopilot_handler.stop()
        radar_handler.stop()

        autopilot_handler.join(timeout=1.0)
        radar_handler.join(timeout=1.0)

        if mode_watcher:
            mode_watcher.stop()
            mode_watcher.join(timeout=1.0)

        # Close radar if still open
        if radar:
            try:
                radar.close()
            except Exception as e:
                print(f"[{datetime.datetime.now().isoformat()}] [Cleanup] Failed to close radar: {e}")

        # Close compensator if still open
        if compensator:
            try:
                compensator.close()
            except Exception as e:
                print(f"[{datetime.datetime.now().isoformat()}] [Cleanup] Failed to close compensator: {e}")

if __name__ == "__main__":
    main()
