# -*- coding: utf-8 -*-

"""
Radar + Autopilot Safety System
-------------------------------
- Radar frames are processed to assess landing zone safety.
- A separate process handles radar connection and data retrieval.
- Main process evaluates radar frames for landing safety and sends MAVLink distance messages.

Core Features:
- Keeps radar connection alive in an isolated process.
- Uses multiprocessing queues and events correctly (no Manager-pickled errors).
- Attitude compensation and landing evaluation logic untouched.
- Optimized for low power CPUs with ProcessPoolExecutor.
"""

import os
import sys
import time
import logging
import threading
import queue
from statistics import median
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Process, Event as MpEvent, Queue
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

# ------------------ Radar Loop (Worker Process) ------------------
def radar_loop(finder, despiker, assessor, frame_queue, stop_event, cli, data, cfg_path, shared):
    """ Continuously read radar frames and put them into frame_queue. """
    while not stop_event.is_set():
        try:
            radar = shared.get('radar')
            if radar is None or not getattr(radar, 'data_serial', None) or not radar.data_serial.is_open:
                if radar:
                    try:
                        radar.close()
                        logger.info("Radar connection closed")
                    except Exception:
                        logger.exception("Error closing radar")
                try:
                    radar = RadarParser(cli, data, cfg_path, debug=False, enable_logging=False)
                    radar.initialize_ports()
                    radar.send_config()
                    shared['radar'] = radar
                    logger.info("Radar connected")
                except Exception:
                    logger.exception("Radar connection failed")
                    shared['radar'] = None
                    time.sleep(1.0)
                continue

            header, det_obj, snr, noise = radar.read_frame()
            frame_queue.put((header, det_obj, snr, noise), block=True, timeout=1)
            shared['last_radar_time'] = time.time()

        except queue.Full:
            logger.warning("Frame queue full, dropping frame")
        except Exception:
            logger.exception("Radar loop error")
            shared['radar'] = None
            time.sleep(1.0)
# ------------------ Sensor Sender ------------------
class DistanceSensorSender:
    """ Sends MAVLink DISTANCE_SENSOR messages at fixed rate. """
    def __init__(self, mav, min_range=0.05, max_range=5.0, rate_hz=10):
        self.mav = mav
        self.min_cm = max(0, min(int(min_range * 100), 65535))
        self.max_cm = max(0, min(int(max_range * 100), 65535))
        self.interval = 1.0 / rate_hz
        self._last_time = time.time()

    def send(self, distance_m):
        now = time.time()
        if now - self._last_time < self.interval:
            return  # Enforce rate limit
        self._last_time = now
        raw_cm = self.max_cm
        if isinstance(distance_m, (int, float)):
            raw_cm = int(round(distance_m * 100))
        dist_cm = max(self.min_cm, min(raw_cm, self.max_cm))
        boot_ms = int(self.mav.time_since('SYSTEM_BOOT') * 1000)
        msg = self.mav.mav.distance_sensor_encode(
            boot_ms,
            self.min_cm,
            self.max_cm,
            dist_cm,
            mavutil.mavlink.MAV_DISTANCE_SENSOR_LASER,
            0,
            mavutil.mavlink.MAV_SENSOR_ROTATION_PITCH_270,
            255
        )
        self.mav.write(msg.pack(self.mav.mav))

# ------------------ Mode Watcher ------------------
def mode_watcher(mav, mode_holder, stop_event):
    """ Listens for MAVLink HEARTBEAT and updates current mode. """
    while not stop_event.is_set():
        try:
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
    """ Monitors radar-evaluated terrain and aborts landing if unsafe. """
    def __init__(self, mav, distance_sender, **cfg):
        self.mav = mav
        self.distance_sender = distance_sender
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
        # Send latest altitude (radar-based) to FCU
        dist = float(min(smoothed['z'])) if smoothed.get('numObj', 0) > 0 else None
        self.distance_sender.send(dist)

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

# ------------------ Main Entry ------------------
def main():
    finder = DevicePortFinder()
    despiker = RadarDespiker()
    assessor = LandingZoneAssessor(0.05, 500, 10, 0.7)

    frame_queue = Queue(maxsize=100)  # Shared between radar and main
    stop_radar = MpEvent()

    shared = {
        'cli': '/dev/ttyUSB0',
        'data': '/dev/ttyUSB1',
        'cfg_path': os.path.join(os.getcwd(), 'best_res_4cm.cfg'),
        'master': None,
        'radar': None,
        'compensator': None,
        'distance_sender': None,
        'landing_monitor': None,
        'mode_holder': {'mode': None},
        'mode_watcher_started': False,
        'last_radar_time': time.time(),
        'finder': finder,
        'despiker': despiker,
        'assessor': assessor
    }

    cpu_executor = ProcessPoolExecutor(max_workers=2)

    radar_proc = Process(target=radar_loop, args=(
        shared['finder'], shared['despiker'], shared['assessor'],
        frame_queue, stop_radar,
        shared['cli'], shared['data'], shared['cfg_path'], shared
    ))

    radar_proc.start()

    try:
        while True:
            header, det_obj, snr, noise = frame_queue.get(timeout=2.0)

            # Radar frame → despike and assess using multiprocessing
            despike_f = cpu_executor.submit(despiker.process, det_obj, snr, noise)
            assess_f = cpu_executor.submit(assessor.assess, det_obj)
            despike_res = despike_f.result()
            safe, assess_res = assess_f.result()

            # Apply IMU compensation if needed
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
                    'x': pts_enu[:, 0],
                    'y': pts_enu[:, 1],
                    'z': pts_enu[:, 2],
                    'numObj': despike_res['numObj']
                }

            # Print landing zone status
            radar = shared.get('radar')
            coeffs = assess_res.get('plane') or []
            if radar:
                if len(coeffs) < 4:
                    radar.warn_print("Landing zone UNSAFE (Insufficient data)")
                else:
                    if safe:
                        radar.info_print(
                            f"Landing zone SAFE slope={assess_res['slope_deg']:.1f}deg, "
                            f"inliers={assess_res['inlier_ratio']*100:.0f}%, "
                            f"res={assess_res['mean_residual']*100:.1f}cm"
                        )
                    else:
                        radar.warn_print(
                            f"Landing zone UNSAFE ({assess_res.get('reason','')}) "
                            f"slope={assess_res.get('slope_deg',0):.1f}deg, "
                            f"inliers={assess_res.get('inlier_ratio',0)*100:.0f}%, "
                            f"res={assess_res.get('mean_residual',0)*100:.1f}cm"
                        )

            # Update landing monitor with fresh result
            lm = shared.get('landing_monitor')
            mode = shared.get('mode_holder', {}).get('mode')
            if lm:
                lm.update(smoothed, assess_res, mode)

    except KeyboardInterrupt:
        logger.info("Interrupted by user, shutting down...")

    except Exception:
        logger.exception("Unexpected error in main loop")

    finally:
        stop_radar.set()
        radar_proc.terminate()
        radar_proc.join()
        cpu_executor.shutdown(wait=False, cancel_futures=True)
        sys.exit(0)

if __name__ == "__main__":
    main()
