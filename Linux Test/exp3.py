# optimized_uav_system.py

import os
import time
import datetime
import traceback
import sys
import threading
import multiprocessing
import numpy as np
from collections import deque
from statistics import median
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pymavlink import mavutil

# External modules
from Parser import RadarParser
from Filter import RadarDespiker
from PortFinder import DevicePortFinder
from PlaneLand import LandingZoneAssessor
from IMUCompensator import AttitudeCompensator

# ============================ sensors.py ============================
class DistanceSensorSender:
    """Send MAVLink DISTANCE_SENSOR messages at a fixed rate."""
    def __init__(self, mav, min_range=0.05, max_range=5.0, rate_hz=10):
        self.mav = mav
        self.min_cm = max(0, min(int(min_range * 100), 65535))
        self.max_cm = max(0, min(int(max_range * 100), 65535))
        self.interval = 1.0 / rate_hz
        self._last_time = time.time()

    def send(self, distance_m):
        now = time.time()
        if now - self._last_time < self.interval:
            return
        self._last_time = now
        try:
            raw_cm = int(round(distance_m * 100)) if isinstance(distance_m, (int, float)) else self.max_cm
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
        except Exception:
            print(f"[{datetime.datetime.now().isoformat()}] [ERROR] DistanceSensorSender.send failed:")
            traceback.print_exc()

# ============================ watchers.py ============================
def mode_watcher_loop(mav, mode_holder, stop_event):
    """Continuously watch HEARTBEAT and update mode."""
    while not stop_event.is_set():
        try:
            msg = mav.recv_match(type='HEARTBEAT', blocking=True, timeout=1)
            if msg:
                try:
                    mode_holder['mode'] = msg.custom_mode.decode('utf-8')
                except Exception:
                    base = msg.base_mode
                    mode_holder['mode'] = 'ARMED' if (base & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED) else 'DISARMED'
        except Exception:
            traceback.print_exc()

# ============================ monitors.py ============================
class LandingMonitor:
    """Evaluate landing safety; issue warnings and abort if necessary."""
    def __init__(self, mav, distance_sender, **cfg):
        self.mav = mav
        self.distance_sender = distance_sender
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
        dist = float(min(smoothed['z'])) if smoothed.get('numObj', 0) > 0 else None
        self.distance_sender.send(dist)

        # reset if not landing/RTL
        if not mode or ('LAND' not in mode.upper() and 'RTL' not in mode.upper()):
            self._reset()
            return

        # assess frame
        safe_frame = False
        if smoothed.get('numObj', 0) >= 3:
            s = assess.get('slope_deg')
            i = assess.get('inlier_ratio')
            self.buf_slope.append(s)
            self.buf_inlier.append(i)
            if len(self.buf_slope) == self.buf_slope.maxlen:
                med_s = median(self.buf_slope)
                med_i = median(self.buf_inlier)
                safe_frame = (med_s < self.th_slope and med_i > self.th_inlier)

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
            pass

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

# ============================ managers.py ============================
def radar_process(cli_port, data_port, cfg_path, radar_queue, stop_event):
    """Maintain radar connection and feed frames into queue."""
    radar = None
    while not stop_event.is_set():
        try:
            if not radar or not getattr(radar, 'data_serial', None) or not radar.data_serial.is_open:
                if radar:
                    radar.close()
                radar = RadarParser(cli_port, data_port, cfg_path, debug=False, enable_logging=False)
                radar.initialize_ports()
                radar.send_config()
                time.sleep(1)
            frame = radar.read_frame()
            radar_queue.put(frame, block=True)
        except Exception:
            traceback.print_exc()
            time.sleep(1)
    if radar:
        radar.close()

class AutopilotManager:
    """Persistent autopilot connection and mode watcher."""
    def __init__(self, finder, shared, io_executor):
        self.finder = finder
        self.shared = shared
        self.io_executor = io_executor
        self.stop_event = threading.Event()

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while not self.stop_event.is_set():
            if self.shared.get('master') is None:
                try:
                    master, _ = self.finder.find_autopilot_connection(
                        timeout=2.0,
                        exclude_ports=[self.shared['cli'], self.shared['data']]
                    )
                    if master:
                        comp = AttitudeCompensator(master)
                        sender = DistanceSensorSender(master)
                        monitor = LandingMonitor(
                            master, sender,
                            buffer_size=20,
                            slope_threshold_deg=5.0,
                            inlier_threshold=0.6,
                            warning_duration=3.0,
                            min_consecutive_to_warn=5,
                            min_consecutive_to_clear=5
                        )
                        self.shared.update({
                            'master': master,
                            'compensator': comp,
                            'distance_sender': sender,
                            'landing_monitor': monitor,
                            'mode_holder': {'mode': None},
                            'stop_mode': threading.Event()
                        })
                        self.io_executor.submit(
                            mode_watcher_loop,
                            master,
                            self.shared['mode_holder'],
                            self.shared['stop_mode']
                        )
                        print(f"[Autopilot] Connected to {master.target_system}/{master.target_component}")
                except Exception:
                    traceback.print_exc()
                    time.sleep(2)
            else:
                time.sleep(1)

    def stop(self):
        self.stop_event.set()

# ============================ main.py ============================
if __name__ == '__main__':
    # Init modules
    finder = DevicePortFinder()
    despiker = RadarDespiker()
    assessor = LandingZoneAssessor(0.05, 500, 10, 0.7)

    # Shared resources
    manager = multiprocessing.Manager()
    radar_queue = manager.Queue(maxsize=50)
    stop_rad = multiprocessing.Event()

    # Shared metadata; autopilot manager will populate runtime handles
    shared = {
        'cli': '/dev/ttyUSB0',
        'data': '/dev/ttyUSB1',
        'last_radar_time': time.time()
    }

    # Start radar process
    cfg_file = os.path.join(os.getcwd(), 'best_res_4cm.cfg')
    radar_proc = multiprocessing.Process(
        target=radar_process,
        args=(shared['cli'], shared['data'], cfg_file, radar_queue, stop_rad),
        daemon=True
    )
    radar_proc.start()

    # Start autopilot manager
    io_executor = ThreadPoolExecutor(max_workers=2)
    cpu_executor = ProcessPoolExecutor(max_workers=2)
    ap_manager = AutopilotManager(finder, shared, io_executor)
    ap_manager.start()

    # Main loop
    try:
        while True:
            header, det_obj, snr, noise = radar_queue.get(timeout=2.0)
            shared['last_radar_time'] = time.time()

            # CPU-bound tasks
            despike_fut = cpu_executor.submit(despiker.process, det_obj, snr, noise)
            assess_fut  = cpu_executor.submit(assessor.assess, det_obj)
            despike_res = despike_fut.result()
            safe, assess_res = assess_fut.result()

            # Pointcloud transform
            smoothed = {'x': [], 'y': [], 'z': [], 'numObj': 0}
            comp = shared.get('compensator')
            if despike_res.get('numObj', 0) >= 3:
                pts = np.vstack((despike_res['x'], despike_res['y'], despike_res['z'])).T
                if comp:
                    try:
                        pts_enu = cpu_executor.submit(comp.transform_pointcloud, pts).result()
                        print(f"[Compensated] first point: {pts_enu[0]}")
                    except Exception:
                        comp.close()
                        shared['compensator'] = None
                        pts_enu = pts
                else:
                    pts_enu = pts
                smoothed = {
                    'x': pts_enu[:,0], 'y': pts_enu[:,1], 'z': pts_enu[:,2],
                    'numObj': despike_res['numObj']
                }

            # Status output
            if not assess_res.get('plane'):
                print('[Landing Monitor] UNSAFE: Insufficient data')
            else:
                status = 'SAFE' if safe else 'UNSAFE'
                print(f"[Landing Monitor] {status} slope={assess_res['slope_deg']:.1f}deg")

            # Landing monitor update
            monitor = shared.get('landing_monitor')
            mode = shared.get('mode_holder', {}).get('mode') if shared.get('mode_holder') else None
            if monitor:
                monitor.update(smoothed, assess_res, mode)

    except KeyboardInterrupt:
        print('Interrupted, shutting down...')

    finally:
        stop_rad.set()
        ap_manager.stop()
        io_executor.shutdown(wait=False, cancel_futures=True)
        cpu_executor.shutdown(wait=False, cancel_futures=True)
        radar_proc.terminate()
        sys.exit(0)
