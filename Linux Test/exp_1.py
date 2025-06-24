import os
import time
import datetime
import traceback
import numpy as np
from collections import deque
from statistics import median
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pymavlink import mavutil

# External modules
from Parser import RadarParser
from Filter import RadarDespiker
from PortFinder import DevicePortFinder
from PlaneLand import LandingZoneAssessor
from IMUCompensator import AttitudeCompensator

# ============================ sensors.py ============================
class DistanceSensorSender:
    """Send MAVLink DISTANCE_SENSOR messages at a fixed rate (I/O-light)."""
    def __init__(self, mav, min_range=0.05, max_range=5.0, send_rate_hz=10):
        self.mav = mav
        self.min_cm = max(0, min(int(min_range * 100), 65535))
        self.max_cm = max(0, min(int(max_range * 100), 65535))
        self.interval = 1.0 / send_rate_hz
        self._last_time = 0.0

    def send(self, distance_m):
        now = time.time()
        if now - self._last_time < self.interval:
            return
        self._last_time = now
        ts = datetime.datetime.now().isoformat()
        try:
            raw_cm = int(round(distance_m * 100)) if isinstance(distance_m, (int, float)) else self.max_cm
            dist_cm = max(self.min_cm, min(raw_cm, self.max_cm))
            print(f"[{ts}] [DEBUG] Distance->{dist_cm}cm")
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
            raw = msg.pack(self.mav.mav)
            self.mav.write(raw)
        except Exception:
            print(f"[{ts}] [ERROR] send() failed:")
            traceback.print_exc()

# ============================ watchers.py ============================
def mode_watcher_loop(mav, mode_holder, stop_event):
    """I/O-bound loop: read HEARTBEAT and update mode."""
    while not stop_event.is_set():
        msg = mav.recv_match(type='HEARTBEAT', blocking=True, timeout=1)
        if msg:
            try:
                mode_holder['mode'] = msg.custom_mode.decode('utf-8')
            except Exception:
                base = msg.base_mode
                mode_holder['mode'] = 'ARMED' if (base & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED) else 'DISARMED'
            print(f"[ModeWatcher] Mode={mode_holder['mode']}")
        time.sleep(0.1)

# ============================ monitors.py ============================
class LandingMonitor:
    """Evaluate landing safety; issue warnings/aborts."""
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
        # Send distance
        dist = float(min(smoothed['z'])) if smoothed.get('numObj', 0) > 0 else None
        self.distance_sender.send(dist)

        # Not landing/RTL => reset
        if not mode or ('LAND' not in mode.upper() and 'RTL' not in mode.upper()):
            self._reset()
            return

        # Safety check
        safe_frame = False
        num = smoothed.get('numObj', 0)
        if num < 3:
            print(f"[LandingMonitor] {num} points -> UNSAFE")
        else:
            slope = assess.get('slope_deg')
            inlier = assess.get('inlier_ratio')
            self.buf_slope.append(slope)
            self.buf_inlier.append(inlier)
            if len(self.buf_slope) == self.buf_slope.maxlen:
                med_s = median(self.buf_slope)
                med_i = median(self.buf_inlier)
                safe_frame = (med_s < self.th_slope and med_i > self.th_inlier)
                print(f"[LandingMonitor] Median slope={med_s:.2f}, inlier={med_i:.2f} -> {'SAFE' if safe_frame else 'UNSAFE'}")
            else:
                print(f"[LandingMonitor] Buffer size={len(self.buf_slope)} -> UNSAFE")

        now = time.time()
        if safe_frame:
            self.count_safe += 1
            self.count_unsafe = 0
        else:
            self.count_unsafe += 1
            self.count_safe = 0

        # Warning
        if self.count_unsafe >= self.min_warn and not self.warned:
            if not self.unsafe_start:
                self.unsafe_start = now
            elif now - self.unsafe_start >= self.warn_time:
                self._abort()

        # Clear
        if (self.warned or self.aborted) and self.count_safe >= self.min_clear:
            print(f"[LandingMonitor] Clearing state")
            self._reset()

    def _issue_warning(self, text):
        try:
            self.mav.mav.statustext_send(mavutil.mavlink.MAV_SEVERITY_WARNING, text.encode('utf-8'))
        except Exception:
            pass
        print(f"[LandingMonitor][WARNING] {text}")

    def _abort(self):
        msg = "Landing unsafe: aborting and switching to LOITER"
        print("[LandingMonitor] Aborting landing (LOITER)")
        try:
            self.mav.set_mode_loiter()
        except Exception:
            # fallback
            self.mav.mav.command_long_send(
                self.mav.target_system,
                self.mav.target_component,
                mavutil.mavlink.MAV_CMD_DO_SET_MODE,
                0,
                mavutil.mavlink.MAV_MODE_FLAG_AUTO_ENABLED,
                2, 0, 0, 0, 0, 0
            )
        self._issue_warning(msg)
        self.warned = True
        self.aborted = True

# ============================ managers.py ============================
def radar_manager(finder, despiker, assessor, shared, stop_event):
    """I/O-bound radar reconnection and init."""
    while not stop_event.is_set():
        now = time.time()
        radar = shared.get('radar')
        last = shared.get('last_radar_time', 0)
        if not radar or not getattr(radar, 'data_serial', None) or not radar.data_serial.is_open or (now - last) > 5.0:
            if radar:
                try:
                    radar.close()
                except:
                    pass
                shared['radar'] = None
            print(f"[{datetime.datetime.now().isoformat()}] [Radar] Reconnecting...")
            try:
                cli, data = '/dev/ttyUSB0', '/dev/ttyUSB1'
                cfg = os.path.join(os.getcwd(), 'best_res_4cm.cfg')
                radar = RadarParser(cli, data, cfg, debug=False, enable_logging=False)
                radar.initialize_ports()
                radar.send_config()
                time.sleep(2)
                radar.info_print('Connected successfully')
                shared.update({'radar': radar, 'cli': cli, 'data': data, 'last_radar_time': time.time()})
            except Exception as e:
                print(f"[Radar][Connection Error] {e}")
                time.sleep(3.0)
        else:
            time.sleep(0.1)

def autopilot_manager(finder, shared, stop_event, executor):
    """I/O-bound autopilot reconnect and setup."""
    while not stop_event.is_set():
        if shared.get('compensator') is None:
            try:
                master, _ = finder.find_autopilot_connection(
                    timeout=2.0,
                    exclude_ports=[shared.get('cli'), shared.get('data')]
                )
            except RuntimeError:
                master = None
            if master:
                comp = AttitudeCompensator(master)
                shared['compensator'] = comp
                shared['master'] = master
                shared['last_att_time'] = time.time()
                # launch mode watcher
                shared['mode_holder'] = {'mode': None}
                shared['stop_mode'] = threading.Event()
                executor.submit(
                    mode_watcher_loop,
                    master,
                    shared['mode_holder'],
                    shared['stop_mode']
                )
                # sensor sender and monitor
                shared['distance_sender'] = DistanceSensorSender(master)
                shared['landing_monitor'] = LandingMonitor(
                    master,
                    shared['distance_sender'],
                    buffer_size=20,
                    slope_threshold_deg=5.0,
                    inlier_threshold=0.6,
                    warning_duration=3.0,
                    min_consecutive_to_warn=5,
                    min_consecutive_to_clear=5
                )
        time.sleep(5.0)

# ============================ main.py ============================
if __name__ == '__main__':
    finder = DevicePortFinder()
    despiker = RadarDespiker()
    assessor = LandingZoneAssessor(0.05, 500, 10, 0.7)

    shared = {
        'radar': None,
        'last_radar_time': 0.0,
        'cli': None,
        'data': None,
        'compensator': None,
        'mode_holder': None
    }

    # Executors
    io_executor = ThreadPoolExecutor(max_workers=4)
    cpu_executor = ProcessPoolExecutor()

    # Start I/O managers
    stop_radar = threading.Event()
    stop_auto = threading.Event()
    io_executor.submit(radar_manager, finder, despiker, assessor, shared, stop_radar)
    io_executor.submit(autopilot_manager, finder, shared, stop_auto, io_executor)

    try:
        while True:
            radar = shared.get('radar')
            if not radar:
                time.sleep(0.1)
                continue
            # I/O-bound: read frame
            future_frame = io_executor.submit(radar.read_frame)
            try:
                header, det_obj, snr, noise = future_frame.result(timeout=5.0)
            except Exception as e:
                print(f"[Radar][Error] {e}")
                try:
                    radar.close()
                except:
                    pass
                shared['radar'] = None
                continue
            shared['last_radar_time'] = time.time()

            # CPU-bound tasks
            futures = []
            futures.append(cpu_executor.submit(despiker.process, det_obj, snr, noise))
            futures.append(cpu_executor.submit(assessor.assess, det_obj))
            comp = shared.get('compensator')
            if comp:
                # transform pointcloud CPU-bound
                def transform(args):
                    pts, c = args
                    return c.transform_pointcloud(pts)
                # will fill after despike
            # collect despike and assess
            despike_res = None
            assess_res = None
            for fut in as_completed(futures):
                res = fut.result()
                if isinstance(res, dict) and 'x' in res:
                    despike_res = res
                else:
                    assess_res = res

            if despike_res and despike_res.get('numObj', 0) >= 3:
                pts_body = np.vstack((despike_res['x'], despike_res['y'], despike_res['z'])).T
                if comp:
                    try:
                        pts_enu = comp.transform_pointcloud(pts_body)
                    except Exception:
                        comp.close()
                        shared['compensator'] = None
                        pts_enu = pts_body
                else:
                    pts_enu = pts_body
                smoothed = {
                    'x': pts_enu[:, 0],
                    'y': pts_enu[:, 1],
                    'z': pts_enu[:, 2],
                    'numObj': despike_res['numObj']
                }
            else:
                smoothed = {'x': [], 'y': [], 'z': [], 'numObj': 0}

            # Landing zone output
            safe, plane_info = assess_res
            if not plane_info.get('plane'):
                radar.warn_print('Landing zone UNSAFE: Insufficient data')
            else:
                msg = (
                    f"SAFE slope={plane_info['slope_deg']:.1f}deg" if safe else
                    f"UNSAFE slope={plane_info['slope_deg']:.1f}deg"
                )
                (radar.info_print if safe else radar.warn_print)(msg)

            # Update monitor
            mode = shared['mode_holder']['mode'] if shared['mode_holder'] else None
            shared['landing_monitor'].update(smoothed, plane_info, mode)

    except KeyboardInterrupt:
        print('Interrupted by user, shutting down...')
    finally:
        stop_radar.set()
        stop_auto.set()
        io_executor.shutdown(wait=True)
        cpu_executor.shutdown(wait=True)
        if shared.get('radar'):
            try:
                shared['radar'].close()
            except:
                pass
        if shared.get('compensator'):
            try:
                shared['compensator'].close()
            except:
                pass
