import os
import sys
import time
import serial
import threading
import traceback
import numpy as np
from PyQt6.QtWidgets import QApplication
#--------------------------------------------------------
from Parser import RadarParser
from Filter import RadarDespiker
# from Plotter import RadarPlotter
from GUI import DroneLandingStatus
from PortFinder import DevicePortFinder
from PlaneLand import LandingZoneAssessor
from IMUCompensator import AttitudeCompensator
from serial.serialutil import SerialException


# main_application.py

# Shared state for autopilot
compensator = None
last_att_time = 0
_autopilot_stop = False
calli_req = True  # Set True once here for a single calibration call

# Radar port names, to avoid scanning them
cli = None
data = None


def _autopilot_reconnect_loop(finder):
    """
    Background thread: periodically tries to establish MAVLink connection.
    When successful, assigns to module-level 'compensator' and updates 'last_att_time'.
    """
    global compensator, last_att_time, _autopilot_stop, cli, data

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
                time.sleep(reconnect_interval)
            except Exception as e:
                print(f"[Autopilot] Background reconnect failed: {e}. Retrying in {reconnect_interval:.1f}s…")
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

    app = QApplication(sys.argv)
    gui = DroneLandingStatus()
    gui.show()

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
                                compensator.internal_calibrate_offsets(num_samples=100, delay=0.01) # Replace with your actual calibration method name
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

            gui.update_status(safe, m)
            app.processEvents()

    except KeyboardInterrupt:
        print("[System] Interrupted by user. Exiting cleanly...")
    except Exception as e:
        print(f"[System] Unexpected error: {e.__class__.__name__}: {e}")
        maybe_traceback()
        time.sleep(2)
    finally:
        _autopilot_stop = True
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
        # try:
        #     # plotter.export_history_csv()
        #     # plotter.show_history_table()
        # except Exception as e:
        #     print(f"[Cleanup] Failed to show history table: {e}")
