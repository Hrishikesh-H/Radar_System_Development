import sys
import datetime
import os
import time
from serial.tools import list_ports
from pymavlink import mavutil
import traceback
import numpy as np
from PyQt6.QtWidgets import QApplication

from Parser import RadarParser
from Filter import RadarDespiker
from Plotter import RadarPlotter
from GUI import DroneLandingStatus
from PlaneLand import LandingZoneAssessor
from IMUCompensator import AttitudeCompensator
from PortFinder import DevicePortFinder


# main_application.py

if __name__ == "__main__":
    finder = DevicePortFinder()
    radar = None
    cli = None
    data = None
    compensator = None

    plotter = RadarPlotter()
    despiker = RadarDespiker()
    assessor = LandingZoneAssessor(0.05, 500, 10, 0.7)

    app = QApplication(sys.argv)
    gui = DroneLandingStatus()
    gui.show()

    last_radar_time = 0
    last_att_time = 0
    RECONNECT_INTERVAL = 3.0

    try:
        while True:
            now = time.time()

            # --- RADAR RECONNECTION LOGIC ---
            if radar is None or (now - last_radar_time > 1.5):
                # If we had a radar instance, close it before attempting to reconnect
                if radar is not None:
                    try:
                        radar.close()
                    except Exception:
                        pass
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
                except Exception as e:
                    print(f"[Radar] Reconnect failed: {e}. Retrying in {RECONNECT_INTERVAL:.1f}s...")
                    time.sleep(RECONNECT_INTERVAL)
                    continue  # Skip downstream processing until radar is back

            # --- COMPENSATOR (MAVLINK) RECONNECTION LOGIC ---
            if compensator is None or (now - last_att_time > 3.0):
                try:
                    exclude_ports = [cli, data] if (cli and data) else []
                    conn_str, conn_baud, ap_name = finder.find_autopilot_connection(timeout=2.0, exclude_ports=exclude_ports)
                    if conn_str is not None:
                        if conn_baud is None:
                            master = mavutil.mavlink_connection(conn_str, source_system=255)
                        else:
                            master = mavutil.mavlink_connection(conn_str, baud=conn_baud, source_system=255)
                        master.wait_heartbeat()
                        compensator = AttitudeCompensator(master)
                        print("[Autopilot] MAVLink connection established for compensation.")
                        last_att_time = time.time()
                    else:
                        raise RuntimeError("No autopilot connection found")
                except Exception as e:
                    compensator = None
                    print(f"[Autopilot] Connection missing — will retry in background (no compensation available). ({e})")
                    # Continue without blocking; raw points will be used

            # --- RADAR DATA ACQUISITION ---
            try:
                header, det_obj, snr, noise = radar.read_frame()
                last_radar_time = time.time()
            except Exception as e:
                print(f"[Radar] Read failed: {e}. Closing radar instance and restarting reconnection...")
                try:
                    radar.close()
                except Exception:
                    pass
                radar = None
                continue  # Go back to radar reconnection

            # --- POINT PROCESSING AND COMPENSATION ---
            smoothed_det_obj = {'x': [], 'y': [], 'z': [], 'numObj': 0}
            if det_obj and det_obj.get('numObj', 0) >= 3:
                spr = despiker.process(det_obj, snr, noise)
                pts_body = np.vstack((spr['x'], spr['y'], spr['z'])).T

                if compensator:
                    try:
                        pts_enu = compensator.transform_pointcloud(pts_body)
                        last_att_time = time.time()
                    except Exception as e:
                        print(f"[Autopilot] Compensation failed: {e}. Using raw points.")
                        pts_enu = pts_body
                else:
                    print("[Autopilot] Compensation unavailable — using raw points.")
                    pts_enu = pts_body

                smoothed_det_obj = {
                    'x': pts_enu[:, 0],
                    'y': pts_enu[:, 1],
                    'z': pts_enu[:, 2],
                    'numObj': det_obj['numObj']
                }

            # --- PLOTTING AND ASSESSMENT ---
            plotter.update(det_obj, smoothed_det_obj)

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
        print(f"[System] Unexpected error: {e}")
        traceback.print_exc()
        time.sleep(2)
    finally:
        # --- CLEANUP ---
        if radar:
            try:
                radar.close()
            except Exception:
                pass
        if compensator:
            try:
                compensator.close()
            except Exception:
                pass
        try:
            plotter.export_history_csv()
            plotter.show_history_table()
        except Exception:
            pass
