import time
import numpy as np
from Parser import RadarParser
from Filter import RadarDespiker
from Plotter import RadarPlotter
import sys
import os
from GUI import DroneLandingStatus
from PyQt6.QtWidgets import QApplication
from PlaneLand import LandingZoneAssessor
from IMUCompensator import AttitudeCompensator
from PortFinder import DevicePortFinder
from pymavlink import mavutil



if __name__ == "__main__":
    finder = DevicePortFinder()
    try:
        cli, data = finder.find_radar_ports_by_description()
        print(f"CLI port: {cli}, DATA port: {data}")
    except RuntimeError as e:
        print(f"Port detection error: {e}")
        sys.exit(1)

    config_path = os.path.join(os.getcwd(), "best_res_4cm.cfg")
    radar = RadarParser(
        cli_port=cli,
        data_port=data,
        config_file=config_path,
        debug=False,
        enable_logging=False,
        log_prefix="radar_log"
    )
    radar.initialize_ports()
    radar.send_config()
    time.sleep(2)
    radar.info_print("Listening for radar data...")

    try:
        conn_str, conn_baud, autopilot = finder.find_autopilot_connection(timeout=2.0, exclude_ports=[cli, data])
    except RuntimeError as e:
        radar.warn_print(f"Autopilot detection failed: {e}")
        conn_str, conn_baud, autopilot = None, None, None

    if conn_str is not None:
        try:
            print(f"[Main] Connecting to autopilot on {conn_str}" + (f" @ {conn_baud}" if conn_baud else ""))
            if conn_baud is None:
                master = mavutil.mavlink_connection(conn_str, source_system=255)
            else:
                master = mavutil.mavlink_connection(conn_str, baud=conn_baud, source_system=255)

            master.wait_heartbeat()
            print(f"[Main] Heartbeat received from autopilot ({autopilot})")
            compensator = AttitudeCompensator(master)
            print(f"[Main] Using autopilot: {compensator.autopilot} on {conn_str}" + (f"@{conn_baud}" if conn_baud else ""))
        except Exception as exc:
            radar.warn_print(f"AttitudeCompensator init failed: {exc}")
            import traceback
            traceback.print_exc()
            compensator = None
    else:
        compensator = None

    plotter = RadarPlotter()
    despiker = RadarDespiker()
    assessor = LandingZoneAssessor(distance_thresh=0.05, min_iterations=500, slope_thresh_deg=10, inlier_ratio_thresh=0.7)

    app = QApplication(sys.argv)
    gui = DroneLandingStatus()
    gui.show()

    try:
        while True:
            header, det_obj, snr, noise = radar.read_frame()
            smoothed_det_obj = {'x': [], 'y': [], 'z': [], 'numObj': 0}

            if det_obj and det_obj.get('numObj', 0) >= 3:
                spr = despiker.process(det_obj, snr, noise)
                pts_body = np.vstack((spr['x'], spr['y'], spr['z'])).T

                if compensator is not None:
                    try:
                        pts_enu = compensator.transform_pointcloud(pts_body)
                    except Exception as e_att:
                        radar.warn_print(f"Attitude compensation skipped: {e_att}")
                        pts_enu = pts_body
                else:
                    pts_enu = pts_body

                smoothed_det_obj = {
                    'x': pts_enu[:, 0],
                    'y': pts_enu[:, 1],
                    'z': pts_enu[:, 2],
                    'numObj': det_obj['numObj']
                }

                print("Num X-spikes fixed:", np.sum(det_obj['x'] != smoothed_det_obj['x']))
                print("Num Y-spikes fixed:", np.sum(det_obj['y'] != smoothed_det_obj['y']))
                print("Num Z-spikes fixed:", np.sum(det_obj['z'] != smoothed_det_obj['z']))

            plotter.update(det_obj, smoothed_det_obj)

            safe, m = assessor.assess(det_obj)

            coeffs = m.get('plane', None)
            if coeffs is None or not hasattr(coeffs, '__len__') or len(coeffs) < 4:
                radar.warn_print("Landing zone UNSAFE (Insufficient data for slope estimation)")
            else:
                if safe:
                    radar.info_print(
                        f"Landing zone SAFE  slope={m['slope_deg']:.1f}°, "
                        f"inliers={m['inlier_ratio']*100:.0f}%, "
                        f"res={m['mean_residual']*100:.1f}cm"
                    )
                else:
                    radar.warn_print(
                        f"Landing zone UNSAFE ({m.get('reason','')})  "
                        f"slope={m.get('slope_deg',0):.1f}°, "
                        f"inliers={m.get('inlier_ratio',0)*100:.0f}%, "
                        f"res={m.get('mean_residual',0)*100:.1f}cm"
                    )

            gui.update_status(safe, m)
            app.processEvents()

    except KeyboardInterrupt:
        radar.info_print("Interrupted by user.")
    finally:
        plotter.export_history_csv()
        plotter.show_history_table()
        radar.close()
        if compensator is not None:
            compensator.close()