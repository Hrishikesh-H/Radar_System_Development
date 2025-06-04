import time
import numpy as np
from Parser import RadarParser
from pyransac3d import Plane
import math
import sys

from serial.tools import list_ports
from scipy.spatial import KDTree

from scipy.spatial import KDTree
import numpy as np
import datetime
import traceback

def debug_print(msg):
    print(f"[{datetime.datetime.now().isoformat()}] [DEBUG] {msg}")

def despike_xyz_sparse(det_obj, snr, noise,
                       radius=0.5, base_z_thresh=0.3, base_xy_thresh=0.3,
                       snr_gate=7.0):
    """
    Filters spikes in sparse 3D radar data using neighborhood smoothing.
    Incorporates SNR and noise information to weight corrections and adjust thresholds.
    """
    x = np.array(det_obj['x'])
    y = np.array(det_obj['y'])
    z = np.array(det_obj['z'])
    num = det_obj.get('numObj', len(x))

    pts = np.vstack((x, y, z)).T
    tree = KDTree(pts)

    # Ensure SNR and noise are proper numpy arrays and have the right shape
    try:
        snr = np.array(snr)
        noise = np.array(noise)
    except Exception as e:
        debug_print(f"Failed to convert snr/noise to arrays: {e}")
        return det_obj

    if snr.ndim != 1 or noise.ndim != 1:
        debug_print(f"SNR or noise is not 1D: snr.ndim={snr.ndim}, noise.ndim={noise.ndim}")
        return det_obj

    if len(snr) != len(x) or len(noise) != len(x):
        debug_print(f"SNR/Noise length mismatch: snr={len(snr)}, noise={len(noise)}, points={len(x)}")
        return det_obj

    x_f, y_f, z_f = x.copy(), y.copy(), z.copy()

    for i, p in enumerate(pts):
        try:
            adaptive_radius = radius * 1.5 if snr[i] < snr_gate else radius
            idxs = tree.query_ball_point(p, r=adaptive_radius)

            if len(idxs) < 3:
                debug_print(f"Skipping index {i} due to low neighbors ({len(idxs)})")
                continue

            neigh_pts = pts[idxs]
            neigh_x, neigh_y, neigh_z = neigh_pts[:, 0], neigh_pts[:, 1], neigh_pts[:, 2]
            neigh_snr = snr[idxs]
            neigh_noise = noise[idxs]

            local_noise = np.mean(neigh_noise)
            z_thresh = base_z_thresh * (1 + local_noise)
            xy_thresh = base_xy_thresh * (1 + local_noise)

            dx = abs(x[i] - np.median(neigh_x))
            dy = abs(y[i] - np.median(neigh_y))
            dz = abs(z[i] - np.median(neigh_z))

            snr_weight = neigh_snr
            noise_weight = 1 / (1 + neigh_noise)
            combined_weight = snr_weight * noise_weight

            if np.sum(combined_weight) == 0:
                w = np.ones_like(combined_weight) / len(combined_weight)
                debug_print(f"[Index {i}] Zero combined weight, using equal weights")
            else:
                w = combined_weight / np.sum(combined_weight)

            if dx > xy_thresh:
                x_f[i] = np.dot(w, neigh_x)
                debug_print(f"[Index {i}] X corrected by smoothing")

            if dy > xy_thresh:
                y_f[i] = np.dot(w, neigh_y)
                debug_print(f"[Index {i}] Y corrected by smoothing")

            if dz > z_thresh:
                z_f[i] = np.dot(w, neigh_z)
                debug_print(f"[Index {i}] Z corrected by smoothing")

        except Exception as e_inner:
            debug_print(f"Error at point {i}: {e_inner}")
            traceback.print_exc()

    return {
        'x': x_f,
        'y': y_f,
        'z': z_f,
        'numObj': len(x)
    }

def find_radar_ports_by_description(keyword="CP2105"):
    """
    Auto-detects CLI and DATA serial ports by looking for `keyword` in the port description.
    Returns:
        (cli_port: str, data_port: str)
    Raises:
        RuntimeError if fewer than two matching ports are found.
    """
    ports = list_ports.comports()
    # Filter ports whose description contains the keyword
    matches = [p.device for p in ports if keyword.lower() in (p.description or "").lower()]

    if len(matches) < 2:
        raise RuntimeError(f"Expected at least 2 ports matching '{keyword}', found: {matches}")

    # Assign first as CLI, second as DATA
    cli_port, data_port = matches[0], matches[1]
    return cli_port, data_port


def assess_landing_zone(det_obj,
                        distance_thresh=0.05,
                        min_iterations=500,
                        slope_thresh_deg=10,
                        inlier_ratio_thresh=0.8):
    """
    Given det_obj from RadarParser, fit a plane and decide landing safety.
    
    Args:
      det_obj (dict): must contain 'x','y','z' numpy arrays and 'numObj'
      distance_thresh (float): RANSAC distance threshold (m)
      min_iterations (int): RANSAC trials
      slope_thresh_deg (float): max allowable tilt from vertical (degrees)
      inlier_ratio_thresh (float): min fraction of points fitting plane
    
    Returns:
      safe (bool): True if landing zone is safe
      metrics (dict): {
         'plane'         : (a,b,c,d),
         'slope_deg'     : float,
         'inlier_ratio'  : float,
         'mean_residual' : float
      }
    """
    # 1) collect points
    x, y, z = det_obj['x'], det_obj['y'], det_obj['z']
    pts = np.vstack((x, y, z)).T
    n = det_obj['numObj']
    if n < 3:
        return False, {'reason': 'not enough points'}

    # 2) fit plane
    model = Plane()
    coeffs, inlier_idxs = model.fit(pts,
                                    thresh=distance_thresh,
                                    maxIteration=min_iterations)
    a, b, c, d = coeffs
    inlier_mask = np.zeros(n, dtype=bool)
    inlier_mask[inlier_idxs] = True
    inlier_count = inlier_mask.sum()
    inlier_ratio = inlier_count / n

    # 3) compute residuals for inliers
    dists = np.abs((pts.dot([a,b,c]) + d))  # unsigned distance
    mean_residual = dists[inlier_mask].mean() if inlier_count>0 else np.inf

    # 4) slope: angle between normal and vertical axis [0,1,0]
    normal = np.array([a, b, c])
    normal = normal / np.linalg.norm(normal)  # ensure unit vector
    cos_theta = abs(np.dot(normal, [0, 1, 0]))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # prevent NaNs
    slope_deg = math.degrees(math.acos(cos_theta))

    # 5) decide
    safe = (slope_deg <= slope_thresh_deg and
            inlier_ratio >= inlier_ratio_thresh and
            mean_residual <= distance_thresh)

    metrics = {
        'plane': (a, b, c, d),
        'slope_deg': slope_deg,
        'inlier_ratio': inlier_ratio,
        'mean_residual': mean_residual
    }
    return safe, metrics

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import datetime

# Assumed imports for RadarParser, despike_xyz_sparse, assess_landing_zone, find_radar_ports_by_description

class RadarPlotter:
    """
    RadarPlotter handles professional ISO-standard plotting & data export:

    - Real-time 3D scatter of raw vs. smoothed points (distinct markers/colors).
    - History 3D scatter of all past points.
    - Temporal line plots of mean X, Y, Z over frames.
    - Tabular view and CSV export of history with timestamps.
    """
    def __init__(self):
        # ISO-standard figure settings
        plt.rc('font', size=10)
        plt.rc('axes', grid=True, facecolor='white')

        # Real-time 3D
        self.fig3d = plt.figure(figsize=(6,6))
        self.ax3d = self.fig3d.add_subplot(111, projection='3d')
        plt.show(block=False)
        self.ax3d.set_title('Real-time 3D Radar View', pad=20)
        self.ax3d.set_box_aspect((1,1,1))
        self._setup_3d_axes(self.ax3d)

        # History 3D
        self.fig_hist = plt.figure(figsize=(6,6))
        self.ax_hist = self.fig_hist.add_subplot(111, projection='3d')
        plt.show(block=False)
        self.ax_hist.set_title('History of All Data Points (3D)', pad=20)
        self.ax_hist.set_box_aspect((1,1,1))
        self._setup_3d_axes(self.ax_hist)

        # Data storage
        self.history_raw = []
        self.history_smooth = []
        self.frame_times = []  # timestamps per frame

        # Temporal
        self.fig_temp, self.ax_temp = plt.subplots(3, 1, figsize=(6, 9))
        plt.show(block=False)
        coords = ['X-axis (m)', 'Y-axis (m)', 'Z-axis (m)']
        for ax, coord in zip(self.ax_temp, coords):
            ax.set_title(f'Temporal {coord.split()[0]}', pad=10)
            ax.set_xlabel('Frame', labelpad=8)
            ax.set_ylabel(coord, labelpad=8)
            ax.grid(True)
        self.temporal_raw = {'x': [], 'y': [], 'z': []}
        self.temporal_smooth = {'x': [], 'y': [], 'z': []}
        self.frame_idx = 0

    def _setup_3d_axes(self, ax):
        ax.clear()
        ax.set_xlabel('X-axis (m)', color='red', labelpad=10)
        ax.set_ylabel('Y-axis (m)', color='green', labelpad=10)
        ax.set_zlabel('Z-axis (m)', color='blue', labelpad=10)
        arrow_len = 1.0
        for vec, color in [((1,0,0),'red'), ((0,1,0),'green'), ((0,0,1),'blue')]:
            ax.quiver(0,0,0, *vec, length=arrow_len, color=color, arrow_length_ratio=0.1)
        ax.text(arrow_len,0,0,'X', color='red')
        ax.text(0,arrow_len,0,'Y', color='green')
        ax.text(0,0,arrow_len,'Z', color='blue')
        ax.set_xlim(-5,5); ax.set_ylim(-5,5); ax.set_zlim(-5,5)

    # rest unchanged

    def update(self, det_obj, sm_obj):
        self.frame_times.append(time.time())
        self.fig_temp, self.ax_temp = plt.subplots(3, 1, figsize=(6, 9))
        plt.show(block=False)  # non-blocking display for temporal plots
        self.fig_temp.show(block=False)
        coords = ['X-axis (m)', 'Y-axis (m)', 'Z-axis (m)']
        for ax, coord in zip(self.ax_temp, coords):
            ax.set_title(f'Temporal {coord.split()[0]}', pad=10)
            ax.set_xlabel('Frame', labelpad=8)
            ax.set_ylabel(coord, labelpad=8)
            ax.grid(True)
        self.temporal_raw = {'x': [], 'y': [], 'z': []}
        self.temporal_smooth = {'x': [], 'y': [], 'z': []}
        self.frame_idx = 0

    def _setup_3d_axes(self, ax):
        ax.clear()
        ax.set_xlabel('X-axis (m)', color='red', labelpad=10)
        ax.set_ylabel('Y-axis (m)', color='green', labelpad=10)
        ax.set_zlabel('Z-axis (m)', color='blue', labelpad=10)
        arrow_len = 1.0
        for vec, color in [((1,0,0),'red'), ((0,1,0),'green'), ((0,0,1),'blue')]:
            ax.quiver(0,0,0, *vec, length=arrow_len, color=color, arrow_length_ratio=0.1)
        ax.text(arrow_len,0,0,'X', color='red')
        ax.text(0,arrow_len,0,'Y', color='green')
        ax.text(0,0,arrow_len,'Z', color='blue')
        ax.set_xlim(-5,5); ax.set_ylim(-5,5); ax.set_zlim(-5,5)

    def update(self, det_obj, sm_obj):
        self.frame_times.append(time.time())

        # Real-time 3D
        self._setup_3d_axes(self.ax3d)
        plotted = False
        if det_obj and det_obj.get('numObj', 0) > 0:
            raw_pts = np.vstack((det_obj['x'], det_obj['y'], det_obj['z'])).T
            self.ax3d.scatter(raw_pts[:,0], raw_pts[:,1], raw_pts[:,2], marker='o', s=20, alpha=0.7, label='Raw', edgecolor='k')
            self.history_raw.append(raw_pts)
            plotted = True
        if sm_obj and sm_obj.get('numObj', 0) > 0:
            sm_pts = np.vstack((sm_obj['x'], sm_obj['y'], sm_obj['z'])).T
            self.ax3d.scatter(sm_pts[:,0], sm_pts[:,1], sm_pts[:,2], marker='^', s=30, alpha=0.7, label='Smoothed', edgecolor='k')
            self.history_smooth.append(sm_pts)
            plotted = True
        if not plotted:
            self.ax3d.text2D(0.5,0.5,'No data', transform=self.ax3d.transAxes, ha='center', va='center', fontsize=12, color='red')
        if plotted:
            self.ax3d.legend(loc='upper left')
        self.fig3d.canvas.draw(); self.fig3d.canvas.flush_events(); plt.pause(0.001)

        # History 3D
        self._setup_3d_axes(self.ax_hist)
        for pts in self.history_raw:
            self.ax_hist.scatter(pts[:,0], pts[:,1], pts[:,2], alpha=0.1, c='tab:blue', marker='o')
        for pts in self.history_smooth:
            self.ax_hist.scatter(pts[:,0], pts[:,1], pts[:,2], alpha=0.3, c='tab:orange', marker='^')
        self.fig_hist.canvas.draw(); self.fig_hist.canvas.flush_events(); plt.pause(0.001)

        # Temporal
        self.frame_idx += 1
        for coord in ['x','y','z']:
            rv = np.array(det_obj.get(coord,[])) if det_obj else np.array([])
            sv = np.array(sm_obj.get(coord,[])) if sm_obj else np.array([])
            self.temporal_raw[coord].append(np.nanmean(rv) if rv.size>0 else np.nan)
            self.temporal_smooth[coord].append(np.nanmean(sv) if sv.size>0 else np.nan)
        for ax, coord in zip(self.ax_temp, ['x','y','z']):
            ax.cla()
            ax.set_xlabel('Frame', labelpad=8)
            ax.set_ylabel(f'{coord.upper()} (m)', labelpad=8)
            ax.grid(True)
            ax.plot(self.temporal_raw[coord], linestyle='-', marker='o', markersize=4, label='Raw')
            ax.plot(self.temporal_smooth[coord], linestyle='-', marker='^', markersize=6, label='Smoothed')
            ax.legend(loc='upper right')
        self.fig_temp.tight_layout();
        self.fig_temp.canvas.draw(); self.fig_temp.canvas.flush_events(); plt.pause(0.001)

    def create_history_dataframe(self):
        records = []
        for idx, pts in enumerate(self.history_raw):
            ts = self.frame_times[idx]
            for x,y,z in pts:
                records.append({'timestamp':ts,'frame':idx,'type':'raw','x':x,'y':y,'z':z})
        for idx, pts in enumerate(self.history_smooth):
            ts = self.frame_times[idx]
            for x,y,z in pts:
                records.append({'timestamp':ts,'frame':idx,'type':'smoothed','x':x,'y':y,'z':z})
        return pd.DataFrame.from_records(records)

    def export_history_csv(self, filename=None):
        df = self.create_history_dataframe()
        if filename is None:
            ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"radar_history_{ts}.csv"
        df.to_csv(filename, index=False)
        print(f"History with timestamps exported to {filename}")

    def show_history_table(self, num_rows=50):
        df = self.create_history_dataframe().head(num_rows)
        print(df.to_string(index=False))

# main() unchanged below

if __name__ == "__main__":
    cli,data = None,None
    try:
        cli,data = find_radar_ports_by_description("CP2105")
        print(f"CLI port: {cli}, DATA port: {data}")
    except RuntimeError as e:
        print(f"Port detection error: {e}"); sys.exit(1)

    config_path = os.path.join(os.getcwd(), "best_res_4cm.cfg")
    radar = RadarParser(cli_port=cli, data_port=data,
                        config_file=config_path, debug=False,
                        enable_logging=False, log_prefix="radar_log")
    radar.initialize_ports(); radar.send_config(); time.sleep(2)
    radar.info_print("Listening for radar data...")

    plotter = RadarPlotter()
    try:
        while True:
            header, det_obj, snr, noise = radar.read_frame()
            smoothed_det_obj = {'x':[], 'y':[], 'z':[], 'numObj':0}
            if det_obj and det_obj.get('numObj',0)>=3:
                spr = despike_xyz_sparse(det_obj, snr, noise)
                pts = np.vstack((spr['x'], spr['y'], spr['z'])).T
                smoothed_det_obj = {'x':pts[:,0],'y':pts[:,1],'z':pts[:,2],'numObj':det_obj['numObj']}
                print("Num X-spikes fixed:",np.sum(det_obj['x']!=smoothed_det_obj['x']))
                print("Num Y-spikes fixed:",np.sum(det_obj['y']!=smoothed_det_obj['y']))
                print("Num Z-spikes fixed:",np.sum(det_obj['z']!=smoothed_det_obj['z']))
            plotter.update(det_obj, smoothed_det_obj)
            safe,m = assess_landing_zone(smoothed_det_obj,0.05,500,10,0.8)
            if safe:
                radar.info_print(f"Landing zone SAFE slope={m['slope_deg']:.1f} deg, inliers={m['inlier_ratio']*100:.0f}%, res={m['mean_residual']*100:.1f}cm")
            else:
                radar.warn_print(f"Landing zone UNSAFE ({m.get('reason','')}) slope={m.get('slope_deg',0):.1f}deg, inliers={m.get('inlier_ratio',0)*100:.0f}%, res={m.get('mean_residual',0)*100:.1f}cm")
    except KeyboardInterrupt:
        radar.info_print("Interrupted by user.")
    finally:
        plotter.export_history_csv()
        plotter.show_history_table()
        radar.close()


