import time
import numpy as np
from Parser import RadarParser
from pyransac3d import Plane
import math
import sys

from serial.tools import list_ports
from scipy.spatial import KDTree

def despike_xyz_sparse(det_obj, radius=0.5, z_thresh=0.3, xy_thresh=0.3):
    """
    Corrects spikes in sparse 3D radar data using neighbor-based smoothing.
    
    Args:
      det_obj (dict): {'x', 'y', 'z', 'numObj'} with numpy arrays
      radius (float): Neighborhood radius (meters)
      z_thresh (float): Max z deviation to consider as spike
      xy_thresh (float): Max x/y deviation to consider as spike
      
    Returns:
      filtered_obj (dict): corrected 'x','y','z','numObj'
    """
    x, y, z = det_obj['x'], det_obj['y'], det_obj['z']
    pts = np.vstack((x, y, z)).T
    tree = KDTree(pts)
    
    x_f, y_f, z_f = x.copy(), y.copy(), z.copy()

    for i in range(len(pts)):
        idxs = tree.query_ball_point(pts[i], r=radius)
        if len(idxs) < 3:
            continue  # not enough neighbors

        neigh_x = x[idxs]
        neigh_y = y[idxs]
        neigh_z = z[idxs]

        # Check each dimension
        dx, dy, dz = abs(x[i] - np.median(neigh_x)), abs(y[i] - np.median(neigh_y)), abs(z[i] - np.median(neigh_z))

        if dx > xy_thresh:
            x_f[i] = np.median(neigh_x)
        if dy > xy_thresh:
            y_f[i] = np.median(neigh_y)
        if dz > z_thresh:
            z_f[i] = np.median(neigh_z)

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

# ...

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

# Your existing RadarParser and helper functions assumed imported here
# e.g., from radar_parser import RadarParser, despike_xyz_sparse, assess_landing_zone, find_radar_ports_by_description

if __name__ == "__main__":
    cli, data = None, None  # Safeguard initialization

    try:
        cli, data = find_radar_ports_by_description("CP2105")
        print(f"CLI port: {cli}, DATA port: {data}")
    except RuntimeError as e:
        print(f"Port detection error: {e}")
        sys.exit(1)  # Exit if ports not found

    # Fix config path to be OS-independent
    config_path = os.path.join(os.getcwd(), "best_res_4cm.cfg")

    radar = RadarParser(
        cli_port=cli,
        data_port=data,
        config_file=config_path,  # Use fixed full path
        debug=False,
        enable_logging=False,
        log_prefix="radar_log"
    )

    radar.initialize_ports()
    radar.send_config()
    time.sleep(2)

    radar.info_print("Listening for radar data...")

    # Initialize interactive plotting
    plt.ion()
    fig_real, ax_real = plt.subplots()
    ax_real.set_title('Real-time Radar View (raw vs smoothed)')
    ax_real.set_xlabel('X (m)')
    ax_real.set_ylabel('Y (m)')
    ax_real.set_aspect('equal', 'box')

    fig_hist, ax_hist = plt.subplots()
    ax_hist.set_title('History of All Data Points')
    ax_hist.set_xlabel('X (m)')
    ax_hist.set_ylabel('Y (m)')
    ax_hist.set_aspect('equal', 'box')

    # Lists to accumulate history
    history_raw = []       # List of (N,2) arrays
    history_smoothed = []  # List of (M,2) arrays

    try:
        while True:
            header, det_obj, snr, noise = radar.read_frame()

            # Prepare empty smoothed output
            smoothed_det_obj = {'x': [], 'y': [], 'z': [], 'numObj': 0}

            if det_obj is not None and det_obj.get('numObj', 0) >= 3:
                x, y, z = det_obj['x'], det_obj['y'], det_obj['z']
                pts = np.vstack((x, y, z)).T

                smoothed_pts_raw = despike_xyz_sparse(det_obj)
                smoothed_pts = np.vstack((smoothed_pts_raw['x'], smoothed_pts_raw['y'], smoothed_pts_raw['z'])).T

                smoothed_det_obj = {
                    'x': smoothed_pts[:, 0],
                    'y': smoothed_pts[:, 1],
                    'z': smoothed_pts[:, 2],
                    'numObj': det_obj['numObj']
                }

                # Print spike counts
                print("Num X-spikes fixed:", np.sum(det_obj['x'] != smoothed_det_obj['x']))
                print("Num Y-spikes fixed:", np.sum(det_obj['y'] != smoothed_det_obj['y']))
                print("Num Z-spikes fixed:", np.sum(det_obj['z'] != smoothed_det_obj['z']))

                # Update history lists
                history_raw.append(np.vstack((x, y)).T)
                history_smoothed.append(np.vstack((smoothed_det_obj['x'], smoothed_det_obj['y'])).T)
            else:
                # Denote no data in real-time view
                ax_real.clear()
                ax_real.text(0.5, 0.5, 'No data', transform=ax_real.transAxes,
                              ha='center', va='center', fontsize=14, color='red')
                fig_real.canvas.draw()
                fig_real.canvas.flush_events()
                continue  # Skip plotting history for this frame

            # Real-time plot
            ax_real.clear()
            ax_real.scatter(x, y, c='red', label='Raw', alpha=0.6)
            ax_real.scatter(smoothed_det_obj['x'], smoothed_det_obj['y'], c='green', label='Smoothed', alpha=0.6)
            ax_real.legend(loc='upper right')
            ax_real.set_xlim(-5, 5)  # Adjust ranges as needed
            ax_real.set_ylim(-5, 5)
            fig_real.canvas.draw()
            fig_real.canvas.flush_events()

            # History plot (append and redraw)
            ax_hist.clear()
            # Plot all raw history points faintly
            for pts2d in history_raw:
                ax_hist.scatter(pts2d[:, 0], pts2d[:, 1], alpha=0.1)
            # Plot all smoothed history points distinctly
            for pts2d in history_smoothed:
                ax_hist.scatter(pts2d[:, 0], pts2d[:, 1], alpha=0.3, marker='x')
            ax_hist.set_xlim(-5, 5)
            ax_hist.set_ylim(-5, 5)
            fig_hist.canvas.draw()
            fig_hist.canvas.flush_events()

            # Assess landing zone as before
            safe, m = assess_landing_zone(
                smoothed_det_obj,
                distance_thresh=0.05,
                min_iterations=500,
                slope_thresh_deg=10,
                inlier_ratio_thresh=0.8
            )

            if safe:
                radar.info_print(
                    f"Landing zone SAFE slope={m['slope_deg']:.1f} deg, "
                    f"inliers={m['inlier_ratio']*100:.0f}%, "
                    f"res={m['mean_residual']*100:.1f}cm"
                )
            else:
                reason = m.get('reason', 'metrics did not meet thresholds')
                radar.warn_print(
                    f"Landing zone UNSAFE ({reason}): "
                    f"slope={m.get('slope_deg', 0):.1f} deg, "
                    f"inliers={m.get('inlier_ratio', 0)*100:.0f}%, "
                    f"res={m.get('mean_residual', 0)*100:.1f}cm"
                )

    except KeyboardInterrupt:
        radar.info_print("Interrupted by user.")
    finally:
        radar.close()
