import time
import numpy as np
from Parser import RadarParser
from pyransac3d import Plane
import math

from serial.tools import list_ports

import numpy as np
from scipy.ndimage import median_filter

import numpy as np
from scipy.ndimage import median_filter

def smooth_point_cloud(points: np.ndarray, size: int = 3, radar=None) -> np.ndarray:
    """
    Apply a median filter to a 3D point cloud to reduce noise and normalize spikes.

    Parameters:
        points (np.ndarray): A numpy array of shape (N, 3) representing 3D points.
        size (int): Size of the median filter kernel (default is 3).
        radar: Logger object with debug_print, warn_print, and error_print methods.

    Returns:
        np.ndarray: Smoothed point cloud of shape (N, 3). If input is invalid, returns original.
    """
    if not isinstance(points, np.ndarray):
        radar.error_print("Point cloud input is not a numpy array. Returning original.")
        return points

    if points.ndim != 2 or points.shape[1] != 3:
        radar.error_print(f"Invalid point cloud shape: {points.shape}. Expected (N, 3). Returning original.")
        return points

    if size < 1 or size % 2 == 0:
        radar.warn_print(f"Median filter size must be a positive odd integer. Got {size}. Using default size 3.")
        size = 3

    radar.debug_print(f"Smoothing point cloud with median filter size {size}")
    smoothed = np.empty_like(points)
    for i in range(3):  # Apply median filter to x, y, z independently
        smoothed[:, i] = median_filter(points[:, i], size=size, mode='nearest')

    radar.debug_print("Point cloud smoothing complete.")
    return smoothed



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

if __name__ == "__main__":
    cli, data = None, None  # Safeguard initialization

    try:
        cli, data = find_radar_ports_by_description("CP2105")
        print(f"CLI port: {cli}, DATA port: {data}")
    except RuntimeError as e:
        print(f"Port detection error: {e}")
        exit(1)  # Exit if ports not found

    # Fix config path to be OS-independent
    config_path = os.path.join(os.getcwd(), "best_res_4cm.cfg")

    radar = RadarParser(
        cli_port=cli,
        data_port=data,
        config_file=config_path,  # Use fixed full path
        debug=False,
        enable_logging=True,
        log_prefix="radar_log"
    )

    radar.initialize_ports()
    radar.send_config()
    time.sleep(2)

    radar.info_print("Listening for radar data...")

    try:
        while True:
            header, det_obj, snr, noise = radar.read_frame()
            if det_obj is not None and snr is not None:
                idx = np.argmax(snr)
                alt = det_obj["y"][idx]
                radar.info_print(f"Altitude from highest SNR object: {alt:.2f} m (SNR: {snr[idx]:.1f} dB)")

            safe, m = False, {'reason': 'no data'}

            if det_obj is not None and det_obj.get('numObj', 0) >= 3:
                # Convert dict to point array first for smoothing
                x, y, z = det_obj['x'], det_obj['y'], det_obj['z']
                pts = np.vstack((x, y, z)).T
                smoothed_pts = smooth_point_cloud(pts)
                
                # Create new det_obj with smoothed values
                smoothed_det_obj = {
                    'x': smoothed_pts[:, 0],
                    'y': smoothed_pts[:, 1],
                    'z': smoothed_pts[:, 2],
                    'numObj': det_obj['numObj']
                }

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
