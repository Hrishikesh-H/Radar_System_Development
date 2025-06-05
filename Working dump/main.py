import time
import numpy as np
from Parser import RadarParser
from pyransac3d import Plane
import math

def assess_landing_zone(det_obj,
                        distance_thresh=0.05,
                        min_iterations=500,
                        slope_thresh_deg=10,
                        inlier_ratio_thresh=0.8,
                        vertical_axis='y'):
    """
    Given det_obj from RadarParser, fit a plane and decide landing safety.
    
    Args:
      det_obj (dict): must contain 'x','y','z' numpy arrays and 'numObj'
      distance_thresh (float): RANSAC distance threshold (m)
      min_iterations (int): RANSAC trials
      slope_thresh_deg (float): max allowable tilt from vertical (degrees)
      inlier_ratio_thresh (float): min fraction of points fitting plane
      vertical_axis (str): 'x', 'y', or 'z' indicating which is up
      
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

    # Debug: show first 5 points
    print("[DEBUG] Sample points:")
    for i in range(min(5, n)):
        print(f"  x={x[i]:.2f}, y={y[i]:.2f}, z={z[i]:.2f}")

    # 2) fit plane
    model = Plane()
    coeffs, inlier_idxs = model.fit(pts,
                                    thresh=distance_thresh,
                                    maxIteration=min_iterations)
    a, b, c, d = coeffs
    print(f"[DEBUG] Plane normal vector: a={a:.3f}, b={b:.3f}, c={c:.3f}")

    inlier_mask = np.zeros(n, dtype=bool)
    inlier_mask[inlier_idxs] = True
    inlier_count = inlier_mask.sum()
    inlier_ratio = inlier_count / n

    # 3) compute residuals for inliers
    dists = np.abs((pts.dot([a, b, c]) + d))  # unsigned distance
    mean_residual = dists[inlier_mask].mean() if inlier_count > 0 else np.inf

    # 4) slope: angle between normal and vertical axis
    normal = np.array([a, b, c])
    normal = normal / np.linalg.norm(normal)

    if vertical_axis == 'y':
        vertical = np.array([0, 1, 0])
    elif vertical_axis == 'z':
        vertical = np.array([0, 0, 1])
    elif vertical_axis == 'x':
        vertical = np.array([1, 0, 0])
    else:
        raise ValueError(f"Invalid vertical_axis '{vertical_axis}', must be 'x', 'y', or 'z'")

    cos_theta = abs(np.dot(normal, vertical))
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


if __name__ == "__main__":
    radar = RadarParser(
        cli_port="/dev/ttyUSB0",
        data_port="/dev/ttyUSB1",
        config_file="best_res_4cm.cfg",
        debug=False,
        enable_logging=False,
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

            # default values
            safe, m = False, {'reason': 'no data'}

            # only call the assessor if we actually have =3 points
            if det_obj is not None and det_obj.get('numObj', 0) >= 3:
                safe, m = assess_landing_zone(
                    det_obj,
                    distance_thresh=0.05,
                    min_iterations=500,
                    slope_thresh_deg=10,
                    inlier_ratio_thresh=0.8,
                    vertical_axis='y'  # change to 'z' or 'x' if needed
                )

            # now safe and m are always initialized
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
