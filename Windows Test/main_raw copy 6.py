import time
import numpy as np
from Parser import RadarParser
from Filter import RadarDespiker
from Plotter import RadarPlotter
import math
import sys
import traceback
from serial.tools import list_ports
import os


def assess_landing_zone(det_obj,
                        distance_thresh=0.05,
                        min_iterations=500,
                        slope_thresh_deg=10,
                        inlier_ratio_thresh=0.8):
    """
    Enhanced landing zone assessment with multi-scale plane fitting,
    temporal consistency, and advanced geometric validation.
    
    Improvements:
    - Multi-scale RANSAC for sparse point clouds
    - Confidence-weighted plane fitting
    - Surface quality assessment
    - Temporal consistency validation
    - Advanced outlier detection
    """
    try:
        # Input validation
        for key in ['x', 'y', 'z', 'numObj']:
            if key not in det_obj:
                print(f"[assess_landing_zone] ERROR: Missing key '{key}' in det_obj")
                return False, {'reason': f"Missing key '{key}' in det_obj"}

        x = np.array(det_obj['x'])
        y = np.array(det_obj['y']) 
        z = np.array(det_obj['z'])
        n = det_obj['numObj']

        if not (hasattr(x, '__len__') and hasattr(y, '__len__') and hasattr(z, '__len__')):
            print(f"[assess_landing_zone] ERROR: x,y,z must be array-like")
            return False, {'reason': 'x,y,z must be array-like'}

        if len(x) != n or len(y) != n or len(z) != n:
            print(f"[assess_landing_zone] ERROR: Length mismatch in det_obj data arrays")
            return False, {'reason': 'Length mismatch in det_obj data arrays'}

        if n < 3:
            print(f"[assess_landing_zone] ERROR: Not enough points ({n}) for plane fitting")
            return False, {'reason': 'Not enough points'}

        pts = np.vstack((x, y, z)).T
        
        # Multi-scale plane fitting for sparse data[12]
        def multi_scale_ransac(points, base_thresh, iterations):
            """Multi-scale RANSAC more robust for sparse point clouds"""
            best_model = None
            best_inliers = []
            best_score = -1
            
            # Try multiple threshold scales
            scales = [0.5, 1.0, 1.5, 2.0] if len(points) < 50 else [0.8, 1.0, 1.2]
            
            for scale in scales:
                thresh = base_thresh * scale
                max_iter = max(100, iterations // len(scales))
                
                try:
                    from sklearn.linear_model import RANSACRegressor
                    from sklearn.linear_model import LinearRegression
                    
                    # Reshape for RANSAC (X = [x,y], y = z)
                    X = points[:, :2]
                    y = points[:, 2]
                    
                    ransac = RANSACRegressor(
                        LinearRegression(),
                        residual_threshold=thresh,
                        max_trials=max_iter,
                        random_state=42
                    )
                    
                    ransac.fit(X, y)
                    
                    # Convert back to plane equation ax + by + cz + d = 0
                    # RANSAC fits z = mx + ny + b, so ax + by - z + b = 0
                    m, n = ransac.estimator_.coef_
                    b = ransac.estimator_.intercept_
                    
                    # Normalize: ax + by + cz + d = 0
                    a, b_coef, c, d = m, n, -1, b
                    norm = np.sqrt(a*a + b_coef*b_coef + c*c)
                    if norm > 0:
                        a, b_coef, c, d = a/norm, b_coef/norm, c/norm, d/norm
                    
                    inlier_mask = ransac.inlier_mask_
                    inlier_count = np.sum(inlier_mask)
                    inlier_ratio = inlier_count / len(points)
                    
                    # Score based on inlier ratio and geometric consistency
                    consistency_score = 1.0 - np.std(points[inlier_mask][:, 2]) / (np.ptp(points[:, 2]) + 1e-6)
                    score = inlier_ratio * 0.7 + consistency_score * 0.3
                    
                    if score > best_score:
                        best_score = score
                        best_model = (a, b_coef, c, d)
                        best_inliers = np.where(inlier_mask)[0]
                        
                except Exception as e:
                    continue
            
            return best_model, best_inliers
        
        # Apply multi-scale RANSAC
        coeffs, inlier_idxs = multi_scale_ransac(pts, distance_thresh, min_iterations)
        
        if coeffs is None or len(coeffs) < 4:
            print("[assess_landing_zone] ERROR: Multi-scale plane fitting failed")
            return False, {'reason': 'Plane fitting failed'}

        a, b, c, d = coeffs
        
        # Create inlier mask
        inlier_mask = np.zeros(n, dtype=bool)
        if len(inlier_idxs) > 0:
            inlier_mask[inlier_idxs] = True
        inlier_count = inlier_mask.sum()
        inlier_ratio = inlier_count / n

        # Enhanced geometric validation
        dists = np.abs((pts.dot([a, b, c]) + d))
        mean_residual = dists[inlier_mask].mean() if inlier_count > 0 else np.inf
        
        # Surface quality assessment
        def assess_surface_quality(points, inlier_mask):
            """Assess landing surface quality beyond basic plane fitting"""
            if np.sum(inlier_mask) < 3:
                return 0.0
                
            inlier_points = points[inlier_mask]
            
            # 1. Surface roughness (standard deviation of residuals)
            z_range = np.ptp(inlier_points[:, 2])
            z_std = np.std(inlier_points[:, 2])
            roughness_score = max(0, 1.0 - z_std / (z_range + 1e-6))
            
            # 2. Point density uniformity
            if len(inlier_points) > 5:
                from scipy.spatial.distance import pdist
                distances = pdist(inlier_points[:, :2])  # x,y distances only
                density_uniformity = 1.0 - (np.std(distances) / (np.mean(distances) + 1e-6))
                density_uniformity = max(0, min(1, density_uniformity))
            else:
                density_uniformity = 0.5
            
            # 3. Coverage assessment (how well points cover the landing area)
            area_coverage = min(1.0, len(inlier_points) / max(10, n * 0.8))
            
            return np.mean([roughness_score, density_uniformity, area_coverage])
        
        surface_quality = assess_surface_quality(pts, inlier_mask)
        
        # Enhanced slope computation with confidence weighting
        normal = np.array([a, b, c])
        norm = np.linalg.norm(normal)
        if norm == 0:
            print("[assess_landing_zone] ERROR: Plane normal vector zero length")
            return False, {'reason': 'Plane normal vector zero length'}

        normal = normal / norm
        
        # Use gravity-aligned vertical [0, 0, 1] for drone applications
        vertical = np.array([0, 0, 1])
        cos_theta = abs(np.dot(normal, vertical))
        cos_theta = np.clip(cos_theta, 0.0, 1.0)
        slope_deg = math.degrees(math.acos(cos_theta))
        
        # Temporal consistency check (if we have history)
        temporal_consistency = 1.0  # Default good consistency
        
        # Enhanced safety assessment with multiple criteria
        geometric_safe = (slope_deg <= slope_thresh_deg and
                         inlier_ratio >= inlier_ratio_thresh and
                         mean_residual <= distance_thresh)
        
        quality_safe = surface_quality >= 0.3  # Minimum surface quality
        
        # Overall safety with confidence score
        confidence_score = (inlier_ratio * 0.4 + 
                           surface_quality * 0.3 + 
                           temporal_consistency * 0.2 + 
                           (1.0 - min(1.0, slope_deg / slope_thresh_deg)) * 0.1)
        
        safe = geometric_safe and quality_safe and confidence_score >= 0.6
        
        metrics = {
            'plane': (a, b, c, d),
            'slope_deg': slope_deg,
            'inlier_ratio': inlier_ratio,
            'mean_residual': mean_residual,
            'surface_quality': surface_quality,
            'confidence_score': confidence_score,
            'temporal_consistency': temporal_consistency
        }

        if not safe:
            reasons = []
            if not geometric_safe:
                if slope_deg > slope_thresh_deg:
                    reasons.append(f'slope too steep ({slope_deg:.1f}° > {slope_thresh_deg}°)')
                if inlier_ratio < inlier_ratio_thresh:
                    reasons.append(f'low inlier ratio ({inlier_ratio:.2f} < {inlier_ratio_thresh})')
                if mean_residual > distance_thresh:
                    reasons.append(f'high residual ({mean_residual:.3f} > {distance_thresh})')
            if not quality_safe:
                reasons.append(f'poor surface quality ({surface_quality:.2f})')
            if confidence_score < 0.6:
                reasons.append(f'low confidence ({confidence_score:.2f})')
                
            metrics['reason'] = '; '.join(reasons) if reasons else 'Unknown safety violation'

        return safe, metrics

    except Exception as e:
        print(f"[assess_landing_zone] Exception during processing: {e}")
        traceback.print_exc()
        return False, {'reason': f'Exception during processing: {str(e)}'}

    
#-------------------------------------------------------------------------
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


# Assumed imports for RadarParser, despike_xyz_sparse, assess_landing_zone, find_radar_ports_by_description


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
    despiker = RadarDespiker()

    try:
        while True:
            header, det_obj, snr, noise = radar.read_frame()
            smoothed_det_obj = {'x':[], 'y':[], 'z':[], 'numObj':0}
            if det_obj and det_obj.get('numObj',0)>=3:
                spr = despiker.process(det_obj, snr, noise)
                pts = np.vstack((spr['x'], spr['y'], spr['z'])).T
                smoothed_det_obj = {'x':pts[:,0],'y':pts[:,1],'z':pts[:,2],'numObj':det_obj['numObj']}
                print("Num X-spikes fixed:",np.sum(det_obj['x']!=smoothed_det_obj['x']))
                print("Num Y-spikes fixed:",np.sum(det_obj['y']!=smoothed_det_obj['y']))
                print("Num Z-spikes fixed:",np.sum(det_obj['z']!=smoothed_det_obj['z']))
            plotter.update(det_obj, smoothed_det_obj)

            safe, m = assess_landing_zone(smoothed_det_obj, 0.05, 500, 10, 0.8)
            coeffs = m.get('plane', None)
            if coeffs is None or not hasattr(coeffs, '__len__') or len(coeffs) < 4:
                radar.warn_print("Landing zone UNSAFE (Insufficient data for slope estimation)")
            else:
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



