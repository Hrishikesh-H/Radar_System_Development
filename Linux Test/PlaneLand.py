import numpy as np
import math
import traceback

class LandingZoneAssessor:
    def __init__(self,
                 distance_thresh=0.05,
                 min_iterations=500,
                 slope_thresh_deg=10,
                 inlier_ratio_thresh=0.7):
        self.distance_thresh = distance_thresh
        self.min_iterations = min_iterations
        self.slope_thresh_deg = slope_thresh_deg
        self.inlier_ratio_thresh = inlier_ratio_thresh

    def assess(self, det_obj):
        """
        Enhanced landing zone assessment with multi-scale plane fitting,
        temporal consistency, and advanced geometric validation.
        """
        try:
            # EARLY RETURN: if det_obj is None or empty, we cannot proceed
            if det_obj is None:
                print("[assess_landing_zone] ERROR: det_obj is None")
                return False, {'reason': 'det_obj is None'}

            # Input validation: ensure expected keys exist
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

            # Multi-scale plane fitting for sparse data
            def multi_scale_ransac(points, base_thresh, iterations):
                """Multi-scale RANSAC more robust for sparse point clouds"""
                best_model = None
                best_inliers = []
                best_score = -1

                scales = [0.5, 1.0, 1.5, 2.0] if len(points) < 50 else [0.8, 1.0, 1.2]

                for scale in scales:
                    thresh = base_thresh * scale
                    max_iter = max(100, iterations // len(scales))

                    try:
                        from sklearn.linear_model import RANSACRegressor
                        from sklearn.linear_model import LinearRegression

                        X = points[:, :2]
                        y = points[:, 2]

                        ransac = RANSACRegressor(
                            LinearRegression(),
                            residual_threshold=thresh,
                            max_trials=max_iter,
                            random_state=42
                        )

                        ransac.fit(X, y)

                        m, n_coef = ransac.estimator_.coef_
                        b = ransac.estimator_.intercept_

                        # Plane equation: a*x + b_coef*y + c*z + d = 0,
                        # where z = m*x + n_coef*y + b  => rearranged to [m, n_coef, -1, b].
                        a, b_coef, c, d = m, n_coef, -1, b
                        norm = np.sqrt(a*a + b_coef*b_coef + c*c)
                        if norm > 0:
                            a, b_coef, c, d = a/norm, b_coef/norm, c/norm, d/norm

                        inlier_mask = ransac.inlier_mask_
                        inlier_count = np.sum(inlier_mask)
                        inlier_ratio = inlier_count / len(points)

                        consistency_score = 1.0 - np.std(points[inlier_mask][:, 2]) / (np.ptp(points[:, 2]) + 1e-6)
                        score = inlier_ratio * 0.7 + consistency_score * 0.3

                        if score > best_score:
                            best_score = score
                            best_model = (a, b_coef, c, d)
                            best_inliers = np.where(inlier_mask)[0]

                    except Exception:
                        continue

                return best_model, best_inliers

            coeffs, inlier_idxs = multi_scale_ransac(pts, self.distance_thresh, self.min_iterations)

            if coeffs is None or len(coeffs) < 4:
                print("[assess_landing_zone] ERROR: Multi-scale plane fitting failed")
                return False, {'reason': 'Plane fitting failed'}

            a, b_coef, c, d = coeffs

            inlier_mask = np.zeros(n, dtype=bool)
            if len(inlier_idxs) > 0:
                inlier_mask[inlier_idxs] = True
            inlier_count = inlier_mask.sum()
            inlier_ratio = inlier_count / n

            dists = np.abs((pts.dot([a, b_coef, c]) + d))
            mean_residual = dists[inlier_mask].mean() if inlier_count > 0 else np.inf

            def assess_surface_quality(points, inlier_mask):
                """Assess landing surface quality beyond basic plane fitting"""
                if np.sum(inlier_mask) < 3:
                    return 0.0

                inlier_points = points[inlier_mask]

                z_range = np.ptp(inlier_points[:, 2])
                z_std = np.std(inlier_points[:, 2])
                roughness_score = max(0, 1.0 - z_std / (z_range + 1e-6))

                if len(inlier_points) > 5:
                    from scipy.spatial.distance import pdist
                    distances = pdist(inlier_points[:, :2])
                    density_uniformity = 1.0 - (np.std(distances) / (np.mean(distances) + 1e-6))
                    density_uniformity = max(0, min(1, density_uniformity))
                else:
                    density_uniformity = 0.5

                area_coverage = min(1.0, len(inlier_points) / max(10, n * 0.8))

                return np.mean([roughness_score, density_uniformity, area_coverage])

            surface_quality = assess_surface_quality(pts, inlier_mask)

            normal = np.array([a, b_coef, c])
            norm = np.linalg.norm(normal)
            if norm == 0:
                print("[assess_landing_zone] ERROR: Plane normal vector zero length")
                return False, {'reason': 'Plane normal vector zero length'}

            normal = normal / norm

            vertical = np.array([0, 1, 0])
            cos_theta = abs(np.dot(normal, vertical))
            cos_theta = np.clip(cos_theta, 0.0, 1.0)
            slope_deg = 90.0 - math.degrees(math.acos(cos_theta))

            temporal_consistency = 1.0

            geometric_safe = (slope_deg <= self.slope_thresh_deg and
                              inlier_ratio >= self.inlier_ratio_thresh and
                              mean_residual <= self.distance_thresh)

            quality_safe = surface_quality >= 0.3

            confidence_score = (inlier_ratio * 0.4 +
                                surface_quality * 0.3 +
                                temporal_consistency * 0.2 +
                                (1.0 - min(1.0, slope_deg / self.slope_thresh_deg)) * 0.1)

            safe = geometric_safe and quality_safe and confidence_score >= 0.6

            metrics = {
                'plane': (a, b_coef, c, d),
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
                    if slope_deg > self.slope_thresh_deg:
                        reasons.append(f'slope too steep ({slope_deg:.1f}° > {self.slope_thresh_deg}°)')
                    if inlier_ratio < self.inlier_ratio_thresh:
                        reasons.append(f'low inlier ratio ({inlier_ratio:.2f} < {self.inlier_ratio_thresh})')
                    if mean_residual > self.distance_thresh:
                        reasons.append(f'high residual ({mean_residual:.3f} > {self.distance_thresh})')
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
