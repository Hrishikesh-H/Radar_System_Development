from system_logger import get_logger
import numpy as np
import math

# Initialize logger for this module
logger = get_logger('LandingZoneAssessor')

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
                logger.error("det_obj is None")
                return False, {'reason': 'det_obj is None'}

            # Input validation: ensure expected keys exist
            for key in ['x', 'y', 'z', 'numObj']:
                if key not in det_obj:
                    logger.error(f"Missing key '{key}' in det_obj")
                    return False, {'reason': f"Missing key '{key}' in det_obj"}

            x = np.asarray(det_obj['x'])
            y = np.asarray(det_obj['y'])
            z = np.asarray(det_obj['z'])
            n = det_obj['numObj']

            if not (x.size and y.size and z.size):
                logger.error("x,y,z must be non-empty arrays")
                return False, {'reason': 'x,y,z must be array-like'}

            if len(x) != n or len(y) != n or len(z) != n:
                logger.error("Length mismatch in det_obj data arrays")
                return False, {'reason': 'Length mismatch in det_obj data arrays'}

            if n < 3:
                logger.error(f"Not enough points ({n}) for plane fitting")
                return False, {'reason': 'Not enough points'}

            # Create points array (reuse memory if possible)
            pts = np.column_stack((x, y, z))

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

                        # Plane equation: a*x + b_coef*y + c*z + d = 0
                        a, b_coef, c, d = m, n_coef, -1, b
                        norm_val = np.sqrt(a*a + b_coef*b_coef + c*c)
                        if norm_val > 0:
                            norm_inv = 1.0 / norm_val
                            a *= norm_inv
                            b_coef *= norm_inv
                            c *= norm_inv
                            d *= norm_inv

                        inlier_mask = ransac.inlier_mask_
                        inlier_count = np.sum(inlier_mask)
                        inlier_ratio = inlier_count / len(points)

                        # Precompute range once
                        z_range = np.ptp(points[:, 2])
                        consistency_score = 1.0 - np.std(points[inlier_mask, 2]) / (z_range + 1e-6)
                        score = inlier_ratio * 0.7 + consistency_score * 0.3

                        if score > best_score:
                            best_score = score
                            best_model = (a, b_coef, c, d)
                            best_inliers = inlier_mask

                    except Exception as e:
                        logger.debug(f"RANSAC scale {scale} failed: {str(e)}")
                        continue

                return best_model, best_inliers

            coeffs, inlier_mask = multi_scale_ransac(pts, self.distance_thresh, self.min_iterations)

            if coeffs is None or len(coeffs) < 4:
                logger.error("Multi-scale plane fitting failed")
                return False, {'reason': 'Plane fitting failed'}

            a, b_coef, c, d = coeffs
            inlier_count = np.sum(inlier_mask)
            inlier_ratio = inlier_count / n

            # Reuse plane normal for distance calculation
            normal = np.array([a, b_coef, c])
            dists = np.abs(pts.dot(normal) + d
            mean_residual = np.mean(dists[inlier_mask]) if inlier_count > 0 else np.inf

            def assess_surface_quality(points, inlier_mask, max_samples=100):
                """Assess landing surface quality with sampling for large sets"""
                inlier_points = points[inlier_mask]
                if len(inlier_points) < 3:
                    return 0.0

                # Precompute z statistics
                z_vals = inlier_points[:, 2]
                z_range = np.ptp(z_vals)
                z_std = np.std(z_vals)
                roughness_score = max(0, 1.0 - z_std / (z_range + 1e-6))

                # Density uniformity with sampling
                if len(inlier_points) > max_samples:
                    indices = np.random.choice(len(inlier_points), max_samples, replace=False)
                    sample_points = inlier_points[indices, :2]
                else:
                    sample_points = inlier_points[:, :2]

                if len(sample_points) > 1:
                    from scipy.spatial.distance import pdist
                    distances = pdist(sample_points)
                    density_uniformity = 1.0 - np.std(distances) / (np.mean(distances) + 1e-6
                    density_uniformity = max(0, min(1, density_uniformity))
                else:
                    density_uniformity = 0.5

                area_coverage = min(1.0, len(inlier_points) / max(10, n * 0.8))

                return np.mean([roughness_score, density_uniformity, area_coverage])

            surface_quality = assess_surface_quality(pts, inlier_mask)

            # Normalize plane normal (reuse existing normal)
            norm_val = np.linalg.norm(normal)
            if norm_val == 0:
                logger.error("Plane normal vector zero length")
                return False, {'reason': 'Plane normal vector zero length'}
            normal /= norm_val

            # Calculate slope angle
            vertical = np.array([0, 1, 0])
            cos_theta = abs(np.dot(normal, vertical))
            slope_deg = 90.0 - math.degrees(math.acos(min(max(cos_theta, 0.0), 1.0)))

            temporal_consistency = 1.0  # Placeholder for actual implementation

            # Safety checks
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
            logger.exception("Exception during landing zone assessment")
            return False, {'reason': f'Exception during processing: {str(e)}'}