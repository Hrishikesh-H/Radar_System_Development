from system_logger import get_logger
import numpy as np
import math
from sklearn.linear_model import RANSACRegressor, LinearRegression

logger = get_logger('LandingZoneAssessor')


class PlaneDetector:
    """Robust plane detector with multi-scale RANSAC for sparse, noisy point clouds"""

    def __init__(self, distance_thresh=0.05, min_iterations=100):
        self.distance_thresh = distance_thresh
        self.min_iterations = min_iterations

    def fit_plane(self, points, weights=None):
        """Multi-scale RANSAC plane fitting optimized for sparse data"""
        n = points.shape[0]
        if n < 3:
            return None, None, float('inf')

        # Use multi-scale approach for sparse clouds
        scales = [0.7, 1.0, 1.3] if n < 15 else [0.9, 1.0, 1.1]
        best_model = None
        best_inliers = None
        best_score = -np.inf

        for scale in scales:
            thresh = self.distance_thresh * scale
            max_iter = max(100, self.min_iterations // len(scales))

            try:
                X = points[:, :2]
                y = points[:, 2]

                ransac = RANSACRegressor(
                    LinearRegression(),
                    residual_threshold=thresh,
                    max_trials=max_iter,
                    random_state=42
                )

                ransac.fit(X, y)

                # Extract plane parameters
                m, b_coef = ransac.estimator_.coef_
                b_val = ransac.estimator_.intercept_
                normal = np.array([m, b_coef, -1])
                norm_val = np.linalg.norm(normal)
                normal /= norm_val
                d = b_val / norm_val

                # Calculate inliers
                dists = np.abs(points @ normal + d)
                inlier_mask = dists <= thresh
                inlier_count = np.sum(inlier_mask)

                # Calculate score (prioritize inlier count and low residual)
                if inlier_count > 0:
                    mean_residual = np.mean(dists[inlier_mask])
                    score = inlier_count / (mean_residual + 1e-6)

                    if score > best_score:
                        best_score = score
                        best_model = (normal, d)
                        best_inliers = inlier_mask
            except Exception as e:
                logger.debug(f"RANSAC scale {scale} failed: {str(e)}")
                continue

        if best_model is None:
            return None, None, float('inf')

        # Final evaluation with original threshold
        normal, d = best_model
        dists = np.abs(points @ normal + d)
        inlier_mask = dists <= self.distance_thresh
        mean_residual = np.mean(dists[inlier_mask]) if np.any(inlier_mask) else float('inf')

        return best_model, inlier_mask, mean_residual


class LandingZoneAssessor:
    def __init__(self,
                 distance_thresh=0.05,
                 slope_thresh_deg=10,
                 inlier_ratio_thresh=0.7,
                 min_plane_points=5,
                 surface_quality_thresh=0.3):
        self.distance_thresh = distance_thresh
        self.slope_thresh_deg = slope_thresh_deg
        self.inlier_ratio_thresh = inlier_ratio_thresh
        self.min_plane_points = min_plane_points
        self.surface_quality_thresh = surface_quality_thresh
        self.plane_detector = PlaneDetector(distance_thresh)

    def assess_surface_quality(self, points, inlier_mask, max_samples=100):
        """Optimized surface quality assessment for sparse clouds"""
        inlier_points = points[inlier_mask]
        if len(inlier_points) < 3:
            return 0.0

        # Roughness assessment
        z_vals = inlier_points[:, 2]
        z_range = np.ptp(z_vals)
        z_std = np.std(z_vals)
        roughness_score = max(0, 1.0 - z_std / (z_range + 1e-6))

        # Density assessment (simplified for sparse clouds)
        if len(inlier_points) > 3:
            # Use minimum bounding circle diameter as area proxy
            from scipy.spatial import ConvexHull
            try:
                hull = ConvexHull(inlier_points[:, :2])
                area = hull.volume  # Actually area for 2D
                density_score = min(1.0, len(inlier_points) / (area + 1e-6))
            except:
                density_score = 0.5
        else:
            density_score = 0.5

        return (roughness_score + density_score) / 2

    def assess(self, det_obj, plane_params=None, inlier_mask=None):
        """
        Landing zone assessment optimized for sparse radar point clouds
        """
        try:
            # Input validation
            if not det_obj or 'x' not in det_obj or 'y' not in det_obj or 'z' not in det_obj:
                logger.error("Invalid det_obj structure")
                return False, {'reason': 'Invalid det_obj structure'}

            x = np.asarray(det_obj['x'])
            y = np.asarray(det_obj['y'])
            z = np.asarray(det_obj['z'])
            n = len(x)

            if n < self.min_plane_points:
                logger.error(f"Not enough points ({n}) for assessment")
                return False, {'reason': 'Not enough points'}

            pts = np.column_stack((x, y, z))
            weights = det_obj.get('quality_weights', np.ones(n))

            # Use precomputed plane if available
            if plane_params is not None and inlier_mask is not None:
                normal, d = plane_params
                # Calculate point-to-plane distances
                dists = np.abs(pts @ normal + d)
                inlier_mask = dists <= self.distance_thresh
                mean_residual = np.mean(dists[inlier_mask]) if np.any(inlier_mask) else float('inf')
            else:
                # Fit new plane with multi-scale RANSAC
                result = self.plane_detector.fit_plane(pts, weights)
                if result[0] is None:
                    logger.error("Plane fitting failed")
                    return False, {'reason': 'Plane fitting failed'}
                (normal, d), inlier_mask, mean_residual = result

            # Calculate inlier ratio
            inlier_count = np.sum(inlier_mask)
            inlier_ratio = inlier_count / n

            # Calculate slope angle
            vertical = np.array([0, 1, 0])  # Y-axis is vertical
            cos_theta = abs(np.dot(normal, vertical))
            slope_deg = 90.0 - math.degrees(math.acos(min(max(cos_theta, 0.0), 1.0)))

            # Surface quality assessment
            surface_quality = self.assess_surface_quality(pts, inlier_mask)

            # Temporal consistency placeholder
            temporal_consistency = 1.0

            # Safety checks
            geometric_safe = (slope_deg <= self.slope_thresh_deg and
                              inlier_ratio >= self.inlier_ratio_thresh and
                              mean_residual <= self.distance_thresh)

            quality_safe = surface_quality >= self.surface_quality_thresh

            confidence_score = (inlier_ratio * 0.5 +
                                surface_quality * 0.3 +
                                temporal_consistency * 0.2)

            safe = geometric_safe and quality_safe and confidence_score >= 0.6

            # Prepare metrics
            metrics = {
                'plane': (normal[0], normal[1], normal[2], d),
                'slope_deg': slope_deg,
                'inlier_ratio': inlier_ratio,
                'mean_residual': mean_residual,
                'surface_quality': surface_quality,
                'confidence_score': confidence_score,
                'temporal_consistency': temporal_consistency
            }

            if not safe:
                reasons = []
                if slope_deg > self.slope_thresh_deg:
                    reasons.append(f'slope too steep ({slope_deg:.1f}° > {self.slope_thresh_deg}°)')
                if inlier_ratio < self.inlier_ratio_thresh:
                    reasons.append(f'low inlier ratio ({inlier_ratio:.2f} < {self.inlier_ratio_thresh})')
                if mean_residual > self.distance_thresh:
                    reasons.append(f'high residual ({mean_residual:.3f} > {self.distance_thresh})')
                if surface_quality < self.surface_quality_thresh:
                    reasons.append(f'poor surface quality ({surface_quality:.2f} < {self.surface_quality_thresh})')
                if confidence_score < 0.6:
                    reasons.append(f'low confidence ({confidence_score:.2f})')

                metrics['reason'] = '; '.join(reasons)

            return safe, metrics

        except Exception as e:
            logger.exception("Assessment error")
            return False, {'reason': f'Processing error: {str(e)}'}