import numpy as np
from system_logger import get_logger
from plane_land import PlaneDetector


class RadarDespiker:
    def __init__(self, radius=0.5, snr_gate=7.0, distance_thresh=0.05):
        self.radius = radius
        self.snr_gate = snr_gate
        self.distance_thresh = distance_thresh
        self.plane_detector = PlaneDetector(distance_thresh)
        self.logger = get_logger("RadarDespiker")

    def statistical_outlier_removal(self, points, k_neighbors=8, std_multiplier=2.0):
        n_points = len(points)
        if n_points < k_neighbors:
            return np.ones(n_points, dtype=bool)

        try:
            from sklearn.neighbors import NearestNeighbors
            nbrs = NearestNeighbors(n_neighbors=min(k_neighbors, n_points - 1)).fit(points)
            distances, _ = nbrs.kneighbors(points)
            mean_distances = np.mean(distances[:, 1:], axis=1)
            mean_dist = np.mean(mean_distances)
            std_dist = np.std(mean_distances)
            threshold = mean_dist + std_multiplier * std_dist
            return mean_distances <= threshold
        except ImportError:
            self.logger.error("scikit-learn not available, skipping statistical outlier removal")
            return np.ones(n_points, dtype=bool)

    def radius_outlier_removal(self, points, radius_thresh=0.3, min_neighbors=3):
        n_points = len(points)
        if n_points < min_neighbors:
            return np.ones(n_points, dtype=bool)

        try:
            from sklearn.neighbors import NearestNeighbors
            nbrs = NearestNeighbors(radius=radius_thresh).fit(points)
            neighbor_counts = np.array([len(indices) - 1 for indices in nbrs.radius_neighbors(points)[1]])
            return neighbor_counts >= min_neighbors
        except ImportError:
            self.logger.error("scikit-learn not available, skipping radius outlier removal")
            return np.ones(n_points, dtype=bool)

    def process(self, det_obj, snr=None, noise=None):
        # Flatten input arrays to ensure 1D
        x = np.asarray(det_obj['x']).flatten()
        y = np.asarray(det_obj['y']).flatten()
        z = np.asarray(det_obj['z']).flatten()

        num = det_obj.get('numObj')
        curr_len = len(x)

        if curr_len == 0:
            return {'x': x, 'y': y, 'z': z, 'numObj': num}

        # Process SNR mask
        snr_arr = np.asarray(snr) if snr is not None else None
        if snr_arr is not None and len(snr_arr) == curr_len:
            snr_mask = snr_arr >= self.snr_gate
            valid_count = np.sum(snr_mask)
            if valid_count < max(3, curr_len // 4):
                top_n = max(3, curr_len // 2)
                n = len(snr_arr)
                if top_n <= 0:
                    top_indices = np.array([], dtype=int)
                elif top_n >= n:
                    top_indices = np.arange(n)
                else:
                    kth_index = n - top_n
                    partitioned_indices = np.argpartition(snr_arr, kth_index)
                    top_indices = partitioned_indices[kth_index:]
                snr_mask = np.zeros(curr_len, dtype=bool)
                snr_mask[top_indices] = True
        else:
            snr_mask = np.ones(curr_len, dtype=bool)

        # Create quality weights
        quality_weights = np.ones(curr_len)
        if snr_arr is not None:
            quality_weights = np.maximum(snr_arr - self.snr_gate, 0)
            if np.sum(quality_weights) > 0:
                quality_weights /= np.max(quality_weights)

        # Dynamic threshold scaling
        point_density = curr_len / (np.ptp(x) * np.ptp(y) * np.ptp(z) + 1e-6)
        density_factor = min(1.0, max(0.2, 1.0 - np.log1p(point_density) / 5.0))
        adaptive_radius = max(0.1, min(self.radius, self.radius * density_factor))

        # Outlier removal
        points_3d = np.column_stack([x, y, z])
        sor_mask = self.statistical_outlier_removal(points_3d, std_multiplier=1.5 + density_factor)
        ror_mask = self.radius_outlier_removal(points_3d,
                                               radius_thresh=adaptive_radius,
                                               min_neighbors=max(1, int(3 * density_factor)))
        valid_mask = snr_mask & sor_mask & ror_mask

        # Ensure sufficient valid points
        if np.sum(valid_mask) < max(3, curr_len // 4):
            sort_arr = snr_arr if snr_arr is not None and len(snr_arr) == curr_len else np.ones(curr_len)
            top_n = max(3, curr_len // 2)
            n = len(sort_arr)
            if top_n <= 0:
                top_indices = np.array([], dtype=int)
            elif top_n >= n:
                top_indices = np.arange(n)
            else:
                kth_index = n - top_n
                partitioned_indices = np.argpartition(sort_arr, kth_index)
                top_indices = partitioned_indices[kth_index:]
            valid_mask = np.zeros(curr_len, dtype=bool)
            valid_mask[top_indices] = True

        # Plane detection for multipath rejection
        plane_params, plane_inlier_mask, _ = None, None, None
        if np.sum(valid_mask) >= 3:
            try:
                # Ensure weights are 1D
                weights = quality_weights[valid_mask].flatten()

                plane_params, plane_inlier_mask, _ = self.plane_detector.fit_plane(
                    points_3d[valid_mask],
                    weights=weights
                )

                # Ensure plane_inlier_mask is 1D
                if plane_inlier_mask is not None:
                    plane_inlier_mask = plane_inlier_mask.flatten()

                    # Create a new mask instead of modifying in-place
                    new_plane_mask = np.zeros(curr_len, dtype=bool)
                    valid_indices = np.where(valid_mask)[0]

                    # Handle case where plane_inlier_mask might be wrong size
                    if len(plane_inlier_mask) == len(valid_indices):
                        new_plane_mask[valid_indices] = plane_inlier_mask
                        valid_mask = new_plane_mask
                    else:
                        self.logger.warning(
                            f"Plane mask size mismatch: {len(plane_inlier_mask)} vs {len(valid_indices)}")
            except Exception as e:
                self.logger.debug(f"Plane detection failed: {str(e)}")

        # Centroid replacement for outliers
        if np.any(valid_mask):
            centroid_x = np.mean(x[valid_mask])
            centroid_y = np.mean(y[valid_mask])
            centroid_z = np.mean(z[valid_mask])
        else:
            centroid_x, centroid_y, centroid_z = 0, 0, 0

        x_out = np.where(valid_mask, x, centroid_x)
        y_out = np.where(valid_mask, y, centroid_y)
        z_out = np.where(valid_mask, z, centroid_z)

        return {
            'x': x_out,
            'y': y_out,
            'z': z_out,
            'numObj': num,
            'plane_params': plane_params,
            'inlier_mask': plane_inlier_mask,
            'quality_weights': quality_weights
        }