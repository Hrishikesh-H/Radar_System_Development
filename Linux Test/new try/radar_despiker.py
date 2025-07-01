import numpy as np
import time
from collections import deque
from scipy.signal import savgol_filter
from system_logger import get_logger


class RadarDespiker:
    def __init__(self, radius=0.5, base_z_thresh=0.3, base_xy_thresh=0.3, snr_gate=7.0):
        self.radius = radius
        self.base_z_thresh = base_z_thresh
        self.base_xy_thresh = base_xy_thresh
        self.snr_gate = snr_gate
        self.history_buffer = deque(maxlen=10)
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

    def adaptive_savgol(self, data, base_window=7, poly=3):
        n_data = len(data)
        if n_data < 3:
            return data

        data_range = np.ptp(data)
        data_std = np.std(data)
        noise_level = data_std / (data_range + 1e-6)
        window_mult = 1.0 + min(2.0, noise_level * 5)
        adaptive_window = int(base_window * window_mult)
        w = min(adaptive_window, n_data if n_data % 2 == 1 else n_data - 1)

        if w < 3:
            return data
        adaptive_poly = min(poly, w - 1, 5)

        try:
            return savgol_filter(data, window_length=w, polyorder=adaptive_poly, mode='interp')
        except Exception as e:
            self.logger.debug(f"Adaptive Savgol filter error: {e}")
            return data

    def process(self, det_obj, snr=None, noise=None):
        start_time = time.perf_counter()
        x = np.asarray(det_obj['x'])
        y = np.asarray(det_obj['y'])
        z = np.asarray(det_obj['z'])
        num = det_obj.get('numObj')
        curr_len = len(x)

        if curr_len == 0:
            result = {'x': x, 'y': y, 'z': z, 'numObj': num}
        else:
            # Process SNR mask
            snr_arr = np.asarray(snr) if snr is not None else None
            if snr_arr is not None and len(snr_arr) == curr_len:
                snr_mask = snr_arr >= self.snr_gate
                valid_count = np.sum(snr_mask)
                if valid_count < max(3, curr_len // 4):
                    top_n = max(3, curr_len // 2)
                    top_indices = np.argpartition(snr_arr, -top_n)[-top_n:]
                    snr_mask = np.zeros(curr_len, dtype=bool)
                    snr_mask[top_indices] = True
            else:
                snr_mask = np.ones(curr_len, dtype=bool)

            # Outlier removal
            points_3d = np.column_stack([x, y, z])
            point_density = curr_len / (np.ptp(x) * np.ptp(y) * np.ptp(z) + 1e-6)
            adaptive_radius = max(0.1, min(self.radius, 1.0 / np.sqrt(point_density + 1e-6)))

            sor_mask = self.statistical_outlier_removal(points_3d)
            ror_mask = self.radius_outlier_removal(points_3d, radius_thresh=adaptive_radius)
            valid_mask = snr_mask & sor_mask & ror_mask

            if np.sum(valid_mask) < max(3, curr_len // 4):
                sort_arr = snr_arr if snr_arr is not None and len(snr_arr) == curr_len else np.ones(curr_len)
                top_indices = np.argpartition(sort_arr, -max(3, curr_len // 2))[-max(3, curr_len // 2):]
                valid_mask = np.zeros(curr_len, dtype=bool)
                valid_mask[top_indices] = True

            # Store current frame in history
            self.history_buffer.append((x.copy(), y.copy(), z.copy(), valid_mask.copy()))
            max_history = max(3, min(10, 20 // max(1, curr_len // 10)))
            while len(self.history_buffer) > max_history:
                self.history_buffer.popleft()

            # Temporal averaging
            if len(self.history_buffer) > 1:
                weights = np.zeros(len(self.history_buffer))
                x_stack = np.empty((len(self.history_buffer), curr_len))
                y_stack = np.empty((len(self.history_buffer), curr_len))
                z_stack = np.empty((len(self.history_buffer), curr_len))

                for i, (hx, hy, hz, hmask) in enumerate(self.history_buffer):
                    n_hist = len(hx)
                    if n_hist >= curr_len:
                        trunc = slice(0, curr_len)
                        age_weight = 0.5 + 0.5 * (i / len(self.history_buffer))
                        quality_weight = np.sum(hmask[trunc]) / curr_len
                        weight = age_weight * quality_weight
                        weights[i] = weight
                        x_stack[i] = hx[trunc]
                        y_stack[i] = hy[trunc]
                        z_stack[i] = hz[trunc]
                    else:
                        weights[i] = 0

                if np.sum(weights) > 0:
                    weights /= np.sum(weights)
                    x_avg = np.average(x_stack, axis=0, weights=weights)
                    y_avg = np.average(y_stack, axis=0, weights=weights)
                    z_avg = np.average(z_stack, axis=0, weights=weights)
                else:
                    x_avg, y_avg, z_avg = x, y, z
            else:
                x_avg, y_avg, z_avg = x, y, z

            # Apply smoothing
            x_smooth = self.adaptive_savgol(x_avg)
            y_smooth = self.adaptive_savgol(y_avg)
            z_smooth = self.adaptive_savgol(z_avg)

            # Handle potential NaNs/Infs
            x_smooth = np.nan_to_num(x_smooth, nan=x_avg, posinf=x_avg, neginf=x_avg)
            y_smooth = np.nan_to_num(y_smooth, nan=y_avg, posinf=y_avg, neginf=y_avg)
            z_smooth = np.nan_to_num(z_smooth, nan=z_avg, posinf=z_avg, neginf=z_avg)

            result = {'x': x_smooth, 'y': y_smooth, 'z': z_smooth, 'numObj': num}

        # Calculate and log processing frequency
        end_time = time.perf_counter()
        elapsed_ms = (end_time - start_time) * 1000
        fps = 1000 / elapsed_ms if elapsed_ms > 0 else float('inf')
        self.logger.info(
            f"Despiker processed {curr_len} points in {elapsed_ms:.2f} ms "
            f"({fps:.1f} FPS)"
        )

        return result