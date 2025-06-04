
import numpy as np
from collections import deque
from scipy.signal import savgol_filter



class RadarDespiker:
    def __init__(self, radius=0.5, base_z_thresh=0.3, base_xy_thresh=0.3, snr_gate=7.0):
        self.radius = radius
        self.base_z_thresh = base_z_thresh
        self.base_xy_thresh = base_xy_thresh
        self.snr_gate = snr_gate
        self.history_buffer = deque(maxlen=10)

    def debug_print(self, msg):
        print(msg)

    def statistical_outlier_removal(self, points, k_neighbors=8, std_multiplier=2.0):
        if len(points) < k_neighbors:
            return np.ones(len(points), dtype=bool)

        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=min(k_neighbors, len(points)-1)).fit(points)
        distances, _ = nbrs.kneighbors(points)
        mean_distances = np.mean(distances[:, 1:], axis=1)
        mean_dist = np.mean(mean_distances)
        std_dist = np.std(mean_distances)
        threshold = mean_dist + std_multiplier * std_dist
        return mean_distances <= threshold

    def radius_outlier_removal(self, points, radius_thresh=0.3, min_neighbors=3):
        if len(points) < min_neighbors:
            return np.ones(len(points), dtype=bool)

        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(radius=radius_thresh).fit(points)
        neighbor_counts = [len(indices) - 1 for indices in nbrs.radius_neighbors(points)[1]]
        return np.array(neighbor_counts) >= min_neighbors

    def adaptive_savgol(self, data, base_window=7, poly=3):
        if len(data) < 3:
            return data

        data_range = np.ptp(data)
        data_std = np.std(data)
        noise_level = data_std / (data_range + 1e-6)
        window_mult = 1.0 + min(2.0, noise_level * 5)
        adaptive_window = int(base_window * window_mult)
        w = adaptive_window if adaptive_window <= len(data) else (len(data) if len(data) % 2 == 1 else len(data) - 1)
        if w < 3:
            return data
        adaptive_poly = min(poly, w - 1, 5)

        try:
            return savgol_filter(data, window_length=w, polyorder=adaptive_poly, mode='interp')
        except Exception as e:
            self.debug_print(f"Adaptive Savgol filter error: {e}")
            return data

    def process(self, det_obj, snr, noise):
        x = np.array(det_obj['x'])
        y = np.array(det_obj['y'])
        z = np.array(det_obj['z'])
        num = det_obj.get('numObj')

        if snr is None:
            snr = np.ones_like(x)
        else:
            snr = np.asarray(snr)

        curr_len = len(x)
        if curr_len == 0:
            return {'x': x, 'y': y, 'z': z, 'numObj': num}

        if len(snr) == curr_len:
            snr_mask = snr >= self.snr_gate
            valid_count = np.sum(snr_mask)
            if valid_count < max(3, curr_len//4):
                top_n = max(3, curr_len//2)
                top_indices = np.argpartition(snr, -top_n)[-top_n:]
                snr_mask = np.zeros_like(snr_mask)
                snr_mask[top_indices] = True
        else:
            snr_mask = np.ones(curr_len, dtype=bool)

        points_3d = np.column_stack([x, y, z])
        point_density = curr_len / (np.ptp(x) * np.ptp(y) * np.ptp(z) + 1e-6)
        adaptive_radius = max(0.1, min(self.radius, 1.0 / np.sqrt(point_density + 1e-6)))

        sor_mask = self.statistical_outlier_removal(points_3d)
        ror_mask = self.radius_outlier_removal(points_3d, radius_thresh=adaptive_radius)
        valid_mask = snr_mask & sor_mask & ror_mask

        if np.sum(valid_mask) < max(3, curr_len // 4):
            top_indices = np.argsort(snr if len(snr) == curr_len else np.ones(curr_len))[-max(3, curr_len//2):]
            valid_mask = np.isin(range(curr_len), top_indices)

        x_filtered = x.copy()
        y_filtered = y.copy()
        z_filtered = z.copy()

        self.history_buffer.append((x_filtered, y_filtered, z_filtered, valid_mask))
        max_history = max(3, min(10, 20 // max(1, curr_len // 10)))
        if len(self.history_buffer) > max_history:
            self.history_buffer.pop(0)

        x_avg = x_filtered.copy()
        y_avg = y_filtered.copy()
        z_avg = z_filtered.copy()

        if len(self.history_buffer) > 1:
            weights = []
            x_stack = []
            y_stack = []
            z_stack = []

            for i, (hx, hy, hz, hmask) in enumerate(self.history_buffer):
                if len(hx) >= curr_len:
                    age_weight = 0.5 + 0.5 * (i / len(self.history_buffer))
                    quality_weight = np.sum(hmask[:curr_len]) / curr_len if len(hmask) >= curr_len else 1.0
                    weight = age_weight * quality_weight
                    weights.append(weight)
                    x_stack.append(hx[:curr_len])
                    y_stack.append(hy[:curr_len])
                    z_stack.append(hz[:curr_len])

            if weights:
                weights = np.array(weights)
                weights /= np.sum(weights)
                x_avg = np.average(np.vstack(x_stack), axis=0, weights=weights)
                y_avg = np.average(np.vstack(y_stack), axis=0, weights=weights)
                z_avg = np.average(np.vstack(z_stack), axis=0, weights=weights)

        x_smooth = self.adaptive_savgol(x_avg)
        y_smooth = self.adaptive_savgol(y_avg)
        z_smooth = self.adaptive_savgol(z_avg)

        x_smooth = np.nan_to_num(x_smooth, nan=x_filtered, posinf=x_filtered, neginf=x_filtered)
        y_smooth = np.nan_to_num(y_smooth, nan=y_filtered, posinf=y_filtered, neginf=y_filtered)
        z_smooth = np.nan_to_num(z_smooth, nan=z_filtered, posinf=z_filtered, neginf=z_filtered)

        return {'x': x_smooth, 'y': y_smooth, 'z': z_smooth, 'numObj': num}
