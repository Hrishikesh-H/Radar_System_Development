import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

from collections import deque
import numpy as np

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

    def adaptive_savgol(self, data, base_window=3, poly=2):
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
            self.history_buffer.popleft()

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
        y_smooth = y_avg
        z_smooth = self.adaptive_savgol(z_avg)

        x_smooth = np.nan_to_num(x_smooth, nan=x_filtered, posinf=x_filtered, neginf=x_filtered)
        y_smooth = np.nan_to_num(y_smooth, nan=y_filtered, posinf=y_filtered, neginf=y_filtered)
        z_smooth = np.nan_to_num(z_smooth, nan=z_filtered, posinf=z_filtered, neginf=z_filtered)

        return {'x': x_smooth, 'y': y_smooth, 'z': z_smooth, 'numObj': num}


st.title("RadarDespiker Interactive Viewer")

csv_file = st.file_uploader("Upload your CSV file", type="csv")
if csv_file:
    df = pd.read_csv(csv_file)
    st.write("Loaded data:", df.head())

    # Check for needed columns
    col_map = {c.lower(): c for c in df.columns}
    has_needed = all(k in col_map for k in ['timestamp', 'x', 'y', 'z'])

    if not has_needed:
        st.error("CSV must contain columns: timestamp, x, y, z. (Case-insensitive)")
    else:
        timestamps = df[col_map['timestamp']].unique()
        timestamps = np.sort(timestamps)  # Sort timestamps for proper ordering
        num_timestamps = len(timestamps)

        if num_timestamps == 0:
            st.error("No frames found! Check your CSV for valid timestamp values.")
        else:
            # Display frame information
            st.info(f"Found {num_timestamps} unique timestamps")

            # Two modes: Single frame or Cumulative frames
            view_mode = st.radio("View Mode", ["Single Frame", "Cumulative Frames"])

            if view_mode == "Single Frame":
                if num_timestamps == 1:
                    st.info("Only one frame available")
                    frame_idx = 0
                else:
                    frame_idx = st.slider("Frame index", min_value=0, max_value=num_timestamps - 1, value=0, step=1)

                frame_time = timestamps[frame_idx]
                frame_df = df[df[col_map['timestamp']] == frame_time]
                st.write(f"Frame {frame_idx}: timestamp {frame_time}, {len(frame_df)} points")

            else:  # Cumulative Frames
                col1, col2, col3 = st.columns(3)

                with col1:
                    center_frame = st.slider("Center frame", min_value=0, max_value=num_timestamps - 1, value=0, step=1)
                with col2:
                    frames_before = st.slider("Frames before", min_value=0, max_value=center_frame, value=0, step=1)
                with col3:
                    frames_after = st.slider("Frames after", min_value=0, max_value=num_timestamps - center_frame - 1,
                                             value=0, step=1)

                # Calculate frame range
                start_idx = max(0, center_frame - frames_before)
                end_idx = min(num_timestamps - 1, center_frame + frames_after)

                selected_timestamps = timestamps[start_idx:end_idx + 1]
                frame_df = df[df[col_map['timestamp']].isin(selected_timestamps)]

                st.write(
                    f"Showing frames {start_idx} to {end_idx} ({len(selected_timestamps)} frames, {len(frame_df)} total points)")

            # Prepare det_obj
            if len(frame_df) == 0:
                st.warning("No data points in selected frame(s)")
            else:
                det_obj = {
                    'x': frame_df[col_map['x']].values,
                    'y': frame_df[col_map['y']].values,
                    'z': frame_df[col_map['z']].values,
                    'numObj': len(frame_df)
                }
                snr = frame_df[col_map['snr']].values if 'snr' in col_map else None
                noise = frame_df[col_map['noise']].values if 'noise' in col_map else None

                # Initialize despiker
                if 'despiker' not in st.session_state:
                    st.session_state['despiker'] = RadarDespiker()
                despiker = st.session_state['despiker']

                # Add reset button
                if st.button("Reset Despiker History"):
                    st.session_state['despiker'] = RadarDespiker()
                    st.success("Despiker history reset!")

                try:
                    result = despiker.process(det_obj, snr, noise)

                    # Plotting
                    # Sliders for interactive 3D view angles
                    colA, colB = st.columns(2)
                    elev_angle = colA.slider("Elevation (vertical)", min_value=0, max_value=90, value=30, step=1)
                    azim_angle = colB.slider("Azimuth (horizontal)", min_value=0, max_value=360, value=60, step=1)

                    # Plotting
                    fig = plt.figure(figsize=(12, 8))
                    ax = fig.add_subplot(111, projection='3d')

                    # Plot raw points
                    if len(det_obj['x']) > 0:
                        ax.scatter(det_obj['x'], det_obj['z'], det_obj['y'],
                                   c='red', label='Raw', alpha=0.6, s=30)

                    # Plot despiked points
                    if len(result['x']) > 0:
                        ax.scatter(result['x'], result['z'], result['y'],
                                   c='green', label='Despiked', alpha=0.7, s=30)

                    ax.set_xlabel('X (m)')
                    ax.set_ylabel('Z (m)')
                    ax.set_zlabel('Y (m)')
                    ax.view_init(elev=elev_angle, azim=azim_angle)

                    if view_mode == "Single Frame":
                        ax.set_title(f"Frame {frame_idx} at timestamp {frame_time}\n{len(frame_df)} points")
                    else:
                        ax.set_title(f"Cumulative: Frames {start_idx}-{end_idx}\n{len(frame_df)} total points")

                    ax.legend()
                    plt.tight_layout()
                    st.pyplot(fig)

                    # Statistics display
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Raw Points", len(det_obj['x']))
                    with col2:
                        st.metric("Processed Points", len(result['x']))
                    with col3:
                        reduction_pct = (1 - len(result['x']) / max(1, len(det_obj['x']))) * 100
                        st.metric("Reduction %", f"{reduction_pct:.1f}%")


                except Exception as e:
                    st.error(f"Error processing data: {str(e)}")
                    st.write("Debug info:")
                    st.write(f"- Points in frame: {len(frame_df)}")
                    st.write(f"- X range: {frame_df[col_map['x']].min():.3f} to {frame_df[col_map['x']].max():.3f}")
                    st.write(f"- Y range: {frame_df[col_map['y']].min():.3f} to {frame_df[col_map['y']].max():.3f}")
                    st.write(f"- Z range: {frame_df[col_map['z']].min():.3f} to {frame_df[col_map['z']].max():.3f}")
