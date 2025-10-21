import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from collections import deque
import numpy as np
from scipy.signal import medfilt


from collections import deque
import numpy as np
from scipy.signal import medfilt, savgol_filter

from collections import deque
import numpy as np
from scipy.signal import medfilt, savgol_filter

from collections import deque
import numpy as np
from scipy.signal import medfilt, savgol_filter

class RadarDespiker:
    def __init__(self, radius=0.5, snr_gate=10.0, k_neighbors=1, std_multiplier=2.0, min_neighbors=1,
                 min_points=5, smooth_mode='median', smooth_param=3):
        self.radius = radius
        self.snr_gate = snr_gate
        self.k_neighbors = k_neighbors
        self.std_multiplier = std_multiplier
        self.min_neighbors = min_neighbors
        self.min_points = min_points
        self.smooth_mode = smooth_mode
        self.smooth_param = smooth_param
        self.point_history = deque(maxlen=5)  # holds previous output batches

    def savgol(self, arr):
        n = len(arr)
        win = min(self.smooth_param, n if n % 2 == 1 else n - 1)
        if win < 3: return arr
        try:
            return savgol_filter(arr, window_length=win, polyorder=min(2, win - 1), mode='interp')
        except Exception:
            return arr

    def medfilt(self, arr):
        if len(arr) < 3:
            return arr
        k = min(self.smooth_param, len(arr) if len(arr) % 2 == 1 else len(arr) - 1)
        if k < 3: return arr
        return medfilt(arr, kernel_size=k)

    def statistical_outlier_removal(self, points):
        if len(points) < self.k_neighbors:
            return np.ones(len(points), dtype=bool)
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=min(self.k_neighbors, len(points)-1)).fit(points)
        distances, _ = nbrs.kneighbors(points)
        mean_distances = np.mean(distances[:, 1:], axis=1)
        mean_dist = np.mean(mean_distances)
        std_dist = np.std(mean_distances)
        threshold = mean_dist + self.std_multiplier * std_dist
        return mean_distances <= threshold

    def radius_outlier_removal(self, points):
        if len(points) < self.min_neighbors:
            return np.ones(len(points), dtype=bool)
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(radius=self.radius).fit(points)
        neighbor_counts = [len(indices) - 1 for indices in nbrs.radius_neighbors(points)[1]]
        return np.array(neighbor_counts) >= self.min_neighbors

    def process(self, det_obj, snr, noise):
        x_raw = np.asarray(det_obj['x'])
        y_raw = np.asarray(det_obj['y'])
        z_raw = np.asarray(det_obj['z'])
        num = det_obj.get('numObj')
        curr_len = len(x_raw)
        # Early exit for empty frame
        if curr_len == 0:
            # No points at all—try to fill at least min_points from history, else return zeros
            hx, hy, hz, hsnr = [], [], [], []
            for h in reversed(self.point_history):
                for xi, yi, zi, snri in zip(h['x'], h['y'], h['z'], h['snr']):
                    hx.append(xi)
                    hy.append(yi)
                    hz.append(zi)
                    hsnr.append(snri)
                    if len(hx) >= self.min_points:
                        break
                if len(hx) >= self.min_points:
                    break
            if len(hx) < self.min_points:
                # Pad with zeros
                hx += [0.] * (self.min_points - len(hx))
                hy += [0.] * (self.min_points - len(hy))
                hz += [0.] * (self.min_points - len(hz))
                hsnr += [0.] * (self.min_points - len(hsnr))
            return {'x': np.array(hx[:self.min_points]), 'y': np.array(hy[:self.min_points]),
                    'z': np.array(hz[:self.min_points]), 'numObj': self.min_points, 'snr': np.array(hsnr[:self.min_points])}

        if snr is None:
            snr = np.ones_like(x_raw)
        else:
            snr = np.asarray(snr)

        if self.smooth_mode == "savgol":
            x = self.savgol(x_raw)
            z = self.savgol(z_raw)
        elif self.smooth_mode == "median":
            x = self.medfilt(x_raw)
            z = self.medfilt(z_raw)
        else:
            x = x_raw
            z = z_raw
        y = y_raw

        snr_mask = (snr >= self.snr_gate) if len(snr) == curr_len else np.ones(curr_len, dtype=bool)
        points_3d = np.column_stack([x, y, z])
        sor_mask = self.statistical_outlier_removal(points_3d)
        ror_mask = self.radius_outlier_removal(points_3d)
        valid_mask = snr_mask & sor_mask & ror_mask

        # Take only filtered points
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        z_valid = z[valid_mask]
        snr_valid = snr[valid_mask]
        N_valid = len(x_valid)

        # Collect recent history for refilling if needed (always freshest first, always unique)
        if N_valid < self.min_points:
            supplement = []
            already = set(zip(map(float, x_valid), map(float, y_valid), map(float, z_valid)))
            for h in reversed(self.point_history):  # Recent to old
                for xi, yi, zi, snri in zip(h['x'], h['y'], h['z'], h['snr']):
                    tup = (float(xi), float(yi), float(zi))
                    if tup not in already:
                        supplement.append((xi, yi, zi, snri))
                        already.add(tup)
                    if len(x_valid) + len(supplement) >= self.min_points:
                        break
                if len(x_valid) + len(supplement) >= self.min_points:
                    break
            for pt in supplement[:self.min_points - N_valid]:
                x_valid = np.append(x_valid, pt[0])
                y_valid = np.append(y_valid, pt[1])
                z_valid = np.append(z_valid, pt[2])
                snr_valid = np.append(snr_valid, pt[3])

        # Save this frame's valid points (for history use in next frame)
        self.point_history.append({'x': x_valid, 'y': y_valid, 'z': z_valid, 'snr': snr_valid})

        return {'x': x_valid, 'y': y_valid, 'z': z_valid, 'numObj': len(x_valid), 'snr': snr_valid}






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
