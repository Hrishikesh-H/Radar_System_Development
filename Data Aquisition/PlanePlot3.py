import streamlit as st
import pandas as pd

import matplotlib.pyplot as plt
from collections import deque
from scipy.signal import medfilt
import math
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import RANSACRegressor, LinearRegression
from scipy.spatial.distance import pdist
import numpy as np
# ---- Styling for IEEE/ISO professional graphs ----
plt.rcParams['font.family'] = 'DejaVu Sans' # Closest to Helvetica, widely available
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['figure.dpi'] = 150
plt.rcParams['axes.linewidth'] = 1.2

class RadarDespiker:
    def __init__(self, radius=0.5, snr_gate=5.0, k_neighbors=4, std_multiplier=2.5, min_neighbors=2,
                 min_points=4, smooth_mode=None, smooth_param=1, min_obstacle_distance=1.5):
        self.radius = radius
        self.snr_gate = snr_gate
        self.k_neighbors = k_neighbors
        self.std_multiplier = std_multiplier
        self.min_neighbors = min_neighbors
        self.min_points = min_points
        self.smooth_mode = smooth_mode
        self.smooth_param = smooth_param
        self.min_obstacle_distance = min_obstacle_distance
        self.point_history = deque(maxlen=7)
        self.max_obstacle_distance = 7.0

    def medfilt(self, arr):
        """Light median filter to remove spikes without changing contour."""
        if len(arr) < 3:
            return arr
        k = min(self.smooth_param, len(arr) if len(arr) % 2 == 1 else len(arr) - 1)
        if k < 3:
            return arr
        return medfilt(arr, kernel_size=k)

    def statistical_outlier_removal(self, points):
        """Remove statistical outliers based on local mean distance."""
        if len(points) < 2:
            return np.ones(len(points), dtype=bool)
        from sklearn.neighbors import NearestNeighbors
        k = min(self.k_neighbors, len(points) - 1)
        if k < 1:
            return np.ones(len(points), dtype=bool)
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(points)
        distances, _ = nbrs.kneighbors(points)
        mean_distances = np.mean(distances[:, 1:], axis=1)
        threshold = np.mean(mean_distances) + self.std_multiplier * np.std(mean_distances)
        return mean_distances <= threshold

    def radius_outlier_removal(self, points):
        """Keep points with enough neighbors in radius."""
        if len(points) == 0:
            return np.array([], dtype=bool)
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(radius=self.radius).fit(points)
        neighbor_counts = [len(indices) - 1 for indices in nbrs.radius_neighbors(points)[1]]
        return np.array(neighbor_counts) >= self.min_neighbors

    def process(self, det_obj, snr, noise):
        x_raw = np.asarray(det_obj['x'])
        y_raw = np.asarray(det_obj['y'])
        z_raw = np.asarray(det_obj['z'])

        if snr is None:
            snr = np.ones_like(x_raw)
        else:
            snr = np.asarray(snr)

        # STEP 1: Discard points with |y| < min_obstacle_distance
        safe_mask = np.abs(y_raw) >= self.min_obstacle_distance
        if np.sum(safe_mask) == 0:
            return {'x': np.array([]), 'y': np.array([]), 'z': np.array([]),
                    'numObj': 0, 'snr': np.array([])}

        x_safe = x_raw[safe_mask]
        y_safe = y_raw[safe_mask]
        z_safe = z_raw[safe_mask]
        snr_safe = snr[safe_mask]

        # STEP 2: Minimal smoothing (optional)
        if self.smooth_mode == "median":
            x_smooth = self.medfilt(x_safe)
            z_smooth = self.medfilt(z_safe)
        else:
            x_smooth, z_smooth = x_safe, z_safe
        y_smooth = y_safe  # y never modified

        # STEP 3: Outlier removal
        snr_mask = snr_safe >= self.snr_gate
        points_3d = np.column_stack([x_smooth, y_smooth, z_smooth])
        sor_mask = self.statistical_outlier_removal(points_3d)
        ror_mask = self.radius_outlier_removal(points_3d)
        valid_mask = snr_mask & sor_mask & ror_mask

        x_valid = x_smooth[valid_mask]
        y_valid = y_smooth[valid_mask]
        z_valid = z_smooth[valid_mask]
        snr_valid = snr_safe[valid_mask]

        # STEP 4: Fallback to top SNR safe points if too few
        if len(x_valid) < self.min_points:
            fallback_mask = snr_safe >= self.snr_gate
            idx_sorted = np.argsort(-snr_safe[fallback_mask])
            for i in idx_sorted:
                if len(x_valid) >= self.min_points:
                    break
                x_valid = np.append(x_valid, x_safe[fallback_mask][i])
                y_valid = np.append(y_valid, y_safe[fallback_mask][i])
                z_valid = np.append(z_valid, z_safe[fallback_mask][i])
                snr_valid = np.append(snr_valid, snr_safe[fallback_mask][i])

        # STEP 5: Fallback to very recent history points if still too few
        if len(x_valid) < self.min_points:
            for h in reversed(self.point_history):
                hx, hy, hz, hsnr = h['x'], h['y'], h['z'], h['snr']
                for xi, yi, zi, snri in zip(hx, hy, hz, hsnr):
                    if len(x_valid) >= self.min_points:
                        break
                    # Add only real points
                    if not ((x_valid == xi) & (y_valid == yi) & (z_valid == zi)).all():
                        x_valid = np.append(x_valid, xi)
                        y_valid = np.append(y_valid, yi)
                        z_valid = np.append(z_valid, zi)
                        snr_valid = np.append(snr_valid, snri)
                if len(x_valid) >= self.min_points:
                    break

        # STEP 6: Do NOT pad with dummy points — strict contour preservation

        # STEP 7: Save only safe points
        final_safe_mask = np.abs(y_valid) >= self.min_obstacle_distance
        self.point_history.append({
            'x': x_valid[final_safe_mask],
            'y': y_valid[final_safe_mask],
            'z': z_valid[final_safe_mask],
            'snr': snr_valid[final_safe_mask]
        })

        return {'x': x_valid[final_safe_mask],
                'y': y_valid[final_safe_mask],
                'z': z_valid[final_safe_mask],
                'numObj': len(x_valid[final_safe_mask]),
                'snr': snr_valid[final_safe_mask]}


class LandingZoneAssessor:
    def __init__(self, distance_thresh=0.05, min_iterations=500, slope_thresh_deg=10, inlier_ratio_thresh=0.4):
        self.distance_thresh = distance_thresh
        self.min_iterations = min_iterations
        self.slope_thresh_deg = slope_thresh_deg
        self.inlier_ratio_thresh = inlier_ratio_thresh

    def assess(self, det_obj):
        try:
            # Check required keys
            for key in ['x', 'y', 'z', 'numObj']:
                if key not in det_obj:
                    return False, {'reason': f"Missing key '{key}'"}

            x = np.array(det_obj['x'])
            y = np.array(det_obj['y'])  # vertical axis
            z = np.array(det_obj['z'])
            n = det_obj['numObj']
            if n < 3:
                return False, {'reason': 'Not enough points'}

            pts = np.vstack((x, y, z)).T  # [X, Y, Z]

            # Multi-scale RANSAC plane fitting (Y = f(X,Z))
            def multi_scale_ransac(points, base_thresh, iterations):
                best_model = None
                best_inliers = []
                best_score = -1
                scales = [0.5, 1.0, 1.5, 2.0] if len(points) < 50 else [0.8, 1.0, 1.2]
                for scale in scales:
                    thresh = base_thresh * scale
                    max_iter = max(100, iterations // len(scales))
                    X_fit = points[:, [0, 2]]  # X and Z as independent
                    y_plane = points[:, 1]      # Y as dependent (vertical)
                    ransac = RANSACRegressor(
                        LinearRegression(),
                        residual_threshold=thresh,
                        max_trials=max_iter,
                        random_state=42)
                    ransac.fit(X_fit, y_plane)
                    m, n_coef = ransac.estimator_.coef_
                    b = ransac.estimator_.intercept_
                    # Plane equation: a*X + b*Z + c*Y + d = 0, set c = -1
                    a, b_coef, c, d = m, n_coef, -1, b
                    norm = np.sqrt(a*a + b_coef*b_coef + c*c)
                    if norm > 0:
                        a, b_coef, c, d = a/norm, b_coef/norm, c/norm, d/norm
                    inlier_mask = ransac.inlier_mask_
                    inlier_count = np.sum(inlier_mask)
                    inlier_ratio = inlier_count / len(points)
                    consistency_score = 1.0 - np.std(points[inlier_mask][:, 1]) / (np.ptp(points[:, 1]) + 1e-6)
                    score = inlier_ratio * 0.7 + consistency_score * 0.3
                    if score > best_score:
                        best_score = score
                        best_model = (a, b_coef, c, d)
                        best_inliers = np.where(inlier_mask)[0]
                return best_model, best_inliers

            coeffs, inlier_idxs = multi_scale_ransac(pts, self.distance_thresh, self.min_iterations)
            if coeffs is None or len(coeffs) < 4:
                return False, {'reason': 'Plane fitting failed'}

            a, b_coef, c, d = coeffs
            inlier_mask = np.zeros(n, dtype=bool)
            if len(inlier_idxs) > 0:
                inlier_mask[inlier_idxs] = True
            inlier_count = inlier_mask.sum()
            inlier_ratio = inlier_count / n
            # Distance to plane for residuals
            dists = np.abs((pts.dot([a, 0, b_coef]) + c*pts[:, 1] + d))  # a*X + b*Z + c*Y + d
            mean_residual = dists[inlier_mask].mean() if inlier_count > 0 else np.inf

            # Surface quality assessment
            def assess_surface_quality(points, inlier_mask):
                if np.sum(inlier_mask) < 3:
                    return 0.0
                inlier_points = points[inlier_mask]
                y_range = np.ptp(inlier_points[:, 1])
                y_std = np.std(inlier_points[:, 1])
                roughness_score = max(0, 1.0 - y_std / (y_range + 1e-6))
                if len(inlier_points) > 5:
                    distances = pdist(inlier_points[:, [0, 2]])
                    density_uniformity = 1.0 - (np.std(distances) / (np.mean(distances) + 1e-6))
                    density_uniformity = max(0, min(1, density_uniformity))
                else:
                    density_uniformity = 0.5
                area_coverage = min(1.0, len(inlier_points) / max(10, n * 0.8))
                return np.mean([roughness_score, density_uniformity, area_coverage])

            surface_quality = assess_surface_quality(pts, inlier_mask)

            # Plane normal vector
            normal = np.array([a, c, b_coef])
            norm = np.linalg.norm(normal)
            if norm == 0:
                return False, {'reason': 'Plane normal vector zero'}
            normal = normal / norm

            # Slope w.r.t vertical Y-axis
            vertical = np.array([0, 1, 0])
            cos_theta = np.dot(normal, vertical)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            slope_deg = math.degrees(math.acos(abs(cos_theta)))

            # Temporal consistency placeholder
            temporal_consistency = 1.0

            # Geometric and quality checks
            geometric_safe = (slope_deg <= self.slope_thresh_deg and
                              inlier_ratio >= self.inlier_ratio_thresh and
                              mean_residual <= self.distance_thresh)
            quality_safe = surface_quality >= 0.4
            confidence_score = (inlier_ratio * 0.4 +
                                surface_quality * 0.3 +
                                temporal_consistency * 0.2 +
                                (1.0 - min(1.0, slope_deg / self.slope_thresh_deg)) * 0.1)

            safe = geometric_safe and quality_safe and confidence_score >= 0.5

            # Metrics dictionary
            metrics = {
                'plane': (a, b_coef, c, d),
                'slope_deg': slope_deg,
                'inlier_ratio': inlier_ratio,
                'mean_residual': mean_residual,
                'surface_quality': surface_quality,
                'confidence_score': confidence_score,
                'temporal_consistency': temporal_consistency
            }

            # Reason for unsafe
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
            return False, {'reason': f'Exception: {e}'}


# Add these sliders before you call plot_plane_fit:
elev = st.slider('3D Elevation Angle (de', min_value=0, max_value=90, value=25)
azim = st.slider('3D Azimuth Angle (deg)', min_value=0, max_value=360, value=35)


# PROFESSIONAL PLOTTING FUNCTIONS -- Place before your app code


def plot_plane_fit(x, y, z, coeffs, metrics, elev=25, azim=35,
                   step_x=0.5, step_y=0.5, step_z=0.5):
    """
    Professional 3D flat terrain visualization.
    Features:
    - Plane light-colored with subtle outline
    - Y-axis and plane normal arrows in contrasting colors
    - Theta (slope) shown as acute angle between arrows in legend
    - Points labeled 'Filtered Points'
    - Helvetica IEEE-style font
    - Top-to-bottom Y-axis
    """
    fig = plt.figure(figsize=(45,10), constrained_layout=True)
    ax = fig.add_subplot(111, projection='3d')
    plt.rcParams.update({'font.family': 'Helvetica', 'font.size': 11})

    # Scatter points
    ax.scatter(x, y, z, c='navy', s=25, alpha=0.9, label='Filtered Points')

    # Plane coefficients
    a, b_coef, c, d = coeffs

    # Plane mesh covering points
    margin_ratio = 0.02
    x_min, x_max = np.min(x), np.max(x)
    z_min, z_max = np.min(z), np.max(z)
    xx, zz = np.meshgrid(
        np.linspace(x_min - margin_ratio*(x_max-x_min), x_max + margin_ratio*(x_max-x_min), 8),
        np.linspace(z_min - margin_ratio*(z_max-z_min), z_max + margin_ratio*(z_max-z_min), 8)
    )
    yy = (-a*xx - b_coef*zz - d)/c

    # Light plane with subtle edges
    ax.plot_surface(xx, yy, zz, color='lightcoral', alpha=0.25, edgecolor='lightgray', linewidth=0.8)

    # Centroid for arrows
    origin = np.array([np.mean(x), np.mean(y), np.mean(z)])
    scale = 0.5 * max(np.ptp(x), np.ptp(y), np.ptp(z))

    # Contrasting arrows
    # Contrasting arrows with small arrowheads
    y_axis_vec = np.array([0, -1, 0]) * scale  # Y-axis top-to-bottom
    plane_normal_vec = np.array([a, c, b_coef])
    plane_normal_vec = plane_normal_vec / np.linalg.norm(plane_normal_vec) * scale

    # Use smaller arrowheads
    ax.quiver(*origin, *y_axis_vec, color='forestgreen', linewidth=2, arrow_length_ratio=0.05)
    ax.quiver(*origin, *plane_normal_vec, color='orange', linewidth=2, arrow_length_ratio=0.05)

    # Theta angle (acute)
    theta = metrics.get('slope_deg', None)
    theta_text = f"θ = {theta:.1f}°" if theta is not None else ""

    # Axis labels
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_ylim(ax.get_ylim()[::-1])  # top-to-bottom Y

    # Axis ticks
    ax.set_xticks(np.arange(np.floor(x_min), np.ceil(x_max)+step_x, step_x))
    ax.set_yticks(np.arange(np.floor(np.min(y)), np.ceil(np.max(y))+step_y, step_y))
    ax.set_zticks(np.arange(np.floor(z_min), np.ceil(z_max)+step_z, step_z))

    # View
    ax.view_init(elev=elev, azim=azim)

    # Professional legend
    custom_lines = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='navy', markersize=6, label='Filtered Points'),
        plt.Line2D([0], [0], color='forestgreen', lw=2, label=f'Y-axis (vertical)'),
        plt.Line2D([0], [0], color='orange', lw=2, label=f'Plane Normal ({theta_text})'),
        plt.Line2D([0], [0], color='lightcoral', lw=4, alpha=0.25, label='Best-fit Plane')
    ]
    ax.legend(handles=custom_lines, loc='upper left', fontsize=10)

    # Title
    ax.set_title('Flat Terrain Detection', fontsize=14, fontweight='bold')

    fig.tight_layout()
    return fig








# ... rest unchanged ...
# After metrics/markdown add this (note axis change also respected):

# ax.set_xlabel('X [m]'); ax.set_ylabel('Y [m]'); ax.set_zlabel('Z [m]')

# ------- Streamlit App ---------

st.title('Radar Point Cloud Landing Zone Assessment (ISO/IEEE Style)')
st.write('Upload a CSV (columns: timestamp, x, y, z, snr, noise). App filters, buffers frames, fits planes, and displays metrics.')

uploaded_file = st.file_uploader("Upload CSV file", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    required = {'timestamp', 'x', 'y', 'z', 'snr', 'noise'}
    if not required.issubset(df.columns):
        st.error(f'CSV must have columns: {",".join(required)}')
    else:
        # Group by timestamp (each timestamp = one scan)
        unique_ts = df['timestamp'].unique()
        buf_len = st.slider('Frame buffer length (step size)', min_value=1, max_value=25, value=8)
        overlap = st.slider('Frame Overlap (0=no overlap)', min_value=0, max_value=buf_len-1, value=2)

        frames = []
        idx = 0
        while idx + buf_len <= len(unique_ts):
            tgroup = unique_ts[idx:idx+buf_len]
            frame = df[df['timestamp'].isin(tgroup)]
            frames.append(frame)
            idx += buf_len - overlap if (buf_len - overlap) > 0 else buf_len

        st.info(f'Detected {len(frames)} buffer frames (total scans: {len(unique_ts)})')
        frame_idx = st.slider('Select Frame Index', min_value=0, max_value=len(frames)-1, value=0)

        radar_despiker = RadarDespiker()
        lz_assessor = LandingZoneAssessor()

        frame = frames[frame_idx]
        det_obj = {
            'x': frame['x'].to_numpy(),
            'y': frame['y'].to_numpy(),
            'z': frame['z'].to_numpy(),
            'numObj': len(frame['x'])
        }
        snr = frame['snr'].to_numpy()
        noise = frame['noise'].to_numpy()
        filtered = radar_despiker.process(det_obj, snr, noise)

        safe, metrics = lz_assessor.assess(filtered)

        # Display metrics
        st.subheader('Landing Zone Assessment Metrics')
        lines = [
            f"- **Safe to Land?** {'✅' if safe else '❌'}",
            f"- **Slope w.r.t Y (deg):** {metrics.get('slope_deg', float('nan')):.2f}",
            f"- **Confidence Score:** {metrics.get('confidence_score', float('nan')):.2f}",
            f"- **Inlier Ratio:** {metrics.get('inlier_ratio', float('nan')):.2f}",
            f"- **Surface Quality:** {metrics.get('surface_quality', float('nan')):.2f}",
            f"- **Mean Residual:** {metrics.get('mean_residual', float('nan')):.4f}"
        ]
        if not safe and metrics.get('reason', ''):
            lines.append(f"- **Reason (if not safe):** {metrics['reason']}")
        st.markdown("\n".join(lines))

        # Sidebar sliders
        st.sidebar.subheader("Axis Step Sizes")
        step_x = st.sidebar.slider("X-axis step", 0.1, 5.0, 0.5, 0.1)
        step_y = st.sidebar.slider("Y-axis step", 0.1, 5.0, 0.5, 0.1)
        step_z = st.sidebar.slider("Z-axis step", 0.1, 5.0, 0.5, 0.1)

        st.sidebar.subheader("View Angle")
        elev = st.sidebar.slider("Elevation", -90, 90, 25)
        azim = st.sidebar.slider("Azimuth", -180, 180, 35)

        # Plot only filtered points for the selected buffer frame
        fig = plot_plane_fit(filtered['x'], filtered['y'], filtered['z'],
                             metrics['plane'], metrics,
                             elev=elev, azim=azim,
                             step_x=step_x, step_y=step_y, step_z=step_z)
        st.pyplot(fig)
        plt.close(fig)




