# import datetime
# import numpy as np
# import time
# import matplotlib.pyplot as plt
# import pandas as pd
# import datetime

# class RadarPlotter:
#     """
#     RadarPlotter handles professional ISO-standard plotting & data export:

#     - Real-time 3D scatter of raw vs. smoothed points (distinct markers/colors).
#     - History 3D scatter of all past points.
#     - Temporal line plots of mean X, Y, Z over frames.
#     - Tabular view and CSV export of history with timestamps.
#     """
#     def __init__(self):
#         # ISO-standard figure settings
#         plt.rc('font', size=10)
#         plt.rc('axes', grid=True, facecolor='white')

#         # Real-time 3D
#         self.fig3d = plt.figure(figsize=(6,6))
#         self.ax3d = self.fig3d.add_subplot(111, projection='3d')
#         plt.show(block=False)
#         self.ax3d.set_title('Real-time 3D Radar View', pad=20)
#         self.ax3d.set_box_aspect((1,1,1))
#         self._setup_3d_axes(self.ax3d)

#         # History 3D
#         self.fig_hist = plt.figure(figsize=(6,6))
#         self.ax_hist = self.fig_hist.add_subplot(111, projection='3d')
#         plt.show(block=False)
#         self.ax_hist.set_title('History of All Data Points (3D)', pad=20)
#         self.ax_hist.set_box_aspect((1,1,1))
#         self._setup_3d_axes(self.ax_hist)

#         # Data storage
#         self.history_raw = []
#         self.history_smooth = []
#         self.frame_times = []  # timestamps per frame

#         # Temporal
#         self.fig_temp, self.ax_temp = plt.subplots(3, 1, figsize=(6, 9))
#         plt.show(block=False)
#         coords = ['X-axis (m)', 'Y-axis (m)', 'Z-axis (m)']
#         for ax, coord in zip(self.ax_temp, coords):
#             ax.set_title(f'Temporal {coord.split()[0]}', pad=10)
#             ax.set_xlabel('Frame', labelpad=8)
#             ax.set_ylabel(coord, labelpad=8)
#             ax.grid(True)
#         self.temporal_raw = {'x': [], 'y': [], 'z': []}
#         self.temporal_smooth = {'x': [], 'y': [], 'z': []}
#         self.frame_idx = 0

#     def _setup_3d_axes(self, ax):
#         ax.clear()
#         ax.set_xlabel('X-axis (m)', color='red', labelpad=10)
#         ax.set_ylabel('Y-axis (m)', color='green', labelpad=10)
#         ax.set_zlabel('Z-axis (m)', color='blue', labelpad=10)
#         arrow_len = 1.0
#         for vec, color in [((1,0,0),'red'), ((0,1,0),'green'), ((0,0,1),'blue')]:
#             ax.quiver(0,0,0, *vec, length=arrow_len, color=color, arrow_length_ratio=0.1)
#         ax.text(arrow_len,0,0,'X', color='red')
#         ax.text(0,arrow_len,0,'Y', color='green')
#         ax.text(0,0,arrow_len,'Z', color='blue')
#         ax.set_xlim(-5,5); ax.set_ylim(-5,5); ax.set_zlim(-5,5)

#     # rest unchanged

#     def update(self, det_obj, sm_obj):
#         self.frame_times.append(time.time())
#         self.fig_temp, self.ax_temp = plt.subplots(3, 1, figsize=(6, 9))
#         plt.show(block=False)  # non-blocking display for temporal plots
#         self.fig_temp.show(block=False)
#         coords = ['X-axis (m)', 'Y-axis (m)', 'Z-axis (m)']
#         for ax, coord in zip(self.ax_temp, coords):
#             ax.set_title(f'Temporal {coord.split()[0]}', pad=10)
#             ax.set_xlabel('Frame', labelpad=8)
#             ax.set_ylabel(coord, labelpad=8)
#             ax.grid(True)
#         self.temporal_raw = {'x': [], 'y': [], 'z': []}
#         self.temporal_smooth = {'x': [], 'y': [], 'z': []}
#         self.frame_idx = 0

#     def _setup_3d_axes(self, ax):
#         ax.clear()
#         ax.set_xlabel('X-axis (m)', color='red', labelpad=10)
#         ax.set_ylabel('Y-axis (m)', color='green', labelpad=10)
#         ax.set_zlabel('Z-axis (m)', color='blue', labelpad=10)
#         arrow_len = 1.0
#         for vec, color in [((1,0,0),'red'), ((0,1,0),'green'), ((0,0,1),'blue')]:
#             ax.quiver(0,0,0, *vec, length=arrow_len, color=color, arrow_length_ratio=0.1)
#         ax.text(arrow_len,0,0,'X', color='red')
#         ax.text(0,arrow_len,0,'Y', color='green')
#         ax.text(0,0,arrow_len,'Z', color='blue')
#         ax.set_xlim(-5,5); ax.set_ylim(-5,5); ax.set_zlim(-5,5)

#     def update(self, det_obj, sm_obj):
#         self.frame_times.append(time.time())

#         # Real-time 3D
#         self._setup_3d_axes(self.ax3d)
#         plotted = False
#         if det_obj and det_obj.get('numObj', 0) > 0:
#             raw_pts = np.vstack((det_obj['x'], det_obj['y'], det_obj['z'])).T
#             self.ax3d.scatter(raw_pts[:,0], raw_pts[:,1], raw_pts[:,2], marker='o', s=20, alpha=0.7, label='Raw', edgecolor='k')
#             self.history_raw.append(raw_pts)
#             plotted = True
#         if sm_obj and sm_obj.get('numObj', 0) > 0:
#             sm_pts = np.vstack((sm_obj['x'], sm_obj['y'], sm_obj['z'])).T
#             self.ax3d.scatter(sm_pts[:,0], sm_pts[:,1], sm_pts[:,2], marker='^', s=30, alpha=0.7, label='Smoothed', edgecolor='k')
#             self.history_smooth.append(sm_pts)
#             plotted = True
#         if not plotted:
#             self.ax3d.text2D(0.5,0.5,'No data', transform=self.ax3d.transAxes, ha='center', va='center', fontsize=12, color='red')
#         if plotted:
#             self.ax3d.legend(loc='upper left')
#         self.fig3d.canvas.draw(); self.fig3d.canvas.flush_events(); plt.pause(0.001)

#         # History 3D
#         self._setup_3d_axes(self.ax_hist)
#         for pts in self.history_raw:
#             self.ax_hist.scatter(pts[:,0], pts[:,1], pts[:,2], alpha=0.1, c='tab:blue', marker='o')
#         for pts in self.history_smooth:
#             self.ax_hist.scatter(pts[:,0], pts[:,1], pts[:,2], alpha=0.3, c='tab:orange', marker='^')
#         self.fig_hist.canvas.draw(); self.fig_hist.canvas.flush_events(); plt.pause(0.001)

#         # Temporal
#         self.frame_idx += 1
#         for coord in ['x','y','z']:
#             rv = np.array(det_obj.get(coord,[])) if det_obj else np.array([])
#             sv = np.array(sm_obj.get(coord,[])) if sm_obj else np.array([])
#             self.temporal_raw[coord].append(np.nanmean(rv) if rv.size>0 else np.nan)
#             self.temporal_smooth[coord].append(np.nanmean(sv) if sv.size>0 else np.nan)
#         for ax, coord in zip(self.ax_temp, ['x','y','z']):
#             ax.cla()
#             ax.set_xlabel('Frame', labelpad=8)
#             ax.set_ylabel(f'{coord.upper()} (m)', labelpad=8)
#             ax.grid(True)
#             ax.plot(self.temporal_raw[coord], linestyle='-', marker='o', markersize=4, label='Raw')
#             ax.plot(self.temporal_smooth[coord], linestyle='-', marker='^', markersize=6, label='Smoothed')
#             ax.legend(loc='upper right')
#         self.fig_temp.tight_layout();
#         self.fig_temp.canvas.draw(); self.fig_temp.canvas.flush_events(); plt.pause(0.001)

#     def create_history_dataframe(self):
#         records = []
#         for idx, pts in enumerate(self.history_raw):
#             ts = self.frame_times[idx]
#             for x,y,z in pts:
#                 records.append({'timestamp':ts,'frame':idx,'type':'raw','x':x,'y':y,'z':z})
#         for idx, pts in enumerate(self.history_smooth):
#             ts = self.frame_times[idx]
#             for x,y,z in pts:
#                 records.append({'timestamp':ts,'frame':idx,'type':'smoothed','x':x,'y':y,'z':z})
#         return pd.DataFrame.from_records(records)

#     def export_history_csv(self, filename=None):
#         df = self.create_history_dataframe()
#         if filename is None:
#             ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
#             filename = f"radar_history_{ts}.csv"
#         df.to_csv(filename, index=False)
#         print(f"History with timestamps exported to {filename}")

#     def show_history_table(self, num_rows=50):
#         df = self.create_history_dataframe().head(num_rows)
#         print(df.to_string(index=False))


#For IDE
import datetime
import numpy as np
import time
import matplotlib
matplotlib.use('QtAgg')  # Force PyCharm to use GUI-compatible backend
import matplotlib.pyplot as plt
import pandas as pd


class RadarPlotter:
    """
    RadarPlotter handles professional ISO-standard plotting & data export:

    - Real-time 3D scatter of raw vs. smoothed points (distinct markers/colors).
    - History 3D scatter of all past points.
    - Temporal line plots of mean X, Y, Z over frames.
    - Tabular view and CSV export of history with timestamps.
    """

    def __init__(self):
        plt.ion()  # Enable interactive mode for PyCharm compatibility

        # ISO-standard figure settings
        plt.rc('font', size=10)
        plt.rc('axes', grid=True, facecolor='white')

        # Real-time 3D
        self.fig3d = plt.figure(figsize=(6, 6))
        self.ax3d = self.fig3d.add_subplot(111, projection='3d')
        self.ax3d.set_title('Real-time 3D Radar View', pad=20)
        self.ax3d.set_box_aspect((1, 1, 1))
        self._setup_3d_axes(self.ax3d)
        plt.show(block=False)

        # History 3D
        self.fig_hist = plt.figure(figsize=(6, 6))
        self.ax_hist = self.fig_hist.add_subplot(111, projection='3d')
        self.ax_hist.set_title('History of All Data Points (3D)', pad=20)
        self.ax_hist.set_box_aspect((1, 1, 1))
        self._setup_3d_axes(self.ax_hist)
        plt.show(block=False)

        # Data storage
        self.history_raw = []
        self.history_smooth = []
        self.frame_times = []  # timestamps per frame

        # Temporal
        self.fig_temp, self.ax_temp = plt.subplots(3, 1, figsize=(6, 9))
        coords = ['X-axis (m)', 'Y-axis (m)', 'Z-axis (m)']
        for ax, coord in zip(self.ax_temp, coords):
            ax.set_title(f'Temporal {coord.split()[0]}', pad=10)
            ax.set_xlabel('Frame', labelpad=8)
            ax.set_ylabel(coord, labelpad=8)
            ax.grid(True)
        plt.show(block=False)

        self.temporal_raw = {'x': [], 'y': [], 'z': []}
        self.temporal_smooth = {'x': [], 'y': [], 'z': []}
        self.frame_idx = 0

    def _setup_3d_axes(self, ax):
        ax.clear()
        ax.set_xlabel('X-axis (m)', color='red', labelpad=10)
        ax.set_ylabel('Y-axis (m)', color='green', labelpad=10)
        ax.set_zlabel('Z-axis (m)', color='blue', labelpad=10)
        arrow_len = 1.0
        for vec, color in [((1, 0, 0), 'red'), ((0, 1, 0), 'green'), ((0, 0, 1), 'blue')]:
            ax.quiver(0, 0, 0, *vec, length=arrow_len, color=color, arrow_length_ratio=0.1)
        ax.text(arrow_len, 0, 0, 'X', color='red')
        ax.text(0, arrow_len, 0, 'Y', color='green')
        ax.text(0, 0, arrow_len, 'Z', color='blue')
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_zlim(-5, 5)

    def update(self, det_obj, sm_obj):
        self.frame_times.append(time.time())

        # Real-time 3D
        self._setup_3d_axes(self.ax3d)
        plotted = False
        if det_obj and det_obj.get('numObj', 0) > 0:
            raw_pts = np.vstack((det_obj['x'], det_obj['y'], det_obj['z'])).T
            self.ax3d.scatter(raw_pts[:, 0], raw_pts[:, 1], raw_pts[:, 2], marker='o', s=20, alpha=0.7,
                              label='Raw', edgecolor='k')
            self.history_raw.append(raw_pts)
            plotted = True
        if sm_obj and sm_obj.get('numObj', 0) > 0:
            sm_pts = np.vstack((sm_obj['x'], sm_obj['y'], sm_obj['z'])).T
            self.ax3d.scatter(sm_pts[:, 0], sm_pts[:, 1], sm_pts[:, 2], marker='^', s=30, alpha=0.7,
                              label='Smoothed', edgecolor='k')
            self.history_smooth.append(sm_pts)
            plotted = True
        if not plotted:
            self.ax3d.text2D(0.5, 0.5, 'No data', transform=self.ax3d.transAxes, ha='center', va='center',
                             fontsize=12, color='red')
        if plotted:
            self.ax3d.legend(loc='upper left')
        self.fig3d.canvas.draw_idle()
        plt.draw()
        self.fig3d.canvas.flush_events()
        plt.pause(0.001)

        # History 3D
        self._setup_3d_axes(self.ax_hist)
        for pts in self.history_raw:
            self.ax_hist.scatter(pts[:, 0], pts[:, 1], pts[:, 2], alpha=0.1, c='tab:blue', marker='o')
        for pts in self.history_smooth:
            self.ax_hist.scatter(pts[:, 0], pts[:, 1], pts[:, 2], alpha=0.3, c='tab:orange', marker='^')
        self.fig_hist.canvas.draw_idle()
        plt.draw()
        self.fig_hist.canvas.flush_events()
        plt.pause(0.001)

        # Temporal
        self.frame_idx += 1
        for coord in ['x', 'y', 'z']:
            rv = np.array(det_obj.get(coord, [])) if det_obj else np.array([])
            sv = np.array(sm_obj.get(coord, [])) if sm_obj else np.array([])
            self.temporal_raw[coord].append(np.nanmean(rv) if rv.size > 0 else np.nan)
            self.temporal_smooth[coord].append(np.nanmean(sv) if sv.size > 0 else np.nan)
        for ax, coord in zip(self.ax_temp, ['x', 'y', 'z']):
            ax.cla()
            ax.set_xlabel('Frame', labelpad=8)
            ax.set_ylabel(f'{coord.upper()} (m)', labelpad=8)
            ax.grid(True)
            ax.plot(self.temporal_raw[coord], linestyle='-', marker='o', markersize=4, label='Raw')
            ax.plot(self.temporal_smooth[coord], linestyle='-', marker='^', markersize=6, label='Smoothed')
            ax.legend(loc='upper right')
        self.fig_temp.tight_layout()
        self.fig_temp.canvas.draw_idle()
        plt.draw()
        self.fig_temp.canvas.flush_events()
        plt.pause(0.001)

    def create_history_dataframe(self):
        records = []
        for idx, pts in enumerate(self.history_raw):
            ts = self.frame_times[idx]
            for x, y, z in pts:
                records.append({'timestamp': ts, 'frame': idx, 'type': 'raw', 'x': x, 'y': y, 'z': z})
        for idx, pts in enumerate(self.history_smooth):
            ts = self.frame_times[idx]
            for x, y, z in pts:
                records.append({'timestamp': ts, 'frame': idx, 'type': 'smoothed', 'x': x, 'y': y, 'z': z})
        return pd.DataFrame.from_records(records)

    def export_history_csv(self, filename=None):
        df = self.create_history_dataframe()
        if filename is None:
            ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"radar_history_{ts}.csv"
        df.to_csv(filename, index=False)
        print(f"History with timestamps exported to {filename}")

    def show_history_table(self, num_rows=50):
        df = self.create_history_dataframe().head(num_rows)
        print(df.to_string(index=False))
