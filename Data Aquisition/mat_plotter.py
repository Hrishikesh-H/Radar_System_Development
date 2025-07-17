import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.widgets import CheckButtons
import numpy as np

class PointCloud3DPlotter:
    def __init__(self, csv_file: str):
        self.data = pd.read_csv(csv_file)
        self.cluster_ids = sorted(self.data['cluster_index'].dropna().unique())
        self.active_clusters = {cluster_id: True for cluster_id in self.cluster_ids}
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(121, projection='3d')
        self.check_ax = self.fig.add_subplot(122)
        self.scatter_objects = {}
        self.annot = None
        self._setup_plot()
        self._setup_checkboxes()
        self._setup_hover()

    def _setup_plot(self):
        self.ax.set_title("3D Point Cloud by Cluster")
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_zlabel("Z (m)")
        self.ax.grid(True)
        self.ax.view_init(elev=15, azim=135)

        cmap = plt.get_cmap('tab10')
        for idx, cluster_id in enumerate(self.cluster_ids):
            cluster_data = self.data[self.data['cluster_index'] == cluster_id]
            color = cmap(idx % 10)
            sc = self.ax.scatter(
                cluster_data['x'], cluster_data['y'], cluster_data['z'],
                label=f"Cluster {cluster_id}",
                color=color, s=20, alpha=0.8, picker=True
            )
            self.scatter_objects[cluster_id] = sc

    def _setup_checkboxes(self):
        self.check_ax.clear()
        self.check_ax.set_title("Toggle Clusters")
        labels = [f"Cluster {cid}" for cid in self.cluster_ids]
        visibility = [self.active_clusters[cid] for cid in self.cluster_ids]
        self.checkbox = CheckButtons(self.check_ax, labels, visibility)
        self.checkbox.on_clicked(self._on_check)

    def _on_check(self, label):
        cid = int(label.split()[-1])
        current = self.active_clusters[cid]
        self.active_clusters[cid] = not current
        self.scatter_objects[cid].set_visible(not current)
        plt.draw()

    def _setup_hover(self):
        # Annotation box for tooltip
        self.annot = self.ax.text2D(0.05, 0.95, "", transform=self.ax.transAxes,
                                    bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7),
                                    fontsize=10, visible=False)

        self.fig.canvas.mpl_connect("motion_notify_event", self._on_hover)

    def _on_hover(self, event):
        if event.inaxes != self.ax:
            self.annot.set_visible(False)
            self.fig.canvas.draw_idle()
            return

        closest_info = None
        min_dist = float('inf')

        for cid in self.cluster_ids:
            if not self.active_clusters[cid]:
                continue
            sc = self.scatter_objects[cid]
            coords = sc._offsets3d
            xdata, ydata, zdata = coords
            xdata = np.asarray(xdata)
            ydata = np.asarray(ydata)
            zdata = np.asarray(zdata)
            proj = self.ax.transData.transform(np.vstack([xdata, ydata]).T)

            for i, (x_screen, y_screen) in enumerate(proj):
                dist = np.hypot(x_screen - event.x, y_screen - event.y)
                if dist < min_dist and dist < 10:  # Threshold in screen px
                    min_dist = dist
                    closest_info = {
                        'x': xdata[i],
                        'y': ydata[i],
                        'z': zdata[i],
                        'cluster': cid
                    }

        if closest_info:
            text = (f"Cluster: {closest_info['cluster']}\n"
                    f"X: {closest_info['x']:.2f}\n"
                    f"Y: {closest_info['y']:.2f}\n"
                    f"Z: {closest_info['z']:.2f}")
            self.annot.set_text(text)
            self.annot.set_visible(True)
        else:
            self.annot.set_visible(False)

        self.fig.canvas.draw_idle()

    def show(self):
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    file_path = r"b:\IISc AIRL\hh_mmwave\Radar_System_Development\Data Aquisition\field_test_9_20250704_182803_4.0s_10.0m.csv"
    plotter = PointCloud3DPlotter(file_path)
    plotter.show()
