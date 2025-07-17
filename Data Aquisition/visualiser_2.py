import sys
import os
import argparse
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
from pyqtgraph.opengl import GLViewWidget, GLAxisItem, GLGridItem, GLScatterPlotItem, GLTextItem

# ===== Logging =====
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ===== Data Model =====
@dataclass
class PointCloud:
    coords: np.ndarray    # shape (N,3)
    snr: np.ndarray       # shape (N,)
    cluster: np.ndarray   # shape (N,)
    timestamp: np.ndarray # shape (N,)

    @classmethod
    def from_csv(cls, path: str) -> 'PointCloud':
        df = pd.read_csv(path)
        required = ['x', 'y', 'z', 'snr', 'timestamp']
        for col in required:
            if col not in df.columns:
                logger.error(f"Missing required column: {col}")
                sys.exit(1)
        coords = df[['x', 'y', 'z']].to_numpy(float)
        snr = df['snr'].to_numpy(float)
        if 'cluster_index' in df.columns:
            cluster = df['cluster_index'].to_numpy(int)
        else:
            cluster = df.get('cluster', pd.Series(0, index=df.index)).to_numpy(int)
        timestamp = pd.to_datetime(df['timestamp'], errors='coerce').astype(np.int64)
        return cls(coords, snr, cluster, timestamp)

# ===== Visualizer =====
class PointCloudVisualizer(QtWidgets.QMainWindow):
    def __init__(self, pc: PointCloud, output_dir: str):
        super().__init__()
        self.setWindowTitle('Point Cloud Explorer')
        self.resize(1600, 1000)
        self.pc = pc
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Unique values
        self.clusters = sorted(np.unique(pc.cluster))
        self.timestamps = np.unique(pc.timestamp)

        # States
        self.snr_min = 0.0
        self.snr_max = pc.snr.max()
        self.time_mode_all = True
        self.point_size = 3.0
        self.cmap = 'viridis'
        self.current_mask = np.zeros(len(pc.coords), dtype=bool)

        # Build UI
        self._init_ui()
        self._init_gl()
        self._update_plot()

    def _init_ui(self):
        cw = QtWidgets.QWidget()
        self.setCentralWidget(cw)
        layout = QtWidgets.QHBoxLayout(cw)

        # 3D View
        self.view = GLViewWidget()
        self.view.opts['distance'] = 60
        layout.addWidget(self.view, stretch=4)

        # Control Panel
        ctrl = QtWidgets.QWidget()
        ctrl.setMinimumWidth(350)
        layout.addWidget(ctrl, stretch=1)
        vbox = QtWidgets.QVBoxLayout(ctrl)

        # Cluster selector
        vbox.addWidget(QtWidgets.QLabel('<b>Clusters</b>'))
        self.list_clusters = QtWidgets.QListWidget()
        self.list_clusters.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        for c in self.clusters:
            item = QtWidgets.QListWidgetItem(f'Cluster {c}')
            item.setData(QtCore.Qt.UserRole, c)
            item.setSelected(True)
            self.list_clusters.addItem(item)
        self.list_clusters.itemSelectionChanged.connect(self._update_plot)
        vbox.addWidget(self.list_clusters)

        # SNR range
        vbox.addWidget(QtWidgets.QLabel('<b>SNR Min/Max</b>'))
        h_snr = QtWidgets.QHBoxLayout()
        self.spin_snr_min = QtWidgets.QDoubleSpinBox()
        self.spin_snr_min.setRange(self.snr_min, self.snr_max)
        self.spin_snr_min.setValue(self.snr_min)
        self.spin_snr_min.valueChanged.connect(self._update_plot)
        self.spin_snr_max = QtWidgets.QDoubleSpinBox()
        self.spin_snr_max.setRange(self.snr_min, self.snr_max)
        self.spin_snr_max.setValue(self.snr_max)
        self.spin_snr_max.valueChanged.connect(self._update_plot)
        h_snr.addWidget(self.spin_snr_min)
        h_snr.addWidget(self.spin_snr_max)
        vbox.addLayout(h_snr)

        # Timestamp mode
        self.btn_time = QtWidgets.QPushButton('Mode: All ≤ time')
        self.btn_time.setCheckable(True)
        self.btn_time.toggled.connect(self._update_plot)
        vbox.addWidget(self.btn_time)
        vbox.addWidget(QtWidgets.QLabel('<b>Timestamp</b>'))
        self.slider_time = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_time.setRange(0, len(self.timestamps) - 1)
        self.slider_time.valueChanged.connect(self._update_plot)
        vbox.addWidget(self.slider_time)
        self.lbl_time = QtWidgets.QLabel('')
        vbox.addWidget(self.lbl_time)

        # Point size and colormap
        vbox.addWidget(QtWidgets.QLabel('<b>Point Size</b>'))
        self.spin_size = QtWidgets.QDoubleSpinBox()
        self.spin_size.setRange(1, 10)
        self.spin_size.setValue(self.point_size)
        self.spin_size.valueChanged.connect(self._update_plot)
        vbox.addWidget(self.spin_size)

        vbox.addWidget(QtWidgets.QLabel('<b>Colormap</b>'))
        self.combo_cmap = QtWidgets.QComboBox()
        self.combo_cmap.addItems(pg.colormap.listMaps())
        self.combo_cmap.setCurrentText(self.cmap)
        self.combo_cmap.currentTextChanged.connect(self._update_plot)
        vbox.addWidget(self.combo_cmap)

        # Export & Inference buttons
        btn_csv = QtWidgets.QPushButton('Export Visible CSV')
        btn_csv.clicked.connect(self._export_csv)
        vbox.addWidget(btn_csv)

        btn_img = QtWidgets.QPushButton('Save Screenshot')
        btn_img.clicked.connect(self._save_image)
        vbox.addWidget(btn_img)

        btn_inf = QtWidgets.QPushButton('Compute Centroids')
        btn_inf.clicked.connect(self._compute_centroids)
        vbox.addWidget(btn_inf)
        self.lbl_inf = QtWidgets.QLabel('')
        vbox.addWidget(self.lbl_inf)

        vbox.addStretch()

    def _init_gl(self):
        self.view.clear()
        # axes and grid
        self.view.addItem(GLAxisItem())
        grid = GLGridItem()
        grid.scale(10, 10, 1)
        self.view.addItem(grid)
        # axis text
        ext = np.max(np.abs(self.pc.coords), axis=0)
        labels = [('X', (ext[0], 0, 0)), ('Y', (0, ext[1], 0)), ('Z', (0, 0, ext[2]))]
        for txt, pos in labels:
            ti = GLTextItem(text=txt, pos=pos,
                            font=QtGui.QFont('Arial', 12), color=(1,1,1,1))
            self.view.addItem(ti)

    def _filter_mask(self):
        mask = np.isin(self.pc.cluster,
                        [it.data(QtCore.Qt.UserRole) for it in self.list_clusters.selectedItems()])
        mn = self.spin_snr_min.value()
        mx = self.spin_snr_max.value()
        mask &= (self.pc.snr >= mn) & (self.pc.snr <= mx)
        idx = self.slider_time.value()
        ts_val = self.timestamps[idx]
        self.lbl_time.setText(
            pd.to_datetime(int(ts_val)).strftime('%Y-%m-%d %H:%M:%S')
        )
        if not self.btn_time.isChecked():
            mask &= (self.pc.timestamp <= ts_val)
        else:
            mask &= (self.pc.timestamp == ts_val)
        return mask

    def _update_plot(self):
        mask = self._filter_mask()
        self.current_mask = mask
        self._init_gl()
        pts = self.pc.coords[mask]
        snr = self.pc.snr[mask]
        if pts.size == 0:
            self.lbl_inf.clear()
            return
        cmap = pg.colormap.get(self.combo_cmap.currentText())
        norm = (snr - snr.min()) / (np.ptp(snr) if np.ptp(snr)>0 else 1)
        colors = cmap.map(norm, mode='float')
        sp = GLScatterPlotItem(pos=pts, size=self.spin_size.value(),
                               color=colors, pxMode=True)
        self.view.addItem(sp)

    def _export_csv(self):
        mask = self.current_mask
        df = pd.DataFrame(self.pc.coords[mask], columns=['x','y','z'])
        df['snr'] = self.pc.snr[mask]
        df['cluster'] = self.pc.cluster[mask]
        df['timestamp'] = self.pc.timestamp[mask]
        path = os.path.join(self.output_dir, 'visible.csv')
        df.to_csv(path, index=False)
        logger.info(f"Exported {len(df)} points to {path}")

    def _save_image(self):
        path = os.path.join(self.output_dir, 'screenshot.png')
        pg.exporters.ImageExporter(self.view.renderToArray()).export(path)
        logger.info(f"Saved screenshot to {path}")

    def _compute_centroids(self):
        mask = self.current_mask
        pts = self.pc.coords[mask]
        clusters = self.pc.cluster[mask]
        out = []
        for c in np.unique(clusters):
            sel = pts[clusters==c]
            cen = sel.mean(axis=0)
            out.append(
                f"Cluster {c}: Count={len(sel)}, Centroid={tuple(cen.round(2))}"
            )
        self.lbl_inf.setText('<br>'.join(out))

    def run(self):
        self.show()

# Entrypoint
if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description='3D Point Cloud Explorer with SNR & clustering'
    )
    parser.add_argument('input_csv', help='Path to point cloud CSV')
    parser.add_argument('output_dir', help='Directory to save outputs')
    args = parser.parse_args()

    pc = PointCloud.from_csv(args.input_csv)
    app = QtWidgets.QApplication([])
    viz = PointCloudVisualizer(pc, args.output_dir)
    viz.run()
    sys.exit(app.exec_())
