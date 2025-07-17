import sys
import os
import argparse
import logging
from dataclasses import dataclass
import time
import numpy as np
import pandas as pd
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import QVector4D, QVector3D
import pyqtgraph as pg
from pyqtgraph.opengl import (GLViewWidget, GLAxisItem, GLGridItem, 
                             GLScatterPlotItem, GLTextItem, GLLinePlotItem,
                             GLMeshItem)

# ===== Logging Configuration =====
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
        """Load point cloud data from CSV file"""
        try:
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
            
            # Parse timestamps
            timestamps = pd.to_datetime(df['timestamp'], errors='coerce')
            return cls(coords, snr, cluster, timestamps)
            
        except FileNotFoundError:
            logger.error(f"File not found: {path}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error reading CSV file {path}: {e}")
            sys.exit(1)

# ===== Visualizer =====
class PointCloudVisualizer(QtWidgets.QMainWindow):
    def __init__(self, pc: PointCloud, output_dir: str):
        super().__init__()
        self.setWindowTitle('Point Cloud Explorer')
        self.resize(1800, 1000)
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
        self.hover_point = None
        self.plane_systems = {
            'x': {'enabled': False, 'planes': [None, None], 'values': [0.0, 0.0]},
            'y': {'enabled': False, 'planes': [None, None], 'values': [0.0, 0.0]},
            'z': {'enabled': False, 'planes': [None, None], 'values': [0.0, 0.0]}
        }
        self.plane_axis = 0  # 0=x, 1=y, 2=z
        self.visual_shape = None  # Current visualization shape (box, sphere, cone)
        self.initial_view = None
        self.filtered_scatter = None  # For points between planes
        self.last_hover_time = 0  # For hover throttling
        self.plane_filter_active = False  # Track if plane filter is active
        self.plane_mask = None  # Store plane filter mask
        self.show_line_graphs = False  # Track line graph visibility

        # Calculate bounding box
        self.min_coords = np.min(pc.coords, axis=0)
        self.max_coords = np.max(pc.coords, axis=0)
        self.range_coords = self.max_coords - self.min_coords
        self.centroid = np.mean(pc.coords, axis=0)
        self.max_dimension = np.max(self.range_coords)

        # Build UI
        self._init_ui()
        self._init_gl()
        self._update_plot()

    def _init_ui(self):
        """Initialize the user interface"""
        cw = QtWidgets.QWidget()
        self.setCentralWidget(cw)
        layout = QtWidgets.QHBoxLayout(cw)

        # 3D View
        self.view = GLViewWidget()
        self.view.opts['distance'] = 60
        self.view.mouseMoveEvent = self._handle_mouse_move
        layout.addWidget(self.view, stretch=4)

        # Control Panel
        ctrl = QtWidgets.QWidget()
        ctrl.setMinimumWidth(400)
        layout.addWidget(ctrl, stretch=1)
        vbox = QtWidgets.QVBoxLayout(ctrl)

        # Status bar for coordinates
        self.status_bar = QtWidgets.QLabel("Hover over a point to see coordinates")
        self.status_bar.setStyleSheet("background-color: #333; color: #EEE; padding: 5px;")
        vbox.addWidget(self.status_bar)

        # Cluster selector
        cluster_header = QtWidgets.QHBoxLayout()
        cluster_header.addWidget(QtWidgets.QLabel('<b>Clusters</b>'))
        
        btn_select_all = QtWidgets.QPushButton('All')
        btn_select_all.clicked.connect(lambda: self._set_cluster_selection(True))
        cluster_header.addWidget(btn_select_all)
        
        btn_select_none = QtWidgets.QPushButton('None')
        btn_select_none.clicked.connect(lambda: self._set_cluster_selection(False))
        cluster_header.addWidget(btn_select_none)
        
        vbox.addLayout(cluster_header)
        
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
        self.btn_time.toggled.connect(self._update_timestamp_mode)
        vbox.addWidget(self.btn_time)
        vbox.addWidget(QtWidgets.QLabel('<b>Timestamp</b>'))
        self.slider_time = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_time.setRange(0, len(self.timestamps) - 1)
        self.slider_time.valueChanged.connect(self._update_plot)
        vbox.addWidget(self.slider_time)
        self.lbl_time = QtWidgets.QLabel('')
        vbox.addWidget(self.lbl_time)

        # Point visualization
        vbox.addWidget(QtWidgets.QLabel('<b>Point Visualization</b>'))
        point_layout = QtWidgets.QHBoxLayout()
        
        # Point size
        point_layout.addWidget(QtWidgets.QLabel('Size:'))
        self.spin_size = QtWidgets.QDoubleSpinBox()
        self.spin_size.setRange(1, 10)
        self.spin_size.setValue(self.point_size)
        self.spin_size.valueChanged.connect(self._update_plot)
        point_layout.addWidget(self.spin_size)
        
        # Colormap
        point_layout.addWidget(QtWidgets.QLabel('Colormap:'))
        self.combo_cmap = QtWidgets.QComboBox()
        self.combo_cmap.addItems(pg.colormap.listMaps())
        self.combo_cmap.setCurrentText(self.cmap)
        self.combo_cmap.currentTextChanged.connect(self._update_plot)
        point_layout.addWidget(self.combo_cmap)
        
        vbox.addLayout(point_layout)

        # Plane filter section
        plane_group = QtWidgets.QGroupBox("Plane Filter")
        plane_layout = QtWidgets.QVBoxLayout(plane_group)
        
        # Axis selection
        axis_layout = QtWidgets.QHBoxLayout()
        axis_layout.addWidget(QtWidgets.QLabel('Axis:'))
        self.combo_plane_axis = QtWidgets.QComboBox()
        self.combo_plane_axis.addItems(['X', 'Y', 'Z'])
        self.combo_plane_axis.setCurrentIndex(0)
        self.combo_plane_axis.currentIndexChanged.connect(self._on_plane_axis_changed)
        axis_layout.addWidget(self.combo_plane_axis)
        plane_layout.addLayout(axis_layout)
        
        # Plane enable checkboxes
        plane_enable_layout = QtWidgets.QHBoxLayout()
        self.cb_plane_x = QtWidgets.QCheckBox('Enable X planes')
        self.cb_plane_x.setChecked(True)  # Enable by default
        self.cb_plane_x.stateChanged.connect(lambda: self._toggle_plane_system('x'))
        plane_enable_layout.addWidget(self.cb_plane_x)
        
        self.cb_plane_y = QtWidgets.QCheckBox('Enable Y planes')
        self.cb_plane_y.stateChanged.connect(lambda: self._toggle_plane_system('y'))
        plane_enable_layout.addWidget(self.cb_plane_y)
        
        self.cb_plane_z = QtWidgets.QCheckBox('Enable Z planes')
        self.cb_plane_z.stateChanged.connect(lambda: self._toggle_plane_system('z'))
        plane_enable_layout.addWidget(self.cb_plane_z)
        plane_layout.addLayout(plane_enable_layout)
        
        # Plane sliders
        self.slider_plane1 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_plane1.valueChanged.connect(self._update_planes)
        plane_layout.addWidget(QtWidgets.QLabel('Plane 1 Position:'))
        plane_layout.addWidget(self.slider_plane1)
        
        self.slider_plane2 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider_plane2.valueChanged.connect(self._update_planes)
        plane_layout.addWidget(QtWidgets.QLabel('Plane 2 Position:'))
        plane_layout.addWidget(self.slider_plane2)
        
        # Plane values display
        self.lbl_plane1 = QtWidgets.QLabel('Plane 1: 0.00')
        self.lbl_plane2 = QtWidgets.QLabel('Plane 2: 0.00')
        plane_layout.addWidget(self.lbl_plane1)
        plane_layout.addWidget(self.lbl_plane2)
        
        # Plane actions
        btn_layout = QtWidgets.QHBoxLayout()
        self.btn_show_planes = QtWidgets.QPushButton('Show Planes')
        self.btn_show_planes.setCheckable(True)
        self.btn_show_planes.setChecked(True)
        self.btn_show_planes.toggled.connect(self._toggle_planes)
        btn_layout.addWidget(self.btn_show_planes)
        
        self.btn_filter_planes = QtWidgets.QPushButton('Filter Points')
        self.btn_filter_planes.clicked.connect(self._filter_points_between_planes)
        btn_layout.addWidget(self.btn_filter_planes)
        
        self.btn_clear_filter = QtWidgets.QPushButton('Clear Filter')
        self.btn_clear_filter.clicked.connect(self._clear_filtered_points)
        btn_layout.addWidget(self.btn_clear_filter)
        plane_layout.addLayout(btn_layout)
        
        vbox.addWidget(plane_group)

        # Visualization tools
        tools_group = QtWidgets.QGroupBox("Visualization Tools")
        tools_layout = QtWidgets.QGridLayout(tools_group)
        
        # Shape visualization
        tools_layout.addWidget(QtWidgets.QLabel('Visualization Shape:'), 0, 0)
        self.combo_shape = QtWidgets.QComboBox()
        self.combo_shape.addItems(['None', 'Bounding Box', 'Sphere', 'Cone'])
        self.combo_shape.currentIndexChanged.connect(self._update_shape_visualization)
        tools_layout.addWidget(self.combo_shape, 0, 1)
        
        # Axes visualization
        self.cb_axes = QtWidgets.QCheckBox('Enhanced Axes')
        self.cb_axes.setChecked(True)
        self.cb_axes.stateChanged.connect(self._toggle_axes)
        tools_layout.addWidget(self.cb_axes, 1, 0, 1, 2)
        
        # Line graphs
        self.btn_line_graphs = QtWidgets.QPushButton('Show Line Graphs')
        self.btn_line_graphs.setCheckable(True)
        self.btn_line_graphs.toggled.connect(self._toggle_line_graphs)
        tools_layout.addWidget(self.btn_line_graphs, 2, 0, 1, 2)
        
        vbox.addWidget(tools_group)

        # Export & Analysis
        export_group = QtWidgets.QGroupBox("Export & Analysis")
        export_layout = QtWidgets.QHBoxLayout(export_group)
        
        # Export buttons
        btn_csv = QtWidgets.QPushButton('Export CSV')
        btn_csv.clicked.connect(self._export_csv)
        export_layout.addWidget(btn_csv)

        btn_img = QtWidgets.QPushButton('Screenshot')
        btn_img.clicked.connect(self._save_image)
        export_layout.addWidget(btn_img)
        
        # Analysis buttons
        btn_inf = QtWidgets.QPushButton('Centroids')
        btn_inf.clicked.connect(self._compute_centroids)
        export_layout.addWidget(btn_inf)
        
        btn_reset = QtWidgets.QPushButton('Reset View')
        btn_reset.clicked.connect(self._reset_view)
        export_layout.addWidget(btn_reset)
        
        vbox.addWidget(export_group)
        
        # Centroid display label
        self.lbl_centroids = QtWidgets.QLabel('')
        vbox.addWidget(self.lbl_centroids)

        vbox.addStretch()
        
        # Initialize plane systems
        self.plane_systems['x']['enabled'] = True  # Enable X planes by default
        self._init_plane_sliders()
        
        # Initialize line graph window
        self._init_line_graph_window()

    def _init_gl(self):
        """Initialize OpenGL view with axes and grid"""
        self.view.clear()
        
        # Add grid
        grid = GLGridItem()
        grid.setSize(x=self.range_coords[0], y=self.range_coords[1], z=1)
        grid.setSpacing(x=1, y=1)
        self.view.addItem(grid)
        
        # Enhanced axes visualization
        self._add_enhanced_axes()
        
        # Store initial view parameters
        self.initial_view = {
            'center': self.view.opts['center'],
            'distance': self.view.opts['distance'],
            'elevation': self.view.opts['elevation'],
            'azimuth': self.view.opts['azimuth']
        }

    def _add_enhanced_axes(self):
        """Create enhanced axes visualization with arrows and labels"""
        # Create axis lines with arrows
        self.axis_items = []
        
        # X-axis (red)
        x_axis = GLLinePlotItem(
            pos=np.array([[0, 0, 0], [self.max_coords[0] * 1.2, 0, 0]]),
            color=(1, 0, 0, 1),
            width=2.0,
            antialias=True
        )
        self.view.addItem(x_axis)
        self.axis_items.append(x_axis)
        
        # Arrow head for X-axis
        x_arrow = GLLinePlotItem(
            pos=np.array([
                [self.max_coords[0] * 1.2, 0, 0],
                [self.max_coords[0] * 1.15, self.max_coords[0] * 0.05, 0],
                [self.max_coords[0] * 1.2, 0, 0],
                [self.max_coords[0] * 1.15, -self.max_coords[0] * 0.05, 0]
            ]),
            color=(1, 0, 0, 1),
            width=2.0,
            antialias=True
        )
        self.view.addItem(x_arrow)
        self.axis_items.append(x_arrow)
        
        # Y-axis (green)
        y_axis = GLLinePlotItem(
            pos=np.array([[0, 0, 0], [0, self.max_coords[1] * 1.2, 0]]),
            color=(0, 1, 0, 1),
            width=2.0,
            antialias=True
        )
        self.view.addItem(y_axis)
        self.axis_items.append(y_axis)
        
        # Arrow head for Y-axis
        y_arrow = GLLinePlotItem(
            pos=np.array([
                [0, self.max_coords[1] * 1.2, 0],
                [self.max_coords[1] * 0.05, self.max_coords[1] * 1.15, 0],
                [0, self.max_coords[1] * 1.2, 0],
                [-self.max_coords[1] * 0.05, self.max_coords[1] * 1.15, 0]
            ]),
            color=(0, 1, 0, 1),
            width=2.0,
            antialias=True
        )
        self.view.addItem(y_arrow)
        self.axis_items.append(y_arrow)
        
        # Z-axis (blue)
        z_axis = GLLinePlotItem(
            pos=np.array([[0, 0, 0], [0, 0, self.max_coords[2] * 1.2]]),
            color=(0, 0, 1, 1),
            width=2.0,
            antialias=True
        )
        self.view.addItem(z_axis)
        self.axis_items.append(z_axis)
        
        # Arrow head for Z-axis
        z_arrow = GLLinePlotItem(
            pos=np.array([
                [0, 0, self.max_coords[2] * 1.2],
                [self.max_coords[2] * 0.05, 0, self.max_coords[2] * 1.15],
                [0, 0, self.max_coords[2] * 1.2],
                [-self.max_coords[2] * 0.05, 0, self.max_coords[2] * 1.15]
            ]),
            color=(0, 0, 1, 1),
            width=2.0,
            antialias=True
        )
        self.view.addItem(z_arrow)
        self.axis_items.append(z_arrow)
        
        # Axis labels
        self.x_label = GLTextItem(
            pos=(self.max_coords[0] * 1.25, 0, 0), 
            text='X', 
            font=QtGui.QFont('Arial', 24), 
            color=(1,0,0,1))
        self.x_label.setGLOptions('additive')
        self.view.addItem(self.x_label)
        
        self.y_label = GLTextItem(
            pos=(0, self.max_coords[1] * 1.25, 0), 
            text='Y', 
            font=QtGui.QFont('Arial', 24), 
            color=(0,1,0,1))
        self.y_label.setGLOptions('additive')
        self.view.addItem(self.y_label)
        
        self.z_label = GLTextItem(
            pos=(0, 0, self.max_coords[2] * 1.25), 
            text='Z', 
            font=QtGui.QFont('Arial', 24), 
            color=(0,0,1,1))
        self.z_label.setGLOptions('additive')
        self.view.addItem(self.z_label)

    def _init_line_graph_window(self):
        """Initialize the line graph visualization window"""
        self.line_graph_window = QtWidgets.QWidget()
        self.line_graph_window.setWindowTitle('Coordinate Distributions')
        self.line_graph_window.resize(1000, 800)
        
        layout = QtWidgets.QVBoxLayout(self.line_graph_window)
        
        # X-coordinate distribution
        self.x_plot = pg.PlotWidget(title="X-coordinate Distribution")
        self.x_plot.setLabel('left', 'Count')
        self.x_plot.setLabel('bottom', 'X Value')
        self.x_plot.showGrid(x=True, y=True)
        layout.addWidget(self.x_plot)
        
        # Y-coordinate distribution
        self.y_plot = pg.PlotWidget(title="Y-coordinate Distribution")
        self.y_plot.setLabel('left', 'Count')
        self.y_plot.setLabel('bottom', 'Y Value')
        self.y_plot.showGrid(x=True, y=True)
        layout.addWidget(self.y_plot)
        
        # Z-coordinate distribution
        self.z_plot = pg.PlotWidget(title="Z-coordinate Distribution")
        self.z_plot.setLabel('left', 'Count')
        self.z_plot.setLabel('bottom', 'Z Value')
        self.z_plot.showGrid(x=True, y=True)
        layout.addWidget(self.z_plot)
        
        # Set line graph visibility to match button state
        self.line_graph_window.setVisible(self.show_line_graphs)

    def _update_line_graphs(self, visible_points):
        """Update the line graphs with current visible points"""
        if not self.show_line_graphs or visible_points.size == 0:
            return
            
        # Clear previous plots
        self.x_plot.clear()
        self.y_plot.clear()
        self.z_plot.clear()
        
        # Create histograms
        bins = 100
        
        # X-coordinate histogram
        x_counts, x_edges = np.histogram(visible_points[:, 0], bins=bins)
        self.x_plot.plot(x_edges, x_counts, stepMode=True, 
                         fillLevel=0, brush=(255, 0, 0, 50),
                         pen=pg.mkPen('r', width=2))
        
        # Y-coordinate histogram
        y_counts, y_edges = np.histogram(visible_points[:, 1], bins=bins)
        self.y_plot.plot(y_edges, y_counts, stepMode=True, 
                         fillLevel=0, brush=(0, 255, 0, 50),
                         pen=pg.mkPen('g', width=2))
        
        # Z-coordinate histogram
        z_counts, z_edges = np.histogram(visible_points[:, 2], bins=bins)
        self.z_plot.plot(z_edges, z_counts, stepMode=True, 
                         fillLevel=0, brush=(0, 0, 255, 50),
                         pen=pg.mkPen('b', width=2))

    def _toggle_line_graphs(self, checked):
        """Toggle visibility of line graphs"""
        self.show_line_graphs = checked
        self.line_graph_window.setVisible(checked)
        
        if checked and hasattr(self, 'scatter') and self.scatter is not None:
            self._update_line_graphs(self.scatter.pos)

    def _toggle_axes(self, state):
        """Toggle enhanced axes visualization"""
        for item in self.axis_items:
            item.setVisible(state == QtCore.Qt.Checked)
            
        self.x_label.setVisible(state == QtCore.Qt.Checked)
        self.y_label.setVisible(state == QtCore.Qt.Checked)
        self.z_label.setVisible(state == QtCore.Qt.Checked)

    def _init_plane_sliders(self):
        """Initialize plane sliders based on point cloud extent"""
        axis_index = self.combo_plane_axis.currentIndex()
        min_val = np.min(self.pc.coords[:, axis_index])
        max_val = np.max(self.pc.coords[:, axis_index])
        
        # Block signals during initialization
        self.slider_plane1.blockSignals(True)
        self.slider_plane2.blockSignals(True)
        
        self.slider_plane1.setRange(0, 1000)
        self.slider_plane2.setRange(0, 1000)
        self.slider_plane1.setValue(250)
        self.slider_plane2.setValue(750)
        
        self.slider_plane1.blockSignals(False)
        self.slider_plane2.blockSignals(False)
        
        self._update_planes()

    def _on_plane_axis_changed(self, index):
        """Handle axis change by reinitializing sliders"""
        self._init_plane_sliders()

    def _toggle_plane_system(self, axis):
        """Toggle plane system for a specific axis"""
        state = self.plane_systems[axis]['enabled'] = not self.plane_systems[axis]['enabled']
        if state:
            # Initialize values with current slider positions
            axis_index = ['x', 'y', 'z'].index(axis)
            min_val = np.min(self.pc.coords[:, axis_index])
            max_val = np.max(self.pc.coords[:, axis_index])
            range_val = max_val - min_val
            val1 = min_val + (self.slider_plane1.value() / 1000) * range_val
            val2 = min_val + (self.slider_plane2.value() / 1000) * range_val
            if val1 > val2:
                val1, val2 = val2, val1
            self.plane_systems[axis]['values'] = [val1, val2]
        
        if self.btn_show_planes.isChecked():
            self._add_planes()

    def _set_cluster_selection(self, select: bool):
        """Select or deselect all clusters"""
        for i in range(self.list_clusters.count()):
            item = self.list_clusters.item(i)
            item.setSelected(select)
        self._update_plot()

    def _update_timestamp_mode(self):
        """Update timestamp mode and refresh plot"""
        if self.btn_time.isChecked():
            self.btn_time.setText('Mode: Exact time')
        else:
            self.btn_time.setText('Mode: All ≤ time')
        self._update_plot()

    def _filter_mask(self):
        mask = np.isin(self.pc.cluster,
                        [it.data(QtCore.Qt.UserRole) for it in self.list_clusters.selectedItems()])
        mn = self.spin_snr_min.value()
        mx = self.spin_snr_max.value()
        mask &= (self.pc.snr >= mn) & (self.pc.snr <= mx)
        idx = self.slider_time.value()
        if idx < len(self.timestamps):
            ts_val = self.timestamps[idx]
            
            # Handle timestamp display
            if isinstance(ts_val, pd.Timestamp):
                self.lbl_time.setText(ts_val.strftime('%Y-%m-%d %H:%M:%S'))
            elif isinstance(ts_val, np.datetime64):
                ts_val = pd.Timestamp(ts_val)
                self.lbl_time.setText(ts_val.strftime('%Y-%m-%d %H:%M:%S'))
            else:
                self.lbl_time.setText(str(ts_val))
            
            # Fixed timestamp filtering logic
            if self.btn_time.isChecked():
                # Exact time mode
                mask &= (self.pc.timestamp == ts_val)
            else:
                # All points up to time mode
                mask &= (self.pc.timestamp <= ts_val)
        
        # Apply plane filter if active
        if self.plane_filter_active and self.plane_mask is not None:
            mask &= self.plane_mask
            
        return mask

    def _update_plot(self):
        """Update the 3D visualization with current filters"""
        mask = self._filter_mask()
        self.current_mask = mask
        pts = self.pc.coords[mask]
        snr = self.pc.snr[mask]
        
        # Remove old scatter plot
        if hasattr(self, 'scatter') and self.scatter is not None:
            self.view.removeItem(self.scatter)
            self.scatter = None
        
        # Add visualization shape if enabled
        self._update_shape_visualization()
        
        # Add planes if enabled
        if self.btn_show_planes.isChecked():
            self._add_planes()
        
        if pts.size == 0:
            self.lbl_centroids.clear()
            return
        
        # Create new scatter plot
        cmap = pg.colormap.get(self.combo_cmap.currentText())
        norm = (snr - snr.min()) / (np.ptp(snr) if np.ptp(snr) > 0 else 1)
        colors = cmap.map(norm, mode='float')
        
        self.scatter = GLScatterPlotItem(
            pos=pts, 
            size=self.spin_size.value(),
            color=colors, 
            pxMode=True
        )
        self.view.addItem(self.scatter)
        
        # Update line graphs if visible
        if self.show_line_graphs:
            self._update_line_graphs(pts)

    def _project_point(self, point):
        """Project a 3D point to 2D screen coordinates"""
        # Get view and projection matrices
        view_mat = self.view.viewMatrix()
        proj_mat = self.view.projectionMatrix()
        viewport = self.view.getViewport()
        
        # Create QVector4D for homogeneous coordinates
        p = QVector4D(float(point[0]), float(point[1]), float(point[2]), 1.0)
        
        # Apply view matrix
        p = view_mat.map(p)
        
        # Apply projection matrix
        p = proj_mat.map(p)
        
        # Perspective divide with zero check
        if abs(p.w()) > 1e-6:
            p.setX(p.x() / p.w())
            p.setY(p.y() / p.w())
            p.setZ(p.z() / p.w())
        else:
            # Handle case where w is zero or very small
            return QVector3D(-10000, -10000, 0)  # Off-screen position
        
        # Convert to screen coordinates
        x = (p.x() + 1) * viewport[2] / 2 + viewport[0]
        y = (p.y() + 1) * viewport[3] / 2 + viewport[1]
        
        return QVector3D(x, y, 0)  # Return as QVector3D for 2D screen position

    def _handle_mouse_move(self, event):
        """Handle mouse movement to show point coordinates while preserving panning"""
        # First call the parent method to preserve panning
        GLViewWidget.mouseMoveEvent(self.view, event)
        
        # Throttle hover checks to improve performance
        current_time = time.time()
        if current_time - self.last_hover_time < 0.05:  # 20fps
            return
        self.last_hover_time = current_time
        
        if not hasattr(self, 'scatter') or self.scatter is None:
            return
            
        # Get mouse position in screen coordinates
        screen_pos = event.pos()
        
        # Only process visible points
        visible_points = self.scatter.pos
        if visible_points.size == 0:
            self.status_bar.setText("No points to display")
            return
        
        # Project all visible points at once
        screen_points = np.array([
            [self._project_point(p).x(), self._project_point(p).y()]
            for p in visible_points
        ])
        
        # Vectorized distance calculation
        mouse_pos = QVector3D(screen_pos.x(), screen_pos.y(), 0)
        dists = np.sqrt(
            (screen_points[:,0] - mouse_pos.x())**2 + 
            (screen_points[:,1] - mouse_pos.y())**2
        )
        min_idx = np.argmin(dists)
        
        if dists[min_idx] < 20:  # 20 pixel threshold
            point = visible_points[min_idx]
            self.status_bar.setText(
                f"Point: X={point[0]:.2f}, Y={point[1]:.2f}, Z={point[2]:.2f}"
            )
            self.hover_point = point
        else:
            self.status_bar.setText("Hover over a point to see coordinates")
            self.hover_point = None

    def _update_planes(self):
        """Update plane positions based on slider values"""
        axis_index = self.combo_plane_axis.currentIndex()
        current_axis = ['x', 'y', 'z'][axis_index]
        min_val = np.min(self.pc.coords[:, axis_index])
        max_val = np.max(self.pc.coords[:, axis_index])
        range_val = max_val - min_val
        
        # Block signals during value adjustment
        self.slider_plane1.blockSignals(True)
        self.slider_plane2.blockSignals(True)
        
        val1 = min_val + (self.slider_plane1.value() / 1000) * range_val
        val2 = min_val + (self.slider_plane2.value() / 1000) * range_val
        
        # Only swap values if needed, don't reset sliders
        if val1 > val2:
            val1, val2 = val2, val1
        
        self.slider_plane1.blockSignals(False)
        self.slider_plane2.blockSignals(False)
        
        # Update only for current axis
        if self.plane_systems[current_axis]['enabled']:
            self.plane_systems[current_axis]['values'] = [val1, val2]
        
        self.lbl_plane1.setText(f"Plane 1: {val1:.2f}")
        self.lbl_plane2.setText(f"Plane 2: {val2:.2f}")
        
        if self.btn_show_planes.isChecked():
            self._add_planes()

    def _create_plane_lines(self, axis, value, color):
        """Create grid lines for a plane at a given position"""
        lines = []
        grid_size = 10  # Number of lines in each direction
        
        # Define the range for the other two axes
        if axis == 'x':  # X plane (YZ plane)
            y_range = np.linspace(self.min_coords[1], self.max_coords[1], grid_size)
            z_range = np.linspace(self.min_coords[2], self.max_coords[2], grid_size)
            
            # Create vertical lines (parallel to Z-axis)
            for y in y_range:
                lines.append([value, y, self.min_coords[2]])
                lines.append([value, y, self.max_coords[2]])
            
            # Create horizontal lines (parallel to Y-axis)
            for z in z_range:
                lines.append([value, self.min_coords[1], z])
                lines.append([value, self.max_coords[1], z])
                
        elif axis == 'y':  # Y plane (XZ plane)
            x_range = np.linspace(self.min_coords[0], self.max_coords[0], grid_size)
            z_range = np.linspace(self.min_coords[2], self.max_coords[2], grid_size)
            
            # Create vertical lines (parallel to Z-axis)
            for x in x_range:
                lines.append([x, value, self.min_coords[2]])
                lines.append([x, value, self.max_coords[2]])
            
            # Create horizontal lines (parallel to X-axis)
            for z in z_range:
                lines.append([self.min_coords[0], value, z])
                lines.append([self.max_coords[0], value, z])
                
        else:  # Z plane (XY plane) or 'z'
            x_range = np.linspace(self.min_coords[0], self.max_coords[0], grid_size)
            y_range = np.linspace(self.min_coords[1], self.max_coords[1], grid_size)
            
            # Create vertical lines (parallel to Y-axis)
            for x in x_range:
                lines.append([x, self.min_coords[1], value])
                lines.append([x, self.max_coords[1], value])
            
            # Create horizontal lines (parallel to X-axis)
            for y in y_range:
                lines.append([self.min_coords[0], y, value])
                lines.append([self.max_coords[0], y, value])
                
        return np.array(lines), color

    def _add_planes(self):
        """Add planes for all enabled axes using grid lines"""
        # Remove all existing planes
        for axis in ['x', 'y', 'z']:
            system = self.plane_systems[axis]
            for i in range(2):
                if system['planes'][i] is not None:
                    try:
                        self.view.removeItem(system['planes'][i])
                    except Exception:
                        pass
                    system['planes'][i] = None
        
        # Add planes for enabled axes
        for axis, system in self.plane_systems.items():
            if not system['enabled']:
                continue
                
            colors = [(1, 0.5, 0.5, 1.0), (0.5, 0.5, 1, 1.0)]  # Reddish, Bluish
            
            for i, value in enumerate(system['values']):
                # Create line positions for the plane
                line_positions, _ = self._create_plane_lines(axis, value, colors[i])
                
                # Create the plane using GLLinePlotItem
                plane = GLLinePlotItem(
                    pos=line_positions,
                    color=colors[i],
                    width=2.0,
                    antialias=True,
                    mode='lines'
                )
                plane.setGLOptions('additive')  # Ensure visibility
                self.view.addItem(plane)
                system['planes'][i] = plane

    def _toggle_planes(self, checked):
        """Show or hide planes"""
        if checked:
            self._add_planes()
        else:
            for axis in ['x', 'y', 'z']:
                system = self.plane_systems[axis]
                for i in range(2):
                    if system['planes'][i] is not None:
                        try:
                            self.view.removeItem(system['planes'][i])
                        except Exception:
                            pass
                        system['planes'][i] = None

    def _add_bounding_box(self):
        """Add bounding box visualization"""
        min_coords = self.min_coords
        max_coords = self.max_coords
        
        # Define the 8 corners of the bounding box
        vertices = np.array([
            [min_coords[0], min_coords[1], min_coords[2]],
            [max_coords[0], min_coords[1], min_coords[2]],
            [max_coords[0], max_coords[1], min_coords[2]],
            [min_coords[0], max_coords[1], min_coords[2]],
            [min_coords[0], min_coords[1], max_coords[2]],
            [max_coords[0], min_coords[1], max_coords[2]],
            [max_coords[0], max_coords[1], max_coords[2]],
            [min_coords[0], max_coords[1], max_coords[2]]
        ])
        
        # Define connections between vertices (edges)
        edges = np.array([
            [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom
            [4, 5], [5, 6], [6, 7], [7, 4],  # Top
            [0, 4], [1, 5], [2, 6], [3, 7]   # Sides
        ])
        
        # Create line segments by repeating vertices
        line_segments = np.zeros((len(edges)*2, 3))
        for i, edge in enumerate(edges):
            line_segments[i*2] = vertices[edge[0]]
            line_segments[i*2+1] = vertices[edge[1]]
        
        # Create line plot item
        box = GLLinePlotItem(
            pos=line_segments,
            color=(1, 1, 1, 0.7),  # White with transparency
            width=2.0,
            antialias=True,
            mode='lines'
        )
        return box

    def _add_sphere(self):
        """Add sphere visualization"""
        # Create sphere mesh
        sphere_mesh = pg.opengl.MeshData.sphere(rows=20, cols=20)
        sphere = GLMeshItem(
            meshdata=sphere_mesh,
            color=(0.8, 0.8, 0.8, 0.3),  # Light gray with transparency
            glOptions='translucent'
        )
        
        # Scale and position the sphere
        radius = self.max_dimension / 2
        sphere.scale(radius, radius, radius)
        sphere.translate(*self.centroid)
        
        return sphere

    def _add_cone(self):
        """Add cone visualization"""
        # Create cone mesh
        cone_mesh = pg.opengl.MeshData.cylinder(rows=1, cols=20, radius=[1.0, 0.0])
        cone = GLMeshItem(
            meshdata=cone_mesh,
            color=(0.8, 0.8, 0.8, 0.3),  # Light gray with transparency
            glOptions='translucent'
        )
        
        # Scale and position the cone
        height = self.max_dimension
        radius = self.max_dimension / 3
        cone.scale(radius, radius, height)
        
        # Position cone base at centroid
        cone.translate(*self.centroid)
        
        return cone

    def _update_shape_visualization(self):
        """Update visualization shape based on current selection"""
        # Remove existing shape if present
        if self.visual_shape is not None:
            try:
                if self.visual_shape in self.view.items:
                    self.view.removeItem(self.visual_shape)
            except Exception as e:
                logger.warning(f"Error removing shape: {e}")
            finally:
                self.visual_shape = None
        
        # Add new shape based on selection
        shape_type = self.combo_shape.currentText()
        if shape_type == 'Bounding Box':
            self.visual_shape = self._add_bounding_box()
        elif shape_type == 'Sphere':
            self.visual_shape = self._add_sphere()
        elif shape_type == 'Cone':
            self.visual_shape = self._add_cone()
        else:  # 'None'
            return
            
        if self.visual_shape:
            self.view.addItem(self.visual_shape)

    def _filter_points_between_planes(self):
        """Filter points between enabled planes"""
        # Start with all points visible
        combined_mask = np.ones(len(self.pc.coords), dtype=bool)
        
        # Apply filtering for each enabled axis
        for axis, system in self.plane_systems.items():
            if not system['enabled']:
                continue
                
            axis_index = ['x', 'y', 'z'].index(axis)
            min_val, max_val = min(system['values']), max(system['values'])
            axis_mask = (self.pc.coords[:, axis_index] >= min_val) & (self.pc.coords[:, axis_index] <= max_val)
            combined_mask &= axis_mask
        
        # Apply the combined mask
        self.plane_mask = combined_mask
        self.plane_filter_active = any(system['enabled'] for system in self.plane_systems.values())
        
        # Update plot
        self._update_plot()
        
        # Show info in status bar
        enabled_axes = [a for a, s in self.plane_systems.items() if s['enabled']]
        self.status_bar.setText(
            f"Filtered to {np.sum(combined_mask)} points within {len(enabled_axes)} plane system(s)"
        )
    
    def _clear_filtered_points(self):
        """Clear all plane filters"""
        for axis in ['x', 'y', 'z']:
            self.plane_systems[axis]['enabled'] = False
            self.cb_plane_x.setChecked(False)
            self.cb_plane_y.setChecked(False)
            self.cb_plane_z.setChecked(False)
        
        self.plane_filter_active = False
        self.plane_mask = None
        self._update_plot()
        self.status_bar.setText("Cleared all plane filters")

    def _reset_view(self):
        """Reset the view to initial state"""
        if self.initial_view:
            self.view.setCameraPosition(
                pos=self.initial_view['center'],
                distance=self.initial_view['distance'],
                elevation=self.initial_view['elevation'],
                azimuth=self.initial_view['azimuth']
            )

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
        # Use QScreen for better screenshot
        screen = QtWidgets.QApplication.primaryScreen()
        screenshot = screen.grabWindow(self.view.winId())
        screenshot.save(path, 'png')
        logger.info(f"Saved screenshot to {path}")

    def _compute_centroids(self):
        mask = self.current_mask
        pts = self.pc.coords[mask]
        clusters = self.pc.cluster[mask]
        out = []
        for c in np.unique(clusters):
            sel = pts[clusters==c]
            if len(sel) > 0:
                cen = sel.mean(axis=0)
                out.append(
                    f"Cluster {c}: Count={len(sel)}, Centroid={tuple(cen.round(2))}"
                )
        self.lbl_centroids.setText('<br>'.join(out))

    def run(self):
        """Start the application"""
        self.show()
        self.line_graph_window.show()

# Entrypoint
if __name__ == '__main__':
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