import csv
from datetime import datetime
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QFrame
from PyQt6.QtGui import QFont, QColor, QPalette
from PyQt6.QtCore import Qt


class DroneLandingStatus(QWidget):
    def __init__(self, export_csv=False):
        super().__init__()
        self.export_csv = export_csv
        self.csv_log = []  # Collect each frame's data

        self.setWindowTitle("Drone Control Station - Landing Zone Monitor")
        self.setFixedSize(640, 320)
        self.setStyleSheet(""" 
            QWidget {
                background-color: #0d0d0d;
                color: #00ffcc;
                font-family: 'Consolas', 'Courier New', monospace;
            }
            QLabel {
                font-size: 16px;
            }
            #titleLabel {
                font-size: 24px;
                color: #00ffff;
                font-weight: bold;
            }
            #statusLabel[status="safe"] {
                color: #00ff00;
            }
            #statusLabel[status="unsafe"] {
                color: #ff0033;
            }
        """)

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.title_label = QLabel("🛸 Landing Zone Analysis")
        self.title_label.setObjectName("titleLabel")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.status_label = QLabel("Status: UNKNOWN")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.slope_label = QLabel("Slope: -")
        self.inliers_label = QLabel("Inliers: -")
        self.residual_label = QLabel("Residual: -")
        self.reason_label = QLabel("Reason: -")

        info_layout = QVBoxLayout()
        info_layout.addWidget(self.slope_label)
        info_layout.addWidget(self.inliers_label)
        info_layout.addWidget(self.residual_label)
        info_layout.addWidget(self.reason_label)

        layout.addWidget(self.title_label)
        layout.addWidget(self.status_label)
        layout.addLayout(info_layout)

        self.setLayout(layout)

    def update_status(self, safe, m):
        coeffs = m.get('plane', None)

        if coeffs is None or not hasattr(coeffs, '__len__') or len(coeffs) < 4:
            self.status_label.setText("Status: UNSAFE (Insufficient data)")
            self.status_label.setProperty("status", "unsafe")
            self.reason_label.setText("Reason: Insufficient data for slope estimation")

            if self.export_csv:
                self.csv_log.append({
                    'timestamp': datetime.now().isoformat(),
                    'status': "UNSAFE",
                    'slope_deg': "",
                    'inlier_ratio': "",
                    'residual_cm': "",
                    'reason': "Insufficient data"
                })

        else:
            slope = m.get("slope_deg", 0)
            inliers = m.get("inlier_ratio", 0) * 100
            residual = m.get("mean_residual", 0) * 100
            reason = m.get("reason", "None")

            self.slope_label.setText(f"Slope: {slope:.1f} deg")
            self.inliers_label.setText(f"Inliers: {inliers:.0f}%")
            self.residual_label.setText(f"Residual: {residual:.1f} cm")
            self.reason_label.setText(f"Reason: {reason}")

            status_str = "SAFE" if safe else "UNSAFE"
            self.status_label.setText(f"Status: {status_str}")
            self.status_label.setProperty("status", status_str.lower())
            self.status_label.style().unpolish(self.status_label)
            self.status_label.style().polish(self.status_label)

            if self.export_csv:
                self.csv_log.append({
                    'timestamp': datetime.now().isoformat(),
                    'status': status_str,
                    'slope_deg': f"{slope:.2f}",
                    'inlier_ratio': f"{inliers:.2f}",
                    'residual_cm': f"{residual:.2f}",
                    'reason': reason
                })

    def closeEvent(self, event):
        if self.export_csv and self.csv_log:
            filename = f"landing_analysis_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            with open(filename, mode='w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=[
                    'timestamp', 'status', 'slope_deg',
                    'inlier_ratio', 'residual_cm', 'reason'
                ])
                writer.writeheader()
                writer.writerows(self.csv_log)
            print(f"[INFO] Exported landing zone analysis to {filename}")
        super().closeEvent(event)

