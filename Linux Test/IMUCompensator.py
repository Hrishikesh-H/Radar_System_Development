import numpy as np
import time
from pymavlink import mavutil  # Needed for command constants

class AttitudeCompensator:
    def __init__(self, master, radar_to_body='Y-down'):
        """
        Initialize the attitude compensator with an existing MAVLink connection.

        Args:
            master: A pymavlink connection object (already connected and ready).
            radar_to_body: either a string preset or a 3x3 np.ndarray to map radar frame to body frame.
                           Supported strings: 'Y-down' (default), 'Z-down'
        """
        self.master = master
        self.autopilot = "EXTERNAL"
        self.roll_offset = 0.0
        self.pitch_offset = 0.0
        self.yaw_offset = 0.0

        self.radar_to_body = self._parse_axis_mapping(radar_to_body)
        self._request_attitude_stream()

        # Give some time for autopilot to start sending ATTITUDE messages
        time.sleep(0.5)

        # Start internal calibration automatically
         # Start internal calibration automatically ONCE
        # if calli_req:
        #     self.internal_calibrate_offsets(num_samples=100, delay=0.01)


    def _parse_axis_mapping(self, mapping):
        if isinstance(mapping, np.ndarray):
            if mapping.shape == (3, 3):
                return mapping
            else:
                raise ValueError("Custom radar_to_body matrix must be 3×3.")
        elif isinstance(mapping, str):
            if mapping.lower() == 'y-down':
                return np.array([
                    [1, 0, 0],
                    [0, 0, -1],
                    [0, -1, 0]
                ])
            elif mapping.lower() == 'z-down':
                return np.eye(3)
            else:
                raise ValueError(f"Unknown radar_to_body preset: {mapping}")
        else:
            raise TypeError("radar_to_body must be a string or 3×3 ndarray.")

    def _request_attitude_stream(self):
        try:
            # Use MAV_CMD_SET_MESSAGE_INTERVAL to request ATTITUDE at 10Hz (100000 microseconds interval)
            self.master.mav.command_long_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                0,
                mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE,  # message ID for ATTITUDE
                100000,  # interval in microseconds (10 Hz)
                0, 0, 0, 0, 0
            )
            print("[AttitudeCompensator] Requested ATTITUDE message interval at 10Hz.")
        except Exception as e:
            print(f"[AttitudeCompensator] Failed to set ATTITUDE message interval: {e}")

    def get_attitude(self, timeout: float = 1.0):
        msg = self.master.recv_match(type='ATTITUDE', blocking=True, timeout=timeout)
        if msg is None:
            raise RuntimeError("[AttitudeCompensator] Timeout waiting for ATTITUDE message.")
        roll = float(msg.roll) - self.roll_offset
        pitch = float(msg.pitch) - self.pitch_offset
        yaw = float(msg.yaw) - self.yaw_offset
        return roll, pitch, yaw

    def transform_pointcloud(self, pointcloud: np.ndarray) -> np.ndarray:
        """
        Transform radar pointcloud from radar frame → NED → back to radar frame,
        applying attitude compensation and returning corrected radar-frame data.

        Args:
            pointcloud: np.ndarray of shape (N, 3), in radar frame.

        Returns:
            np.ndarray of shape (N, 3), attitude-compensated back in radar frame.
        """
        if pointcloud.ndim != 2 or pointcloud.shape[1] != 3:
            raise ValueError("[AttitudeCompensator] `pointcloud` must be shape (N, 3).")

        print(f"[Debug] Input pointcloud shape: {pointcloud.shape}")

        # Step 1: Radar → Body
        pts_body = pointcloud @ self.radar_to_body.T

        # Step 2: Get attitude
        roll, pitch, yaw = self.get_attitude()
        print(f"[Debug] Current attitude - Roll: {roll:.4f}, Pitch: {pitch:.4f}, Yaw: {yaw:.4f}")

        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

        R_x = np.array([[1, 0, 0],
                        [0, cr, -sr],
                        [0, sr, cr]])
        R_y = np.array([[cp, 0, sp],
                        [0, 1, 0],
                        [-sp, 0, cp]])
        R_z = np.array([[cy, -sy, 0],
                        [sy, cy, 0],
                        [0, 0, 1]])

        # Step 3: Body → Earth (NED)
        R_b2e = R_z @ R_y @ R_x
        pts_ned = pts_body @ R_b2e.T

        # [You could apply NED-space filtering/logic here if needed]

        # Step 4: Earth (NED) → Body (invert the rotation)
        pts_body_corrected = pts_ned @ R_b2e

        # Step 5: Body → Radar (using transpose to invert radar_to_body)
        pts_radar_corrected = pts_body_corrected @ self.radar_to_body.T

        print(f"[Debug] Output (corrected) pointcloud shape: {pts_radar_corrected.shape}")
        print(f"[Debug] Sample output points (first 5):")
        for i, pt in enumerate(pts_radar_corrected[:5]):
            print(f" Point {i}: x={pt[0]:.4f}, y={pt[1]:.4f}, z={pt[2]:.4f}")

        return pts_radar_corrected

    # def inverse_transform_pointcloud(self, pointcloud_ned: np.ndarray) -> np.ndarray:
    #     """
    #     Transform pointcloud from earth frame back to radar frame.

    #     Args:
    #         pointcloud_ned: np.ndarray of shape (N, 3), points in NED frame.

    #     Returns:
    #         np.ndarray of shape (N, 3), points in radar frame.
    #     """
    #     if pointcloud_ned.ndim != 2 or pointcloud_ned.shape[1] != 3:
    #         raise ValueError("[AttitudeCompensator] `pointcloud_ned` must be shape (N, 3).")

    #     roll, pitch, yaw = self.get_attitude()

    #     cr = np.cos(roll)
    #     sr = np.sin(roll)
    #     cp = np.cos(pitch)
    #     sp = np.sin(pitch)
    #     cy = np.cos(yaw)
    #     sy = np.sin(yaw)

    #     R_x = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    #     R_y = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    #     R_z = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])

    #     R_b2e = R_z @ R_y @ R_x

    #     # Earth → Body (invert R_b2e transpose)
    #     pts_body = pointcloud_ned @ R_b2e

    #     # Body → Radar (use transpose to invert radar_to_body)
    #     pts_radar = pts_body @ self.radar_to_body.T

    #     return pts_radar

    def get_transformation_matrix(self) -> np.ndarray:
        roll, pitch, yaw = self.get_attitude()

        cr = np.cos(roll)
        sr = np.sin(roll)
        cp = np.cos(pitch)
        sp = np.sin(pitch)
        cy = np.cos(yaw)
        sy = np.sin(yaw)

        R_x = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        R_y = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        R_z = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])

        return R_z @ R_y @ R_x

    def internal_calibrate_offsets(self, num_samples, delay):
        rolls, pitches, yaws = [], [], []
        print(f"[AttitudeCompensator] Starting internal calibration with {num_samples} samples...")
        for _ in range(num_samples):
            try:
                r, p, y = self.get_attitude(timeout=1.0)
            except RuntimeError:
                continue
            rolls.append(r)
            pitches.append(p)
            yaws.append(y)
            time.sleep(delay)
        if rolls and pitches and yaws:
            self.roll_offset = np.mean(rolls)
            self.pitch_offset = np.mean(pitches)
            # If you wish to remove yaw bias as well, uncomment the next line:
            # self.yaw_offset = self.circular_mean(np.array(yaws))
            print(f"[AttitudeCompensator] Internal calibration done - "
                  f"Roll offset: {self.roll_offset}, "
                  f"Pitch offset: {self.pitch_offset}, "
                  f"Yaw offset: {self.yaw_offset}")
        else:
            print("[AttitudeCompensator] Warning: Internal calibration failed to get samples.")

    @staticmethod
    def circular_mean(angles):
        sin_sum = np.sum(np.sin(angles))
        cos_sum = np.sum(np.cos(angles))
        return np.arctan2(sin_sum, cos_sum)

    def close(self):
        pass  # External connection managed elsewhere

