#!/usr/bin/env python3

import argparse
import random
import time
import sys
from pymavlink import mavutil

# Sensor configuration
SENSOR_ID = 0
MIN_DISTANCE = 20
MAX_DISTANCE = 400
SENSOR_TYPE = 0
SENSOR_ORIENTATION = 25  # MAV_SENSOR_ORIENTATION_ROLL_180 (downward)
UPDATE_RATE = 10  # Hz

# Valid unit quaternion for downward orientation (w, x, y, z format)
# Represents no rotation (identity quaternion)
QUATERNION = [1.0, 0.0, 0.0, 0.0]

def send_distance_sensor(master, distance_cm):
    """Send DISTANCE_SENSOR MAVLink message to flight controller"""
    # Use either system time or 0 if unknown
    time_boot_ms = getattr(master, 'time_since_boot_ms', lambda: 0)()
    
    master.mav.distance_sensor_send(
        time_boot_ms=time_boot_ms,
        min_distance=MIN_DISTANCE,
        max_distance=MAX_DISTANCE,
        current_distance=int(distance_cm),
        type=SENSOR_TYPE,
        id=SENSOR_ID,
        orientation=SENSOR_ORIENTATION,
        covariance=0,
        horizontal_fov=0.0,  # Must be float
        vertical_fov=0.0,    # Must be float
        quaternion=QUATERNION,
        signal_quality=100
    )

def main():
    parser = argparse.ArgumentParser(description='Downward Distance Sensor Simulator')
    parser.add_argument('--connect', default='udpin:localhost:14550',
                        help='Connection string (udpin, udpout, tcp, or serial port)')
    parser.add_argument('--baud', type=int, default=57600,
                        help='Baud rate for serial connections')
    args = parser.parse_args()

    print(f"Connecting to: {args.connect} at {args.baud} baud")
    
    try:
        # Attempt connection
        master = mavutil.mavlink_connection(
            args.connect,
            baud=args.baud,
            autoreconnect=True,
            source_system=1,
            source_component=93
        )
        
        # Wait for heartbeat
        master.wait_heartbeat(timeout=5)
        print(f"Heartbeat from system {master.target_system}, component {master.target_component}")
        print(f"Using MAVLink {'2.0' if master.mavlink20() else '1.0'}")

        print(f"Sending simulated distance data ({MIN_DISTANCE}-{MAX_DISTANCE}cm) at {UPDATE_RATE}Hz")
        print("Press Ctrl+C to exit...")
        
        while True:
            distance = random.randint(MIN_DISTANCE, MAX_DISTANCE)
            send_distance_sensor(master, distance)
            time.sleep(1.0 / UPDATE_RATE)
            
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()