# import sys
# import datetime
# from serial.tools import list_ports
# from pymavlink import mavutil


# class DevicePortFinder:
#     """
#     Class to encapsulate logic for:
#     - Finding radar serial ports by description.
#     - Automatically detecting PX4/ArduPilot via UDP or serial.
#     """

#     def __init__(self, radar_keyword="CP2105"):
#         self.radar_keyword = radar_keyword.lower()

#     def log(self, level, msg):
#         ts = datetime.datetime.now().isoformat()
#         print(f"[{ts}] [{level}] {msg}")

#     def find_radar_ports_by_description(self):
#         """
#         Auto-detects CLI and DATA serial ports by looking for `radar_keyword`
#         in the port description or hardware ID. Supports both Windows and Linux.

#         Returns:
#             (cli_port: str, data_port: str)
#         Raises:
#             RuntimeError if fewer than two matching ports are found.
#         """
#         ports = list_ports.comports()
#         matches = []

#         for p in ports:
#             desc = (p.description or "").lower()
#             hwid = (p.hwid or "").lower()
#             manuf = (p.manufacturer or "").lower()

#             if self.radar_keyword in desc or self.radar_keyword in hwid or self.radar_keyword in manuf:
#                 matches.append(p.device)

#         if len(matches) < 2 and sys.platform.startswith('linux'):
#             usb_ports = [p.device for p in ports if ('ttyusb' in p.device.lower() or 'ttyacm' in p.device.lower())]
#             for port in usb_ports:
#                 if port not in matches:
#                     matches.append(port)
#                 if len(matches) >= 2:
#                     break

#         if len(matches) < 2:
#             raise RuntimeError(f"Expected at least 2 ports matching '{self.radar_keyword}', found: {matches}")

#         cli_port, data_port = matches[0], matches[1]
#         return cli_port, data_port

#     def find_autopilot_connection(self, timeout: float = 20.0, exclude_ports=None):
#         """
#         Attempt to find a live PX4/ArduPilot autopilot by:
#           1) Trying a common UDP endpoint (127.0.0.1:14550).
#           2) Scanning all serial ports at 9600, 57600, and 115200 baud.
#         Returns:
#             (conn_str, baud, autopilot_name)
#         Raises:
#             RuntimeError if neither UDP nor serial yielded a heartbeat.
#         """
#         UDP_ADDR = "127.0.0.1"
#         UDP_PORT = 14550
#         udp_conn = f"udp:{UDP_ADDR}:{UDP_PORT}"
#         master = None

#         try:
#             self.log("INFO", f"Connecting to MAVLink on {udp_conn} …")
#             master = mavutil.mavlink_connection(udp_conn, source_system=255)
#             self.log("INFO", "Waiting for MAVLink heartbeat …")
#             hb = master.recv_match(type='HEARTBEAT', blocking=True, timeout=timeout)
#             if hb:
#                 ap_t = hb.autopilot
#                 ap_name = mavutil.mavlink.enums['MAV_AUTOPILOT'][ap_t].description
#                 self.log("INFO", f"SUCCESS: autopilot via {udp_conn} → {ap_name}")
#                 master.close()
#                 return udp_conn, None, ap_name
#             else:
#                 self.log("WARN", f"No heartbeat on {udp_conn}")
#         except Exception as e:
#             self.log("WARN", f"UDP connection failed: {e}")
#         finally:
#             if master:
#                 try:
#                     master.close()
#                 except:
#                     pass

#         self.log("INFO", "Scanning serial ports…")
#         ports = [p for p in list_ports.comports()
#                  if exclude_ports is None or p.device not in exclude_ports]

#         if not ports:
#             self.log("INFO", "No ports detected—falling back to COM3–COM10")
#             ports = [type('P', (), {'device': f'COM{i}'}) for i in range(3, 11)]

#         for portinfo in ports:
#             port = portinfo.device
#             for baud in (9600, 57600, 115200):
#                 master = None
#                 try:
#                     self.log("INFO", f"Trying serial {port} @ {baud} baud …")
#                     master = mavutil.mavlink_connection(port, baud=baud, source_system=255)
#                     self.log("INFO", "Waiting for MAVLink heartbeat …")
#                     hb = master.recv_match(type='HEARTBEAT', blocking=True, timeout=timeout)
#                     if hb:
#                         ap_t = hb.autopilot
#                         ap_name = mavutil.mavlink.enums['MAV_AUTOPILOT'][ap_t].description
#                         self.log("INFO", f"SUCCESS: autopilot on {port}@{baud} → {ap_name}")
#                         master.close()
#                         return port, baud, ap_name
#                     else:
#                         self.log("WARN", f"No heartbeat on {port}@{baud}")
#                 except Exception as e:
#                     self.log("WARN", f"Error opening {port}@{baud}: {e}")
#                 finally:
#                     if master:
#                         try:
#                             master.close()
#                         except:
#                             pass

#         ts = datetime.datetime.now().isoformat()
#         raise RuntimeError(f"[{ts}] [ERROR] No PX4/ArduPilot heartbeat detected on UDP or any serial port.")

# import sys
# import datetime
# from serial.tools import list_ports
# from pymavlink import mavutil


# class DevicePortFinder:
#     def __init__(self, radar_keyword="CP2105"):
#         self.radar_keyword = radar_keyword.lower()

#     def log(self, level, msg):
#         ts = datetime.datetime.now().isoformat()
#         print(f"[{ts}] [{level}] {msg}")

#     def find_radar_ports_by_description(self):
#         ports = list_ports.comports()
#         matches = []

#         for p in ports:
#             desc = (p.description or "").lower()
#             hwid = (p.hwid or "").lower()
#             manuf = (p.manufacturer or "").lower()

#             if self.radar_keyword in desc or self.radar_keyword in hwid or self.radar_keyword in manuf:
#                 matches.append(p.device)

#         if len(matches) < 2 and sys.platform.startswith('linux'):
#             usb_ports = [p.device for p in ports if ('ttyusb' in p.device.lower() or 'ttyacm' in p.device.lower())]
#             for port in usb_ports:
#                 if port not in matches:
#                     matches.append(port)
#                 if len(matches) >= 2:
#                     break

#         if len(matches) < 2:
#             raise RuntimeError(f"Expected at least 2 ports matching '{self.radar_keyword}', found: {matches}")

#         return matches[0], matches[1]

#     def find_autopilot_connection(self, timeout: float = 20.0, exclude_ports=None):
#         UDP_ADDR = "127.0.0.1"
#         UDP_PORT = 14550
#         udp_conn = f"udp:{UDP_ADDR}:{UDP_PORT}"
#         master = None

#         try:
#             self.log("INFO", f"Connecting to MAVLink on {udp_conn} …")
#             master = mavutil.mavlink_connection(udp_conn, source_system=255)
#             self.log("INFO", "Waiting for MAVLink heartbeat …")
#             hb = master.recv_match(type='HEARTBEAT', blocking=True, timeout=timeout)
#             if hb:
#                 ap_t = hb.autopilot
#                 ap_name = mavutil.mavlink.enums['MAV_AUTOPILOT'][ap_t].description
#                 self.log("INFO", f"SUCCESS: autopilot via {udp_conn} → {ap_name}")
#                 return udp_conn, None, ap_name
#             else:
#                 self.log("WARN", f"No heartbeat on {udp_conn}")
#         except Exception as e:
#             self.log("WARN", f"UDP connection failed: {e}")
#         finally:
#             if master:
#                 try:
#                     master.close()
#                 except Exception:
#                     pass

#         self.log("INFO", "Scanning serial ports…")
#         ports = [p for p in list_ports.comports() if exclude_ports is None or p.device not in exclude_ports]

#         if not ports:
#             self.log("INFO", "No ports detected—falling back to COM3–COM10")
#             ports = [type('P', (), {'device': f'COM{i}'}) for i in range(3, 11)]

#         for portinfo in ports:
#             port = portinfo.device
#             for baud in (9600, 57600, 115200):
#                 master = None
#                 try:
#                     self.log("INFO", f"Trying serial {port} @ {baud} baud …")
#                     master = mavutil.mavlink_connection(port, baud=baud, source_system=255)
#                     self.log("INFO", "Waiting for MAVLink heartbeat …")
#                     hb = master.recv_match(type='HEARTBEAT', blocking=True, timeout=timeout)
#                     if hb:
#                         ap_t = hb.autopilot
#                         ap_name = mavutil.mavlink.enums['MAV_AUTOPILOT'][ap_t].description
#                         self.log("INFO", f"SUCCESS: autopilot on {port}@{baud} → {ap_name}")
#                         return port, baud, ap_name
#                     else:
#                         self.log("WARN", f"No heartbeat on {port}@{baud}")
#                 except Exception as e:
#                     self.log("WARN", f"Error opening {port}@{baud}: {e}")
#                 finally:
#                     if master:
#                         try:
#                             master.close()
#                         except Exception:
#                             pass

#         raise RuntimeError("No PX4/ArduPilot heartbeat detected on UDP or any serial port.")



# device_port_finder.py

# import sys
# import datetime
# from serial.tools import list_ports
# from pymavlink import mavutil


# class DevicePortFinder:
#     def __init__(self, radar_keyword="CP2105"):
#         self.radar_keyword = radar_keyword.lower()

#     def log(self, level, msg):
#         ts = datetime.datetime.now().isoformat()
#         print(f"[{ts}] [{level}] {msg}")

#     def find_radar_ports_by_description(self):
#         ports = list_ports.comports()
#         matches = []

#         for p in ports:
#             desc = (p.description or "").lower()
#             hwid = (p.hwid or "").lower()
#             manuf = (p.manufacturer or "").lower()

#             if self.radar_keyword in desc or self.radar_keyword in hwid or self.radar_keyword in manuf:
#                 matches.append(p.device)

#         if len(matches) < 2 and sys.platform.startswith('linux'):
#             usb_ports = [p.device for p in ports if ('ttyusb' in p.device.lower() or 'ttyacm' in p.device.lower())]
#             for port in usb_ports:
#                 if port not in matches:
#                     matches.append(port)
#                 if len(matches) >= 2:
#                     break

#         if len(matches) < 2:
#             raise RuntimeError(f"Expected at least 2 ports matching '{self.radar_keyword}', found: {matches}")

#         return matches[0], matches[1]

#     def find_autopilot_connection(self, timeout: float = 20.0, exclude_ports=None):
#         UDP_ADDR = "127.0.0.1"
#         UDP_PORT = 14550
#         udp_conn = f"udp:{UDP_ADDR}:{UDP_PORT}"
#         master = None

#         try:
#             self.log("INFO", f"Connecting to MAVLink on {udp_conn} …")
#             master = mavutil.mavlink_connection(udp_conn, source_system=255)
#             self.log("INFO", "Waiting for MAVLink heartbeat …")
#             hb = master.recv_match(type='HEARTBEAT', blocking=True, timeout=timeout)
#             if hb:
#                 ap_t = hb.autopilot
#                 ap_name = mavutil.mavlink.enums['MAV_AUTOPILOT'][ap_t].description
#                 self.log("INFO", f"SUCCESS: autopilot via {udp_conn} → {ap_name}")
#                 return udp_conn, None, ap_name
#             else:
#                 self.log("WARN", f"No heartbeat on {udp_conn}")
#         except Exception as e:
#             self.log("WARN", f"UDP connection failed: {e}")
#         finally:
#             if master:
#                 try:
#                     master.close()
#                 except Exception:
#                     pass

#         self.log("INFO", "Scanning serial ports…")
#         ports = [p for p in list_ports.comports() if exclude_ports is None or p.device not in exclude_ports]

#         if not ports:
#             self.log("INFO", "No ports detected—falling back to COM3–COM10")
#             ports = [type('P', (), {'device': f'COM{i}'}) for i in range(3, 11)]

#         for portinfo in ports:
#             port = portinfo.device
#             for baud in (9600, 57600, 115200):
#                 master = None
#                 try:
#                     self.log("INFO", f"Trying serial {port} @ {baud} baud …")
#                     master = mavutil.mavlink_connection(port, baud=baud, source_system=255)
#                     self.log("INFO", "Waiting for MAVLink heartbeat …")
#                     hb = master.recv_match(type='HEARTBEAT', blocking=True, timeout=timeout)
#                     if hb:
#                         ap_t = hb.autopilot
#                         ap_name = mavutil.mavlink.enums['MAV_AUTOPILOT'][ap_t].description
#                         self.log("INFO", f"SUCCESS: autopilot on {port}@{baud} → {ap_name}")
#                         return port, baud, ap_name
#                     else:
#                         self.log("WARN", f"No heartbeat on {port}@{baud}")
#                 except Exception as e:
#                     self.log("WARN", f"Error opening {port}@{baud}: {e}")
#                 finally:
#                     if master:
#                         try:
#                             master.close()
#                         except Exception:
#                             pass

#         # Instead of raising, log error and return None triple
#         self.log("ERROR", "No PX4/ArduPilot heartbeat detected on UDP or any serial port.")
#         return None, None, None



# device_port_finder.py

import sys
import datetime
from serial.tools import list_ports
from pymavlink import mavutil
import platform
import glob
      
     



class DevicePortFinder:
    def __init__(self, radar_keyword="CP2105"):
        self.radar_keyword = radar_keyword.lower()

    def log(self, level, msg):
        ts = datetime.datetime.now().isoformat()
        print(f"[{ts}] [{level}] {msg}")

    def find_radar_ports_by_description(self):
        ports = list_ports.comports()
        matches = []

        for p in ports:
            desc = (p.description or "").lower()
            hwid = (p.hwid or "").lower()
            manuf = (p.manufacturer or "").lower()

            if self.radar_keyword in desc or self.radar_keyword in hwid or self.radar_keyword in manuf:
                matches.append(p.device)

        if len(matches) < 2 and sys.platform.startswith('linux'):
            usb_ports = [p.device for p in ports if ('ttyusb' in p.device.lower() or 'ttyacm' in p.device.lower())]
            for port in usb_ports:
                if port not in matches:
                    matches.append(port)
                if len(matches) >= 2:
                    break

        if len(matches) < 2:
            raise RuntimeError(f"Expected at least 2 ports matching '{self.radar_keyword}', found: {matches}")

        return matches[0], matches[1]

    def find_autopilot_connection(self, timeout: float = 20.0, exclude_ports=None):
        import os
        import platform
        import glob
        from serial.tools import list_ports
        from pymavlink import mavutil

        UDP_ADDR = "127.0.0.1"
        UDP_PORT = 14550
        udp_conn = f"udp:{UDP_ADDR}:{UDP_PORT}"
        master = None
        keep_master = False

        # Try UDP first
        try:
            self.log("INFO", "Connecting to MAVLink on {} ...".format(udp_conn))
            master = mavutil.mavlink_connection(
                udp_conn,
                source_system=255,
                mavlink_version=1  # Modified to force MAVLink 1.0
            )
            self.log("INFO", "Waiting for MAVLink heartbeat ...")
            hb = master.recv_match(type='HEARTBEAT', blocking=True, timeout=timeout)
            if hb:
                ap_t = hb.autopilot
                ap_name = mavutil.mavlink.enums['MAV_AUTOPILOT'][ap_t].description
                self.log("INFO", "SUCCESS: autopilot via {} -> {}".format(udp_conn, ap_name))
                keep_master = True
                return master, ap_name
            else:
                self.log("WARN", "No heartbeat on {}".format(udp_conn))
        except Exception as e:
            self.log("WARN", "UDP connection failed: {}".format(e))
        finally:
            if master and not keep_master:
                try:
                    master.close()
                except Exception:
                    pass

        # Scan serial ports next
        self.log("INFO", "Scanning serial ports...")

        ports = []
        system_os = platform.system()

        if system_os == "Windows":
            ports = [p for p in list_ports.comports() if exclude_ports is None or p.device not in exclude_ports]
            if not ports:
                self.log("INFO", "No ports detected - falling back to COM3 to COM10")
                ports = [type('P', (), {'device': 'COM{}'.format(i)}) for i in range(3, 11)]
        else:
            # Linux-compatible ports: USB, ACM, AMA, and S0
            linux_port_patterns = ['/dev/ttyUSB*', '/dev/ttyACM*', '/dev/ttyAMA*', '/dev/ttyS0']
            detected = []
            for pattern in linux_port_patterns:
                detected += glob.glob(pattern)
            detected = sorted(set(detected))
            if exclude_ports:
                detected = [p for p in detected if p not in exclude_ports]
            ports = [type('P', (), {'device': p}) for p in detected]

        for portinfo in ports:
            port = portinfo.device
            for baud in (9600, 57600, 115200):
                master = None
                try:
                    self.log("INFO", "Trying serial {} @ {} baud ...".format(port, baud))
                    master = mavutil.mavlink_connection(
                        port,
                        baud=baud,
                        source_system=255,
                        mavlink_version=1  # Modified to force MAVLink 1.0
                    )
                    self.log("INFO", "Waiting for MAVLink heartbeat ...")
                    hb = master.recv_match(type='HEARTBEAT', blocking=True, timeout=timeout)
                    if hb:
                        ap_t = hb.autopilot
                        ap_name = mavutil.mavlink.enums['MAV_AUTOPILOT'][ap_t].description
                        self.log("INFO", "SUCCESS: autopilot on {}@{} -> {}".format(port, baud, ap_name))
                        keep_master = True
                        return master, ap_name
                    else:
                        self.log("WARN", "No heartbeat on {}@{}".format(port, baud))
                except Exception as e:
                    self.log("WARN", "Error opening {}@{}: {}".format(port, baud, e))
                finally:
                    if master and not keep_master:
                        try:
                            master.close()
                        except Exception:
                            pass

        self.log("ERROR", "No PX4/ArduPilot heartbeat detected on UDP or any serial port.")
        return None, None


