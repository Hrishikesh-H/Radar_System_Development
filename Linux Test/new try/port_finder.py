# port_finder.py
import sys
from serial.tools import list_ports
from pymavlink import mavutil
import platform
import glob
import traceback

# Import our centralized logger
from system_logger import get_logger

class DevicePortFinder:
    def __init__(self, radar_keyword="CP2105"):
        self.radar_keyword = radar_keyword.lower()
        # Storage for device details
        self.device_details = {
            'radar': {'cli': None, 'data': None},
            'autopilot': {'connection': None, 'info': None, 'port': None, 'baud': None}
        }
        
        # Configure logging with subsystem name
        self.logger = get_logger('PortFinder')

    def _log_exception(self, context, e):
        """Log exception with full traceback and context details"""
        self.logger.error(f"Exception in {context}: {str(e)}")
        self.logger.debug(f"Exception details: {type(e).__name__}")
        self.logger.debug(f"Traceback:\n{traceback.format_exc()}")
        self.logger.debug(f"System state: {self.device_details}")

    def find_radar_ports_by_description(self):
        try:
            self.logger.info(f"Searching for radar ports with keyword: '{self.radar_keyword}'")
            ports = list_ports.comports()
            matches = []
            self.logger.debug(f"Found {len(ports)} total serial ports")

            for p in ports:
                desc = (p.description or "").lower()
                hwid = (p.hwid or "").lower()
                manuf = (p.manufacturer or "").lower()
                self.logger.debug(f"Checking port: {p.device} | Desc: {desc} | HWID: {hwid} | Manuf: {manuf}")

                if (self.radar_keyword in desc or
                        self.radar_keyword in hwid or
                        self.radar_keyword in manuf):
                    self.logger.info(f"Found matching radar port: {p.device}")
                    matches.append(p.device)

            # Fallback for Linux systems
            if len(matches) < 2 and sys.platform.startswith('linux'):
                self.logger.warning(f"Only {len(matches)} radar ports found, trying Linux fallback")
                usb_ports = [p.device for p in ports if ('ttyusb' in p.device.lower() or 'ttyacm' in p.device.lower())]
                self.logger.debug(f"Found {len(usb_ports)} USB/ACM ports: {usb_ports}")

                for port in usb_ports:
                    if port not in matches:
                        matches.append(port)
                        self.logger.info(f"Added fallback port: {port}")
                    if len(matches) >= 2:
                        self.logger.info("Found required 2 ports via fallback")
                        break

            if len(matches) < 2:
                error_msg = f"Expected at least 2 ports matching '{self.radar_keyword}', found: {matches}"
                self.logger.error(error_msg)
                self.logger.debug(f"All scanned ports: {[p.device for p in ports]}")
                raise RuntimeError(error_msg)

            # Try to distinguish CLI and DATA ports using baudrate test
            self.logger.info("Attempting to distinguish CLI and DATA ports...")

            cli_candidate = None
            data_candidate = None

            for port in matches:
                try:
                    self.logger.debug(f"Probing port {port} as CLI at baud {self.cli_baud}")
                    with serial.Serial(port, self.cli_baud, timeout=0.5) as ser:
                        ser.write(b'version\r\n')
                        time.sleep(0.1)
                        resp = ser.read(100).decode('ascii', errors='ignore').lower()
                        if "mmwave" in resp or "sdk" in resp or "version" in resp:
                            self.logger.info(f"Identified {port} as CLI port")
                            cli_candidate = port
                        else:
                            self.logger.debug(f"No CLI response from {port}, assuming data port")
                            data_candidate = port
                except Exception as e:
                    self.logger.warning(f"Failed to test port {port}: {e}")

            # If auto-detection fails, fallback to original order
            if not cli_candidate or not data_candidate:
                self.logger.warning("Could not auto-distinguish ports reliably; using default order")
                cli_candidate, data_candidate = matches[0], matches[1]

            # Store radar ports in device details
            self.device_details['radar']['cli'] = cli_candidate
            self.device_details['radar']['data'] = data_candidate

            self.logger.info(f"Radar ports identified: CLI={cli_candidate}, DATA={data_candidate}")
            return cli_candidate, data_candidate

        except Exception as e:
            self._log_exception("find_radar_ports_by_description", e)
            raise

    def find_autopilot_connection(self, timeout: float = 20.0, exclude_ports=None):
        try:
            self.logger.info(f"Starting autopilot discovery (timeout={timeout}s, exclude={exclude_ports})")
            UDP_ADDR = "127.0.0.1"
            UDP_PORT = 14550
            udp_conn = f"udp:{UDP_ADDR}:{UDP_PORT}"
            master = None
            keep_master = False

            # Try UDP first
            try:
                self.logger.info(f"Attempting UDP connection: {udp_conn}")
                master = mavutil.mavlink_connection(
                    udp_conn,
                    source_system=255,
                    mavlink_version=1
                )
                self.logger.debug(f"UDP connection object created: {master}")
                self.logger.info("Listening for MAVLink heartbeat...")
                hb = master.recv_match(type='HEARTBEAT', blocking=True, timeout=timeout)
                
                if hb:
                    ap_t = hb.autopilot
                    ap_name = mavutil.mavlink.enums['MAV_AUTOPILOT'][ap_t].description
                    self.logger.info(f"SUCCESS: Autopilot via UDP | Type: {ap_name} | System: {hb.get_srcSystem()}")
                    
                    # Store autopilot details
                    self.device_details['autopilot']['connection'] = master
                    self.device_details['autopilot']['info'] = ap_name
                    self.device_details['autopilot']['port'] = udp_conn
                    self.device_details['autopilot']['baud'] = None
                    
                    return master, ap_name
                else:
                    self.logger.warning(f"No heartbeat received on UDP connection after {timeout}s")
            except Exception as e:
                self._log_exception("UDP connection", e)
            finally:
                if master and not keep_master:
                    try:
                        self.logger.debug("Closing UDP connection")
                        master.close()
                    except Exception as e:
                        self.logger.warning(f"Error closing UDP connection: {str(e)}")

            # Scan serial ports next
            self.logger.info("Starting serial port scan...")
            ports = []
            system_os = platform.system()
            self.logger.debug(f"Operating system: {system_os}")

            if system_os == "Windows":
                ports = [p for p in list_ports.comports() if exclude_ports is None or p.device not in exclude_ports]
                if not ports:
                    self.logger.warning("No COM ports detected, falling back to COM3-COM10")
                    ports = [type('P', (), {'device': f'COM{i}'}) for i in range(3, 11)]
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
            
            self.logger.info(f"Scanning {len(ports)} serial ports")
            self.logger.debug(f"Ports to scan: {[p.device for p in ports]}")

            for portinfo in ports:
                port = portinfo.device
                self.logger.info(f"Processing port: {port}")
                
                for baud in (9600, 57600, 115200):
                    master = None
                    try:
                        self.logger.debug(f"Trying {port} @ {baud} baud")
                        master = mavutil.mavlink_connection(
                            port,
                            baud=baud,
                            source_system=255,
                            mavlink_version=1
                        )
                        self.logger.info(f"Listening for heartbeat on {port} @ {baud} baud")
                        hb = master.recv_match(type='HEARTBEAT', blocking=True, timeout=timeout)
                        
                        if hb:
                            ap_t = hb.autopilot
                            ap_name = mavutil.mavlink.enums['MAV_AUTOPILOT'][ap_t].description
                            self.logger.info(f"SUCCESS: Autopilot on {port} @ {baud} | Type: {ap_name} | System: {hb.get_srcSystem()}")
                            
                            # Store autopilot details
                            self.device_details['autopilot']['connection'] = master
                            self.device_details['autopilot']['info'] = ap_name
                            self.device_details['autopilot']['port'] = port
                            self.device_details['autopilot']['baud'] = baud
                            
                            return master, ap_name
                        else:
                            self.logger.warning(f"No heartbeat on {port} @ {baud} after {timeout}s")
                    except Exception as e:
                        self._log_exception(f"serial connection {port} @ {baud}", e)
                    finally:
                        if master and not keep_master:
                            try:
                                self.logger.debug(f"Closing {port} connection")
                                master.close()
                            except Exception as e:
                                self.logger.warning(f"Error closing {port}: {str(e)}")

            # If we reach here, no connection was successful
            error_msg = "No autopilot heartbeat detected on any interface"
            self.logger.error(error_msg)
            self.logger.debug(f"Scanned ports: {[p.device for p in ports]}")
            self.logger.debug(f"Excluded ports: {exclude_ports}")
            return None, None
            
        except Exception as e:
            self._log_exception("find_autopilot_connection", e)
            raise

    def get_device_details(self):
        """Return discovered device details"""
        self.logger.debug("Retrieving device details")
        return self.device_details