# main.py
import logging
import time
import argparse
from system_logger import init_logger, get_logger, set_console_log_level
from port_finder import DevicePortFinder
from radar_interface import RadarParser
import os


def main():
    # Initialize centralized logging with INFO level
    init_logger(console_level=logging.INFO)
    logger = get_logger('Main')

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Radar and Autopilot Interface')
    parser.add_argument('--config', default="best_res_4cm.cfg", help='Path to radar configuration file')
    args = parser.parse_args()

    config_path = os.path.join(os.getcwd(), args.config)

    try:
        # Find radar ports
        port_finder = DevicePortFinder()
        cli_port, data_port = port_finder.find_radar_ports_by_description()
        logger.info(f"Found radar ports: CLI={cli_port}, DATA={data_port}")

        # # Find autopilot (exclude radar ports)
        # ap_conn, ap_info = port_finder.find_autopilot_connection(
        #     exclude_ports=[cli_port, data_port]
        # )
        # if ap_info:
        #     logger.info(f"Autopilot connected: {ap_info}")

        # Initialize radar interface
        radar = RadarParser(
            cli_port=cli_port,
            data_port=data_port,
            config_file=config_path,
            cli_baud=115200,
            data_baud=921600,
        )
        radar.initialize_ports()
        radar.send_config()
        time.sleep(2)
        logger.info("Radar initialized and configured")

        # Radar data processing loop
        logger.info("Starting radar data processing...")
        # In main loop
        last_log_time = time.time()
        frame_count = 0

        time.sleep(2.0)

        while True:
            header, det_obj, snr, noise = radar.read_frame()

            if header is not None:
                frame_count += 1
                logger.info(f"Frame {frame_count}: {header['num_detected_obj']} objects")

                if det_obj is not None:
                    for i in range(det_obj['numObj']):
                        logger.info(f"  Object {i + 1}: "
                                    f"x={det_obj['x'][i]:.2f}m, "
                                    f"y={det_obj['y'][i]:.2f}m, "
                                    f"z={det_obj['z'][i]:.2f}m, "
                                    f"vel={det_obj['velocity'][i]:.2f}m/s")

            # Periodic status log
            if time.time() - last_log_time > 5:
                logger.info(f"Status: Processed {frame_count} frames | "
                            f"Buffer: dont know bytes")
                last_log_time = time.time()

            time.sleep(0.001)

    except KeyboardInterrupt:
        logger.info("Application terminated by user")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
    finally:
        if 'radar' in locals():
            radar.close()


if __name__ == "__main__":
    main()