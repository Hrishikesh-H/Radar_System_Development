# main.py
import logging
import time
import argparse
import os
import csv
from datetime import datetime
from system_logger import init_logger, get_logger, set_console_log_level
from port_finder import DevicePortFinder
from radar_interface import RadarParser
from radar_despiker import RadarDespiker

def main():
    # Initialize centralized logging with INFO level
    init_logger(console_level=logging.INFO)
    logger = get_logger('Main')

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Radar and Autopilot Interface')
    parser.add_argument('--config', default="best_res_4cm.cfg", help='Path to radar configuration file')
    parser.add_argument('--output', default="radar_data", help='Output CSV file name')
    args = parser.parse_args()

    config_path = os.path.join(os.getcwd(), args.config)
    output_file = os.path.join(os.getcwd(), args.output)

    # Prepare CSV file for logging data
    csv_file = None
    csv_writer = None
    
    try:
        # Open CSV file and write header
        csv_file = open(output_file, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            'timestamp', 'frame_id', 'object_id', 
            'original_x', 'original_y', 'original_z',
            'despiked_x', 'despiked_y', 'despiked_z',
            'snr', 'noise', 'num_objects'
        ])
        logger.info(f"Data logging to: {output_file}")

        # Find radar ports
        port_finder = DevicePortFinder()
        cli_port, data_port = port_finder.find_radar_ports_by_description()
        logger.info(f"Found radar ports: CLI={cli_port}, DATA={data_port}")

        # Initialize radar interface
        radar = RadarParser(
            cli_port=data_port,
            data_port=cli_port,
            config_file=config_path,
            cli_baud=115200,
            data_baud=921600,
        )
        radar.initialize_ports()
        radar.send_config()
        time.sleep(2)
        logger.info("Radar initialized and configured")

        # Initialize radar despiker
        despiker = RadarDespiker()
        logger.info("Radar despiker initialized")

        # Radar data processing loop
        logger.info("Starting radar data processing...")
        last_log_time = time.time()
        frame_count = 0

        time.sleep(2.0)

        while True:
            header, det_obj, snr, noise = radar.read_frame()

            if header is not None:
                frame_count += 1
                timestamp = datetime.now().isoformat()
                num_objects = header['num_detected_obj']
                
                # Store original data before processing
                original_det_obj = det_obj

                # Apply despiker to the detection objects
                if det_obj is not None:
                    despiked_det_obj = despiker.process(det_obj, snr)
                else:
                    despiked_det_obj = None

                logger.info(f"Frame {frame_count}: {num_objects} objects")

                # Log to CSV
                if original_det_obj is not None and despiked_det_obj is not None:
                    num_objects = min(len(original_det_obj['x']), len(despiked_det_obj['x']))
                    for i in range(num_objects):
                        # Get SNR and noise values if available
                        snr_val = snr[i] if snr is not None and i < len(snr) else -1
                        noise_val = noise[i] if noise is not None and i < len(noise) else -1
                        
                        csv_writer.writerow([
                            timestamp, frame_count, i+1,
                            original_det_obj['x'][i], original_det_obj['y'][i], original_det_obj['z'][i],
                            despiked_det_obj['x'][i], despiked_det_obj['y'][i], despiked_det_obj['z'][i],
                            snr_val, noise_val, num_objects
                        ])
                
                # For debugging: log first object
                if despiked_det_obj is not None and despiked_det_obj['numObj'] > 0:
                    logger.info(f"  Object 1: "
                                f"x={despiked_det_obj['x'][0]:.2f}m, "
                                f"y={despiked_det_obj['y'][0]:.2f}m, "
                                f"z={despiked_det_obj['z'][0]:.2f}m")

            # Periodic status log and CSV flush
            if time.time() - last_log_time > 5:
                logger.info(f"Status: Processed {frame_count} frames")
                last_log_time = time.time()
                if csv_file:
                    csv_file.flush()  # Ensure data is written to disk

            time.sleep(0.001)

    except KeyboardInterrupt:
        logger.info("Application terminated by user")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
    finally:
        if 'radar' in locals():
            radar.close()
        if csv_file:
            csv_file.close()
            logger.info(f"CSV file closed: {output_file}")


if __name__ == "__main__":
    main()