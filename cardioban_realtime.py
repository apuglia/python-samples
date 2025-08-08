import platform
import sys
import time
import csv
from datetime import datetime

osDic = {
    "Darwin": f"MacOS/Intel{''.join(platform.python_version().split('.')[:2])}",
    "Linux": "Linux64",
    "Windows": f"Win{platform.architecture()[0][:2]}_{''.join(platform.python_version().split('.')[:2])}",
}
if platform.mac_ver()[0] != "":
    import subprocess
    from os import linesep

    p = subprocess.Popen("sw_vers", stdout=subprocess.PIPE)
    result = p.communicate()[0].decode("utf-8").split(str("\t"))[2].split(linesep)[0]
    if result.startswith("12."):
        print("macOS version is Monterrey!")
        osDic["Darwin"] = "MacOS/Intel310"
        if (
            int(platform.python_version().split(".")[0]) <= 3
            and int(platform.python_version().split(".")[1]) < 10
        ):
            print(f"Python version required is ≥ 3.10. Installed is {platform.python_version()}")
            exit()

sys.path.append(f"PLUX-API-Python3/{osDic[platform.system()]}")

import plux


class CardioBanDevice(plux.SignalsDev):
    def __init__(self, address):
        plux.SignalsDev.__init__(address)
        self.duration = 0
        self.frequency = 0
        self.start_time = time.time()
        self.csv_writer = None
        self.csv_file = None
        self.sample_count = 0
        
    def setup_csv_logging(self, filename=None):
        """Setup CSV file for data logging"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"cardioban_data_{timestamp}.csv"
        
        self.csv_file = open(filename, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        # Write header
        self.csv_writer.writerow(['timestamp', 'sequence', 'channel_1', 'channel_2', 'channel_3', 'channel_4', 'channel_5', 'channel_6'])
        print(f"Data logging to: {filename}")

    def onRawFrame(self, nSeq, data):
        """Process incoming data frames"""
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        # Print real-time data every 1000 samples (adjust as needed)
        if nSeq % 1000 == 0:
            print(f"Time: {elapsed_time:.2f}s | Sample: {nSeq} | Data: {data}")
        
        # Log to CSV if enabled
        if self.csv_writer:
            self.csv_writer.writerow([elapsed_time, nSeq] + list(data))
        
        self.sample_count = nSeq
        
        # Stop condition: either duration reached or manual stop
        return nSeq > self.duration * self.frequency
    
    def close(self):
        """Clean shutdown"""
        if self.csv_file:
            self.csv_file.close()
        super().close()


def start_cardioban_acquisition(
    address="BTH38:5B:44:77:FF:EF",  # Your converted address
    duration=10,                     # Duration in seconds
    frequency=1000,                  # Sampling frequency (Hz)
    active_ports=[1, 2, 3],         # Active sensor ports
    log_to_csv=True                  # Enable CSV logging
):
    """
    Start real-time acquisition from cardioban device
    
    Args:
        address: Bluetooth address of your cardioban device
        duration: Recording duration in seconds
        frequency: Sampling frequency (1000 Hz recommended for cardio)
        active_ports: List of active sensor ports [1, 2, 3, 4, 5, 6]
        log_to_csv: Save data to CSV file
    """
    
    print(f"Starting cardioban acquisition...")
    print(f"Device: {address}")
    print(f"Duration: {duration} seconds")
    print(f"Frequency: {frequency} Hz")
    print(f"Active ports: {active_ports}")
    print("Press Ctrl+C to stop early\n")
    
    device = CardioBanDevice(address)
    device.duration = int(duration)
    device.frequency = int(frequency)
    
    # Setup CSV logging if requested
    if log_to_csv:
        device.setup_csv_logging()
    
    try:
        # Start acquisition
        device.start(device.frequency, active_ports, 16)
        
        # Main acquisition loop
        device.loop()
        
        print(f"\nAcquisition completed!")
        print(f"Total samples collected: {device.sample_count}")
        
    except KeyboardInterrupt:
        print(f"\nAcquisition stopped by user")
        print(f"Samples collected: {device.sample_count}")
        
    except Exception as e:
        print(f"Error during acquisition: {e}")
        
    finally:
        # Clean shutdown
        device.stop()
        device.close()


if __name__ == "__main__":
    # You can run this script with command line arguments or modify the defaults below
    
    if len(sys.argv) > 1:
        # Use command line arguments
        start_cardioban_acquisition(*sys.argv[1:])
    else:
        # Default configuration - modify these values as needed
        start_cardioban_acquisition(
            address="BTH38:5B:44:77:FF:EF",  # Your device address
            duration=30,                     # 30 seconds
            frequency=1000,                  # 1000 Hz
            active_ports=[1, 2, 3],         # Use ports 1, 2, 3
            log_to_csv=True                  # Save to CSV
        )