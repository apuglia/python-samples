import platform
import sys
import time
import csv
from datetime import datetime
import threading
from collections import deque

# Import plotting libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import numpy as np
    PLOTTING_AVAILABLE = True
except ImportError:
    print("Matplotlib not available. Install with: pip install matplotlib")
    PLOTTING_AVAILABLE = False

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

# Handle Windows compatibility - find available version
if platform.system() == "Windows":
    import os
    target_dir = f"PLUX-API-Python3/{osDic[platform.system()]}"
    if not os.path.exists(target_dir):
        print(f"Directory {target_dir} not found")
        # Try available Windows versions in order of preference (prioritize older versions for compatibility)
        fallback_versions = ["Win64_310", "Win64_39", "Win64_38", "Win64_37", "Win64_312"]
        for version in fallback_versions:
            if os.path.exists(f"PLUX-API-Python3/{version}"):
                print(f"Using {version} instead")
                osDic["Windows"] = version
                break
        else:
            print("No compatible Windows version found!")
            exit(1)
    
    # Try to load the plux library with proper DLL handling
    plux_dir = f"PLUX-API-Python3/{osDic[platform.system()]}"
    dll_path = os.path.abspath(plux_dir)
    
    # Add DLL path to PATH environment variable
    os.environ["PATH"] = dll_path + ";" + os.environ["PATH"]
    
    # Change to DLL directory temporarily
    original_cwd = os.getcwd()
    try:
        os.chdir(dll_path)
        sys.path.append(dll_path)
        
        # Try to import plux
        import plux
        print(f"Successfully loaded plux from {osDic['Windows']}")
        
    except ImportError as e:
        print(f"Failed to load plux from {osDic['Windows']}: {e}")
        # Try other versions if the first one fails
        fallback_versions = ["Win64_310", "Win64_39", "Win64_38", "Win64_37", "Win64_312"]
        for version in fallback_versions:
            if version != osDic["Windows"] and os.path.exists(f"PLUX-API-Python3/{version}"):
                print(f"Trying {version}...")
                try:
                    os.chdir(original_cwd)
                    new_dll_path = os.path.abspath(f"PLUX-API-Python3/{version}")
                    os.chdir(new_dll_path)
                    sys.path.clear()
                    sys.path.append(new_dll_path)
                    
                    import plux
                    print(f"Successfully loaded plux from {version}")
                    osDic["Windows"] = version
                    break
                except ImportError:
                    continue
        else:
            print("Could not load plux from any available version")
            exit(1)
    
    finally:
        os.chdir(original_cwd)
        
else:
    sys.path.append(f"PLUX-API-Python3/{osDic[platform.system()]}")
    import plux


class CardioBanRealTimePlotter:
    def __init__(self, num_channels=3, window_size=5000):
        self.num_channels = num_channels
        self.window_size = window_size
        
        # Data storage - using deque for efficient append/pop operations
        self.data_buffers = [deque(maxlen=window_size) for _ in range(num_channels)]
        self.time_buffer = deque(maxlen=window_size)
        
        # Threading control
        self.data_lock = threading.Lock()
        self.is_running = False
        
        # Setup matplotlib
        if PLOTTING_AVAILABLE:
            self.setup_plot()
    
    def setup_plot(self):
        """Initialize the matplotlib plot"""
        self.fig, self.axes = plt.subplots(self.num_channels, 1, figsize=(12, 8))
        if self.num_channels == 1:
            self.axes = [self.axes]
        
        # Configure each subplot
        self.lines = []
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        
        for i, ax in enumerate(self.axes):
            line, = ax.plot([], [], color=colors[i % len(colors)], linewidth=1)
            self.lines.append(line)
            ax.set_title(f'Channel {i+1}')
            ax.set_ylabel('Amplitude')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, self.window_size)
        
        self.axes[-1].set_xlabel('Samples')
        plt.tight_layout()
        plt.ion()  # Enable interactive mode
        plt.show()
    
    def add_data(self, timestamp, data):
        """Add new data point to buffers"""
        with self.data_lock:
            self.time_buffer.append(timestamp)
            for i, value in enumerate(data):
                if i < self.num_channels:
                    self.data_buffers[i].append(value)
    
    def update_plot(self):
        """Update the plot with current data"""
        if not PLOTTING_AVAILABLE:
            return
        
        with self.data_lock:
            # Update each line
            for i, line in enumerate(self.lines):
                if len(self.data_buffers[i]) > 0:
                    x_data = list(range(len(self.data_buffers[i])))
                    y_data = list(self.data_buffers[i])
                    line.set_data(x_data, y_data)
                    
                    # Auto-scale y-axis
                    if y_data:
                        y_min, y_max = min(y_data), max(y_data)
                        margin = (y_max - y_min) * 0.1
                        self.axes[i].set_ylim(y_min - margin, y_max + margin)
        
        # Update x-axis
        current_len = len(self.time_buffer)
        if current_len > 0:
            for ax in self.axes:
                ax.set_xlim(0, max(current_len, 100))
        
        plt.pause(0.001)  # Small pause to allow plot update


class CardioBanDevice(plux.SignalsDev):
    def __init__(self, address, plotter=None):
        plux.SignalsDev.__init__(address)
        self.duration = 0
        self.frequency = 0
        self.start_time = time.time()
        self.csv_writer = None
        self.csv_file = None
        self.sample_count = 0
        self.plotter = plotter
        
    def setup_csv_logging(self, filename=None):
        """Setup CSV file for data logging"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"cardioban_data_{timestamp}.csv"
        
        self.csv_file = open(filename, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        # Write header
        header = ['timestamp', 'sequence'] + [f'channel_{i+1}' for i in range(6)]
        self.csv_writer.writerow(header)
        print(f"Data logging to: {filename}")

    def onRawFrame(self, nSeq, data):
        """Process incoming data frames"""
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        # Add data to plotter
        if self.plotter:
            self.plotter.add_data(elapsed_time, data)
        
        # Print real-time data every 1000 samples
        if nSeq % 1000 == 0:
            print(f"Time: {elapsed_time:.2f}s | Sample: {nSeq} | Data: {data}")
        
        # Log to CSV if enabled
        if self.csv_writer:
            self.csv_writer.writerow([elapsed_time, nSeq] + list(data))
        
        self.sample_count = nSeq
        
        # Stop condition
        return nSeq > self.duration * self.frequency
    
    def close(self):
        """Clean shutdown"""
        if self.csv_file:
            self.csv_file.close()
        super().close()


def plot_update_thread(plotter, device):
    """Thread function for updating plots"""
    while plotter.is_running:
        try:
            plotter.update_plot()
            time.sleep(0.05)  # 20 FPS update rate
        except Exception as e:
            print(f"Plot update error: {e}")
            break


def start_cardioban_realtime_plot(
    address="BTH38:5B:44:77:FF:EF",
    duration=60,
    frequency=1000,
    active_ports=[1, 2, 3],
    log_to_csv=True,
    show_plot=True
):
    """
    Start real-time acquisition with live plotting
    
    Args:
        address: Bluetooth address of cardioban device
        duration: Recording duration in seconds
        frequency: Sampling frequency (Hz)
        active_ports: List of active sensor ports
        log_to_csv: Save data to CSV file
        show_plot: Show real-time plot
    """
    
    print(f"Starting cardioban real-time acquisition with visualization...")
    print(f"Device: {address}")
    print(f"Duration: {duration} seconds")
    print(f"Frequency: {frequency} Hz")
    print(f"Active ports: {active_ports}")
    print("Press Ctrl+C to stop early\n")
    
    # Setup plotter
    plotter = None
    plot_thread = None
    
    if show_plot and PLOTTING_AVAILABLE:
        plotter = CardioBanRealTimePlotter(num_channels=len(active_ports))
        plotter.is_running = True
        plot_thread = threading.Thread(target=plot_update_thread, args=(plotter, None))
        plot_thread.daemon = True
        plot_thread.start()
    elif show_plot:
        print("Warning: Plotting not available. Install matplotlib to enable visualization.")
    
    # Setup device
    device = CardioBanDevice(address, plotter)
    device.duration = int(duration)
    device.frequency = int(frequency)
    
    # Setup CSV logging if requested
    if log_to_csv:
        device.setup_csv_logging()
    
    try:
        # Start acquisition
        device.start(device.frequency, active_ports, 16)
        
        print("Acquisition started! Real-time plot should appear...")
        
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
        if plotter:
            plotter.is_running = False
        if plot_thread:
            plot_thread.join(timeout=1)
        
        device.stop()
        device.close()
        
        if PLOTTING_AVAILABLE:
            plt.ioff()
            plt.close('all')


if __name__ == "__main__":
    if not PLOTTING_AVAILABLE:
        print("To enable real-time plotting, install matplotlib:")
        print("pip install matplotlib")
        print()
    
    if len(sys.argv) > 1:
        # Use command line arguments
        start_cardioban_realtime_plot(*sys.argv[1:])
    else:
        # Default configuration
        start_cardioban_realtime_plot(
            address="BTH38:5B:44:77:FF:EF",
            duration=60,                     # 60 seconds
            frequency=1000,                  # 1000 Hz
            active_ports=[1, 2, 3],         # Use ports 1, 2, 3
            log_to_csv=True,                 # Save to CSV
            show_plot=True                   # Show real-time plot
        )