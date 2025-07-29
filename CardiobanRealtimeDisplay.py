import platform
import sys
import threading
import time
from collections import deque
import numpy as np

# Platform-specific library loading (same as other examples)
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

try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Install with: pip install matplotlib")
    print("Falling back to console output only.")


class CardiobanRealtimeDevice(plux.SignalsDev):
    def __init__(self, address, channels=1, buffer_size=1000, display_mode='both'):
        plux.MemoryDev.__init__(address)
        self.channels = channels
        self.frequency = 0
        self.running = False
        self.display_mode = display_mode  # 'console', 'plot', 'both'
        
        # Data buffering
        self.buffer_size = buffer_size
        self.data_buffers = [deque(maxlen=buffer_size) for _ in range(channels)]
        self.time_buffer = deque(maxlen=buffer_size)
        self.sample_count = 0
        self.start_time = 0
        
        # Thread safety
        self.data_lock = threading.Lock()
        
        # Statistics
        self.min_values = [float('inf')] * channels
        self.max_values = [float('-inf')] * channels
        self.last_values = [0] * channels

    def onRawFrame(self, nSeq, data):
        current_time = time.time() - self.start_time
        
        with self.data_lock:
            # Store data in buffers
            self.time_buffer.append(current_time)
            
            for i in range(min(len(data), self.channels)):
                value = data[i]
                self.data_buffers[i].append(value)
                self.last_values[i] = value
                
                # Update statistics
                if value < self.min_values[i]:
                    self.min_values[i] = value
                if value > self.max_values[i]:
                    self.max_values[i] = value
            
            self.sample_count = nSeq
        
        # Console output
        if self.display_mode in ['console', 'both']:
            if nSeq % 100 == 0:  # Print every 100 samples to avoid overwhelming output
                print(f"Sample {nSeq:6d} | Time: {current_time:7.2f}s | Data: {data[:self.channels]}")
        
        return not self.running

    def start_acquisition(self, frequency=1000, duration=None):
        self.frequency = frequency
        self.running = True
        self.start_time = time.time()
        
        # Calculate channel code based on number of channels
        channel_codes = {1: 0x01, 2: 0x03, 3: 0x07, 4: 0x0F, 
                        5: 0x1F, 6: 0x3F, 7: 0x7F, 8: 0xFF}
        code = channel_codes.get(self.channels, 0x01)
        
        print(f"Starting acquisition...")
        print(f"Channels: {self.channels} (code: 0x{code:02X})")
        print(f"Frequency: {frequency} Hz")
        print(f"Duration: {'Continuous' if duration is None else f'{duration}s'}")
        print("-" * 50)
        
        # Start the device
        self.start(frequency, code, 16)
        
        if duration:
            # Run for specific duration
            threading.Timer(duration, self.stop_acquisition).start()
        
        # Start the acquisition loop in a separate thread
        self.acquisition_thread = threading.Thread(target=self.loop)
        self.acquisition_thread.start()

    def stop_acquisition(self):
        print("\nStopping acquisition...")
        self.running = False
        if hasattr(self, 'acquisition_thread'):
            self.acquisition_thread.join()
        self.stop()
        self.close()
        self.print_statistics()

    def print_statistics(self):
        print("\n" + "="*50)
        print("ACQUISITION STATISTICS")
        print("="*50)
        print(f"Total samples: {self.sample_count}")
        print(f"Duration: {len(self.time_buffer) / self.frequency:.2f}s")
        print(f"Average rate: {self.sample_count / (len(self.time_buffer) / self.frequency):.1f} Hz")
        
        for i in range(self.channels):
            print(f"\nChannel {i+1}:")
            print(f"  Last value: {self.last_values[i]}")
            print(f"  Min value: {self.min_values[i]}")
            print(f"  Max value: {self.max_values[i]}")
            print(f"  Range: {self.max_values[i] - self.min_values[i]}")

    def get_plot_data(self):
        with self.data_lock:
            if len(self.time_buffer) == 0:
                return [], [[] for _ in range(self.channels)]
            
            times = list(self.time_buffer)
            data_arrays = [list(buffer) for buffer in self.data_buffers]
            return times, data_arrays


class RealtimePlotter:
    def __init__(self, device, update_interval=50):
        self.device = device
        self.update_interval = update_interval
        
        # Create the plot
        self.fig, self.axes = plt.subplots(device.channels, 1, figsize=(12, 3*device.channels))
        if device.channels == 1:
            self.axes = [self.axes]
        
        self.lines = []
        for i, ax in enumerate(self.axes):
            line, = ax.plot([], [], 'b-', linewidth=1)
            self.lines.append(line)
            ax.set_title(f'Channel {i+1} - Cardioban Real-time Data')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Signal Value')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()

    def update_plot(self, frame):
        times, data_arrays = self.device.get_plot_data()
        
        if len(times) > 1:
            for i, (line, data) in enumerate(zip(self.lines, data_arrays)):
                if len(data) > 0:
                    line.set_data(times[:len(data)], data)
                    
                    # Auto-scale axes
                    ax = self.axes[i]
                    if len(times) > 0 and len(data) > 0:
                        ax.set_xlim(min(times), max(times))
                        if len(data) > 1:
                            data_min, data_max = min(data), max(data)
                            margin = (data_max - data_min) * 0.1
                            ax.set_ylim(data_min - margin, data_max + margin)
        
        return self.lines

    def start_animation(self):
        self.ani = animation.FuncAnimation(
            self.fig, self.update_plot, interval=self.update_interval, blit=False
        )
        plt.show()


def main():
    # Configuration
    address = "BTH00:07:80:4D:2E:76"  # Default Cardioban address
    channels = 1  # Number of channels to acquire
    frequency = 1000  # Sampling frequency in Hz
    duration = None  # Duration in seconds (None for continuous)
    display_mode = 'both'  # 'console', 'plot', 'both'
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        address = sys.argv[1]
    if len(sys.argv) > 2:
        channels = int(sys.argv[2])
    if len(sys.argv) > 3:
        frequency = int(sys.argv[3])
    if len(sys.argv) > 4:
        duration = float(sys.argv[4])
    if len(sys.argv) > 5:
        display_mode = sys.argv[5]
    
    # Validate parameters
    if channels < 1 or channels > 8:
        print("Error: Channels must be between 1 and 8")
        return
    
    max_freq = {1: 8000, 2: 5000, 3: 4000, 4: 3000, 5: 3000, 6: 2000, 7: 2000, 8: 2000}
    if frequency > max_freq.get(channels, 1000):
        print(f"Warning: Frequency {frequency} Hz may be too high for {channels} channels")
        print(f"Maximum recommended: {max_freq.get(channels, 1000)} Hz")
    
    # Create device
    device = CardiobanRealtimeDevice(address, channels=channels, display_mode=display_mode)
    
    try:
        # Start acquisition
        device.start_acquisition(frequency=frequency, duration=duration)
        
        # Start plotting if matplotlib is available and requested
        if HAS_MATPLOTLIB and display_mode in ['plot', 'both']:
            plotter = RealtimePlotter(device)
            plotter.start_animation()
        else:
            # Wait for acquisition to complete
            if hasattr(device, 'acquisition_thread'):
                device.acquisition_thread.join()
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if device.running:
            device.stop_acquisition()


if __name__ == "__main__":
    print("Cardioban Real-time Data Display")
    print("Usage: python CardiobanRealtimeDisplay.py [address] [channels] [frequency] [duration] [display_mode]")
    print("Example: python CardiobanRealtimeDisplay.py BTH00:07:80:4D:2E:76 2 1000 30 both")
    print("Display modes: 'console', 'plot', 'both'")
    print("Press Ctrl+C to stop acquisition\n")
    
    main()