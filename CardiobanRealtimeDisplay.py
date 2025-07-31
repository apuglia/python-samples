import platform
import sys
import threading
import time
from collections import deque
import numpy as np
from scipy import signal
from ecg_analysis import ECGQualityAnalyzer

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
    def __init__(self, address, channels=1, buffer_size=1000, display_mode='both', enable_filters=True):
        plux.MemoryDev.__init__(address)
        self.channels = channels
        self.frequency = 0
        self.running = False
        self.display_mode = display_mode  # 'console', 'plot', 'both'
        self.enable_filters = enable_filters
        
        # Data buffering
        self.buffer_size = buffer_size
        self.data_buffers = [deque(maxlen=buffer_size) for _ in range(channels)]
        self.filtered_buffers = [deque(maxlen=buffer_size) for _ in range(channels)]
        self.time_buffer = deque(maxlen=buffer_size)
        self.sample_count = 0
        self.start_time = 0
        
        # Thread safety
        self.data_lock = threading.Lock()
        
        # Statistics
        self.min_values = [float('inf')] * channels
        self.max_values = [float('-inf')] * channels
        self.last_values = [0] * channels
        
        # Filter states
        self.filter_initialized = False
        self.highpass_zi = None
        self.lowpass_zi = None
        self.notch_zi = None
        
        # ECG Quality Analyzer (will be initialized with proper frequency later)
        self.quality_analyzer = None

    def initialize_filters(self, frequency):
        """Initialize ECG signal filters"""
        if not self.enable_filters:
            return
            
        # High-pass filter: removes baseline wander (0.5 Hz cutoff)
        self.highpass_sos = signal.butter(4, 0.5, btype='high', fs=frequency, output='sos')
        self.highpass_zi = [signal.sosfilt_zi(self.highpass_sos) for _ in range(self.channels)]
        
        # Low-pass filter: removes high-frequency noise (40 Hz cutoff)
        self.lowpass_sos = signal.butter(4, 40, btype='low', fs=frequency, output='sos')
        self.lowpass_zi = [signal.sosfilt_zi(self.lowpass_sos) for _ in range(self.channels)]
        
        # Notch filter: removes 50 Hz powerline interference
        b_notch, a_notch = signal.iirnotch(50, 30, fs=frequency)
        self.notch_sos = signal.tf2sos(b_notch, a_notch)
        self.notch_zi = [signal.sosfilt_zi(self.notch_sos) for _ in range(self.channels)]
        
        self.filter_initialized = True
        print("ECG filters initialized:")
        print("  - High-pass: 0.5 Hz (baseline wander removal)")
        print("  - Low-pass: 40 Hz (noise reduction)")
        print("  - Notch: 50 Hz (powerline interference removal)")

    def apply_filters(self, data):
        """Apply real-time ECG filters to data"""
        if not self.enable_filters or not self.filter_initialized:
            return data
            
        filtered_data = []
        for i, value in enumerate(data[:self.channels]):
            # Convert to float array for filtering
            x = np.array([float(value)])
            
            # Apply high-pass filter
            x, self.highpass_zi[i] = signal.sosfilt(self.highpass_sos, x, zi=self.highpass_zi[i])
            
            # Apply low-pass filter
            x, self.lowpass_zi[i] = signal.sosfilt(self.lowpass_sos, x, zi=self.lowpass_zi[i])
            
            # Apply notch filter
            x, self.notch_zi[i] = signal.sosfilt(self.notch_sos, x, zi=self.notch_zi[i])
            
            filtered_data.append(x[0])
            
        return filtered_data

    def onRawFrame(self, nSeq, data):
        current_time = time.time() - self.start_time
        
        # Apply filters to the data
        filtered_data = self.apply_filters(data)
        
        with self.data_lock:
            # Store data in buffers
            self.time_buffer.append(current_time)
            
            for i in range(min(len(data), self.channels)):
                # Store raw data
                raw_value = data[i]
                self.data_buffers[i].append(raw_value)
                
                # Store filtered data
                if self.enable_filters and filtered_data:
                    filtered_value = filtered_data[i]
                    self.filtered_buffers[i].append(filtered_value)
                    self.last_values[i] = filtered_value
                    
                    # Update statistics with filtered data
                    if filtered_value < self.min_values[i]:
                        self.min_values[i] = filtered_value
                    if filtered_value > self.max_values[i]:
                        self.max_values[i] = filtered_value
                else:
                    # Use raw data if no filtering
                    self.filtered_buffers[i].append(raw_value)
                    self.last_values[i] = raw_value
                    
                    if raw_value < self.min_values[i]:
                        self.min_values[i] = raw_value
                    if raw_value > self.max_values[i]:
                        self.max_values[i] = raw_value
            
            self.sample_count = nSeq
        
        # Add samples to quality analyzer (use filtered data if available)
        if self.quality_analyzer is not None:
            for i in range(min(len(data), self.channels)):
                analysis_value = filtered_data[i] if (self.enable_filters and filtered_data) else data[i]
                self.quality_analyzer.add_sample(analysis_value, current_time)
                
                # Debug: Print first few samples to verify data flow
                if nSeq < 5:
                    print(f"DEBUG: Adding sample {nSeq}: value={analysis_value}, time={current_time}")
                    print(f"DEBUG: Buffer size now: {len(self.quality_analyzer.signal_buffer)}")
        
        # Console output
        if self.display_mode in ['console', 'both']:
            if nSeq % 100 == 0:  # Print every 100 samples to avoid overwhelming output
                if self.enable_filters and filtered_data:
                    print(f"Sample {nSeq:6d} | Time: {current_time:7.2f}s | Raw: {data[:self.channels]} | Filtered: {[f'{x:.1f}' for x in filtered_data[:self.channels]]}")
                else:
                    print(f"Sample {nSeq:6d} | Time: {current_time:7.2f}s | Data: {data[:self.channels]}")
                
                # Show quality metrics every 500 samples
                if nSeq % 500 == 0 and self.quality_analyzer is not None:
                    quality_str = self.quality_analyzer.get_quality_string()
                    print(f"         >>> {quality_str}")
        
        return not self.running

    def start_acquisition(self, frequency=1000, duration=None):
        self.frequency = frequency
        self.running = True
        self.start_time = time.time()
        
        # Initialize filters
        self.initialize_filters(frequency)
        
        # Initialize quality analyzer
        self.quality_analyzer = ECGQualityAnalyzer(sampling_rate=frequency, buffer_size=self.buffer_size)
        
        # Calculate channel code based on number of channels
        channel_codes = {1: 0x01, 2: 0x03, 3: 0x07, 4: 0x0F, 
                        5: 0x1F, 6: 0x3F, 7: 0x7F, 8: 0xFF}
        code = channel_codes.get(self.channels, 0x01)
        
        print(f"Starting acquisition...")
        print(f"Channels: {self.channels} (code: 0x{code:02X})")
        print(f"Frequency: {frequency} Hz")
        print(f"Duration: {'Continuous' if duration is None else f'{duration}s'}")
        print(f"Filtering: {'Enabled' if self.enable_filters else 'Disabled'}")
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
                return [], [[] for _ in range(self.channels)], [[] for _ in range(self.channels)]
            
            times = list(self.time_buffer)
            raw_data_arrays = [list(buffer) for buffer in self.data_buffers]
            filtered_data_arrays = [list(buffer) for buffer in self.filtered_buffers]
            return times, raw_data_arrays, filtered_data_arrays


class RealtimePlotter:
    def __init__(self, device, update_interval=50, show_both=False):
        self.device = device
        self.update_interval = update_interval
        self.show_both = show_both
        
        # Create subplots - if showing both raw and filtered, create 2 rows per channel
        if show_both and device.enable_filters:
            rows = device.channels * 2
            self.fig, self.axes = plt.subplots(rows, 1, figsize=(12, 2*rows))
            if rows == 1:
                self.axes = [self.axes]
        else:
            self.fig, self.axes = plt.subplots(device.channels, 1, figsize=(12, 3*device.channels))
            if device.channels == 1:
                self.axes = [self.axes]
        
        self.raw_lines = []
        self.filtered_lines = []
        self.quality_text = None
        
        if show_both and device.enable_filters:
            # Create lines for both raw and filtered data
            for i in range(device.channels):
                # Raw data subplot
                raw_ax = self.axes[i*2]
                raw_line, = raw_ax.plot([], [], 'r-', linewidth=1, label='Raw')
                self.raw_lines.append(raw_line)
                raw_ax.set_title(f'Channel {i+1} - Raw ECG Data')
                raw_ax.set_ylabel('Raw Signal')
                raw_ax.grid(True, alpha=0.3)
                raw_ax.legend()
                
                # Filtered data subplot
                filtered_ax = self.axes[i*2 + 1]
                filtered_line, = filtered_ax.plot([], [], 'b-', linewidth=1, label='Filtered')
                self.filtered_lines.append(filtered_line)
                filtered_ax.set_title(f'Channel {i+1} - Filtered ECG Data')
                filtered_ax.set_xlabel('Time (s)')
                filtered_ax.set_ylabel('Filtered Signal')
                filtered_ax.grid(True, alpha=0.3)
                filtered_ax.legend()
        else:
            # Single plot per channel (raw or filtered based on device setting)
            for i, ax in enumerate(self.axes):
                if device.enable_filters:
                    line, = ax.plot([], [], 'b-', linewidth=1, label='Filtered')
                    self.filtered_lines.append(line)
                    ax.set_title(f'Channel {i+1} - Filtered ECG Data')
                else:
                    line, = ax.plot([], [], 'r-', linewidth=1, label='Raw')
                    self.raw_lines.append(line)
                    ax.set_title(f'Channel {i+1} - Raw ECG Data')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Signal Value')
                ax.grid(True, alpha=0.3)
                ax.legend()
        
        # Add quality metrics text box
        if device.enable_filters:
            # Add text box for quality metrics at the top
            self.quality_text = self.fig.text(0.02, 0.98, 'Quality Metrics: Initializing...', 
                                            fontsize=10, verticalalignment='top',
                                            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)  # Make room for quality text

    def update_plot(self, frame):
        times, raw_data_arrays, filtered_data_arrays = self.device.get_plot_data()
        
        if len(times) > 1:
            if self.show_both and self.device.enable_filters:
                # Update both raw and filtered plots
                all_lines = []
                for i in range(self.device.channels):
                    # Update raw data
                    if len(raw_data_arrays[i]) > 0:
                        self.raw_lines[i].set_data(times[:len(raw_data_arrays[i])], raw_data_arrays[i])
                        raw_ax = self.axes[i*2]
                        if len(times) > 0 and len(raw_data_arrays[i]) > 0:
                            raw_ax.set_xlim(min(times), max(times))
                            if len(raw_data_arrays[i]) > 1:
                                data_min, data_max = min(raw_data_arrays[i]), max(raw_data_arrays[i])
                                margin = (data_max - data_min) * 0.1
                                raw_ax.set_ylim(data_min - margin, data_max + margin)
                        all_lines.append(self.raw_lines[i])
                    
                    # Update filtered data
                    if len(filtered_data_arrays[i]) > 0:
                        self.filtered_lines[i].set_data(times[:len(filtered_data_arrays[i])], filtered_data_arrays[i])
                        filtered_ax = self.axes[i*2 + 1]
                        if len(times) > 0 and len(filtered_data_arrays[i]) > 0:
                            filtered_ax.set_xlim(min(times), max(times))
                            if len(filtered_data_arrays[i]) > 1:
                                data_min, data_max = min(filtered_data_arrays[i]), max(filtered_data_arrays[i])
                                margin = (data_max - data_min) * 0.1
                                filtered_ax.set_ylim(data_min - margin, data_max + margin)
                        all_lines.append(self.filtered_lines[i])
                return all_lines
            else:
                # Single plot mode - show either raw or filtered
                lines_to_update = self.filtered_lines if self.device.enable_filters else self.raw_lines
                data_to_use = filtered_data_arrays if self.device.enable_filters else raw_data_arrays
                
                for i, (line, data) in enumerate(zip(lines_to_update, data_to_use)):
                    if len(data) > 0:
                        line.set_data(times[:len(data)], data)
                        ax = self.axes[i]
                        if len(times) > 0 and len(data) > 0:
                            ax.set_xlim(min(times), max(times))
                            if len(data) > 1:
                                data_min, data_max = min(data), max(data)
                                margin = (data_max - data_min) * 0.1
                                ax.set_ylim(data_min - margin, data_max + margin)
                return lines_to_update
        
        # Update quality metrics text
        if self.quality_text and self.device.quality_analyzer:
            try:
                metrics = self.device.quality_analyzer.get_quality_metrics()
                
                # Check if we have enough data for meaningful metrics
                if len(self.device.quality_analyzer.signal_buffer) < 100:
                    quality_text = "📊 SIGNAL QUALITY: Collecting data... 📈"
                    color = 'lightgray'
                else:
                    # Create quality level indicator
                    quality_level = "Poor"
                    color = 'lightcoral'
                    if metrics['quality_score'] > 80:
                        quality_level = "Excellent"
                        color = 'lightgreen'
                    elif metrics['quality_score'] > 60:
                        quality_level = "Good"
                        color = 'lightblue'
                    elif metrics['quality_score'] > 40:
                        quality_level = "Fair"
                        color = 'lightyellow'
                    
                    # Format metrics text
                    quality_text = (
                        f"📊 SIGNAL QUALITY: {quality_level} ({metrics['quality_score']:.1f}/100)\n"
                        f"🔊 SNR: {metrics['snr']:.1f} dB  |  "
                        f"💓 Heart Rate: {metrics['heart_rate']:.1f} BPM  |  "
                        f"🔄 HRV: {metrics['hrv']:.1f} ms\n"
                        f"📈 QRS Correlation: {metrics['template_correlation']:.3f}  |  "
                        f"🫀 QRS Count: {metrics['qrs_count']}"
                    )
                
                self.quality_text.set_text(quality_text)
                self.quality_text.set_bbox(dict(boxstyle='round', facecolor=color, alpha=0.8))
            except Exception as e:
                # Fallback if there's an error
                self.quality_text.set_text(f"📊 SIGNAL QUALITY: Starting analysis... (samples: {len(times)})")
                self.quality_text.set_bbox(dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        return []

    def start_animation(self):
        self.ani = animation.FuncAnimation(
            self.fig, self.update_plot, interval=self.update_interval, blit=False
        )
        plt.show()


def main():
    # Configuration
    address = "BTH00:07:80:4D:2E:76"  # Default Cardioban address
    channels = 1  # Number of channels to acquire
    frequency = 500  # Sampling frequency in Hz
    duration = None  # Duration in seconds (None for continuous)
    display_mode = 'both'  # 'console', 'plot', 'both'
    enable_filters = True  # Enable ECG filtering by default
    
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
    if len(sys.argv) > 6:
        enable_filters = sys.argv[6].lower() in ['true', '1', 'yes', 'on']
    
    # Validate parameters
    if channels < 1 or channels > 8:
        print("Error: Channels must be between 1 and 8")
        return
    
    max_freq = {1: 8000, 2: 5000, 3: 4000, 4: 3000, 5: 3000, 6: 2000, 7: 2000, 8: 2000}
    if frequency > max_freq.get(channels, 1000):
        print(f"Warning: Frequency {frequency} Hz may be too high for {channels} channels")
        print(f"Maximum recommended: {max_freq.get(channels, 1000)} Hz")
    
    # Create device
    device = CardiobanRealtimeDevice(address, channels=channels, display_mode=display_mode, enable_filters=enable_filters)
    
    try:
        # Start acquisition
        device.start_acquisition(frequency=frequency, duration=duration)
        
        # Start plotting if matplotlib is available and requested
        if HAS_MATPLOTLIB and display_mode in ['plot', 'both']:
            # Show both raw and filtered plots if filters are enabled
            show_both = enable_filters
            plotter = RealtimePlotter(device, show_both=show_both)
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
    print("Usage: python CardiobanRealtimeDisplay.py [address] [channels] [frequency] [duration] [display_mode] [filters]")
    print("Example: python CardiobanRealtimeDisplay.py BTH00:07:80:4D:2E:76 2 1000 30 both true")
    print("Display modes: 'console', 'plot', 'both'")
    print("Filters: 'true'/'false' - Enable/disable ECG filtering")
    print("Press Ctrl+C to stop acquisition\n")
    
    main()