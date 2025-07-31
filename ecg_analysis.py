"""
ECG Signal Quality and QRS Analysis Module

This module provides real-time ECG signal quality assessment and QRS complex analysis.
"""

import numpy as np
from scipy import signal
from collections import deque
import threading


class QRSDetector:
    """Pan-Tompkins QRS detection algorithm implementation"""
    
    def __init__(self, sampling_rate=500):
        self.sampling_rate = sampling_rate
        self.window_size = int(0.15 * sampling_rate)  # 150ms window
        self.peak_buffer = deque(maxlen=10)
        self.rr_intervals = deque(maxlen=20)
        self.last_peak_time = 0
        self.threshold = 0
        
    def detect_qrs(self, ecg_signal):
        """Detect QRS complexes using simplified Pan-Tompkins algorithm"""
        if len(ecg_signal) < self.window_size:
            return [], {}
            
        # Differentiate
        diff_signal = np.diff(ecg_signal)
        
        # Square
        squared_signal = diff_signal ** 2
        
        # Moving average
        window = np.ones(self.window_size) / self.window_size
        filtered_signal = np.convolve(squared_signal, window, mode='same')
        
        # Adaptive threshold - use lower threshold for better detection
        max_val = np.max(filtered_signal)
        if max_val > 0:
            threshold = max_val * 0.1  # Lower threshold
        else:
            return [], {}
        
        # Find peaks with more sensitive parameters
        peaks, properties = signal.find_peaks(
            filtered_signal, 
            height=threshold,
            distance=int(0.3 * self.sampling_rate),  # 300ms minimum distance (50-200 BPM)
            prominence=threshold * 0.5
        )
        
        # Debug print
        if len(peaks) > 0:
            print(f"DEBUG QRS: Found {len(peaks)} peaks, max_val={max_val:.1f}, threshold={threshold:.1f}")
        
        return peaks, properties


class ECGQualityAnalyzer:
    """Real-time ECG signal quality analysis"""
    
    def __init__(self, sampling_rate=500, buffer_size=5000):
        self.sampling_rate = sampling_rate
        self.buffer_size = buffer_size
        
        # Data buffers
        self.signal_buffer = deque(maxlen=buffer_size)
        self.time_buffer = deque(maxlen=buffer_size)
        
        # QRS detection
        self.qrs_detector = QRSDetector(sampling_rate)
        self.qrs_templates = deque(maxlen=20)  # Store last 20 QRS complexes
        self.qrs_positions = deque(maxlen=50)
        self.qrs_times = deque(maxlen=50)
        
        # Quality metrics
        self.snr = 0
        self.template_correlation = 0
        self.quality_score = 0
        self.heart_rate = 0
        self.hrv = 0
        
        # Template matching
        self.master_template = None
        self.template_window_samples = int(0.3 * sampling_rate)  # 300ms window
        
        # Thread safety
        self.lock = threading.Lock()
        
    def add_sample(self, value, timestamp):
        """Add new ECG sample for analysis"""
        with self.lock:
            self.signal_buffer.append(value)
            self.time_buffer.append(timestamp)
            
            # Analyze when we have enough data
            if len(self.signal_buffer) >= self.sampling_rate:  # 1 second of data
                self._analyze_signal()
    
    def _analyze_signal(self):
        """Perform signal quality analysis"""
        if len(self.signal_buffer) < self.sampling_rate:
            return
            
        # Convert to numpy array for analysis
        signal_array = np.array(list(self.signal_buffer))
        time_array = np.array(list(self.time_buffer))
        
        # Calculate SNR
        self._calculate_snr(signal_array)
        
        # Detect QRS complexes
        self._detect_and_analyze_qrs(signal_array, time_array)
        
        # Calculate overall quality score
        self._calculate_quality_score()
    
    def _calculate_snr(self, signal_array):
        """Calculate Signal-to-Noise Ratio"""
        # Use high-frequency components as noise estimate
        b, a = signal.butter(4, 35, btype='high', fs=self.sampling_rate)
        noise = signal.filtfilt(b, a, signal_array)
        
        # Signal power (variance of original signal)
        signal_power = np.var(signal_array)
        
        # Noise power (variance of high-frequency components)
        noise_power = np.var(noise)
        
        # SNR in dB
        if noise_power > 0:
            self.snr = 10 * np.log10(signal_power / noise_power)
        else:
            self.snr = 60  # Very high SNR if no noise detected
    
    def _detect_and_analyze_qrs(self, signal_array, time_array):
        """Detect QRS complexes and analyze morphology"""
        # Use only recent data for QRS detection
        recent_samples = min(len(signal_array), int(2 * self.sampling_rate))
        recent_signal = signal_array[-recent_samples:]
        recent_time = time_array[-recent_samples:]
        
        # Detect QRS peaks
        peaks, _ = self.qrs_detector.detect_qrs(recent_signal)
        
        if len(peaks) > 0:
            # Convert peak indices to timestamps
            current_time = time_array[-1]
            for peak_idx in peaks:
                peak_time = recent_time[peak_idx]
                
                # Only process new peaks
                if len(self.qrs_times) == 0 or peak_time > self.qrs_times[-1]:
                    self.qrs_positions.append(len(signal_array) - recent_samples + peak_idx)
                    self.qrs_times.append(peak_time)
                    
                    # Extract QRS template
                    self._extract_qrs_template(signal_array, len(signal_array) - recent_samples + peak_idx)
        
        # Calculate heart rate and HRV
        self._calculate_heart_metrics()
        
        # Analyze QRS template similarity
        self._analyze_template_similarity()
    
    def _extract_qrs_template(self, signal_array, peak_position):
        """Extract QRS complex template around detected peak"""
        half_window = self.template_window_samples // 2
        start_idx = max(0, peak_position - half_window)
        end_idx = min(len(signal_array), peak_position + half_window)
        
        if end_idx - start_idx >= self.template_window_samples:
            template = signal_array[start_idx:end_idx]
            
            # Normalize template
            template_std = np.std(template)
            if template_std > 1e-8:  # Only normalize if there's variation
                template = (template - np.mean(template)) / template_std
                self.qrs_templates.append(template)
                print(f"DEBUG: Extracted QRS template #{len(self.qrs_templates)}, std={template_std:.3f}")
            else:
                print(f"DEBUG: Skipped flat template (std={template_std:.3f})")
    
    def _calculate_heart_metrics(self):
        """Calculate heart rate and heart rate variability"""
        if len(self.qrs_times) < 2:
            return
            
        # Calculate R-R intervals
        rr_intervals = []
        for i in range(1, len(self.qrs_times)):
            rr_interval = self.qrs_times[i] - self.qrs_times[i-1]
            if 0.3 < rr_interval < 2.0:  # Valid heart rate range (30-200 BPM)
                rr_intervals.append(rr_interval)
        
        if len(rr_intervals) > 0:
            # Heart rate (BPM)
            mean_rr = np.mean(rr_intervals)
            self.heart_rate = 60.0 / mean_rr
            
            # Heart rate variability (RMSSD)
            if len(rr_intervals) > 1:
                rr_diff = np.diff(rr_intervals)
                self.hrv = np.sqrt(np.mean(rr_diff ** 2)) * 1000  # in milliseconds
    
    def _analyze_template_similarity(self):
        """Analyze QRS template similarity using cross-correlation"""
        if len(self.qrs_templates) < 2:
            self.template_correlation = 0
            return
        
        # Create or update master template
        if self.master_template is None and len(self.qrs_templates) >= 3:
            # Use median template as master
            templates_array = np.array(list(self.qrs_templates))
            self.master_template = np.median(templates_array, axis=0)
        
        if self.master_template is not None:
            # Calculate correlation with recent templates
            recent_templates = list(self.qrs_templates)[-5:]  # Last 5 templates
            correlations = []
            
            for template in recent_templates:
                if len(template) == len(self.master_template):
                    correlation = np.corrcoef(template, self.master_template)[0, 1]
                    if not np.isnan(correlation):
                        correlations.append(correlation)
            
            if correlations:
                self.template_correlation = np.mean(correlations)
            else:
                self.template_correlation = 0
    
    def _calculate_quality_score(self):
        """Calculate overall signal quality score (0-100)"""
        # Weighted combination of quality metrics
        snr_score = min(100, max(0, (self.snr + 10) * 5))  # SNR component
        template_score = max(0, self.template_correlation * 100)  # Template similarity
        
        # Heart rate reasonableness (50-150 BPM is good)
        hr_score = 0
        if 50 <= self.heart_rate <= 150:
            hr_score = 100
        elif 40 <= self.heart_rate <= 180:
            hr_score = 50
        
        # Weighted average
        weights = [0.4, 0.4, 0.2]  # SNR, template, HR
        scores = [snr_score, template_score, hr_score]
        
        self.quality_score = sum(w * s for w, s in zip(weights, scores))
    
    def get_quality_metrics(self):
        """Get current quality metrics"""
        with self.lock:
            return {
                'snr': self.snr,
                'template_correlation': self.template_correlation,
                'quality_score': self.quality_score,
                'heart_rate': self.heart_rate,
                'hrv': self.hrv,
                'qrs_count': len(self.qrs_templates),
                'master_template': self.master_template.copy() if self.master_template is not None else None,
                'recent_templates': list(self.qrs_templates)[-3:] if len(self.qrs_templates) >= 3 else []
            }
    
    def get_quality_string(self):
        """Get formatted quality assessment string"""
        metrics = self.get_quality_metrics()
        
        quality_level = "Poor"
        if metrics['quality_score'] > 80:
            quality_level = "Excellent"
        elif metrics['quality_score'] > 60:
            quality_level = "Good"
        elif metrics['quality_score'] > 40:
            quality_level = "Fair"
        
        return (f"Quality: {quality_level} ({metrics['quality_score']:.1f}/100) | "
                f"SNR: {metrics['snr']:.1f}dB | "
                f"Template Corr: {metrics['template_correlation']:.3f} | "
                f"HR: {metrics['heart_rate']:.1f}bpm | "
                f"HRV: {metrics['hrv']:.1f}ms")


class QRSVisualization:
    """Visualization tools for QRS analysis"""
    
    @staticmethod
    def plot_qrs_overlay(templates, master_template=None):
        """Create overlay plot of QRS templates"""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot individual templates
        for i, template in enumerate(templates):
            ax.plot(template, alpha=0.3, color='blue', linewidth=1)
        
        # Plot master template if available
        if master_template is not None:
            ax.plot(master_template, color='red', linewidth=3, label='Master Template')
            ax.legend()
        
        ax.set_title('QRS Complex Overlay')
        ax.set_xlabel('Samples')
        ax.set_ylabel('Amplitude (normalized)')
        ax.grid(True, alpha=0.3)
        
        return fig, ax
    
    @staticmethod
    def create_quality_dashboard():
        """Create a quality metrics dashboard"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # SNR plot
        axes[0, 0].set_title('Signal-to-Noise Ratio')
        axes[0, 0].set_ylabel('SNR (dB)')
        
        # Template correlation plot
        axes[0, 1].set_title('QRS Template Correlation')
        axes[0, 1].set_ylabel('Correlation Coefficient')
        
        # Heart rate plot
        axes[1, 0].set_title('Heart Rate')
        axes[1, 0].set_ylabel('BPM')
        
        # Quality score plot
        axes[1, 1].set_title('Overall Quality Score')
        axes[1, 1].set_ylabel('Quality (0-100)')
        
        plt.tight_layout()
        return fig, axes