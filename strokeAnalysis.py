import cv2
import numpy as np
from collections import deque
import time

class CarrotRubbingDetector:
    def __init__(self, buffer_seconds=2, fps=30, roi_mode='auto', update_every=3):
        """
        Initialize the carrot rubbing frequency detector.
        
        Args:
            buffer_seconds: How many seconds of motion data to analyze (shorter = more responsive)
            fps: Frames per second of the video
            roi_mode: 'auto' (full frame), 'center' (center region), or 'manual' (click to select)
            update_every: Calculate frequency every N frames (lower = more responsive)
        """
        self.buffer_seconds = buffer_seconds
        self.fps = fps
        self.buffer_size = buffer_seconds * fps
        self.roi_mode = roi_mode
        self.update_every = update_every
        
        # Motion history buffer for vertical movement
        self.motion_history = deque(maxlen=self.buffer_size)
        
        # Smoothed frequency tracking
        self.frequency_history = deque(maxlen=10)
        self.last_frequency = 0.0
        self.last_confidence = 0.0
        
        # Optical flow parameters
        self.flow_params = dict(
            pyr_scale=0.5,      # Image scale (<1) to build pyramids
            levels=3,           # Number of pyramid layers
            winsize=15,         # Averaging window size
            iterations=3,       # Iterations at each pyramid level
            poly_n=5,           # Size of pixel neighborhood
            poly_sigma=1.2,     # Gaussian std for smoothing derivatives
            flags=0
        )
        
        # ROI coordinates
        self.roi = None
        self.roi_selected = False
        
        # For visualization
        self.prev_gray = None
        self.frame_count = 0
        
    def select_roi_callback(self, event, x, y, flags, param):
        """Mouse callback for ROI selection"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.roi_start = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.roi_end = (x, y)
            x1, y1 = self.roi_start
            x2, y2 = self.roi_end
            self.roi = (min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
            self.roi_selected = True
    
    def setup_roi(self, frame):
        """Setup ROI based on mode"""
        h, w = frame.shape[:2]
        
        if self.roi_mode == 'auto':
            # Use full frame
            self.roi = (0, 0, w, h)
            self.roi_selected = True
            
        elif self.roi_mode == 'center':
            # Use center 60% of frame
            margin_w, margin_h = int(w * 0.2), int(h * 0.2)
            self.roi = (margin_w, margin_h, w - 2*margin_w, h - 2*margin_h)
            self.roi_selected = True
            
        elif self.roi_mode == 'manual':
            # Let user select ROI
            print("Click and drag to select the region containing hand + carrot")
            cv2.namedWindow('Select ROI')
            cv2.setMouseCallback('Select ROI', self.select_roi_callback)
            
            while not self.roi_selected:
                display = frame.copy()
                cv2.putText(display, "Click and drag to select ROI, then press SPACE", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow('Select ROI', display)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' ') and self.roi_selected:
                    break
            
            cv2.destroyWindow('Select ROI')
    
    def extract_vertical_motion(self, flow, roi):
        """Extract dominant vertical motion from optical flow in ROI"""
        x, y, w, h = roi
        
        # Extract flow in ROI
        flow_roi = flow[y:y+h, x:x+w]
        
        # Separate horizontal and vertical components
        flow_x = flow_roi[:, :, 0]
        flow_y = flow_roi[:, :, 1]
        
        # Calculate motion magnitude
        magnitude = np.sqrt(flow_x**2 + flow_y**2)
        
        # METHOD 1: Mean vertical motion (weighted by magnitude)
        # This gives more weight to pixels with stronger motion
        weights = magnitude / (np.sum(magnitude) + 1e-6)
        vertical_motion = np.sum(flow_y * weights)
        
        # METHOD 2: Median vertical motion (alternative - more robust to outliers)
        # vertical_motion = np.median(flow_y[magnitude > np.percentile(magnitude, 75)])
        
        # METHOD 3: Histogram-based dominant direction
        # Find the most common vertical motion direction
        # hist, bins = np.histogram(flow_y[magnitude > 2], bins=50)
        # vertical_motion = bins[np.argmax(hist)]
        
        return vertical_motion, magnitude
    
    def calculate_frequency_fft(self):
        """Calculate rubbing frequency using FFT with adaptive windowing"""
        min_samples = max(self.fps, 20)  # Reduced minimum required samples
        
        if len(self.motion_history) < min_samples:
            return self.last_frequency, self.last_confidence  # Return last known value
        
        # Convert motion history to numpy array
        signal = np.array(self.motion_history)
        
        # Remove DC component (mean)
        signal = signal - np.mean(signal)
        
        # Apply Hamming window to reduce spectral leakage
        window = np.hamming(len(signal))
        signal_windowed = signal * window
        
        # Compute FFT
        fft_vals = np.fft.fft(signal_windowed)
        fft_freq = np.fft.fftfreq(len(signal), 1/self.fps)
        
        # Only consider positive frequencies
        positive_freq_idx = fft_freq > 0
        fft_magnitude = np.abs(fft_vals[positive_freq_idx])
        fft_freq_positive = fft_freq[positive_freq_idx]
        
        # Find dominant frequency (peak in reasonable range: 0.5-10 Hz)
        valid_range = (fft_freq_positive > 0.5) & (fft_freq_positive < 10)
        
        if np.any(valid_range):
            valid_magnitudes = fft_magnitude[valid_range]
            valid_frequencies = fft_freq_positive[valid_range]
            
            peak_idx = np.argmax(valid_magnitudes)
            dominant_frequency = valid_frequencies[peak_idx]
            confidence = valid_magnitudes[peak_idx] / (np.sum(fft_magnitude) + 1e-6)
            
            # Temporal smoothing with exponential moving average
            alpha = 0.3  # Smoothing factor (higher = more responsive)
            if self.last_frequency > 0:
                dominant_frequency = alpha * dominant_frequency + (1 - alpha) * self.last_frequency
                confidence = alpha * confidence + (1 - alpha) * self.last_confidence
            
            self.last_frequency = dominant_frequency
            self.last_confidence = confidence
            
            return dominant_frequency, confidence
        
        return self.last_frequency, self.last_confidence
    
    def calculate_frequency_autocorrelation(self):
        """Alternative: Calculate frequency using autocorrelation"""
        if len(self.motion_history) < self.buffer_size // 2:
            return 0.0
        
        signal = np.array(self.motion_history)
        signal = signal - np.mean(signal)
        
        # Compute autocorrelation
        autocorr = np.correlate(signal, signal, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Find peaks in autocorrelation (excluding zero lag)
        # Look for first significant peak
        threshold = 0.3 * np.max(autocorr)
        peaks = []
        
        for i in range(int(self.fps * 0.1), len(autocorr)):  # Start from 0.1s lag
            if autocorr[i] > threshold:
                if i == 0 or (autocorr[i] > autocorr[i-1] and 
                             (i == len(autocorr)-1 or autocorr[i] > autocorr[i+1])):
                    peaks.append(i)
                    break
        
        if peaks:
            period_frames = peaks[0]
            frequency = self.fps / period_frames
            return frequency
        
        return 0.0
    
    def process_frame(self, frame):
        """Process a single frame and update motion history"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Setup ROI on first frame
        if self.roi is None:
            self.setup_roi(frame)
        
        # Calculate optical flow
        if self.prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, gray, None, **self.flow_params
            )
            
            # Extract vertical motion in ROI
            vertical_motion, magnitude = self.extract_vertical_motion(flow, self.roi)
            
            # Add to history
            self.motion_history.append(vertical_motion)
            
            # Calculate frequency more frequently for responsiveness
            if self.frame_count % self.update_every == 0:
                frequency, confidence = self.calculate_frequency_fft()
            else:
                frequency, confidence = self.last_frequency, self.last_confidence
            
            # Visualization
            vis_frame = self.visualize(frame, flow, vertical_motion, frequency, confidence, magnitude)
            
            self.frame_count += 1
            self.prev_gray = gray
            
            return vis_frame, frequency, confidence
        
        self.prev_gray = gray
        return frame, 0.0, 0.0
    
    def visualize(self, frame, flow, vertical_motion, frequency, confidence, magnitude):
        """Create visualization of the detection"""
        vis = frame.copy()
        
        # Draw ROI
        x, y, w, h = self.roi
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Draw optical flow vectors (subsample for visibility)
        step = 20
        for y_pos in range(y, y+h, step):
            for x_pos in range(x, x+w, step):
                if y_pos < flow.shape[0] and x_pos < flow.shape[1]:
                    fx, fy = flow[y_pos, x_pos]
                    # Only draw significant motion
                    if np.sqrt(fx**2 + fy**2) > 1:
                        end_x = int(x_pos + fx * 2)
                        end_y = int(y_pos + fy * 2)
                        cv2.arrowedLine(vis, (x_pos, y_pos), (end_x, end_y), 
                                      (0, 255, 255), 1, tipLength=0.3)
        
        # Display information
        info_y = 30
        cv2.putText(vis, f"Frequency: {frequency:.2f} Hz ({frequency*60:.1f} rubs/min)", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(vis, f"Confidence: {confidence:.2f}", 
                   (10, info_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.putText(vis, f"Vertical Motion: {vertical_motion:.2f}", 
                   (10, info_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Draw motion history graph
        if len(self.motion_history) > 1:
            graph_h = 100
            graph_w = 300
            graph_x, graph_y = vis.shape[1] - graph_w - 10, 10
            
            # Create graph background
            cv2.rectangle(vis, (graph_x, graph_y), 
                         (graph_x + graph_w, graph_y + graph_h), (0, 0, 0), -1)
            cv2.rectangle(vis, (graph_x, graph_y), 
                         (graph_x + graph_w, graph_y + graph_h), (255, 255, 255), 1)
            
            # Plot motion history
            history_array = np.array(list(self.motion_history))
            if len(history_array) > 0:
                # Normalize to graph height
                max_val = np.max(np.abs(history_array)) + 1e-6
                normalized = (history_array / max_val) * (graph_h / 2)
                
                points = []
                for i, val in enumerate(normalized):
                    px = graph_x + int((i / len(history_array)) * graph_w)
                    py = graph_y + graph_h // 2 - int(val)
                    points.append([px, py])
                
                if len(points) > 1:
                    points = np.array(points, dtype=np.int32)
                    cv2.polylines(vis, [points], False, (0, 255, 0), 2)
        
        return vis

def main():
    """Main function to run the detector"""
    # Open video capture (0 for webcam, or provide video file path)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open video capture")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = 30  # Default if FPS cannot be determined
    
    print(f"Video FPS: {fps}")
    print("Instructions:")
    print("  - Position hand rubbing carrot in view")
    print("  - Press 'q' to quit")
    print("  - Press 'r' to reset motion history")
    
    # Initialize detector
    # Try 'auto', 'center', or 'manual' for roi_mode
    # Lower buffer_seconds and update_every for faster response
    detector = CarrotRubbingDetector(
        buffer_seconds=2,      # Reduced from 3 for faster updates
        fps=fps, 
        roi_mode='center',
        update_every=3          # Update frequency calculation every 3 frames
    )
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or cannot read frame")
            break
        
        # Process frame
        vis_frame, frequency, confidence = detector.process_frame(frame)
        
        # Display
        cv2.imshow('Carrot Rubbing Detector', vis_frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            detector.motion_history.clear()
            print("Motion history reset")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()