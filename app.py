from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
from collections import deque
import time
import json
import threading
import base64

app = Flask(__name__)

class CarrotRubbingDetector:
    def __init__(self, buffer_seconds=2, fps=30, roi_mode='auto', update_every=3):
        self.buffer_seconds = buffer_seconds
        self.fps = fps
        self.buffer_size = buffer_seconds * fps
        self.roi_mode = roi_mode
        self.update_every = update_every
        
        self.motion_history = deque(maxlen=self.buffer_size)
        self.frequency_history = deque(maxlen=10)
        self.last_frequency = 0.0
        self.last_confidence = 0.0
        self.last_vertical_motion = 0.0
        
        self.flow_params = dict(
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        
        self.roi = None
        self.roi_selected = False
        self.prev_gray = None
        self.frame_count = 0
        
        # Timer-related data
        self.timer_active = False
        self.timer_start_time = None
        self.timer_duration = 0.0
        self.timer_motion_log = []  # Log of (timestamp, vertical_motion) during timer
        
    def setup_roi(self, frame):
        """Setup ROI based on mode"""
        h, w = frame.shape[:2]
        
        if self.roi_mode == 'auto':
            self.roi = (0, 0, w, h)
            self.roi_selected = True
        elif self.roi_mode == 'center':
            margin_w, margin_h = int(w * 0.2), int(h * 0.2)
            self.roi = (margin_w, margin_h, w - 2*margin_w, h - 2*margin_h)
            self.roi_selected = True
    
    def extract_vertical_motion(self, flow, roi):
        """Extract dominant vertical motion from optical flow in ROI"""
        x, y, w, h = roi
        
        flow_roi = flow[y:y+h, x:x+w]
        flow_x = flow_roi[:, :, 0]
        flow_y = flow_roi[:, :, 1]
        
        magnitude = np.sqrt(flow_x**2 + flow_y**2)
        weights = magnitude / (np.sum(magnitude) + 1e-6)
        vertical_motion = np.sum(flow_y * weights)
        
        return vertical_motion, magnitude
    
    def calculate_frequency_fft(self):
        """Calculate rubbing frequency using FFT"""
        min_samples = max(self.fps, 20)
        
        if len(self.motion_history) < min_samples:
            return self.last_frequency, self.last_confidence
        
        signal = np.array(self.motion_history)
        signal = signal - np.mean(signal)
        
        window = np.hamming(len(signal))
        signal_windowed = signal * window
        
        fft_vals = np.fft.fft(signal_windowed)
        fft_freq = np.fft.fftfreq(len(signal), 1/self.fps)
        
        positive_freq_idx = fft_freq > 0
        fft_magnitude = np.abs(fft_vals[positive_freq_idx])
        fft_freq_positive = fft_freq[positive_freq_idx]
        
        valid_range = (fft_freq_positive > 0.5) & (fft_freq_positive < 10)
        
        if np.any(valid_range):
            valid_magnitudes = fft_magnitude[valid_range]
            valid_frequencies = fft_freq_positive[valid_range]
            
            peak_idx = np.argmax(valid_magnitudes)
            dominant_frequency = valid_frequencies[peak_idx]
            confidence = valid_magnitudes[peak_idx] / (np.sum(fft_magnitude) + 1e-6)
            
            alpha = 0.3
            if self.last_frequency > 0:
                dominant_frequency = alpha * dominant_frequency + (1 - alpha) * self.last_frequency
                confidence = alpha * confidence + (1 - alpha) * self.last_confidence
            
            self.last_frequency = dominant_frequency
            self.last_confidence = confidence
            
            return dominant_frequency, confidence
        
        return self.last_frequency, self.last_confidence
    
    def process_frame(self, frame, show_vectors=True):
        """Process a single frame and update motion history"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.roi is None:
            self.setup_roi(frame)
        
        vis = frame.copy()
        vertical_motion = 0.0
        frequency = 0.0
        confidence = 0.0
        
        if self.prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, gray, None, **self.flow_params
            )
            
            vertical_motion, magnitude = self.extract_vertical_motion(flow, self.roi)
            
            self.motion_history.append(vertical_motion)
            self.last_vertical_motion = vertical_motion
            
            # Log motion during timer
            if self.timer_active:
                current_time = time.time()
                elapsed = current_time - self.timer_start_time if self.timer_start_time else 0.0
                self.timer_motion_log.append((elapsed, float(vertical_motion)))
            
            if self.frame_count % self.update_every == 0:
                frequency, confidence = self.calculate_frequency_fft()
            else:
                frequency, confidence = self.last_frequency, self.last_confidence
            
            vis = self.visualize(frame, flow, vertical_motion, frequency, confidence, magnitude, show_vectors)
            
            self.frame_count += 1
        
        self.prev_gray = gray
        return vis, vertical_motion, frequency, confidence
    
    def visualize(self, frame, flow, vertical_motion, frequency, confidence, magnitude, show_vectors=True):
        """Create visualization of the detection"""
        vis = frame.copy()
        
        # Draw ROI
        x, y, w, h = self.roi
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Draw optical flow vectors if enabled
        if show_vectors:
            step = 20
            for y_pos in range(y, y+h, step):
                for x_pos in range(x, x+w, step):
                    if y_pos < flow.shape[0] and x_pos < flow.shape[1]:
                        fx, fy = flow[y_pos, x_pos]
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
        
        # Timer display
        if self.timer_active and self.timer_start_time:
            elapsed = time.time() - self.timer_start_time
            timer_text = f"Timer: {elapsed:.2f}s"
            cv2.putText(vis, timer_text, 
                       (10, info_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw motion history graph
        if len(self.motion_history) > 1:
            graph_h = 100
            graph_w = 300
            graph_x, graph_y = vis.shape[1] - graph_w - 10, 10
            
            cv2.rectangle(vis, (graph_x, graph_y), 
                         (graph_x + graph_w, graph_y + graph_h), (0, 0, 0), -1)
            cv2.rectangle(vis, (graph_x, graph_y), 
                         (graph_x + graph_w, graph_y + graph_h), (255, 255, 255), 1)
            
            history_array = np.array(list(self.motion_history))
            if len(history_array) > 0:
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
    
    def start_timer(self):
        """Start the timer"""
        if not self.timer_active:
            self.timer_active = True
            self.timer_start_time = time.time()
            self.timer_motion_log = []
            print(f"[DEBUG] Timer started at {self.timer_start_time}")
            return True
        return False
    
    def stop_timer(self):
        """Stop the timer and return analytics"""
        if self.timer_active:
            self.timer_active = False
            if self.timer_start_time:
                self.timer_duration = time.time() - self.timer_start_time
            else:
                self.timer_duration = 0.0
            
            # Calculate analytics
            analytics = self.get_analytics()
            print(f"[DEBUG] Timer stopped. Duration: {self.timer_duration:.2f}s, Log entries: {len(self.timer_motion_log)}")
            return analytics
        return None
    
    def get_analytics(self):
        """Get analytics from timer motion log"""
        if not self.timer_motion_log:
            return {
                'duration': self.timer_duration,
                'average_motion': 0.0,
                'std_dev': 0.0,
                'motion_over_time': []
            }
        
        motions = [motion for _, motion in self.timer_motion_log]
        
        if len(motions) == 0:
            return {
                'duration': self.timer_duration,
                'average_motion': 0.0,
                'std_dev': 0.0,
                'motion_over_time': []
            }
        
        avg_motion = np.mean(motions)
        std_dev = np.std(motions)
        
        return {
            'duration': self.timer_duration,
            'average_motion': float(avg_motion),
            'std_dev': float(std_dev),
            'motion_over_time': self.timer_motion_log  # List of (time, motion) tuples
        }

# Global detector instance
detector = CarrotRubbingDetector(
    buffer_seconds=2,
    fps=30,
    roi_mode='center',
    update_every=3
)

# Global camera instance
camera = None
camera_lock = threading.Lock()

def get_camera():
    """Get or create camera instance"""
    global camera
    with camera_lock:
        if camera is None or not camera.isOpened():
            camera = cv2.VideoCapture(0)
            if not camera.isOpened():
                print("[DEBUG] Error: Could not open camera")
                return None
            print("[DEBUG] Camera opened successfully")
        return camera

def generate_frames(show_vectors=True):
    """Generate video frames"""
    cap = get_camera()
    if cap is None:
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[DEBUG] Failed to read frame")
            break
        
        vis_frame, vertical_motion, frequency, confidence = detector.process_frame(frame, show_vectors)
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', vis_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            continue
        
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    show_vectors = request.args.get('vectors', 'true').lower() == 'true'
    return Response(generate_frames(show_vectors),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get current stats"""
    return jsonify({
        'frequency': float(detector.last_frequency),
        'confidence': float(detector.last_confidence),
        'vertical_motion': float(detector.last_vertical_motion),
        'timer_active': detector.timer_active,
        'timer_duration': detector.timer_duration if detector.timer_active and detector.timer_start_time else 0.0
    })

@app.route('/api/timer/start', methods=['POST'])
def start_timer():
    """Start the timer"""
    success = detector.start_timer()
    return jsonify({'success': success, 'message': 'Timer started' if success else 'Timer already active'})

@app.route('/api/timer/stop', methods=['POST'])
def stop_timer():
    """Stop the timer and return analytics"""
    analytics = detector.stop_timer()
    if analytics:
        return jsonify({'success': True, 'analytics': analytics})
    return jsonify({'success': False, 'message': 'Timer not active'})

@app.route('/api/timer/status', methods=['GET'])
def timer_status():
    """Get timer status"""
    elapsed = 0.0
    if detector.timer_active and detector.timer_start_time:
        elapsed = time.time() - detector.timer_start_time
    
    return jsonify({
        'active': detector.timer_active,
        'elapsed': elapsed
    })

if __name__ == '__main__':
    print("[DEBUG] Starting Flask application...")
    print("[DEBUG] Open http://localhost:5000 in your browser")
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)

