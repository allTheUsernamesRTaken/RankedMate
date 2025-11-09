from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
from collections import deque
import time
import json
import threading
import base64
import os
from werkzeug.utils import secure_filename
import sqlite3
from datetime import datetime

app = Flask(__name__)

# Database setup
DB_NAME = 'leaderboard.db'

def init_database():
    """Initialize the leaderboard database"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS scores (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            total_score REAL NOT NULL,
            jerk_score REAL NOT NULL,
            nut_value REAL NOT NULL,
            timestamp TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()
    print("[DEBUG] Database initialized")

# Initialize database on startup
init_database()

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
        self.last_analytics = None  # Store last analytics for score retrieval
        
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
            graph_w = 200  # Reduced from 300 to make it less wide
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
            self.last_analytics = analytics  # Store for later retrieval
            print(f"[DEBUG] Timer stopped. Duration: {self.timer_duration:.2f}s, Log entries: {len(self.timer_motion_log)}")
            return analytics
        return None
    
    def calculate_pattern_consistency(self, timer_motion_log):
        """
        Calculate pattern consistency by measuring regularity of oscillation periods.
        Finds zero-crossings and measures consistency of time between direction changes.
        Lower std dev of periods = more consistent pattern.
        """
        if len(timer_motion_log) < 4:  # Need at least a few points
            return 1.0  # Default to perfect consistency if not enough data
        
        times = np.array([t for t, _ in timer_motion_log])
        motions = np.array([m for _, m in timer_motion_log])
        
        # Find zero-crossings (where motion changes sign)
        # We'll look for sign changes in the motion values
        zero_crossings = []
        for i in range(1, len(motions)):
            if (motions[i-1] >= 0 and motions[i] < 0) or (motions[i-1] < 0 and motions[i] >= 0):
                # Linear interpolation to find exact zero-crossing time
                if motions[i-1] != motions[i]:
                    t_ratio = -motions[i-1] / (motions[i] - motions[i-1])
                    zero_time = times[i-1] + t_ratio * (times[i] - times[i-1])
                    zero_crossings.append(zero_time)
        
        # If we don't have enough zero-crossings, use peak detection instead
        if len(zero_crossings) < 2:
            # Simple peak/valley detection
            # Find local maxima (peaks) and minima (valleys)
            peaks = []
            valleys = []
            window = max(1, len(motions) // 20)  # Minimum distance between peaks
            
            for i in range(window, len(motions) - window):
                # Check for local maximum (peak)
                if motions[i] == np.max(motions[i-window:i+window+1]):
                    peaks.append(i)
                # Check for local minimum (valley)
                elif motions[i] == np.min(motions[i-window:i+window+1]):
                    valleys.append(i)
            
            # Combine and sort extrema
            extrema_indices = sorted(set(peaks + valleys))
            
            if len(extrema_indices) >= 2:
                extrema_times = times[extrema_indices]
                # Calculate periods between extrema
                periods = np.diff(extrema_times)
                if len(periods) > 0:
                    period_std = np.std(periods)
                    period_mean = np.mean(periods)
                    # Coefficient of variation (normalized std dev)
                    if period_mean > 0:
                        cv = period_std / period_mean
                        # Convert to consistency score (lower CV = higher consistency)
                        # Use 1 / (1 + CV) so perfect consistency (CV=0) = 1.0
                        consistency = 1.0 / (1.0 + cv)
                        return float(consistency)
            
            # Fallback: if we can't find pattern, return low consistency
            return 0.1
        
        # Calculate periods between zero-crossings
        periods = np.diff(zero_crossings)
        
        if len(periods) == 0:
            return 0.1
        
        # Calculate coefficient of variation (std dev / mean)
        # This normalizes the std dev by the mean period
        period_mean = np.mean(periods)
        if period_mean > 0:
            period_std = np.std(periods)
            cv = period_std / period_mean
            # Convert to consistency score: lower CV = higher consistency
            # Use 1 / (1 + CV) so perfect consistency (CV=0) = 1.0, higher CV = lower score
            consistency = 1.0 / (1.0 + cv)
            return float(consistency)
        
        return 0.1
    
    def get_analytics(self):
        """Get analytics from timer motion log"""
        if not self.timer_motion_log:
            return {
                'duration': self.timer_duration,
                'average_motion': 0.0,
                'std_dev': 0.0,
                'jerk_score': 0.0,
                'motion_over_time': []
            }
        
        motions = [motion for _, motion in self.timer_motion_log]
        
        if len(motions) == 0:
            return {
                'duration': self.timer_duration,
                'average_motion': 0.0,
                'std_dev': 0.0,
                'jerk_score': 0.0,
                'motion_over_time': []
            }
        
        avg_motion = np.mean(motions)
        
        # Calculate pattern consistency (0.0 to 1.0, higher = more consistent pattern)
        pattern_consistency = self.calculate_pattern_consistency(self.timer_motion_log)
        
        # For display, we'll show pattern consistency as a percentage (0-100%)
        # and also keep a "pattern std dev" metric that's inversely related for backward compatibility
        pattern_std_dev = (1.0 - pattern_consistency) * 10.0  # Scale for display (lower = better)
        
        # Calculate jerk score: duration * abs(avg_motion) * pattern_consistency
        # Higher pattern_consistency = higher score (consistent patterns are rewarded)
        jerk_score = self.timer_duration * abs(avg_motion) * pattern_consistency
        
        return {
            'duration': self.timer_duration,
            'average_motion': float(avg_motion),
            'std_dev': float(pattern_std_dev),  # Now represents pattern inconsistency
            'pattern_consistency': float(pattern_consistency),  # Add this for reference
            'jerk_score': float(jerk_score),
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

# Global nut value storage
nut_value = 0.0  # Stores the current nut number (0-10)

def detect_square_and_calculate_brightness(image_path):
    """
    Detects a black square on white paper and calculates normalized brightness (0-1).
    Based on detect_square.py logic.
    
    Args:
        image_path: Path to the input image
    
    Returns:
        float: Normalized brightness value (0.0 to 1.0), or None if error
    """
    if not os.path.exists(image_path):
        print(f"[DEBUG] Error: Image file '{image_path}' not found.")
        return None
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"[DEBUG] Error: Could not read image '{image_path}'.")
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding to handle varying lighting conditions
    thresh = cv2.adaptiveThreshold(
        blurred, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        11, 
        2
    )
    
    # Find contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("[DEBUG] No contours found in the image.")
        return None
    
    # Find the largest contour (should be the black square)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Approximate the contour to a polygon
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # Create a mask for the detected square region
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [approx], 255)
    
    # Extract the region of interest (ROI) from the original grayscale image
    roi = cv2.bitwise_and(gray, gray, mask=mask)
    
    # Calculate average brightness within the detected square
    pixel_count = np.count_nonzero(mask)
    if pixel_count == 0:
        print("[DEBUG] No pixels found in detected square.")
        return None
    
    total_brightness = np.sum(roi[mask > 0])
    average_brightness = total_brightness / pixel_count
    
    # Normalize to 0-1 range (0 = black, 255 = white, so divide by 255)
    normalized_brightness = average_brightness / 255.0
    
    print(f"[DEBUG] Square Detection Results:")
    print(f"  Average brightness: {average_brightness:.2f}")
    print(f"  Normalized brightness: {normalized_brightness:.4f}")
    print(f"  Pixel count: {pixel_count}")
    
    return normalized_brightness

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

@app.route('/api/nut/upload', methods=['POST'])
def upload_nut_photo():
    """Handle photo upload for nut meter analysis"""
    global nut_value
    
    if 'photo' not in request.files:
        return jsonify({'success': False, 'message': 'No file provided'}), 400
    
    file = request.files['photo']
    
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No file selected'}), 400
    
    # Check if file is an image
    if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        return jsonify({'success': False, 'message': 'Invalid file type. Please upload a JPG or PNG image.'}), 400
    
    # Save uploaded file temporarily
    filename = secure_filename(file.filename)
    upload_path = os.path.join('uploads', filename)
    
    # Create uploads directory if it doesn't exist
    os.makedirs('uploads', exist_ok=True)
    
    file.save(upload_path)
    print(f"[DEBUG] File uploaded: {upload_path}")
    
    try:
        # Analyze the image
        normalized_brightness = detect_square_and_calculate_brightness(upload_path)
        
        if normalized_brightness is None:
            # Clean up uploaded file
            if os.path.exists(upload_path):
                os.remove(upload_path)
            return jsonify({'success': False, 'message': 'Could not detect square in image. Make sure there is a black square on white paper.'}), 400
        
        # Calculate nut number (0-10)
        nut_value = normalized_brightness * 10.0
        
        # Clean up uploaded file
        if os.path.exists(upload_path):
            os.remove(upload_path)
        
        print(f"[DEBUG] Nut value calculated: {nut_value:.2f}")
        
        return jsonify({
            'success': True,
            'nut_value': float(nut_value),
            'normalized_brightness': float(normalized_brightness)
        })
    
    except Exception as e:
        # Clean up uploaded file
        if os.path.exists(upload_path):
            os.remove(upload_path)
        print(f"[DEBUG] Error processing image: {str(e)}")
        return jsonify({'success': False, 'message': f'Error processing image: {str(e)}'}), 500

@app.route('/api/nut/value', methods=['GET'])
def get_nut_value():
    """Get current nut value"""
    global nut_value
    return jsonify({'nut_value': float(nut_value)})

@app.route('/api/scores', methods=['GET'])
def get_scores():
    """Get all scores (jerk score, nut value, total)"""
    global nut_value
    
    # Get jerk score from last analytics if available
    jerk_score = 0.0
    if hasattr(detector, 'last_analytics') and detector.last_analytics is not None:
        jerk_score = detector.last_analytics.get('jerk_score', 0.0)
    
    total_score = jerk_score + nut_value
    
    print(f"[DEBUG] Scores API: jerk_score={jerk_score}, nut_value={nut_value}, total={total_score}")
    
    return jsonify({
        'jerk_score': float(jerk_score),
        'nut_value': float(nut_value),
        'total_score': float(total_score)
    })

@app.route('/api/leaderboard/submit', methods=['POST'])
def submit_score():
    """Submit a score to the leaderboard"""
    global nut_value
    
    data = request.get_json()
    username = data.get('username', 'Daniel Guo')
    
    # Get jerk score from last analytics if available
    jerk_score = 0.0
    if hasattr(detector, 'last_analytics') and detector.last_analytics is not None:
        jerk_score = detector.last_analytics.get('jerk_score', 0.0)
    
    total_score = jerk_score + nut_value
    
    if total_score <= 0:
        return jsonify({'success': False, 'message': 'No score to submit'}), 400
    
    # Save to database
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    timestamp = datetime.now().isoformat()
    c.execute('''
        INSERT INTO scores (username, total_score, jerk_score, nut_value, timestamp)
        VALUES (?, ?, ?, ?, ?)
    ''', (username, total_score, jerk_score, nut_value, timestamp))
    conn.commit()
    conn.close()
    
    print(f"[DEBUG] Score submitted: {username} - Total: {total_score}, Jerk: {jerk_score}, Nut: {nut_value}")
    
    return jsonify({
        'success': True,
        'message': 'Score submitted successfully'
    })

@app.route('/leaderboard')
def leaderboard():
    """Display the leaderboard page"""
    return render_template('leaderboard.html')

@app.route('/api/leaderboard', methods=['GET'])
def get_leaderboard():
    """Get leaderboard data"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    # Get all scores grouped by username
    c.execute('''
        SELECT username, total_score
        FROM scores
        ORDER BY username, timestamp DESC
    ''')
    
    rows = c.fetchall()
    conn.close()
    
    # Group scores by username and calculate statistics
    user_scores = {}
    for username, total_score in rows:
        if username not in user_scores:
            user_scores[username] = []
        user_scores[username].append(float(total_score))
    
    # Calculate statistics for each user
    leaderboard_data = []
    for username, scores in user_scores.items():
        avg_score = np.mean(scores)
        max_score = np.max(scores)
        num_scores = len(scores)
        std_dev = np.std(scores) if num_scores > 1 else 0.0
        
        leaderboard_data.append({
            'username': username,
            'avg_score': float(avg_score),
            'max_score': float(max_score),
            'num_scores': int(num_scores),
            'std_dev': float(std_dev)
        })
    
    # Sort by average score descending
    leaderboard_data.sort(key=lambda x: x['avg_score'], reverse=True)
    
    return jsonify({'leaderboard': leaderboard_data})

if __name__ == '__main__':
    print("[DEBUG] Starting Flask application...")
    print("[DEBUG] Open http://localhost:5000 in your browser")
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)

