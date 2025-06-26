from flask import Flask, Response, jsonify, render_template, request, redirect, url_for
from flask_cors import CORS
import cv2
import numpy as np
import threading
import time
from ultralytics import YOLO

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load models
# Replace 'your_weapon_model.pt' with the actual filename you downloaded
weapon_model = YOLO('best.pt')  # Your downloaded weapon model
object_model = YOLO('yolov8n.pt')  # Default YOLO model for general objects

# Configuration
FRAME_TIMEOUT = 10  # seconds - increased for network streams
cap = None
output_frame = None
detection_lock = threading.Lock()
current_detections = []
current_port = 5001  # Default port

# Stream configuration - modify these as needed
USE_EXTERNAL_STREAM = True  # Set to False to use laptop camera
STREAM_URL = "http://raspberrypi:8080/?action=stream"  # Your MJPEG stream URL

def init_camera():
    global cap
    while True:
        try:
            if cap is not None:
                cap.release()
            
            if USE_EXTERNAL_STREAM:
                print(f"Connecting to external stream: {STREAM_URL}")
                cap = cv2.VideoCapture(STREAM_URL)
                # Set additional properties for network streams
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size to minimize latency
                if not cap.isOpened():
                    print("Could not connect to external stream. Retrying...")
                    time.sleep(5)  # Wait longer for network streams
                    continue
                print("External stream connected successfully.")
            else:
                print("Connecting to laptop camera...")
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    print("Could not open laptop camera. Retrying...")
                    time.sleep(2)
                    continue
                print("Laptop camera initialized successfully.")
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FPS, 30)
            
            return
            
        except Exception as e:
            print(f"Camera/Stream initialization error: {str(e)}")
            if cap is not None:
                cap.release()
                cap = None
        
        print("Retrying in 5 seconds...")
        time.sleep(5)

def detect_objects():
    global output_frame, current_detections
    init_camera()
    
    frame_skip_counter = 0
    consecutive_failures = 0
    
    while True:
        try:
            ret = False
            start_time = time.time()
            
            # Try to read frame with timeout
            while time.time() - start_time < FRAME_TIMEOUT:
                ret, frame = cap.read()
                if ret:
                    break
                time.sleep(0.1)
            
            if not ret:
                consecutive_failures += 1
                print(f"Failed to read frame (attempt {consecutive_failures})")
                
                # If too many consecutive failures, reinitialize
                if consecutive_failures >= 10:
                    print("Too many consecutive failures, reinitializing connection...")
                    raise RuntimeError("Too many frame read failures")
                
                # Skip this iteration and try again
                time.sleep(0.5)
                continue
            
            # Reset failure counter on successful frame read
            consecutive_failures = 0
            
            # Skip frames for performance (optional - adjust as needed)
            frame_skip_counter += 1
            if frame_skip_counter % 2 != 0:  # Process every 2nd frame
                continue
            
            # Run weapon detection with adjustable confidence
            weapon_results = weapon_model(frame, conf=0.3)  # Lower confidence for better detection
            # Comment out the next line if you only want weapon detection
            object_results = object_model(frame, conf=0.5)
            
            detections = []
            weapon_count = 0

            # Process weapon detections
            for result in weapon_results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Get class name from model
                        if hasattr(weapon_model, 'names') and class_id < len(weapon_model.names):
                            class_name = weapon_model.names[class_id]
                        else:
                            # Fallback class mapping - adjust these based on your model
                            weapon_classes = {
                                0: 'Gun',
                                1: 'Knife', 
                                2: 'Pistol',
                                3: 'Rifle',
                                4: 'Weapon'  # Generic weapon class
                            }
                            class_name = weapon_classes.get(class_id, f'Weapon_{class_id}')
                        
                        # Draw bounding box for weapons (red)
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                        label = f'{class_name} {confidence:.2f}'
                        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        
                        detections.append({'class': class_name, 'confidence': float(confidence)})
                        weapon_count += 1

            # Process general object detections (optional)
            for result in object_results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        class_name = object_model.names[class_id]
                        
                        # Draw bounding box for objects (green)
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        label = f'{class_name} {confidence:.2f}'
                        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        detections.append({'class': class_name, 'confidence': float(confidence)})

            # Add stream source indicator
            source_text = "Stream: External" if USE_EXTERNAL_STREAM else "Stream: Laptop Camera"
            cv2.putText(frame, source_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            with detection_lock:
                current_detections = detections
                output_frame = frame.copy()
                
        except Exception as e:
            print(f"Detection error: {str(e)}, reinitializing connection...")
            consecutive_failures = 0  # Reset counter
            init_camera()

def generate():
    while True:
        with detection_lock:
            if output_frame is None:
                # Provide a default frame if no frame is available
                default_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(default_frame, "Connecting to stream...", (50, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                frame = default_frame
            else:
                frame = output_frame.copy()
        
        try:
            _, jpeg = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        except Exception as e:
            print(f"Frame encoding error: {str(e)}")
            time.sleep(0.1)

@app.route('/')
def index():
    return render_template('index.html', current_port=current_port)

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detections')
def get_detections():
    with detection_lock:
        return jsonify(current_detections)

@app.route('/switch_port', methods=['POST'])
def switch_port():
    global current_port
    new_port = request.form.get('port')
    try:
        new_port = int(new_port)
        print(f"Switching to port {new_port}...")
        time.sleep(1)
        return redirect(f'http://127.0.0.1:{new_port}/')
    except ValueError:
        return "Invalid port", 400

@app.route('/switch_source', methods=['POST'])
def switch_source():
    global USE_EXTERNAL_STREAM, cap
    source = request.form.get('source')
    
    if source == 'external':
        USE_EXTERNAL_STREAM = True
    elif source == 'camera':
        USE_EXTERNAL_STREAM = False
    
    # Reinitialize the camera/stream
    if cap is not None:
        cap.release()
        cap = None
    
    return jsonify({'status': 'success', 'source': 'external' if USE_EXTERNAL_STREAM else 'camera'})

@app.route('/update_stream_url', methods=['POST'])
def update_stream_url():
    global STREAM_URL, cap
    new_url = request.form.get('url')
    
    if new_url:
        STREAM_URL = new_url
        # Reinitialize if using external stream
        if USE_EXTERNAL_STREAM and cap is not None:
            cap.release()
            cap = None
        return jsonify({'status': 'success', 'url': STREAM_URL})
    
    return jsonify({'status': 'error', 'message': 'Invalid URL'})

if __name__ == '__main__':
    detection_thread = threading.Thread(target=detect_objects, daemon=True)
    detection_thread.start()
    app.run(port=current_port, threaded=True, use_reloader=False)