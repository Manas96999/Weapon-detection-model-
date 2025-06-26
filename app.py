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
FRAME_TIMEOUT = 5  # seconds
cap = None
output_frame = None
detection_lock = threading.Lock()
current_detections = []
current_port = 5001  # Default port

def init_camera():
    global cap
    while True:
        try:
            if cap is not None:
                cap.release()
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
            print(f"Camera initialization error: {str(e)}")
        print("Retrying in 2 seconds...")
        time.sleep(2)

def detect_objects():
    global output_frame, current_detections
    init_camera()
    while True:
        try:
            ret = False
            start_time = time.time()
            while time.time() - start_time < FRAME_TIMEOUT:
                ret, frame = cap.read()
                if ret:
                    break
                time.sleep(0.1)
            if not ret:
                raise RuntimeError("Frame read timeout")
            
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

            with detection_lock:
                current_detections = detections
                output_frame = frame.copy()
                
        except Exception as e:
            print(f"Error: {str(e)}, reinitializing camera...")
            init_camera()

def generate():
    while True:
        with detection_lock:
            if output_frame is None:
                continue
            frame = output_frame.copy()
        _, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

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

if __name__ == '__main__':
    detection_thread = threading.Thread(target=detect_objects, daemon=True)
    detection_thread.start()
    app.run(port=current_port, threaded=True, use_reloader=False)