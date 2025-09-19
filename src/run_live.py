import cv2
import mediapipe as mp
import numpy as np
import torch
from collections import deque
import time
import argparse
import csv
from datetime import datetime
import glob
import asyncio
import websockets
import os
import http.server
import socketserver
import threading
from functools import partial
import json
from .models import get_model
from .preprocessing import preprocess_sequence
from . import config as cfg
from .scaler import StandardScaler
from .spotter_data_loader import SPOTTER_WINDOW_SIZE
from .head_pose_estimation import HeadPoseEstimator
from .lightning_module import GestureLightningModule, SpotterLightningModule

# --- WebSocket Server State ---
CONNECTED_CLIENTS = set()

async def register_client(websocket, controller):
    """Adds a new client and sends them the initial system state."""
    CONNECTED_CLIENTS.add(websocket)
    print(f"New client connected. Total clients: {len(CONNECTED_CLIENTS)}")
    # Send initial lock state
    initial_state_event = "system_locked" if controller.is_locked else "system_unlocked"
    await websocket.send(json.dumps({"event": initial_state_event}))
    try:
        await websocket.wait_closed()
    finally:
        CONNECTED_CLIENTS.remove(websocket)
        print(f"Client disconnected. Total clients: {len(CONNECTED_CLIENTS)}")

async def broadcast_message(message):
    """Sends a JSON message to all connected WebSocket clients."""
    if CONNECTED_CLIENTS:
        # Create a list of tasks to send messages concurrently
        json_message = json.dumps(message)
        tasks = [client.send(json_message) for client in CONNECTED_CLIENTS]
        await asyncio.gather(*tasks, return_exceptions=True)

# --- HTTP Server for Slideshow ---
class SlideshowHttpRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Custom handler to serve slideshow.html as the default file."""
    def do_GET(self):
        if self.path == '/':
            self.path = '/slideshow.html'
        return http.server.SimpleHTTPRequestHandler.do_GET(self)

def run_http_server():
    """Runs a simple HTTP server to serve the slideshow files."""
    PORT = 8000
    # Define project root as the directory containing the 'src' folder
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    web_dir = os.path.join(project_root, 'slideshow', 'slideshow')
    
    # Create a handler that is bound to the specific directory
    handler = partial(SlideshowHttpRequestHandler, directory=web_dir)
    
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), handler) as httpd:
        print(f"HTTP server started at http://localhost:{PORT}")
        print(f"Serving files from: {web_dir}")
        httpd.serve_forever()

class LiveGestureController:
    """
    Manages the real-time gesture recognition pipeline and WebSocket communication.
    """
    def __init__(self, model_path: str, mode='gru_only', debug=False):
        self.model_path = model_path
        self.mode = mode
        self.debug = debug
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        ordered_keypoints_with_dupes = ['_'.join(f.split('_')[:-1]) for f in cfg.FEATURES_TO_USE if f.endswith(('_x', '_y', '_z'))]
        self.ordered_keypoints = list(dict.fromkeys(ordered_keypoints_with_dupes))
        self.mediapipe_name_map = {"left_mouth": "MOUTH_LEFT", "right_mouth": "MOUTH_RIGHT"}
        self.landmark_map = {}
        for name in self.ordered_keypoints:
            mediapipe_name = self.mediapipe_name_map.get(name, name.upper())
            try:
                self.landmark_map[name] = self.mp_pose.PoseLandmark[mediapipe_name]
            except KeyError:
                print(f"FATAL: Keypoint '{name}' not found. Check config.py.")
                exit()

        # --- State Management ---
        self.is_locked = True
        self.waiting_for_idle = False
        self.last_gesture_time = 0
        self.toggle_lock_gesture = cfg.TOGGLE_LOCK_GESTURE

        # --- Attention Tracking State ---
        self.attentive_start_time = None
        self.user_is_being_watched = False
        self.user_lost_time = None
        self.last_activity_time = time.time()
        self.tooltip_show_time = None # Tracks when the tooltip was shown

        # --- Buffers ---
        self.spotter_raw_kps_buffer = deque(maxlen=SPOTTER_WINDOW_SIZE)
        self.spotter_pred_buffer = deque(maxlen=cfg.SPOTTER_BUFFER_SIZE)
        self.is_motion_active = False
        self.main_model_raw_kps_buffer = deque(maxlen=cfg.MAX_SEQ_LENGTH)
        self.main_model_pred_buffer = deque(maxlen=cfg.PREDICTION_SMOOTHING_BUFFER_SIZE)

        # --- UI/Feedback ---
        self.feedback_duration = 0.2
        self.feedback_end_time = 0
        self.colors = {"locked": (0, 0, 255), "unlocked": (0, 255, 0), "feedback": (0, 255, 255), "text": (255, 255, 255)}
        
        self.scaler = None
        self.head_pose_estimator = None
        if cfg.ENABLE_HEAD_POSE_DETECTION:
            self.head_pose_estimator = HeadPoseEstimator(debug=self.debug)

        self._load_main_model()
        if self.mode == 'spotter': self._load_spotter_model()
        self._load_scaler_if_needed()
        self._define_gesture_handlers()

        if self.debug: self._setup_debug()

    def _setup_debug(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.debug_filename = f"debug_predictions_{timestamp}.csv"
        self.debug_file = open(self.debug_filename, 'w', newline='')
        self.csv_writer = csv.writer(self.debug_file)
        self.csv_writer.writerow(['timestamp'] + list(GESTURE_LABELS.keys()))
        self.last_debug_time = 0
        print(f"DEBUG MODE: Logging predictions to {self.debug_filename}")

    def _load_main_model(self):
        print("Loading main gesture recognition model...")
        try:
            self.main_model = GestureLightningModule.load_from_checkpoint(self.model_path, map_location=cfg.DEVICE).model
            self.main_model.to(cfg.DEVICE).eval()
            print(f"Successfully loaded main model from: {self.model_path}")
        except FileNotFoundError:
            print(f"FATAL: Main model not found at {self.model_path}."); exit()

    def _load_spotter_model(self):
        print("Loading gesture spotter model...")
        try:
            self.spotter_model = SpotterLightningModule.load_from_checkpoint(cfg.SPOTTER_MODEL_PATH, map_location=cfg.DEVICE).model
            self.spotter_model.to(cfg.DEVICE).eval()
            print(f"Successfully loaded spotter model from: {SPOTTER_MODEL_PATH}")
        except FileNotFoundError:
            print(f"FATAL: Spotter model not found at {SPOTTER_MODEL_PATH}."); exit()

    def _load_scaler_if_needed(self):
        if cfg.PREPROCESSING_STRATEGY == 'standardize':
            if not os.path.exists(cfg.SCALER_PATH):
                print(f"FATAL: Scaler file not found at {cfg.SCALER_PATH}."); exit()
            self.scaler = StandardScaler().load(cfg.SCALER_PATH)
            print(f"Loaded scaler from {cfg.SCALER_PATH}.")

    def _define_gesture_handlers(self):
        """Maps gesture names to WebSocket messages based on the config."""
        self.gesture_handlers = {
            gesture: (lambda event=event: asyncio.create_task(broadcast_message({"event": event})))
            for gesture, event in cfg.GESTURE_ACTIONS.items()
        }

    def _trigger_handler(self, gesture_name):
        """Invokes a gesture handler and manages cooldown."""
        current_time = time.time()
        # Use gesture-specific cooldown period from config
        cooldown = cfg.GESTURE_COOLDOWN_PERIODS.get(gesture_name, 1.0)
        if current_time - self.last_gesture_time > cooldown:
            print(f"Action: Triggering {gesture_name}")
            if gesture_name in self.gesture_handlers:
                self.gesture_handlers[gesture_name]()
            self.last_gesture_time = current_time
            self.feedback_end_time = current_time + self.feedback_duration
            self.main_model_pred_buffer.clear()
            # Any action should hide the tooltip
            if self.tooltip_show_time:
                asyncio.create_task(broadcast_message({"event": "hide_idle_tooltip"}))
                self.tooltip_show_time = None

    def _handle_debug_output(self, probs):
        current_time = time.time()
        if current_time - self.last_debug_time > 0.5:
            self.csv_writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")] + probs.cpu().numpy().tolist())
            self.last_debug_time = current_time

    def _update_ui(self, frame, gesture_name, confidence, all_probs=None, is_attentive=None):
        if time.time() < self.feedback_end_time:
            cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), self.colors["feedback"], 10)
        
        if is_attentive is not None:
            attentive_text = "ATTENTIVE" if is_attentive else "LOOKING AWAY"
            attentive_color = (0, 255, 0) if is_attentive else (0, 0, 255)
            cv2.putText(frame, attentive_text, (frame.shape[1] - 400, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, attentive_color, 2)

        status_color = self.colors["locked"] if self.is_locked else self.colors["unlocked"]
        status_text = "LOCKED" if self.is_locked else "ACTIVE"
        if self.waiting_for_idle: status_text, status_color = "RETURN TO IDLE", (0, 165, 255)

        cv2.circle(frame, (40, 40), 25, status_color, -1)
        cv2.putText(frame, status_text, (80, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, self.colors["text"], 2)

        if self.mode == 'spotter':
            spotter_text = "MOTION" if self.is_motion_active else "NO MOTION"
            spotter_color = (0, 255, 0) if self.is_motion_active else (0, 0, 255)
            cv2.putText(frame, f"Spotter: {spotter_text}", (frame.shape[1] - 250, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, spotter_color, 2)

        if gesture_name:
            cv2.putText(frame, f"{gesture_name.upper()} ({confidence:.2f})", (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.colors["text"], 2)

        if self.debug and all_probs is not None:
            for i, (label, prob) in enumerate(zip(GESTURE_LABELS.keys(), all_probs)):
                color = (0, 255, 0) if prob == torch.max(all_probs) else (0, 255, 255)
                cv2.putText(frame, f"{label}: {prob:.3f}", (frame.shape[1] - 220, 100 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    async def run_gesture_recognition(self):
        """Main async loop for camera capture, processing, and interaction."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return

        with self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while True:
                ret, frame = cap.read()
                if not ret: break

                is_attentive = None
                if self.head_pose_estimator:
                    is_attentive, frame, _ = self.head_pose_estimator.process_frame(frame)

                    # --- New Attention Logic ---
                    current_time = time.time()
                    if is_attentive:
                        self.user_lost_time = None # Reset timer when user looks back

                        # --- Idle Tooltip Logic ---
                        # Check if we should SHOW the tooltip
                        if not self.is_locked and self.tooltip_show_time is None and (current_time - self.last_activity_time > cfg.IDLE_TOOLTIP_THRESHOLD):
                            print("EVENT: User is idle, showing tooltip.")
                            self.tooltip_show_time = current_time
                            asyncio.create_task(broadcast_message({"event": "show_idle_tooltip"}))
                        
                        # Check if we should HIDE the tooltip
                        if self.tooltip_show_time and (current_time - self.tooltip_show_time > cfg.IDLE_TOOLTIP_DURATION):
                            print("EVENT: Tooltip timed out, hiding.")
                            self.tooltip_show_time = None
                            self.last_activity_time = current_time # Reset idle timer to prevent immediate re-showing
                            asyncio.create_task(broadcast_message({"event": "hide_idle_tooltip"}))

                        if self.attentive_start_time is None:
                            self.attentive_start_time = current_time
                        elif not self.user_is_being_watched and (current_time - self.attentive_start_time > cfg.ATTENTION_DETECT_THRESHOLD):
                            self.user_is_being_watched = True
                            asyncio.create_task(broadcast_message({"event": "user_detected"}))
                            print("EVENT: User detected, sending message.")
                    else: # User is looking away
                        self.attentive_start_time = None
                        # If user was being watched and is now looking away, start a timer.
                        if self.user_is_being_watched:
                            if self.user_lost_time is None:
                                self.user_lost_time = current_time
                            # If the timer exceeds the timeout, reset the system.
                            elif (current_time - self.user_lost_time > cfg.DISENGAGEMENT_TIMEOUT):
                                print("EVENT: User disengaged. Resetting system.")
                                self.is_locked = True
                                self.user_is_being_watched = False
                                self.user_lost_time = None
                                asyncio.create_task(broadcast_message({"event": "system_reset"}))
                else:
                    frame = cv2.flip(frame, 1)

                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(img_rgb)

                gesture_name, confidence, all_probs = None, 0.0, None

                if results.pose_landmarks:
                    self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
                    
                    raw_kps_flat = np.array([c for kp in self.ordered_keypoints for c in [results.pose_landmarks.landmark[self.landmark_map[kp]].x, results.pose_landmarks.landmark[self.landmark_map[kp]].y, results.pose_landmarks.landmark[self.landmark_map[kp]].z]], dtype=np.float32)
                    
                    # Using gru_only pipeline logic directly for simplicity
                    self.main_model_raw_kps_buffer.append(raw_kps_flat)
                    if len(self.main_model_raw_kps_buffer) == cfg.MAX_SEQ_LENGTH:
                        final_features = preprocess_sequence(np.array(self.main_model_raw_kps_buffer), scaler=self.scaler)
                        input_tensor = torch.tensor(final_features, dtype=torch.float32).unsqueeze(0).to(cfg.DEVICE)

                        with torch.no_grad():
                            out = self.main_model(input_tensor)
                            probs = torch.softmax(out, dim=1).squeeze()
                            all_probs = probs
                            if self.debug: self._handle_debug_output(probs)
                            self.main_model_pred_buffer.append(list(cfg.GESTURE_LABELS.keys())[torch.argmax(probs).item()])

                        if len(self.main_model_pred_buffer) == cfg.PREDICTION_SMOOTHING_BUFFER_SIZE:
                            most_common = max(set(self.main_model_pred_buffer), key=self.main_model_pred_buffer.count)
                            confidence = self.main_model_pred_buffer.count(most_common)
                            
                            # Use gesture-specific confidence threshold from config
                            required_confidence = cfg.GESTURE_CONFIDENCE_THRESHOLDS.get(most_common, 10)
                            if confidence >= required_confidence:
                                gesture_name = most_common
                                if self.waiting_for_idle:
                                    if gesture_name == 'idle':
                                        self.waiting_for_idle = False
                                        self.main_model_pred_buffer.clear()
                                else:
                                    if gesture_name == 'idle':
                                        self.main_model_pred_buffer.clear()
                                    elif gesture_name == self.toggle_lock_gesture:
                                        self.is_locked = not self.is_locked # Toggle the lock state
                                        self.waiting_for_idle = True
                                        self.main_model_pred_buffer.clear()
                                        self.last_activity_time = time.time() # Reset activity timer
                                        if self.tooltip_show_time:
                                            asyncio.create_task(broadcast_message({"event": "hide_idle_tooltip"}))
                                            self.tooltip_show_time = None
                                        lock_event = "system_locked" if self.is_locked else "system_unlocked"
                                        asyncio.create_task(broadcast_message({"event": lock_event}))
                                    elif gesture_name == cfg.RESET_GESTURE:
                                        print("EVENT: Manual reset triggered.")
                                        self.is_locked = True
                                        self.waiting_for_idle = True
                                        self.main_model_pred_buffer.clear()
                                        asyncio.create_task(broadcast_message({"event": "system_reset"}))
                                    elif not self.is_locked and gesture_name in self.gesture_handlers:
                                        self._trigger_handler(gesture_name)
                                        self.waiting_for_idle = True
                                        self.last_activity_time = time.time() # Reset activity timer
                                        if self.tooltip_show_time:
                                            asyncio.create_task(broadcast_message({"event": "hide_idle_tooltip"}))
                                            self.tooltip_show_time = None
                            else:
                                self.main_model_pred_buffer.popleft()

                self._update_ui(frame, gesture_name, confidence, all_probs, is_attentive)
                cv2.imshow('Gesture Presentation Control', frame)

                if cv2.waitKey(5) & 0xFF == 27: break
                await asyncio.sleep(0.001) # Yield control to the event loop

        if self.debug and hasattr(self, 'debug_file'): self.debug_file.close()
        if self.head_pose_estimator: self.head_pose_estimator.close()
        cap.release()
        cv2.destroyAllWindows()

def find_latest_model_path(model_dir=cfg.MODEL_DIR):
    search_path = os.path.join(model_dir, "gru_main_model-*.ckpt")
    model_files = glob.glob(search_path)
    if not model_files: return None
    return max(model_files, key=os.path.getmtime)

async def main():
    """Main async function to set up and run all components."""
    latest_model_path = find_latest_model_path()
    if not latest_model_path:
        print(f"FATAL: No 'gru_main_model-*.ckpt' files found in '{cfg.MODEL_DIR}'."); exit()
    print(f"Found latest model: {os.path.basename(latest_model_path)}")

    parser = argparse.ArgumentParser(description="Run the live gesture recognition system with WebSocket server.")
    parser.add_argument('--model_path', type=str, default=latest_model_path, help="Path to the trained model checkpoint.")
    parser.add_argument('--mode', type=str, default='gru_only', choices=['spotter', 'gru_only'], help="Recognition mode.")
    parser.add_argument('--debug', action='store_true', help="Enable debug mode.")
    args = parser.parse_args()

    # Start the HTTP server in a separate thread
    http_thread = threading.Thread(target=run_http_server, daemon=True)
    http_thread.start()

    # Initialize the gesture controller BEFORE starting the WebSocket server.
    # This ensures that any slow, blocking operations (like loading the model)
    # are completed before we start accepting connections, preventing handshake failures.
    controller = LiveGestureController(model_path=args.model_path, mode=args.mode, debug=args.debug)

    # Create a partial function to pass the controller instance to the registration handler
    register_handler = partial(register_client, controller=controller)

    # Start the WebSocket server
    websocket_server = await websockets.serve(register_handler, "localhost", 8765)
    print("WebSocket server started at ws://localhost:8765")
    
    # Run gesture recognition loop
    await controller.run_gesture_recognition()

    # Clean up
    websocket_server.close()
    await websocket_server.wait_closed()

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nSession ended by user.")
