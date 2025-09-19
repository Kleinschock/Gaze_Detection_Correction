import sys
import os
sys.path.append(r'C:\Users\s392804\Documents\test\final_project')
import time
import json
import asyncio
import numpy as np
import cv2
import mediapipe as mp
import yaml
from sanic import Sanic
from sanic.response import html
import pathlib

# Set up project paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

# Configuration paths

SCALER_PATH = os.path.join(PROJECT_ROOT, "DataPreprocessing", "Config", "scaler_params.npz")
PCA_MEAN_PATH = os.path.join(PROJECT_ROOT, "DataPreprocessing", "Config", "pca_mean.npy")
PCA_EIGEN_PATH = os.path.join(PROJECT_ROOT, "DataPreprocessing", "Config", "pca_eigenvectors.npy")
MODEL_PATH = os.path.join(PROJECT_ROOT, "different_models", "differentGradientDescents_next_try_no_L2","momentum" , "model_checkpoint.npz")
'''

SCALER_PATH = os.path.join(PROJECT_ROOT, "DataPreprocessing", "Config", "scaler_params.npz")
PCA_MEAN_PATH = os.path.join(PROJECT_ROOT, "DataPreprocessing", "Config", "pca_mean.npy")
PCA_EIGEN_PATH = os.path.join(PROJECT_ROOT, "DataPreprocessing", "Config", "pca_eigenvectors.npy")
MODEL_PATH = os.path.join(PROJECT_ROOT, "best1", "model_checkpoint.npz")'''


# Import project modules
import DataPreprocessing.Config.GLOBAL_Var as g_var
from data_preprocessing_toolkit.data_preprocessing import DataPreprecessor
from nn_framework.neuralNetwork import NeuralNetwork as nn

# Application settings
DEMO_MODE = False
DATA_COLLECTION_MODE = False
FLIP_IMAGE = True  # Set to True when webcam flips your image

# Initialize Sanic app
app = Sanic("slideshow_server")
slideshow_root_path = pathlib.Path(__file__).parent.joinpath("slideshow")
script_dir = pathlib.Path(__file__).parent
app.static("/static", slideshow_root_path)


# Load keypoint mappings
def load_keypoint_mappings():
    with open("./process_videos/keypoint_mapping.yml") as yaml_file:
        mappings = yaml.safe_load(yaml_file)
        keypoint_names = mappings["face"] + mappings["body"]
    print(f"Loaded {len(keypoint_names)} keypoints from mapping file")
    return keypoint_names


# Initialize data preprocessor
def init_preprocessor():
    dp = DataPreprecessor()

    # Load Scaler
    try:
        dp.data_scaler.load(SCALER_PATH)
        print("Successfully loaded scaler parameters")
    except Exception as e:
        print(f"Error loading scaler: {e}")

    # Load PCA parameters
    n_components = g_var.PCA_COUNT
    dp.data_pca.pca_mean_dir = PCA_MEAN_PATH
    dp.data_pca.pca_eigenvectors_dir = PCA_EIGEN_PATH
    dp.data_pca.number_of_components = n_components
    dp.data_pca.load_pca_parameters(PCA_MEAN_PATH, PCA_EIGEN_PATH)

    # Set window parameters
    dp.window_size = g_var.WINDOW_SIZE
    dp.window_stride = g_var.WINDOW_STRIDE

    print(f"Using {n_components} PCA components")
    print(f"Using window size: {dp.window_size}, stride: {dp.window_stride}")

    return dp


# Load neural network model
def load_model():
    try:
        model = nn.load_model(filepath=MODEL_PATH)
        if model:
            print(f"Successfully loaded neural network model from {MODEL_PATH}")
        else:
            print("Failed to load model - returned None")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


# Set up gesture recognition
def setup_gesture_recognition():
    # Configure gesture mapping
    gesture_map = g_var.gesture_map
    gesture_id_map = {label: i for i, label in gesture_map.items()}
    print(f"Gesture map: {gesture_map}")

    # Initialize MediaPipe utilities
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose

    return gesture_map, gesture_id_map, mp_drawing, mp_drawing_styles, mp_pose


# Extract features from pose landmarks
def extract_features(results, keypoint_names):
    current_features = []

    for feature in g_var.MediaPipe_features:
        parts = feature.rsplit('_', 1)
        if len(parts) == 2:
            joint_name, coordinate = parts

            try:
                # Get the index of the joint from the keypoint names list
                if joint_name in keypoint_names:
                    joint_idx = keypoint_names.index(joint_name)
                    joint_data = results.pose_landmarks.landmark[joint_idx]

                    # Extract the appropriate coordinate
                    if coordinate == 'x':
                        current_features.append(joint_data.x)
                    elif coordinate == 'y':
                        current_features.append(joint_data.y)
                    elif coordinate == 'z':
                        current_features.append(joint_data.z)
                else:
                    print(f"WARNING: Joint {joint_name} not found in keypoint_names, using zero value")
                    current_features.append(0.0)
            except Exception as e:
                print(f"ERROR: Processing joint {joint_name}: {e}")
                current_features.append(0.0)

    return current_features


# Display gesture probabilities on image
def display_probabilities(image, all_probabilities, predicted_gesture):
    y_offset = 70
    cv2.putText(image, "Gesture Probabilities:", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    y_offset += 30

    # Sort probabilities from highest to lowest
    sorted_probs = sorted(all_probabilities.items(), key=lambda x: x[1], reverse=True)

    for gesture_name, prob in sorted_probs:
        # Determine bar length based on probability
        bar_length = int(prob * 200)

        # Highlight the predicted gesture
        if gesture_name == predicted_gesture:
            text_color = (0, 255, 255)  # Yellow for the chosen gesture
            bar_color = (0, 255, 255)
            thickness = 2
            prefix = "►"
        else:
            text_color = (200, 200, 200)  # Light gray for other gestures
            bar_color = (100, 100, 100)
            thickness = 1
            prefix = " "

        # Draw text label and probability value
        text = f"{prefix} {gesture_name}: {prob:.4f}"
        cv2.putText(image, text, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, thickness, cv2.LINE_AA)

        # Draw visual probability bar
        cv2.rectangle(image, (220, y_offset - 15), (220 + bar_length, y_offset - 5),
                      bar_color, -1)  # -1 for filled rectangle

        y_offset += 25

    return image


# Save collected data
def save_collected_data(data_dir, unprocessed_data, preprocessed_data, prediction_data):
    timestamp = int(time.time())
    data_path = data_dir / f"session_{timestamp}"
    data_path.mkdir(exist_ok=True)

    print(f"Saving collected data to {data_path}...")

    # Save unprocessed data
    if unprocessed_data:
        np.save(data_path / "unprocessed_data.npy", np.array(unprocessed_data))
        print(f"Saved {len(unprocessed_data)} unprocessed data samples")

    # Save preprocessed data
    if preprocessed_data:
        np.save(data_path / "preprocessed_data.npy", np.array(preprocessed_data))
        print(f"Saved {len(preprocessed_data)} preprocessed data samples")

    # Save predictions
    if prediction_data:
        with open(data_path / "predictions.json", 'w') as f:
            json.dump(prediction_data, f, indent=2)
        print(f"Saved {len(prediction_data)} prediction records")

    print("Data collection complete!")


# Sanic routes
@app.route("/")
async def index(request):
    return html(open(slideshow_root_path.joinpath("slideshow.html"), "r").read())


@app.websocket("/events")
async def emitter(_request, ws):
    print("==============================================================")
    print("WebSocket connection opened - starting handler")
    print("==============================================================")

    if DEMO_MODE:
        # Demo mode - send predefined commands
        await run_demo_mode(ws)
    else:
        # Gesture recognition mode
        await run_gesture_recognition(ws)


async def run_demo_mode(ws):
    # Simple event emission for testing
    while True:
        print("emitting 'right'")
        await ws.send("right")
        await asyncio.sleep(2)

        print("emitting 'rotate'")
        await ws.send("rotate")
        await asyncio.sleep(2)

        print("emitting 'left'")
        await ws.send("left")
        await asyncio.sleep(2)


async def run_gesture_recognition(ws):
    print("Starting gesture recognition mode...")

    # Initialize components
    KEYPOINT_NAMES = load_keypoint_mappings()
    dp = init_preprocessor()
    model = load_model()
    gesture_map, gesture_id_map, mp_drawing, mp_drawing_styles, mp_pose = setup_gesture_recognition()

    # Set up data collection if enabled
    unprocessed_data = []
    preprocessed_data = []
    prediction_data = []

    if DATA_COLLECTION_MODE:
        print("DATA COLLECTION MODE ENABLED - Data will be saved when ESC is pressed")
        data_dir = pathlib.Path("collected_data")
        data_dir.mkdir(exist_ok=True)

    # Check if model is loaded
    if model is None:
        print("ERROR: Model not loaded properly, cannot continue with gesture recognition")
        return

    # Initialize pose detector and webcam
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    # Initialize window buffer and tracking variables
    window_size = g_var.WINDOW_SIZE
    feature_window = []

    # Gesture tracking variables
    last_gesture = g_var.idle
    cooldown = g_var.COOLDOWN_FRAMES
    confirmation_threshold = g_var.CONFIRMATION_FRAMES
    confirmation_counters = {gesture: 0 for gesture in gesture_map.values()}

    print(f"Using cooldown: {cooldown} frames")
    print(f"Using confirmation threshold: {confirmation_threshold} frames")

    try:
        while True:
            # Allow server to handle other tasks
            await asyncio.sleep(0.001)

            # Capture frame
            success, image = cap.read()
            if not success:
                print("ERROR: Failed to capture frame")
                continue

            # Flip image if needed
            if FLIP_IMAGE:
                image = cv2.flip(image, 1)

            # Process frame with MediaPipe
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                # Extract features
                current_features = extract_features(results, KEYPOINT_NAMES)

                # Add features to window
                if len(current_features) == len(g_var.MediaPipe_features):
                    feature_window.append(current_features)

                    # Keep window at the right size
                    if len(feature_window) > window_size:
                        feature_window.pop(0)
                else:
                    print(
                        f"WARNING: Feature count mismatch. Expected {len(g_var.MediaPipe_features)}, got {len(current_features)}")

                # Preprocess and predict when window is full
                if len(feature_window) == window_size:
                    # Convert feature window to numpy array
                    np_window = np.array(feature_window)

                    # Store unprocessed data if in collection mode
                    if DATA_COLLECTION_MODE:
                        unprocessed_data.append(np_window.copy())

                    # Preprocess the data
                    processed_data = dp.preprocess_live_data(np_window)
                    if processed_data is None:
                        print("ERROR: Data preprocessing failed")
                        continue

                    # Store preprocessed data if in collection mode
                    if DATA_COLLECTION_MODE:
                        preprocessed_data.append(processed_data.copy())

                    # Reshape to add batch dimension
                    processed_data = processed_data.reshape(1, -1)

                    # Predict gesture
                    prediction_probs = model.forward_logits(processed_data)

                    # Apply softmax if needed
                    if not np.isclose(np.sum(prediction_probs[0]), 1.0):
                        prediction_probs = nn.softmax(prediction_probs)

                    # Get the predicted gesture
                    predicted_gesture_idx = np.argmax(prediction_probs[0])
                    confidence = prediction_probs[0, predicted_gesture_idx]

                    # Map the index to the gesture name
                    try:
                        predicted_gesture = gesture_map[predicted_gesture_idx]
                        print(f"DEBUG: Predicted gesture: {predicted_gesture}, confidence: {confidence:.4f}")

                        # Store prediction data if in collection mode
                        if DATA_COLLECTION_MODE:
                            timestamp = time.time()
                            prediction_entry = {
                                "timestamp": timestamp,
                                "predicted_gesture": predicted_gesture,
                                "confidence": float(confidence),
                                "all_probabilities": {gesture_map[idx]: float(prediction_probs[0, idx])
                                                      for idx in gesture_map}
                            }
                            prediction_data.append(prediction_entry)

                        # Gather all probabilities for display
                        all_probabilities = {}
                        for idx, gesture_name in gesture_map.items():
                            prob = float(prediction_probs[0, idx])
                            all_probabilities[gesture_name] = prob
                            print(f"  - {gesture_name}: {prob:.4f}")

                        # Process gesture if it meets criteria
                        confidence_threshold = 0.6

                        if (predicted_gesture != g_var.idle and confidence > confidence_threshold):
                            # Increment counter for this gesture
                            confirmation_counters[predicted_gesture] += 1

                            # Reset counters for other gestures
                            for gesture in confirmation_counters:
                                if gesture != predicted_gesture:
                                    confirmation_counters[gesture] = 0

                            print(
                                f"Confirmation progress for {predicted_gesture}: {confirmation_counters[predicted_gesture]}/{confirmation_threshold}")

                            # Check if we've reached the confirmation threshold and cooldown is ready
                            if confirmation_counters[predicted_gesture] >= confirmation_threshold and cooldown <= 0:
                                print(
                                    f"CONFIRMED: Detected gesture: {predicted_gesture} with confidence {confidence:.4f}")

                                # Map gesture to slideshow control
                                if predicted_gesture == g_var.swipe_left:
                                    await ws.send("left")
                                elif predicted_gesture == g_var.swipe_right:
                                    await ws.send("right")
                                elif predicted_gesture == g_var.rotate:
                                    await ws.send("rotate")
                                elif predicted_gesture == g_var.rolling:
                                    await ws.send("rolling")
                                elif predicted_gesture == g_var.flip:
                                    await ws.send("zoom_out")

                                # Reset confirmation counter for all gestures
                                for gesture in confirmation_counters:
                                    confirmation_counters[gesture] = 0

                                # Reset cooldown and update last gesture
                                cooldown = g_var.COOLDOWN_FRAMES
                                last_gesture = predicted_gesture
                        else:
                            # Reset all confirmation counters when idle or low confidence
                            for gesture in confirmation_counters:
                                confirmation_counters[gesture] = 0

                    except KeyError:
                        print(f"ERROR: Gesture index {predicted_gesture_idx} not found in gesture map")
                        all_probabilities = {}

                    # Update cooldown counter
                    if cooldown > 0:
                        cooldown -= 1

                    # Display processed image with skeleton overlay and decision map
                    image.flags.writeable = True
                    image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

                    # Draw pose landmarks using MediaPipe drawing utilities
                    mp_drawing.draw_landmarks(
                        image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

                    # Add status information
                    current_status = f"Predicted: {predicted_gesture} ({confidence:.2f}), Cooldown: {cooldown}"
                    if predicted_gesture != g_var.idle:
                        current_status += f", Confirmation: {confirmation_counters[predicted_gesture]}/{confirmation_threshold}"

                    cv2.putText(image, current_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 0), 2, cv2.LINE_AA)

                    # Add probability display
                    if all_probabilities:
                        image = display_probabilities(image, all_probabilities, predicted_gesture)

                    # Add last recognized gesture
                    last_gesture_text = f"Last recognized: {last_gesture}"
                    cv2.putText(image, last_gesture_text, (10, image.shape[0] - 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

                    # Add instruction text
                    if DATA_COLLECTION_MODE:
                        collection_text = f"DATA COLLECTION ACTIVE - {len(prediction_data)} frames collected"
                        cv2.putText(image, collection_text, (10, image.shape[0] - 70),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

                    cv2.putText(image, "Press ESC to exit" + (" and save data" if DATA_COLLECTION_MODE else ""),
                                (10, image.shape[0] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                    cv2.imshow('Gesture Recognition', image)

            # Check for exit command
            if cv2.waitKey(5) & 0xFF == 27:  # ESC key
                # Save collected data if in data collection mode
                if DATA_COLLECTION_MODE and (unprocessed_data or preprocessed_data or prediction_data):
                    save_collected_data(data_dir, unprocessed_data, preprocessed_data, prediction_data)
                break

    except Exception as e:
        print(f"Error in websocket handler: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose.close()
        print("Websocket connection closed")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)