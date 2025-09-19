# src/config.py

import torch

# -- Project Structure --
DATA_ROOT = 'Data/train'
MODEL_DIR = 'models'
RESULTS_DIR = 'results'

# -- Data Configuration --
# The labels must match the folder names in DATA_ROOT
GESTURE_LABELS = {
    "swipe_left": 0,
    "swipe_right": 1,
    "rotate": 2,
    "flip": 3,
    "rolling": 4,
    "idle": 5
}
# -- Feature Configuration --
# List of all features to be used in the model.
# This list is used in data_loader.py to select the correct columns from the CSV.
FEATURES_TO_USE = [
    "nose_x", "nose_y", "nose_z", "nose_confidence",
    "left_eye_inner_x", "left_eye_inner_y", "left_eye_inner_z", "left_eye_inner_confidence",
    "left_eye_x", "left_eye_y", "left_eye_z", "left_eye_confidence",
    "left_eye_outer_x", "left_eye_outer_y", "left_eye_outer_z", "left_eye_outer_confidence",
    "right_eye_inner_x", "right_eye_inner_y", "right_eye_inner_z", "right_eye_inner_confidence",
    "right_eye_x", "right_eye_y", "right_eye_z", "right_eye_confidence",
    "right_eye_outer_x", "right_eye_outer_y", "right_eye_outer_z", "right_eye_outer_confidence",
    "left_ear_x", "left_ear_y", "left_ear_z", "left_ear_confidence",
    "right_ear_x", "right_ear_y", "right_ear_z", "right_ear_confidence",
    "left_mouth_x", "left_mouth_y", "left_mouth_z", "left_mouth_confidence",
    "right_mouth_x", "right_mouth_y", "right_mouth_z", "right_mouth_confidence",
    "left_shoulder_x", "left_shoulder_y", "left_shoulder_z", "left_shoulder_confidence",
    "right_shoulder_x", "right_shoulder_y", "right_shoulder_z", "right_shoulder_confidence",
    "left_elbow_x", "left_elbow_y", "left_elbow_z", "left_elbow_confidence",
    "right_elbow_x", "right_elbow_y", "right_elbow_z", "right_elbow_confidence",
    "left_wrist_x", "left_wrist_y", "left_wrist_z", "left_wrist_confidence",
    "right_wrist_x", "right_wrist_y", "right_wrist_z", "right_wrist_confidence",
    "left_pinky_x", "left_pinky_y", "left_pinky_z", "left_pinky_confidence",
    "right_pinky_x", "right_pinky_y", "right_pinky_z", "right_pinky_confidence",
    "left_index_x", "left_index_y", "left_index_z", "left_index_confidence",
    "right_index_x", "right_index_y", "right_index_z", "right_index_confidence",
    "left_thumb_x", "left_thumb_y", "left_thumb_z", "left_thumb_confidence",
    "right_thumb_x", "right_thumb_y", "right_thumb_z", "right_thumb_confidence",
]

# Dynamically create SELECTED_KEYPOINTS from FEATURES_TO_USE for consistent ordering
# This derived constant is used by both the data loader and preprocessing modules.
_keypoint_names = set()
_coordinate_columns = [f for f in FEATURES_TO_USE if f.endswith(('_x', '_y', '_z'))]
for col in _coordinate_columns:
    _keypoint_names.add('_'.join(col.split('_')[:-1]))
SELECTED_KEYPOINTS = sorted(list(_keypoint_names))


# Gestures for state control
MAX_SEQ_LENGTH = 20  # Fixed sequence length for padding/truncating
WINDOW_STRIDE = 5  # Stride for the sliding window in data creation
# Strategy for labeling sequences: 'any' or 'majority'
# 'any': sequence is a gesture if any frame is a gesture
# 'majority': sequence is a gesture if >50% of frames are gestures
LABELING_STRATEGY = 'majority'

# -- Training Data Strategy --
# If True, the data loaders will ignore the 'idle' portions of gesture files
# and instead use only the dedicated CSVs in the 'Data/train/idle' folder.
# If False, it uses all data as before.
USE_DEDICATED_IDLE_DATA = True

# -- Preprocessing Configuration --
# Defines the global preprocessing strategy. Can be one of:
# 'body_centric': Uses shoulder-based normalization for position/size/rotation invariance.
# 'standardize': Uses Z-score standardization (mean=0, std=1) across the training set.
PREPROCESSING_STRATEGY = 'body_centric'  # or 'standardize'

# Path to save/load the fitted standard scaler. Only used if strategy is 'standardize'.
SCALER_PATH = f'{MODEL_DIR}/standard_scaler.json'

# -- Feature Engineering Configuration --
# A dictionary to enable or disable specific feature engineering steps.
# This allows for modular experimentation with different feature sets.
FEATURE_ENGINEERING_CONFIG = {
    'velocity': True,
    'acceleration': True,
    'distances': False,  # Placeholder for future implementation
    'angles': False,     # Placeholder for future implementation
}

# -- Model Configuration --
# Choose 'ffnn' for the baseline or 'gru' for the main model
DEFAULT_MODEL_TYPE = 'gru'

# A dictionary to hold the hyperparameters for each model architecture.
# This makes it easy to tune model-specific parameters from one central place.
MODEL_PARAMS = {
    'gru': {
        'hidden_sizes': [128, 64, 32],
        'dropout_rate': 0.5
    },
    'ffnn': {
        'hidden_sizes': [256, 128],
        'dropout_rate': 0.5
    },
    'spotter': {
        'hidden_sizes': [64, 32],
        'dropout_rate': 0.3
    }
}

# -- Training Hyperparameters --
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 0.0001
VALIDATION_SPLIT = 0.2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -- Early Stopping Configuration --
# Patience: How many epochs to wait for improvement before stopping.
EARLY_STOPPING_PATIENCE = 5
# Min Delta: Minimum change in the monitored quantity to qualify as an improvement.
EARLY_STOPPING_MIN_DELTA = 0.001

# -- Real-time Inference Configuration --
# Maps the gesture name recognized by the model to the event sent over the WebSocket.
GESTURE_ACTIONS = {
  "swipe_left": "left",
  "swipe_right": "right",
  "flip": "down"
}

# Defines which gesture is used to toggle the lock state.
TOGGLE_LOCK_GESTURE = "rolling"

# Defines which gesture is used to manually reset the presentation to the first slide.
RESET_GESTURE = "rotate"

# --- Main GRU Model ---
# Confidence threshold for each gesture (how many times it must appear in the buffer)
GESTURE_CONFIDENCE_THRESHOLDS = {
    "flip": 9,
    "rolling": 9,
    "swipe_left": 8,
    "swipe_right": 8,
    "idle": 10
}
PREDICTION_SMOOTHING_BUFFER_SIZE = 10
# Cooldown period in seconds for each gesture to prevent spamming
GESTURE_COOLDOWN_PERIODS = {
    "flip": 1.5,
    "rolling": 2.0,
    "swipe_left": 1.0,
    "swipe_right": 1.0,
}

# --- Spotter Model ---
# Path to the trained spotter model checkpoint.
SPOTTER_MODEL_PATH = f'{MODEL_DIR}/spotter_model-epoch=29-val_acc=0.8985.ckpt'
# Confidence threshold for the spotter to classify a frame as "motion".
SPOTTER_THRESHOLD = 0.8
# Buffer to determine if a gesture has started or stopped.
# A gesture is considered "active" if `SPOTTER_BUFFER_SIZE` consecutive frames are classified as motion.
SPOTTER_BUFFER_SIZE = 5

# -- Head Pose Estimation and Tutorial Timers (in seconds) --
ENABLE_HEAD_POSE_DETECTION = True
# Thresholds for pitch and yaw to determine if the user is looking forward.
# Values are in degrees.
HEAD_POSE_THRESHOLDS = {
    'pitch': 15,  # Max up/down deviation
    'yaw': 18     # Max left/right deviation
}
# How long a user must be looking at the screen to trigger the tutorial.
ATTENTION_DETECT_THRESHOLD = 4.0
# How long a user must look away for the system to lock and reset.
DISENGAGEMENT_TIMEOUT = 10.0
# How long an unlocked, attentive user must be inactive before a helpful tooltip is shown.
IDLE_TOOLTIP_THRESHOLD = 6.0
# How long the idle tooltip should remain visible before automatically hiding.
IDLE_TOOLTIP_DURATION = 5.0

# =================================================================
# Fine-Tuning Configuration
# =================================================================
FINETUNING_CONFIG = {
    # A global switch to activate the fine-tuning data pipeline.
    # This is set to True at runtime by the finetune.py script.
    "ENABLED": False,

    # Use a much smaller learning rate to gently adjust the model's weights.
    "FINETUNE_LEARNING_RATE": 1e-4,

    # Specify exactly which idle datasets to use for fine-tuning.
    # This allows you to experiment with different levels of "chaotic" data.
    "IDLE_FILES_TO_USE": [
        "idle_confusion_hard.csv"
    ],

    # The core of catastrophic forgetting prevention. This sets the fraction
    # of the ORIGINAL gesture data to mix in with the idle data.
    # 0.0 = Pure idle data.
    # 0.1 = All idle data + a random 10% of the original gestures.
    "GESTURE_DATA_SAMPLE_FRACTION": 0.1,
}
