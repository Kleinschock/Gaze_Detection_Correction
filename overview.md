# Project Overview & System Architecture

This project implements a real-time gesture recognition system to control a computer using hand and arm movements. It is built on a robust, **two-stage machine learning pipeline** that leverages full-body pose estimation from a standard webcam to achieve high accuracy while minimizing false positives during idle periods.

1.  **Stage 1: Gesture Spotting**: A lightweight binary classifier (`SpotterNet`) analyzes a small, sliding window of the most recent frames to determine if the user is performing *any* intentional motion. Although it makes a prediction for each new frame, it uses the context of a small window (`SPOTTER_WINDOW_SIZE`) to calculate dynamic features like velocity. This allows it to reliably distinguish between static poses and deliberate movements, acting as an efficient gate to prevent ambiguous or idle actions from being processed further.
2.  **Stage 2: Gesture Classification**: Only when the spotter confirms that a deliberate motion is underway does the system begin feeding data to the main, more complex recurrent neural network (`GRUNet`). This model's sole focus is to classify the *type* of gesture being performed (e.g., "swipe left").

This architecture ensures both high responsiveness and high precision, making the system practical for real-world use. The entire pipeline is designed with modularity and scientific rigor in mind, incorporating advanced feature engineering, modern deep learning practices with PyTorch Lightning, and sound Human-Computer Interaction (HCI) principles.


# File-by-File Breakdown

This section details the purpose and functionality of each key file in the `src` directory.

### `config.py`
*   **Purpose**: Centralized configuration hub for the entire project.
*   **Functionality**: Defines all critical constants and hyperparameters. This includes gesture labels, data paths, model parameters, data creation settings (`WINDOW_STRIDE`), the `LABELING_STRATEGY`, the `PREPROCESSING_STRATEGY`, and the `FEATURES_TO_USE` list. **Crucially, it now also contains a `FEATURE_ENGINEERING_CONFIG` dictionary, which allows individual features like velocity and acceleration to be enabled or disabled globally with a simple boolean flag. All stale, hardcoded feature counts have been removed.**

### `data_loader.py`
*   **Purpose**: The core of the data loading and preparation pipeline for the main GRU model.
*   **Functionality**:
    *   Loads raw keypoint data from CSV files.
    *   Creates overlapping data sequences using a sliding window approach (`MAX_SEQ_LENGTH`, `WINDOW_STRIDE`).
    *   Implements a modular labeling strategy (`LABELING_STRATEGY`) to assign a label to each window.
    *   **Supports two distinct data loading strategies controlled by `USE_DEDICATED_IDLE_DATA` in `config.py`**:
        1.  **Standard Strategy**: Loads all data, including the 'idle' portions found within gesture recordings.
        2.  **Dedicated Idle Strategy**: To improve the model's robustness, this strategy first creates all possible windows from the gesture files and then **filters out** any windows that were labeled as 'idle'. It then separately loads the more challenging 'idle' data from the dedicated `Data/train/idle` folder. This provides the model with cleaner gesture examples and more realistic negative examples.
    *   **Crucially, it now delegates all feature engineering and normalization to the centralized `preprocessing.py` module**, ensuring consistency across the project. **It no longer contains any hardcoded logic for calculating the number of features.**
    *   Defines the `GestureDataset` class, which serves as the foundation for the main model.

### `preprocessing.py`
*   **Purpose**: A centralized, configurable module for all data preprocessing and feature engineering.
*   **Functionality**:
    *   Contains functions for the two mutually exclusive normalization strategies (`body_centric` or `standardize`).
    *   The main entry point is the `preprocess_sequence` function. This function is called by both the training pipeline (`data_loader.py`) and the live inference pipeline (`run_live.py`).
    *   Inside `preprocess_sequence`, it first applies the normalization strategy defined by `PREPROCESSING_STRATEGY` in `config.py`.
    *   **Crucially, it then reads the `FEATURE_ENGINEERING_CONFIG` dictionary from `config.py`. Based on which flags in this dictionary are set to `True`, it dynamically calls the corresponding calculation functions (e.g., `calculate_velocity`) from the `feature_engineering.py` module and appends the results.** This confirms that the `FEATURE_ENGINEERING_CONFIG` is the central switchboard that controls which features are generated across the entire project.

### `feature_engineering.py`
*   **Purpose**: A dedicated module containing all logic for calculating individual features.
*   **Functionality**:
    *   Provides clean, isolated functions for each feature calculation (e.g., `calculate_velocity`, `calculate_acceleration`).
    *   This modular approach makes it easy to add, modify, or debug individual feature calculations without altering the main preprocessing pipeline.

### `models.py`
*   **Purpose**: Defines the neural network architectures for gesture classification.
*   **Functionality**:
    *   `FFNN`: A simple Feed-Forward Neural Network that serves as a performance baseline. It flattens the time series data, ignoring temporal dynamics.
    *   `GRUNet`: The main classification model, a Gated Recurrent Unit network, which excels at learning from sequential data.
    *   **Crucially, both model classes (`FFNN`, `GRUNet`) are now fully decoupled from the project's configuration. They accept `input_size` as a constructor argument, making them self-contained and robust.**
    *   `get_model()`: A factory function that now takes the dynamically calculated `input_size` and passes it to the requested model, ensuring the architecture always matches the data.

### `lightning_datamodule.py` & `lightning_module.py`
*   **Purpose**: Encapsulates all PyTorch Lightning logic for the main classification task.
*   **Functionality**:
    *   `GestureDataModule`: Manages the creation of data loaders (train, validation, test). **Crucially, it now ensures that if the `standardize` strategy is used, the scaler is fitted *only* on the training data *after* the train/val/test split has occurred.** This prevents data leakage and guarantees that the scaler used for validation, testing, and live inference is identical to the one derived from the training set.
    *   `GestureLightningModule`: Contains the complete training, validation, and testing logic. It defines the loss function (Cross-Entropy), optimizer (AdamW), and calculates and logs all relevant metrics, including accuracy and a confusion matrix visualized in `wandb`.

### `class_balancing.py`
*   **Purpose**: Provides strategies to handle imbalanced datasets.
*   **Functionality**:
    *   `calculate_inverse_frequency_weights`: Calculates class weights for use in a weighted loss function. This is a common technique where the loss for samples from minority classes is scaled up.
    *   `undersample_data`: Implements random undersampling. It reduces the number of samples in the majority classes to match the number in the minority class, creating a balanced dataset at the data level.

### `train.py`
*   **Purpose**: Executable scripts to orchestrate the training process for the two models.
*   **Functionality**:
    *   Handle command-line arguments for selecting models and strategies.
    *   **`train.py` now contains a `get_input_size()` helper function that dynamically calculates the exact number of features based on the settings in `config.py` (`FEATURES_TO_USE` and `FEATURE_ENGINEERING_CONFIG`).**
    *   This dynamically calculated `input_size` is then passed through the `LightningModule` to the model constructor. This is the core of the fix that ensures the model's input layer always matches the feature pipeline.
    *   Initialize the appropriate `DataModule` and `LightningModule`.
    *   Configure PyTorch Lightning `Trainer` with callbacks like `ModelCheckpoint` (to save the best model) and `EarlyStopping` (to prevent overfitting).
    *   Integrate with `WandbLogger` for comprehensive experiment tracking.
### `run_live.py`
*   **Purpose**: The main entry point for running the real-time gesture recognition system.
*   **Functionality**:
    *   **Automatic Model Loading**: At startup, the script automatically finds the most recently created `gru_main_model-*.ckpt` file in the `models/` directory and uses it by default. This can be overridden with the `--model_path` argument.
    *   Initializes the camera and MediaPipe for live pose estimation.
    *   **Supports two distinct operating modes via the `--mode` flag**:
        1.  **`gru_only` (Default)**: Implements a simplified, direct-to-classification pipeline. It loads only the main `GRUNet` model and continuously feeds it sequences of frames. This is the standard mode for direct gesture recognition.
        2.  **`spotter`**: Implements the full two-stage pipeline, which must be explicitly enabled. It loads both the lightweight `SpotterNet` and the main `GRUNet` models.
            *   **Stage 1 Execution**: For each frame, it preprocesses a small window of data and feeds it to the spotter model to detect "motion" vs. "no motion". A buffer (`SPOTTER_BUFFER_SIZE`) is used to smooth these predictions and reliably detect the start and end of a gesture.
            *   **Stage 2 Execution**: Only when the spotter confirms a gesture is active does it begin collecting a full sequence of frames (`MAX_SEQ_LENGTH`) for the main GRU model.
    *   **It calls the centralized `preprocess_sequence` function from `preprocessing.py` for all models**, ensuring that the live data undergoes the exact same, configurable transformations as the training data.
    *   Manages the system state (e.g., locked/unlocked) and uses a `prediction_buffer` to stabilize and confirm the final gesture classification before triggering actions.
    *   Translates recognized gestures into WebSocket events (e.g., `{"event": "right"}`) that are broadcast to the frontend to control the slideshow. It also provides rich visual feedback directly on the camera feed. **The UI is now context-aware, showing spotter-related information only when the `spotter` mode is active.**
    *   **Debug Mode**: Includes a `--debug` flag to log prediction probabilities for detailed analysis.

### `spotter_data_loader.py` & `train_spotter.py`
*   **Purpose**: A dedicated data loader and training script for the binary spotter model.
*   **Functionality**:
    *   `spotter_data_loader.py`: Loads all training data and correctly labels each frame as either "motion" (1) or "no motion" (0). **Crucially, it does not treat each frame in isolation.** It uses a small sliding window, defined by `SPOTTER_WINDOW_SIZE` in `config.py`, to create a small sequence for *every single frame* in the dataset. This context is essential for the preprocessing pipeline to calculate dynamic features like velocity and acceleration, which are impossible to derive from a single frame. This allows the spotter model to learn the difference between a static pose and an intentional movement.
    *   **Supports two distinct data loading strategies controlled by `USE_DEDICATED_IDLE_DATA` in `config.py`**:
        1.  **Standard Strategy**: Loads all frames from all files.
        2.  **Dedicated Idle Strategy**: This strategy first loads all gesture files and **filters out** any rows marked as 'idle' *before* processing. It then separately loads all data from the dedicated `Data/train/idle` folder. This provides the spotter with cleaner "motion" examples and more realistic "no motion" examples.
    *   It also supports class balancing strategies like undersampling.
    *   `train_spotter.py`: An executable script that orchestrates the training of the `SpotterNet` model using the `SpotterDataset` and a dedicated `SpotterLightningModule` configured for binary classification.

### `tuning.py`
*   **Purpose**: An executable script to perform hyperparameter optimization for all models (`GRU`, `FFNN`, `Spotter`) using Weights & Biases (W&B) Sweeps.
*   **Functionality**:
    *   **Model-Specific Tuning**: The script now accepts a `--model_type` argument (`gru`, `ffnn`, or `spotter`) to specify which model to tune.
    *   **Separate Sweep Configurations**: It contains three distinct `sweep_config` dictionaries (`sweep_config_gru`, `sweep_config_ffnn`, and `sweep_config_spotter`), each tailored with appropriate hyperparameter ranges for its respective model architecture.
    *   **W&B Agent Integration**: It uses `wandb.agent()` to run the sweep. The agent calls a dedicated training function based on the selected model type.
    *   **Reusable Training Logic**: For each trial, the sweep function calls the appropriate refactored training function (`run_training()` for `gru`/`ffnn` or `run_spotter_training()` for `spotter`), ensuring a clean and modular workflow.
    *   **Usage**:
        *   To tune the GRU model: `python -m src.tuning --model_type gru --count <number_of_runs>`
        *   To tune the FFNN model: `python -m src.tuning --model_type ffnn --count <number_of_runs>`
        *   To tune the Spotter model: `python -m src.tuning --model_type spotter --count <number_of_runs>`

### `finetune.py`
*   **Purpose**: A dedicated, executable script for fine-tuning a pre-trained model on a specialized dataset.
*   **Functionality**:
    *   **Targeted Specialization**: Allows you to take a high-performing general model and further train it on a specific subset of data (e.g., challenging "idle" examples) to improve its performance on that specific task without retraining from scratch.
    *   **Configurable Data Mix**: The fine-tuning data is controlled by the `FINETUNING_CONFIG` dictionary in `config.py`. This allows you to specify exactly which idle files to use and what fraction of the original gesture data to mix in, which is crucial for preventing "catastrophic forgetting."
    *   **Controlled Learning**: It automatically uses a much lower learning rate (`FINETUNE_LEARNING_RATE`) to make small, precise adjustments to the model's weights.
    *   **Non-Destructive**: Saves the resulting model with a `finetuned-` prefix, ensuring your original base model is never overwritten.
    *   **Integrated Logging**: Automatically tags runs in Weights & Biases with `finetune` for easy tracking and comparison.
    *   **Usage**: `python -m src.finetune --checkpoint_path <path/to/base_model.ckpt>`

### `head_pose_estimation.py`
*   **Purpose**: A modular component for detecting the user's head orientation.
*   **Functionality**:
    *   Uses Google's MediaPipe `FaceMesh` to detect facial landmarks in real-time.
    *   Calculates the 3D head pose (pitch, yaw, roll) from these landmarks using OpenCV's `solvePnP` function.
    *   Provides a simple method to determine if the user is looking forward, based on configurable angle thresholds.
    *   This module is designed to be integrated into `run_live.py` to add a layer of attention detection, ensuring the system can differentiate between a user who is actively engaged and one who is looking away.

### `evaluate.py`
*   **Purpose**: A dedicated script for the quantitative evaluation of trained models.
*   **Functionality**:
    *   Loads the test dataset and the final trained models (`ffnn`, `gru`).
    *   Runs inference on the test set to gather predictions.
    *   Generates and prints a detailed `classification_report` (including precision, recall, F1-score) for each model.
    *   Creates and saves a visual confusion matrix (`.png`) for each model to the `results/` directory.
    *   Generates and saves a `performance_comparison.csv` file summarizing the key metrics (accuracy, F1-score) for all evaluated models, enabling easy comparison.

### `scaler.py`
*   **Purpose**: Provides a stateful `StandardScaler` class for Z-score normalization.
*   **Functionality**:
    *   Implements `fit`, `transform`, and `fit_transform` methods, mirroring the behavior of `scikit-learn`.
    *   Crucially, it can save its calculated `mean_` and `std_` parameters to a JSON file and load them back.
    *   This ensures that the exact same scaling is applied during training, validation, testing, and live inference, which is a critical requirement for the `'standardize'` preprocessing strategy and prevents data leakage.

### `debug_data_loader.py`
*   **Purpose**: A simple utility script for debugging the data loading process.
*   **Functionality**:
    *   Instantiates the `GestureDataset` with a `debug=True` flag.
    *   This activates detailed print statements within the `GestureDataset` to trace how data files are found, loaded, and how windows are created and labeled, which is invaluable for troubleshooting data preparation issues.

---

# Raw Data Structure & Content

The foundation of this project is the raw data collected and stored in the `Data/train/` directory. Understanding its structure is key to understanding the entire pipeline.

*   **Directory Structure**: The data is organized into subdirectories, where each directory name corresponds to a specific gesture (e.g., `swipe_left`, `idle`, `rotate`).
*   **File Content**: Each directory contains a small number of CSV files (typically 2-4 per gesture). Each CSV represents a single, continuous recording session.
*   **CSV Format**:
    *   Each row in a CSV file represents a single frame captured from the camera.
    *   There are **134 columns** in total.
    *   **Column 1**: `timestamp`.
    *   **Columns 2-133**: Raw keypoint data from MediaPipe Pose. For each of the 33 keypoints, there are four values: `_x`, `_y`, `_z`, and `_confidence`.
    *   **Column 134**: `ground_truth`. This is a manually or semi-automatically labeled binary flag (0 or 1) indicating whether the frame is considered part of the intended gesture. This column is essential for the `LABELING_STRATEGY` and for training the binary `SpotterNet`.
*   **Data Volume**: While the number of individual recordings is low, each recording is several thousand frames long, providing a large amount of sequential data for the sliding-window mechanism to generate training samples from.

# Data and Feature Engineering

The quality of the input data is critical for any machine learning model. This project employs a scientifically-grounded, multi-level feature engineering pipeline to transform raw 3D keypoint coordinates into a rich, informative feature set. This is performed identically for both training (`data_loader.py`) and live inference (`run_live.py`).

*   **Level 1: Configurable Keypoint Selection**: The exact keypoints used for feature engineering are now defined in the `FEATURES_TO_USE` list in `config.py`. The `data_loader.py` and `run_live.py` scripts dynamically adapt to this list, ensuring that the entire pipeline uses a consistent set of features.
*   **Level 2: Configurable Normalization/Standardization**: The project now supports two mutually exclusive preprocessing strategies, controlled by `PREPROCESSING_STRATEGY` in `config.py`:
    1.  **`'body_centric'` (Default)**: To ensure the model is invariant to the user's position, size, and orientation, we apply:
        *   **Position Invariance**: Centering all keypoints relative to the midpoint of the shoulders.
        *   **Size Invariance**: Scaling all keypoints based on the user's shoulder width.
        *   **Rotation Invariance**: Creating a body-centric coordinate system.
    2.  **`'standardize'`**: Applies traditional Z-score standardization. It computes the mean and standard deviation for each feature across the entire training set and scales the data accordingly. The fitted scaler is saved to `models/standard_scaler.json` to ensure identical transformations are used during training, validation, and live inference.
*   **Level 3: Configurable Feature Engineering**: After normalization, the pipeline generates additional features. This process is now fully modular and controlled by the `FEATURE_ENGINEERING_CONFIG` in `config.py`.
    *   The logic for each feature (e.g., **velocity**, **acceleration**) is isolated in its own function within `src/feature_engineering.py`.
    *   The main `preprocess_sequence` function dynamically calls these functions based on the config, allowing for easy experimentation by simply toggling features on or off.

The final feature vector for each frame combines the normalized keypoints with any additional features enabled in the configuration, creating a rich, flexible, and consistent input for the neural networks.

---

# Slideshow Interaction & Real-time UI Control

The project includes a fully integrated, self-contained slideshow demonstration that showcases the gesture recognition engine's capabilities. This system relies on a tightly coupled architecture between the Python backend (`run_live.py`) and a JavaScript-powered frontend (`slideshow/slideshow/`).

### Backend (`run_live.py`)
The backend is an `asyncio`-based application that serves two roles:
1.  **HTTP Server**: A simple `http.server` is launched in a background thread on port `8000`. It serves the static files (`slideshow.html`, `event_listeners.js`, etc.) from the `slideshow/slideshow/` directory, making the demo self-hosting.
2.  **WebSocket Server**: A `websockets` server runs on port `8765`, acting as the real-time communication bridge to the frontend.

The backend is responsible for translating high-level system states and recognized gestures into specific, structured JSON messages that are broadcast to all connected clients.

### Frontend (`event_listeners.js`)
The frontend JavaScript connects to the WebSocket server and listens for incoming messages. A large `switch` statement handles the different event types, manipulating the DOM to update the UI in real-time. This includes:
*   Changing slide content and visibility using the Reveal.js API.
*   Showing/hiding instructional toolbars and tooltips.
*   Applying CSS classes to give visual feedback (e.g., changing the background color based on the lock state).

### The Event-Driven Workflow

This section details the key event chains that create the interactive experience.

1.  **Initial Connection & State Sync**:
    *   When the browser opens `slideshow.html`, `event_listeners.js` connects to `ws://localhost:8765`.
    *   The backend's `register_client` function immediately sends either a `{"event": "system_locked"}` or `{"event": "system_unlocked"}` message so the UI instantly reflects the current state.

2.  **Attention-Based Tutorial**:
    *   **Backend**: If `ENABLE_HEAD_POSE_DETECTION` is true, `run_live.py` monitors head orientation. If the user looks at the screen for `ATTENTION_DETECT_THRESHOLD` seconds, it sends `{"event": "user_detected"}`.
    *   **Frontend**: On receiving `user_detected`, it displays the first tutorial message ("Hey you...") and starts a timer for the next instruction.
    *   **Backend**: When the user performs the `TOGGLE_LOCK_GESTURE`, the backend sends `{"event": "system_unlocked"}`.
    *   **Frontend**: On `system_unlocked`, it shows the next tutorial step ("Great! System unlocked..."). This continues for subsequent gestures (`right`, `down`), guiding the user through the controls.

3.  **Lock/Unlock Mechanism**:
    *   **Backend**: When the `TOGGLE_LOCK_GESTURE` is recognized, the backend flips its internal `is_locked` state and immediately broadcasts `{"event": "system_locked"}` or `{"event": "system_unlocked"}`.
    *   **Frontend**: On `system_locked`, it adds the `.locked` class to the `<body>`, tinting the background red. On `system_unlocked`, it removes this class and triggers a green "pulse" animation for clear visual feedback.

4.  **Gesture Navigation**:
    *   **Backend**: When a navigation gesture (e.g., `swipe_right`) is confirmed, the backend looks up the corresponding action in `cfg.GESTURE_ACTIONS` and sends the associated event, e.g., `{"event": "right"}`.
    *   **Frontend**: On receiving `right`, `left`, or `down`, it calls the corresponding Reveal.js API function (`Reveal.right()`, etc.) to change the slide.

5.  **Idle/Disengagement Handling**:
    *   **Idle Tooltip**: If the user is attentive but inactive for `IDLE_TOOLTIP_THRESHOLD` seconds, the backend sends `{"event": "show_idle_tooltip"}`. The frontend then displays a random hint. The backend later sends `{"event": "hide_idle_tooltip"}` after `IDLE_TOOLTIP_DURATION` or if any action is performed.
    *   **System Reset**: If the user looks away for `DISENGAGEMENT_TIMEOUT` seconds or performs the `RESET_GESTURE`, the backend sends `{"event": "system_reset"}`. The frontend responds by navigating to the very first slide and re-applying the locked state UI, ensuring the system is ready for the next user.

This robust, event-driven architecture decouples the gesture recognition logic from the UI presentation, allowing for a complex and responsive user experience controlled entirely from the Python backend.

---

# Human-Computer Interaction (HCI) Concepts

A usable real-time system requires more than just an accurate model. This project incorporates several HCI principles:

*   **Lock/Unlock Mechanism**: The system starts in a "locked" state to prevent accidental activations. A single, dedicated gesture (`rolling`, as defined by `TOGGLE_LOCK_GESTURE` in the config) is used to switch between the "locked" and "unlocked" states. This toggle-based approach provides deliberate control, a core HCI concept known as "explicit engagement," while being simple for the user to remember.
*   **Feedback**: The system provides immediate visual feedback to the user. The on-screen UI shows the current state (Locked/Active), the recognized gesture, and a visual flash to confirm when an action has been triggered. This keeps the user informed about what the system is doing.
*   **Feedforward**: The UI also serves as feedforward, passively informing the user of the system's current state *before* they act. Seeing the "LOCKED" status tells the user they need to perform the unlock gesture first.
*   **Cooldown and Buffering**: To prevent a single gesture from being spammed and to ensure predictions are stable, the system uses a cooldown period after each action and a prediction buffer to confirm a gesture over several frames.

---

# Changelog


*   **2025-07-28 09:52**: Performed a full framework analysis and updated the documentation.
    *   **`overview.md`**: Read all relevant source code files and analyzed the training data structure. Updated the overview to include descriptions for previously undocumented scripts (`evaluate.py`, `scaler.py`, `debug_data_loader.py`) and added a new, detailed section on the raw data format (directory structure, CSV content, and data volume).
    *   **`data_analysis/analyze_data_structure.py`**: Created and executed this new script to programmatically analyze the training data, providing concrete numbers for the documentation.
    *   **Problem Solved**: The `overview.md` was missing descriptions for several utility and evaluation scripts, and lacked a clear explanation of the raw data itself. This update makes the documentation comprehensive, ensuring any developer can understand the entire project, from raw data to final evaluation.

*   **2025-07-27 23:26**: Aligned data loading for all models to ensure a direct architectural comparison.
    *   **`src/data_loader.py`**: Removed the specialized FFNN data loading strategy. The `GestureDataset` now uses the exact same sliding window data generation method for both `ffnn` and `gru` model types.
    *   **Problem Solved**: The previous logic created a much smaller, non-augmented dataset for the FFNN, causing a confusing discrepancy in the number of training steps per epoch (e.g., 44 vs. 440). This change ensures both models are trained on the identical, augmented dataset, making the comparison a true, apples-to-apples test of architectural performance and resolving the steps-per-epoch mismatch. This corrects a flawed methodological assumption.

*   **2025-07-27 23:07**: Implemented model-aware data loading for fair comparison.
    *   **`src/data_loader.py`**: Modified the `GestureDataset` to accept a `model_type`. It now uses a different data loading strategy for the FFNN, treating each file as a single sample instead of using a sliding window. This prevents data augmentation that is not suitable for the FFNN architecture.
    *   **`src/lightning_datamodule.py`**: Updated to pass the `model_type` down to the `GestureDataset` during instantiation.
    *   **Problem Solved**: The previous implementation used a sliding window for all models, which created a much larger, augmented dataset for the GRU than for the FFNN. This led to a significant and unfair difference in the number of training steps per epoch. The new logic ensures that each model is trained on a dataset that is appropriate for its architecture, resolving the data discrepancy and enabling a scientifically sound comparison.

*   **2025-07-27 22:50**: Unified hyperparameter search spaces for fair model comparison.
    *   **`src/tuning.py`**: Aligned the `batch_size`, `dropout_rate`, and `weight_decay` parameter ranges in `sweep_config_gru` and `sweep_config_ffnn`.
    *   **Problem Solved**: The previous configurations used different search spaces for common hyperparameters, which could have biased the comparison between the GRU and FFNN models. This change ensures that both architectures are optimized over the exact same set of possibilities, leading to a more scientifically sound and unbiased comparison of their performance.

*   **2025-07-27 22:38**: Implemented hyperparameter tuning for the FFNN classification model.
    *   **`src/tuning.py`**: Enhanced the script to support the `ffnn` model as a tunable architecture alongside `gru` and `spotter`. Added a dedicated `sweep_config_ffnn` with a hyperparameter space suitable for deep feed-forward networks. The main sweep function was generalized to handle both `gru` and `ffnn` classifier models.
    *   **Problem Solved**: This fulfills the requirement to quantitatively compare the GRU model against a well-tuned FFNN baseline. The system now supports comprehensive, automated hyperparameter optimization for all three key models in the project, enabling fair and data-driven model comparison.

*   **2025-07-27 22:33**: Implemented hyperparameter tuning for the spotter model.
    *   **`src/tuning.py`**: Major refactoring to support model-specific sweeps. The script now accepts a `--model_type` argument (`main` or `spotter`). It contains separate `sweep_config` dictionaries and dedicated sweep functions for each model, ensuring that the correct hyperparameter space is explored for each architecture.
    *   **`src/train_spotter.py`**: The script was refactored to encapsulate the core logic into a reusable `run_spotter_training(config)` function. This allows the `wandb` agent to programmatically call it with different hyperparameter configurations during a sweep.
    *   **Problem Solved**: The project's powerful hyperparameter optimization capabilities were previously limited to the main classification model. This enhancement extends the same functionality to the spotter model, enabling a data-driven approach to finding the most performant and efficient architecture for the crucial first stage of the recognition pipeline.

*   **2025-07-27 17:22**: Added a new tooltip hint for the reset gesture.
    *   **`slideshow/slideshow/event_listeners.js`**: Added a new hint to the `idleHints` array to inform users that the 'rotate' gesture can be used to reset the presentation.
    *   **Problem Solved**: This improves user guidance by explicitly documenting a key feature within the contextual help system.

*   **2025-07-27 17:21**: Implemented a dynamic and timed idle tooltip system.
    *   **`src/config.py`**: Added `IDLE_TOOLTIP_DURATION` and adjusted `IDLE_TOOLTIP_THRESHOLD` to make the tooltip's timing fully configurable.
    *   **`src/run_live.py`**: Refactored the idle tooltip logic. The system now tracks when a tooltip is shown and sends a `hide_idle_tooltip` event after a configured duration or immediately after any user action. This prevents the tooltip from getting stuck on screen.
    *   **`slideshow/slideshow/event_listeners.js`**: Added a handler for the new `hide_idle_tooltip` event and also added event listeners to hide the tooltip on any key press or mouse click.
    *   **Problem Solved**: The previous tooltip was static and would remain on screen indefinitely. The new system is dynamic: it appears after a short period of inactivity (4s), disappears automatically after a few seconds (3s) or on user action, and can reappear later, creating a much more fluid and helpful user experience.

*   **2025-07-27 17:05**: Fixed a `NameError` in the live inference script.
    *   **`src/run_live.py`**: Corrected a `NameError` by adding the `cfg.` prefix to `SPOTTER_BUFFER_SIZE` and added missing `GestureLightningModule` and `SpotterLightningModule` imports.
    *   **Problem Solved**: A recent refactoring introduced a runtime error that prevented the live script from starting. This fix resolves the error and ensures the application is runnable.

*   **2025-07-27 17:03**: Simplified and clarified the user disengagement logic.
    *   **`src/config.py`**: Removed the redundant `AWAY_TIMEOUT` and `ATTENTION_RESET_THRESHOLD` variables. Replaced them with a single, clearer `DISENGAGEMENT_TIMEOUT` to handle all cases where a user looks away.
    *   **`src/run_live.py`**: Refactored the attention tracking logic to use the single `DISENGAGEMENT_TIMEOUT`. This removes the confusing dual-timeout system and makes the code cleaner and more maintainable.
    *   **Problem Solved**: The previous implementation had two separate, confusingly named timers for handling user disengagement. This refactoring simplifies the configuration and the logic into a single, robust mechanism.

*   **2025-07-27 17:00**: Added a manual reset gesture.
    *   **`src/config.py`**: Added a `RESET_GESTURE` variable, mapping it to the "rotate" gesture.
    *   **`src/run_live.py`**: The main gesture processing loop now checks for the `RESET_GESTURE`. When detected, it triggers the same robust `system_reset` event used by the automatic timeout, forcing the system to lock and return to the first slide.
    *   **Problem Solved**: This provides the user with a quick and reliable way to manually reset the presentation to its initial state at any time, which is especially useful during live demonstrations.

*   **2025-07-27 16:57**: Corrected and consolidated all live mode settings into `src/config.py`.
    *   **`config/live_config.yml`**: This redundant file was deleted to enforce a single source of truth.
    *   **`src/config.py`**: All parameters related to the live mode, including gesture mappings, tutorial timers, and prediction tuning, have been moved into this central configuration file under a new `Real-time Inference Configuration` section.
    *   **`src/run_live.py`**: The script was refactored to remove the YAML-loading logic and now imports all its settings directly from `src/config.py`.
    *   **Problem Solved**: This corrects a design flaw and properly centralizes all project configuration. The live mode is now fully customizable from the main `config.py` file, improving consistency and maintainability.

*   **2025-07-27 16:49**: Enhanced system reset logic and fixed tutorial UI bugs.
    *   **`src/run_live.py`**: Consolidated the user disengagement logic. When a user looks away for more than 4 seconds, the system now immediately locks and sends a single, robust `system_reset` event, removing the previous, less comprehensive `user_lost` event.
    *   **`slideshow/slideshow/event_listeners.js`**: Fixed a bug where the tutorial toolbar could get "stuck" on screen. The logic is now more flexible, allowing any valid gesture to advance the tutorial once the system is unlocked.
    *   **Problem Solved**: This makes the system's state management more robust and predictable. The presentation will now always return to a clean, locked, and ready state for the next user. The tutorial experience is also smoother and less rigid.

*   **2025-07-27 16:46**: Redesigned the tutorial UI to use a top toolbar.
    *   **`slideshow/slideshow/slideshow.html`**: Replaced the intrusive full-screen overlay with a new `div` styled as a toolbar that slides down from the top of the screen. The problematic `filter`-based dimming effect has been completely removed.
    *   **`slideshow/slideshow/event_listeners.js`**: The JavaScript logic was updated to manage the visibility of the new toolbar, sliding it in and out of view as the tutorial progresses.
    *   **Problem Solved**: This resolves a persistent visual bug where the background dimming effect would not correctly revert. The new toolbar design is a cleaner, more professional, and more robust solution for displaying interactive instructions without interfering with other visual feedback like the lock-state background color.

*   **2025-07-27 16:37**: Added an idle user tooltip system to encourage interaction.
    *   **`src/run_live.py`**: Implemented a 10-second inactivity timer. If a user is attentive but performs no gestures, the backend sends a `show_idle_tooltip` event. The timer resets after any user activity.
    *   **`slideshow/slideshow/slideshow.html`**: Added a new `div` and CSS for a non-intrusive tooltip in the bottom-right corner.
    *   **`slideshow/slideshow/event_listeners.js`**: The frontend now handles the `show_idle_tooltip` event by selecting a random hint from a predefined list and displaying it. The tooltip is automatically hidden on the next user action.
    *   **Problem Solved**: This feature prevents a user from getting "stuck" if they are unsure what to do next. By providing gentle, contextual hints during periods of inactivity, the system encourages further exploration and engagement with the presentation.

*   **2025-07-27 14:58**: Implemented an automatic system reset on user disengagement.
    *   **`src/run_live.py`**: Added a 15-second timeout. If the system is unlocked and the user looks away for more than 15 seconds, the system automatically re-locks itself and sends a `system_reset` event.
    *   **`slideshow/slideshow/event_listeners.js`**: Added a handler for the `system_reset` event, which navigates the slideshow back to the first slide and updates the UI to the locked state.
    *   **Problem Solved**: This makes the system more robust for public or kiosk use. If a user walks away without locking the system, it will automatically reset for the next user, ensuring a consistent starting experience.

*   **2025-07-27 14:55**: Implemented an extended tutorial and remapped core gestures.
    *   **`src/run_live.py`**: The gesture mapping has been updated based on user feedback. The `rolling` gesture now acts as a toggle for locking and unlocking the system. The `flip` gesture has been freed up and is now mapped to a new `down` WebSocket event for vertical slide navigation.
    *   **`slideshow/slideshow/slideshow.html`**: Added a new text element for the `flip` instruction.
    *   **`slideshow/slideshow/event_listeners.js`**: The tutorial sequence is now more comprehensive. It guides the user through unlocking, explains the `swipe right` gesture in more detail, and then introduces the `flip` gesture for downward navigation. The WebSocket handler now also processes the `down` event to control the Reveal.js `down()` function.
    *   **Problem Solved**: This change resolves a gesture conflict and provides a more complete and intuitive tutorial that covers all primary navigation actions (`left`, `right`, `down`) and the new lock/unlock toggle mechanism.
*   **2025-07-27 14:45**: Refined the interactive tutorial with lock-state awareness and visual feedback.
    *   **`src/run_live.py`**: Modified the WebSocket server to broadcast the system's lock state (`system_locked`, `system_unlocked`) to all clients upon connection and whenever the state changes.
    *   **`slideshow/slideshow/slideshow.html`**: Added CSS to apply a red tint to the background when the system is locked and a green "pulse" animation when it is unlocked.
    *   **`slideshow/slideshow/event_listeners.js`**: The tutorial logic was expanded. It now explicitly instructs the user to perform the "rolling" gesture to unlock the system and provides clear visual confirmation ("System Unlocked!") upon success, creating a more comprehensive and intuitive onboarding experience.
    *   **Problem Solved**: The initial tutorial was good but lacked feedback on the crucial lock/unlock mechanism. This refinement provides a richer, more guided experience that teaches the user the complete interaction flow, from getting the system's attention to unlocking it and performing their first action.

*   **2025-07-27 14:40**: Implemented an interactive, attention-based user tutorial.
    *   **`src/run_live.py`**: Enhanced the main loop to track user attention using the `HeadPoseEstimator`. It now sends timed `user_detected` and `user_lost` events via a JSON-based WebSocket protocol.
    *   **`slideshow/slideshow/slideshow.html`**: Added a new overlay `div` and CSS for displaying instructional messages.
    *   **`slideshow/slideshow/event_listeners.js`**: Reworked the WebSocket `onmessage` handler to parse JSON. It now controls a progressive tutorial: it shows a welcome message on `user_detected`, and upon the first `swipe_right` gesture, it transitions to the next instruction.
    *   **Problem Solved**: The system now provides an intuitive, on-screen guide for new users. Instead of relying on prior knowledge, it actively invites engagement when a user looks at the screen and teaches them the core gestures interactively, significantly improving usability.

*   **2025-07-27 14:34**: Finalized and fixed the fine-tuning data pipeline.
    *   **`src/lightning_datamodule.py`**: Implemented a comprehensive fix to ensure data object consistency. The fine-tuning training dataset is now always wrapped in a `torch.utils.data.Subset`, making its structure identical to the datasets created during a standard training run. This resolves all `AttributeError` exceptions caused by inconsistent object types.
    *   **Problem Solved**: The previous implementation of the scientifically correct validation strategy introduced a bug where the training dataset object had a different type during fine-tuning, causing crashes in downstream functions (like class balancing and scaler fitting) that expected a `Subset` object. This final fix makes the data pipeline robust and reliable for both standard training and fine-tuning.

*   **2025-07-27 14:30**: Implemented a scientifically correct validation strategy for fine-tuning.
    *   **`src/lightning_datamodule.py`**: The `setup` method was significantly refactored. When the `FINETUNING_CONFIG['ENABLED']` flag is active, it now correctly separates the data sources. The training set is built from the specialized fine-tuning data mix, while the validation and test sets are now built from a completely separate, standard dataset instance.
    *   **`src/data_loader.py`**: Added a `force_standard_load` boolean parameter to the `GestureDataset` constructor. This allows the `DataModule` to instantiate a "clean" version of the dataset for validation, ignoring the global fine-tuning configuration.
    *   **Problem Solved**: Previously, the fine-tuning process would split the mixed (idle + sampled gestures) dataset into training and validation sets. This meant the model was being validated on a different data distribution than its original task, making the validation accuracy a poor indicator of "catastrophic forgetting." This new approach ensures that during fine-tuning, the model is always validated against the original, unbiased data distribution, providing a true measure of whether the model's core gesture recognition capabilities are being preserved.

*   **2025-07-27 14:02**: Fixed a WebSocket handshake race condition.
    *   **`src/run_live.py`**: Reordered the main initialization sequence. The `LiveGestureController` (which performs slow, blocking I/O by loading the model) is now fully initialized *before* the `websockets.serve` function is called.
    *   **Problem Solved**: A race condition was causing the WebSocket server to fail the handshake if a client connected immediately upon startup. By ensuring all blocking operations are complete before the server starts listening, we guarantee the `asyncio` event loop is free to handle the handshake instantly, making the connection robust and reliable.

*   **2025-07-27 13:52**: Implemented gesture-specific confidence thresholds and cooldowns.
    *   **`src/config.py`**: Replaced the global `CONFIDENCE_THRESHOLD` and `COOLDOWN_PERIOD` with `GESTURE_CONFIDENCE_THRESHOLDS` and `GESTURE_COOLDOWN_PERIODS` dictionaries. This allows for fine-tuning the real-time performance for each gesture individually based on its unique characteristics.
    *   **`src/run_live.py`**: Modified the live inference loop to look up the specific confidence and cooldown values for each recognized gesture from the new dictionaries in the config.
    *   **Problem Solved**: This provides a much more nuanced and effective way to control the live system. By setting higher confidence requirements for shorter gestures and adjusting cooldowns based on gesture length, the system's responsiveness and reliability are significantly improved, reducing both false positives and missed activations.

*   **2025-07-27 13:29**: Implemented a dedicated, configurable fine-tuning pipeline.
    *   **`src/finetune.py`**: Created a new, dedicated script to handle the fine-tuning process. This provides a clear separation between training from scratch and specializing an existing model.
    *   **`src/config.py`**: Added a `FINETUNING_CONFIG` dictionary to provide centralized, scientific control over the fine-tuning data mix, including which idle files to use and the sample fraction of gesture data to prevent catastrophic forgetting.
    *   **`src/data_loader.py`**: Implemented a new `_load_finetuning_strategy` to prepare the specific data mix defined in the configuration.
    *   **`src/train.py`**: Refactored the script to move the core logic into a reusable `run_training` function. This function now handles loading models from checkpoints, applying the correct learning rate, and tagging runs appropriately in W&B, making the codebase more modular and avoiding duplication.
    *   **Problem Solved**: This provides a robust and scientifically sound workflow for improving model performance on specific, challenging data (like complex idle movements) without needing to retrain from scratch. The process is non-destructive, clearly logged, and highly configurable.

*   **2025-07-27 12:17**: Integrated a self-contained, gesture-controlled slideshow feature.
    *   **`src/run_live.py`**: Major refactoring to use Python's `asyncio` library. The script now runs an asynchronous event loop to manage multiple tasks concurrently. It starts a WebSocket server (`websockets`) to broadcast gesture commands and a simple HTTP server to serve the slideshow files directly, making the entire feature self-contained. The core gesture logic was updated to send string commands (`"left"`, `"right"`) via the WebSocket instead of simulating keypresses. The lock/unlock gestures were remapped to `flip` and `rolling` as requested.
    *   **`slideshow/slideshow/event_listeners.js`**: Simplified the WebSocket client to connect to the new, dedicated server port (`8765`) and to only handle the essential `left` and `right` commands for slide navigation.
    *   **`slideshow/slideshow/slideshow.html`**: Updated the user instructions on the main slide to reflect the new, simplified control scheme (`rolling` to unlock, `flip` to lock, `swipe` to navigate). Removed outdated gesture hints from sub-slides.
    *   **Problem Solved**: This provides a fully integrated and robust method for controlling a web-based UI (the Reveal.js slideshow) with gestures. By refactoring `run_live.py` into a self-contained server application, we have created a clean, modular, and easily demonstrable use case for the gesture recognition engine, avoiding the need to merge disparate and complex demo scripts.

*   **2025-07-27 10:42**: Refined the live UI to be context-aware.
    *   **`src/run_live.py`**: Modified the `_update_ui` function to only display the "Spotter: MOTION/NO MOTION" status text when the application is running in the `--mode spotter`.
    *   **Problem Solved**: This declutters the user interface. The spotter status is irrelevant when running in the default `gru_only` mode, so hiding it makes the UI cleaner and more intuitive for the standard use case.

*   **2025-07-27 10:37**: Integrated a new Head Pose Estimation (HPE) feature to detect user attention.
    *   **`src/head_pose_estimation.py`**: Created a new, dedicated module to encapsulate all logic for head pose estimation using MediaPipe FaceMesh and OpenCV. The module calculates pitch, yaw, and roll angles.
    *   **`src/config.py`**: Added an `ENABLE_HEAD_POSE_DETECTION` flag and `HEAD_POSE_THRESHOLDS` to make the new feature fully configurable.
    *   **`src/run_live.py`**: Modified the live script to conditionally initialize and use the `HeadPoseEstimator`. It now processes frames to determine if the user is looking at the screen and displays this status ("ATTENTIVE" or "LOOKING AWAY") on the UI.
    *   **Problem Solved**: This new feature provides a robust and reliable way to gauge user attention, which is a critical first step towards building more context-aware interactions. It allows the system to understand if the user is actively engaged before processing gestures, potentially reducing unintended activations.

*   **2025-07-27 09:36**: Started live gesture recognition with a specific high-performance model.
    *   **`src/run_live.py`**: Executed the script using the `--model_path` argument to load `models/gru_main_model-epoch=21-val_acc=0.9944.ckpt`.
    *   **Problem Solved**: This allows for the immediate use and evaluation of a specific, well-performing model in the live environment, bypassing the default behavior of using the most recently created one. This is essential for testing and deploying specific model versions.

*   **2025-07-26 19:04**: Implemented a fully integrated hyperparameter tuning pipeline using Weights & Biases Sweeps.
    *   **`src/tuning.py`**: Created a new executable script to define and run hyperparameter sweeps. It contains a `sweep_config` for defining the search space and uses `wandb.agent` to manage the tuning process.
    *   **`src/train.py`**: Refactored the main training script. The core logic was moved into a reusable `run_training(config)` function, which can now be called programmatically with different hyperparameter configurations. The `main()` function is now a simple wrapper for handling command-line arguments for single runs.
    *   **Problem Solved**: The project previously had an outdated, non-functional tuning script. This new implementation provides a powerful, modern, and fully integrated solution for hyperparameter optimization that leverages the existing PyTorch Lightning framework and W&B integration. This allows for efficient, automated discovery of the best model configurations.

*   **2025-07-26 18:56**: Improved the live inference script with smarter defaults.
    *   **`src/run_live.py`**: Implemented a `find_latest_model_path` function that automatically detects and uses the most recently created GRU model checkpoint from the `models/` directory as the default. Also, the default execution mode was changed to `gru_only` to make the primary function of the application more accessible. The two-stage `spotter` pipeline now needs to be explicitly enabled with `--mode spotter`.
    *   **Problem Solved**: This enhances user experience by removing the need to manually update the script to use the latest trained model. It also aligns the default behavior with the most common use case (direct gesture recognition), while still providing the more advanced spotter pipeline as a clear option.

*   **2025-07-26 18:43**: Added a `gru_only` mode to the live inference script for simplified testing.
    *   **`src/run_live.py`**: Refactored the script to support two execution modes via a `--mode` command-line argument. The default `spotter` mode runs the full two-stage pipeline, while the new `gru_only` mode bypasses the spotter and sends data directly to the main GRU model. The main `run` method was split into `run_spotter_pipeline` and `run_gru_only_pipeline` to cleanly separate the logic for each mode.
    *   **Problem Solved**: This provides a crucial debugging and testing tool. It allows for the evaluation of the main GRU model's performance in a live setting without the influence or potential issues of the spotter model, simplifying the process of isolating and analyzing classification behavior.

*   **2025-07-26 18:13**: Fixed a `ModuleNotFoundError` by correcting an import path.
    *   **`src/lightning_datamodule.py`**: Changed the import for `SpotterLightningModule` from the non-existent `src.spotter` to the correct `src.lightning_module`.
    *   **Problem Solved**: The application was crashing at startup because a file was moved, but the corresponding import statement was not updated. This fix aligns the code with the current project structure and resolves the `ModuleNotFoundError`.

*   **2025-07-26 17:56**: Implemented a new, configurable data loading strategy to improve idle/gesture differentiation.
    *   **`src/config.py`**: Added a `USE_DEDICATED_IDLE_DATA` boolean flag. When `True`, the data loaders use a new strategy to create a cleaner training set.
    *   **`src/data_loader.py`**: Refactored to support two loading strategies for the main GRU model. The new strategy creates all possible windows from gesture files, then filters out any windows labeled as 'idle'. It then separately loads data from the dedicated `Data/train/idle` folder. This prevents windowing artifacts while providing the model with pure gesture examples and more realistic idle examples.
    *   **`src/spotter_data_loader.py`**: Refactored to support two loading strategies for the spotter model. The new strategy filters out all 'idle' rows from the gesture CSVs *before* processing them, creating a clean "motion" dataset. It then separately loads the dedicated "chaotic idle" data.
    *   **Problem Solved**: The model's understanding of "idle" was weak because the idle data within gesture recordings was too clean and simple. This new strategy allows the models to be trained on pure gesture examples versus more realistic and challenging idle examples, which should significantly improve the model's ability to reject non-gesture movements.

*   **2025-07-26 17:22**: Implemented a full two-stage gesture recognition pipeline to improve real-time accuracy and reduce false positives.
    *   **`src/config.py`**: Added a new configuration section for the spotter model, including the model path (`SPOTTER_MODEL_PATH`), confidence threshold, and buffer size.
    *   **`src/run_live.py`**: Major refactoring to integrate the two-stage logic. The script now loads both the spotter and main GRU models. It uses the spotter to continuously monitor for motion. Only when motion is detected does it collect a full sequence and pass it to the main model for classification. This prevents the main model from processing ambiguous idle movements.
    *   **`src/train_spotter.py`**: Added support for command-line arguments to select a class balancing strategy (`--balancing_strategy`), specifically enabling undersampling to handle the imbalanced motion/no-motion dataset.
    *   **`src/spotter_data_loader.py`**: Made the data loader more robust. It now correctly handles the `idle_confusion.csv` file, pads missing keypoint columns to prevent errors, and uses a small sliding window to ensure dynamic features (velocity, acceleration) are calculated correctly, mirroring the main preprocessing pipeline. Also integrated the `undersample_data` function.
    *   **`src/lightning_module.py`**: Added a new `SpotterLightningModule` specifically for training the binary classification spotter model, using `BCEWithLogitsLoss` and binary accuracy metrics.
    *   **`src/models.py`**: Added the `SpotterNet` model, a lightweight FFNN designed for fast, frame-by-frame binary classification.
    *   **Problem Solved**: This major architectural enhancement directly addresses the problem of the model misinterpreting idle movements (like scratching one's head) as actual gestures. The spotter acts as an intelligent gate, significantly improving the system's reliability and usability in a real-world setting.

*   **2025-07-26 16:09**: Fixed critical model loading failure in the live inference script.
    *   **`src/run_live.py`**: Refactored the `_load_model` function to remove all hardcoded, dummy hyperparameters. It now calls `GestureLightningModule.load_from_checkpoint(model_path)` directly, allowing PyTorch Lightning to correctly and automatically reconstruct the model using the hyperparameters saved inside the checkpoint file.
    *   **Problem Solved**: The live script was previously ignoring the hyperparameters saved in the checkpoint and trying to load the saved weights into a new, default-architected model, causing a `RuntimeError` due to layer size mismatches. This fix ensures that the live inference script loads models in a robust and correct way, fully leveraging the backward-compatibility logic in the `GestureLightningModule` and resolving the final bug in the model loading pipeline.

*   **2025-07-26 14:17**: Implemented a robust, backward-compatible model loading architecture.
    *   **`src/lightning_module.py`**: Added a compatibility layer to the `__init__` method. This layer intelligently inspects the incoming `model_params`. If it detects the old hyperparameter format (`hidden_size`, `num_layers`), it automatically converts them to the new `hidden_sizes` list format before passing them to the model constructor.
    *   **`src/models.py`**: Removed the temporary backward-compatibility code from the model classes, making them clean and reliant solely on the new `hidden_sizes` parameter.
    *   **Problem Solved**: The previous fix, while functional, placed the compatibility logic in the wrong place. This more robust solution correctly centralizes the backward-compatibility logic within the `GestureLightningModule`, which is responsible for model creation and loading. This resolves both the initial `TypeError` and the subsequent `RuntimeError` related to state dictionary mismatches, ensuring that old checkpoints can be loaded seamlessly without any errors.

*   **2025-07-26 14:11**: Implemented variable-size multi-layer FFNN architecture.
    *   **`src/config.py`**: Updated the `MODEL_PARAMS` for the `ffnn` model. Replaced `hidden_size` with a `hidden_sizes` list (e.g., `[256, 128]`) to allow for defining a stack of fully connected layers with different sizes.
    *   **`src/models.py`**: Refactored the `FFNN` class to dynamically construct a `nn.Sequential` model based on the `hidden_sizes` list. This provides the same architectural flexibility for the FFNN as was previously implemented for the GRU.
    *   **Problem Solved**: This change brings architectural consistency to the project. Both the GRU and FFNN models can now be configured with multiple, variable-sized hidden layers directly from the central `config.py` file, enhancing modularity and experimental flexibility.

*   **2025-07-26 14:09**: Implemented variable-size multi-layer GRU architecture.
    *   **`src/config.py`**: Modified the `MODEL_PARAMS` for the `gru` model. Replaced `hidden_size` and `num_layers` with a single `hidden_sizes` list (e.g., `[128, 64]`). This allows for defining a stack of GRU layers where each can have a different size.
    *   **`src/models.py`**: Refactored the `GRUNet` class. It now dynamically constructs a `nn.ModuleList` of single-layer GRU modules based on the `hidden_sizes` list. This enables the creation of more complex, hierarchical recurrent architectures.
    *   **`src/train.py`**: Adjusted the script to correctly handle the new `hidden_sizes` parameter and pass it to the model.
    *   **Problem Solved**: The previous GRU implementation was limited to a stack of layers that all shared the same hidden size. This change provides the flexibility to design more sophisticated "funnel" architectures, which can improve model efficiency and performance by forcing the network to learn more compressed data representations at each successive layer.

*   **2025-07-26 14:00**: Centralized model architecture hyperparameters for full configurability.
    *   **`src/config.py`**: Added a `MODEL_PARAMS` dictionary to define model-specific hyperparameters (e.g., `hidden_size`, `num_layers`, `dropout_rate`) for both `gru` and `ffnn` architectures. Removed the old standalone `DROPOUT_RATE` constant.
    *   **`src/train.py`**: Modified the script to import and use the new `MODEL_PARAMS` dictionary. It now dynamically selects the correct parameters based on the chosen model, removing hardcoded values from the training logic.
    *   **`src/models.py`**: Removed the direct dependency on `config.py`. The models are now fully decoupled, receiving all necessary parameters during instantiation.
    *   **Problem Solved**: Previously, key model architecture parameters were hardcoded in `src/models.py` and `src/train.py`, making them difficult to tune. This change centralizes all hyperparameters in `config.py`, making the entire model architecture configurable from a single location, which significantly improves modularity and simplifies experimentation.

*   **2025-07-26 13:53**: Enhanced experiment tracking by logging a comprehensive configuration to Weights & Biases.
    *   **`src/train.py`**: Modified the script to assemble a single, detailed dictionary containing all relevant settings before initializing the `WandbLogger`. This includes command-line arguments, key hyperparameters from `config.py` (like `LEARNING_RATE`, `BATCH_SIZE`), data preparation settings (`PREPROCESSING_STRATEGY`, `LABELING_STRATEGY`), the full `FEATURE_ENGINEERING_CONFIG`, and the actual model parameters (`dropout`, `hidden_size`, etc.).
    *   **Problem Solved**: Previously, only a small subset of parameters were logged, making it difficult to reproduce experiments or compare runs accurately. This change ensures that every run in `wandb` is now fully documented with the exact configuration used to generate it, significantly improving reproducibility and analytical capabilities.

*   **2025-07-26 13:25**: Fixed a critical data shape mismatch between training and live inference.
    *   **`src/run_live.py`**: Refactored the live keypoint extraction logic. Instead of creating a `(num_keypoints, 3)` array for each frame and then reshaping the full buffer, the code now flattens the `(x, y, z)` coordinates into a single flat array for each frame *before* appending to the buffer.
    *   **Problem Solved**: The previous `reshape` operation on the `(20, 23, 3)` buffer created an incorrect memory layout, grouping all x-coordinates together, then all y's, etc. This did not match the interleaved `(x, y, z, x, y, z, ...)` structure of the training data, causing the model to receive scrambled and meaningless input. The new approach ensures the live data has the exact same shape and memory layout as the training data from the very beginning, resolving the final and most subtle data corruption bug.

*   **2025-07-26 13:14**: Fixed a critical data scrambling bug in the live inference pipeline.
    *   **`src/run_live.py`**: Modified the script to derive the keypoint processing order directly from the `FEATURES_TO_USE` list in `config.py`, instead of using the alphabetically sorted `SELECTED_KEYPOINTS` list.
    *   **Problem Solved**: The model was trained on data in a specific order defined by `FEATURES_TO_USE`, but the live script was assembling the data in a different, alphabetical order. This caused the feature data to be completely scrambled, leading to catastrophic model performance. This fix guarantees that the live data is processed in the exact same order as the training data, resolving the core inconsistency.

*   **2025-07-26 12:45**: Fixed a critical `RuntimeError` caused by a mismatch between model and data dimensions.
    *   **`src/train.py`**: Implemented a `get_input_size()` helper function that dynamically calculates the number of features at runtime based on `FEATURES_TO_USE` and `FEATURE_ENGINEERING_CONFIG` from `config.py`. This calculated size is now passed to the model at instantiation.
    *   **`src/lightning_module.py`**: Modified the `GestureLightningModule` to accept the `input_size` and pass it to the model factory.
    *   **`src/models.py`**: Decoupled the `GRUNet` and `FFNN` models from global state by making them accept `input_size` as a constructor argument. Removed all reliance on imported constants for model dimensions.
    *   **`src/data_loader.py` & `src/config.py`**: Removed all stale, hardcoded feature count constants (`NUM_FEATURES_TOTAL`, `GRU_INPUT_FEATURES`, etc.) that were the source of the error.
    *   **Problem Solved**: The system previously relied on hardcoded constants to define the model's input layer size. When the feature configuration was changed, these constants became outdated, causing a crash. This fix makes the entire pipeline robust and dynamic. The model's architecture is now guaranteed to match the data pipeline's output, completely eliminating the source of the bug.

*   **2025-07-26 12:36**: Implemented a fully modular and configurable feature engineering pipeline.
    *   **`src/feature_engineering.py`**: Created this new, dedicated module to house the logic for all individual feature calculations (e.g., velocity, acceleration). This isolates feature logic, making it clean and easy to maintain.
    *   **`src/config.py`**: Added the `FEATURE_ENGINEERING_CONFIG` dictionary. This acts as a central control panel to enable or disable specific features across the entire project with simple `True`/`False` flags.
    *   **`src/preprocessing.py`**: Refactored the main `preprocess_sequence` function. It now dynamically imports and calls the functions from `feature_engineering.py` based on the settings in `FEATURE_ENGINEERING_CONFIG`. This removes hardcoded feature generation and makes the pipeline highly modular.
    *   **Problem Solved**: This major enhancement provides a flexible and scientific framework for experimenting with different feature sets. The choice of features is now controlled entirely from the central configuration file, ensuring consistency and eliminating the need to modify the core processing logic for experimentation.

*   **2025-07-26 12:29**: Completed a new training run with the undersampling strategy.
    *   **`src/train.py`**: Executed with `--balancing_strategy undersample`.
    *   **Result**: The model training completed successfully, with Early Stopping halting the run at epoch 17. The best model was saved at epoch 12, achieving a validation accuracy of **0.8741**. The resulting model (`gru_main_model-epoch=12-val_acc=0.8741.ckpt`) is now saved and ready for use. This provides a new baseline model trained on a balanced dataset.

*   **2025-07-26 12:27**: Fixed a critical data shape mismatch between training and live inference.
    *   **`src/run_live.py`**: Added a `reshape` operation to the live inference loop. This ensures that the sequence of keypoints collected from the camera, which has a shape of `(20, 23, 3)`, is flattened into the correct `(20, 69)` shape before being passed to the preprocessing function.
    *   **Problem Solved**: The live preprocessing pipeline was receiving data in a 3D format, while the training pipeline used a 2D format. This caused the feature engineering and standardization steps to produce meaningless results, leading to extremely poor model performance in the live mode. This fix guarantees that the data structure is identical for both training and inference, resolving the core inconsistency.

*   **2025-07-26 12:15**: Fixed a critical data leakage bug in the training pipeline.
    *   **`src/lightning_datamodule.py`**: Refactored the `setup` method to ensure the `StandardScaler` is fitted *only* on the training data subset *after* the random train/validation/test split is performed. The same fitted scaler is then correctly applied to all three datasets (train, val, test).
    *   **Problem Solved**: The previous implementation fitted the scaler on an unshuffled, approximate version of the training data *before* the random split. This created a mismatch between the data used to fit the scaler and the data used to train the model, leading to a skewed scaler being saved for live inference. This fix resolves the data leakage, ensuring the preprocessing is 100% consistent between training and live deployment and should significantly improve live performance.

*   **2025-07-26 12:05**: Fixed a critical bug in the live preprocessing pipeline.
    *   **`src/preprocessing.py`**: Removed a redundant and incorrect `reshape` operation within the `preprocess_sequence` function.
    *   **Problem Solved**: This fixed a `ValueError` that occurred during live inference. The error was caused by an incorrect attempt to reshape the keypoint data array, which was introduced during a recent refactoring. The fix ensures the live preprocessing pipeline now runs without crashing.

*   **2025-07-26 12:02**: Added configurable Early Stopping to the training process.
    *   **`src/config.py`**: Added `EARLY_STOPPING_PATIENCE` and `EARLY_STOPPING_MIN_DELTA` parameters. This allows for fine-tuning the conditions under which training will halt if the validation accuracy stops improving.
    *   **`src/train.py`**: Modified the script to import and use these new configuration values when initializing the PyTorch Lightning `EarlyStopping` callback.
    *   **Problem Solved**: This makes a critical training hyperparameter easily accessible and configurable from the central config file, improving experimental flexibility and reproducibility.

*   **2025-07-26 12:00**: Made the GRU model's dropout rate configurable.
    *   **`src/config.py`**: Added a `DROPOUT_RATE` parameter under the `Model Configuration` section. This allows for easy adjustment of the dropout regularization from a central location.
    *   **`src/models.py`**: Modified the `GRUNet` class to use the new `DROPOUT_RATE` from the config file as the default value for its `dropout` parameter.
    *   **Problem Solved**: This change enhances modularity and makes it simpler to tune the model's regularization strength without modifying the model's source code directly.

*   **2025-07-26 11:52**: Implemented a modular and configurable standardization strategy as an alternative to body-centric normalization.
    *   **`src/scaler.py`**: Created a new `StandardScaler` class to handle fitting, transforming, saving, and loading Z-score normalization statistics. This ensures that the exact same mean/std values are used across training and inference.
    *   **`src/config.py`**: Replaced the `PREPROCESSING_STEPS` dictionary with a single `PREPROCESSING_STRATEGY` string ('body_centric' or 'standardize') to make the choice of normalization explicit and mutually exclusive. Added `SCALER_PATH` to define where the fitted scaler is stored.
    *   **`src/lightning_datamodule.py`**: Updated the `setup` method to automatically handle the scaler. When `PREPROCESSING_STRATEGY` is 'standardize', it now fits the `StandardScaler` on the training data, saves it to a file, and passes it to the datasets. For inference, it loads the pre-fitted scaler.
    *   **`src/data_loader.py`**: Modified `GestureDataset` to accept the optional `scaler` object and pass it to the preprocessing function.
    *   **`src/preprocessing.py`**: The main `preprocess_sequence` function was updated to read the `PREPROCESSING_STRATEGY` and apply either the body-centric normalization or the standardization via the provided scaler.
    *   **`src/run_live.py`**: The live inference script now loads the saved scaler from `SCALER_PATH` at startup if the strategy is 'standardize', ensuring perfect consistency with the training pipeline.
    *   **Problem Solved**: This provides a robust, scientifically sound way to experiment with two different, powerful feature scaling techniques. The choice is controlled by a single configuration setting, and the system automatically handles the fitting and application of the scaler, preventing data leakage.

*   **2025-07-26 11:44**: Refactored the entire preprocessing pipeline to be modular and configurable.
    *   **`src/preprocessing.py`**: Created this new file to centralize all feature engineering and normalization logic. It contains functions for each step (centering, scaling, rotation, etc.) and a main `preprocess_sequence` function that applies them based on flags in the config.
    *   **`src/config.py`**: Added a `PREPROCESSING_STEPS` dictionary to allow individual preprocessing steps to be enabled or disabled globally. This is crucial for conducting controlled experiments on the impact of each normalization technique.
    *   **`src/data_loader.py`**: Removed all hardcoded feature engineering logic from the `GestureDataset`. It now imports and calls `preprocess_sequence`, ensuring the training data pipeline is consistent and configurable.
    *   **`src/run_live.py`**: Removed all duplicated feature engineering logic from the live inference pipeline. It now also calls `preprocess_sequence`, guaranteeing that the live data is processed in the exact same way as the training data.
    *   **Problem Solved**: This major refactoring eliminates significant code duplication, reduces the risk of inconsistencies between training and inference, and provides a flexible, scientific framework for experimenting with different preprocessing strategies simply by changing the configuration.

*   **2025-07-26 11:29**: Completed a new training run with the updated feature set and undersampling.
    *   **`src/train.py`**: Executed with `--balancing_strategy undersample` after reconfiguring the feature set in `config.py`.
    *   **Result**: The model training completed successfully, achieving a best validation accuracy of **0.9371**. The resulting model (`gru_main_model-epoch=28-val_acc=0.9371.ckpt`) is now saved and ready for use. This validates the new feature engineering pipeline and provides an improved model for live inference.
*   **2025-07-26 11:25**: Implemented a centralized and configurable feature selection system.
    *   **`src/config.py`**: Added a `FEATURES_TO_USE` list to centrally define all keypoints for the model. This removes hardcoded feature selection and allows for easy experimentation. The default list now excludes hip and leg keypoints.
    *   **`src/data_loader.py`**: Modified the script to dynamically import and use `FEATURES_TO_USE`. The feature engineering pipeline was updated to ensure robust normalization (position, size, and rotation invariance) without relying on hip keypoints. It now uses the shoulder midpoint for centering and the shoulder-to-nose vector for establishing rotation.
    *   **`src/run_live.py`**: Updated the live inference script to mirror the changes in `data_loader.py`. It now uses the same `FEATURES_TO_USE` list and the same modified normalization logic, ensuring complete consistency between training and real-time deployment. This solved the problem of having hardcoded keypoint selections and adapted the system to work with a more focused, upper-body feature set.
*   **2025-07-26 11:14**: Started a new training run with undersampling and 'majority' window labeling.
    *   **`src/config.py`**: Changed `LABELING_STRATEGY` from `'any'` to `'majority'`.
    *   **`src/train.py`**: Executed with the `--balancing_strategy undersample` flag. The best model achieved a validation accuracy of `0.9301`. This addresses the class imbalance issue at the data level and uses a more robust labeling strategy for gesture windows.
*   **2025-07-26 11:10**: Clarified real-time buffer logic for improved consistency.
    *   **`src/config.py`**: Renamed `REALTIME_BUFFER_SIZE` to `PREDICTION_SMOOTHING_BUFFER_SIZE` to more accurately reflect its purpose of stabilizing output predictions, not collecting input features.
    *   **`src/run_live.py`**: Updated the script to use the new, more descriptive variable name. This resolves confusion between the feature collection buffer (which must match `MAX_SEQ_LENGTH`) and the prediction smoothing buffer.
*   **2025-07-26 11:05**: Implemented modular window labeling strategy.
    *   **`src/config.py`**: Added `LABELING_STRATEGY` setting, allowing a choice between 'any' (at least one gesture frame) and 'majority' (>50% gesture frames) for labeling a window.
    *   **`src/data_loader.py`**: Updated `GestureDataset` to use the selected `LABELING_STRATEGY`, making the process for assigning labels to data windows modular and easy to extend.
*   **2025-07-26 11:03**: Made data windowing stride configurable.
    *   **`src/config.py`**: Added a new `WINDOW_STRIDE` parameter to allow easy configuration of the sliding window's step size during data preparation.
    *   **`src/data_loader.py`**: Modified the `GestureDataset` to use the `WINDOW_STRIDE` from the config file, replacing the hardcoded value. This enhances modularity and makes it easier to experiment with different data augmentation strategies.
*   **2025-07-26 10:56**: Implemented a live debugging mode.
    *   **`src/run_live.py`**: Added a `--debug` command-line argument. When active, the script now logs the model's raw prediction probabilities for every gesture to a timestamped CSV file every 0.5 seconds. It also displays these probabilities on the live camera feed, providing real-time insight into the model's decision-making process. This feature is crucial for analyzing model behavior, diagnosing misclassifications, and fine-tuning performance.
*   **2025-07-26 10:36**: Added a modular undersampling feature.
    *   **`src/class_balancing.py`**: Created a new `undersample_data` function to perform random undersampling on the dataset.
    *   **`src/data_loader.py`**: Modified `GestureDataset` to accept a `balancing_strategy` and apply undersampling during data loading if the strategy is set to `'undersample'`.
    *   **`src/lightning_datamodule.py`**: Updated `GestureDataModule` to propagate the `balancing_strategy` to the dataset.
    *   **`src/train.py`**: Added a `--balancing_strategy` command-line argument with the choice `'undersample'`, allowing users to easily enable this feature during training. This solves the problem of class imbalance by modifying the dataset itself rather than just the loss function.
*   **2025-07-26**: Initial creation of `overview.md`. Documented the existing project architecture, including the two-stage pipeline, file structure, feature engineering, and HCI concepts. This serves as the baseline understanding of the project for future development.
